"""V7 six degree-of-freedom motion and collision planning.

The V6 controller keeps its fixed Cartesian waypoints.  This module is an
independent V7 planning boundary: inverse kinematics, collision queries and
OMPL planning all use a private :class:`mujoco.MjData` snapshot.  In
particular, a planner cannot make a simulation pass by writing to the live
``qpos`` or by reading an unrecorded object pose.

``V7Kinematics`` solves the seven KUKA arm joints (``joint1`` ... ``joint7``)
for a position and MuJoCo ``(w, x, y, z)`` quaternion.  ``CollisionChecker``
checks the complete arm and Sharpa hand against the declared fixed geometry,
the table and non-target objects.  ``RRTConnectPlanner`` is a small adapter
around OMPL's geometric RRT-Connect implementation; no local random-tree
implementation is hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Iterable, Mapping, Sequence

import mujoco
import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
DEFAULT_EE_BODY = "link7"


def _vector(values: Sequence[float], size: int, field_name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite vector of length {size}") from error
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must be a finite vector of length {size}")
    return result.copy()


def _unit_quaternion(values: Sequence[float], field_name: str = "quaternion") -> np.ndarray:
    result = _vector(values, 4, field_name)
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError(f"{field_name} must be non-zero")
    return result / norm


def _quat_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    result = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(result, _unit_quaternion(quaternion))
    return result.reshape(3, 3)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=float).reshape(9))
    return _unit_quaternion(result)


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a bounded world-frame rotation vector from current to target."""
    relative = target @ current.T
    quaternion = _matrix_to_quat(relative)
    if quaternion[0] < 0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, max(float(quaternion[0]), 1e-12))
    angle = min(angle, math.pi)
    return quaternion[1:] / vector_norm * angle


def _body_ancestors(model: mujoco.MjModel, body_id: int) -> set[int]:
    result: set[int] = set()
    current = int(body_id)
    # World body (0) owns the table and fixtures; it is an environment body,
    # never part of the KUKA collision set.
    while current > 0:
        result.add(current)
        parent = int(model.body_parentid[current])
        if parent <= 0 or parent == current:
            break
        current = parent
    return result


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(index)) or ""


class V7Kinematics:
    """Seven-joint KUKA pose solver backed by MuJoCo's Jacobian."""

    def __init__(self, model: mujoco.MjModel, *, body_name: str = DEFAULT_EE_BODY,
                 joint_names: Sequence[str] = ARM_JOINT_NAMES,
                 position_tolerance: float = 2e-4,
                 orientation_tolerance: float = 2e-3,
                 max_iterations: int = 250,
                 damping: float = 1e-4) -> None:
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"model is missing end-effector body {body_name!r}")
        self.joint_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            for name in joint_names
        )
        if len(self.joint_ids) != 7 or any(index < 0 for index in self.joint_ids):
            raise ValueError("model must expose seven named KUKA arm joints")
        if any(model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE for index in self.joint_ids):
            raise ValueError("KUKA arm joints must be hinge joints")
        self.qpos_adrs = np.asarray([model.jnt_qposadr[index] for index in self.joint_ids], dtype=int)
        self.dof_adrs = np.asarray([model.jnt_dofadr[index] for index in self.joint_ids], dtype=int)
        ranges = np.asarray([model.jnt_range[index] for index in self.joint_ids], dtype=float)
        if not np.all(np.isfinite(ranges)) or np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError("KUKA arm joints must have finite increasing model limits")
        self.lower = ranges[:, 0].copy()
        self.upper = ranges[:, 1].copy()
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        if (self.position_tolerance <= 0 or self.orientation_tolerance <= 0
                or self.max_iterations < 1 or self.damping <= 0):
            raise ValueError("IK tolerances, iterations and damping must be positive")

    def _set_arm(self, joints: Sequence[float]) -> np.ndarray:
        values = _vector(joints, 7, "joints")
        clipped = np.clip(values, self.lower, self.upper)
        self.scratch.qpos[self.qpos_adrs] = clipped
        return clipped

    def pose(self, joints: Sequence[float], hand_qpos: Sequence[float] | Mapping[str, float] | None = None,
             *, reset_other_qpos: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return end-effector position/quaternion in a private scratch state."""
        if reset_other_qpos:
            self.scratch.qpos[:] = self.model.qpos0
        self._set_arm(joints)
        if reset_other_qpos:
            # The arm pose is independent of hand/object state, but retaining a
            # deterministic scratch state avoids accidentally using stale free
            # joints when callers request collision-aware hand poses.
            self.scratch.qvel[:] = 0.0
        if hand_qpos is not None:
            _apply_hand_qpos(self.model, self.scratch, hand_qpos)
        mujoco.mj_forward(self.model, self.scratch)
        position = self.scratch.xpos[self.body_id].copy()
        quaternion = _matrix_to_quat(self.scratch.xmat[self.body_id].reshape(3, 3))
        return position, quaternion

    def solve(self, position: Sequence[float], quaternion: Sequence[float],
              seed: Sequence[float] | None = None) -> np.ndarray | None:
        """Solve a 6DoF wrist pose and return a limit-clipped 7-vector.

        The seed is always copied into scratch state.  Failure returns ``None``
        rather than exposing a partially converged configuration.
        """
        target_position = _vector(position, 3, "position")
        target_rotation = _quat_to_matrix(quaternion)
        if seed is None:
            # Mid-range is reproducible and avoids assuming the V6 home pose.
            q = (self.lower + self.upper) / 2.0
        else:
            q = _vector(seed, 7, "seed")
            if np.any(q < self.lower - 1e-10) or np.any(q > self.upper + 1e-10):
                raise ValueError("seed is outside the seven KUKA model limits")
        q = np.clip(q, self.lower, self.upper)
        jac_position = np.zeros((3, self.model.nv), dtype=float)
        jac_rotation = np.zeros((3, self.model.nv), dtype=float)
        for _ in range(self.max_iterations):
            self._set_arm(q)
            mujoco.mj_forward(self.model, self.scratch)
            current_position = self.scratch.xpos[self.body_id]
            current_rotation = self.scratch.xmat[self.body_id].reshape(3, 3)
            position_error = target_position - current_position
            orientation_error = _rotation_error(current_rotation, target_rotation)
            if (float(np.linalg.norm(position_error)) <= self.position_tolerance
                    and float(np.linalg.norm(orientation_error)) <= self.orientation_tolerance):
                return q.copy()
            mujoco.mj_jacBody(self.model, self.scratch, jac_position, jac_rotation, self.body_id)
            jacobian = np.vstack((jac_position[:, self.dof_adrs], jac_rotation[:, self.dof_adrs]))
            error = np.concatenate((position_error, orientation_error))
            normal = jacobian @ jacobian.T + self.damping * np.eye(6)
            try:
                delta = jacobian.T @ np.linalg.solve(normal, error)
            except np.linalg.LinAlgError:
                return None
            # Limit individual Newton steps to keep iterations stable near a
            # joint boundary and to avoid jumping through a collision wall.
            q = np.clip(q + np.clip(delta, -0.18, 0.18), self.lower, self.upper)
        return None


def _apply_hand_qpos(model: mujoco.MjModel, data: mujoco.MjData,
                     hand_qpos: Sequence[float] | Mapping[str, float]) -> None:
    """Apply declared hand joint positions to scratch data only."""
    if isinstance(hand_qpos, Mapping):
        for joint_name, value in hand_qpos.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
            if joint_id < 0:
                raise ValueError(f"unknown hand joint {joint_name!r}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"hand joint {joint_name!r} is not a hinge")
            qadr = int(model.jnt_qposadr[joint_id])
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("hand qpos must be finite")
            data.qpos[qadr] = np.clip(value, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
        return
    values = np.asarray(hand_qpos, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("hand qpos must be finite")
    hand_joints = [index for index in range(model.njnt)
                   if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
                   and _name(model, mujoco.mjtObj.mjOBJ_JOINT, index) not in ARM_JOINT_NAMES]
    if values.size != len(hand_joints):
        raise ValueError(f"hand qpos must contain {len(hand_joints)} hinge values or a name mapping")
    for value, joint_id in zip(values, hand_joints):
        data.qpos[model.jnt_qposadr[joint_id]] = np.clip(
            value, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])


@dataclass(frozen=True)
class HeldToolProxy:
    """Declared conservative geometry attached to the wrist while planning."""

    half_extents: tuple[float, float, float]
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_id: str = "held_tool"
    source: str = "declared_tool_geometry"

    def __post_init__(self) -> None:
        extents = _vector(self.half_extents, 3, "half_extents")
        local = _vector(self.local_position, 3, "local_position")
        quat = _unit_quaternion(self.local_quaternion, "local_quaternion")
        if np.any(extents <= 0):
            raise ValueError("held tool half_extents must be positive")
        if not self.object_id or not self.source:
            raise ValueError("held tool object_id and source must be non-empty")
        object.__setattr__(self, "half_extents", tuple(float(value) for value in extents))
        object.__setattr__(self, "local_position", tuple(float(value) for value in local))
        object.__setattr__(self, "local_quaternion", tuple(float(value) for value in quat))

    @property
    def bounding_radius(self) -> float:
        return float(np.linalg.norm(np.asarray(self.half_extents)))


@dataclass(frozen=True)
class CollisionReport:
    collision_free: bool
    minimum_distance: float
    collisions: tuple[tuple[str, str, float], ...] = ()
    target_contacts: tuple[tuple[str, str, float], ...] = ()
    checked_pairs: int = 0

    @property
    def clearance(self) -> float:
        return float(self.minimum_distance)


@dataclass(frozen=True)
class PlanningResult:
    path: tuple[np.ndarray, ...]
    status: str
    clearance: float
    request_scope: str = "declared_map"
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "exact_solution"


class CollisionChecker:
    """Collision checker over a declared scene snapshot.

    ``sceneconfig`` is normally :class:`V7SceneConfig`; ``explicit_geom_ids``
    can be supplied by a perception adapter as a mapping of geometry groups.
    Both describe what is checked.  Current MuJoCo poses are copied into the
    private scratch state, never read from a controller's live data object.
    """

    def __init__(self, model: mujoco.MjModel, sceneconfig: object | None = None,
                 explicit_geom_ids: Mapping[str, Iterable[int | str]] | None = None,
                 *, safety_margin: float = 0.004) -> None:
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.safety_margin = float(safety_margin)
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError("safety_margin must be finite and non-negative")
        link7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, DEFAULT_EE_BODY)
        if link7 < 0:
            raise ValueError("model is missing link7")
        self.robot_bodies = _body_ancestors(model, link7)
        # Sharpa is attached below link7; ancestor-only traversal omits every
        # palm/finger collision geometry in the real vendor model.
        for body in range(1, model.nbody):
            if link7 in _body_ancestors(model, body):
                self.robot_bodies.add(body)
        # The vendor pedestal/base is a fixed mounting body.  Including its
        # own geometry would report the intended base-to-pedestal contact as a
        # path collision, so moving robot checks start at link1.
        self.robot_bodies = {
            body for body in self.robot_bodies
            if _name(model, mujoco.mjtObj.mjOBJ_BODY, body) not in {"base", "world"}
        }
        self.robot_geom_ids = tuple(
            gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in self.robot_bodies
            and int(model.geom_contype[gid]) != 0
        )
        self.hand_geom_ids = tuple(gid for gid in self.robot_geom_ids
                                   if any(token in _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid).lower()
                                          for token in ("palm", "fingertip", "sharpa", "left_")))
        self.finger_geom_ids = tuple(gid for gid in self.hand_geom_ids
                                     if "fingertip" in _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid).lower())
        self.fixed_geom_ids = tuple(
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == 0 and int(model.geom_contype[gid]) != 0
        )
        self.object_geom_ids = tuple(
            gid for gid in range(model.ngeom)
            if _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])).startswith("v7_obj_")
            or _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) == "target"
        )
        self.groups = {"robot": self.robot_geom_ids, "hand": self.hand_geom_ids,
                       "fingers": self.finger_geom_ids, "fixed": self.fixed_geom_ids,
                       "objects": self.object_geom_ids}
        if explicit_geom_ids is not None:
            for group, values in explicit_geom_ids.items():
                self.groups[str(group)] = tuple(self._resolve_geom(value) for value in values)
        self.sceneconfig = sceneconfig
        self._held_model = copy.copy(model)
        self._held_data = mujoco.MjData(self._held_model)
        self._proxy_geom_id = self.robot_geom_ids[0] if self.robot_geom_ids else None

    def _resolve_geom(self, value: int | str) -> int:
        if isinstance(value, str):
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, value)
        else:
            index = int(value)
        if index < 0 or index >= self.model.ngeom:
            raise ValueError(f"unknown geometry {value!r}")
        return index

    def _target_geom_ids(self, selected_target: str | int | Iterable[int | str] | None) -> set[int]:
        if selected_target is None:
            return set()
        if isinstance(selected_target, (str, int)):
            value = selected_target
            if isinstance(value, str):
                # Accept either the declared object ID or a model geom/body name.
                candidates = [f"v7_geom_{value}", value]
                geom = next((mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
                             for candidate in candidates
                             if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate) >= 0), -1)
                if geom >= 0:
                    return {geom}
                body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, value)
                if body >= 0:
                    return {gid for gid in self.object_geom_ids if int(self.model.geom_bodyid[gid]) == body}
                return set()
            return {self._resolve_geom(value)}
        return {self._resolve_geom(value) for value in selected_target}

    def _object_radius(self, gid: int) -> float:
        geom_type = int(self.model.geom_type[gid])
        size = np.asarray(self.model.geom_size[gid], dtype=float)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(size[0])
        if geom_type in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            return float(np.linalg.norm(size[:3]))
        return float(size[0] + size[1])

    def _heldtool_distance(self, joints: Sequence[float], heldtool: HeldToolProxy,
                           ignored_geom_ids: Iterable[int] = ()) -> float:
        if self._proxy_geom_id is None:
            raise ValueError("held tool planning requires a robot collision geometry")
        wrist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DEFAULT_EE_BODY)
        wrist_rotation = self.scratch.xmat[wrist].reshape(3, 3)
        center = self.scratch.xpos[wrist] + wrist_rotation @ np.asarray(heldtool.local_position)
        rotation = wrist_rotation @ _quat_to_matrix(heldtool.local_quaternion)
        proxy = self._proxy_geom_id
        model, data = self._held_model, self._held_data
        # MuJoCo model arrays are backed by native storage; copy.copy(model)
        # is shallow. Restore every temporary proxy field so a planning query
        # cannot silently change the caller's live scene geometry.
        old_type = int(model.geom_type[proxy])
        old_dataid = int(model.geom_dataid[proxy])
        old_size = model.geom_size[proxy].copy()
        old_rbound = float(model.geom_rbound[proxy])
        try:
            model.geom_type[proxy] = mujoco.mjtGeom.mjGEOM_BOX
            model.geom_dataid[proxy] = -1
            model.geom_size[proxy] = heldtool.half_extents
            model.geom_rbound[proxy] = heldtool.bounding_radius
            data.qpos[:] = self.scratch.qpos
            mujoco.mj_forward(model, data)
            # Only private collision data is transformed. Native geometry
            # distance handles the oriented proxy against scene geometry.
            data.geom_xpos[proxy] = center
            data.geom_xmat[proxy] = rotation.reshape(9)
            minimum = math.inf
            ignored = set(ignored_geom_ids) | set(self._target_geom_ids(heldtool.object_id))
            obstacles = (set(self.groups.get("fixed", self.fixed_geom_ids))
                         | set(self.groups.get("obstacles", ()))
                         | set(self.groups.get("objects", self.object_geom_ids))
                         | set(self.groups.get("non_target", ())))
            for gid in obstacles:
                if gid in ignored or gid == proxy:
                    continue
                distance = mujoco.mj_geomDistance(model, data, proxy, gid, 1.0, None)
                minimum = min(minimum, float(distance))
            return minimum
        finally:
            model.geom_type[proxy] = old_type
            model.geom_dataid[proxy] = old_dataid
            model.geom_size[proxy] = old_size
            model.geom_rbound[proxy] = old_rbound

    def check(self, joints: Sequence[float], hand_positions: Sequence[float] | Mapping[str, float] | None = None,
              selected_target: str | int | Iterable[int | str] | None = None, *,
              approach: bool = True, heldtool: HeldToolProxy | None = None) -> CollisionReport:
        values = _vector(joints, 7, "joints")
        qpos = np.asarray(self.scratch.qpos)
        # The caller supplies a configuration snapshot, not the planner's
        # mutable live data.  Other qpos values remain at the declared scene
        # initial state in this private data object.
        for index, name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"model is missing {name}")
            qpos[self.model.jnt_qposadr[joint_id]] = values[index]
        if hand_positions is not None:
            _apply_hand_qpos(self.model, self.scratch, hand_positions)
        mujoco.mj_forward(self.model, self.scratch)
        target_ids = self._target_geom_ids(selected_target)
        robot_ids = set(self.groups.get("robot", self.robot_geom_ids))
        fixed_ids = set(self.groups.get("fixed", self.fixed_geom_ids)) | set(self.groups.get("obstacles", ()))
        object_ids = set(self.groups.get("objects", self.object_geom_ids)) | set(self.groups.get("non_target", ()))
        finger_ids = set(self.groups.get("fingers", self.finger_geom_ids))
        collisions: list[tuple[str, str, float]] = []
        target_contacts: list[tuple[str, str, float]] = []
        minimum = math.inf
        checked = 0
        # Check robot against fixed geometry and all non-target objects.  Pair
        # filtering here intentionally differs from MuJoCo contact exclusions:
        # a target is allowed only against fingertips during approach.
        obstacle_ids = fixed_ids | object_ids
        if heldtool is not None:
            # A carried target has left its declared map pose. Its attached
            # proxy is checked below, so its old geometry is not an obstacle.
            obstacle_ids -= target_ids | self._target_geom_ids(heldtool.object_id)
        for robot_gid in robot_ids:
            for obstacle_gid in obstacle_ids:
                if robot_gid == obstacle_gid:
                    continue
                if obstacle_gid in target_ids and approach and robot_gid in finger_ids:
                    continue
                checked += 1
                distance = float(mujoco.mj_geomDistance(self.model, self.scratch,
                                                        robot_gid, obstacle_gid, 1.0, None))
                minimum = min(minimum, distance)
                if distance < self.safety_margin:
                    pair = (_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_gid),
                            _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_gid), distance)
                    if obstacle_gid in target_ids:
                        target_contacts.append(pair)
                        # Target contact is permitted only for a fingertip in
                        # the declared approach phase.  A target touching a
                        # palm/arm (or during transfer) is a collision.
                        if not (approach and robot_gid in finger_ids):
                            collisions.append(pair)
                    else:
                        collisions.append(pair)
        if heldtool is not None:
            held_distance = self._heldtool_distance(values, heldtool, target_ids)
            minimum = min(minimum, held_distance)
            if held_distance < self.safety_margin:
                collisions.append((heldtool.object_id, "declared_scene_geometry", held_distance))
        if not math.isfinite(minimum):
            minimum = math.inf
        return CollisionReport(not collisions, minimum, tuple(collisions), tuple(target_contacts), checked)

    def is_collision_free(self, *args, **kwargs) -> bool:
        return bool(self.check(*args, **kwargs).collision_free)

    def plan(self, start: Sequence[float], goal: Sequence[float],
             hand_positions: Sequence[float] | Mapping[str, float] | None = None,
             selected_target: str | int | Iterable[int | str] | None = None,
             timebudget: float = 1.0, **kwargs) -> PlanningResult:
        """Convenience entry point using the OMPL RRTConnect adapter."""
        return RRTConnectPlanner(self).plan(start, goal, hand_positions,
                                             selected_target, timebudget, **kwargs)


class RRTConnectPlanner:
    """OMPL RRTConnect adapter for KUKA joint-space paths."""

    def __init__(self, checker: CollisionChecker, *, environment_scope: str = "declared_map") -> None:
        self.checker = checker
        self.environment_scope = str(environment_scope)
        if self.environment_scope not in {"declared_map", "rgbd", "simulation_snapshot"}:
            raise ValueError("environment_scope must be declared_map, rgbd, or simulation_snapshot")
        self.kinematics = V7Kinematics(checker.model)

    def plan(self, start: Sequence[float], goal: Sequence[float],
             hand_positions: Sequence[float] | Mapping[str, float] | None = None,
             selected_target: str | int | Iterable[int | str] | None = None,
             timebudget: float = 1.0, *, scene_snapshot: Mapping[str, object] | None = None,
             environment_scope: str | None = None, approach: bool = True,
             heldtool: HeldToolProxy | None = None) -> PlanningResult:
        start_values = _vector(start, 7, "start")
        goal_values = _vector(goal, 7, "goal")
        if np.any(start_values < self.kinematics.lower) or np.any(start_values > self.kinematics.upper):
            raise ValueError("start is outside KUKA model limits")
        if np.any(goal_values < self.kinematics.lower) or np.any(goal_values > self.kinematics.upper):
            raise ValueError("goal is outside KUKA model limits")
        budget = float(timebudget)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("timebudget must be finite and positive")
        scope = self.environment_scope if environment_scope is None else str(environment_scope)
        if scope not in {"declared_map", "rgbd", "simulation_snapshot"}:
            raise ValueError("environment_scope must be declared_map, rgbd, or simulation_snapshot")
        if scope == "rgbd" and scene_snapshot is None:
            raise ValueError("rgbd planning requires an explicit scene_snapshot")
        start_report = self.checker.check(start_values, hand_positions, selected_target,
                                          approach=approach, heldtool=heldtool)
        goal_report = self.checker.check(goal_values, hand_positions, selected_target,
                                         approach=approach, heldtool=heldtool)
        if not start_report.collision_free or not goal_report.collision_free:
            return PlanningResult((), "invalid_start_or_goal", min(start_report.clearance, goal_report.clearance), scope,
                                  {"start": start_report, "goal": goal_report})
        try:
            import ompl.base as ob
            import ompl.geometric as og
        except ImportError as error:
            return PlanningResult((), "ompl_unavailable", min(start_report.clearance, goal_report.clearance), scope,
                                  {"error": str(error)})
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for index, (lower, upper) in enumerate(zip(self.kinematics.lower, self.kinematics.upper)):
            bounds.setLow(index, float(lower))
            bounds.setHigh(index, float(upper))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        setup.getSpaceInformation().setStateValidityCheckingResolution(.001)

        def valid(state) -> bool:
            values = np.asarray([state[index] for index in range(7)], dtype=float)
            return self.checker.is_collision_free(values, hand_positions,
                                                  selected_target, approach=approach, heldtool=heldtool)

        # The current OMPL Python wheel exposes the callback through a
        # StateValidityChecker subclass (older wheels had Fn aliases).  This
        # adapter keeps all validity calls in CollisionChecker's scratch state.
        class _ValidityChecker(ob.StateValidityChecker):
            def __init__(self, space_information):
                super().__init__(space_information)

            def isValid(self, state):  # noqa: N802 - OMPL's Python API name
                return valid(state)

        setup.setStateValidityChecker(_ValidityChecker(setup.getSpaceInformation()))
        ompl_start = space.allocState()
        ompl_goal = space.allocState()
        for index in range(7):
            ompl_start[index] = float(start_values[index])
            ompl_goal[index] = float(goal_values[index])
        setup.setStartAndGoalStates(ompl_start, ompl_goal)
        setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
        solved = setup.solve(budget)
        if not solved:
            return PlanningResult((), "timeout", min(start_report.clearance, goal_report.clearance), scope,
                                  {"scene_snapshot": scene_snapshot or {}})
        path_obj = setup.getSolutionPath()
        path_obj.interpolate()
        sparse_path = tuple(np.asarray([path_obj.getState(index)[joint] for joint in range(7)], dtype=float)
                            for index in range(path_obj.getStateCount()))
        dense_path = [sparse_path[0]] if sparse_path else []
        for first, second in zip(sparse_path, sparse_path[1:]):
            segments = max(1, int(math.ceil(float(np.max(np.abs(second - first))) / .005)))
            dense_path.extend(first + (second - first) * index / segments
                              for index in range(1, segments + 1))
        path = tuple(dense_path)
        reports = tuple(self.checker.check(q, hand_positions, selected_target,
                                           approach=approach, heldtool=heldtool) for q in path)
        clearance = min((report.clearance for report in reports), default=math.inf)
        if not all(report.collision_free for report in reports):
            return PlanningResult((), "post_validation_failed", float(clearance), scope,
                                  {"planner": "OMPL_RRTConnect", "validated_states": len(reports)})
        status = "exact_solution" if solved == ob.PlannerStatus.EXACT_SOLUTION else "approximate_solution"
        return PlanningResult(path, status, float(clearance), scope,
                              {"scene_snapshot": scene_snapshot or {}, "planner": "OMPL_RRTConnect",
                               "validated_states": len(reports), "max_joint_segment_rad": .005})


V7Planner = RRTConnectPlanner

__all__ = ["ARM_JOINT_NAMES", "CollisionChecker", "CollisionReport", "HeldToolProxy",
           "PlanningResult", "RRTConnectPlanner", "V7Kinematics", "V7Planner"]
