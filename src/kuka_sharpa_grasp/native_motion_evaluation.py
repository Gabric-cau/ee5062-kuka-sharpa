"""Read target-privileged strict labels after a caller's native mj_forward.

This adapter does not step, forward, change state or write a motor command.
Its output belongs exclusively to evaluation, never grasp planning or task
phase selection. The caller must run mj_forward at the integrated qpos/qvel
immediately before sampling. An expected timestamp catches mismatched times,
not a stale dynamics cache: a timestamp alone cannot prove freshness.
"""

from __future__ import annotations

from numbers import Real

import mujoco
import numpy as np

from .strict_motion_monitor import CHANNELS, StrictMotionSample


def _finite(value, shape, name):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + " must have finite real values and the declared shape")
    return result.astype(float, copy=True)


class NativeMotionEvaluator:
    """Bind target/table identities for evaluation, not simulated perception.

    Channels sum absolute native normal forces from every external contact of
    a ``left_`` hand geometry. Internal hand-hand contacts are excluded. Loaded
    target-hand/table tests use maximum absolute per-contact normals, preserving
    the strict monitor's any-contact > .05 N definition. Clearance covers every
    collision-enabled hand geometry (nonzero contype or conaffinity), and is
    capped at 1 m by the distance query, well beyond the .10 m release gate.
    Robot-table minimum signed distance covers contacts involving named link/
    hand bodies and all robot ancestors of the hand. It is not a clutter gate.
    """

    def __init__(self, model, *, target_body_name="target", target_geom_name="target_geom",
                 table_geom_name="table"):
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
        target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom_name)
        if target <= 0 or target_geom < 0 or table < 0 or target_geom == table:
            raise ValueError("distinct target body/geometry and table geometry are required")
        owner = int(model.geom_bodyid[target_geom])
        while owner and owner != target:
            owner = int(model.body_parentid[owner])
        if owner != target:
            raise ValueError("target geometry must descend from the target body")
        channels = {}
        hand_bodies = set()
        robot_bodies = set()
        for body in range(1, model.nbody):
            name = model.body(body).name or ""
            if name.startswith(("link", "left_")):
                robot_bodies.add(body)
            if name.startswith("left_"):
                hand_bodies.add(body)
                ancestor = body
                while ancestor:
                    robot_bodies.add(ancestor)
                    ancestor = int(model.body_parentid[ancestor])
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if body not in hand_bodies:
                continue
            name = model.body(body).name or ""
            channels[geom] = next((finger for finger in CHANNELS[:-1]
                                  if name.startswith("left_" + finger + "_")), "palm")
        clearance = tuple(g for g in channels if model.geom_contype[g] or model.geom_conaffinity[g])
        if not clearance or target_geom in channels or table in channels:
            raise ValueError("collision hand geometry must be separate from target and table")
        self.model = model
        self.target_body_id, self.target_geom_id, self.table_geom_id = target, target_geom, table
        self._channels = channels
        self._clearance_geoms = clearance
        self._robot_geoms = frozenset(g for g in range(model.ngeom)
                                     if int(model.geom_bodyid[g]) in robot_bodies)

    def sample_after_forward(self, data, *, expected_timestamp_s, release_commanded, phase):
        """Read one caller-refreshed native sample; leave live data unchanged.

        Native step-start force caches are intentionally unsupported: the runner
        retains those separately for force peaks and motor-prediction auditing.
        Call at every integrated 2 ms timestamp after mj_forward, including
        failures/recovery. The strict monitor handles continuity and durations.
        """
        if (isinstance(expected_timestamp_s, (bool, np.bool_))
                or not isinstance(expected_timestamp_s, Real)
                or not np.isfinite(expected_timestamp_s) or expected_timestamp_s < 0.):
            raise ValueError("expected_timestamp_s must be finite and nonnegative")
        if not np.isfinite(data.time) or abs(float(data.time) - float(expected_timestamp_s)) > 1e-9:
            raise ValueError("evaluation timestamp must match the refreshed native sample")
        if type(release_commanded) is not bool or not isinstance(phase, str) or not phase:
            raise ValueError("release flag must be a boolean and phase a nonempty string")
        position = _finite(data.xpos[self.target_body_id], (3,), "target position")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.target_body_id, velocity, 0)
        velocity = _finite(velocity, (6,), "target world spatial velocity")
        totals = dict.fromkeys(CHANNELS, 0.)
        target_hand_contact, hand_peak, table_peak, minimum_table_distance = False, 0., 0., None
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            first, second = int(contact.geom1), int(contact.geom2)
            mujoco.mj_contactForce(self.model, data, index, wrench)
            normal = abs(float(_finite(wrench, (6,), "native contact wrench")[0]))
            distance = float(_finite(contact.dist, (), "native contact distance"))
            first_hand, second_hand = first in self._channels, second in self._channels
            if first_hand != second_hand:
                hand_geom = first if first_hand else second
                totals[self._channels[hand_geom]] += normal
                if self.target_geom_id in (first, second):
                    target_hand_contact = True
                    hand_peak = max(hand_peak, normal)
            if {first, second} == {self.target_geom_id, self.table_geom_id}:
                table_peak = max(table_peak, normal)
            if self.table_geom_id in (first, second):
                other = second if first == self.table_geom_id else first
                if other in self._robot_geoms:
                    minimum_table_distance = (distance if minimum_table_distance is None
                                              else min(minimum_table_distance, distance))
        clearance = min(float(mujoco.mj_geomDistance(self.model, data, self.target_geom_id,
                            geom, 1., None)) for geom in self._clearance_geoms)
        return StrictMotionSample(timestamp_s=float(data.time), target_position_world_m=tuple(position),
            target_linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
            target_angular_speed_rad_s=float(np.linalg.norm(velocity[:3])),
            target_hand_contact=target_hand_contact, target_hand_peak_normal_n=hand_peak,
            target_table_peak_normal_n=table_peak, hand_clearance_m=clearance,
            hand_channel_forces_n=tuple(totals[name] for name in CHANNELS),
            minimum_robot_table_distance_m=minimum_table_distance,
            release_commanded=release_commanded, phase=phase)
