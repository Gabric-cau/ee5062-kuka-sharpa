"""Robot-only wrist task and tactile rigid-transport references.

This module does not step physics or write a motor command. Its soft task
velocity is an input to coordinated_admittance, whose joint, rate, deflection
and frozen-state motor-torque boxes still apply. No contact force, retention or
object-motion guarantee follows from a kinematic reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_policy import ContactObservation


def _array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and real numeric entries")
    result = np.asarray(array, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, positive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if value < 0. or (positive and value == 0.):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _tuple(array):
    return tuple(float(value) for value in array)


def _limit_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


@dataclass(frozen=True)
class MotionReference:
    """One sample, all tuples immutable; world linear/angular rows are separate.

    The wrist Jacobian has three m/rad linear rows then three rad/rad angular
    rows. Pose errors are desired minus actual translation and the world-frame
    rotation vector log(R_desired R_actual.T). Transport uses the achieved
    DLS/rate-limited wrist twist, not an unreachable requested twist. Patches
    retain their validated observation order and original native input indices.
    """

    timestamp_s: float
    joint_names: tuple[str, ...]
    patch_input_indices: tuple[int, ...]
    wrist_position_world_m: tuple[float, ...]
    wrist_rotation_world_rows: tuple[tuple[float, ...], ...]
    wrist_jacobian_world: tuple[tuple[float, ...], ...]
    position_error_world_m: tuple[float, ...]
    orientation_error_world_rad: tuple[float, ...]
    requested_linear_velocity_world_m_s: tuple[float, ...]
    requested_angular_velocity_world_rad_s: tuple[float, ...]
    limited_linear_velocity_world_m_s: tuple[float, ...]
    limited_angular_velocity_world_rad_s: tuple[float, ...]
    achieved_linear_velocity_world_m_s: tuple[float, ...]
    achieved_angular_velocity_world_rad_s: tuple[float, ...]
    task_reference_velocity_rad_s: tuple[float, ...]
    rigid_transport_normal_velocity_m_s: tuple[float, ...]
    joint_velocity_scale: float
    linear_velocity_residual_world_m_s: tuple[float, ...]
    angular_velocity_residual_world_rad_s: tuple[float, ...]


class CoordinatedMotionReference:
    """Use actual wrist-ancestor hinge positions in a private kinematic scratch.

    No live object qpos/qvel, external pose, contact dynamics, model material or
    material estimate is read. Only mj_kinematics, mj_comPos and mj_jacBody run.
    Finger columns are present to match the coordinated solver's joint order,
    but have zero wrist Jacobian and zero soft task velocity.
    """

    def __init__(self, model, *, joint_names, wrist_body_name="link7"):
        names = tuple(joint_names)
        if (not names or isinstance(joint_names, (str, bytes))
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("joint_names must be distinct nonempty robot joint names")
        wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body_name)
        if wrist <= 0:
            raise ValueError("A named robot wrist body is required")
        ancestors = set()
        body = wrist
        while body:
            ancestors.add(body)
            body = int(model.body_parentid[body])
        ancestor_joints = [joint for joint in range(model.njnt)
            if int(model.jnt_bodyid[joint]) in ancestors]
        if any(model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE for joint in ancestor_joints):
            raise ValueError("Only hinge robot ancestors are supported")
        joints = []
        for name in names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Each controlled coordinate must be a named robot hinge")
            body = int(model.jnt_bodyid[joint])
            if body not in ancestors:
                if not (model.body(body).name or "").startswith("left_") or not self._descends_from(model, body, wrist):
                    raise ValueError("Controlled joints must be wrist ancestors or descendant hand joints")
            joints.append(joint)
        self.model, self.joint_names, self.wrist_body_id = model, names, wrist
        self._qpos_indices = tuple(int(model.jnt_qposadr[joint]) for joint in ancestor_joints)
        self._dof_indices = tuple(int(model.jnt_dofadr[joint]) for joint in joints)
        self._scratch = mujoco.MjData(model)

    @staticmethod
    def _descends_from(model, body, ancestor):
        while body:
            if body == ancestor:
                return True
            body = int(model.body_parentid[body])
        return False

    def prepare(self, data, observation: ContactObservation, *, desired_linear_velocity_world_m_s,
                joint_velocity_limits_rad_s, desired_angular_velocity_world_rad_s=(0., 0., 0.),
                desired_position_world_m=None, desired_rotation_world=None,
                position_gain_s_inv=1., orientation_gain_s_inv=1., max_linear_speed_m_s=.004,
                max_angular_speed_rad_s=.05, damping_m_rad=1e-4, rotational_length_scale_m=.1):
        """Build current-sample references without integrating any pose reference.

        The caller advances its declared pose trajectory at native sample time.
        Desired feedforward twist plus proportional pose correction is norm
        limited, mapped by weighted damped least squares, then uniformly scaled
        to declared joint speed limits. Those are soft-reference limits, not a
        replacement for the downstream servo command box. A caller must retain
        its force/contact/support/held and Cartesian tracking rejection gates.
        """
        if not isinstance(observation, ContactObservation):
            raise ValueError("A validated ContactObservation is required")
        timestamp = _scalar(data.time, "data.time", positive=False)
        observed_at = _scalar(observation.timestamp_s, "observation.timestamp_s", positive=False)
        if abs(timestamp - observed_at) > 1e-9:
            raise ValueError("Observation must belong to the current native sample")
        linear = _array(desired_linear_velocity_world_m_s, (3,), "desired linear velocity")
        angular = _array(desired_angular_velocity_world_rad_s, (3,), "desired angular velocity")
        limits = _array(joint_velocity_limits_rad_s, (len(self.joint_names),), "joint velocity limits")
        if np.any(limits < 0.):
            raise ValueError("joint velocity limits must be nonnegative")
        position_gain = _scalar(position_gain_s_inv, "position gain", positive=False)
        orientation_gain = _scalar(orientation_gain_s_inv, "orientation gain", positive=False)
        max_linear = _scalar(max_linear_speed_m_s, "maximum linear speed")
        max_angular = _scalar(max_angular_speed_rad_s, "maximum angular speed")
        damping = _scalar(damping_m_rad, "DLS damping")
        rotation_scale = _scalar(rotational_length_scale_m, "rotational length scale")
        scratch = self._scratch
        scratch.qpos[list(self._qpos_indices)] = _array(data.qpos[list(self._qpos_indices)],
            (len(self._qpos_indices),), "robot wrist-ancestor positions")
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        position = scratch.xpos[self.wrist_body_id].copy()
        rotation = scratch.xmat[self.wrist_body_id].reshape(3, 3).copy()
        position_error, orientation_error = np.zeros(3), np.zeros(3)
        if desired_position_world_m is not None:
            position_error = _array(desired_position_world_m, (3,), "desired wrist position") - position
        if desired_rotation_world is not None:
            desired_rotation = _array(desired_rotation_world, (3, 3), "desired wrist rotation")
            if (not np.allclose(desired_rotation.T @ desired_rotation, np.eye(3), rtol=0., atol=1e-8)
                    or abs(float(np.linalg.det(desired_rotation)) - 1.) > 1e-8):
                raise ValueError("desired wrist rotation must be a proper rotation matrix")
            orientation_error = Rotation.from_matrix(desired_rotation @ rotation.T).as_rotvec()
        requested_linear = linear + position_gain * position_error
        requested_angular = angular + orientation_gain * orientation_error
        limited_linear = _limit_norm(requested_linear, max_linear)
        limited_angular = _limit_norm(requested_angular, max_angular)
        jac_linear, jac_angular = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, scratch, jac_linear, jac_angular, self.wrist_body_id)
        jacobian = np.vstack([jac_linear[:, self._dof_indices], jac_angular[:, self._dof_indices]])
        weighted = np.vstack([jacobian[:3], rotation_scale * jacobian[3:]])
        desired = np.r_[limited_linear, rotation_scale * limited_angular]
        velocity = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping ** 2 * np.eye(6), desired)
        if not np.isfinite(velocity).all():
            raise ValueError("Nonfinite damped joint task velocity")
        ratios = np.divide(limits, np.abs(velocity), out=np.ones_like(limits), where=velocity != 0.)
        scale = float(min(1., np.min(ratios)))
        velocity *= scale
        achieved = jacobian @ velocity
        indices, transport = [], []
        for patch in observation.patches:
            index = patch.input_index
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or index < 0 or index in indices:
                raise ValueError("Patch input indices must be distinct nonnegative integers")
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, patch.hand_body_name)
            if (body <= 0 or not patch.hand_body_name.startswith(f"left_{patch.finger}_")
                    or not self._descends_from(self.model, body, self.wrist_body_id)):
                raise ValueError("Patch must belong to the descendant robot hand")
            point = _array(patch.position_world_m, (3,), "tactile point")
            normal = _array(patch.normal_on_object_world, (3,), "tactile normal")
            length = float(np.linalg.norm(normal))
            if length <= 1e-12:
                raise ValueError("Tactile normal must have positive norm")
            rigid_velocity = achieved[:3] + np.cross(achieved[3:], point - position)
            transport.append(float(np.dot(normal / length, rigid_velocity)))
            indices.append(int(index))
        values = np.r_[position, rotation.ravel(), jacobian.ravel(), position_error, orientation_error,
            requested_linear, requested_angular, limited_linear, limited_angular, velocity, achieved, transport]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite motion reference")
        return MotionReference(timestamp, self.joint_names, tuple(indices), _tuple(position),
            tuple(_tuple(row) for row in rotation), tuple(_tuple(row) for row in jacobian),
            _tuple(position_error), _tuple(orientation_error), _tuple(requested_linear), _tuple(requested_angular),
            _tuple(limited_linear), _tuple(limited_angular), _tuple(achieved[:3]), _tuple(achieved[3:]),
            _tuple(velocity), tuple(transport), scale, _tuple(achieved[:3] - limited_linear),
            _tuple(achieved[3:] - limited_angular))
