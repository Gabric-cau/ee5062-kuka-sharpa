"""Robot-only kinematics and motor bindings for coordinated contact references.

Native contact patches are simulated tactile observations, not object identity
or pose estimates. The Jacobian holds the external surface fixed locally.
Returned commands are actual position-servo inputs, including any feedforward
offset the caller chooses; adding gravity compensation afterwards invalidates
the reported frozen-state torque prediction. This module does not step physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np

from .contact_policy import CONTACT_CHANNELS, ContactObservation, FingerContactPolicy
from .tactile import TactileContactReader


@dataclass(frozen=True)
class NativeContactSample:
    timestamp_s: float
    patches: tuple[Mapping, ...]
    measured_pressures_n: Mapping[str, float]


def read_native_contact_sample(model, data, *, hand_body_prefix="left_"):
    """Keep every external hand contact without inspecting its other body.

    The contact frame normal points from geom 0 to geom 1. Its sign is selected
    from the hand side, so a positive point velocity in that direction means
    compression into the external surface. Material friction is deliberately
    not read from contact/model data: an allocator must use a declared prior.
    Contacts/forces are the most recent MuJoCo dynamics cache, just as in the
    existing tactile reader; timestamp_s is the observation time, not a claim
    of a force recomputation at the integrated qpos.
    """
    reader = TactileContactReader(model, hand_body_prefix=hand_body_prefix)
    channels = reader.geometry_channels
    hand_geoms = {gid for gid in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[gid])).name or "").startswith(hand_body_prefix)}
    totals = dict.fromkeys(CONTACT_CHANNELS, 0.)
    patches = []
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = map(int, contact.geom)
        if (first in hand_geoms) == (second in hand_geoms):
            continue
        hand_geom = first if first in hand_geoms else second
        if hand_geom not in channels:
            raise ValueError("External hand contact has no tactile channel")
        finger = channels[hand_geom]
        mujoco.mj_contactForce(model, data, index, wrench)
        pressure = abs(float(wrench[0]))
        point = np.asarray(contact.pos, dtype=float)
        normal = np.asarray(contact.frame[:3], dtype=float) * (1. if first == hand_geom else -1.)
        if not np.isfinite(np.r_[pressure, point, normal]).all():
            raise ValueError("Nonfinite native contact observation")
        totals[finger] += pressure
        patches.append(MappingProxyType({
            "input_index": index, "finger": finger,
            "hand_body_name": model.body(int(model.geom_bodyid[hand_geom])).name,
            "position_world_m": tuple(point.tolist()),
            "normal_on_object_world": tuple(normal.tolist()),
            "measured_pressure_n": pressure,
        }))
    if not np.isfinite([data.time, *totals.values()]).all():
        raise ValueError("Nonfinite tactile timestamp or channel sum")
    return NativeContactSample(float(data.time), tuple(patches), MappingProxyType(totals))


@dataclass(frozen=True)
class PositionServo:
    name: str
    joint_id: int
    actuator_id: int
    qpos_index: int
    dof_index: int
    lower_rad: float
    upper_rad: float
    kp_nm_rad: float
    kd_nm_s_rad: float
    lower_torque_nm: float
    upper_torque_nm: float


def _position_servo(model, name):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"Controlled joint requires a hinge position servo: {name}")
    actuators = [index for index in range(model.nu)
        if model.actuator_trntype[index] == mujoco.mjtTrn.mjTRN_JOINT
        and int(model.actuator_trnid[index, 0]) == joint]
    if len(actuators) != 1:
        raise ValueError(f"Controlled hinge requires exactly one direct position servo: {name}")
    actuator = actuators[0]
    gain, bias = model.actuator_gainprm[actuator], model.actuator_biasprm[actuator]
    gear = np.asarray(model.actuator_gear[actuator])
    if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
            or int(model.actuator_trnid[actuator, 0]) != joint
            or not np.array_equal(gear, [1., 0., 0., 0., 0., 0.])
            or model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
            or gain[0] <= 0. or bias[0] != 0. or bias[1] != -gain[0] or bias[2] > 0.
            or np.any(gain[1:] != 0.) or np.any(bias[3:] != 0.)
            or not np.isfinite(np.r_[gain, bias]).all()):
        raise ValueError(f"Unsupported position-servo transmission or force law: {name}")
    if not model.jnt_limited[joint]:
        raise ValueError(f"Controlled hinge needs finite position limits: {name}")
    lower, upper = map(float, model.jnt_range[joint])
    if model.actuator_ctrllimited[actuator]:
        lower = max(lower, float(model.actuator_ctrlrange[actuator, 0]))
        upper = min(upper, float(model.actuator_ctrlrange[actuator, 1]))
    torque_ranges = []
    if model.actuator_forcelimited[actuator]:
        torque_ranges.append(model.actuator_forcerange[actuator])
    if model.jnt_actfrclimited[joint]:
        torque_ranges.append(model.jnt_actfrcrange[joint])
    if not torque_ranges:
        raise ValueError(f"A declared motor torque limit is required: {name}")
    torque_lower = max(float(pair[0]) for pair in torque_ranges)
    torque_upper = min(float(pair[1]) for pair in torque_ranges)
    if (not np.isfinite([lower, upper, torque_lower, torque_upper]).all()
            or lower >= upper or torque_lower >= torque_upper):
        raise ValueError(f"Empty or nonfinite motor bounds: {name}")
    return PositionServo(name, joint, actuator, int(model.jnt_qposadr[joint]),
        int(model.jnt_dofadr[joint]), lower, upper, float(gain[0]), float(-bias[2]),
        torque_lower, torque_upper)


class ContactExecutionAdapter:
    """Bind one contact policy to real arm/hand position-servo coordinates.

    Only robot proprioception is copied to scratch data. No live object qpos,
    qvel, contact dynamics, or material coefficients enter the kinematic model.
    The supplied tactile points/normals remain local surface observations.
    """

    def __init__(self, model, policy: FingerContactPolicy, *, arm_joint_names=()):
        arm = tuple(arm_joint_names)
        if any(not isinstance(name, str) or not name for name in arm) or len(set(arm)) != len(arm):
            raise ValueError("Arm joint names must be distinct nonempty strings")
        if any(name.startswith("left_") for name in arm):
            raise ValueError("Arm and finger coordinate groups must be separate")
        policy.bind_mujoco_model(model)
        self.model, self.policy = model, policy
        self.joint_names = arm + policy.controlled_joint_names
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Duplicate controlled coordinate")
        self.servos = tuple(_position_servo(model, name) for name in self.joint_names)
        self._scratch = mujoco.MjData(model)
        hand_bodies = {index for index in range(model.nbody)
            if (model.body(index).name or "").startswith("left_")}
        ancestors = set(hand_bodies)
        for body in tuple(hand_bodies):
            while body:
                body = int(model.body_parentid[body])
                ancestors.add(body)
        if any(int(model.jnt_bodyid[servo.joint_id]) not in ancestors for servo in self.servos):
            raise ValueError("Controlled coordinates must belong to the hand or its robot ancestors")
        self._robot_qpos = tuple(int(model.jnt_qposadr[joint]) for joint in range(model.njnt)
            if int(model.jnt_bodyid[joint]) in ancestors
            and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE)

    def compression_jacobian(self, data, observation: ContactObservation):
        """Return J[P,N] in m/rad at integrated robot qpos in private scratch.

        J predicts hand-point motion into a locally fixed external surface.
        It omits external-body motion and contact compliance; it does not
        predict dynamic normal force or prove physical controllability.
        """
        timestamp = observation.timestamp_s
        if (timestamp is None or isinstance(timestamp, (bool, np.bool_))
                or not np.isfinite([data.time, timestamp]).all()
                or abs(float(data.time) - float(timestamp)) > 1e-9):
            raise ValueError("Contact observation must belong to the current native sample")
        checked = self.policy.validate_contacts(observation.as_allocator_patches(),
            observation.measured_pressures_n,
            controlled_joint_names=self.policy.controlled_joint_names)
        if tuple(patch.input_index for patch in checked.patches) != tuple(
                patch.input_index for patch in observation.patches):
            raise ValueError("Contact observation rows must already match the active policy")
        scratch = self._scratch
        scratch.qpos[list(self._robot_qpos)] = data.qpos[list(self._robot_qpos)]
        if not np.isfinite(scratch.qpos[list(self._robot_qpos)]).all():
            raise ValueError("Nonfinite robot proprioception")
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        columns = [servo.dof_index for servo in self.servos]
        jacobian = np.empty((len(checked.patches), len(columns)))
        point_jac = np.zeros((3, self.model.nv))
        for row, patch in enumerate(checked.patches):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, patch.hand_body_name)
            if body < 0 or not patch.hand_body_name.startswith(f"left_{patch.finger}_"):
                raise ValueError("Tactile patch has no matching hand body")
            mujoco.mj_jac(self.model, scratch, point_jac, None,
                np.asarray(patch.position_world_m), body)
            jacobian[row] = np.asarray(patch.normal_on_object_world) @ point_jac[:, columns]
        if not np.isfinite(jacobian).all():
            raise ValueError("Nonfinite compression Jacobian")
        return jacobian

    def solve(self, data, observation, target_normal_velocity_m_s, *,
              command_rate_lower_rad_s, command_rate_upper_rad_s,
              deflection_lower_rad, deflection_upper_rad,
              task_reference_velocity_rad_s=None, contact_weights=None,
              task_weights=None, maximum_contact_residual_m_s=1e-4):
        """Produce a candidate actual motor command; leave live data untouched."""
        from .coordinated_admittance import solve_coordinated_admittance

        jacobian = self.compression_jacobian(data, observation)
        qpos = [servo.qpos_index for servo in self.servos]
        dofs = [servo.dof_index for servo in self.servos]
        actuators = [servo.actuator_id for servo in self.servos]
        return solve_coordinated_admittance(jacobian, target_normal_velocity_m_s,
            actual_position_rad=data.qpos[qpos].copy(), actual_velocity_rad_s=data.qvel[dofs].copy(),
            previous_command_rad=data.ctrl[actuators].copy(), timestep_s=float(self.model.opt.timestep),
            joint_lower_rad=[servo.lower_rad for servo in self.servos],
            joint_upper_rad=[servo.upper_rad for servo in self.servos],
            servo_kp_nm_rad=[servo.kp_nm_rad for servo in self.servos],
            servo_kd_nm_s_rad=[servo.kd_nm_s_rad for servo in self.servos],
            torque_lower_nm=[servo.lower_torque_nm for servo in self.servos],
            torque_upper_nm=[servo.upper_torque_nm for servo in self.servos],
            command_rate_lower_rad_s=command_rate_lower_rad_s,
            command_rate_upper_rad_s=command_rate_upper_rad_s,
            deflection_lower_rad=deflection_lower_rad, deflection_upper_rad=deflection_upper_rad,
            task_reference_velocity_rad_s=task_reference_velocity_rad_s,
            contact_weights=contact_weights, task_weights=task_weights,
            maximum_contact_residual_m_s=maximum_contact_residual_m_s)
