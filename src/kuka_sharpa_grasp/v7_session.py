"""Physical V7 shared teleoperation session.

Only reset writes generalized positions. User commands change bounded motor
targets; contacts and object motion always come from MuJoCo integration.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .control import FINGERS, ForceReading
from .hand_grasp import HandConfig, HandGraspController
from .motion import ArmCommandLimiter
from .physics import arm_joint_acceleration_limits, arm_joint_velocity_limits
from .pick_place import RobotGravityCompensation, RobotKinematics
from .tactile import TactileContactReader
from .teleop import TeleopCommand


class WorkbenchSession:
    """One physical workbench with operator-selected catalogue identity.

    Scene geometry is a declared static workcell map, not inferred vision.
    Selecting an object does not supply its current pose to the controller.
    Evaluation snapshots containing object poses are kept under ``evaluation``.
    """

    def __init__(self, model, config, shared_control=True, *, hand_config=None,
                 tactile_retention=None):
        self.model = model
        self.config = config
        self.data = mujoco.MjData(model)
        self.shared_control = bool(shared_control)
        # Preserve the original manual protocol's tactile hold behaviour;
        # pass ``tactile_retention=False`` for a declared ablation.
        self.tactile_retention = True if tactile_retention is None else bool(tactile_retention)
        if hand_config is not None and not isinstance(hand_config, HandConfig):
            raise TypeError("hand_config must be a HandConfig")
        self.hand = HandGraspController(model, hand_config or HandConfig(
            contact_threshold_n=2.0, maximum_force_n=12.,
            retention_rate=.10, retention_extra=.25,
            retention_minimum_force_n=1.0, thumb_opposition_rad=1.1,
            thumb_mcp_flexion_ratio=.4, thumb_ip_flexion_ratio=1.2))
        self.tactile = TactileContactReader(model, touch_sensor_names={
            finger: f"v7_touch_{finger}" for finger in FINGERS})
        self.gravity = RobotGravityCompensation(model)
        self.wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.robot_geoms = set()
        self.hand_geoms = set(self.tactile.geometry_channels)
        self.object_geoms = {}
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if body in {"base", *(f"link{i}" for i in range(1, 8))} or body.startswith("left_"):
                self.robot_geoms.add(gid)
            if name.startswith("v7_geom_"):
                self.object_geoms[gid] = name[len("v7_geom_"):]
        # A local import keeps the scene/contract builder usable without OMPL.
        from .v7_motion import V7Kinematics
        self.kinematics = V7Kinematics(model)
        self.selected_target = config.registry.actionable[0].object_id
        self.requested_closure = dict.fromkeys(FINGERS, 0.)
        self.finger_overrides = {}
        self.finger_override_commands = {}
        self.events = []
        self.trace = []
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        legacy_kin = RobotKinematics(self.model)
        start = legacy_kin.solve(np.array([0., 0., .25]))
        if start is None:
            raise RuntimeError("V7 start pose is unreachable")
        self.data.qpos[:7] = start
        self.hand.closure = dict.fromkeys(FINGERS, 0.)
        self.hand.opposition = self.hand.config.thumb_opposition_rad
        self.hand.hold_reference = None
        self.hand.engaged.clear()
        self.hand.slip_adjustments = 0
        self.hand.initialize(self.data)
        self.data.ctrl[:7] = start
        mujoco.mj_forward(self.model, self.data)
        self.limiter = ArmCommandLimiter(start,
            arm_joint_velocity_limits(self.model), arm_joint_acceleration_limits(self.model))
        self.arm_goal = start.copy()
        self.position_goal = self.data.xpos[self.wrist_body].copy()
        self.quaternion_goal = self.data.xquat[self.wrist_body].copy()
        self.requested_closure = dict.fromkeys(FINGERS, 0.)
        self.finger_overrides = {}
        self.finger_override_commands = {}
        self.status = "READY"
        self.stopped = False
        self._stopped_hand_ctrl = None
        self.last_command_time = None
        self.overload_seconds = 0.
        self.interventions = 0
        self.events.clear()
        self.trace.clear()
        self._last_snapshot = self.tactile.read(self.data)

    def set_finger_target(self, finger, value):
        if finger not in FINGERS:
            raise ValueError("unknown finger")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("finger target must be finite and in [0, 1]")
        if self.stopped:
            return
        self.requested_closure[finger] = value

    def set_joint_target(self, name, angle):
        """Expose individual finger joints for supported regrasp experiments."""
        if name not in self.hand.actuators or not name.startswith("left_"):
            raise ValueError("unknown hand actuator")
        angle = float(angle)
        actuator = self.hand.actuators[name]
        joint = self.model.actuator_trnid[actuator, 0]
        if not math.isfinite(angle) or not self.model.jnt_range[joint, 0] <= angle <= self.model.jnt_range[joint, 1]:
            raise ValueError("joint command exceeds model range")
        if self.stopped:
            return
        if name not in self.finger_override_commands:
            self.finger_override_commands[name] = float(self.data.ctrl[actuator])
        self.finger_overrides[name] = angle

    def select_target(self, object_id):
        spec = self.config.registry.select_target(object_id)
        self.selected_target = spec.object_id
        self.events.append({"time_s": float(self.data.time), "event": "OPERATOR_TARGET", "target": spec.object_id})

    def stop(self, reason="OPERATOR_STOP"):
        if not self.stopped:
            self._stopped_hand_ctrl = {
                name: float(self.data.ctrl[aid]) for name, aid in self.hand.actuators.items()
                if name.startswith("left_")
            }
            self.requested_closure = self.hand.closure.copy()
            self.finger_overrides = {
                name: self._stopped_hand_ctrl[name] for name in self.finger_overrides
            }
            self.finger_override_commands = self.finger_overrides.copy()
            self.hand.hold_reference = None
            self.hand.engaged.clear()
            self.arm_goal = self.limiter.position.copy()
            self.limiter.velocity[:] = 0.
            self.position_goal, self.quaternion_goal = self.kinematics.pose(self.arm_goal)
        self.stopped = True
        self.status = reason
        self.events.append({"time_s": float(self.data.time), "event": reason})

    def command(self, command):
        if not isinstance(command, TeleopCommand):
            raise TypeError("command must be TeleopCommand")
        if self.last_command_time is not None and command.timestamp < self.last_command_time:
            raise ValueError("command timestamps must be monotonic")
        self.last_command_time = command.timestamp
        if command.selected_target is not None:
            self.select_target(command.selected_target)
        if "escape" in command.buttons or "space" in command.buttons:
            self.stop()
            return
        if "r" in command.buttons:
            self.stopped = False
            self.status = "READY"
        if self.stopped:
            return
        if "hold" in command.buttons:
            self.hand.hold_reference = self.hand.closure.copy()
        for finger in FINGERS:
            self.requested_closure[finger] = float(np.clip(
                self.requested_closure[finger] + command.grip_delta, 0., 1.))
        if not any(command.translation_m) and not any(command.rotation_rad):
            return
        position = self.position_goal + command.translation_m
        rotation = np.asarray(command.rotation_rad, dtype=float)
        angle = np.linalg.norm(rotation)
        delta = np.array([1., 0., 0., 0.])
        if angle > 0:
            mujoco.mju_axisAngle2Quat(delta, rotation / angle, angle)
        quaternion = np.empty(4)
        mujoco.mju_mulQuat(quaternion, delta, self.quaternion_goal)
        # Bounds constrain commanded wrist motion, not live physical positions.
        if not (.30 <= position[0] <= 1.05 and -.40 <= position[1] <= .40 and .40 <= position[2] <= 1.10):
            self.status = "WORKSPACE_COMMAND_REJECTED"
            self.interventions += 1
            return
        solution = self.kinematics.solve(position, quaternion, self.arm_goal)
        if solution is None:
            self.status = "IK_COMMAND_REJECTED"
            self.interventions += 1
            return
        self.position_goal, self.quaternion_goal = position, quaternion
        self.arm_goal = solution
        self.status = "MOVING"

    def _collision_events(self):
        wrench = np.zeros(6)
        events = []
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.robot_geoms) == (b in self.robot_geoms):
                continue
            robot, other = (a, b) if a in self.robot_geoms else (b, a)
            # Selected object contact is expected only on the hand surface.
            if robot in self.hand_geoms and self.object_geoms.get(other) == self.selected_target:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            if contact.dist < -.002 and abs(wrench[0]) > .5:
                events.append({"time_s": float(self.data.time),
                    "robot_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot) or f"geom_{robot}",
                    "other_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}",
                    "penetration_m": -float(contact.dist), "normal_force_n": abs(float(wrench[0]))})
        return events

    def step(self, command=None):
        if command is not None:
            self.command(command)
        dt = float(self.model.opt.timestep)
        tactile = self.tactile.read(self.data)
        force_readings = {
            finger: ForceReading(
                normal=reading.normal_force_n,
                tangential=reading.tangential_force_n,
                contacts=reading.contacts,
            )
            for finger, reading in tactile.fingers.items()
            if finger in FINGERS
        }
        # Tactile is target-independent: any hand-side contact can engage a
        # finger, but it never identifies which object caused it. The
        # retention loop only changes hand targets; object motion remains
        # entirely MuJoCo-generated.
        maximum = max(item.normal_force_n for item in tactile.fingers.values())
        self.overload_seconds = self.overload_seconds + dt if maximum > 30. else 0.
        if self.shared_control and self.overload_seconds >= .02 and not self.stopped:
            self.interventions += 1
            self.stop("TACTILE_OVERLOAD_STOP")
        if self.stopped:
            # Hold motor references while MuJoCo still integrates inertia and contacts.
            for name, angle in self._stopped_hand_ctrl.items():
                self.data.ctrl[self.hand.actuators[name]] = angle
            arm_command = self.arm_goal.copy()
        else:
            for finger, reading in force_readings.items():
                if reading.normal > .2:
                    self.hand.engaged.add(finger)
            for finger in FINGERS:
                reading = tactile.fingers[finger]
                current = self.hand.closure[finger]
                goal = self.requested_closure[finger]
                if self.shared_control and reading.normal_force_n > 12.:
                    goal = max(0., current - .2 * dt)
                elif self.shared_control and reading.normal_force_n >= 3. and goal > current:
                    goal = current
                self.hand.closure[finger] = current + float(np.clip(goal - current, -.4 * dt, .4 * dt))
            if (self.tactile_retention
                    and any(force.contacts > 0 for force in force_readings.values())
                    and any(goal > 0.03 for goal in self.requested_closure.values())):
                self.hand.retain(force_readings, dt)
            self.hand.write(self.data)
            for name, angle in self.finger_overrides.items():
                aid = self.hand.actuators[name]
                joint = self.model.actuator_trnid[aid, 0]
                finger = next((finger for finger in FINGERS if name.startswith(f"left_{finger}_")), None)
                # Manual individual joints still obey local overload relief.
                if self.shared_control and finger and tactile.fingers[finger].normal_force_n > 12.:
                    self.data.ctrl[aid] = min(self.data.ctrl[aid], angle)
                else:
                    previous = self.finger_override_commands.get(name, float(self.data.ctrl[aid]))
                    self.data.ctrl[aid] = previous + float(np.clip(angle - previous, -.5 * dt, .5 * dt))
                self.data.ctrl[aid] = np.clip(self.data.ctrl[aid], *self.model.jnt_range[joint])
                self.finger_override_commands[name] = float(self.data.ctrl[aid])
            arm_command = self.limiter.update(self.arm_goal, dt)
        self.gravity.apply(self.data, arm_command)
        mujoco.mj_step(self.model, self.data)
        self._last_snapshot = self.tactile.read(self.data)
        collision = self._collision_events()
        self.events.extend(collision)
        if self.shared_control and collision and not self.stopped:
            self.interventions += 1
            self.stop("ENVIRONMENT_COLLISION_STOP")
        return self.snapshot()

    def snapshot(self):
        """Separate operator signals from evaluation-only object state."""
        data = self.data
        evaluation = {}
        for spec in self.config.registry.objects:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"v7_obj_{spec.object_id}")
            evaluation[spec.object_id] = {"position": data.xpos[body].tolist(),
                "quaternion": data.xquat[body].tolist()}
        return {"time_s": float(data.time), "status": self.status,
            "shared_control": self.shared_control, "selected_target": self.selected_target,
            "wrist_position": data.xpos[self.wrist_body].tolist(),
            "wrist_quaternion": data.xquat[self.wrist_body].tolist(),
            "wrist_goal": self.position_goal.tolist(), "arm_command": self.limiter.position.tolist(),
            "arm_joint_position": data.qpos[:7].tolist(),
            "arm_joint_velocity": data.qvel[:7].tolist(),
            "hand_joint_position": {
                name: float(data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]])
                for name, aid in self.hand.actuators.items() if name.startswith("left_")
            },
            "arm_torque_nm": data.qfrc_actuator[:7].tolist(),
            "finger_closure": dict(self.hand.closure), "tactile": self._last_snapshot.as_dict(),
            "shared_control": self.shared_control, "tactile_retention": self.tactile_retention,
            "interventions": self.interventions, "evaluation": evaluation,
            "task_success": None}


__all__ = ["WorkbenchSession"]
