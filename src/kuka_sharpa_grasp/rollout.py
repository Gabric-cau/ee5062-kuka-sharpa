"""Deterministic force-control benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from .control import ContactForceReader, FINGERS, ForceReading, OuterForceController, hand_synergy_targets
from .evaluator import evaluate_trial
from .physics import capture_object_reference, set_object_parameters


@dataclass
class TrialResult:
    seed: int
    controller: str
    steps: int
    contacts: int
    final_contacts: int
    peak_force: float
    settled_peak_force: float
    final_force: float
    object_height: float
    object_mass: float
    lifted: bool
    status: str
    # Appended fields keep the original positional constructor compatible.
    failure_reason: str = "none"
    stage: str = "hold"
    object_x: float = 0.0
    object_y: float = 0.0
    object_z: float = 0.0
    object_friction: float = 0.0


class GraspRollout:
    """A small, inspectable rollout used before adding planning and recovery."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.reader = ContactForceReader(model)
        self.arm_names = [f"joint{i}" for i in range(1, 8)]
        self.hand_joint_names = {
            finger: [name for name in self._joint_names() if name.startswith(f"left_{finger}_")]
            for finger in FINGERS
        }
        self.hand_actuator_ids = {
            finger: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self._actuator_names(finger)]
            for finger in FINGERS
        }
        self.arm_actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]
        self.target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        # Read the compiled body pose so mesh objects with a non-zero local
        # origin (such as the YCB can) reset to the same physical pose as the
        # scene XML. This keeps box and mesh trials comparable.
        self.object_initial_pos = np.array(model.body_pos[self.target_body_id], dtype=float)
        self.base_object_mass = float(model.body_mass[self.target_body_id])
        self.base_object_friction = np.array(model.geom_friction[self.target_geom_id], dtype=float)
        self.base_object_inertia = np.array(model.body_inertia[self.target_body_id], dtype=float)
        self.object_parameter_reference = capture_object_reference(model, self.target_body_id, self.target_geom_id)
        self.arm_home = np.asarray((0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0), dtype=float)
        # This simple joint-space waypoint raises the hand after the grasp phase.
        self.arm_lift = np.asarray((0.0, 0.40, 0.0, -1.20, 0.0, 0.0, 0.0), dtype=float)
        self.target_qpos = hand_synergy_targets(0.0)

    def _trial_object_pos(self, seed: int) -> tuple[float, float, float]:
        """Return the scene-authored pose until randomized placement is enabled."""
        del seed
        return tuple(float(value) for value in self.object_initial_pos)

    def _joint_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(self.model.njnt)]

    def _actuator_names(self, finger: str) -> list[str]:
        names = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith(f"left_{finger}_"):
                names.append(name)
        return names

    def _set_joint(self, name: str, value: float) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr] = value

    def reset(self, seed: int = 0, object_pos: tuple[float, float, float] | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for name, value in zip(self.arm_names, self.arm_home):
            self._set_joint(name, value)
        target_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        if object_pos is None:
            object_pos = self._trial_object_pos(seed)
        self.data.qpos[self.model.jnt_qposadr[target_jid]:self.model.jnt_qposadr[target_jid] + 3] = np.asarray(object_pos)
        self.data.qpos[self.model.jnt_qposadr[target_jid] + 3:self.model.jnt_qposadr[target_jid] + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)
        self.target_qpos = hand_synergy_targets(0.0)
        self._write_arm_ctrl(tuple(self.arm_home))
        self._write_hand_ctrl(self.target_qpos)

    def configure_object(self, mass: float | None = None, friction: float | None = None) -> tuple[float, float]:
        """Set object mass/sliding friction and refresh constants before a trial."""
        return set_object_parameters(
            self.model, self.data, self.target_body_id, self.target_geom_id,
            reference=self.object_parameter_reference, mass=mass, friction=friction,
        )

    def _write_arm_ctrl(self, values: tuple[float, ...]) -> None:
        for actuator_id, value in zip(self.arm_actuator_ids, values):
            self.data.ctrl[actuator_id] = value

    def _arm_target(self, step: int) -> tuple[float, ...]:
        """Move only after the grasp has had time to establish contact."""
        # Keep the object in a static preload before lifting. This margin makes
        # the contact loop less sensitive to platform-specific floating-point
        # differences and gives the hand time to establish all contacts.
        start = int(1.30 / self.model.opt.timestep)
        duration = max(1, int(1.10 / self.model.opt.timestep))
        alpha = np.clip((step - start) / duration, 0.0, 1.0)
        return tuple((1.0 - alpha) * self.arm_home + alpha * self.arm_lift)

    def _write_hand_ctrl(self, targets: dict[str, float]) -> None:
        for finger, actuator_ids in self.hand_actuator_ids.items():
            for rank, actuator_id in enumerate(actuator_ids):
                # A conservative per-finger synergy; distal joints close later.
                value = targets[finger] * (0.25 + 0.75 * min(1.0, (rank + 1) / max(1, len(actuator_ids))))
                jid = self.model.actuator_trnid[actuator_id, 0]
                lo, hi = self.model.jnt_range[jid]
                self.data.ctrl[actuator_id] = float(lo + (hi - lo) * value)

    def run(self, seed: int = 0, controller: str = "force", duration: float = 2.5,
            object_mass: float | None = None,
            object_pos: tuple[float, float, float] | None = None,
            object_friction: float | None = None) -> TrialResult:
        mass, friction = self.configure_object(object_mass, object_friction)
        trial_pos = self._trial_object_pos(seed) if object_pos is None else object_pos
        self.reset(seed=seed, object_pos=trial_pos)
        # The initial preload is 0.85 normalized closure. The force target is
        # per fingertip; the reported total is the sum across all contacts.
        force_controller = OuterForceController(target_force=8.0) if controller == "force" else None
        peak_force = 0.0
        settled_peak_force = 0.0
        max_contacts = 0
        steps = int(duration / self.model.opt.timestep)
        settle_step = int(0.35 / self.model.opt.timestep)
        for step in range(steps):
            readings = self.reader.read(self.data)
            total = self.reader.total(readings)
            peak_force = max(peak_force, total.normal)
            if step >= settle_step:
                settled_peak_force = max(settled_peak_force, total.normal)
            max_contacts = max(max_contacts, total.contacts)
            if step <= int(0.35 / self.model.opt.timestep):
                self.target_qpos = hand_synergy_targets(0.0)
            elif step <= int(1.20 / self.model.opt.timestep):
                self.target_qpos = hand_synergy_targets(0.85)
            elif force_controller is not None:
                self.target_qpos = force_controller.update(readings, self.target_qpos, self.model.opt.timestep)
            else:
                self.target_qpos = hand_synergy_targets(0.85)
            self._write_arm_ctrl(self._arm_target(step))
            self._write_hand_ctrl(self.target_qpos)
            mujoco.mj_step(self.model, self.data)
        readings = self.reader.read(self.data)
        total = self.reader.total(readings)
        target_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        height = float(self.data.qpos[self.model.jnt_qposadr[target_jid] + 2])
        evaluation = evaluate_trial(
            max_contacts=max_contacts,
            final_contacts=total.contacts,
            object_height=height,
        )
        lifted = evaluation.status == "success"
        return TrialResult(
            seed, controller, steps, max_contacts, total.contacts, peak_force,
            settled_peak_force, total.normal, height, mass, lifted,
            evaluation.status, evaluation.failure_reason.value, evaluation.stage.value,
            float(trial_pos[0]), float(trial_pos[1]), float(trial_pos[2]),
            friction,
        )

    def run_trace(self, seed: int = 0, controller: str = "force", duration: float = 2.5,
                  object_mass: float | None = None,
                  object_pos: tuple[float, float, float] | None = None,
                  object_friction: float | None = None,
                  frame_callback: Callable[[int, mujoco.MjData, dict[str, ForceReading]], None] | None = None,
                  ) -> tuple[TrialResult, list[dict[str, float]]]:
        """Run one trial and return per-step signals for plots and debugging."""
        mass, friction = self.configure_object(object_mass, object_friction)
        trial_pos = self._trial_object_pos(seed) if object_pos is None else object_pos
        self.reset(seed=seed, object_pos=trial_pos)
        force_controller = OuterForceController(target_force=8.0) if controller == "force" else None
        rows: list[dict[str, float]] = []
        peak_force = 0.0
        settled_peak_force = 0.0
        max_contacts = 0
        steps = int(duration / self.model.opt.timestep)
        settle_step = int(0.35 / self.model.opt.timestep)
        for step in range(steps):
            readings = self.reader.read(self.data)
            total = self.reader.total(readings)
            peak_force = max(peak_force, total.normal)
            if step >= settle_step:
                settled_peak_force = max(settled_peak_force, total.normal)
            max_contacts = max(max_contacts, total.contacts)
            if step <= int(0.35 / self.model.opt.timestep):
                self.target_qpos = hand_synergy_targets(0.0)
            elif step <= int(1.20 / self.model.opt.timestep):
                self.target_qpos = hand_synergy_targets(0.85)
            elif force_controller is not None:
                self.target_qpos = force_controller.update(readings, self.target_qpos, self.model.opt.timestep)
            else:
                self.target_qpos = hand_synergy_targets(0.85)
            self._write_arm_ctrl(self._arm_target(step))
            self._write_hand_ctrl(self.target_qpos)
            rows.append({
                "time": float(self.data.time),
                "total_normal_force": total.normal,
                "total_tangential_force": total.tangential,
                "contacts": float(total.contacts),
                "target_force": force_controller.target_force if force_controller is not None else 0.0,
                **{f"{finger}_normal_force": readings[finger].normal for finger in FINGERS},
            })
            mujoco.mj_step(self.model, self.data)
            if frame_callback is not None:
                frame_callback(step + 1, self.data, self.reader.read(self.data))
        readings = self.reader.read(self.data)
        total = self.reader.total(readings)
        target_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        height = float(self.data.qpos[self.model.jnt_qposadr[target_jid] + 2])
        evaluation = evaluate_trial(
            max_contacts=max_contacts,
            final_contacts=total.contacts,
            object_height=height,
        )
        lifted = evaluation.status == "success"
        result = TrialResult(
            seed, controller, steps, max_contacts, total.contacts, peak_force,
            settled_peak_force, total.normal, height, mass, lifted,
            evaluation.status, evaluation.failure_reason.value, evaluation.stage.value,
            float(trial_pos[0]), float(trial_pos[1]), float(trial_pos[2]),
            friction,
        )
        return result, rows
