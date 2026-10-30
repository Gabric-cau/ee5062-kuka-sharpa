"""Physical multi-object V7 task execution.

The executor is intentionally a small, inspectable state machine.  It sends
Cartesian goals through :class:`WorkbenchSession`; MuJoCo integrates the arm,
hand and free object bodies.  No object qpos is written by this module.  The
object poses in the result are evaluation snapshots only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping

import mujoco
import numpy as np

from .teleop import TeleopCommand
from .v7_session import WorkbenchSession


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _finite_vector(values, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} finite values") from error
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class TaskSpec:
    """One tool transfer contract.

    The default wrist offset is calibrated from the V6 physical can transfer
    and is exposed as a parameter because tool geometry and hand posture need
    to be calibrated experimentally.  It is a declared map prior, never a
    read from a live object body.
    """

    object_id: str
    slot_id: str | None = None
    grasp_offset_m: tuple[float, float, float] = (-0.083, 0.027, 0.237)
    approach_axis_world: tuple[float, float, float] = (0.70710678, 0.0, -0.70710678)
    wrist_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_distance_m: float = 0.13
    lift_height_m: float = 0.15
    pregrasp_closure: float = 0.0
    grasp_closure: float = 0.85
    tactile_force_min_n: float = 0.25
    tactile_contacts_min: int = 1
    close_settle_seconds: float = 2.0
    release_verify_seconds: float = 0.5
    hold_seconds: float = 0.08
    phase_timeout_s: float = 5.0
    dock_tolerance_m: float = 0.065

    def __post_init__(self) -> None:
        if not self.object_id or any(character.isspace() for character in self.object_id):
            raise ValueError("object_id must be a non-empty token")
        if self.slot_id is not None and (not self.slot_id or any(character.isspace() for character in self.slot_id)):
            raise ValueError("slot_id must be a non-empty token")
        object.__setattr__(self, "grasp_offset_m", _finite_vector(self.grasp_offset_m, 3, "grasp_offset_m"))
        axis = np.asarray(_finite_vector(self.approach_axis_world, 3, "approach_axis_world"), dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError("approach_axis_world must be non-zero")
        object.__setattr__(self, "approach_axis_world", tuple(float(value) for value in axis / norm))
        object.__setattr__(self, "wrist_rotation_rad", _finite_vector(self.wrist_rotation_rad, 3, "wrist_rotation_rad"))
        for name in ("approach_distance_m", "lift_height_m", "hold_seconds", "phase_timeout_s", "dock_tolerance_m", "close_settle_seconds", "release_verify_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("pregrasp_closure", "grasp_closure"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(float(self.tactile_force_min_n)) or self.tactile_force_min_n < 0:
            raise ValueError("tactile_force_min_n must be finite and non-negative")
        if int(self.tactile_contacts_min) != self.tactile_contacts_min or self.tactile_contacts_min < 0:
            raise ValueError("tactile_contacts_min must be a non-negative integer")


@dataclass(frozen=True)
class PhaseRecord:
    name: str
    start_time_s: float
    end_time_s: float
    status: str
    steps: int
    tactile_peak_force_n: float
    tactile_peak_contacts: int
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "status": self.status,
            "steps": self.steps,
            "tactile_peak_force_n": self.tactile_peak_force_n,
            "tactile_peak_contacts": self.tactile_peak_contacts,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TaskResult:
    object_id: str
    slot_id: str | None
    success: bool
    status: str
    failure_reason: str | None
    phases: tuple[PhaseRecord, ...]
    trace: tuple[dict[str, object], ...]
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "slot_id": self.slot_id,
            "success": self.success,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "phases": [phase.as_dict() for phase in self.phases],
            "trace": list(self.trace),
            "metrics": dict(self.metrics),
        }


class V7TaskExecutor:
    """Execute physical pick/transfer/dock episodes on a workbench session."""

    def __init__(self, session: WorkbenchSession, *, reset_each_episode: bool = True,
                 step_callback=None) -> None:
        self.session = session
        self.reset_each_episode = bool(reset_each_episode)
        self.step_callback = step_callback
        self._timestamp = float(session.data.time)
        self.actions: list[dict[str, object]] = []
        self.current_phase = "INITIALIZE"

    @property
    def config(self):
        return self.session.config

    def default_tasks(self, object_ids: Iterable[str] | None = None) -> tuple[TaskSpec, ...]:
        requested = None if object_ids is None else tuple(object_ids)
        actionable = self.config.registry.actionable
        if requested is None:
            selected = actionable
        else:
            by_id = {item.object_id: item for item in actionable}
            unknown = [object_id for object_id in requested if object_id not in by_id]
            if unknown:
                raise ValueError(f"unknown actionable object(s): {unknown}")
            selected = tuple(by_id[object_id] for object_id in requested)
        tasks = []
        for spec in selected:
            tasks.append(calibrated_task_spec(spec.object_id, slot_id=spec.dock_slot))
        return tuple(tasks)

    def _object_body(self, object_id: str) -> int:
        body = mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_BODY, f"v7_obj_{object_id}")
        if body < 0:
            raise ValueError(f"model is missing object body for {object_id!r}")
        return int(body)

    def _slot_pose(self, spec: TaskSpec):
        if spec.slot_id is None:
            return None
        slots = {slot.slot_id: slot for slot in self.config.slots}
        slot = slots.get(spec.slot_id)
        if slot is None:
            raise ValueError(f"unknown dock slot {spec.slot_id!r}")
        if slot.accepted_object_id != spec.object_id:
            raise ValueError(f"dock slot {spec.slot_id!r} accepts {slot.accepted_object_id!r}, not {spec.object_id!r}")
        return np.asarray(slot.pose, dtype=float)

    def _object_map_pose(self, object_id: str) -> np.ndarray:
        """Return the declared layout pose; live qpos remains untouched."""
        return np.asarray(self.config.registry.by_id(object_id).pose, dtype=float)

    def _command(self, target: np.ndarray, rotation_rad: tuple[float, float, float] | None = None) -> None:
        delta = np.asarray(target, dtype=float) - np.asarray(self.session.position_goal, dtype=float)
        self._timestamp = max(self._timestamp, float(self.session.data.time))
        command = TeleopCommand(timestamp=self._timestamp, translation_m=tuple(delta),
                                rotation_rad=(0.0, 0.0, 0.0) if rotation_rad is None else rotation_rad,
                                source="task_executor")
        self.session.command(command)
        self.actions.append(command.as_dict())

    def _step(self, closure: float | None, trace: list[dict[str, object]]) -> dict[str, object]:
        grip_delta = 0.0 if closure is None else float(closure) - self.session.requested_closure[FINGERS[0]]
        self._timestamp = max(self._timestamp, float(self.session.data.time))
        command = TeleopCommand(timestamp=self._timestamp, grip_delta=grip_delta, source="task_executor")
        snapshot = self.session.step(command)
        if grip_delta:
            self.actions.append(command.as_dict())
        snapshot["phase"] = self.current_phase
        trace.append(snapshot)
        if self.step_callback is not None:
            self.step_callback(self.session.data, snapshot)
        return snapshot

    def _phase(self, name: str, target: np.ndarray, closure: float, *, trace: list[dict[str, object]],
               timeout_s: float, done=None, rotation_rad: tuple[float, float, float] | None = None) -> tuple[PhaseRecord, str | None]:
        start = float(self.session.data.time)
        self.current_phase = name
        if name == "LIFT":
            hold_command = TeleopCommand(timestamp=start, buttons=("hold",), source="task_executor")
            self.session.command(hold_command)
            self.actions.append(hold_command.as_dict())
        origin = np.asarray(self.session.position_goal, dtype=float).copy()
        rotation = np.asarray((0.0, 0.0, 0.0) if rotation_rad is None else rotation_rad, dtype=float)
        path_seconds = max(float(np.linalg.norm(target - origin)) / .08,
                           float(np.linalg.norm(rotation)) / .3, .02)
        previous_fraction = 0.0
        command_interval = max(1, round(.02 / float(self.session.model.opt.timestep)))
        limit = max(1, int(math.ceil(timeout_s / float(self.session.model.opt.timestep))))
        peak_force, peak_contacts = 0.0, 0
        status = "TIMEOUT"
        notes: list[str] = []
        for step_index in range(limit):
            if step_index % command_interval == 0:
                fraction = min(1.0, (step_index * float(self.session.model.opt.timestep) + .02) / path_seconds)
                # Solve small Cartesian segments at a fixed wrist pose rather
                # than letting independently limited joints arc a held tool.
                point = origin + fraction * (target - origin)
                self._command(point, tuple(rotation * (fraction - previous_fraction)))
                previous_fraction = fraction
                if self.session.status in {"IK_COMMAND_REJECTED", "WORKSPACE_COMMAND_REJECTED"}:
                    status = "COMMAND_REJECTED"
                    notes.append(self.session.status)
                    break
            snapshot = self._step(closure, trace)
            tactile = snapshot["tactile"]
            force = float(tactile["total_normal_force_n"])
            contacts = int(tactile["contact_count"])
            peak_force, peak_contacts = max(peak_force, force), max(peak_contacts, contacts)
            wrist = np.asarray(snapshot["wrist_position"], dtype=float)
            reached = previous_fraction >= 1.0 and float(np.linalg.norm(wrist - target)) <= 0.012
            condition = reached if done is None else bool(done(snapshot, reached))
            if snapshot["status"] in {"TACTILE_OVERLOAD_STOP", "ENVIRONMENT_COLLISION_STOP", "OPERATOR_STOP"}:
                status = "SAFETY_STOP"
                notes.append(str(snapshot["status"]))
                break
            if condition:
                status = "COMPLETE"
                break
        else:
            step_index = limit
        end = float(self.session.data.time)
        record = PhaseRecord(name, start, end, status, step_index + 1, peak_force, peak_contacts, tuple(notes))
        return record, None if status == "COMPLETE" else f"{name.upper()}_{status}"

    def run(self, task: TaskSpec) -> TaskResult:
        if self.reset_each_episode:
            self.session.reset()
            self._timestamp = float(self.session.data.time)
        event_start = len(self.session.events)
        self.actions = []
        selection = TeleopCommand(timestamp=float(self.session.data.time), selected_target=task.object_id,
                                 source="task_executor")
        self.session.command(selection)
        self.actions.append(selection.as_dict())
        self._object_body(task.object_id)
        initial_snapshot = self.session.snapshot()
        initial_position = np.asarray(initial_snapshot["evaluation"][task.object_id]["position"], dtype=float)
        target_pose = self._object_map_pose(task.object_id)
        grasp = target_pose + np.asarray(task.grasp_offset_m, dtype=float)
        pregrasp = grasp - np.asarray(task.approach_axis_world, dtype=float) * task.approach_distance_m
        lift = grasp + np.array([0.0, 0.0, task.lift_height_m])
        slot = self._slot_pose(task)
        dock = None if slot is None else slot + np.asarray(task.grasp_offset_m, dtype=float)
        trace: list[dict[str, object]] = [initial_snapshot]
        phases: list[PhaseRecord] = []
        failure: str | None = None

        phase, failure = self._phase("APPROACH", pregrasp, task.pregrasp_closure, trace=trace, timeout_s=task.phase_timeout_s,
                                     rotation_rad=task.wrist_rotation_rad)
        phases.append(phase)
        if failure is None:
            phase, failure = self._phase("ALIGN", grasp, task.pregrasp_closure, trace=trace, timeout_s=task.phase_timeout_s)
            phases.append(phase)
        if failure is None:
            close_start = float(self.session.data.time)
            def held_condition(snapshot, reached):
                return reached and float(snapshot["time_s"]) - close_start >= task.close_settle_seconds and (
                                    float(snapshot["tactile"]["total_normal_force_n"]) >= task.tactile_force_min_n
                                    and int(snapshot["tactile"]["contact_count"]) >= task.tactile_contacts_min)
            phase, failure = self._phase("CLOSE", grasp, task.grasp_closure, trace=trace,
                                         timeout_s=task.phase_timeout_s, done=held_condition)
            phases.append(phase)
            # Require tactile evidence to persist, rather than accepting one
            # collision impulse as a grasp.
            if failure is None:
                hold_steps = max(1, int(math.ceil(task.hold_seconds / float(self.session.model.opt.timestep))))
                for _ in range(hold_steps):
                    snap = self._step(task.grasp_closure, trace)
                    if (float(snap["tactile"]["total_normal_force_n"]) < task.tactile_force_min_n
                            or int(snap["tactile"]["contact_count"]) < task.tactile_contacts_min):
                        failure = "GRASP_CONTACT_NOT_PERSISTENT"
                        break
                if failure is not None:
                    phases[-1] = PhaseRecord(phases[-1].name, phases[-1].start_time_s,
                                             float(self.session.data.time), "FAILED", phases[-1].steps,
                                             phases[-1].tactile_peak_force_n, phases[-1].tactile_peak_contacts,
                                             phases[-1].notes + (failure,))
        if failure is None:
            body_before_lift = np.asarray(trace[-1]["evaluation"][task.object_id]["position"], dtype=float)
            def lift_condition(snapshot, reached):
                position = np.asarray(snapshot["evaluation"][task.object_id]["position"], dtype=float)
                tactile = snapshot["tactile"]
                return reached and position[2] > body_before_lift[2] + 0.04 and (
                    float(tactile["total_normal_force_n"]) >= task.tactile_force_min_n)
            phase, failure = self._phase("LIFT", lift, task.grasp_closure, trace=trace,
                                         timeout_s=task.phase_timeout_s, done=lift_condition)
            phases.append(phase)
            if failure is None and np.asarray(trace[-1]["evaluation"][task.object_id]["position"])[2] <= initial_position[2] + .04:
                failure = "OBJECT_NOT_LIFTED"
        if failure is None and dock is not None:
            phase, failure = self._phase("TRANSFER", dock + np.array([0., 0., .10]), task.grasp_closure,
                                         trace=trace, timeout_s=task.phase_timeout_s)
            phases.append(phase)
        if failure is None and dock is not None:
            phase, failure = self._phase("DOCK", dock, task.grasp_closure, trace=trace, timeout_s=task.phase_timeout_s)
            phases.append(phase)
        if failure is None:
            release_target = dock if dock is not None else lift
            def open_condition(snapshot, reached):
                return reached and max(snapshot["finger_closure"].values()) <= .03
            phase, failure = self._phase("RELEASE", release_target, 0.0, trace=trace,
                                         timeout_s=task.phase_timeout_s, done=open_condition)
            phases.append(phase)
        if failure is None and slot is not None:
            previous_position = np.asarray(trace[-1]["evaluation"][task.object_id]["position"], dtype=float)
            stable_since = None
            def stable_dock_condition(snapshot, reached):
                nonlocal previous_position, stable_since
                position = np.asarray(snapshot["evaluation"][task.object_id]["position"], dtype=float)
                speed = float(np.linalg.norm(position - previous_position)) / float(self.session.model.opt.timestep)
                previous_position = position
                in_slot = float(np.linalg.norm(position - slot)) <= task.dock_tolerance_m
                stable = reached and in_slot and speed < .02 and max(snapshot["finger_closure"].values()) <= .03
                if not stable:
                    stable_since = None
                elif stable_since is None:
                    stable_since = float(snapshot["time_s"])
                return stable_since is not None and float(snapshot["time_s"]) - stable_since >= task.release_verify_seconds
            phase, failure = self._phase("VERIFY", dock, 0.0, trace=trace,
                                         timeout_s=task.phase_timeout_s, done=stable_dock_condition)
            phases.append(phase)
        final_position = np.asarray(trace[-1]["evaluation"][task.object_id]["position"], dtype=float)
        dock_error = None if slot is None else float(np.linalg.norm(final_position - slot))
        success = failure is None and dock_error is not None and dock_error <= task.dock_tolerance_m
        if failure is None and not success:
            failure = "DOCK_NOT_REQUESTED" if dock_error is None else "DOCK_TOLERANCE_FAILED"
        metrics = {
            "initial_object_position_m": initial_position.tolist(),
            "final_object_position_m": final_position.tolist(),
            "object_height_change_m": float(final_position[2] - initial_position[2]),
            "dock_error_m": dock_error,
            "peak_tactile_force_n": max((float(row["tactile"]["total_normal_force_n"]) for row in trace), default=0.0),
            "peak_tactile_contacts": max((int(row["tactile"]["contact_count"]) for row in trace), default=0),
            "physics_object_body": f"v7_obj_{task.object_id}",
            "qpos_write_policy": "executor_never_writes_object_qpos",
            "task_scope": "pick_transfer_dock_release" if slot is not None else "lift_calibration_probe",
            "trace_samples": len(trace),
            "events": list(self.session.events[event_start:]),
            "action_log": list(self.actions),
        }
        return TaskResult(task.object_id, task.slot_id, bool(success), "SUCCESS" if success else "FAILED",
                          failure, tuple(phases), tuple(trace), metrics)

    def run_batch(self, tasks: Iterable[TaskSpec] | None = None) -> tuple[TaskResult, ...]:
        selected = self.default_tasks() if tasks is None else tuple(tasks)
        return tuple(self.run(task) for task in selected)


def calibrated_task_spec(object_id: str, *, slot_id: str | None = None) -> TaskSpec:
    """Declared development presets; marker calibration used seed 0 only."""
    if object_id == "tool_marker":
        return TaskSpec(object_id, slot_id=slot_id,
                        grasp_offset_m=(-.065, .037, .233),
                        approach_axis_world=(0., 0., -1.), grasp_closure=.65,
                        lift_height_m=.10, phase_timeout_s=6., hold_seconds=.1,
                        tactile_force_min_n=.15)
    return TaskSpec(object_id, slot_id=slot_id, phase_timeout_s=6.)


__all__ = ["PhaseRecord", "TaskResult", "TaskSpec", "V7TaskExecutor", "calibrated_task_spec"]
