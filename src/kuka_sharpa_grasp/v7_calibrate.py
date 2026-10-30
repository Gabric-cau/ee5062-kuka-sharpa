"""Physical grasp calibration with phase-bounded, target-specific evidence.

Candidates are executed through ``V7TaskExecutor``. The extra contact reader
is evaluation instrumentation: it labels recorded MuJoCo contacts, never
feeds object poses into the controller and never writes object qpos.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Mapping

import mujoco
import numpy as np

from .v7_task import TaskResult, TaskSpec, V7TaskExecutor


EVIDENCE_KEY = "calibration_evaluation"


@dataclass(frozen=True)
class GraspCandidate:
    grasp_offset_m: tuple[float, float, float]
    wrist_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    closure: float = 0.35

    def __post_init__(self):
        for name in ("grasp_offset_m", "wrist_rotation_rad", "approach_axis_world"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)
        norm = float(np.linalg.norm(self.approach_axis_world))
        if norm < 1e-9:
            raise ValueError("approach_axis_world must be non-zero")
        object.__setattr__(self, "approach_axis_world",
                           tuple(value / norm for value in self.approach_axis_world))
        if not math.isfinite(float(self.closure)) or not 0 <= self.closure <= 1:
            raise ValueError("closure must be finite and in [0, 1]")

    def as_dict(self):
        return {"grasp_offset_m": list(self.grasp_offset_m),
                "wrist_rotation_rad": list(self.wrist_rotation_rad),
                "approach_axis_world": list(self.approach_axis_world),
                "closure": self.closure}


class CalibrationContactReader:
    """Evaluation-only attribution of contacts to the requested free object."""

    def __init__(self, session, object_id: str):
        self.session, self.object_id = session, object_id
        self.model = session.model
        self.target_geoms = {gid for gid, identity in session.object_geoms.items()
                             if identity == object_id}
        self.geometry_channels = session.tactile.geometry_channels
        if not self.target_geoms:
            raise ValueError(f"model has no geometry for {object_id!r}")

    def read(self) -> dict[str, object]:
        session, model, data = self.session, self.model, self.session.data
        target_force, target_contacts = 0.0, 0
        target_fingers: set[str] = set()
        support_force = 0.0
        channel_forces: dict[str, float] = {}
        collisions = []
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            first, second = int(contact.geom1), int(contact.geom2)
            mujoco.mj_contactForce(model, data, index, wrench)
            force = abs(float(wrench[0]))
            if (first in session.hand_geoms) != (second in session.hand_geoms):
                hand = first if first in session.hand_geoms else second
                other = second if hand == first else first
                channel = self.geometry_channels.get(hand, "palm")
                # These names identify hand surfaces, never an object identity
                # available to the physical controller.
                channel_forces[channel] = channel_forces.get(channel, 0.0) + force
                if other in self.target_geoms:
                    target_force += force
                    target_contacts += int(force > .05)
                    if channel != "palm" and force > .05:
                        target_fingers.add(channel)
            if first in self.target_geoms or second in self.target_geoms:
                other = second if first in self.target_geoms else first
                if other not in session.robot_geoms:
                    support_force += force
            if (first in session.robot_geoms) != (second in session.robot_geoms):
                robot, other = (first, second) if first in session.robot_geoms else (second, first)
                allowed_target = robot in session.hand_geoms and other in self.target_geoms
                if not allowed_target and contact.dist < -.002 and force > .5:
                    collisions.append({
                        "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot),
                        "other_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        "penetration_m": -float(contact.dist), "normal_force_n": force,
                    })
        return {
            "target_id": self.object_id, "target_hand_force_n": target_force,
            "target_hand_contacts": target_contacts, "target_contact_fingers": sorted(target_fingers),
            "target_support_force_n": support_force,
            "maximum_hand_channel_force_n": max(channel_forces.values(), default=0.0),
            "hand_channel_forces_n": channel_forces, "environment_collisions": collisions,
            "source": "mujoco_contact_attribution_evaluation_only",
        }


@contextmanager
def _record_contacts(executor: V7TaskExecutor, object_id: str):
    session = executor.session
    if executor.reset_each_episode:
        # Isolate each candidate's hand controller history as well as its
        # physics reset. Retention references must not leak between trials.
        session.hand.hold_reference = None
        session.hand.engaged.clear()
        session.hand.slip_adjustments = 0
    reader = CalibrationContactReader(session, object_id)
    previous_step = session.step

    def instrumented_step(command=None):
        snapshot = previous_step(command)
        snapshot[EVIDENCE_KEY] = reader.read()
        return snapshot

    session.step = instrumented_step
    try:
        yield
    finally:
        session.step = previous_step


@dataclass(frozen=True)
class LiftEvidence:
    accepted: bool
    lift_phase_complete: bool
    evidence_samples: int
    sustained_lift_seconds: float
    sustained_height_change_m: float
    lift_endpoint_height_change_m: float | None
    lift_endpoint_target_held: bool
    collision_samples: int
    overload_samples: int
    maximum_hand_channel_force_n: float
    reasons: tuple[str, ...]
    phase_metrics: tuple[Mapping[str, object], ...]

    def as_dict(self):
        return {
            "accepted": self.accepted, "lift_phase_complete": self.lift_phase_complete,
            "evidence_samples": self.evidence_samples,
            "sustained_lift_seconds": self.sustained_lift_seconds,
            "sustained_height_change_m": self.sustained_height_change_m,
            "lift_endpoint_height_change_m": self.lift_endpoint_height_change_m,
            "lift_endpoint_target_held": self.lift_endpoint_target_held,
            "collision_samples": self.collision_samples, "overload_samples": self.overload_samples,
            "maximum_hand_channel_force_n": self.maximum_hand_channel_force_n,
            "reasons": list(self.reasons), "phase_metrics": list(self.phase_metrics),
            "criteria": {
                "height_change_m": .04, "hold_seconds": .10,
                "target_hand_force_n": .25, "target_hand_contacts": 2,
                "target_finger_channels": 2, "maximum_support_force_n": .10,
                "maximum_relative_drift_m": .03, "maximum_sample_gap_s": .02,
                "overload_channel_force_n": 30.0,
            },
        }


def _target_relative_position(row, object_id: str):
    pose = row["evaluation"][object_id]
    inverse = np.asarray(row["wrist_quaternion"], dtype=float).copy()
    inverse[1:] *= -1
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, inverse)
    return rotation.reshape(3, 3) @ (
        np.asarray(pose["position"], dtype=float) - np.asarray(row["wrist_position"], dtype=float))


def score_lift_evidence(result: TaskResult, object_id: str) -> LiftEvidence:
    """Require a completed LIFT with continuous target contact and no support.

    Whole-episode maximum height remains diagnostic only. A brief impulse or
    a contact with another workbench object cannot qualify a held lift.
    """
    initial_height = float(result.metrics["initial_object_position_m"][2])
    collision_samples, overload_samples, maximum_force, evidence_samples = 0, 0, 0.0, 0
    for row in result.trace:
        evidence = row.get(EVIDENCE_KEY)
        if evidence is None:
            continue
        evidence_samples += 1
        collision_samples += bool(evidence["environment_collisions"])
        force = float(evidence["maximum_hand_channel_force_n"])
        maximum_force = max(maximum_force, force)
        overload_samples += force > 30.0

    lift_phase_complete = False
    best_seconds, best_height = 0.0, 0.0
    endpoint_height = None
    endpoint_target_held = False
    phase_metrics = []
    for phase in result.phases:
        rows = [row for row in result.trace
                if phase.start_time_s < float(row["time_s"]) <= phase.end_time_s + 1e-9]
        attributed = [row for row in rows if EVIDENCE_KEY in row]
        target_forces = [float(row[EVIDENCE_KEY]["target_hand_force_n"]) for row in attributed]
        heights = [float(row["evaluation"][object_id]["position"][2]) - initial_height for row in rows]
        metric = {
            **phase.as_dict(), "evaluation_samples": len(attributed),
            "target_hand_peak_force_n": max(target_forces, default=0.0),
            "target_hand_contact_samples": sum(value >= .25 for value in target_forces),
            "maximum_height_change_m": max(heights, default=0.0),
            "endpoint_height_change_m": heights[-1] if heights else None,
            "collision_samples": sum(bool(row[EVIDENCE_KEY]["environment_collisions"]) for row in attributed),
            "overload_samples": sum(float(row[EVIDENCE_KEY]["maximum_hand_channel_force_n"]) > 30 for row in attributed),
        }
        if phase.name == "LIFT":
            lift_phase_complete |= phase.status == "COMPLETE"
            endpoint_height = heights[-1] if heights else None
            endpoint_evidence = rows[-1].get(EVIDENCE_KEY) if rows else None
            endpoint_target_held = endpoint_evidence is not None and (
                float(endpoint_evidence["target_hand_force_n"]) >= .25
                and int(endpoint_evidence["target_hand_contacts"]) >= 2
                and len(endpoint_evidence["target_contact_fingers"]) >= 2
                and float(endpoint_evidence["target_support_force_n"]) <= .10
                and not endpoint_evidence["environment_collisions"]
                and float(endpoint_evidence["maximum_hand_channel_force_n"]) <= 30.)
            run_start = None
            run_minimum = math.inf
            relative_origin = None
            previous_time = None
            phase_best_seconds, phase_best_height = 0.0, 0.0
            for row in rows:
                timestamp = float(row["time_s"])
                evidence = row.get(EVIDENCE_KEY)
                height = float(row["evaluation"][object_id]["position"][2]) - initial_height
                valid = evidence is not None and height >= .04 and (
                    float(evidence["target_hand_force_n"]) >= .25
                    and int(evidence["target_hand_contacts"]) >= 2
                    and len(evidence["target_contact_fingers"]) >= 2
                    and float(evidence["target_support_force_n"]) <= .10
                    and not evidence["environment_collisions"]
                    and float(evidence["maximum_hand_channel_force_n"]) <= 30.0)
                if previous_time is not None and timestamp - previous_time > .02 + 1e-9:
                    valid = False
                relative = _target_relative_position(row, object_id)
                if valid and run_start is not None and np.linalg.norm(relative - relative_origin) > .03:
                    valid = False
                if valid:
                    if run_start is None:
                        run_start, relative_origin = timestamp, relative
                        run_minimum = height
                    run_minimum = min(run_minimum, height)
                    seconds = timestamp - run_start
                    if seconds > phase_best_seconds:
                        phase_best_seconds, phase_best_height = seconds, run_minimum
                else:
                    run_start, relative_origin, run_minimum = None, None, math.inf
                previous_time = timestamp
            metric["sustained_lift_seconds"] = phase_best_seconds
            metric["sustained_height_change_m"] = phase_best_height
            if phase_best_seconds > best_seconds:
                best_seconds, best_height = phase_best_seconds, phase_best_height
        phase_metrics.append(metric)
    reasons = []
    if not lift_phase_complete:
        reasons.append("LIFT_PHASE_NOT_COMPLETE")
    if best_seconds < .10 - 1e-9:
        reasons.append("NO_SUSTAINED_TARGET_HELD_LIFT")
    if endpoint_height is None or endpoint_height < .04:
        reasons.append("LIFT_ENDPOINT_BELOW_HEIGHT_THRESHOLD")
    if not endpoint_target_held:
        reasons.append("LIFT_ENDPOINT_TARGET_NOT_HELD")
    if collision_samples:
        reasons.append("ENVIRONMENT_COLLISION")
    if overload_samples:
        reasons.append("HAND_CHANNEL_OVERLOAD")
    if evidence_samples != max(0, len(result.trace) - 1):
        reasons.append("CONTACT_EVIDENCE_INCOMPLETE")
    accepted = (lift_phase_complete and best_seconds >= .10 - 1e-9
                and endpoint_height is not None and endpoint_height >= .04
                and endpoint_target_held
                and collision_samples == 0 and overload_samples == 0
                and evidence_samples == max(0, len(result.trace) - 1))
    return LiftEvidence(accepted, lift_phase_complete, evidence_samples, best_seconds, best_height,
                        endpoint_height, endpoint_target_held, collision_samples, overload_samples, maximum_force,
                        tuple(reasons), tuple(phase_metrics))


@dataclass(frozen=True)
class CalibrationRecord:
    candidate: GraspCandidate
    result: TaskResult
    max_height_change_m: float
    max_tactile_force_n: float
    max_contacts: int
    lift_evidence: LiftEvidence

    @property
    def score(self) -> float:
        evidence = self.lift_evidence
        safety_penalty = 0.0
        if evidence.collision_samples:
            safety_penalty += 10.0 + min(10.0, evidence.collision_samples * .01)
        if evidence.overload_samples:
            safety_penalty += 20.0 + min(20.0, evidence.maximum_hand_channel_force_n / 30.0)
        # Contact magnitude and instantaneous height never provide a bonus.
        signal = (100.0 if evidence.accepted else 0.0)
        signal += min(1.0, evidence.sustained_lift_seconds) + 5.0 * evidence.sustained_height_change_m
        progress = sum(phase.status == "COMPLETE" and phase.name in {"APPROACH", "ALIGN", "CLOSE"}
                       for phase in self.result.phases)
        return float(signal + .001 * progress - safety_penalty)

    def as_dict(self):
        return {"candidate": self.candidate.as_dict(), "success": self.lift_evidence.accepted,
                "success_scope": "target_held_lift_only_no_docking",
                "executor_success": self.result.success,
                "failure_reason": self.result.failure_reason,
                "max_height_change_m": self.max_height_change_m,
                "max_tactile_force_n": self.max_tactile_force_n,
                "max_contacts": self.max_contacts, "score": self.score,
                "lift_evidence": self.lift_evidence.as_dict(),
                "phases": [phase.as_dict() for phase in self.result.phases]}


def evaluate_candidate(executor: V7TaskExecutor, object_id: str, candidate: GraspCandidate,
                       *, phase_timeout_s: float = 6.0, hold_seconds: float = .08) -> CalibrationRecord:
    task = TaskSpec(object_id, grasp_offset_m=candidate.grasp_offset_m,
                    wrist_rotation_rad=candidate.wrist_rotation_rad,
                    approach_axis_world=candidate.approach_axis_world,
                    grasp_closure=candidate.closure, phase_timeout_s=phase_timeout_s,
                    hold_seconds=hold_seconds, tactile_force_min_n=.25,
                    close_settle_seconds=2.0, lift_height_m=.10)
    with _record_contacts(executor, object_id):
        result = executor.run(task)
    initial = float(result.metrics["initial_object_position_m"][2])
    heights = [float(row["evaluation"][object_id]["position"][2]) for row in result.trace]
    forces = [float(row["tactile"]["total_normal_force_n"]) for row in result.trace]
    contacts = [int(row["tactile"]["contact_count"]) for row in result.trace]
    return CalibrationRecord(candidate, result, max(heights, default=initial) - initial,
                             max(forces, default=0.0), max(contacts, default=0),
                             score_lift_evidence(result, object_id))


def calibrate(executor: V7TaskExecutor, object_id: str,
              candidates: Iterable[GraspCandidate], *, phase_timeout_s: float = 6.0,
              hold_seconds: float = .08,
              record_callback: Callable[[int, CalibrationRecord], None] | None = None,
              retain_traces: bool = True) -> tuple[CalibrationRecord, ...]:
    records = []
    for index, candidate in enumerate(candidates):
        record = evaluate_candidate(executor, object_id, candidate,
                                    phase_timeout_s=phase_timeout_s, hold_seconds=hold_seconds)
        if record_callback is not None:
            record_callback(index, record)
        if not retain_traces:
            record = replace(record, result=replace(record.result, trace=()))
        records.append(record)
    return tuple(sorted(records, key=lambda record: record.score, reverse=True))


__all__ = ["CalibrationContactReader", "CalibrationRecord", "GraspCandidate", "LiftEvidence",
           "calibrate", "evaluate_candidate", "score_lift_evidence"]
