"""Evidence-driven batch metrics for the V7 workbench.

This module deliberately separates *observed* outcomes from missing evidence.
An empty collision log means zero recorded collisions; a missing collision log
means the collision metric is unknown.  The same rule applies to contact,
regrasp and docking evidence, so a teleoperation demo cannot silently become
an end-to-end success claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v7_evaluator import (
    CollisionEvent,
    RegraspEvidence,
    RegraspMeasurement,
    RegraspSpec,
    RegraspTraceSample,
    PoseSnapshot,
    aggregate_collision_events,
    evaluate_regrasp,
)
from .v7_scene import DockSlot, V7SceneConfig, default_v7_scene_config


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "fail", "failed"}:
            return False
    return None


def _position(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return None
    return result


def _quaternion(value: Any) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    norm = math.sqrt(sum(item * item for item in values))
    if norm < 1e-12:
        return None
    return tuple(item / norm for item in values)  # type: ignore[return-value]


def _quaternion_angle(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(float(np.dot(np.asarray(first, dtype=float), np.asarray(second, dtype=float))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


@dataclass(frozen=True)
class BatchMetricSpec:
    """Thresholds used by :func:`evaluate_episode`.

    These are evaluation tolerances, not claims about calibrated hardware.
    """

    lift_height_m: float = 0.020
    lift_hold_samples: int = 3
    drop_tolerance_m: float = 0.015
    grasp_force_n: float = 0.5
    # Includes the declared V7 wrist-to-tool grasp offset (~0.25 m) while
    # still requiring same-sample contact and object motion.
    grasp_proximity_m: float = 0.30
    dock_position_tolerance_m: float = 0.035
    dock_orientation_tolerance_rad: float = math.radians(20.0)
    require_regrasp: bool = False

    def __post_init__(self) -> None:
        numeric = (self.lift_height_m, self.drop_tolerance_m, self.grasp_force_n, self.grasp_proximity_m,
                   self.dock_position_tolerance_m, self.dock_orientation_tolerance_rad)
        if not all(math.isfinite(float(item)) and float(item) >= 0 for item in numeric):
            raise ValueError("metric thresholds must be finite and non-negative")
        if int(self.lift_hold_samples) != self.lift_hold_samples or self.lift_hold_samples < 1:
            raise ValueError("lift_hold_samples must be a positive integer")


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode outcomes with explicit unknown values for missing evidence."""

    episode_id: str
    seed: int | None
    target_id: str | None
    duration_s: float | None
    target_selection_correct: bool | None
    target_selection_source: str
    grasp_evidence: bool | None
    lift_success: bool | None
    drop_detected: bool | None
    non_target_collision_count: int | None
    obstacle_collision_count: int | None
    target_collision_count: int | None
    unknown_collision_count: int | None
    collision_evidence_available: bool
    peak_normal_force_n: float | None
    peak_tangential_force_n: float | None
    peak_slip_score: float | None
    docking_position_error_m: float | None
    docking_orientation_error_rad: float | None
    docking_success: bool | None
    intervention_count: int | None
    regrasp: dict[str, Any] | None
    success: bool | None
    unknown_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def no_drop(self) -> bool | None:
        return None if self.drop_detected is None else not self.drop_detected

    @property
    def collision_free(self) -> bool | None:
        if not self.collision_evidence_available:
            return None
        return bool(self.obstacle_collision_count == 0 and self.non_target_collision_count == 0
                    and self.unknown_collision_count == 0)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "target_id": self.target_id,
            "duration_s": self.duration_s,
            "target_selection_correct": self.target_selection_correct,
            "target_selection_source": self.target_selection_source,
            "grasp_evidence": self.grasp_evidence,
            "lift_success": self.lift_success,
            "drop_detected": self.drop_detected,
            "non_target_collision_count": self.non_target_collision_count,
            "obstacle_collision_count": self.obstacle_collision_count,
            "target_collision_count": self.target_collision_count,
            "unknown_collision_count": self.unknown_collision_count,
            "collision_evidence_available": self.collision_evidence_available,
            "peak_normal_force_n": self.peak_normal_force_n,
            "peak_tangential_force_n": self.peak_tangential_force_n,
            "peak_slip_score": self.peak_slip_score,
            "docking_position_error_m": self.docking_position_error_m,
            "docking_orientation_error_rad": self.docking_orientation_error_rad,
            "docking_success": self.docking_success,
            "intervention_count": self.intervention_count,
            "regrasp": self.regrasp,
            "success": self.success,
            "unknown_reasons": list(self.unknown_reasons),
        }
        return payload


@dataclass(frozen=True)
class BatchSummary:
    """Counts and rates with a denominator for every reported rate."""

    episodes: tuple[EpisodeMetrics, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    def _rate(self, field: str) -> dict[str, Any]:
        observed = [getattr(item, field) for item in self.episodes]
        known = [value for value in observed if isinstance(value, bool)]
        numerator = sum(value is True for value in known)
        denominator = len(known)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "unknown": len(observed) - denominator,
            "rate": (numerator / denominator if denominator else None),
        }

    def as_dict(self) -> dict[str, Any]:
        success = self._rate("success")
        return {
            "episode_count": self.episode_count,
            "success_count": success["numerator"],
            "success_denominator": success["denominator"],
            "success_unknown": success["unknown"],
            "success_rate": success["rate"],
            "metric_rates": {
                field: self._rate(field)
                for field in ("target_selection_correct", "grasp_evidence", "lift_success",
                              "drop_detected", "no_drop", "collision_free", "docking_success", "success")
            },
            "collision_observation": {
                "episodes_with_logs": sum(item.collision_evidence_available for item in self.episodes),
                "episodes_without_logs": sum(not item.collision_evidence_available for item in self.episodes),
            },
            "episodes": [item.as_dict() for item in self.episodes],
        }

    def write_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        rows = [item.as_dict() for item in self.episodes]
        fieldnames = list(EpisodeMetrics.__dataclass_fields__.keys())
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["regrasp"] = json.dumps(row["regrasp"], sort_keys=True) if row["regrasp"] is not None else ""
                row["unknown_reasons"] = ";".join(row["unknown_reasons"])
                writer.writerow(row)


def _slot_for(config: V7SceneConfig, target_id: str | None) -> DockSlot | None:
    if target_id is None:
        return None
    spec = next((item for item in config.registry.objects if item.object_id == target_id), None)
    if spec is None or spec.dock_slot is None:
        return None
    return next((slot for slot in config.slots if slot.slot_id == spec.dock_slot), None)


def _target_trajectory(trace: Sequence[Mapping[str, Any]], target_id: str) -> tuple[list[np.ndarray], list[tuple[float, float, float, float] | None], list[np.ndarray | None]]:
    positions: list[np.ndarray] = []
    orientations: list[tuple[float, float, float, float] | None] = []
    wrists: list[np.ndarray | None] = []
    for sample in trace:
        evaluation = sample.get("evaluation", {})
        state = evaluation.get(target_id, {}) if isinstance(evaluation, Mapping) else {}
        position = _position(state.get("position")) if isinstance(state, Mapping) else None
        quaternion = _quaternion(state.get("quaternion")) if isinstance(state, Mapping) else None
        if position is not None:
            positions.append(position)
            orientations.append(quaternion)
            wrists.append(_position(sample.get("wrist_position")))
    return positions, orientations, wrists


def _force_metrics(trace: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None, float | None, bool | None]:
    normals: list[float] = []
    tangentials: list[float] = []
    slips: list[float] = []
    contact_seen = False
    reading_seen = False
    for sample in trace:
        tactile = sample.get("tactile", {})
        fingers = tactile.get("fingers", {}) if isinstance(tactile, Mapping) else {}
        if not isinstance(fingers, Mapping):
            continue
        for reading in fingers.values():
            if not isinstance(reading, Mapping):
                continue
            normal = _finite(reading.get("normal_force_n"))
            tangent = _finite(reading.get("tangential_force_n"))
            slip = _finite(reading.get("slip_score"))
            if normal is not None:
                reading_seen = True
                normals.append(max(0.0, normal))
                contact_seen = contact_seen or normal > 0.0 or int(reading.get("contacts", 0) or 0) > 0
            if tangent is not None:
                tangentials.append(max(0.0, tangent))
            if slip is not None:
                slips.append(float(np.clip(slip, 0.0, 1.0)))
    return (max(normals) if normals else None, max(tangentials) if tangentials else None,
            max(slips) if slips else None, contact_seen if reading_seen else None)


def _target_contact_near_wrist(trace: Sequence[Mapping[str, Any]], target_id: str,
                               proximity_m: float, minimum_force_n: float) -> bool | None:
    """Return target-proximity/contact evidence from the same sample.

    Aggregate force peaks alone cannot identify which object was touched.  The
    target trajectory and wrist pose are evaluation-only state, so they are
    used here only to require temporal overlap with the tactile signal.
    """
    reading_seen = False
    for sample in trace:
        evaluation = sample.get("evaluation", {})
        state = evaluation.get(target_id, {}) if isinstance(evaluation, Mapping) else {}
        target = _position(state.get("position")) if isinstance(state, Mapping) else None
        wrist = _position(sample.get("wrist_position"))
        tactile = sample.get("tactile", {})
        fingers = tactile.get("fingers", {}) if isinstance(tactile, Mapping) else {}
        normal_total = 0.0
        contact_count = 0
        if isinstance(fingers, Mapping):
            for reading in fingers.values():
                if not isinstance(reading, Mapping):
                    continue
                normal = _finite(reading.get("normal_force_n"))
                if normal is None:
                    continue
                reading_seen = True
                normal_total += max(0.0, normal)
                try:
                    contact_count += int(reading.get("contacts", 0) or 0)
                except (TypeError, ValueError):
                    pass
        if target is not None and wrist is not None and normal_total >= minimum_force_n and contact_count > 0:
            if float(np.linalg.norm(target - wrist)) <= proximity_m:
                return True
    return None if not reading_seen else False


def _coerce_collision_events(raw_events: Iterable[Mapping[str, Any]] | None) -> list[CollisionEvent] | None:
    if raw_events is None:
        return None
    events: list[CollisionEvent] = []
    for raw in raw_events:
        if isinstance(raw, CollisionEvent):
            events.append(raw)
            continue
        if not isinstance(raw, Mapping):
            continue
        geometry_a = raw.get("geometry_a", raw.get("robot_geom"))
        geometry_b = raw.get("geometry_b", raw.get("other_geom"))
        timestamp = _finite(raw.get("timestamp", raw.get("time_s")))
        force = _finite(raw.get("normal_force_n", 0.0))
        if geometry_a is None or geometry_b is None or timestamp is None or force is None:
            continue
        try:
            events.append(CollisionEvent(timestamp, str(geometry_a), str(geometry_b), force))
        except ValueError:
            continue
    return events


def _regrasp_from_trace(trace: Sequence[Mapping[str, Any]], target_id: str, spec: RegraspSpec) -> RegraspEvidence | None:
    """Build a regrasp contract only when all continuous fields are recorded."""
    samples: list[RegraspTraceSample] = []
    for sample in trace:
        evaluation = sample.get("evaluation", {})
        state = evaluation.get(target_id, {}) if isinstance(evaluation, Mapping) else {}
        wrist_position = _position(sample.get("wrist_position"))
        object_position = _position(state.get("position")) if isinstance(state, Mapping) else None
        wrist_quaternion = _quaternion(sample.get("wrist_quaternion"))
        object_quaternion = _quaternion(state.get("quaternion")) if isinstance(state, Mapping) else None
        regrasp = sample.get("regrasp", {})
        tactile = sample.get("tactile", {})
        fingers = tactile.get("fingers", {}) if isinstance(tactile, Mapping) else {}
        if not (wrist_position is not None and object_position is not None and wrist_quaternion and object_quaternion
                and isinstance(regrasp, Mapping)):
            return None
        hand_force = _finite(regrasp.get("hand_contact_force_n"))
        aperture = _finite(regrasp.get("finger_aperture_m"))
        contact_count = regrasp.get("finger_contact_count")
        support = _bool(regrasp.get("support_contact"))
        height = _finite(regrasp.get("object_height_m"))
        if hand_force is None:
            # A tactile trace can provide force, but contact identity remains
            # an explicit regrasp field to avoid treating any obstacle touch as
            # a grasp.
            hand_force = _finite(tactile.get("total_normal_force_n")) if isinstance(tactile, Mapping) else None
        if aperture is None or support is None or height is None or contact_count is None or hand_force is None:
            return None
        before_pose = PoseSnapshot(0.0, tuple(wrist_position), tuple(object_position), tuple(wrist_quaternion), tuple(object_quaternion))
        pose = PoseSnapshot(float(_finite(sample.get("time_s")) or 0.0), tuple(wrist_position), tuple(object_position), tuple(wrist_quaternion), tuple(object_quaternion))
        samples.append(RegraspTraceSample(pose, hand_force, aperture, int(contact_count), support, height))
    if len(samples) < 2:
        return None
    return evaluate_regrasp(RegraspMeasurement(samples[0].pose, samples[-1].pose, tuple(samples)), spec)


def evaluate_episode(
    episode_id: str,
    trace: Sequence[Mapping[str, Any]],
    *,
    target_id: str | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    config: V7SceneConfig | None = None,
    seed: int | None = None,
    thresholds: BatchMetricSpec = BatchMetricSpec(),
    regrasp_spec: RegraspSpec = RegraspSpec(),
) -> EpisodeMetrics:
    """Compute one episode's metrics from recorded evidence.

    ``trace`` must be the JSON list written by ``run_v7_workbench.py``.
    Collision evidence is intentionally tri-state: pass ``events=[]`` for an
    observed collision-free run and ``events=None`` when no event log exists.
    """
    config = default_v7_scene_config(seed or 0) if config is None else config
    trace = tuple(item for item in trace if isinstance(item, Mapping))
    raw_events = None if events is None else tuple(events)
    expected = target_id or (str(trace[-1].get("selected_target")) if trace and trace[-1].get("selected_target") else None)
    actionable = {item.object_id for item in config.registry.actionable}
    if expected not in actionable:
        expected = None
    target_event_values = [str(item.get("target")) for item in raw_events or ()
                          if isinstance(item, Mapping) and item.get("event") == "OPERATOR_TARGET" and item.get("target")]
    selected_values = target_event_values or [str(item.get("selected_target")) for item in trace if item.get("selected_target")]
    target_selection_correct: bool | None
    if not trace or expected is None:
        target_selection_correct = None
        selection_source = "missing"
    else:
        target_selection_correct = expected in selected_values
        selection_source = "events.OPERATOR_TARGET" if target_event_values else "trace.selected_target"
    positions, orientations, wrist_positions = _target_trajectory(trace, expected) if expected else ([], [], [])
    initial = positions[0] if positions else None
    heights = [float(position[2]) for position in positions]
    lift_success: bool | None = None
    drop_detected: bool | None = None
    if initial is not None and len(positions) >= thresholds.lift_hold_samples:
        elevated = [float(position[2] - initial[2]) >= thresholds.lift_height_m for position in positions]
        lift_success = any(all(elevated[index:index + thresholds.lift_hold_samples])
                           for index in range(max(1, len(elevated) - thresholds.lift_hold_samples + 1)))
    peak_normal, peak_tangent, peak_slip, contact_seen = _force_metrics(trace)
    # Contact is evidence only when the target also moved relative to its
    # initial state.  This avoids counting a finger touching an obstacle as a
    # successful target grasp.
    target_contact_near_wrist = (_target_contact_near_wrist(trace, expected,
                                                             thresholds.grasp_proximity_m,
                                                             thresholds.grasp_force_n)
                                 if expected else None)
    # Force on any finger is not target identity.  Require target trajectory
    # evidence and a recorded wrist-to-target proximity before calling this a
    # grasp; otherwise report a measured failure (or unknown when no contact
    # trace exists), never a success from arbitrary obstacle contact.
    grasp_evidence = None if target_contact_near_wrist is None else bool(target_contact_near_wrist and lift_success)

    collision_events = _coerce_collision_events(raw_events)
    collision_available = collision_events is not None
    if collision_events is None:
        obstacle_count = non_target_count = target_count = unknown_count = None
    else:
        obstacles = tuple(item.obstacle_id for item in config.obstacles)
        non_targets = tuple(item.object_id for item in config.registry.distractors)
        targets = (expected,) if expected else tuple(item.object_id for item in config.registry.actionable)
        audit = aggregate_collision_events(collision_events,
            obstacle_ids=obstacles, non_target_ids=non_targets, target_ids=targets)
        obstacle_count, non_target_count = audit.obstacle_events, audit.non_target_events
        target_count, unknown_count = audit.target_events, audit.unknown_robot_events

    slot = _slot_for(config, expected)
    dock_pos_error = dock_rot_error = None
    docking_success = None
    if slot is not None and positions:
        dock_pos_error = float(np.linalg.norm(positions[-1] - np.asarray(slot.pose, dtype=float)))
        expected_quat = (math.cos(slot.yaw / 2.0), 0.0, 0.0, math.sin(slot.yaw / 2.0))
        if orientations[-1] is not None:
            dock_rot_error = _quaternion_angle(orientations[-1], expected_quat)
            docking_success = (dock_pos_error <= thresholds.dock_position_tolerance_m
                               and dock_rot_error <= thresholds.dock_orientation_tolerance_rad)
    if heights:
        peak_height = max(heights)
        # A controlled descent into the declared slot is a placement, not a
        # drop.  Without a docking pass, a large post-lift height loss remains
        # a drop even if the object happens to finish near the table.
        drop_detected = bool(lift_success and (peak_height - heights[-1]) > thresholds.drop_tolerance_m
                             and docking_success is not True)

    intervention_values = [_finite(item.get("interventions")) for item in trace]
    intervention_count = int(max(value for value in intervention_values if value is not None)) if any(value is not None for value in intervention_values) else None
    regrasp_evidence = _regrasp_from_trace(trace, expected, regrasp_spec) if expected else None
    regrasp_payload = regrasp_evidence.as_dict() if regrasp_evidence else None

    unknown_reasons: list[str] = []
    for value, reason in ((target_selection_correct, "target_selection"), (grasp_evidence, "grasp"),
                          (lift_success, "lift"), (drop_detected, "drop"), (docking_success, "docking"),
                          (intervention_count, "interventions")):
        if value is None:
            unknown_reasons.append(f"missing_{reason}_evidence")
    if not collision_available:
        unknown_reasons.append("missing_collision_log")
    if thresholds.require_regrasp and regrasp_evidence is None:
        unknown_reasons.append("missing_regrasp_trace")
    elif thresholds.require_regrasp and regrasp_evidence is not None and not regrasp_evidence.accepted:
        unknown_reasons.append("regrasp_rejected")

    required = (target_selection_correct, grasp_evidence, lift_success, drop_detected,
                docking_success)
    if thresholds.require_regrasp:
        required += (None if regrasp_evidence is None else regrasp_evidence.accepted,)
    if any(value is None for value in required) or not collision_available:
        success = None
    else:
        success = bool(target_selection_correct and grasp_evidence and lift_success
                       and not drop_detected and docking_success
                       and obstacle_count == 0 and non_target_count == 0 and unknown_count == 0
                       and (not thresholds.require_regrasp or bool(regrasp_evidence and regrasp_evidence.accepted)))
    duration = _finite(trace[-1].get("time_s")) if trace else None
    return EpisodeMetrics(
        episode_id=str(episode_id), seed=seed, target_id=expected, duration_s=duration,
        target_selection_correct=target_selection_correct, target_selection_source=selection_source,
        grasp_evidence=grasp_evidence, lift_success=lift_success, drop_detected=drop_detected,
        non_target_collision_count=non_target_count, obstacle_collision_count=obstacle_count,
        target_collision_count=target_count, unknown_collision_count=unknown_count,
        collision_evidence_available=collision_available, peak_normal_force_n=peak_normal,
        peak_tangential_force_n=peak_tangent, peak_slip_score=peak_slip,
        docking_position_error_m=dock_pos_error, docking_orientation_error_rad=dock_rot_error,
        docking_success=docking_success, intervention_count=intervention_count,
        regrasp=regrasp_payload, success=success, unknown_reasons=tuple(unknown_reasons),
    )


def evaluate_batch(episodes: Iterable[Mapping[str, Any]], *, config: V7SceneConfig | None = None,
                   thresholds: BatchMetricSpec = BatchMetricSpec(),
                   regrasp_spec: RegraspSpec = RegraspSpec()) -> BatchSummary:
    """Evaluate records containing ``episode_id``, ``trace`` and optional events."""
    values = []
    for index, record in enumerate(episodes):
        if not isinstance(record, Mapping):
            raise TypeError("episodes must contain mapping records")
        values.append(evaluate_episode(
            str(record.get("episode_id", f"episode_{index:04d}")),
            record.get("trace", ()),
            target_id=record.get("target_id"), events=record.get("events"), config=config,
            seed=record.get("seed"), thresholds=thresholds, regrasp_spec=regrasp_spec))
    return BatchSummary(tuple(values))


__all__ = [
    "BatchMetricSpec", "BatchSummary", "EpisodeMetrics", "evaluate_batch", "evaluate_episode",
]
