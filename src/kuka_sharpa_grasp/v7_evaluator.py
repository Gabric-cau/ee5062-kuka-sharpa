"""Evidence-only evaluation contracts for the V7 dexterity milestone.

The functions in this module consume recorded poses and collision events.  No
MuJoCo state is read and no command is issued, so a controller cannot make a
regrasp pass by writing an oracle pose into the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def _finite_vector(values: Sequence[float], length: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain {length} finite values") from error
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain {length} finite values")
    return result


def _unit_quaternion(values: Sequence[float], field: str) -> tuple[float, float, float, float]:
    result = _finite_vector(values, 4, field)
    norm = math.sqrt(sum(value * value for value in result))
    if norm < 1e-12:
        raise ValueError(f"{field} must be non-zero")
    return tuple(value / norm for value in result)  # type: ignore[return-value]


def _quat_conjugate(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]


def _quat_multiply(first: tuple[float, float, float, float],
                   second: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    pure = (0.0, vector[0], vector[1], vector[2])
    rotated = _quat_multiply(_quat_multiply(quaternion, pure), _quat_conjugate(quaternion))
    return rotated[1], rotated[2], rotated[3]


def _quaternion_angle(first: tuple[float, float, float, float],
                      second: tuple[float, float, float, float]) -> float:
    relative = _quat_multiply(_quat_conjugate(first), second)
    scalar = min(1.0, max(-1.0, abs(float(relative[0]))))
    return 2.0 * math.acos(scalar)


@dataclass(frozen=True)
class RegraspSpec:
    """Thresholds for a supported finger-level regrasp evidence check."""

    min_relative_translation_m: float = 0.015
    min_relative_rotation_rad: float = math.radians(15.0)
    max_wrist_translation_m: float = 0.020
    max_wrist_rotation_rad: float = math.radians(10.0)
    min_hand_contact_force_n: float = 0.5
    min_finger_motion_m: float = 0.003
    min_contact_transitions: int = 1
    max_object_drop_m: float = 0.005
    requirement: str = "either"

    def __post_init__(self) -> None:
        fields = (
            self.min_relative_translation_m, self.min_relative_rotation_rad,
            self.max_wrist_translation_m, self.max_wrist_rotation_rad,
            self.min_hand_contact_force_n, self.min_finger_motion_m, self.max_object_drop_m,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in fields):
            raise ValueError("regrasp thresholds must be finite and non-negative")
        if self.max_wrist_translation_m <= 0 or self.max_wrist_rotation_rad <= 0:
            raise ValueError("wrist tolerances must be positive")
        if int(self.min_contact_transitions) != self.min_contact_transitions or self.min_contact_transitions < 0:
            raise ValueError("min_contact_transitions must be a non-negative integer")
        if self.requirement not in {"either", "both"}:
            raise ValueError("requirement must be 'either' or 'both'")


@dataclass(frozen=True)
class PoseSnapshot:
    """Recorded wrist and object poses at one instant.

    Quaternions use MuJoCo's ``(w, x, y, z)`` convention.  The evaluator
    computes the object pose in the wrist frame before comparing the two
    snapshots, so moving the wrist and object together does not fake a pass.
    """

    timestamp: float
    wrist_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    wrist_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "wrist_position", _finite_vector(self.wrist_position, 3, "wrist_position"))
        object.__setattr__(self, "object_position", _finite_vector(self.object_position, 3, "object_position"))
        object.__setattr__(self, "wrist_quaternion", _unit_quaternion(self.wrist_quaternion, "wrist_quaternion"))
        object.__setattr__(self, "object_quaternion", _unit_quaternion(self.object_quaternion, "object_quaternion"))


@dataclass(frozen=True)
class RegraspTraceSample:
    """One continuous simulator sample used to rule out endpoint-only claims."""

    pose: PoseSnapshot
    hand_contact_force_n: float
    finger_aperture_m: float
    finger_contact_count: int
    support_contact: bool
    object_height_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose, PoseSnapshot):
            raise TypeError("pose must be a PoseSnapshot")
        force = float(self.hand_contact_force_n)
        aperture = float(self.finger_aperture_m)
        height = float(self.object_height_m)
        if not math.isfinite(force) or force < 0:
            raise ValueError("hand_contact_force_n must be finite and non-negative")
        if not math.isfinite(aperture) or aperture < 0:
            raise ValueError("finger_aperture_m must be finite and non-negative")
        if int(self.finger_contact_count) != self.finger_contact_count or int(self.finger_contact_count) < 0:
            raise ValueError("finger_contact_count must be non-negative")
        if not math.isfinite(height):
            raise ValueError("object_height_m must be finite")
        object.__setattr__(self, "hand_contact_force_n", force)
        object.__setattr__(self, "finger_aperture_m", aperture)
        object.__setattr__(self, "finger_contact_count", int(self.finger_contact_count))
        object.__setattr__(self, "support_contact", bool(self.support_contact))
        object.__setattr__(self, "object_height_m", height)


@dataclass(frozen=True)
class RegraspMeasurement:
    """Derived pose deltas for a pair of pre/post snapshots."""

    before: PoseSnapshot
    after: PoseSnapshot
    trace: tuple[RegraspTraceSample, ...] = ()

    def __post_init__(self) -> None:
        if self.after.timestamp < self.before.timestamp:
            raise ValueError("regrasp snapshots must be time ordered")
        trace = tuple(self.trace)
        if any(not isinstance(sample, RegraspTraceSample) for sample in trace):
            raise TypeError("trace must contain RegraspTraceSample values")
        if any(trace[index + 1].pose.timestamp < trace[index].pose.timestamp for index in range(len(trace) - 1)):
            raise ValueError("regrasp trace timestamps must be nondecreasing")
        object.__setattr__(self, "trace", trace)

    @staticmethod
    def _relative_pose(snapshot: PoseSnapshot) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        wrist_inverse = _quat_conjugate(snapshot.wrist_quaternion)
        displacement = tuple(
            snapshot.object_position[index] - snapshot.wrist_position[index] for index in range(3)
        )
        return _quat_rotate(wrist_inverse, displacement), _quat_multiply(wrist_inverse, snapshot.object_quaternion)

    @property
    def duration_s(self) -> float:
        return self.after.timestamp - self.before.timestamp

    @property
    def wrist_translation_m(self) -> float:
        if not self.trace:
            return float(np.linalg.norm(np.asarray(self.after.wrist_position) - self.before.wrist_position))
        origin = np.asarray(self.trace[0].pose.wrist_position)
        return max(float(np.linalg.norm(np.asarray(sample.pose.wrist_position) - origin)) for sample in self.trace)

    @property
    def wrist_rotation_rad(self) -> float:
        if not self.trace:
            return _quaternion_angle(self.before.wrist_quaternion, self.after.wrist_quaternion)
        origin = self.trace[0].pose.wrist_quaternion
        return max(_quaternion_angle(origin, sample.pose.wrist_quaternion) for sample in self.trace)

    @property
    def relative_translation_m(self) -> float:
        first, _ = self._relative_pose(self.before)
        second, _ = self._relative_pose(self.after)
        return float(np.linalg.norm(np.asarray(second) - first))

    @property
    def relative_rotation_rad(self) -> float:
        _, first = self._relative_pose(self.before)
        _, second = self._relative_pose(self.after)
        return _quaternion_angle(first, second)

    @property
    def finger_motion_m(self) -> float:
        if not self.trace:
            return 0.0
        apertures = [sample.finger_aperture_m for sample in self.trace]
        return max(apertures) - min(apertures)

    @property
    def contact_transitions(self) -> int:
        if not self.trace:
            return 0
        return sum(self.trace[index].finger_contact_count != self.trace[index - 1].finger_contact_count
                   for index in range(1, len(self.trace)))

    @property
    def continuous_hand_contact(self) -> bool:
        return bool(self.trace) and all(sample.hand_contact_force_n > 0 for sample in self.trace)

    @property
    def continuous_support_contact(self) -> bool:
        return bool(self.trace) and all(sample.support_contact for sample in self.trace)

    @property
    def max_object_drop_m(self) -> float:
        if not self.trace:
            return 0.0
        heights = [sample.object_height_m for sample in self.trace]
        return max(0.0, heights[0] - min(heights))


@dataclass(frozen=True)
class RegraspEvidence:
    """Reportable result of :func:`evaluate_regrasp`."""

    accepted: bool
    relative_translation_m: float
    relative_rotation_rad: float
    wrist_translation_m: float
    wrist_rotation_rad: float
    duration_s: float
    finger_motion_m: float
    contact_transitions: int
    max_object_drop_m: float
    continuous_hand_contact: bool
    continuous_support_contact: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "relative_translation_m": self.relative_translation_m,
            "relative_rotation_rad": self.relative_rotation_rad,
            "wrist_translation_m": self.wrist_translation_m,
            "wrist_rotation_rad": self.wrist_rotation_rad,
            "duration_s": self.duration_s,
            "finger_motion_m": self.finger_motion_m,
            "contact_transitions": self.contact_transitions,
            "max_object_drop_m": self.max_object_drop_m,
            "continuous_hand_contact": self.continuous_hand_contact,
            "continuous_support_contact": self.continuous_support_contact,
            "reasons": list(self.reasons),
        }


def evaluate_regrasp(measurement: RegraspMeasurement, spec: RegraspSpec = RegraspSpec()) -> RegraspEvidence:
    """Require relative motion plus continuous support/contact trace evidence."""
    translation_ok = measurement.relative_translation_m >= spec.min_relative_translation_m
    rotation_ok = measurement.relative_rotation_rad >= spec.min_relative_rotation_rad
    wrist_ok = (
        measurement.wrist_translation_m <= spec.max_wrist_translation_m
        and measurement.wrist_rotation_rad <= spec.max_wrist_rotation_rad
    )
    trace_ok = bool(measurement.trace)
    contact_ok = bool(measurement.trace) and all(
        sample.hand_contact_force_n >= spec.min_hand_contact_force_n for sample in measurement.trace
    )
    support_ok = measurement.continuous_support_contact
    finger_ok = measurement.finger_motion_m >= spec.min_finger_motion_m
    transition_ok = measurement.contact_transitions >= spec.min_contact_transitions
    drop_ok = measurement.max_object_drop_m <= spec.max_object_drop_m
    motion_ok = (translation_ok or rotation_ok) if spec.requirement == "either" else (translation_ok and rotation_ok)
    reasons: list[str] = []
    if not wrist_ok:
        reasons.append("WRIST_MOVED_TOO_FAR")
    if not motion_ok:
        reasons.append("INSUFFICIENT_RELATIVE_OBJECT_MOTION")
    if not trace_ok:
        reasons.append("NO_CONTINUOUS_TRACE")
    if not contact_ok:
        reasons.append("HAND_CONTACT_NOT_CONTINUOUS")
    if not support_ok:
        reasons.append("SUPPORT_CONTACT_NOT_CONTINUOUS")
    if not finger_ok:
        reasons.append("INSUFFICIENT_FINGER_MOTION")
    if not transition_ok:
        reasons.append("NO_FINGER_CONTACT_TRANSITION")
    if not drop_ok:
        reasons.append("OBJECT_DROPPED")
    if translation_ok:
        reasons.append("RELATIVE_TRANSLATION_THRESHOLD")
    if rotation_ok:
        reasons.append("RELATIVE_ROTATION_THRESHOLD")
    return RegraspEvidence(
        accepted=bool(wrist_ok and motion_ok and trace_ok and contact_ok and support_ok and finger_ok and transition_ok and drop_ok),
        relative_translation_m=measurement.relative_translation_m,
        relative_rotation_rad=measurement.relative_rotation_rad,
        wrist_translation_m=measurement.wrist_translation_m,
        wrist_rotation_rad=measurement.wrist_rotation_rad,
        duration_s=measurement.duration_s,
        finger_motion_m=measurement.finger_motion_m,
        contact_transitions=measurement.contact_transitions,
        max_object_drop_m=measurement.max_object_drop_m,
        continuous_hand_contact=contact_ok,
        continuous_support_contact=support_ok,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CollisionEvent:
    """One recorded MuJoCo contact classified later from stable geometry IDs."""

    timestamp: float
    geometry_a: str
    geometry_b: str
    normal_force_n: float = 0.0

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        force = float(self.normal_force_n)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("collision timestamp must be finite and non-negative")
        if not isinstance(self.geometry_a, str) or not isinstance(self.geometry_b, str):
            raise ValueError("collision geometries must be strings")
        if not self.geometry_a or not self.geometry_b:
            raise ValueError("collision geometries must be non-empty")
        if not math.isfinite(force) or force < 0:
            raise ValueError("normal_force_n must be finite and non-negative")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "normal_force_n", force)


def _matches(name: str, identifiers: set[str]) -> bool:
    return any(name == identifier or name.endswith("_" + identifier) for identifier in identifiers)


@dataclass(frozen=True)
class CollisionAudit:
    """Aggregated collision counts for a workspace transfer."""

    total_events: int
    obstacle_events: int
    non_target_events: int
    target_events: int
    unclassified_events: int
    ignored_environment_events: int
    unknown_robot_events: int
    peak_obstacle_force_n: float
    peak_non_target_force_n: float
    first_obstacle_time_s: float | None
    first_non_target_time_s: float | None

    @property
    def workspace_safe(self) -> bool:
        return self.obstacle_events == 0 and self.non_target_events == 0 and self.unknown_robot_events == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "obstacle_events": self.obstacle_events,
            "non_target_events": self.non_target_events,
            "target_events": self.target_events,
            "unclassified_events": self.unclassified_events,
            "ignored_environment_events": self.ignored_environment_events,
            "unknown_robot_events": self.unknown_robot_events,
            "peak_obstacle_force_n": self.peak_obstacle_force_n,
            "peak_non_target_force_n": self.peak_non_target_force_n,
            "first_obstacle_time_s": self.first_obstacle_time_s,
            "first_non_target_time_s": self.first_non_target_time_s,
            "workspace_safe": self.workspace_safe,
        }


def aggregate_collision_events(
    events: Iterable[CollisionEvent],
    *,
    obstacle_ids: Iterable[str] = (),
    non_target_ids: Iterable[str] = (),
    target_ids: Iterable[str] = (),
    robot_ids: Iterable[str] = (),
) -> CollisionAudit:
    """Count robot workspace contacts, ignoring passive object/table contacts.

    ``robot_ids`` should contain stable hand/arm geometry IDs when available.
    With no explicit set, common KUKA/Sharpa prefixes are used.  IDs are
    required to be disjoint so a mislabeled geometry cannot be silently counted
    as both a target and an obstacle.
    """
    obstacles, non_targets, targets, robots = (set(map(str, values)) for values in
                                               (obstacle_ids, non_target_ids, target_ids, robot_ids))
    if obstacles & non_targets or obstacles & targets or non_targets & targets:
        raise ValueError("obstacle, non-target, and target IDs must be disjoint")
    if not robots:
        robots = {"__auto_robot_prefixes__"}

    def robot_name(name: str) -> bool:
        if "__auto_robot_prefixes__" in robots:
            return name.startswith(("left_", "right_", "link", "arm", "wrist", "iiwa", "sharpa_"))
        return _matches(name, robots)

    counts = {"obstacle": 0, "non_target": 0, "target": 0, "unclassified": 0}
    peak_obstacle = peak_non_target = 0.0
    first_obstacle = first_non_target = None
    total = ignored = unknown_robot = 0
    for event in events:
        if not isinstance(event, CollisionEvent):
            raise TypeError("events must contain CollisionEvent values")
        total += 1
        names = (event.geometry_a, event.geometry_b)
        if not any(robot_name(name) for name in names):
            ignored += 1
            continue
        is_obstacle = any(_matches(name, obstacles) for name in names)
        is_non_target = any(_matches(name, non_targets) for name in names)
        is_target = any(_matches(name, targets) for name in names)
        if is_obstacle:
            counts["obstacle"] += 1
            peak_obstacle = max(peak_obstacle, event.normal_force_n)
            first_obstacle = event.timestamp if first_obstacle is None else min(first_obstacle, event.timestamp)
        if is_non_target:
            counts["non_target"] += 1
            peak_non_target = max(peak_non_target, event.normal_force_n)
            first_non_target = event.timestamp if first_non_target is None else min(first_non_target, event.timestamp)
        if is_target:
            counts["target"] += 1
        if not (is_obstacle or is_non_target or is_target):
            counts["unclassified"] += 1
            unknown_robot += 1
    return CollisionAudit(
        total_events=total, obstacle_events=counts["obstacle"], non_target_events=counts["non_target"],
        target_events=counts["target"], unclassified_events=counts["unclassified"],
        ignored_environment_events=ignored, unknown_robot_events=unknown_robot,
        peak_obstacle_force_n=peak_obstacle, peak_non_target_force_n=peak_non_target,
        first_obstacle_time_s=first_obstacle, first_non_target_time_s=first_non_target,
    )


__all__ = [
    "CollisionAudit", "CollisionEvent", "PoseSnapshot", "RegraspEvidence",
    "RegraspMeasurement", "RegraspSpec", "aggregate_collision_events", "evaluate_regrasp",
]

