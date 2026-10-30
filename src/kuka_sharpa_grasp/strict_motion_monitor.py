"""Read-only, target-privileged evaluation of native 2 ms motion records.

These inputs are evaluation truth, never planner observations or phase gates.
The caller supplies integrated target state, target-specific contact identity,
and geometric clearance; this module neither imports nor writes MuJoCo. Force
channels cover every external hand contact, including failed/recovery phases.
The predicates retain ``verify_public_ycb_strict.py`` boundaries. Passing them
does not certify collision avoidance against arbitrary clutter or evidence
authenticity: the runner must independently retain its native journal/archive.
"""

from dataclasses import dataclass
import math
from numbers import Real


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
TIMESTEP_S = .002
TIME_TOLERANCE_S = 1e-9


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a real number, not a boolean")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.):
        raise ValueError(name + " must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _vector(values, name, length, *, nonnegative=False):
    try:
        result = tuple(_number(value, name, nonnegative=nonnegative) for value in values)
    except TypeError as error:
        raise ValueError(name + " must be a numeric sequence") from error
    if len(result) != length:
        raise ValueError(name + " has the wrong length")
    return result


@dataclass(frozen=True)
class StrictMotionSample:
    """An immutable, same-native-sample evaluation record, all SI units.

    ``target_hand_contact`` includes unloaded/zero-force contacts. The two
    ``peak_normal_n`` fields are maximum absolute *per-contact* normal forces,
    matching the original any-contact > .05 N predicates, not contact sums.
    Channel forces instead sum absolute normals per finger/palm, excluding
    internal hand-hand contacts. ``minimum_robot_table_distance_m=None`` means
    no robot-table contacts; otherwise it is the minimum signed contact distance
    over all arm/hand versus table pairs. The clearance is signed geometry
    distance from target to all collision-enabled hand geometries. Timestamp
    presence alone cannot establish that these fields came from the same state.
    """

    timestamp_s: float
    target_position_world_m: tuple
    target_linear_speed_m_s: float
    target_angular_speed_rad_s: float
    target_hand_contact: bool
    target_hand_peak_normal_n: float
    target_table_peak_normal_n: float
    hand_clearance_m: float
    hand_channel_forces_n: tuple
    minimum_robot_table_distance_m: float | None
    release_commanded: bool
    phase: str = "UNKNOWN"

    def __post_init__(self):
        for name in ("timestamp_s", "target_linear_speed_m_s", "target_angular_speed_rad_s",
                     "target_hand_peak_normal_n", "target_table_peak_normal_n"):
            object.__setattr__(self, name, _number(getattr(self, name), name, nonnegative=True))
        object.__setattr__(self, "hand_clearance_m", _number(self.hand_clearance_m, "hand_clearance_m"))
        object.__setattr__(self, "target_position_world_m", _vector(
            self.target_position_world_m, "target_position_world_m", 3))
        object.__setattr__(self, "hand_channel_forces_n", _vector(
            self.hand_channel_forces_n, "hand_channel_forces_n", len(CHANNELS), nonnegative=True))
        if self.minimum_robot_table_distance_m is not None:
            object.__setattr__(self, "minimum_robot_table_distance_m", _number(
                self.minimum_robot_table_distance_m, "minimum_robot_table_distance_m"))
        if type(self.target_hand_contact) is not bool or type(self.release_commanded) is not bool:
            raise ValueError("contact and release flags must be booleans")
        if not self.target_hand_contact and self.target_hand_peak_normal_n > 0.:
            raise ValueError("a loaded target-hand contact requires target_hand_contact")
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("phase must be a nonempty string")


class StrictMotionMonitor:
    """Aggregate privileged labels without commanding or terminating a task.

    Always observe every native step, including prefix, abort and recovery. The
    first native timestamp must be .002 s. A missing/irregular 2 ms step resets
    both current intervals and permanently invalidates whole-episode continuity.
    A repeated/reversed timestamp also resets intervals and raises ValueError;
    its force/penetration peaks remain recorded. There is no reset method.
    """

    def __init__(self, *, initial_target_height_m, goal_position_world_m):
        self.initial_target_height_m = _number(initial_target_height_m, "initial_target_height_m")
        self.goal_position_world_m = _vector(goal_position_world_m, "goal_position_world_m", 3)
        self._previous_time = None
        self._held_since = self._released_since = None
        self._held_seconds = self._released_seconds = 0.
        self._contiguous = True
        self._observed_samples = self._accepted_samples = self._rejected_timestamps = 0
        self._discontinuities = 0
        self._peaks = dict.fromkeys(CHANNELS, 0.)
        self._phase_peaks = {}
        self._minimum_table_distance = None
        self._last_held = self._last_released = False

    def observe(self, sample):
        """Consume one frozen sample; return its instantaneous held/release flags.

        Invalid timestamp order is rejected after retaining its safety extrema.
        A caller may abort on rejection; it must preserve the resulting report.
        """
        if not isinstance(sample, StrictMotionSample):
            raise TypeError("StrictMotionSample is required")
        self._observed_samples += 1
        for name, value in zip(CHANNELS, sample.hand_channel_forces_n, strict=True):
            self._peaks[name] = max(self._peaks[name], value)
        self._phase_peaks[sample.phase] = max(self._phase_peaks.get(sample.phase, 0.),
                                            max(sample.hand_channel_forces_n))
        distance = sample.minimum_robot_table_distance_m
        if distance is not None:
            self._minimum_table_distance = (distance if self._minimum_table_distance is None
                                            else min(self._minimum_table_distance, distance))
        timestamp = sample.timestamp_s
        if self._previous_time is not None and timestamp <= self._previous_time:
            self._contiguous = False
            self._rejected_timestamps += 1
            self._held_since = self._released_since = None
            self._last_held = self._last_released = False
            raise ValueError("native timestamps must be strictly increasing; duplicate/reversed sample rejected")
        contiguous_step = (abs(timestamp - TIMESTEP_S) <= TIME_TOLERANCE_S
                           if self._previous_time is None else
                           abs(timestamp - self._previous_time - TIMESTEP_S) <= TIME_TOLERANCE_S)
        if not contiguous_step:
            self._contiguous = False
            self._discontinuities += 1
            self._held_since = self._released_since = None
        self._accepted_samples += 1
        self._previous_time = timestamp
        position = sample.target_position_world_m
        supported = sample.target_table_peak_normal_n > .05
        held = (position[2] > self.initial_target_height_m + .07
                and sample.target_hand_peak_normal_n > .05 and not supported)
        released = (sample.release_commanded and not sample.target_hand_contact and supported
                    and math.hypot(position[0] - self.goal_position_world_m[0],
                                   position[1] - self.goal_position_world_m[1]) < .025
                    and abs(position[2] - self.goal_position_world_m[2]) < .025
                    and sample.target_linear_speed_m_s < .04
                    and sample.target_angular_speed_rad_s < .2
                    and sample.hand_clearance_m > .10)
        if held:
            self._held_since = timestamp if self._held_since is None else self._held_since
            self._held_seconds = max(self._held_seconds, timestamp - self._held_since)
        else:
            self._held_since = None
        if released:
            self._released_since = timestamp if self._released_since is None else self._released_since
            self._released_seconds = max(self._released_seconds, timestamp - self._released_since)
        else:
            self._released_since = None
        self._last_held, self._last_released = bool(held), bool(released)
        return {"held": bool(held), "released_stable": bool(released)}

    def result(self, *, external_target_qpos_writes=None, integration_steps=None, native_samples=None,
               original_task_success=None):
        """Return a fresh JSON-compatible report; no success is required to report.

        ``task_success`` stays None unless the caller supplies its original task
        label and all native audit counts. It can only retain/weaken that label,
        never create one. The original evaluator must establish task ordering,
        metadata and artifact identity. Audit counts are runner assertions, not
        independently authenticated here. Omitting a count leaves proof unknown.
        """
        audit = dict(external_target_qpos_writes=external_target_qpos_writes,
                     integration_steps=integration_steps, native_samples=native_samples)
        for name, value in audit.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(name + " must be a nonnegative integer or None")
        if original_task_success is not None and type(original_task_success) is not bool:
            raise ValueError("original_task_success must be a boolean or None")
        audit_supplied = all(value is not None for value in audit.values())
        no_target_writes = (external_target_qpos_writes == 0 and
                            integration_steps == native_samples == self._accepted_samples
                            if audit_supplied else None)
        checks = {
            "native_contiguous_2ms": bool(self._accepted_samples and self._contiguous),
            "held_lift": self._held_seconds + TIME_TOLERANCE_S >= .1,
            "released_stable_3s": self._released_seconds + TIME_TOLERANCE_S >= 3.,
            "all_channels_within_12n": max(self._peaks.values()) <= 12.,
            "no_robot_table_penetration": (self._minimum_table_distance is None
                                            or self._minimum_table_distance >= -.002),
        }
        gates_passed = all(checks.values())
        checks["no_external_target_qpos_writes"] = no_target_writes
        task_success = (bool(original_task_success and gates_passed and no_target_writes is True)
                        if original_task_success is not None and audit_supplied else None)
        return {
            "format": "strict_motion_monitor_r28_v1", "target_privileged_evaluation_only": True,
            "strict_motion_gates_passed": gates_passed,
            "task_success": task_success, "original_task_success": original_task_success, "checks": checks,
            "native_audit_supplied": audit_supplied, "native_audit_assertions": audit,
            "observed_samples": self._observed_samples, "accepted_samples": self._accepted_samples,
            "rejected_timestamps": self._rejected_timestamps,
            "discontinuities": self._discontinuities, "last_timestamp_s": self._previous_time,
            "held_seconds": self._held_seconds, "released_stable_seconds": self._released_seconds,
            "last_held": self._last_held, "last_released_stable": self._last_released,
            "channel_peaks_n": self._peaks.copy(), "phase_peaks_n": self._phase_peaks.copy(),
            "whole_episode_peak_n": max(self._peaks.values()),
            "minimum_robot_table_distance_m": self._minimum_table_distance,
            "thresholds": dict(held_rise_m=.07, held_seconds=.1, contact_normal_minimum_n=.05,
                xy_error_m=.025, height_error_m=.025, released_seconds=3., channel_peak_n=12.,
                linear_speed_m_s=.04, angular_speed_rad_s=.2, hand_clearance_m=.10,
                robot_table_penetration_m=.002, native_timestep_s=TIMESTEP_S),
            "scope": ("Privileged target-state motion predicates only; no arbitrary-clutter safety "
                      "or archive authenticity proof. Strict >70mm rise and >.05N loaded contact; "
                      "strict release-error/speed/clearance inequalities; whole-episode extrema retained."),
        }
