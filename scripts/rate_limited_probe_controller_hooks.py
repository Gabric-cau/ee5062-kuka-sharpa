"""Next-run-only reference governor; the pinned R24 arm source stays unchanged."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

import actual_wrist_probe_controller_hooks as inherited
from settled_alignment_controller_hooks import _code_semantics


INHERITED_SOURCE_SHA256 = "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"
INHERITED_HOOK_SHA256 = "a60164c9c2538693ed89dc5855dad44944b85ba0ee83b60396f838b368aeb39e"
PARAMETERS = {"filter_time_constant_s": .10, "maximum_target_speed_m_s": .004,
    "reference_interval_s": .02, "clock_tolerance_s": 1e-7,
    "filter_discretization": "exact_first_order_zero_order_hold",
    "initialization": "admission_actual_wrist_once"}


class RateLimitedProbeTracker(inherited.ProbeTracker):
    """Filter the original raw reference, then limit its vector displacement."""

    def __init__(self, timestamp, admission_timestamp, admission_position):
        super().__init__(timestamp, admission_timestamp, admission_position)
        self.filtered_target = self.start_position.copy()
        self.applied_target = self.start_position.copy()
        self.previous_endpoint_s = 0.
        self.previous_ik_actual = self.start_position.copy()
        self.previous_ik_clock_s = self.start_s

    def target(self, data, kin, elapsed_s, failure):
        try:
            endpoint = float(elapsed_s)
            clock = float(data.time)
            dt = endpoint - self.previous_endpoint_s
            interval = PARAMETERS["reference_interval_s"]
            tolerance = PARAMETERS["clock_tolerance_s"]
            if (not np.isfinite([endpoint, clock, dt]).all()
                    or abs(dt - interval) > tolerance
                    or endpoint > inherited.PARAMETERS["maximum_probe_seconds"] + tolerance
                    or abs(clock - (self.start_s + endpoint - interval)) > tolerance):
                raise ValueError("Reference endpoint and current native clocks differ")
        except (TypeError, ValueError, OverflowError) as error:
            raise failure("RATE_LIMITED_PROBE_INVALID_REFERENCE_CLOCK") from error

        raw = super().target(data, kin, endpoint, failure)
        previous_filtered, previous_applied = self.filtered_target.copy(), self.applied_target.copy()
        alpha = float(-np.expm1(-dt / PARAMETERS["filter_time_constant_s"]))
        filtered = previous_filtered + alpha * (raw - previous_filtered)
        requested = filtered - previous_applied
        distance = float(np.linalg.norm(requested))
        maximum_distance = PARAMETERS["maximum_target_speed_m_s"] * dt
        delta = requested * (maximum_distance / distance) if distance > maximum_distance else requested
        applied = previous_applied + delta
        row = self.ik_samples[-1]
        actual = np.asarray(row["actual_wrist_at_ik_sample_world_m"], dtype=float)
        actual_dt = clock - self.previous_ik_clock_s
        row.update(raw_cartesian_ik_target_world_m=raw.tolist(),
            filtered_cartesian_ik_target_world_m=filtered.tolist(),
            applied_cartesian_ik_target_world_m=applied.tolist(),
            cartesian_ik_target_world_m=applied.tolist(),
            reference_endpoint_elapsed_s=endpoint, reference_dt_s=dt, filter_alpha=alpha,
            filtered_target_delta_world_m=(filtered - previous_filtered).tolist(),
            applied_target_delta_world_m=delta.tolist(),
            applied_target_speed_m_s=float(np.linalg.norm(delta) / dt),
            actual_wrist_delta_since_previous_ik_world_m=(actual - self.previous_ik_actual).tolist(),
            actual_wrist_sample_dt_s=actual_dt,
            actual_wrist_speed_since_previous_ik_m_s=None if actual_dt == 0. else
                float(np.linalg.norm(actual - self.previous_ik_actual) / actual_dt),
            reference_governor_parameters=dict(PARAMETERS))
        self.filtered_target, self.applied_target = filtered, applied
        self.previous_endpoint_s = endpoint
        self.previous_ik_actual, self.previous_ik_clock_s = actual.copy(), clock
        return applied.copy()

    def sample(self, data, kin, hand, failure):
        super().sample(data, kin, hand, failure)
        row = self.samples[-1]
        latest = self.ik_samples[-1] if self.ik_samples else None
        row["reference_governor"] = None if latest is None else {
            name: copy.deepcopy(latest[name]) for name in (
                "raw_cartesian_ik_target_world_m", "filtered_cartesian_ik_target_world_m",
                "applied_cartesian_ik_target_world_m", "reference_endpoint_elapsed_s",
                "reference_dt_s", "applied_target_delta_world_m", "applied_target_speed_m_s")}
        return copy.deepcopy(row)


class RateLimitedProbeTelemetry(inherited.ProbeTelemetry):
    def begin(self, data, hand, cartesian_motion, failure):
        try:
            if not cartesian_motion:
                raise ValueError("An admitted governed probe requires Cartesian IK")
            tracker = RateLimitedProbeTracker(float(data.time), hand.probe_start_s,
                hand.probe_start_world_m)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise failure("ACTUAL_WRIST_PROBE_INVALID_START") from error
        self.attempts.append(tracker)
        return tracker

    def as_dict(self):
        result = super().as_dict()
        result.update(format="rate_limited_actual_wrist_probe_policy_v1",
            inherited_format="actual_wrist_probe_policy_v1", reference_governor=dict(PARAMETERS),
            cartesian_ik_target_semantics="Applied governed target; original raw R12 target is recorded separately",
            scope="Only active unconfirmed probe IK reference is filtered and rate limited. "
                "Original hand, confirmation, actual-speed abort and strict task gates remain unchanged.")
        return result


def transform(original, source):
    """Clone the pinned callable and replace only its private telemetry global."""
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    if source_sha != INHERITED_SOURCE_SHA256:
        raise ValueError("Inherited arm source SHA256 differs from executed R24")
    checked = {}
    exec(compile(source, "<checked_rate_limited_probe_arm>", "exec"), checked)
    if _code_semantics(original.__code__) != _code_semantics(checked["run_task"].__code__):
        raise ValueError("Loaded arm callable differs from its pinned readable source")
    if hashlib.sha256(Path(inherited.__file__).read_bytes()).hexdigest() != INHERITED_HOOK_SHA256:
        raise ValueError("Inherited actual-wrist probe hook source differs")
    inherited._check_reference()
    old_telemetry = original.__globals__.get("_r23_probe_telemetry")
    if (type(old_telemetry) is not inherited.ProbeTelemetry
            or old_telemetry.attempts
            or original.__globals__.get("_r23_probe_parameters") != inherited.PARAMETERS
            or "r25_rate_limited_probe_transform" in original.__dict__):
        raise ValueError("Expected an unused original R23 probe telemetry binding")
    telemetry = RateLimitedProbeTelemetry()
    namespace = dict(original.__globals__, _r23_probe_telemetry=telemetry)
    wrapped = FunctionType(original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__)
    namespace[original.__name__] = wrapped
    wrapped.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    wrapped.__annotations__ = dict(original.__annotations__)
    wrapped.__dict__.update(original.__dict__)
    wrapped.actual_wrist_probe_telemetry = telemetry
    wrapped.rate_limited_probe_telemetry = telemetry
    wrapped.r25_rate_limited_probe_transform = {
        "format": "r25_rate_limited_probe_global_binding_v1",
        "inherited_source_sha256": source_sha, "readable_source_sha256": source_sha,
        "readable_source_unchanged": True, "compiled_arm_code_unchanged": True,
        "replacement_global": "_r23_probe_telemetry",
        "inherited_telemetry_class": inherited.ProbeTelemetry.__name__,
        "replacement_telemetry_class": type(telemetry).__name__,
        "replacement_tracker_class": RateLimitedProbeTracker.__name__,
        "inherited_probe_hook_sha256": INHERITED_HOOK_SHA256,
        "raw_reference_source_sha256": inherited.REFERENCE_SOURCE_SHA256,
        "raw_reference_parameters": dict(inherited.PARAMETERS),
        "reference_governor": dict(PARAMETERS),
        "original_decisive_confirmation_called": True,
        "all_original_native_safety_checks_retained": True,
        "runtime_target_state_used": False, "target_qpos_writes_added": False}
    return wrapped, source, telemetry
