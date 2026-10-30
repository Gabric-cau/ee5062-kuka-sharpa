"""Next-run-only actual-wrist probe motion around the frozen R22 arm and hand."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np

import develop_public_ycb_cartesian_probe_r12 as reference
from settled_alignment_controller_hooks import _code_semantics


INHERITED_SOURCE_SHA256 = "206db2d2b6f08dedc1757ba9ec9445957f96bf6fab36fbba139a670acba0d1eb"
REFERENCE_SOURCE_SHA256 = "8729e049bae598a397933d69e8bf40f1e74534025c81c297effdfe04ec454ddf"
bounded_probe_reference = reference.bounded_probe_reference
PARAMETERS = {
    "path_seconds": 3.,
    "settle_seconds": 3.,
    "maximum_probe_seconds": 6.,
    "distance_m": .005,
    "actual_error_gain": 2.,
    "reference_correction_limit_m": .002,
    "cartesian_ik_sample_seconds": .02,
}


def _position(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Probe wrist position must be a finite three-vector")
    return result


def _finite_number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


class ProbeTracker:
    """Observe one admitted probe without changing its hand or confirmation state."""

    def __init__(self, timestamp, admission_timestamp, admission_position):
        if (not np.isfinite([timestamp, admission_timestamp]).all()
                or abs(timestamp - admission_timestamp) > 1e-9):
            raise ValueError("Actual-wrist probe move must begin at its recorded admission clock")
        self.start_s = float(admission_timestamp)
        self.start_position = _position(admission_position).copy()
        self.ik_samples, self.samples = [], []
        self.outcome = None

    def target(self, data, kin, elapsed_s, failure):
        try:
            timestamp = float(data.time)
            if (not np.isfinite(timestamp) or timestamp < self.start_s - 1e-9
                    or self.ik_samples and timestamp <= self.ik_samples[-1]["timestamp_s"]
                    or not 0. <= elapsed_s <= PARAMETERS["maximum_probe_seconds"] + 1e-9):
                raise ValueError("Invalid actual-wrist IK sample clock")
            row = bounded_probe_reference(self.start_position,
                _position(data.xpos[kin.body]).copy(), float(elapsed_s))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise failure("ACTUAL_WRIST_PROBE_INVALID_REFERENCE") from error
        row.update(timestamp_s=timestamp, actual_elapsed_s=timestamp - self.start_s,
            parameters=dict(PARAMETERS),
            time_semantics="Current cached wrist sampled for the next 20ms IK segment endpoint")
        self.ik_samples.append(row)
        return np.asarray(row["cartesian_ik_target_world_m"])

    def sample(self, data, kin, hand, failure):
        try:
            timestamp = float(data.time)
            position = _position(data.xpos[kin.body]).copy()
            if (not np.isfinite(timestamp) or timestamp <= self.start_s
                    or timestamp - self.start_s > PARAMETERS["maximum_probe_seconds"] + 1e-9
                    or self.samples and timestamp <= self.samples[-1]["timestamp_s"]):
                raise ValueError("Invalid actual-wrist native sample clock")
            confirmed_seconds = float(hand.probe_confirmed_seconds)
            if not np.isfinite(confirmed_seconds) or confirmed_seconds < 0.:
                raise ValueError("Invalid inherited confirmation clock")
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise failure("ACTUAL_WRIST_PROBE_INVALID_MEASUREMENT") from error
        latest = None if not self.ik_samples else self.ik_samples[-1]
        row = {"timestamp_s": timestamp, "elapsed_s": timestamp - self.start_s,
            "parameters": dict(PARAMETERS),
            "actual_admission_start_world_m": self.start_position.tolist(),
            "current_actual_wrist_world_m": position.tolist(),
            "currently_probe_active": bool(hand.probe_active),
            "inherited_confirmed_seconds": confirmed_seconds,
            "latest_ik_timestamp_s": None if latest is None else latest["timestamp_s"],
            "ik_sample_age_s": None if latest is None else timestamp - latest["timestamp_s"],
            "desired_actual_path_world_m": None if latest is None else list(latest["desired_actual_path_world_m"]),
            "bounded_reference_correction_world_m": None if latest is None else list(latest["bounded_reference_correction_world_m"]),
            "cartesian_ik_target_world_m": None if latest is None else list(latest["cartesian_ik_target_world_m"]),
            "runtime_target_state_used": False}
        self.samples.append(row)
        return copy.deepcopy(row)

    def finish(self, data, hand, reason, *, motion_error=None):
        previous = {} if self.outcome is None else self.outcome
        timestamp = _finite_number(data.time)
        self.outcome = {**previous, "timestamp_s": timestamp,
            "elapsed_s": None if timestamp is None else timestamp - self.start_s,
            "arm_exit_reason": previous.get("arm_exit_reason", reason),
            "final_reason": str(motion_error) if motion_error is not None else reason,
            "motion_failure": None if motion_error is None else str(motion_error),
            "probe_active_at_motion_end": bool(hand.probe_active),
            "inherited_confirmed_seconds": _finite_number(hand.probe_confirmed_seconds),
            "original_confirmation_attempted": False,
            "original_confirmation_accepted": None}

    def as_dict(self):
        return {"start_s": self.start_s,
            "actual_admission_start_world_m": self.start_position.tolist(),
            "ik_samples": copy.deepcopy(self.ik_samples),
            "samples": copy.deepcopy(self.samples), "outcome": copy.deepcopy(self.outcome)}


class ProbeTelemetry:
    def __init__(self):
        self.attempts = []

    def begin(self, data, hand, cartesian_motion, failure):
        try:
            if not cartesian_motion:
                raise ValueError("An admitted actual-wrist probe requires Cartesian IK")
            tracker = ProbeTracker(float(data.time), hand.probe_start_s, hand.probe_start_world_m)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise failure("ACTUAL_WRIST_PROBE_INVALID_START") from error
        self.attempts.append(tracker)
        return tracker

    def confirm_probe(self, hand, data):
        """Log the original decisive check; keep its return and state transitions."""
        tracker = self.attempts[-1] if self.attempts else None
        if tracker is not None and abs(tracker.start_s - hand.probe_start_s) > 1e-9:
            raise ValueError("Decisive probe confirmation does not match the tracked admission")
        try:
            accepted = hand.confirm_probe(data)
        except Exception as error:
            if tracker is not None:
                tracker.finish(data, hand, "DECISIVE_CONFIRMATION_EXCEPTION", motion_error=error)
                tracker.outcome["original_confirmation_attempted"] = True
            raise
        if tracker is not None:
            tracker.outcome.update(original_confirmation_attempted=True,
                original_confirmation_accepted=bool(accepted),
                confirmation_timestamp_s=float(data.time),
                final_reason="ORIGINAL_DECISIVE_CONFIRMATION_ACCEPTED" if accepted else
                    "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED")
        return accepted

    def as_dict(self):
        return {"format": "actual_wrist_probe_policy_v1", "parameters": dict(PARAMETERS),
            "attempts": [item.as_dict() for item in self.attempts],
            "reference_source_sha256": REFERENCE_SOURCE_SHA256,
            "wrist_measurement_time_semantics": "Inherited cached xpos observed at the native clock; no additional live forward evaluation",
            "ik_time_semantics": "end_step * timestep next-segment lookahead, matching the original R12 reference",
            "scope": "Active unconfirmed Cartesian probe only; wrist proprioception controls the reference. Original hand, confirmation, safety and strict success remain unchanged."}


def _one(nodes, predicate, label):
    matches = [node for node in nodes if predicate(node)]
    if len(matches) != 1:
        raise ValueError("Expected one inherited actual-wrist probe anchor: " + label)
    return matches[0]


def _check_reference():
    path = Path(inspect.getsourcefile(bounded_probe_reference)).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SOURCE_SHA256:
        raise ValueError("Frozen R12 bounded reference source SHA256 differs")
    if bounded_probe_reference is not reference.bounded_probe_reference or any(
            reference.CARTESIAN_PROBE_PARAMETERS[key] != value for key, value in PARAMETERS.items()):
        raise ValueError("Loaded R12 bounded reference or its parameters differ")


def transform(original, source):
    """Extend a pinned R22 factory result; retain all hand/global/metadata bindings."""
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    if source_sha != INHERITED_SOURCE_SHA256:
        raise ValueError("Inherited arm source SHA256 differs from the executed R22 arm")
    checked = {}
    exec(compile(source, "<checked_actual_wrist_probe_arm>", "exec"), checked)
    if _code_semantics(original.__code__) != _code_semantics(checked["run_task"].__code__):
        raise ValueError("Loaded arm callable differs from its pinned R22 readable source")
    _check_reference()
    tree = ast.parse(source)
    task = _one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "run_task", "task")
    move = _one(task.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "move", "move")
    loop = _one(move.body, lambda n: isinstance(n, ast.For), "native loop")
    if ast.unparse(loop.iter) != "range(steps + _r22_extra_steps)":
        raise ValueError("Inherited R22 native loop clock differs")
    duration = _one(move.body, lambda n: isinstance(n, ast.Assign)
        and len(n.targets) == 1 and ast.unparse(n.targets[0]) == "duration", "duration")
    setup = ast.parse("""
_r23_probe = None
if state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed:
    _r23_probe = _r23_probe_telemetry.begin(data, hand, cartesian_motion, MotionFailure)
    duration = _r23_probe_parameters['maximum_probe_seconds']
""").body
    move.body[move.body.index(duration) + 1:move.body.index(duration) + 1] = setup
    segment = _one(ast.walk(loop), lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "cartesian_motion and step % segment_steps == 0", "Cartesian segment")
    position = _one(segment.body, lambda n: isinstance(n, ast.Assign)
        and len(n.targets) == 1 and ast.unparse(n.targets[0]) == "next_position", "Cartesian target")
    branch = ast.parse("""
if _r23_probe is not None and state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed:
    next_position = _r23_probe.target(data, kin, end_step * model.opt.timestep, MotionFailure)
else:
    pass
""").body[0]
    branch.orelse = [position]
    segment.body[segment.body.index(position)] = branch
    recording = _one(loop.body, lambda n: isinstance(n, ast.Assign)
        and len(n.targets) == 1 and ast.unparse(n.targets[0]) == "trace[-1]['probe_policy']", "probe trace")
    loop.body[loop.body.index(recording) + 1:loop.body.index(recording) + 1] = ast.parse("""
if _r23_probe is not None:
    trace[-1]['actual_wrist_probe'] = _r23_probe.sample(data, kin, hand, MotionFailure)
""").body
    loop.body.extend(ast.parse("""
if _r23_probe is not None and state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed and hand.probe_confirmed_seconds >= _probe_parameters['confirmation_seconds']:
    _r23_probe.finish(data, hand, 'ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED')
    trace[-1]['actual_wrist_probe_termination'] = 'ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED'
    break
""").body)
    tail_start = move.body.index(setup[-1]) + 1
    inherited_tail = move.body[tail_start:]
    end = ast.parse("""
if _r23_probe is not None and _r23_probe.outcome is None:
    _r23_probe.finish(data, hand, 'MAXIMUM_PROBE_DURATION_REACHED')
""").body
    end_index = inherited_tail.index(loop) + 1
    inherited_tail[end_index:end_index] = end
    guard = ast.parse("""
try:
    pass
except Exception as _r23_motion_error:
    if _r23_probe is not None:
        _r23_probe.finish(data, hand, 'MOTION_EXCEPTION', motion_error=_r23_motion_error)
    raise
""").body[0]
    guard.body = inherited_tail
    move.body[tail_start:] = [guard]
    confirmation = _one(ast.walk(task), lambda n: isinstance(n, ast.Call)
        and ast.unparse(n) == "hand.confirm_probe(data)", "original decisive confirmation")
    confirmation.func = ast.parse("_r23_probe_telemetry.confirm_probe", mode="eval").body
    confirmation.args.insert(0, ast.Name(id="hand", ctx=ast.Load()))
    tree = ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree) + "\n"
    telemetry = ProbeTelemetry()
    additions = {"_r23_probe_telemetry": telemetry, "_r23_probe_parameters": dict(PARAMETERS)}
    if additions.keys() & original.__globals__.keys():
        raise ValueError("Actual-wrist probe globals would replace an inherited binding")
    namespace = dict(original.__globals__, **additions)
    exec(compile(rendered, "<actual_wrist_probe_arm>", "exec"), namespace)
    wrapped = namespace["run_task"]
    wrapped.__dict__.update(original.__dict__)
    wrapped.r23_actual_wrist_probe_transform = {
        "format": "actual_wrist_probe_ast_extension_v1",
        "inherited_source_sha256": source_sha,
        "transformed_source_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "reference_source_sha256": REFERENCE_SOURCE_SHA256, "parameters": dict(PARAMETERS),
        "settled_alignment_source_retained": True, "close_motion_source_unchanged": True,
        "original_decisive_confirmation_called": True,
        "all_original_native_safety_checks_retained": True,
        "runtime_target_state_used": False, "target_qpos_writes_added": False}
    return wrapped, rendered, telemetry
