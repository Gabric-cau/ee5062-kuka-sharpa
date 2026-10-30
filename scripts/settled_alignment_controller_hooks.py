"""Next-run-only open-hand alignment settling around the unchanged arm task."""

from __future__ import annotations

import ast
import copy
import hashlib
from types import CodeType

import numpy as np


INHERITED_SOURCE_SHA256 = "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485"
PARAMETERS = {
    "maximum_extra_seconds": 6.,
    "position_tolerance_m": .001,
    "maximum_actual_speed_m_s": .002,
    "continuous_seconds": .25,
    "command_tolerance_rad": 1e-10,
    "open_command_tolerance_rad": 1e-10,
    "orientation_error_is_a_gate": False,
}


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid alignment " + label)
    return result


def open_command_observation(expected, actual):
    """Verify actual output controls, including opposition and zero flexion."""
    valid = bool(expected and set(actual) == set(expected))
    error = None
    if valid:
        try:
            values = [(float(expected[name]), float(actual[name])) for name in expected]
            valid = bool(np.isfinite(values).all())
            if valid:
                error = max(abs(a - b) for a, b in values)
        except (TypeError, ValueError, OverflowError):
            valid = False
    def finite_values(mapping):
        result = {}
        for name, value in mapping.items():
            try:
                number = float(value)
                result[name] = number if np.isfinite(number) else None
            except (TypeError, ValueError, OverflowError):
                result[name] = None
        return result
    return {"expected_command_rad": finite_values(expected), "actual_command_rad": finite_values(actual),
        "maximum_error_rad": error,
        "accepted": bool(valid and error <= PARAMETERS["open_command_tolerance_rad"])}


class AlignmentTracker:
    """Finite-clock readiness from wrist proprioception and applied commands."""

    def __init__(self, timestamp, nominal_seconds, target_position, target_rotation,
            target_command, expected_open):
        if not np.isfinite([timestamp, nominal_seconds]).all() or nominal_seconds <= 0.:
            raise ValueError("Invalid alignment clock")
        self.start_s, self.nominal_seconds = float(timestamp), float(nominal_seconds)
        self.target_position = _array(target_position, (3,), "target position").copy()
        self.target_rotation = _array(target_rotation, (3, 3), "target rotation").copy()
        self.target_command = _array(target_command, (7,), "target command").copy()
        self.expected_open = dict(expected_open)
        if not open_command_observation(expected_open, expected_open)["accepted"]:
            raise ValueError("Invalid expected open-hand command")
        self.previous = None
        self.window_start_s = None
        self.events = []
        self.ready = False
        self.outcome = None

    def sample(self, timestamp, position, rotation, command, open_commands):
        position = _array(position, (3,), "actual position")
        rotation = _array(rotation, (3, 3), "actual rotation")
        command = _array(command, (7,), "actual arm command")
        if (not np.isfinite(timestamp) or timestamp <= self.start_s
                or self.previous is not None and timestamp <= self.previous[0]):
            raise ValueError("Alignment observations must advance the actual clock")
        timestamp = float(timestamp)
        speed = None if self.previous is None else float(np.linalg.norm(position - self.previous[1]) /
            (timestamp - self.previous[0]))
        self.previous = timestamp, position.copy()
        position_error = float(np.linalg.norm(position - self.target_position))
        command_error = float(np.max(np.abs(command - self.target_command)))
        open_proof = open_command_observation(self.expected_open, open_commands)
        orientation_error = float(np.arccos(np.clip(
            (np.trace(self.target_rotation.T @ rotation) - 1.) / 2., -1., 1.)))
        reasons = []
        if not open_proof["accepted"]:
            reasons.append("OPEN_HAND_OUTPUT_REQUIRED")
        if command_error > PARAMETERS["command_tolerance_rad"]:
            reasons.append("FULL_GRASP_COMMAND_NOT_REACHED")
        if position_error > PARAMETERS["position_tolerance_m"]:
            reasons.append("ACTUAL_WRIST_POSITION")
        if speed is None or speed > PARAMETERS["maximum_actual_speed_m_s"]:
            reasons.append("ACTUAL_WRIST_SPEED")
        if reasons:
            self.window_start_s = None
        elif self.window_start_s is None:
            self.window_start_s = timestamp
        continuous = 0. if self.window_start_s is None else timestamp - self.window_start_s
        nominal_finished = timestamp - self.start_s >= self.nominal_seconds - 1e-9
        self.ready = bool(nominal_finished and not reasons
            and continuous >= PARAMETERS["continuous_seconds"] - 1e-9)
        row = {"timestamp_s": timestamp, "elapsed_s": timestamp - self.start_s,
            "nominal_finished": nominal_finished,
            "extra_elapsed_s": max(0., timestamp - self.start_s - self.nominal_seconds),
            "actual_wrist_position_world_m": position.tolist(),
            "target_wrist_position_world_m": self.target_position.tolist(),
            "position_error_m": position_error, "actual_wrist_speed_m_s": speed,
            "orientation_error_rad": orientation_error, "orientation_error_is_a_gate": False,
            "maximum_arm_command_error_rad": command_error,
            "open_hand_output": open_proof, "continuous_seconds": continuous,
            "reasons": reasons, "ready": self.ready,
            "runtime_target_state_used": False}
        self.events.append(row)
        return row

    def finish(self, success, reason):
        self.outcome = {"settled": bool(success), "reason": reason,
            "timestamp_s": None if not self.events else self.events[-1]["timestamp_s"]}

    def as_dict(self):
        return {"start_s": self.start_s, "nominal_seconds": self.nominal_seconds,
            "target_arm_command_rad": self.target_command.tolist(),
            "target_wrist_rotation_world": self.target_rotation.tolist(),
            "expected_open_command_rad": dict(self.expected_open),
            "events": copy.deepcopy(self.events), "outcome": copy.deepcopy(self.outcome)}


class AlignmentTelemetry:
    def __init__(self):
        self.attempts = []

    def begin(self, data, steps, kin, target, hand):
        expected = {name: float(hand.config.thumb_opposition_rad)
            if name == "left_thumb_CMC_FE" else 0.
            for name in hand.actuators if name.startswith("left_")}
        position = kin.position(target)
        rotation = kin.scratch.xmat[kin.body].reshape(3, 3).copy()
        tracker = AlignmentTracker(float(data.time), steps * float(kin.model.opt.timestep),
            position, rotation, target, expected)
        self.attempts.append(tracker)
        return tracker

    def as_dict(self):
        return {"format": "settled_open_alignment_policy_v1", "parameters": dict(PARAMETERS),
            "attempts": [item.as_dict() for item in self.attempts],
            "wrist_measurement_time_semantics": "Post-mj_step observation clock with inherited cached xpos/xmat; no additional live forward evaluation",
            "speed_estimator": "Finite difference of consecutive cached wrist positions, not independently recomputed postintegration Cartesian velocity",
            "arm_command_semantics": "Post-limiter kinematic joint command; raw arm motor ctrl includes the unchanged gravity compensation term",
            "scope": "Applied open-hand controls and wrist proprioception only; no object-state input, force equilibrium, probe confirmation or task-success claim."}


def _open_outputs(hand, data):
    return {name: float(data.ctrl[aid]) for name, aid in hand.actuators.items()
        if name.startswith("left_")}


def _verify_open(tracker, hand, data, failure):
    proof = open_command_observation(tracker.expected_open, _open_outputs(hand, data))
    if not proof["accepted"]:
        tracker.finish(False, "ALIGN_HAND_NOT_OPEN")
        tracker.outcome["timestamp_s"] = float(data.time)
        tracker.outcome["open_hand_output"] = proof
        raise failure("ALIGN_HAND_NOT_OPEN")


def _sample_alignment(tracker, hand, kin, data, command, failure):
    try:
        return tracker.sample(float(data.time), data.xpos[kin.body].copy(),
            data.xmat[kin.body].reshape(3, 3).copy(), command, _open_outputs(hand, data))
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        tracker.finish(False, "ALIGN_SETTLE_INVALID_MEASUREMENT")
        raise failure("ALIGN_SETTLE_INVALID_MEASUREMENT") from error


def _code_semantics(code):
    return (code.co_code, tuple(_code_semantics(value) if isinstance(value, CodeType) else value
        for value in code.co_consts), code.co_names, code.co_varnames,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_flags & 0xffff, code.co_freevars, code.co_cellvars)


def _one(nodes, predicate, label):
    matches = [node for node in nodes if predicate(node)]
    if len(matches) != 1:
        raise ValueError("Expected one inherited alignment anchor: " + label)
    return matches[0]


def transform(original, source):
    """Wrap a pinned factory result without changing inherited hand bindings."""
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    if source_sha != INHERITED_SOURCE_SHA256:
        raise ValueError("Inherited arm source SHA256 differs from the declared R19-R21 arm")
    baseline_namespace = {}
    exec(compile(source, "<checked_inherited_alignment_arm>", "exec"), baseline_namespace)
    if _code_semantics(original.__code__) != _code_semantics(baseline_namespace["run_task"].__code__):
        raise ValueError("Loaded arm callable differs from its pinned readable source")
    tree = ast.parse(source)
    task = _one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "run_task", "task")
    move = _one(task.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "move", "move")
    loop = _one(move.body, lambda n: isinstance(n, ast.For), "native loop")
    if ast.unparse(loop.iter) != "range(steps)":
        raise ValueError("Inherited native loop clock differs")
    setup = ast.parse("""
_r22_alignment = None
_r22_extra_steps = 0
if state == State.ALIGN:
    _r22_alignment = _r22_alignment_telemetry.begin(data, steps, kin, target, hand)
    _r22_extra_steps = round(_r22_alignment_parameters['maximum_extra_seconds'] / model.opt.timestep)
""").body
    move.body[move.body.index(loop):move.body.index(loop)] = setup
    loop.iter = ast.parse("range(steps + _r22_extra_steps)", mode="eval").body
    first_arm = _one(loop.body, lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "cartesian_motion and step % segment_steps == 0", "Cartesian segment")
    arm_index = loop.body.index(first_arm)
    second_arm = loop.body[arm_index + 1]
    if not isinstance(second_arm, ast.If) or ast.unparse(second_arm.test) != "cartesian_motion":
        raise ValueError("Inherited Cartesian command branch differs")
    arm_branch = ast.parse("""
if state == State.ALIGN and step >= steps:
    desired_command = target.copy()
else:
    pass
""").body[0]
    arm_branch.orelse = [first_arm, second_arm]
    loop.body[arm_index:arm_index + 2] = [arm_branch]
    alpha = _one(ast.walk(loop), lambda n: isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "alpha"
        and ast.unparse(n.value) == "(step + 1) / steps", "hand blend")
    alpha.value = ast.parse("1.0 if state == State.ALIGN and step >= steps else (step + 1) / steps",
        mode="eval").body
    write = _one(loop.body, lambda n: isinstance(n, ast.Expr)
        and ast.unparse(n) == "hand.write(data)", "hand output")
    loop.body[loop.body.index(write) + 1:loop.body.index(write) + 1] = ast.parse("""
if state == State.ALIGN:
    _r22_verify_open(_r22_alignment, hand, data, MotionFailure)
""").body
    final_gate = _one(loop.body, lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "state == State.CLOSE and hand.preload_ready", "original close termination")
    loop.body[loop.body.index(final_gate):loop.body.index(final_gate)] = ast.parse("""
if state == State.ALIGN:
    _r22_row = _r22_sample_alignment(_r22_alignment, hand, kin, data, command, MotionFailure)
    trace[-1]['alignment_settling'] = _r22_row.copy()
    if _r22_row['ready']:
        break
""").body
    move.body[move.body.index(loop) + 1:move.body.index(loop) + 1] = ast.parse("""
if state == State.ALIGN:
    if not _r22_alignment.ready:
        _r22_alignment.finish(False, 'ALIGN_SETTLE_TIMEOUT')
        raise MotionFailure('ALIGN_SETTLE_TIMEOUT')
    _r22_alignment.finish(True, 'OPEN_FULL_GRASP_COMMAND_AND_ACTUAL_WRIST_SETTLED')
""").body
    tree = ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree) + "\n"
    telemetry = AlignmentTelemetry()
    namespace = dict(original.__globals__, _r22_alignment_telemetry=telemetry,
        _r22_alignment_parameters=dict(PARAMETERS), _r22_verify_open=_verify_open,
        _r22_sample_alignment=_sample_alignment)
    exec(compile(rendered, "<settled_open_alignment_arm>", "exec"), namespace)
    wrapped = namespace["run_task"]
    wrapped.__dict__.update(original.__dict__)
    wrapped.r22_alignment_transform = {"format": "settled_open_alignment_ast_extension_v1",
        "inherited_source_sha256": source_sha,
        "transformed_source_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "parameters": dict(PARAMETERS), "close_motion_source_unchanged": True,
        "all_original_native_safety_checks_retained": True,
        "target_qpos_writes_added": False}
    return wrapped, rendered, telemetry
