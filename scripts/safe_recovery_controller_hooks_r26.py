"""Recovery ordering over the frozen R25 probe controller.

The failed grasp stays failed. This extension brakes, lowers to instrumented
support, and opens at a stationary wrist before the inherited retry.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

import numpy as np

import rate_limited_probe_controller_hooks as inherited


INHERITED_SOURCE_SHA256 = inherited.INHERITED_SOURCE_SHA256
INHERITED_HOOK_SHA256 = "27691d998994be32b640d1155f01fa3152342f0c81441b4aa16b21450835616d"
PARAMETERS = {
    "brake_timeout_s": 1.5, "settling_window_s": .15,
    "settling_speed_limit_m_s": .005, "lower_speed_m_s": .004,
    "maximum_lower_distance_m": .05, "lower_timeout_s": 13.,
    "force_abort_limit_n": 12., "hold_hand_phase": "R26_RECOVERY_HOLD",
    "sequence": "brake_then_lower_to_original_table_support_then_stationary_release",
    "scope": "Failure recovery only; 12N is an abort threshold, not a post-step force guarantee",
}


class RecoveryTelemetry:
    def __init__(self):
        self.attempts = []
        self.current = None

    def begin(self, mode, data, kin, hand, support_gate, failure):
        if mode not in ("BRAKE", "LOWER", "RELEASE"):
            raise failure("R26_INVALID_RECOVERY_MODE")
        if float(hand.model.opt.timestep) != .002:
            raise failure("R26_INVALID_NATIVE_TIMESTEP")
        if mode == "RELEASE" and (support_gate is None or not support_gate.fresh_confirmation(float(data.time))):
            raise failure("R26_RELEASE_REQUIRES_ORIGINAL_TABLE_SUPPORT")
        # Avoid the inherited RECOVER reset, which begins opening immediately.
        # Captured joint references and added-bias limits are left intact.
        if mode != "RELEASE":
            hand.phase_begin(PARAMETERS["hold_hand_phase"], data)
            hand.inhibit_admission()
        self.current = {"mode": mode, "start_s": float(data.time),
            "start_wrist_world_m": data.xpos[kin.body].copy(),
            "previous_wrist_world_m": data.xpos[kin.body].copy(),
            "previous_s": float(data.time), "settled_seconds": 0., "samples": [],
            "ready": False, "finished": False, "target_world_m": data.xpos[kin.body].copy(),
            "initial_hand_references_rad": copy.deepcopy(hand.joint_references)}
        self.attempts.append(self.current)

    def target(self, data, kin, command, failure):
        item = self.current
        endpoint = item["start_wrist_world_m"].copy()
        if item["mode"] == "LOWER":
            distance = min(PARAMETERS["maximum_lower_distance_m"],
                PARAMETERS["lower_speed_m_s"] * max(0., float(data.time) - item["start_s"]))
            endpoint[2] -= distance
        solved = kin.solve(endpoint - kin.origin, seed=command, tolerance=.0001)
        if solved is None:
            raise failure("R26_RECOVERY_TARGET_IK")
        item["target_world_m"] = endpoint
        return solved

    def sample(self, data, kin, hand, support_gate, failure):
        item = self.current
        now = float(data.time)
        position = data.xpos[kin.body].copy()
        dt = now - item["previous_s"]
        if abs(dt - .002) > 1e-7:
            raise failure("R26_NONCONTIGUOUS_NATIVE_SAMPLE")
        speed = float(np.linalg.norm(position - item["previous_wrist_world_m"]) / dt)
        pressures = {name: float(value.normal) for name, value in hand.reader.read(data).items()}
        support = bool(support_gate is not None and support_gate.fresh_confirmation(now))
        item["settled_seconds"] = item["settled_seconds"] + dt if speed <= PARAMETERS["settling_speed_limit_m_s"] else 0.
        settled = item["settled_seconds"] + 1e-9 >= PARAMETERS["settling_window_s"]
        item["ready"] = settled and (item["mode"] == "BRAKE" or support)
        row = {"timestamp_s": now, "mode": item["mode"], "actual_wrist_world_m": position.tolist(),
            "actual_wrist_speed_m_s": speed, "settled_seconds": item["settled_seconds"],
            "channel_pressures_n": pressures, "original_table_support_confirmed": support,
            "ready": item["ready"], "hand_phase": hand.phase,
            "scope": "Recovery telemetry only; no task-success certificate"}
        item["samples"].append(row)
        item["previous_s"], item["previous_wrist_world_m"] = now, position
        if not np.isfinite([speed, *pressures.values()]).all():
            raise failure("R26_NONFINITE_RECOVERY_OBSERVATION")
        if max(pressures.values(), default=0.) > PARAMETERS["force_abort_limit_n"]:
            raise failure("R26_HAND_FORCE_LIMIT")
        if item["mode"] == "RELEASE" and not support:
            raise failure("R26_TABLE_SUPPORT_LOST_DURING_RELEASE")
        return copy.deepcopy(row)

    def finish(self, failure):
        item = self.current
        if item is None:
            raise failure("R26_MISSING_RECOVERY_SESSION")
        if item["mode"] != "RELEASE" and not item["ready"]:
            reason = "WRIST_NOT_SETTLED" if item["mode"] == "BRAKE" else "TABLE_SUPPORT_NOT_CONFIRMED"
            raise failure("R26_" + reason)
        item["finished"] = True

    def as_dict(self):
        return {"format": "r26_brake_supported_lower_release_telemetry_v1", "parameters": dict(PARAMETERS),
            "attempts": [{key: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                for key, value in item.items() if key not in ("previous_wrist_world_m", "previous_s")}
                for item in self.attempts]}


R26_RECOVERY = '''
try:
    hand.abort_probe(controller.failure or 'R26_RECOVERY_ENTERED')
    move(State.RECOVER, command.copy(), _r26_parameters['brake_timeout_s'], 0., recovery_mode='BRAKE')
    if support_gate is None or not support_gate.fresh_confirmation(float(data.time)):
        move(State.RECOVER, command.copy(), _r26_parameters['lower_timeout_s'], 0., recovery_mode='LOWER')
    move(State.RECOVER, command.copy(), _r26_parameters['brake_timeout_s'], 0., recovery_mode='BRAKE')
    move(State.RELEASE, command.copy(), 4., 0., recovery_mode='RELEASE')
    if max(value.normal for value in hand_reader.read(data).values()) > 1.:
        raise MotionFailure('R26_RELEASE_CONTACT_NOT_CLEARED')
    retreat_position = kin.position(command) - kin.approach_axis * .13 + [0., 0., .10]
    safe = kin.solve(retreat_position - kin.origin, seed=command)
    if safe is None:
        raise MotionFailure('RECOVERY_IK')
    move(State.RECOVER, safe, 2., 0.)
    safe_height = .34 if physical_v6 and scenario.object_kind == 'ycb_cracker_box' else .18
    move(State.RECOVER, kin.solve(np.array([0., 0., safe_height])), 2., 0.)
except MotionFailure as error:
    controller.fail(data.time, 'RECOVERY_' + str(error))
    controller.enter(State.ABORT, data.time, controller.failure)
    break
'''


def _one(nodes, predicate, description):
    found = [node for node in nodes if predicate(node)]
    if len(found) != 1:
        raise ValueError("R26 expected exactly one " + description)
    return found[0]


def transform(original, source):
    """Return the actually compiled R26 source, never label R24 text as R26."""
    if hashlib.sha256(source.encode()).hexdigest() != INHERITED_SOURCE_SHA256:
        raise ValueError("Inherited arm source SHA256 differs from executed R24")
    if hashlib.sha256(Path(inherited.__file__).read_bytes()).hexdigest() != INHERITED_HOOK_SHA256:
        raise ValueError("Inherited R25 hook source changed")
    base, _, probe_telemetry = inherited.transform(original, source)
    tree = ast.parse(source)
    task = tree.body[0]
    move = _one(ast.walk(task), lambda n: isinstance(n, ast.FunctionDef) and n.name == "move", "move function")
    move.args.args.append(ast.arg(arg="recovery_mode"))
    move.args.defaults.append(ast.Constant(value=None))
    duration = _one(move.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "duration", "duration clamp")
    move.body[move.body.index(duration)] = ast.If(test=ast.parse("recovery_mode is None", mode="eval").body,
        body=[duration], orelse=[])
    trial = _one(move.body, lambda n: isinstance(n, ast.Try), "motion try")
    begin = _one(trial.body, lambda n: isinstance(n, ast.Expr) and ast.unparse(n).startswith("_contact_guard.begin("), "guard begin")
    index = trial.body.index(begin)
    trial.body[index:index + 1] = ast.parse('''
if recovery_mode is None or recovery_mode == 'RELEASE':
    _contact_guard.begin(state, data)
if recovery_mode is not None:
    _r26_recovery.begin(recovery_mode, data, kin, hand, support_gate, MotionFailure)
''').body
    loop = _one(ast.walk(move), lambda n: isinstance(n, ast.For) and ast.unparse(n.target) == "step", "native step loop")
    visual = loop.body[0]
    visual.test = ast.BoolOp(op=ast.And(), values=[ast.BoolOp(op=ast.Or(), values=[
        ast.parse("recovery_mode is not None", mode="eval").body, visual.test.values[0]]), visual.test.values[1]])
    reference = _one(loop.body, lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "state == State.ALIGN and step >= steps", "original Cartesian reference branch")
    loop.body[loop.body.index(reference)] = ast.If(
        test=ast.parse("recovery_mode is not None", mode="eval").body,
        body=ast.parse("desired_command = _r26_recovery.target(data, kin, command, MotionFailure)").body,
        orelse=[reference])
    guide = _one(loop.body, lambda n: isinstance(n, ast.Assign) and "_contact_guard.guide" in ast.unparse(n.value), "contact guide")
    loop.body[loop.body.index(guide)] = ast.parse('''
if recovery_mode is None:
    desired_command = _contact_guard.guide(state, desired_command, command, kin, data, MotionFailure)
else:
    _contact_guard.read(data)
''').body[0]
    hand_branch = _one(loop.body, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "state == State.CLOSE", "hand motion branch")
    loop.body[loop.body.index(hand_branch)] = ast.If(
        test=ast.parse("recovery_mode in ('BRAKE', 'LOWER')", mode="eval").body,
        body=ast.parse("hand.phase_begin(_r26_parameters['hold_hand_phase'], data)\nhand.inhibit_admission()").body,
        orelse=[hand_branch])
    record = _one(loop.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n.value) == "record()", "native record")
    loop.body.insert(loop.body.index(record) + 1, ast.parse('''
if recovery_mode is not None:
    trace[-1]['safe_recovery'] = _r26_recovery.sample(data, kin, hand, support_gate, MotionFailure)
''').body[0])
    loop.body.append(ast.parse('''
if recovery_mode in ('BRAKE', 'LOWER') and _r26_recovery.current['ready']:
    break
''').body[0])
    trial.body.insert(trial.body.index(loop) + 1, ast.parse("if recovery_mode is not None:\n    _r26_recovery.finish(MotionFailure)").body[0])
    tracking = _one(trial.body, lambda n: isinstance(n, ast.If)
        and "APPROACH_TRACKING_TIMEOUT" in ast.unparse(n), "endpoint tracking gate")
    position_call = _one(ast.walk(tracking.test), lambda n: isinstance(n, ast.Call)
        and ast.unparse(n) == "kin.position(target)", "endpoint tracking position")
    for node in ast.walk(tracking.test):
        for name, value in ast.iter_fields(node):
            if value is position_call:
                setattr(node, name, ast.parse("kin.position(target) if recovery_mode is None else _r26_recovery.current['target_world_m']", mode="eval").body)
    attempts = _one(task.body, lambda n: isinstance(n, ast.For) and ast.unparse(n.target) == "attempt", "attempt loop")
    recovery = _one(attempts.body, lambda n: isinstance(n, ast.Try) and "move(State.RELEASE, command.copy()" in ast.unparse(n), "recovery block")
    attempts.body[attempts.body.index(recovery)] = ast.parse(R26_RECOVERY).body[0]
    rendered = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    telemetry = RecoveryTelemetry()
    namespace = dict(base.__globals__, _r26_recovery=telemetry, _r26_parameters=dict(PARAMETERS))
    exec(compile(rendered, "<r26_brake_lower_release_arm>", "exec"), namespace)
    wrapped = namespace["run_task"]
    wrapped.__dict__.update(base.__dict__)
    wrapped.actual_wrist_probe_telemetry = probe_telemetry
    wrapped.rate_limited_probe_telemetry = probe_telemetry
    wrapped.safe_recovery_telemetry = telemetry
    wrapped.r26_safe_recovery_transform = {
        "format": "r26_safe_recovery_ast_extension_v2", "inherited_source_sha256": INHERITED_SOURCE_SHA256,
        "compiled_source_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "readable_source_unchanged": False, "inherited_r25_transform": copy.deepcopy(base.r25_rate_limited_probe_transform),
        "replacement_scope": "recovery sequencing, wrist reference, private hand hold and abort-only force gate",
        "parameters": dict(PARAMETERS), "original_task_and_strict_gates_retained": True,
        "release_requires_original_table_support": True, "force_limit_is_abort_only": True,
        "runtime_target_state_used": False, "target_qpos_writes_added": False,
    }
    return wrapped, rendered, telemetry
