#!/usr/bin/env python3
"""Independent finite R23 probe reference, native-clock and retained-source audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np

import audit_public_ycb_settled_alignment as align
from public_ycb_evidence import load_evidence_model


FORMAT = "r23_actual_wrist_probe_reference_native_and_retained_source_audit_v1"
RUNNER = "public_ycb_actual_wrist_probe_development_r23"
DERIVED_SHA256 = "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"
BASELINE = "results/public_ycb_settled_alignment_20261022_r22_bias10_close32_seed3414"
BASELINE_RECEIPT = "results/r22_settled_alignment_audit_20261022/receipt.json"
PARAMETERS = {"path_seconds": 3., "settle_seconds": 3., "maximum_probe_seconds": 6.,
    "distance_m": .005, "actual_error_gain": 2., "reference_correction_limit_m": .002,
    "cartesian_ik_sample_seconds": .02}
PINS = {
    "scripts/actual_wrist_probe_controller_hooks.py": "a60164c9c2538693ed89dc5855dad44944b85ba0ee83b60396f838b368aeb39e",
    "tests/test_actual_wrist_probe_controller_hooks.py": "038e2511253fd6c45927f8ca421fe144efdac809d1b21ffff02c452062a8ba2b",
    "docs/R23_ACTUAL_WRIST_PROBE_POLICY_20261022.md": "25fbdba20bc2fb237e4b922333fb77a6c6835c6d1dc11496f0cb6d62bc6428fa",
    "scripts/develop_public_ycb_actual_wrist_probe_r23.py": "3a6911195b5eb48892f8debdd9b6a9105815fe26577f6f1b14237036526683ee",
    "tests/test_public_ycb_actual_wrist_probe_r23.py": "3abbd841140e3ef5ab34e81ca1f380e34bf8e0a9dc844afa355544708e829c72",
    "scripts/audit_public_ycb_settled_alignment.py": "c313cd881fb482d689604162bdd4854c75655cf02999b058fe1ae1ea5e74d17b",
    "scripts/public_ycb_evidence.py": align.LOADER_SHA256,
    "scripts/develop_public_ycb_cartesian_probe_r12.py": "8729e049bae598a397933d69e8bf40f1e74534025c81c297effdfe04ec454ddf",
    BASELINE + "/summary.json": "f67343f1e534694bebf6b4f5c3f50bac619ed5a276d9cc621c374a08a7498618",
    BASELINE_RECEIPT: "66368f6f499b32a3957e676f779759e4e0ec68b84a72bd92978e902ac23f82c3",
    BASELINE + "/ycb_can_bias_budget_r17_seed3414/derived_controller.py": align.DERIVED_SHA256,
}
DT = .002
digest, finite, array, equal, safe_file = align.digest, align.finite, align.array, align.equal, align.safe_file


def one(items, predicate, name):
    values = [item for item in items if predicate(item)]
    if len(values) != 1:
        raise ValueError("Expected exactly one R23 source anchor: " + name)
    return values[0]


def same_node(node, source, name):
    if ast.dump(node) != ast.dump(ast.parse(source).body[0]):
        raise ValueError("Unexpected R23 extension structure: " + name)


def verify_ast_extension(inherited, transformed):
    """Remove only exact declared extensions, then compare the complete R22 AST."""
    tree = ast.parse(transformed)
    task = one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "run_task", "task")
    move = one(task.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "move", "move")
    setup = one(move.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_r23_probe", "setup")
    same_node(setup, "_r23_probe = None", "setup")
    move.body.remove(setup)
    begin = one(move.body, lambda n: isinstance(n, ast.If) and "_r23_probe_telemetry.begin" in ast.unparse(n), "begin")
    same_node(begin, """if state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed:
    _r23_probe = _r23_probe_telemetry.begin(data, hand, cartesian_motion, MotionFailure)
    duration = _r23_probe_parameters['maximum_probe_seconds']""", "begin")
    move.body.remove(begin)
    guard = one(move.body, lambda n: isinstance(n, ast.Try) and any(h.name == "_r23_motion_error" for h in n.handlers), "exception logging")
    expected = ast.parse("""try:
    pass
except Exception as _r23_motion_error:
    if _r23_probe is not None:
        _r23_probe.finish(data, hand, 'MOTION_EXCEPTION', motion_error=_r23_motion_error)
    raise""").body[0]
    if (ast.dump(ast.Module(body=guard.handlers, type_ignores=[])) != ast.dump(ast.Module(body=expected.handlers, type_ignores=[]))
            or guard.orelse or guard.finalbody):
        raise ValueError("R23 exception guard changes inherited control behavior")
    index = move.body.index(guard)
    move.body[index:index + 1] = guard.body
    finish = one(move.body, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "_r23_probe is not None and _r23_probe.outcome is None", "bounded end")
    same_node(finish, """if _r23_probe is not None and _r23_probe.outcome is None:
    _r23_probe.finish(data, hand, 'MAXIMUM_PROBE_DURATION_REACHED')""", "bounded end")
    move.body.remove(finish)
    loop = one(move.body, lambda n: isinstance(n, ast.For), "native loop")
    segment = one(ast.walk(loop), lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "cartesian_motion and step % segment_steps == 0", "20ms segment")
    branch = one(segment.body, lambda n: isinstance(n, ast.If) and "_r23_probe.target" in ast.unparse(n), "reference")
    if len(branch.orelse) != 1:
        raise ValueError("Original Cartesian reference is not retained")
    original_position = branch.orelse[0]
    branch.orelse = []
    same_node(branch, """if _r23_probe is not None and state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed:
    next_position = _r23_probe.target(data, kin, end_step * model.opt.timestep, MotionFailure)""", "reference")
    segment.body[segment.body.index(branch)] = original_position
    sample = one(loop.body, lambda n: isinstance(n, ast.If) and "_r23_probe.sample" in ast.unparse(n), "sample")
    same_node(sample, """if _r23_probe is not None:
    trace[-1]['actual_wrist_probe'] = _r23_probe.sample(data, kin, hand, MotionFailure)""", "sample")
    loop.body.remove(sample)
    early = one(loop.body, lambda n: isinstance(n, ast.If) and "_r23_probe.finish" in ast.unparse(n), "early exit")
    same_node(early, """if _r23_probe is not None and state == State.TRIAL_LIFT and hand.probe_active and not _estimated.payload_confirmed and hand.probe_confirmed_seconds >= _probe_parameters['confirmation_seconds']:
    _r23_probe.finish(data, hand, 'ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED')
    trace[-1]['actual_wrist_probe_termination'] = 'ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED'
    break""", "early exit")
    loop.body.remove(early)
    confirm = one(ast.walk(task), lambda n: isinstance(n, ast.Call) and ast.unparse(n.func) == "_r23_probe_telemetry.confirm_probe", "decisive confirmation")
    if ast.unparse(confirm) != "_r23_probe_telemetry.confirm_probe(hand, data)":
        raise ValueError("Original decisive confirmation arguments changed")
    confirm.func, confirm.args = ast.parse("hand.confirm_probe", mode="eval").body, [ast.Name(id="data", ctx=ast.Load())]
    if ast.dump(tree) != ast.dump(ast.parse(inherited)):
        raise ValueError("Removing declared extensions does not recover the entire executed R22 arm/guards")
    return {"entire_r22_arm_ast_restored": True, "all_original_guards_and_close_and_align_retained": True,
        "original_decisive_confirmation_call_retained_once": True}


def reference_values(start, actual, elapsed):
    """Independent arithmetic; never call the runtime R12 reference function."""
    start, actual = array(start, (3,), "admission wrist"), array(actual, (3,), "IK cached wrist")
    fraction = min(1., max(0., finite(elapsed, "reference elapsed") / 3.))
    progress = fraction * fraction * (3. - 2. * fraction)
    desired = start + np.array([0., 0., .005 * progress])
    error = desired - actual
    correction = 2. * error
    length = float(np.linalg.norm(correction))
    if length > .002:
        correction *= .002 / length
    return {"cubic_progress": progress, "desired_actual_path_world_m": desired,
        "actual_wrist_at_ik_sample_world_m": actual, "actual_path_error_world_m": error,
        "bounded_reference_correction_world_m": correction, "cartesian_ik_target_world_m": desired + correction}


def table_load(native, target, table):
    result = 0.
    for contact in native["native_contacts"]:
        if set(contact["geom_ids"]) != {target, table}:
            continue
        frame = align.rotation(contact["contact_frame_world_rows"], "native contact frame")
        wrench = array(contact["force_torque_contact_frame"], (6,), "native contact wrench")
        result += float((frame.T @ wrench[:3])[2]) * (1. if contact["geom_ids"][1] == target else -1.)
    return max(0., result)


def source_proof(directory, protocol):
    before = protocol["source_hashes_before"]
    if (len(before) != 167 or before != protocol["source_hashes_after"]
            or protocol["source_changed_during_run"] is not False
            or any(before.get(name) != value for name, value in PINS.items())):
        raise ValueError("R23 finalized 167-source identity differs")
    archive_path = safe_file(directory, "frozen_source.zip")
    archive_sha = digest(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(before):
            raise ValueError("R23 source ZIP manifest differs")
        for name, expected in before.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("R23 archived source bytes differ: " + name)
        baseline = json.loads(archive.read(BASELINE + "/summary.json"))
        old = baseline["protocol"]["source_hashes_before"]
        if len(old) != 146 or any(before.get(name) != value for name, value in old.items()):
            raise ValueError("R22 frozen inherited 146 sources differ")
        inherited = archive.read(BASELINE + "/ycb_can_bias_budget_r17_seed3414/derived_controller.py").decode()
    if digest(archive_path) != archive_sha:
        raise ValueError("Source ZIP changed during verification")
    return {"source_count": 167, "inherited_r22_source_count": 146,
        "before_after_manifest_equal": True, "all_archived_input_hashes_match": True,
        "archive_sha256": archive_sha}, inherited, baseline["rows"][0]


def tick(timestamp):
    timestamp = finite(timestamp, "native clock")
    index = round(timestamp / DT)
    equal(timestamp, index * DT, "native 2ms clock", 1e-7)
    return index


def collect_streams(native_path, trace_path, model, alignments, probes, expected_open):
    groups = iter(itertools.groupby(align.json_array(trace_path), key=lambda row: row["timestamp"]))
    scratch, wrist = mujoco.MjData(model), model.body("link7").id
    target, table = model.geom("target_geom").id, model.geom("table").id
    hand_ids = {name: model.actuator(name).id for name in expected_open}
    def fk(command):
        scratch.qpos[:7] = command
        mujoco.mj_kinematics(model, scratch)
        return scratch.xpos[wrist].copy(), scratch.xmat[wrist].reshape(3, 3).copy()
    wanted = {tick(item["start_s"]) for item in probes}
    wanted.update(tick(row["timestamp_s"]) for item in probes for row in item["ik_samples"])
    caches, align_rows, probe_rows, closes = {}, [[] for _ in alignments], [[] for _ in probes], []
    previous_q, previous_state, count, max_residual = None, None, 0, 0.
    for count, native in enumerate(align.json_array(native_path), 1):
        timestamp = finite(native["timestamp"], "native timestamp")
        equal(timestamp, count * DT, "contiguous native clock", 1e-7)
        try:
            clock, group = next(groups)
        except StopIteration as error:
            raise ValueError("Trace ends before native stream") from error
        equal(clock, timestamp, "native/first trace clock", 1e-7)
        row = next(group)
        # Exhaust phase-end duplicates; only record()'s first occurrence invokes NativeAudit.sample.
        for duplicate in group:
            if tick(duplicate["timestamp"]) != count:
                raise ValueError("Malformed duplicate trace clock")
        motor = array(native["motor_commands"], (29,), "native motor controls")
        if row["state"] != native["state"] or not np.allclose(array(row["hand_command"], (22,), "trace hand controls"), motor[7:], atol=1e-12, rtol=0.):
            raise ValueError("Native state/22 hand outputs differ from the first integrated trace row")
        position = array(row["wrist_position"], (3,), "cached wrist position")
        rotation = align.rotation(row["wrist_rotation_world"], "cached wrist")
        fp, fr = (None, None) if previous_q is None else fk(previous_q)
        if fp is not None:
            residual = max(float(np.max(np.abs(fp - position))), float(np.max(np.abs(fr - rotation))))
            if residual > 1e-8:
                raise ValueError("Cached wrist differs from the previous integrated raw arm encoder FK")
            max_residual = max(max_residual, residual)
        if count in wanted:
            if fp is None:
                raise ValueError("Probe reference requires a preceding integrated encoder observation")
            caches[count] = fp.copy()
        command = array(row["arm_command"], (7,), "post-limiter logical arm command")
        if native["state"] == "ALIGN":
            choices = [i for i, part in enumerate(alignments) if part["start_s"] + DT / 2 < timestamp <= part["end_s"] + 1e-7]
            if len(choices) != 1:
                raise ValueError("Native ALIGN has no unique transition interval")
            align_rows[choices[0]].append({"timestamp_s": timestamp, "position": position, "rotation": rotation,
                "fk_position": fp, "fk_rotation": fr, "command": command,
                "open_commands": {name: float(motor[aid]) for name, aid in hand_ids.items()},
                "alignment_settling": row.get("alignment_settling")})
        for ordinal, item in enumerate(probes):
            end = finite(item["outcome"]["timestamp_s"], "probe outcome timestamp")
            if item["start_s"] + DT / 2 < timestamp <= end + 1e-7:
                if native["state"] != "TRIAL_LIFT" or fp is None:
                    raise ValueError("Probe native observation is outside TRIAL_LIFT/FK coverage")
                probe_rows[ordinal].append({"timestamp_s": timestamp, "position": fp.copy(),
                    "trace_sample": row.get("actual_wrist_probe"), "policy": row["probe_policy"],
                    "termination": row.get("actual_wrist_probe_termination"),
                    "table_load_n": table_load(native, target, table)})
        if native["state"] == "CLOSE" and previous_state != "CLOSE":
            closes.append({"first_native_s": timestamp, "command": command.tolist()})
        previous_q, previous_state = array(row["arm_joint_position"], (7,), "postintegration arm encoders"), native["state"]
    if next(groups, None) is not None or set(caches) != wanted:
        raise ValueError("Unmatched trace clocks or missing probe reference observation")
    return align_rows, probe_rows, caches, closes, count, max_residual, fk


def vector_equal(actual, expected, name):
    if not np.allclose(array(actual, (3,), name), expected, atol=1e-9, rtol=0.):
        raise ValueError(name + " differs from independent cached FK/reference")


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + " must retain an actual Boolean")
    return value


def geometry_policy(policy, position, previous_position, start, table, weight, required):
    displacement = position - start
    speed = float(np.linalg.norm(position - previous_position) / DT)
    endpoint = bool(np.linalg.norm(displacement - [0., 0., .005]) <= .0005)
    vector_equal(policy["actual_wrist_position_world_m"], position, "hand-policy cached wrist")
    vector_equal(policy["actual_probe_displacement_m"], displacement, "hand-policy probe displacement")
    equal(policy["actual_wrist_speed_m_s"], speed, "hand-policy cached speed", 1e-8)
    equal(policy["table_load_n"], table, "native target-specific table load", 1e-7)
    equal(policy["catalogue_weight_n"], weight, "catalogue weight", 1e-9)
    unloaded = table <= .2 * weight
    if (boolean(policy["actual_probe_endpoint_reached"], "endpoint") is not endpoint
            or boolean(policy["table_unloaded"], "table unloaded") is not unloaded):
        raise ValueError("Hand-policy endpoint/table flags differ from native geometry/contacts")
    reasons = list(policy["motion_safety_reasons"])
    if speed > .01:
        reasons.insert(0, "SPEED")
    if displacement[2] > .0055:
        reasons.insert(0, "DISTANCE_EXCEEDED")
    if displacement[2] < -.0005:
        reasons.insert(0, "DOWNWARD_DRIFT")
    if np.linalg.norm(displacement[:2]) > .002:
        reasons.insert(0, "HORIZONTAL_DRIFT")
    if not required <= set(policy["associated_active_fingers"]):
        reasons.insert(0, "CONTACT_LOSS")
    fixed = boolean(policy["sampled_fixed_normal_1g_feasible"], "recorded fixed-normal 1g")
    current = boolean(policy["sampled_solve_current"], "recorded current solve")
    visual = boolean(policy["accepted_visual_after_probe_start"], "recorded fresh visual")
    confirmed = bool(not reasons and endpoint and unloaded and fixed and current and visual)
    return confirmed, None if not reasons else "PROBE_" + reasons[0], speed


def check_bias_abort(event, timestamp, motion_error):
    """Bind only the source-defined R16 abort that runs before R23 sample()."""
    equal(event["timestamp_s"] + DT, timestamp, "R16 abort preceding write clock", 1e-7)
    equal(event["post_step_timestamp_s"], timestamp, "R16 abort post-step clock", 1e-7)
    before, after = event["admission_before"], event["admission_after_step"]
    if (event["format"] != "r17_bias_budget_write_v1"
            or event["inherited_write_format"] != "r16_bias_recovery_write_v1"
            or event["additional_bias_limit_rad"] != .10 or event["phase"] != "TRIAL_LIFT"
            or event["probe_abort_applied"] is not True
            or not (event["recovery_active"] is True or event["failure_latched"] is True)
            or before["probe_active"] is not True or before["payload_confirmed"] is not False
            or after["probe_active"] is not False or after["payload_confirmed"] is not False
            or after["probe_confirmed_seconds"] != 0. or after["probe_eligible_seconds"] != 0.
            or after["pressure_window_samples"] != 0 or after["preload_ready"] is not False
            or event["post_step_returned_failure"] != motion_error
            or event["resolved_abort_reason"] != motion_error):
        raise ValueError("Inactive probe sample lacks its exact original R16 abort/inhibit transition")


def audit_probe(item, rows, caches, admission, decisive, transitions, weight, bias_aborts=None):
    start = finite(item["start_s"], "probe start")
    outcome = item["outcome"]
    end = finite(outcome["timestamp_s"], "probe end")
    duration = end - start
    if not -1e-9 <= duration <= 6. + 1e-7:
        raise ValueError("Probe exceeded its six-second bound")
    equal(admission["timestamp_s"], start, "accepted admission clock", 1e-7)
    if admission["admission"] is not True or admission["decisive_current_solve"] is not True:
        raise ValueError("R23 probe lacks accepted original hand admission")
    actual_start = caches[tick(start)]
    vector_equal(item["actual_admission_start_world_m"], actual_start, "actual admission start")
    vector_equal(admission["actual_wrist_position_world_m"], actual_start, "original admission cached wrist")
    required = set(admission["associated_active_fingers"])
    events, ik_rows = item["samples"], item["ik_samples"]
    if len(events) != len(rows) or abs(len(rows) * DT - duration) > 1e-7:
        raise ValueError("Probe native/sample clocks differ from its actual duration")
    minimum_ik = (len(rows) + 9) // 10
    if not minimum_ik <= len(ik_rows) <= minimum_ik + (1 if len(rows) % 10 == 0 and outcome["motion_failure"] else 0):
        raise ValueError("Probe 20ms IK sample count differs")
    for index, row in enumerate(ik_rows):
        timestamp, elapsed = start + index * .02, (index + 1) * .02
        equal(row["timestamp_s"], timestamp, "IK preintegration clock", 1e-7)
        equal(row["actual_elapsed_s"], timestamp - start, "actual IK elapsed", 1e-7)
        equal(row["path_elapsed_s"], elapsed, "20ms next-segment lookahead", 1e-9)
        if row["parameters"] != PARAMETERS or row["target_pose_used"] is not False:
            raise ValueError("IK reference parameters or state-input scope differ")
        vector_equal(row["actual_admission_start_world_m"], actual_start, "IK admission start")
        for name, expected in reference_values(actual_start, caches[tick(timestamp)], elapsed).items():
            if name == "cubic_progress":
                equal(row[name], expected, "IK cubic progress", 1e-12)
            else:
                vector_equal(row[name], expected, "IK " + name)
    bias_aborts = {} if bias_aborts is None else bias_aborts
    confirmed_seconds, previous_position, last_stop, first_ready, observed_bias_abort = 0., actual_start, None, None, False
    for index, (event, raw) in enumerate(zip(events, rows)):
        timestamp = raw["timestamp_s"]
        equal(timestamp, start + (index + 1) * DT, "probe contiguous native clock", 1e-7)
        if event != raw["trace_sample"]:
            raise ValueError("Probe result/metadata telemetry differs from the first integrated trace")
        if event["parameters"] != PARAMETERS or event["runtime_target_state_used"] is not False:
            raise ValueError("Probe sample parameter/state semantics differ")
        abort = bias_aborts.get(tick(timestamp))
        if abort is not None:
            check_bias_abort(abort, timestamp, outcome["motion_failure"])
            if (index != len(rows) - 1 or event["currently_probe_active"] is not False
                    or outcome["motion_failure"] is None or outcome["probe_active_at_motion_end"] is not False):
                raise ValueError("R16 pre-sample abort was not the final inactive failed observation")
            observed_bias_abort = True
            # R16 inhibits during write, before parent after_step, then inhibits again after it.
            confirmed_seconds = 0.
        elif event["currently_probe_active"] is not True:
            raise ValueError("Inactive probe sample lacks a bound original R16 abort")
        equal(event["timestamp_s"], timestamp, "probe sample clock", 1e-7)
        equal(event["elapsed_s"], timestamp - start, "probe elapsed", 1e-7)
        vector_equal(event["actual_admission_start_world_m"], actual_start, "sample admission start")
        vector_equal(event["current_actual_wrist_world_m"], raw["position"], "sample cached wrist")
        ik = ik_rows[index // 10]
        equal(event["latest_ik_timestamp_s"], ik["timestamp_s"], "sample current IK timestamp", 1e-7)
        equal(event["ik_sample_age_s"], timestamp - ik["timestamp_s"], "sample IK age", 1e-7)
        for key in ("desired_actual_path_world_m", "bounded_reference_correction_world_m", "cartesian_ik_target_world_m"):
            vector_equal(event[key], ik[key], "sample " + key)
        policy = raw["policy"]
        equal(policy["probe_start_s"], start, "original hand admission start", 1e-7)
        confirm, stop, speed = geometry_policy(policy, raw["position"], previous_position, actual_start,
            raw["table_load_n"], weight, required)
        confirmed_seconds = confirmed_seconds + DT if confirm else 0.
        equal(policy["probe_confirmed_seconds"], confirmed_seconds, "original continuous hand window", 1e-8)
        if abort is not None:
            if abort["original_post_step_failure"] != stop:
                raise ValueError("R16 abort changed the reconstructed original parent failure")
            confirmed_seconds = 0.
        equal(event["inherited_confirmed_seconds"], confirmed_seconds, "sample hand window", 1e-8)
        if policy["stop_reason"] != stop:
            raise ValueError("Recorded original hand stop reason differs")
        if confirmed_seconds >= .25 and first_ready is None:
            first_ready = index
        if stop and index != len(rows) - 1:
            raise ValueError("Probe continued after an inherited hand stop")
        previous_position, last_stop = raw["position"], stop
    equal(outcome["elapsed_s"], duration, "probe actual outcome duration", 1e-7)
    equal(outcome["inherited_confirmed_seconds"], confirmed_seconds, "outcome hand window", 1e-8)
    motion_error = outcome["motion_failure"]
    arm_reason = outcome["arm_exit_reason"]
    if motion_error is not None:
        if (not isinstance(motion_error, str) or not motion_error
                or outcome["original_confirmation_attempted"] is not False
                or outcome["original_confirmation_accepted"] is not None or decisive is not None):
            raise ValueError("Motion failure attempted decisive confirmation")
        if last_stop is not None and motion_error != last_stop:
            raise ValueError("Original hand stop was replaced by a different motion failure")
        if outcome["final_reason"] != motion_error:
            raise ValueError("Probe failure reason was not retained")
        classification = "motion_failure"
    else:
        if arm_reason == "MAXIMUM_PROBE_DURATION_REACHED":
            equal(duration, 6., "full probe deadline", 1e-7)
            if first_ready is not None:
                raise ValueError("Probe missed the first original eligible early exit")
        elif arm_reason == "ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED":
            if first_ready != len(rows) - 1 or not rows or rows[-1]["termination"] != arm_reason:
                raise ValueError("Probe early exit lacks the first original continuous confirmation window")
        else:
            raise ValueError("Unsupported normal probe arm termination")
        if outcome["original_confirmation_attempted"] is not True or decisive is None:
            raise ValueError("Normal probe lacks its original decisive current-patch confirmation")
        equal(outcome["confirmation_timestamp_s"], end, "decisive confirmation clock", 1e-7)
        equal(decisive["timestamp_s"], end, "original confirmation artifact clock", 1e-7)
        equal(decisive["probe_confirmed_seconds"], confirmed_seconds, "decisive hand window", 1e-8)
        if not rows:
            raise ValueError("Normal decisive confirmation lacks a native wrist/contact observation")
        # The decisive force solve changes tactile flags, but does not integrate the arm or contacts.
        vector_equal(decisive["actual_wrist_position_world_m"], rows[-1]["position"], "decisive cached wrist")
        vector_equal(decisive["actual_probe_displacement_m"], rows[-1]["position"] - actual_start,
            "decisive displacement")
        equal(decisive["actual_wrist_speed_m_s"], speed, "decisive cached speed", 1e-8)
        equal(decisive["table_load_n"], rows[-1]["table_load_n"], "decisive native table load", 1e-7)
        equal(decisive["catalogue_weight_n"], weight, "decisive catalogue weight", 1e-9)
        if (boolean(decisive["actual_probe_endpoint_reached"], "decisive endpoint") is not
                bool(np.linalg.norm(rows[-1]["position"] - actual_start - [0., 0., .005]) <= .0005)
                or boolean(decisive["table_unloaded"], "decisive unloaded") is not
                    (rows[-1]["table_load_n"] <= .2 * weight)):
            raise ValueError("Decisive geometry/table flags differ from independent native evidence")
        for flag in ("probe_active", "sampled_fixed_normal_1g_feasible", "accepted_visual_after_probe_start"):
            boolean(decisive[flag], "original decisive " + flag)
        accepted = bool(decisive["probe_active"] and confirmed_seconds >= .25 and not decisive["motion_safety_reasons"]
            and decisive["actual_probe_endpoint_reached"] and decisive["table_unloaded"]
            and decisive["sampled_fixed_normal_1g_feasible"] and decisive["accepted_visual_after_probe_start"]
            and decisive["actual_wrist_speed_m_s"] <= .01
            and required <= set(decisive["associated_active_fingers"]))
        if (decisive["decisive_current_solve"] is not True or type(decisive["confirmation"]) is not bool
                or decisive["confirmation"] is not accepted or outcome["original_confirmation_accepted"] is not accepted
                or outcome["final_reason"] != ("ORIGINAL_DECISIVE_CONFIRMATION_ACCEPTED" if accepted else "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED")):
            raise ValueError("Original decisive confirmation result/control outcome differs")
        classification = "original_confirmation_accepted_probe_only" if accepted else "original_confirmation_rejected"
    if motion_error is not None or classification == "original_confirmation_rejected":
        reason = motion_error or "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED"
        if not any(abs(row["timestamp"] - end) <= 1e-7 and row["state"] == "RECOVER" and row["reason"] == reason for row in transitions):
            raise ValueError("Failed probe does not retain its original recovery transition")
    return {"classification": classification, "start_s": start, "end_s": end, "duration_s": duration,
        "native_samples": len(rows), "ik_samples": len(ik_rows), "arm_exit_reason": arm_reason,
        "original_confirmation_attempted": outcome["original_confirmation_attempted"],
        "original_confirmation_accepted": outcome["original_confirmation_accepted"],
        "final_reason": outcome["final_reason"], "final_confirmed_seconds": confirmed_seconds,
        "original_r16_pre_sample_abort_observed": observed_bias_abort,
        "all_reference_rows_independently_recomputed": True, "all_native_clock_and_cached_wrist_checks_consistent": True,
        "recorded_lp_association_and_visual_flags_independently_recomputed": False}


def audit_directory(directory, expected_summary_sha256):
    directory = Path(directory).resolve(strict=True)
    if (not isinstance(expected_summary_sha256, str) or len(expected_summary_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_summary_sha256)):
        raise ValueError("An explicit lowercase final-summary SHA256 is required")
    paths = {name: safe_file(directory, filename) for name, filename in
        (("summary", "summary.json"), ("protocol", "protocol.json"), ("archive", "frozen_source.zip"))}
    summary_bytes, protocol_bytes = paths["summary"].read_bytes(), paths["protocol"].read_bytes()
    if hashlib.sha256(summary_bytes).hexdigest() != expected_summary_sha256:
        raise ValueError("Final summary differs from explicitly expected SHA256")
    summary, protocol = json.loads(summary_bytes), json.loads(protocol_bytes)
    if (summary["complete"] is not True or summary["runner_errors"] or summary["protocol"] != protocol
            or summary["runner_revision"] != RUNNER or len(summary["rows"]) != 1
            or protocol["format"] != RUNNER or protocol["runner_revision"] != RUNNER
            or protocol["controller_class"] != "ProvisionalPreloadBudgetHand"
            or protocol["target_allocator_revision"] != "public_ycb_shared_normal_development_r20"
            or protocol["admission_policy_revision"] != "public_ycb_provisional_preload_development_r21"
            or protocol["alignment_settling"] != align.POLICY
            or protocol["actual_wrist_probe"] != PARAMETERS or protocol["maximum_attempts"] != 2):
        raise ValueError("Expected finalized explicitly versioned R23 evidence")
    source, inherited, baseline_row = source_proof(directory, protocol)
    row = summary["rows"][0]
    count = row["effective_hand_instances"]
    if (row["runner_error"] is not None or row["kind"] != "ycb_can" or row["seed"] != 3414
            or row["runner_revision"] != RUNNER or type(count) is not int or not 1 <= count <= 2
            or count != len(row["attempts"]) or row["effective_hand_budgets_rad"] != [.10] * count
            or row["effective_hand_controller_types"] != ["ProvisionalPreloadBudgetHand"] * count
            or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or row["derived_controller_sha256"] != DERIVED_SHA256
            or row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != align.SCENE_SHA256
            or row["scene_sha256_at_finalize"] != align.SCENE_SHA256):
        raise ValueError("R23 case/attempt/physical source identity differs")
    case = safe_file(directory, str(Path(row["path"]) / "scene.xml")).parent
    for name, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"),
            ("metadata", "metadata.json"), ("result", "result.json")):
        paths[name] = safe_file(case, filename)
    paths.update(model_loader=Path(__file__).with_name("public_ycb_evidence.py"),
        alignment_auditor=Path(align.__file__).resolve())
    for name in ("native", "trace", "load_transfer_probe", "bias_recovery"):
        artifact = row["artifacts"][name]
        paths[name] = safe_file(case, artifact["path"])
        if digest(paths[name]) != artifact["sha256"] or paths[name].stat().st_size != artifact["bytes"]:
            raise ValueError("Finalized raw artifact hash/size differs: " + name)
    before = {name: digest(path) for name, path in paths.items()}
    metadata, result = json.loads(paths["metadata"].read_text()), json.loads(paths["result"].read_text())
    if (before["summary"] != expected_summary_sha256
            or before["protocol"] != hashlib.sha256(protocol_bytes).hexdigest()
            or before["archive"] != source["archive_sha256"] or metadata != row
            or before["controller"] != DERIVED_SHA256 or before["scene"] != align.SCENE_SHA256
            or before["model_loader"] != align.LOADER_SHA256
            or before["alignment_auditor"] != PINS["scripts/audit_public_ycb_settled_alignment.py"]
            or result["runner_revision"] != RUNNER or result["success"] != row["success"]
            or any(result[key] != row[key] for key in ("attempts", "alignment_settling",
                "alignment_source_transform", "actual_wrist_probe", "actual_wrist_probe_source_transform"))
            or row["alignment_source_transform"] != baseline_row["alignment_source_transform"]
            or row["arm_source_transform"] != baseline_row["arm_source_transform"]
            or row["arm_source_transform_is_inherited_r19_prefix_only"] is not True):
        raise ValueError("R23 summary/result/metadata/inherited source binding differs")
    declaration = protocol["r23_actual_wrist_probe_experiment"]
    if (declaration["r22_comparison_summary_sha256"] != PINS[BASELINE + "/summary.json"]
            or declaration["baseline_derived_controller_sha256"] != align.DERIVED_SHA256
            or declaration["baseline_scene_sha256"] != align.SCENE_SHA256
            or declaration["baseline_child_exit_observed"] is not True
            or declaration["baseline_child_returncode"] != 0
            or declaration["baseline_r22_alignment_evidence_accepted"] is not True
            or declaration["changed_parameters"] != {"unconfirmed_TRIAL_LIFT_actual_wrist_probe": PARAMETERS}):
        raise ValueError("R23 inherited finalized R22 declaration differs")
    transform = row["actual_wrist_probe_source_transform"]
    if (transform["format"] != "actual_wrist_probe_ast_extension_v1"
            or transform["inherited_source_sha256"] != align.DERIVED_SHA256
            or transform["transformed_source_sha256"] != DERIVED_SHA256
            or transform["reference_source_sha256"] != PINS["scripts/develop_public_ycb_cartesian_probe_r12.py"]
            or transform["parameters"] != PARAMETERS
            or any(transform[key] is not True for key in ("settled_alignment_source_retained",
                "close_motion_source_unchanged", "original_decisive_confirmation_called",
                "all_original_native_safety_checks_retained"))
            or transform["runtime_target_state_used"] is not False or transform["target_qpos_writes_added"] is not False):
        raise ValueError("Unsupported R23 arm source transformation")
    source.update(verify_ast_extension(inherited, paths["controller"].read_text()))
    telemetry, alignment = row["actual_wrist_probe"], row["alignment_settling"]
    probes = telemetry["attempts"]
    if (telemetry["format"] != "actual_wrist_probe_policy_v1" or telemetry["parameters"] != PARAMETERS
            or telemetry["reference_source_sha256"] != transform["reference_source_sha256"]
            or not 0 <= len(probes) <= count
            or alignment["format"] != "settled_open_alignment_policy_v1" or alignment["parameters"] != align.POLICY):
        raise ValueError("Unsupported actual probe/retained ALIGN telemetry")
    if any(probes[i]["outcome"]["timestamp_s"] >= probes[i + 1]["start_s"] for i in range(len(probes) - 1)):
        raise ValueError("Probe attempts overlap or repeat")
    intervals = align.alignment_intervals(result["transitions"])
    if len(intervals) != len(alignment["attempts"]) or not 0 <= len(intervals) <= count:
        raise ValueError("Retained ALIGN transition/attempt count differs")
    model = load_evidence_model(paths["scene"])
    if model.nu != 29 or abs(float(model.opt.timestep) - DT) > 1e-15:
        raise ValueError("Expected unchanged seven-arm/22-hand actuator model at 2ms")
    names = [model.actuator(aid).name for aid in range(7, model.nu)]
    if len(set(names)) != 22 or "left_thumb_CMC_FE" not in names or any(not n.startswith("left_") for n in names):
        raise ValueError("Unknown native hand actuator mapping")
    equal(row["hand_config"]["thumb_opposition_rad"], 1.3, "unchanged open thumb opposition")
    expected_open = {name: 1.3 if name == "left_thumb_CMC_FE" else 0. for name in names}
    observations, raw_probes, caches, closes, native_count, max_fk, fk = collect_streams(
        paths["native"], paths["trace"], model, intervals, probes, expected_open)
    if (native_count != row["native_audit"]["native_samples"]
            or native_count != row["native_audit"]["integration_steps"]):
        raise ValueError("R23 native sample count differs from finalized metadata")
    alignment_audits = [align.reconstruct_attempt(attempt, samples, interval, expected_open, fk)
        for attempt, samples, interval in zip(alignment["attempts"], observations, intervals)]
    align.verify_close_linkage(closes, alignment_audits)
    admissions, confirmations = [], []
    for event in align.json_array(paths["load_transfer_probe"]):
        if event.get("admission") is True:
            admissions.append(event)
        if "confirmation" in event:
            confirmations.append(event)
    if len(admissions) != len(probes):
        raise ValueError("Accepted original hand admissions differ from tracked probes")
    bias_aborts = {}
    for event in align.json_array(paths["bias_recovery"]):
        timestamp = event.get("post_step_timestamp_s")
        if event.get("probe_abort_applied") is not True or timestamp is None:
            continue
        if any(item["start_s"] < timestamp <= item["outcome"]["timestamp_s"] + 1e-7 for item in probes):
            index = tick(timestamp)
            if index in bias_aborts:
                raise ValueError("Repeated original R16 probe abort at one native clock")
            bias_aborts[index] = event
    audits, used_confirmations = [], 0
    weight = finite(row["catalogue_mass_prior_kg"], "catalogue mass") * 9.81
    equal(weight, 3.42369, "frozen can catalogue weight", 1e-10)
    for item, samples, admission in zip(probes, raw_probes, admissions):
        matches = [event for event in confirmations
            if abs(event["timestamp_s"] - item["outcome"]["timestamp_s"]) <= 1e-7]
        if len(matches) > 1:
            raise ValueError("Repeated decisive confirmation at a probe outcome")
        used_confirmations += len(matches)
        audits.append(audit_probe(item, samples, caches, admission, matches[0] if matches else None,
            result["transitions"], weight, bias_aborts))
    if used_confirmations != len(confirmations):
        raise ValueError("Unbound original decisive confirmation artifacts")
    if any(digest(path) != before[name] for name, path in paths.items()):
        raise ValueError("An input changed during the read-only probe reconstruction")
    return {"format": FORMAT, "accepted_probe_evidence": True,
        "summary_sha256": before["summary"], "protocol_sha256": before["protocol"],
        "script_sha256": digest(__file__), "source": source,
        "inputs": {name: {"path": str(path), "sha256": before[name], "bytes": path.stat().st_size}
            for name, path in paths.items()}, "inputs_unchanged_during_audit": True,
        "checks": {"all_evidence_consistent": True,
            "all_probe_reference_and_native_clock_checks_consistent": True,
            "all_probe_termination_and_original_confirmation_checks_consistent": True,
            "all_retained_arm_source_and_alignment_checks_consistent": True},
        "case": row["path"], "actual_hand_instances": count, "probes_exercised": bool(probes),
        "native_samples": native_count, "native_timestep_s": DT,
        "alignments": alignment_audits, "probes": audits,
        "first_global_native_cached_wrist_fk_excluded": 1,
        "maximum_cached_wrist_fk_residual": max_fk,
        "task_success": row["success"], "held_lift": row["held_lift"], "mujoco_version": mujoco.__version__,
        "scope": "Finite R23 same-can probe audit. The entire inherited R22 arm AST is recovered by removing only exact R23 extensions, retaining ALIGN/CLOSE and all existing guards. ALIGN is freshly reconstructed from this R23 native/trace stream with narrow pinned helpers, not certified by reusing the old R22 receipt. Cached wrist FK uses the previous integrated raw arm encoders; the first global native sample is excluded. R12 reference arithmetic, native clocks, target-specific table load, geometry, confirmation-window timing and original decisive-call outcome are checked independently. Original LP/pressure, association, visual freshness flags and other safety-guard predicates are reconciled as recorded conditions, not independently rebuilt; gravity mapping, final physical grasp success and whole task score are outside this audit. Logical arm_command is distinct from gravity-compensated native arm motor controls. No integration or live state modification is performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Probe audit requires a new output receipt")
    try:
        receipt, status = audit_directory(args.directory, args.expected_summary_sha256), 0
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        receipt, status = {"format": FORMAT, "accepted_probe_evidence": False,
            "script_sha256": digest(__file__), "error": str(error)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"accepted_probe_evidence": receipt["accepted_probe_evidence"], "receipt": str(args.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
