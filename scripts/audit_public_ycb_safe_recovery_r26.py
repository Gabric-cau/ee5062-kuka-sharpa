#!/usr/bin/env python3
"""Read-only R26 recovery reconstruction; never recompute or replace task score."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile

import mujoco
import numpy as np

import audit_public_ycb_settled_alignment as stream
from audit_public_ycb_actual_wrist_probe import table_load
from public_ycb_evidence import load_evidence_model
from verify_public_ycb_strict import channel_pressures, hand_geometry


FORMAT = "r26_supported_recovery_native_reconstruction_audit_v1"
RUNNER = "public_ycb_safe_recovery_development_r26"
TELEMETRY = "r26_brake_supported_lower_release_telemetry_v1"
HOOK = "scripts/safe_recovery_controller_hooks_r26.py"
DT, CLOCK_TOL = .002, 1e-7
POLICY = {"brake_timeout_s": 1.5, "settling_window_s": .15,
    "settling_speed_limit_m_s": .005, "lower_speed_m_s": .004,
    "maximum_lower_distance_m": .05, "lower_timeout_s": 13.,
    "force_abort_limit_n": 12., "hold_hand_phase": "R26_RECOVERY_HOLD"}
digest, safe_file, number, vector, equal = stream.digest, stream.safe_file, stream.finite, stream.array, stream.equal


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tick(value):
    clock = number(value, "clock")
    index = round(clock / DT)
    require(abs(clock - index * DT) <= CLOCK_TOL, "Clock is off the native 2ms grid")
    return index


def same_vector(actual, expected, name):
    require(np.allclose(vector(actual, (3,), name), expected, rtol=0., atol=1e-9), name + " differs")


def one(nodes, predicate, label):
    matches = [node for node in nodes if predicate(node)]
    require(len(matches) == 1, "Expected one source anchor: " + label)
    return matches[0]


def literal(tree, name):
    node = one(tree.body, lambda n: isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets), name)
    return ast.literal_eval(node.value)


def source_contract(source, frozen_hook, transform):
    """Check compiled text binding and declared control branches without running it."""
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    require(source_sha == transform["compiled_source_sha256"], "Actual derived/compiled source SHA differs")
    require(transform["readable_source_unchanged"] is False
        and transform["force_limit_is_abort_only"] is True
        and transform["runtime_target_state_used"] is False
        and transform["target_qpos_writes_added"] is False, "Unsupported R26 transform declaration")
    tree, hook = ast.parse(source), ast.parse(frozen_hook)
    parameters = literal(hook, "PARAMETERS")
    require(all(parameters.get(k) == v for k, v in POLICY.items())
        and transform["parameters"] == parameters, "Recovery parameters differ from audited contract")
    move = one(ast.walk(tree), lambda n: isinstance(n, ast.FunctionDef) and n.name == "move", "move")
    require(move.args.args[-1].arg == "recovery_mode"
        and isinstance(move.args.defaults[-1], ast.Constant)
        and move.args.defaults[-1].value is None, "Normal/clearance move does not default to ordinary control")
    hold = one(ast.walk(move), lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "recovery_mode in ('BRAKE', 'LOWER')"
        and any(isinstance(x, ast.Expr) and "hand.phase_begin" in ast.unparse(x) for x in n.body), "hand hold")
    require(ast.unparse(ast.Module(body=hold.body, type_ignores=[])) ==
        "hand.phase_begin(_r26_parameters['hold_hand_phase'], data)\nhand.inhibit_admission()",
        "BRAKE/LOWER do not retain the declared private hold/admission block")
    reference = one(ast.walk(move), lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "recovery_mode is not None"
        and any("_r26_recovery.target(" in ast.unparse(x) for x in n.body), "fixed/lower reference")
    require(ast.unparse(reference.body[0]) ==
        "desired_command = _r26_recovery.target(data, kin, command, MotionFailure)", "Wrong recovery reference")
    guide = one(ast.walk(move), lambda n: isinstance(n, ast.If)
        and ast.unparse(n.test) == "recovery_mode is None"
        and any("_contact_guard.guide(" in ast.unparse(x) for x in n.body), "ordinary tactile escape")
    require(ast.unparse(guide.orelse[0]) == "_contact_guard.read(data)", "Recovery still invokes arm escape")
    declared = ast.parse(literal(hook, "R26_RECOVERY")).body[0]
    actual = one(ast.walk(tree), lambda n: isinstance(n, ast.Try)
        and any(isinstance(x, ast.Expr) and "hand.abort_probe(controller.failure" in ast.unparse(x)
            for x in n.body), "recovery sequence")
    require(ast.dump(declared) == ast.dump(actual), "Actual recovery block differs from frozen hook")
    recovery_moves = [n for n in ast.walk(actual) if isinstance(n, ast.Call) and ast.unparse(n.func) == "move"]
    ordinary = [n for n in recovery_moves if not any(k.arg == "recovery_mode" for k in n.keywords)]
    require(len(ordinary) == 2 and all(ast.unparse(n.args[0]) == "State.RECOVER" for n in ordinary),
        "Inherited clearance is missing or frozen by recovery mode")
    target = one(ast.walk(hook), lambda n: isinstance(n, ast.FunctionDef) and n.name == "target", "target method")
    target_text = ast.unparse(target)
    require("endpoint = item['start_wrist_world_m'].copy()" in target_text
        and "if item['mode'] == 'LOWER':" in target_text
        and "endpoint[2] -= distance" in target_text,
        "Source lacks the admission anchor/vertical-only reference contract")
    return {"derived_equals_recorded_compiled_source_sha256": True, "derived_sha256": source_sha,
        "frozen_hook_sha256": hashlib.sha256(frozen_hook.encode()).hexdigest(),
        "ordinary_clearance_not_in_fixed_reference_branch": True,
        "historical_process_identity_independently_proven": False,
        "per_sample_ik_reference_independently_recomputed": False}, parameters


class SupportReconstruction:
    """Independent arithmetic for the existing instrumented table filter/gate."""
    def __init__(self, config):
        for key in ("minimum_load_n", "confirmation_seconds", "filter_seconds", "maximum_sample_gap_seconds"):
            require(number(config[key], key) > 0., "Invalid support configuration")
        self.config, self.last, self.filtered, self.since = config, None, 0., None

    def update(self, timestamp, load):
        timestamp, load = number(timestamp, "support clock"), number(load, "support load")
        require(load >= 0. and (self.last is None or timestamp > self.last), "Invalid support sample")
        dt = 0. if self.last is None else timestamp - self.last
        if self.last is None or dt > self.config["maximum_sample_gap_seconds"]:
            self.filtered, self.since = 0., None
        else:
            self.filtered += -math.expm1(-dt / self.config["filter_seconds"]) * (load - self.filtered)
        supported = load >= self.config["minimum_load_n"] and self.filtered >= self.config["minimum_load_n"]
        if not supported:
            self.since = None
        elif self.since is None:
            self.since = timestamp
        confirmed = bool(supported and timestamp - self.since + 1e-12 >= self.config["confirmation_seconds"])
        self.last = timestamp
        return confirmed


def reconstruct_session(item, observations, writes, failure):
    mode, start = item["mode"], tick(item["start_s"])
    require(mode in ("BRAKE", "LOWER", "RELEASE"), "Unknown recovery mode")
    require(start in observations, "Recovery start lacks a native-clock wrist/support observation")
    anchor = vector(item["start_wrist_world_m"], (3,), "stage anchor")
    same_vector(anchor, observations[start]["position"], "Stage admission wrist anchor")
    for name, reference in item["initial_hand_references_rad"].items():
        require(name in observations[start]["controls"], "Initial hand reference lacks native actuator mapping")
        equal(reference, observations[start]["controls"][name], "initial native hand reference " + name, 1e-9)
    if mode == "RELEASE":
        require(observations[start]["support"] is True, "Release began before original support confirmation")
    samples = item["samples"]
    require(isinstance(samples, list), "Recovery samples are not an array")
    previous, settled, ready, peak, missing_inline, emergency_writes = anchor, 0., False, 0., 0, 0
    all_hold_commands_nonreleasing = True
    violation = None
    for ordinal, sample in enumerate(samples, 1):
        index = tick(sample["timestamp_s"])
        require(index == start + ordinal and index in observations, "Missing/noncontiguous recovery native sample")
        raw = observations[index]
        require(raw["state"] == ("RELEASE" if mode == "RELEASE" else "RECOVER"), "Native recovery phase differs")
        position = vector(raw["position"], (3,), "native-clock cached wrist")
        same_vector(sample["actual_wrist_world_m"], position, "Logged recovery wrist")
        speed = float(np.linalg.norm(position - previous) / DT)
        settled = settled + DT if speed <= POLICY["settling_speed_limit_m_s"] else 0.
        ready = bool(settled + 1e-9 >= POLICY["settling_window_s"] and (mode == "BRAKE" or raw["support"]))
        equal(sample["actual_wrist_speed_m_s"], speed, "recovery speed", 1e-8)
        equal(sample["settled_seconds"], settled, "continuous settling window", 1e-8)
        require(sample["mode"] == mode and sample["ready"] is ready
            and sample["original_table_support_confirmed"] is raw["support"], "Recovery mode/support/readiness differs")
        require(set(sample["channel_pressures_n"]) == set(raw["pressures"]), "Incomplete hand force channels")
        for channel, pressure in raw["pressures"].items():
            equal(sample["channel_pressures_n"][channel], pressure, "native channel " + channel, 1e-7)
        peak = max(peak, max(raw["pressures"].values(), default=0.))
        force_violation = max(raw["pressures"].values(), default=0.) > POLICY["force_abort_limit_n"]
        support_violation = mode == "RELEASE" and not raw["support"]
        if force_violation or support_violation:
            require(ordinal == len(samples) and item["finished"] is False,
                "Recovery continued/finished after an abort threshold violation")
            violation = "R26_HAND_FORCE_LIMIT" if force_violation else "R26_TABLE_SUPPORT_LOST_DURING_RELEASE"
            require(failure == "RECOVERY_" + violation, "Recovery failure does not retain native abort reason")
        if raw["inline"] is None:
            require(violation is not None and ordinal == len(samples), "Missing inline recovery sample without final abort")
            missing_inline += 1
        else:
            require(raw["inline"] == sample, "Trace inline recovery telemetry differs from result")
        require(index - 1 in writes, "Recovery sample lacks original hand writer evidence")
        writer = writes[index - 1]
        if mode in ("BRAKE", "LOWER"):
            require(sample["hand_phase"] == POLICY["hold_hand_phase"]
                and writer["phase"] == POLICY["hold_hand_phase"], "BRAKE/LOWER entered an opening hand phase")
            require(writer["mode"] not in ("RELEASING", "OPEN_HOLD"), "Hand writer explicitly opened during hold")
            admission = writer["admission_after_write"]
            require(admission["preload_ready"] is False and admission["probe_active"] is False
                and admission["payload_confirmed"] is False, "Failure recovery regained grasp admission")
            equal(admission["probe_eligible_seconds"], 0., "recovery admission window")
            equal(admission["probe_confirmed_seconds"], 0., "recovery confirmation window")
            emergency_writes += int(writer["final_safety_override"] is True)
            for finger in writer["per_finger"]:
                for name, joint in finger["joints"].items():
                    equal(raw["controls"][name], joint["applied_command_rad"], "applied native hand control", 1e-9)
            all_hold_commands_nonreleasing &= writer["mode"] not in ("RELEASING", "OPEN_HOLD")
        else:
            require(sample["hand_phase"] == "RELEASE" and writer["phase"] == "RELEASE", "Missing explicit release phase")
        require(not ready or mode == "RELEASE" or ordinal == len(samples), "Stage missed its first settled termination")
        previous = position
    require(type(item["ready"]) is bool and type(item["finished"]) is bool
        and item["ready"] is ready, "Final recovery flags differ")
    equal(item["settled_seconds"], settled, "final settling window", 1e-8)
    expected = anchor.copy()
    if mode == "LOWER" and samples:
        expected[2] -= min(POLICY["maximum_lower_distance_m"],
            POLICY["lower_speed_m_s"] * max(0., (len(samples) - 1) * DT))
    same_vector(item["target_world_m"], expected, "Final issued recovery target")
    if item["finished"]:
        require(samples and (ready or mode == "RELEASE") and violation is None, "Unsupported completed recovery stage")
        if mode == "RELEASE":
            equal(len(samples) * DT, 4., "original stationary release duration", CLOCK_TOL)
    elif violation is None:
        require(isinstance(failure, str) and failure.startswith("RECOVERY_"), "Unfinished recovery lacks a failed outcome")
    return {"mode": mode, "start_s": start * DT, "end_s": (start + len(samples)) * DT,
        "native_samples": len(samples), "finished": item["finished"], "ready": ready,
        "reconstructed_peak_channel_n": peak, "abort_reason": violation,
        "final_abort_samples_without_inline_assignment": missing_inline,
        "continuous_settled_seconds": settled, "emergency_override_writes": emergency_writes,
        "no_explicit_opening_in_hold": all_hold_commands_nonreleasing if mode != "RELEASE" else None,
        "initial_reference_commands_bound": True,
        "physical_closed_hand_or_force_invariant_proven": False,
        "reference_check_scope": "Fixed anchor/vertical lowering source plus final issued endpoint; individual IK targets are not logged"}


def sequence_contract(items, reports, observations, result, native_last_tick):
    state, completed = "INITIAL", 0
    for ordinal, (item, report) in enumerate(zip(items, reports)):
        mode = item["mode"]
        allowed = {"INITIAL": {"BRAKE"}, "AFTER_FIRST": {"LOWER", "BRAKE"},
            "AFTER_LOWER": {"BRAKE"}, "AFTER_SECOND": {"RELEASE"}, "AFTER_RELEASE": {"BRAKE"}}
        require(mode in allowed[state], "Recovery mode sequence is invalid")
        if ordinal:
            previous = reports[ordinal - 1]
            require(previous["finished"] is True and report["start_s"] >= previous["end_s"] - CLOCK_TOL,
                "Recovery phases overlap or continue an unfinished phase")
            if state != "AFTER_RELEASE":
                equal(report["start_s"], previous["end_s"], "adjacent recovery phase clock", CLOCK_TOL)
        if state == "AFTER_FIRST":
            supported = observations[tick(item["start_s"])]["support"]
            require((mode == "LOWER") is (not supported), "Lowering bypass/selection differs from original support gate")
        if state in ("INITIAL", "AFTER_RELEASE"):
            state = "AFTER_FIRST"
        elif mode == "LOWER":
            state = "AFTER_LOWER"
        elif mode == "BRAKE":
            state = "AFTER_SECOND"
        else:
            state = "AFTER_RELEASE"
            completed += int(report["finished"])
        if report["abort_reason"] is not None:
            require(ordinal == len(items) - 1 and tick(report["end_s"]) == native_last_tick,
                "Native integration continued after recovery force/support abort")
            require(result["success"] is False, "Abort was relabeled as task success")
            require(any(tick(t["timestamp"]) == native_last_tick and t["state"] == "ABORT"
                and t["reason"] == result["failure"] for t in result["transitions"]), "Abort lacks final state transition")
    recoveries = result["successful_recoveries"]
    require(type(recoveries) is int and 0 <= recoveries <= completed,
        "Original successful-recovery count exceeds completed supported releases")
    return {"ordered_phase_prefix_valid": True, "supported_release_sequences_finished": completed,
        "original_successful_recoveries": recoveries, "recovery_completed": recoveries > 0,
        "recovery_exercised": bool(items), "all_recorded_stages_finished": all(r["finished"] for r in reports)}


def collect_native(native_path, trace_path, model, sessions, config):
    needed = set()
    for item in sessions:
        needed.add(tick(item["start_s"]))
        needed.update(tick(s["timestamp_s"]) for s in item["samples"])
    groups = itertools.groupby(stream.json_array(trace_path), lambda row: tick(row["timestamp"]))
    trace_groups = iter(groups)
    group = next(trace_groups, None)
    hand, target, table = hand_geometry(model), model.geom("target_geom").id, model.geom("table").id
    wrist = model.body("link7").id
    scratch, previous_arm, observations = mujoco.MjData(model), None, {}
    support, count, maximum_fk, whole_peak = SupportReconstruction(config), 0, 0., 0.
    names = [model.actuator(i).name for i in range(model.nu)]
    for native in stream.json_array(native_path):
        count += 1
        index = tick(native["timestamp"])
        require(index == count, "Whole native stream is not contiguous from 2ms")
        while group is not None and group[0] < index:
            stale = list(group[1])
            require(group[0] == 0, "Trace has a non-native timestamp")
            group = next(trace_groups, None)
        require(group is not None and group[0] == index, "Native sample lacks raw trace")
        traces = list(group[1])
        raw = traces[0]
        group = next(trace_groups, None)
        require(raw["state"] == native["state"], "First integrated trace/native state differs")
        position = vector(raw["wrist_position"], (3,), "trace wrist")
        if previous_arm is not None:
            scratch.qpos[:7] = previous_arm
            mujoco.mj_kinematics(model, scratch)
            residual = float(np.max(np.abs(scratch.xpos[wrist] - position)))
            require(residual <= 1e-8, "Cached wrist differs from previous integrated encoders")
            maximum_fk = max(maximum_fk, residual)
        previous_arm = vector(raw["arm_joint_position"], (7,), "actual arm encoders")
        pressures = channel_pressures(native["native_contacts"], hand)
        whole_peak = max(whole_peak, max(pressures.values()))
        for name, value in pressures.items():
            equal(raw["hand_segment_forces"][name], value, "trace/native hand pressure", 1e-7)
        load = table_load(native, target, table)
        confirmed = support.update(number(native["timestamp"], "native time"), load)
        equal(raw["table_load_sensor"]["raw_load_n"], load, "native table support load", 1e-7)
        equal(raw["table_load_sensor"]["filtered_load_n"], support.filtered, "original table filter", 1e-7)
        require(raw["table_load_sensor"]["confirmed"] is confirmed, "Original support confirmation differs")
        controls = vector(native["motor_commands"], (model.nu,), "native motor controls")
        require(np.allclose(controls[7:], raw["hand_command"], atol=1e-10, rtol=0.), "Trace/native hand controls differ")
        inline = [t["safe_recovery"] for t in traces if "safe_recovery" in t]
        require(len(inline) <= 1, "Repeated inline recovery sample at one native tick")
        if index in needed:
            observations[index] = {"position": position, "pressures": pressures, "support": confirmed,
                "state": native["state"], "controls": {name: float(value) for name, value in zip(names, controls)},
                "inline": inline[0] if inline else None}
        elif inline:
            raise ValueError("Unaccounted recovery telemetry in native trace")
    require(group is None and count > 0 and needed <= observations.keys(), "Unbound/empty native or recovery data")
    return observations, count, maximum_fk, whole_peak


def audit_directory(directory, expected_summary_sha256):
    directory = Path(directory).resolve()
    paths = {name: safe_file(directory, relative) for name, relative in (
        ("summary", "summary.json"), ("protocol", "protocol.json"), ("archive", "frozen_source.zip"),
        ("native_receipt", "verification_receipt.json"), ("strict_receipt", "strict_native_verification.json"))}
    initial_hashes = {name: digest(path) for name, path in paths.items()}
    summary, protocol = (json.loads(paths[k].read_text()) for k in ("summary", "protocol"))
    require(digest(paths["summary"]) == expected_summary_sha256 and summary["protocol"] == protocol
        and summary["complete"] is True and summary["runner_errors"] == 0 and protocol["format"] == RUNNER,
        "Run is incomplete, failed, wrong revision or not bound to expected summary")
    manifest = protocol["source_hashes_before"]
    require(manifest == protocol["source_hashes_after"] and protocol["source_changed_during_run"] is False,
        "Source manifest changed during run")
    with zipfile.ZipFile(paths["archive"]) as archive:
        require(len(archive.namelist()) == len(manifest) and set(archive.namelist()) == set(manifest),
            "Frozen source ZIP manifest differs")
        for name, sha in manifest.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and hashlib.sha256(archive.read(name)).hexdigest() == sha, "Frozen source bytes differ: " + name)
        frozen_hook = archive.read(HOOK).decode()
    require(len(summary["rows"]) == len(protocol["schedule"]) == 1, "Expected isolated R26 case")
    row = summary["rows"][0]
    require(row["kind"] == "ycb_can" and row["seed"] == 3414 and row["runner_error"] is None,
        "Unexpected/failed R26 case")
    case = directory / row["path"]
    safe_file(directory, row["path"] + "/metadata.json")
    paths.update({name: safe_file(case, relative) for name, relative in (
        ("metadata", "metadata.json"), ("result", "result.json"), ("source", "derived_controller.py"), ("scene", "scene.xml"))})
    for name in ("native", "trace", "bias_recovery"):
        artifact = row["artifacts"][name]
        path = safe_file(case, artifact["path"])
        require(digest(path) == artifact["sha256"] and path.stat().st_size == artifact["bytes"], "Changed artifact: " + name)
        paths[name] = path
    before = {name: digest(path) for name, path in paths.items()}
    require(all(before[name] == expected for name, expected in initial_hashes.items()),
        "Finalized inputs changed during initial binding")
    metadata, result = (json.loads(paths[k].read_text()) for k in ("metadata", "result"))
    require(metadata == row and result["success"] is row["success"] and result["failure"] == row["failure"]
        and result["runner_revision"] == RUNNER, "Result/summary/metadata outcomes differ")
    for name in ("safe_recovery", "safe_recovery_source_transform"):
        require(result[name] == row[name], "Result/metadata recovery evidence differs")
    transform, telemetry = result["safe_recovery_source_transform"], result["safe_recovery"]
    require(digest(paths["source"]) == row["derived_controller_sha256"], "Recorded derived source changed")
    source, parameters = source_contract(paths["source"].read_text(), frozen_hook, transform)
    require(telemetry["format"] == TELEMETRY and telemetry["parameters"] == parameters,
        "Recovery telemetry parameters/format differ")
    receipts = {key: json.loads(paths[key].read_text()) for key in ("native_receipt", "strict_receipt")}
    require(all(r["summary_sha256"] == expected_summary_sha256 for r in receipts.values())
        and receipts["native_receipt"]["complete"] is True
        and receipts["native_receipt"]["native_samples_contiguous_2ms"] is True
        and receipts["strict_receipt"]["accepted_evidence"] is True, "Unbound/unaccepted native or strict receipt")
    strict_row = one(receipts["strict_receipt"]["rows"], lambda r: r["path"] == row["path"], "strict case")
    require(strict_row["result_sha256"] == before["result"] and strict_row["unknown"] is False
        and strict_row["scene_sha256"] == before["scene"]
        and row["scene_sha256_before_load"] == row["scene_sha256_at_finalize"] == before["scene"]
        and row["scene_unchanged_since_load"] is True,
        "Strict receipt is not bound to current result")
    model = load_evidence_model(paths["scene"])
    equal(float(model.opt.timestep), DT, "native model timestep", 1e-12)
    config = result["table_support_parameters"]
    for name, expected in (("confirmation_seconds", .12), ("filter_seconds", .03),
            ("maximum_sample_gap_seconds", .02)):
        equal(config[name], expected, "original support " + name, 1e-12)
    equal(config["minimum_load_n"], .7 * float(model.body_mass[model.body("target").id])
        * abs(float(model.opt.gravity[2])), "original support threshold", 1e-9)
    observations, count, fk_residual, whole_peak = collect_native(
        paths["native"], paths["trace"], model, telemetry["attempts"], config)
    require(row["native_audit"]["native_samples"] == row["native_audit"]["integration_steps"] == count
        and row["native_audit"]["external_target_qpos_writes"] == 0, "Native count/target write audit differs")
    needed_writes = {tick(s["timestamp_s"]) - 1 for item in telemetry["attempts"] for s in item["samples"]}
    writes = {}
    for event in stream.json_array(paths["bias_recovery"]):
        index = tick(event["timestamp_s"])
        if index in needed_writes:
            require(index not in writes, "Repeated hand writer at one recovery tick")
            writes[index] = event
    reports = [reconstruct_session(item, observations, writes, result["failure"]) for item in telemetry["attempts"]]
    sequence = sequence_contract(telemetry["attempts"], reports, observations, result, count)
    require(not any(r["abort_reason"] for r in reports) or strict_row["strict_success"] is False,
        "Strict success relabels a recovery abort")
    require(all(digest(path) == before[name] for name, path in paths.items()), "Evidence changed during read-only audit")
    return {"format": FORMAT, "accepted_recovery_evidence": True, "summary_sha256": expected_summary_sha256,
        "script_sha256": digest(__file__), "source": source,
        "inputs": {name: {"path": str(path), "sha256": before[name], "bytes": path.stat().st_size} for name, path in paths.items()},
        "inputs_unchanged_during_audit": True, "native_samples": count,
        "maximum_cached_wrist_fk_residual_m": fk_residual, "whole_episode_peak_channel_n": whole_peak,
        "sequence": sequence, "stages": reports, "task_success": row["success"], "held_lift": row["held_lift"],
        "strict_success": strict_row["strict_success"], "original_failure": result["failure"],
        "score_recomputed_or_rewritten": False, "force_invariant_proven": False,
        "scope": "Recorded recovery ordering, cached wrist finite differences, original table-support arithmetic and native hand channels only. The <=5mm/s,0.15s window is low-speed settling, not zero velocity. Hold excludes explicit opening; phase synchronization and original emergency relief may change references or unload. Source and final reference endpoints are checked, individual IK targets are not recorded. No LP/QP/contact dynamics reconstruction, integration, live-state mutation or historical process-identity proof. Accepted failed evidence and recovery completion never certify task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Audit requires a new output receipt")
    try:
        receipt, status = audit_directory(args.directory, args.expected_summary_sha256), 0
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        receipt, status = {"format": FORMAT, "accepted_recovery_evidence": False,
            "script_sha256": digest(__file__), "error": str(error), "score_recomputed_or_rewritten": False}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"accepted_recovery_evidence": receipt["accepted_recovery_evidence"], "receipt": str(args.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
