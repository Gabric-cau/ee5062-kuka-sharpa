#!/usr/bin/env python3
"""Narrow independent R22 ALIGN timing, open-output and cached-wrist audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np

from public_ycb_evidence import load_evidence_model


FORMAT = "r22_settled_alignment_native_output_and_cached_wrist_audit_v1"
RUNNER = "public_ycb_settled_alignment_development_r22"
POLICY = {"maximum_extra_seconds": 6., "position_tolerance_m": .001,
    "maximum_actual_speed_m_s": .002, "continuous_seconds": .25,
    "command_tolerance_rad": 1e-10, "open_command_tolerance_rad": 1e-10,
    "orientation_error_is_a_gate": False}
PINS = {
    "scripts/settled_alignment_controller_hooks.py": "83269930d9333ae3797c73fd417c4f302e082884a40321a92ff7775047bc7b48",
    "scripts/develop_public_ycb_settled_alignment_r22.py": "5297ff8c9bf36fa283e4bc3b1a3094e0cfd2a13423b278d48f5600a441d72529",
    "tests/test_settled_alignment_controller_hooks.py": "6a4e448d058fc5051d792dfcb5f6c78a63a90e2a424aebfcf7bfa821d566c653",
    "tests/test_public_ycb_settled_alignment_r22.py": "da0d1a134fc41b2a4b932bcc8e67dcf069397dc9d84c3eb413d0d0a0f558670a",
    "docs/SETTLED_ALIGNMENT_POLICY_20261022.md": "7d73ab9ac24194529e978a2e6c4c1184690cedd88d51d2ce2ffef9e87cfd7136",
}
DERIVED_SHA256 = "206db2d2b6f08dedc1757ba9ec9445957f96bf6fab36fbba139a670acba0d1eb"
INHERITED_SHA256 = "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485"
BASELINE_SUMMARY = "results/public_ycb_provisional_preload_20261022_r21_bias10_close32_seed3414/summary.json"
BASELINE_SHA256 = "6c1308f33e0ce2f15a7160b97d11b74eb882d358bc3a818d47618926145b212c"
SCENE_SHA256 = "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e"
LOADER_SHA256 = "fc0194592711e37a888f1fe85d11a795a1af234c0b1d277789037ee5ff7c9954"
DT, CLOCK_TOLERANCE, VALUE_TOLERANCE = .002, 1e-7, 1e-9
SAFETY_FAILURES = {"SUSTAINED_HAND_OVERLOAD", "ARM_VELOCITY_LIMIT", "ROBOT_TABLE_PENETRATION",
    "WORKSPACE_VIOLATION", "CONTACT_GUARD_NOT_RELIEVED", "CONTACT_RETREAT_IK",
    "ARM_COMMAND_JOINT_LIMIT", "CARTESIAN_PATH_IK"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value, name):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("Nonfinite or nonnumeric " + name)
    return float(value)


def array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid " + name)
    return result


def rotation(value, name):
    result = array(value, (3, 3), name)
    if not np.allclose(result @ result.T, np.eye(3), atol=1e-8, rtol=0.) or abs(np.linalg.det(result) - 1.) > 1e-8:
        raise ValueError("Invalid rotation " + name)
    return result


def equal(actual, expected, name, tolerance=VALUE_TOLERANCE):
    if expected is None:
        if actual is not None:
            raise ValueError(name + " must be absent for the first sampled speed")
    elif abs(finite(actual, name) - expected) > tolerance:
        raise ValueError(name + " differs from independently reconstructed value")


def safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Evidence path escapes its directory")
    path = Path(root)
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Symlink evidence is not accepted")
    if not path.is_file():
        raise ValueError("Missing evidence file: " + str(path))
    return path


def json_array(path):
    """Stream compressed JSON without loading native/trace arrays in memory."""
    decoder, buffer, position, state = json.JSONDecoder(), "", 0, "start"
    with gzip.open(path, "rt") as stream:
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position == len(buffer):
                    break
                if state == "start":
                    if buffer[position] != "[":
                        raise ValueError("Expected a JSON array")
                    state, position = "first", position + 1
                    continue
                if buffer[position] == "]":
                    if state not in ("first", "separator"):
                        raise ValueError("Trailing comma in evidence JSON array")
                    if (buffer[position + 1:] + stream.read()).strip():
                        raise ValueError("Trailing stream content")
                    return
                if state == "separator":
                    if buffer[position] != ",":
                        raise ValueError("Missing evidence JSON array separator")
                    state, position = "value", position + 1
                    continue
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                position, state = end, "separator"
                yield value
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise ValueError("Incomplete JSON array")
            buffer, position = buffer[position:] + chunk, 0


def source_proof(directory, protocol):
    before = protocol["source_hashes_before"]
    if (not isinstance(before, dict) or len(before) != 146
            or protocol["source_hashes_after"] != before
            or protocol["source_changed_during_run"] is not False
            or any(before.get(name) != value for name, value in PINS.items())
            or before.get(BASELINE_SUMMARY) != BASELINE_SHA256):
        raise ValueError("R22 frozen 146-source identity changed")
    archive_path = safe_file(directory, "frozen_source.zip")
    archive_sha256 = digest(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(before):
            raise ValueError("Frozen source ZIP manifest differs")
        for name, expected in before.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Frozen source ZIP bytes differ: " + name)
    if digest(archive_path) != archive_sha256:
        raise ValueError("Frozen source ZIP changed during identity verification")
    return {"source_count": 146, "all_archived_input_hashes_match": True,
        "before_after_manifest_equal": True, "r21_summary_sha256": BASELINE_SHA256,
        "archive_sha256": archive_sha256}


def verify_open(expected, actual, name):
    if set(expected) != set(actual) or len(expected) != 22:
        raise ValueError(name + " must contain exactly the 22 hand output names")
    errors = [abs(finite(actual[key], name) - value) for key, value in expected.items()]
    return max(errors)


def reconstruct_attempt(attempt, observations, interval, expected_open, fk_target):
    """Recompute policy using native controls, raw trace commands and scratch FK."""
    start, end = interval["start_s"], interval["end_s"]
    equal(attempt["start_s"], start, "ALIGN start", CLOCK_TOLERANCE)
    equal(attempt["nominal_seconds"], 1.5, "original nominal ALIGN seconds")
    target_command = array(attempt["target_arm_command_rad"], (7,), "full grasp command")
    target_position, target_rotation = fk_target(target_command)
    target_rotation = rotation(target_rotation, "target FK")
    if not np.allclose(rotation(attempt["target_wrist_rotation_world"], "recorded target"), target_rotation, atol=1e-9, rtol=0.):
        raise ValueError("Recorded target rotation differs from full-command FK")
    if verify_open(expected_open, attempt["expected_open_command_rad"], "expected open controls") > 1e-12:
        raise ValueError("Expected open controls differ from frozen hand configuration")
    events, outcome = attempt["events"], attempt["outcome"]
    if len(events) > len(observations) or len(observations) - len(events) > 1:
        raise ValueError("Missing or extra ALIGN telemetry/native samples")
    previous_position, previous_timestamp, window_start = None, None, None
    ready, longest, fk_count = False, 0., 0
    max_fk_position = max_fk_rotation = max_open_error = 0.
    for ordinal, (event, observation) in enumerate(zip(events, observations)):
        timestamp = observation["timestamp_s"]
        equal(timestamp, start + (ordinal + 1) * DT, "contiguous ALIGN clock", CLOCK_TOLERANCE)
        if observation["alignment_settling"] != event:
            raise ValueError("Inline trace ALIGN row differs from result/metadata telemetry")
        equal(event["timestamp_s"], timestamp, "ALIGN telemetry clock", CLOCK_TOLERANCE)
        position = observation["position"]
        actual_rotation = observation["rotation"]
        if observation["fk_position"] is not None:
            fk_count += 1
            residual = float(np.max(np.abs(position - observation["fk_position"])))
            rot_residual = float(np.max(np.abs(actual_rotation - observation["fk_rotation"])))
            if residual > 1e-8 or rot_residual > 1e-8:
                raise ValueError("Cached wrist differs from previous integrated arm encoder FK")
            max_fk_position, max_fk_rotation = max(max_fk_position, residual), max(max_fk_rotation, rot_residual)
            position, actual_rotation = observation["fk_position"], observation["fk_rotation"]
        if not np.allclose(array(event["actual_wrist_position_world_m"], (3,), "logged wrist"), position, atol=1e-8, rtol=0.):
            raise ValueError("ALIGN wrist telemetry differs from native-clock FK/trace")
        if not np.allclose(array(event["target_wrist_position_world_m"], (3,), "logged target"), target_position, atol=1e-9, rtol=0.):
            raise ValueError("ALIGN target position differs from full-command FK")
        elapsed = timestamp - start
        speed = None if previous_position is None else float(np.linalg.norm(position - previous_position) /
            (timestamp - previous_timestamp))
        previous_position, previous_timestamp = position.copy(), timestamp
        position_error = float(np.linalg.norm(position - target_position))
        command_error = float(np.max(np.abs(observation["command"] - target_command)))
        open_error = verify_open(expected_open, observation["open_commands"], "native hand outputs")
        max_open_error = max(max_open_error, open_error)
        proof = event["open_hand_output"]
        if (verify_open(expected_open, proof["expected_command_rad"], "telemetry expected output") > 1e-12
                or verify_open(observation["open_commands"], proof["actual_command_rad"], "telemetry actual output") > 1e-12):
            raise ValueError("Open-hand telemetry differs from native motor commands")
        equal(proof["maximum_error_rad"], open_error, "open-output maximum error", 1e-12)
        open_ok = open_error <= POLICY["open_command_tolerance_rad"]
        if proof["accepted"] is not open_ok:
            raise ValueError("Open-output acceptance differs")
        if not open_ok:
            raise ValueError("A non-open ALIGN output was integrated despite the pre-step guard")
        reasons = []
        if not open_ok:
            reasons.append("OPEN_HAND_OUTPUT_REQUIRED")
        if command_error > POLICY["command_tolerance_rad"]:
            reasons.append("FULL_GRASP_COMMAND_NOT_REACHED")
        if position_error > POLICY["position_tolerance_m"]:
            reasons.append("ACTUAL_WRIST_POSITION")
        if speed is None or speed > POLICY["maximum_actual_speed_m_s"]:
            reasons.append("ACTUAL_WRIST_SPEED")
        if reasons:
            window_start = None
        elif window_start is None:
            window_start = timestamp
        continuous = 0. if window_start is None else timestamp - window_start
        longest = max(longest, continuous)
        nominal_finished = elapsed >= 1.5 - 1e-9
        ready = bool(nominal_finished and not reasons and continuous >= .25 - 1e-9)
        if ready and ordinal != len(events) - 1:
            raise ValueError("ALIGN continued after the first eligible termination")
        for key, value in {"elapsed_s": elapsed, "extra_elapsed_s": max(0., elapsed - 1.5),
                "position_error_m": position_error, "actual_wrist_speed_m_s": speed,
                "maximum_arm_command_error_rad": command_error, "continuous_seconds": continuous,
                "orientation_error_rad": float(np.arccos(np.clip((np.trace(target_rotation.T @ actual_rotation) - 1.) / 2., -1., 1.)))}.items():
            equal(event[key], value, "ALIGN " + key, 1e-7 if key == "orientation_error_rad" else 1e-8)
        if (event["reasons"] != reasons or event["ready"] is not ready
                or event["nominal_finished"] is not nominal_finished
                or event["orientation_error_is_a_gate"] is not False
                or event["runtime_target_state_used"] is not False):
            raise ValueError("ALIGN reconstructed readiness/window/reasons differ")
    if end - start > 7.5 + CLOCK_TOLERANCE:
        raise ValueError("ALIGN exceeded its nominal plus six-second bound")
    trailing = len(observations) - len(events)
    if trailing and observations[-1]["alignment_settling"] is not None:
        raise ValueError("A trailing safety-abort native sample contains unaccounted policy telemetry")
    failure = interval["next_reason"]
    if outcome is None:
        if failure not in SAFETY_FAILURES or ready or interval["next_state"] == "CLOSE":
            raise ValueError("Unexplained unfinished ALIGN telemetry")
        classification = "inherited_safety_abort"
    else:
        if type(outcome["settled"]) is not bool:
            raise ValueError("ALIGN outcome must retain an actual Boolean")
        if outcome["settled"]:
            if (not ready or trailing or interval["next_state"] != "CLOSE"
                    or outcome["reason"] != "OPEN_FULL_GRASP_COMMAND_AND_ACTUAL_WRIST_SETTLED"):
                raise ValueError("Settled ALIGN lacks reconstructed continuous readiness/CLOSE transition")
            classification = "settled"
        elif outcome["reason"] == "ALIGN_SETTLE_TIMEOUT":
            if ready or trailing or interval["next_state"] == "CLOSE" or failure != "ALIGN_SETTLE_TIMEOUT":
                raise ValueError("Invalid failed ALIGN timeout transition")
            equal(end - start, 7.5, "bounded ALIGN timeout", CLOCK_TOLERANCE)
            classification = "timeout"
        elif outcome["reason"] == "ALIGN_HAND_NOT_OPEN":
            if ready or trailing or interval["next_state"] == "CLOSE" or failure != "ALIGN_HAND_NOT_OPEN":
                raise ValueError("Invalid pre-integration non-open ALIGN failure")
            proof = outcome["open_hand_output"]
            if (verify_open(expected_open, proof["expected_command_rad"], "failed expected output") > 1e-12
                    or set(proof["actual_command_rad"]) != set(expected_open)
                    or proof["accepted"] is not False):
                raise ValueError("Malformed declared pre-integration open-output failure")
            actual = proof["actual_command_rad"]
            if any(value is None for value in actual.values()):
                if proof["maximum_error_rad"] is not None:
                    raise ValueError("Nonfinite failed output must retain an unknown maximum error")
                for name, value in actual.items():
                    if value is not None:
                        finite(value, "failed hand output " + name)
            else:
                error = verify_open(expected_open, actual, "failed actual output")
                equal(proof["maximum_error_rad"], error, "failed output error", 1e-12)
                if error <= POLICY["open_command_tolerance_rad"]:
                    raise ValueError("Declared non-open failure contains fully open outputs")
            classification = "declared_unintegrated_nonopen_output_abort"
        else:
            raise ValueError("Unsupported invalid-measurement ALIGN outcome remains unverified")
        equal(outcome["timestamp_s"], end, "ALIGN outcome timestamp", CLOCK_TOLERANCE)
    if observations:
        equal(observations[-1]["timestamp_s"], end, "last native ALIGN clock", CLOCK_TOLERANCE)
    elif abs(start - end) > CLOCK_TOLERANCE:
        raise ValueError("Nonempty ALIGN duration has no native samples")
    return {"classification": classification, "start_s": start, "end_s": end,
        "native_samples": len(observations), "policy_samples": len(events),
        "trailing_safety_abort_native_samples": trailing, "settled": classification == "settled",
        "extra_seconds": max(0., end - start - 1.5), "longest_reconstructed_continuous_seconds": longest,
        "cached_wrist_fk_samples": fk_count, "maximum_cached_wrist_fk_position_residual_m": max_fk_position,
        "maximum_cached_wrist_fk_rotation_element_residual": max_fk_rotation,
        "maximum_native_open_output_error_rad": max_open_error,
        "preintegration_failure_outputs_independently_observed": False if
            classification == "declared_unintegrated_nonopen_output_abort" else None,
        "all_recorded_policy_rows_consistent": True, "target_command_rad": target_command.tolist()}


def alignment_intervals(transitions):
    result = []
    for index, item in enumerate(transitions):
        if item["state"] != "ALIGN":
            continue
        if index + 1 >= len(transitions):
            raise ValueError("Completed result has an unterminated ALIGN transition")
        following = transitions[index + 1]
        result.append({"start_s": finite(item["timestamp"], "ALIGN transition start"),
            "end_s": finite(following["timestamp"], "ALIGN transition end"),
            "next_state": following["state"], "next_reason": following["reason"]})
    return result


def collect_native(native_path, trace_path, model, intervals, expected_open):
    hand_ids = {name: model.actuator(name).id for name in expected_open}
    scratch, body = mujoco.MjData(model), model.body("link7").id
    def fk(command):
        scratch.qpos[:7] = command
        mujoco.mj_kinematics(model, scratch)
        return scratch.xpos[body].copy(), scratch.xmat[body].reshape(3, 3).copy()
    groups = iter(itertools.groupby(json_array(trace_path), key=lambda item: item["timestamp"]))
    observations = [[] for _ in intervals]
    closes = []
    previous_post_arm, count, previous_state = None, 0, None
    for native in json_array(native_path):
        count += 1
        timestamp = finite(native["timestamp"], "native timestamp")
        equal(timestamp, count * DT, "native contiguous 2ms clock", CLOCK_TOLERANCE)
        try:
            trace_timestamp, group = next(groups)
        except StopIteration as error:
            raise ValueError("Trace ends before native stream") from error
        equal(trace_timestamp, timestamp, "raw trace/native clock", CLOCK_TOLERANCE)
        rows = list(group)
        row = rows[0]
        if row["state"] != native["state"]:
            raise ValueError("First trace observation differs from native phase")
        motor = array(native["motor_commands"], (model.nu,), "native motor inputs")
        if not np.allclose(array(row["hand_command"], (22,), "trace hand outputs"), motor[7:], atol=1e-12, rtol=0.):
            raise ValueError("Native hand controls differ from first integrated trace output")
        command = array(row["arm_command"], (7,), "logical arm command")
        if native["state"] == "ALIGN":
            selected = [i for i, part in enumerate(intervals)
                if part["start_s"] + DT / 2 < timestamp <= part["end_s"] + CLOCK_TOLERANCE]
            if len(selected) != 1:
                raise ValueError("Native ALIGN sample has no unique transition interval")
            position = array(row["wrist_position"], (3,), "cached trace wrist")
            rot = rotation(row["wrist_rotation_world"], "cached trace wrist")
            fk_position, fk_rotation = (None, None) if previous_post_arm is None else fk(previous_post_arm)
            tagged = [value["alignment_settling"] for value in rows if "alignment_settling" in value]
            if len(tagged) > 1:
                raise ValueError("Repeated inline ALIGN policy observation at a native clock")
            observations[selected[0]].append({"timestamp_s": timestamp, "position": position,
                "rotation": rot, "fk_position": fk_position, "fk_rotation": fk_rotation,
                "command": command, "open_commands": {name: float(motor[aid]) for name, aid in hand_ids.items()},
                "alignment_settling": tagged[0] if tagged else None})
        if native["state"] == "CLOSE" and previous_state != "CLOSE":
            closes.append({"first_native_s": timestamp, "command": command.tolist()})
        previous_post_arm = array(row["arm_joint_position"], (7,), "postintegration arm encoders")
        previous_state = native["state"]
    if next(groups, None) is not None:
        raise ValueError("Trace contains unmatched clocks after native completion")
    return observations, closes, count, fk


def verify_close_linkage(closes, audits):
    settled = [item for item in audits if item["settled"]]
    if len(closes) != len(settled):
        raise ValueError("CLOSE started without exactly one preceding settled ALIGN")
    for close, item in zip(closes, settled):
        equal(close["first_native_s"], item["end_s"] + DT, "first CLOSE native timestamp", CLOCK_TOLERANCE)
        error = float(np.max(np.abs(array(close["command"], (7,), "first CLOSE command") -
            array(item["target_command_rad"], (7,), "settled full grasp command"))))
        if error > POLICY["command_tolerance_rad"] + 1e-12:
            raise ValueError("CLOSE continued incomplete insertion in command space")
        item["first_close_native_s"] = close["first_native_s"]
        item["first_close_full_target_command_error_rad"] = error


def audit_directory(directory):
    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = safe_file(directory, "summary.json"), safe_file(directory, "protocol.json")
    summary_bytes, protocol_bytes = summary_path.read_bytes(), protocol_path.read_bytes()
    summary, protocol = json.loads(summary_bytes), json.loads(protocol_bytes)
    if (summary["complete"] is not True or summary["runner_errors"] or summary["protocol"] != protocol
            or summary["runner_revision"] != RUNNER or len(summary["rows"]) != 1
            or protocol["format"] != RUNNER or protocol["runner_revision"] != RUNNER
            or protocol["controller_class"] != "ProvisionalPreloadBudgetHand"
            or protocol["target_allocator_revision"] != "public_ycb_shared_normal_development_r20"
            or protocol["admission_policy_revision"] != "public_ycb_provisional_preload_development_r21"
            or protocol["alignment_settling"] != POLICY or protocol["maximum_attempts"] != 2):
        raise ValueError("Expected finalized explicitly versioned R22 evidence")
    source = source_proof(directory, protocol)
    row = summary["rows"][0]
    count = row["effective_hand_instances"]
    if (row["runner_error"] is not None or row["kind"] != "ycb_can" or row["seed"] != 3414
            or row["runner_revision"] != RUNNER or type(count) is not int or not 1 <= count <= 2
            or count != len(row["attempts"]) or row["effective_hand_controller_types"] != ["ProvisionalPreloadBudgetHand"] * count
            or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or row["derived_controller_sha256"] != DERIVED_SHA256
            or row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != row["scene_sha256_at_finalize"]):
        raise ValueError("R22 case/attempt/source/outcome identity differs")
    case = directory / row["path"]
    if not case.resolve().is_relative_to(directory) or case.is_symlink():
        raise ValueError("R22 case escapes evidence root")
    paths = {"summary": summary_path, "protocol": protocol_path,
        "archive": safe_file(directory, "frozen_source.zip"), "scene": safe_file(case, "scene.xml"),
        "controller": safe_file(case, "derived_controller.py"), "metadata": safe_file(case, "metadata.json"),
        "result": safe_file(case, "result.json"),
        "model_loader": Path(__file__).with_name("public_ycb_evidence.py")}
    for name in ("native", "trace"):
        artifact = row["artifacts"][name]
        paths[name] = safe_file(case, artifact["path"])
        if digest(paths[name]) != artifact["sha256"] or paths[name].stat().st_size != artifact["bytes"]:
            raise ValueError("Raw " + name + " hash/size differs from finalized summary")
    before = {name: digest(path) for name, path in paths.items()}
    metadata, result = json.loads(paths["metadata"].read_text()), json.loads(paths["result"].read_text())
    if (before["summary"] != hashlib.sha256(summary_bytes).hexdigest()
            or before["protocol"] != hashlib.sha256(protocol_bytes).hexdigest()
            or before["archive"] != source["archive_sha256"]
            or metadata != row or before["controller"] != DERIVED_SHA256
            or before["scene"] != SCENE_SHA256 or before["scene"] != row["scene_sha256_before_load"]
            or before["model_loader"] != LOADER_SHA256
            or protocol["r22_settled_alignment_experiment"]["baseline_scene_sha256"] != SCENE_SHA256
            or result["alignment_settling"] != row["alignment_settling"]
            or result["alignment_source_transform"] != row["alignment_source_transform"]
            or result["attempts"] != row["attempts"]
            or result["runner_revision"] != RUNNER or result["success"] != row["success"]):
        raise ValueError("R22 summary/metadata/result/physical source bindings differ")
    transform = row["alignment_source_transform"]
    if (transform["format"] != "settled_open_alignment_ast_extension_v1"
            or transform["inherited_source_sha256"] != INHERITED_SHA256
            or transform["transformed_source_sha256"] != DERIVED_SHA256
            or transform["parameters"] != POLICY or transform["target_qpos_writes_added"] is not False
            or transform["close_motion_source_unchanged"] is not True):
        raise ValueError("Unknown R22 arm source transformation")
    telemetry = row["alignment_settling"]
    if telemetry["format"] != "settled_open_alignment_policy_v1" or telemetry["parameters"] != POLICY:
        raise ValueError("Unknown alignment telemetry format or thresholds")
    intervals = alignment_intervals(result["transitions"])
    if len(intervals) != len(telemetry["attempts"]) or not 0 <= len(intervals) <= count:
        raise ValueError("ALIGN transitions/telemetry do not match actual attempt count")
    model = load_evidence_model(paths["scene"])
    if model.nu != 29 or abs(float(model.opt.timestep) - DT) > 1e-15:
        raise ValueError("Expected native KUKA seven plus Sharpa 22 actuator model at 2ms")
    names = [model.actuator(aid).name for aid in range(7, model.nu)]
    if (len(set(names)) != 22 or "left_thumb_CMC_FE" not in names
            or any(not name.startswith("left_") for name in names)):
        raise ValueError("Unknown hand actuator identity")
    equal(row["hand_config"]["thumb_opposition_rad"], 1.3, "frozen thumb opposition")
    expected_open = {name: 1.3 if name == "left_thumb_CMC_FE" else 0. for name in names}
    observations, closes, native_count, fk = collect_native(paths["native"], paths["trace"], model, intervals, expected_open)
    if native_count != row["native_audit"]["native_samples"]:
        raise ValueError("Native sample count differs from final metadata")
    audits = [reconstruct_attempt(attempt, samples, interval, expected_open, fk)
        for attempt, samples, interval in zip(telemetry["attempts"], observations, intervals)]
    verify_close_linkage(closes, audits)
    if any(digest(path) != before[name] for name, path in paths.items()):
        raise ValueError("An audit input changed during this read-only reconstruction")
    return {"format": FORMAT, "accepted_alignment_evidence": True,
        "summary_sha256": before["summary"], "protocol_sha256": before["protocol"],
        "script_sha256": digest(__file__), "source": source,
        "inputs": {name: {"path": str(path), "sha256": before[name], "bytes": path.stat().st_size}
            for name, path in paths.items()}, "inputs_unchanged_during_audit": True,
        "case": row["path"], "actual_hand_instances": count, "alignments": audits,
        "native_samples": native_count, "native_timestep_s": DT,
        "mujoco_version": mujoco.__version__,
        "all_native_open_outputs_and_recorded_alignment_rows_consistent": True,
        "all_recorded_cached_wrist_fk_checks_consistent": True,
        "all_alignment_termination_and_close_transitions_consistent": True,
        "task_success": row["success"], "held_lift": row["held_lift"],
        "first_global_native_cached_wrist_fk_excluded": 1,
        "scope": "Narrow ALIGN policy and output/timing consistency only. Cached wrist FK uses previous integrated raw-trace arm encoders; the first global native sample lacks that predecessor and is excluded. Actual native 22 hand motor inputs independently check open output. Logical arm target/completion uses trace arm_command, not gravity-compensated native arm motor ctrl. Target choice/planning, gravity compensation mapping, pressure/QP budget, contact shear, probe confirmation, held-object success and whole-task scoring are not independently audited here. Failed ALIGN remains a failed alignment, not unknown success. No integration or live state modification is performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Alignment audit requires a new output receipt")
    try:
        receipt, exit_code = audit_directory(args.directory), 0
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        receipt, exit_code = {"format": FORMAT, "accepted_alignment_evidence": False,
            "script_sha256": digest(__file__), "error": str(error)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"accepted_alignment_evidence": receipt["accepted_alignment_evidence"],
        "receipt": str(args.output)}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
