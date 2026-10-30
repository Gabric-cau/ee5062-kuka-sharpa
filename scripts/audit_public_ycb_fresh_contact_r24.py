#!/usr/bin/env python3
"""Narrow R24 source, current-contact reconstruction and recorded QP linkage audit."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

import audit_public_ycb_settled_alignment as stream


PROJECT = Path(__file__).resolve().parents[1]
FORMAT = "r24_fresh_contact_source_reconstruction_and_qp_linkage_audit_v1"
RUNNER = "public_ycb_fresh_contact_tracking_development_r24"
HAND = "FreshContactTrackingHand"
HOOK = "scripts/fresh_contact_tracking_controller_hooks.py"
R13 = "scripts/develop_public_ycb_joint_admittance_r13.py"
BASELINE = "results/public_ycb_actual_wrist_probe_20261022_r23_bias10_close32_seed3414_retry1"
BASELINE_SHA256 = "6b87f56530c87666b34eb45b177ed4b7732d7927f2a944138bef390ea260a210"
DERIVED_SHA256 = "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"
HOOK_SHA256 = "e64bff5553e42aa263c86ebbc98127296b24e288668ffacc326379f7c20c16e8"
R13_SHA256 = "ea37244e99045959c6ad841477b2d275823c1599f1ac9624af9e94c16bf866ed"
ACTIVE = ("thumb", "middle", "ring")
NAMES = {finger: ["left_" + finger + "_" + suffix for suffix in
    (("MCP_FE", "IP") if finger == "thumb" else ("MCP_FE", "PIP", "DIP"))] for finger in ACTIVE}
digest, safe_file, finite, equal, array = stream.digest, stream.safe_file, stream.finite, stream.equal, stream.array


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ast_sha_for_version(tree, version):
    """Reproduce the recorded interpreter's AST dump for the existing no-type-parameter code."""
    tree = copy.deepcopy(tree)
    if tuple(version[:2]) < (3, 12):
        for node in ast.walk(tree):
            if "type_params" in node._fields:
                require(not getattr(node, "type_params", []), "Unexpected typed method parameters")
                node._fields = tuple(name for name in node._fields if name != "type_params")
    options = {"show_empty": True} if tuple(version[:2]) < (3, 13) and "show_empty" in (ast.dump.__kwdefaults__ or {}) else {}
    return sha_text(ast.dump(tree, **options))


def method_ast(source, class_name, method_name):
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def source_contract(hook_source, r13_source, version):
    original = method_ast(r13_source, "JointAdmittanceProbeHand", "regulate")
    parsed = ast.parse(hook_source)
    fragments = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                "_OBSOLETE_IF", "_FRESH_TELEMETRY", "_REBUILT_TELEMETRY"):
            parse_call = next(child for child in ast.walk(node.value) if isinstance(child, ast.Call)
                and ast.unparse(child.func) == "ast.parse")
            fragments[node.targets[0].id] = ast.parse(ast.literal_eval(parse_call.args[0])).body[0]
    require(len(fragments) == 3, "Missing exact R24 method fragments")
    expected_if = ast.parse('''if self.joint_activation_s is not None and changed:
    self.joint_target.update(feasible=False, reason="CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE")
    self._mode("HOLD_INVALID", data, self.joint_target["reason"])
    return
''').body[0]
    require(ast.dump(fragments["_OBSOLETE_IF"]) == ast.dump(expected_if), "Removed branch differs")
    transformed = copy.deepcopy(original)
    matches = [i for i, node in enumerate(transformed.body) if ast.dump(node) == ast.dump(expected_if)]
    require(len(matches) == 1, "Original method has no unique contact-change branch")
    index = matches[0]
    transformed.body[index] = copy.deepcopy(fragments["_FRESH_TELEMETRY"])
    transformed.body.append(copy.deepcopy(fragments["_REBUILT_TELEMETRY"]))
    restored = copy.deepcopy(transformed)
    restored.body[index] = copy.deepcopy(expected_if)
    restored.body.pop()
    require(ast.dump(restored) == ast.dump(original), "Complete original regulate AST was not restored")
    wrapper = method_ast(hook_source, HAND, "regulate")
    expected_wrapper = ast.parse("def regulate(self, readings, dt, carrying):\n    return _fresh_regulate(self, readings, dt, carrying)\n").body[0]
    require(ast.dump(wrapper) == ast.dump(expected_wrapper), "Wrapper does more than delegate")
    lines = hook_source.splitlines(keepends=True)
    return {"original_method_ast_sha256": ast_sha_for_version(original, version),
        "transformed_method_ast_sha256": ast_sha_for_version(transformed, version),
        "restored_method_ast_sha256": ast_sha_for_version(restored, version),
        "wrapper_method_source_sha256": sha_text("".join(lines[wrapper.lineno - 1:wrapper.end_lineno])),
        "entire_original_method_restored": True, "wrapper_is_exact_thin_delegate": True}


def check_runtime_binding(hand, binding, contract):
    proof, tracking = hand["method_binding"], binding["tracking"]
    require(proof["format"] == "r24_fresh_contact_method_binding_v1", "Unexpected inner method binding")
    require(binding["controller_class"] == HAND and binding["additional_bias_limit_rad"] == .10, "Actual hand class/budget differs")
    for key in ("wrapper_source_file", "wrapper_source_file_sha256"):
        require(proof[key] == {"wrapper_source_file": HOOK, "wrapper_source_file_sha256": HOOK_SHA256}[key], "Wrapper file binding differs")
    for key in ("original_method_ast_sha256", "transformed_method_ast_sha256", "restored_method_ast_sha256", "wrapper_method_source_sha256"):
        require(proof[key] == contract[key], "Recorded inner/wrapper AST binding differs: " + key)
    require(proof["original_r13_source_sha256"] == R13_SHA256
        and proof["inner_compilation_filename"] == "<fresh_contact_tracking_policy>"
        and proof["ast_hash_format"] == "sha256(ast.dump(tree) without location attributes)", "Inner source interpretation differs")
    for flag in ("wrapper_is_thin_delegate", "inner_same_closure_as_original", "inner_original_class_cell_retained",
            "inner_other_globals_match_except_allocator_and_warning_registry", "inner_allocator_is_shared_normal",
            "wrapper_allocator_metadata_matches_inner"):
        require(proof[flag] is True, "Recorded inner binding check failed: " + flag)
    require(tracking["method_source_file"] == HOOK and tracking["method_source_file_sha256"] == HOOK_SHA256
        and tracking["method_source_sha256"] == contract["wrapper_method_source_sha256"]
        and tracking["allocator_is_shared_normal"] is True, "Runtime wrapper source/allocator differs")
    for flag in ("same_code_as_original", "same_closure_as_original", "other_globals_match_except_warning_registry"):
        require(tracking[flag] is False, "New wrapper was incorrectly represented as an original method")
    for flag in ("same_code_as_original", "same_closure_as_original", "other_globals_match_except_warning_registry",
            "allocator_is_shared_normal", "r13_proxy_other_members_match"):
        require(binding["gate_only"][flag] is True, "Retained gate allocator binding differs")
    for flag in ("writer_is_frozen_r16", "after_step_is_frozen_r16", "admit_probe_is_frozen_r16"):
        require(binding[flag] is True, "Retained R16 method differs")
    require(binding["unchanged_methods"]["regulate"] is False
        and all(value is True for key, value in binding["unchanged_methods"].items() if key != "regulate"), "Undeclared downstream method differs")


def current_groups(patches):
    """Reconstruct only the declared same-body grouping and pressure arithmetic."""
    groups = []
    for patch in sorted(patches, key=lambda p: (p["finger"], p["hand_body_name"], *p["position_world_m"], p["raw_input_index"])):
        point = array(patch["position_world_m"], (3,), "current patch point")
        normal = array(patch["normal_on_object_world"], (3,), "current patch normal")
        normal = normal / np.linalg.norm(normal)
        group = next((g for g in groups if g["finger"] == patch["finger"] and g["hand_body_name"] == patch["hand_body_name"]
            and all(normal @ member[1] >= .9 and np.linalg.norm(point - member[2]) <= .015 for member in g["members"])), None)
        if group is None:
            group = {"finger": patch["finger"], "hand_body_name": patch["hand_body_name"], "members": []}
            groups.append(group)
        group["members"].append((patch, normal, point))
    output = []
    for group in groups:
        members = group.pop("members")
        weights = [max(.05, finite(member[0]["measured_pressure_n"], "patch pressure")) for member in members]
        normal = np.average([member[1] for member in members], axis=0, weights=weights)
        normal /= np.linalg.norm(normal)
        output.append(dict(group, raw_input_indices=[member[0]["raw_input_index"] for member in members],
            position_world_m=np.average([member[2] for member in members], axis=0, weights=weights).tolist(),
            normal_on_object_world=normal.tolist(),
            measured_normal_n=sum(member[0]["measured_pressure_n"] for member in members),
            target_normal_n=sum(member[0]["target_normal_n"] for member in members)))
    return output


def same_array(value, expected, name, tolerance=1e-10):
    expected = np.asarray(expected, dtype=float)
    require(np.allclose(array(value, expected.shape, name), expected, atol=tolerance, rtol=0.), name + " differs")


def target_sample(target):
    patches = target["patches"]
    expected = [p["input_index"] for p in target["association"]["patches"] if p["finger"] in ACTIVE and p["measured_pressure_n"] >= .05]
    require([p["raw_input_index"] for p in patches] == expected, "Target raw indices differ from current association")
    associated = {p["input_index"]: p for p in target["association"]["patches"]}
    for patch in patches:
        source = associated[patch["raw_input_index"]]
        require(patch["finger"] == source["finger"] and patch["input_index"] == source["input_index"], "Current patch identity differs")
        for key in ("position_world_m", "normal_on_object_world", "measured_pressure_n"):
            same_array(patch[key], source[key], "Current association " + key)
    # Full matcher reconstruction is available when every retained contact is represented.
    count = sum(p["pressure_n"] >= .05 for p in target["association"]["contact_association"])
    require(count == len(patches), "This narrow audit requires all >=.05N sample contacts in the current target")
    return sorted([{"finger": p["finger"], "hand_body_name": p["hand_body_name"], "position_world_m": p["position_world_m"],
        "normal": (-np.asarray(p["normal_on_object_world"])).tolist(), "pressure_n": p["measured_pressure_n"]} for p in patches],
        key=lambda p: (p["finger"], p["hand_body_name"], *p["position_world_m"]))


def sample_change(current, previous):
    if previous is None:
        return False, []
    if len(current) != len(previous):
        return True, ["count"]
    reasons = set()
    for now, old in zip(current, previous):
        if (now["finger"], now["hand_body_name"]) != (old["finger"], old["hand_body_name"]):
            reasons.add("identity")
        if np.linalg.norm(np.asarray(now["position_world_m"]) - old["position_world_m"]) > .001:
            reasons.add("position")
        if np.asarray(now["normal"]) @ old["normal"] < .99:
            reasons.add("normal")
        if abs(now["pressure_n"] - old["pressure_n"]) > .1 + .2 * old["pressure_n"]:
            reasons.add("pressure")
    return bool(reasons), sorted(reasons)


def check_reconstruction(event):
    target, solves = event["current_target"], event["joint_solves"]
    telemetry = target["fresh_contact_tracking"]
    require(target["feasible"] is True and target["association"]["accepted"] is True
        and target["runtime_target_state_used"] is False, "Reconstruction lacks accepted current inputs")
    equal(telemetry["timestamp_s"], target["timestamp_s"], "Target/telemetry clock")
    equal(event["target_age_s"], event["timestamp_s"] - target["timestamp_s"], "Target write age", 1e-8)
    require(telemetry["format"] == "r24_current_contact_tracking_v1"
        and telemetry["status"] == "CURRENT_CONTACT_JACOBIANS_REBUILT", "Incomplete reconstruction telemetry")
    require(type(telemetry["changed_from_previous_sample"]) is bool and type(telemetry["activated_before_current_solve"]) is bool,
        "Matcher/activation telemetry is not Boolean")
    sample = target_sample(target)
    require(telemetry["current_contact_sample_count"] == len(sample)
        and telemetry["current_target_raw_input_indices"] == [p["raw_input_index"] for p in target["patches"]], "Current sample/raw-index telemetry differs")
    groups = current_groups(target["patches"])
    require(telemetry["rebuilt_group_count"] == len(groups)
        and [s["finger"] for s in solves] == list(ACTIVE), "Rebuilt group/finger count differs")
    expected_telemetry = []
    for solve in solves:
        finger = solve["finger"]
        expected = [g for g in groups if g["finger"] == finger]
        require(solve["joint_names"] == NAMES[finger] and len(solve["groups"]) == len(expected), "Current finger joints/groups differ")
        for actual_group, expected_group in zip(solve["groups"], expected):
            for key in ("finger", "hand_body_name", "raw_input_indices"):
                require(actual_group[key] == expected_group[key], "Current group identity/index differs")
            for key in ("position_world_m", "normal_on_object_world", "measured_normal_n", "target_normal_n"):
                same_array(actual_group[key], expected_group[key], "Current group " + key)
        shape = (len(expected), len(NAMES[finger]))
        array(solve["compression_jacobian_m_rad"], shape, "Current Jacobian dimensions")
        errors = np.array([g["target_normal_n"] - g["measured_normal_n"] for g in expected])
        same_array(solve["pressure_errors_n"], errors, "Current group pressure errors")
        same_array(solve["desired_normal_velocity_m_s"], np.clip(errors * .0005, -.001, .001), "Current desired normal velocity")
        expected_telemetry.append({"finger": finger, "joint_names": NAMES[finger],
            "raw_input_indices_by_group": [g["raw_input_indices"] for g in expected], "jacobian_shape": list(shape)})
    require(telemetry["rebuilt_finger_solves"] == expected_telemetry, "Reconstruction metadata differs from actual solves")
    return sample


def check_qp_linkage(event, bias, capture):
    fresh = abs(event["timestamp_s"] - event["current_target"]["timestamp_s"]) < 1e-9
    require(bias["format"] == "r17_bias_budget_write_v1" and bias["additional_bias_limit_rad"] == .10
        and bias["fresh_target"] is fresh, "R16 same-write freshness/budget differs")
    if not bias["per_finger"]:
        require(event["phase"] in ("RELEASE", "VERIFY", "RECOVER") or bias["final_safety_override"] is True,
            "Current reconstruction unexpectedly bypassed its R16 writer")
        return {"current": False, "qp_fingers": 0, "rejected_qp_fingers": 0}
    if not fresh:
        require(all(f.get("nominal_qp") is None for f in bias["per_finger"]), "Stale retained reconstruction reached nominal QP")
        return {"current": False, "qp_fingers": 0, "rejected_qp_fingers": 0}
    equal(event["target_age_s"], 0., "Current QP target age")
    require([f["finger"] for f in bias["per_finger"]] == list(ACTIVE), "Current R16 finger records missing")
    rejected = 0
    for solve, finger in zip(event["joint_solves"], bias["per_finger"]):
        qp = finger["nominal_qp"]
        require(qp is not None and finger["joint_names"] == solve["joint_names"], "Current Jacobian not supplied to the R14 QP")
        for holder in (finger, qp):
            same_array(holder["compression_jacobian_m_rad"], solve["compression_jacobian_m_rad"], "Current Jacobian/QP linkage")
            same_array(holder["desired_normal_velocity_m_s"], solve["desired_normal_velocity_m_s"], "Current desired/QP linkage")
        low, high = [], []
        for name in solve["joint_names"]:
            joint = finger["joints"][name]
            equal(joint["fixed_capture_bias_rad"], capture["captured_command_minus_actual_rad"][name], "Fixed loaded bias")
            equal(joint["capture_timestamp_s"], capture["timestamp_s"], "Original activation capture")
            require(joint["additional_bias_limit_rad"] == .10, "Added bias budget changed")
            previous = joint["previous_reference_rad"]
            low.append((joint["intersection_rad"][0] - previous) / event["dt_s"])
            high.append((joint["intersection_rad"][1] - previous) / event["dt_s"])
        same_array(qp["lower_velocity_rad_s"], low, "Current box lower velocity", 1e-8)
        same_array(qp["upper_velocity_rad_s"], high, "Current box upper velocity", 1e-8)
        require(type(qp["accepted"]) is bool, "QP acceptance must be Boolean")
        if qp["accepted"]:
            require(finger["mode"] == "NOMINAL" and finger["nominal_direction_satisfied"] is True, "Accepted current QP lost direction verification")
        else:
            rejected += 1
            require(finger["mode"] == "BIAS_RECOVERY" and bias["recovery_active"] is True
                and bias["preload_inhibited"] is True, "Rejected current QP did not retain recovery")
            if bias["admission_before"]["probe_active"]:
                require(bias["abort_reason"] == "PROBE_BIAS_RECOVERY", "Infeasible active probe did not retain abort")
    return {"current": True, "qp_fingers": 3, "rejected_qp_fingers": rejected}


def clock(value):
    value = finite(value, "write clock")
    tick = round(value / .002)
    equal(value, tick * .002, "Native write clock", 1e-7)
    return tick


def check_result_binding(row, result):
    # R19 arm metadata and held_lift are summary-only; result carries later transforms.
    for key in ("fresh_contact_tracking", "attempts", "success", "actual_wrist_probe_source_transform",
            "alignment_source_transform"):
        require(key in result and key in row and result[key] == row[key], "Result/metadata differs or lacks field: " + key)


def audit_streams(joint_path, bias_path, hands):
    expected = iter((i, event) for i, hand in enumerate(hands) for event in hand["events"])
    grouped = iter(itertools.groupby(stream.json_array(bias_path), key=lambda e: clock(e["timestamp_s"])))
    current_tick, current_bias = -1, []
    reports = [{"retained_records": 0, "current_solve_records": 0, "retained_without_current_qp_records": 0,
        "changed_current_samples": 0, "same_count_pressure_changes": 0, "qp_fingers": 0,
        "rejected_qp_fingers": 0, "activation_captures": 0, "changed_examples": []} for _ in hands]
    samples, last_keys, captures = [None] * len(hands), [None] * len(hands), []
    for event in stream.json_array(joint_path):
        if event.get("event") == "ACTIVATION_CAPTURE":
            captures.append(event)
        telemetry = event.get("current_target", {}).get("fresh_contact_tracking")
        if event.get("event") != "NOMINAL_REFERENCE_WRITE" or telemetry is None:
            continue
        pair = next(expected, None)
        require(pair is not None and pair[1] == telemetry, "Actual writer telemetry differs from serialized hand records")
        i, _ = pair
        report = reports[i]
        report["retained_records"] += 1
        sample = check_reconstruction(event)
        tick = clock(event["timestamp_s"])
        while current_tick < tick:
            pair = next(grouped, None)
            require(pair is not None, "Bias stream ended before a reconstruction write")
            current_tick, rows = pair
            current_bias = list(rows)
        require(current_tick == tick, "Current reconstruction lacks same-clock R16 writer")
        choices = [k for k, b in enumerate(current_bias) if b["phase"] == event["phase"]
            and abs(b["dt_s"] - event["dt_s"]) < 1e-9]
        require(bool(choices), "Current reconstruction lacks matching R16 write interval/phase")
        bias = current_bias.pop(choices[0])
        capture = next((c for c in reversed(captures) if c["timestamp_s"] <= telemetry["timestamp_s"] + 1e-9), None)
        require(capture is not None, "Current reconstruction lacks loaded activation capture")
        linkage = check_qp_linkage(event, bias, capture)
        report["qp_fingers"] += linkage["qp_fingers"]
        report["rejected_qp_fingers"] += linkage["rejected_qp_fingers"]
        if not linkage["current"]:
            report["retained_without_current_qp_records"] += 1
            continue
        key = (telemetry["timestamp_s"], sha_text(json.dumps(telemetry, sort_keys=True)))
        if key == last_keys[i]:
            continue
        last_keys[i] = key
        report["current_solve_records"] += 1
        changed, reasons = sample_change(sample, samples[i])
        require(telemetry["changed_from_previous_sample"] is changed, "Original sample matcher change flag differs")
        require(telemetry["activated_before_current_solve"] is (samples[i] is not None), "Loaded activation was recaptured or reset")
        if changed:
            report["changed_current_samples"] += 1
            report["same_count_pressure_changes"] += int("pressure" in reasons and "count" not in reasons)
            if len(report["changed_examples"]) < 32:
                report["changed_examples"].append({"timestamp_s": telemetry["timestamp_s"], "change_reasons": reasons,
                    "sample_count": len(sample), "groups": telemetry["rebuilt_group_count"],
                    "all_current_qps_accepted": linkage["rejected_qp_fingers"] == 0, "phase": event["phase"]})
        samples[i] = sample
    require(next(expected, None) is None, "Serialized reconstruction records lack actual writer events")
    require(len(captures) == len(hands), "Loaded activation capture count differs from actual hands")
    for i, hand in enumerate(hands):
        reports[i]["activation_captures"] = 1
        require(reports[i]["retained_records"] == hand["accepted_reconstruction_count"]
            and sum(e["changed_from_previous_sample"] for e in hand["events"]) == hand["changed_current_sample_count"], "Serialized reconstruction counts differ")
    return reports


def audit_directory(directory, expected_summary_sha256):
    directory = Path(directory).resolve(strict=True)
    require(isinstance(expected_summary_sha256, str) and len(expected_summary_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_summary_sha256), "Explicit final summary SHA256 required")
    paths = {name: safe_file(directory, filename) for name, filename in
        (("summary", "summary.json"), ("protocol", "protocol.json"), ("archive", "frozen_source.zip"))}
    require(digest(paths["summary"]) == expected_summary_sha256, "Final summary SHA256 differs")
    summary, protocol = (json.loads(paths[name].read_text()) for name in ("summary", "protocol"))
    require(summary["complete"] is True and summary["runner_errors"] == 0 and summary["protocol"] == protocol
        and summary["runner_revision"] == RUNNER and protocol["format"] == RUNNER
        and protocol["controller_class"] == HAND and len(summary["rows"]) == 1, "Expected finalized R24 evidence")
    before = protocol["source_hashes_before"]
    require(len(before) == 186 and before == protocol["source_hashes_after"]
        and protocol["source_changed_during_run"] is False and before.get(HOOK) == HOOK_SHA256
        and before.get(R13) == R13_SHA256, "Frozen R24 source manifest differs")
    with zipfile.ZipFile(paths["archive"]) as archive:
        require(len(archive.namelist()) == 186 and set(archive.namelist()) == set(before), "R24 source ZIP manifest differs")
        for relative, expected in before.items():
            require(hashlib.sha256(archive.read(relative)).hexdigest() == expected
                and digest(safe_file(PROJECT, relative)) == expected, "Archived/current frozen source differs: " + relative)
        baseline = json.loads(archive.read(BASELINE + "/summary.json"))
        inherited = baseline["protocol"]["source_hashes_before"]
        require(before[BASELINE + "/summary.json"] == BASELINE_SHA256 and len(inherited) == 167
            and all(before.get(k) == v for k, v in inherited.items()), "Inherited R23 sources differ")
        hook_source, r13_source = archive.read(HOOK).decode(), archive.read(R13).decode()
    row = summary["rows"][0]
    require(row["runner_error"] is None and row["kind"] == "ycb_can" and row["seed"] == 3414
        and row["effective_hand_instances"] == len(row["attempts"]) == 2
        and row["effective_hand_controller_types"] == [HAND] * 2 and row["effective_hand_budgets_rad"] == [.10] * 2
        and row["derived_controller_sha256"] == DERIVED_SHA256 and row["scene_unchanged_since_load"] is True,
        "Actual case/hand/arm identity differs")
    case = safe_file(directory, str(Path(row["path"]) / "metadata.json")).parent
    for name, filename in (("metadata", "metadata.json"), ("result", "result.json"),
            ("controller", "derived_controller.py"), ("scene", "scene.xml")):
        paths[name] = safe_file(case, filename)
    for name in ("joint_admittance", "bias_recovery"):
        artifact = row["artifacts"][name]
        paths[name] = safe_file(case, artifact["path"])
        require(digest(paths[name]) == artifact["sha256"] and paths[name].stat().st_size == artifact["bytes"], "Raw artifact hash/size differs: " + name)
    paths["json_stream_loader"] = Path(stream.__file__).resolve()
    require(digest(paths["json_stream_loader"]) == before["scripts/audit_public_ycb_settled_alignment.py"], "Bound stream loader differs")
    hashes = {name: digest(path) for name, path in paths.items()}
    result, metadata = (json.loads(paths[name].read_text()) for name in ("result", "metadata"))
    require(metadata == row and hashes["controller"] == DERIVED_SHA256
        and hashes["scene"] == row["scene_sha256_before_load"] == row["scene_sha256_at_finalize"], "Final metadata/arm/scene binding differs")
    check_result_binding(row, result)
    require(all(row[key] == baseline["rows"][0][key] for key in ("actual_wrist_probe_source_transform",
        "alignment_source_transform", "arm_source_transform")), "R23 arm transformation declarations differ")
    declaration = protocol["r24_fresh_contact_tracking_experiment"]
    require(declaration["arm_source_unchanged_from_r23"] is True and declaration["physics_and_strict_task_gates_unchanged"] is True
        and declaration["additional_bias_limit_rad"] == .10 and declaration["close_seconds"] == 32.
        and declaration["arm_interpolation_seconds"] == 16., "Undeclared R24 policy change")
    hands = row["fresh_contact_tracking"]["hands"]
    bindings = row["shared_normal_runtime_bindings"]
    require(row["fresh_contact_tracking"]["format"] == "r24_current_contact_tracking_v1"
        and len(hands) == len(bindings) == 2, "Missing actual hand reconstruction records")
    contracts = []
    for hand, binding in zip(hands, bindings):
        contract = source_contract(hook_source, r13_source, binding["python_version"])
        check_runtime_binding(hand, binding, contract)
        contracts.append(contract)
    reports = audit_streams(paths["joint_admittance"], paths["bias_recovery"], hands)
    require(any(report["changed_current_samples"] for report in reports), "No actual changed-contact reconstruction was observed")
    require(all(digest(path) == hashes[name] for name, path in paths.items()), "Audit input changed while reading")
    rejected = sum(report["rejected_qp_fingers"] for report in reports)
    return {"format": FORMAT, "script_sha256": digest(Path(__file__)), "summary_sha256": expected_summary_sha256,
        "accepted_fresh_contact_evidence": True,
        "checks": {"all_evidence_consistent": True, "all_frozen_source_and_recorded_method_bindings_consistent": True,
            "all_current_contact_reconstruction_and_qp_linkages_consistent": True},
        "source": {"source_count": 186, "inherited_r23_source_count": 167, "all_archived_and_current_hashes_match": True,
            "arm_source_unchanged_from_r23": True, "method_contracts": contracts},
        "inputs": {name: {"path": str(path), "sha256": hashes[name], "bytes": path.stat().st_size} for name, path in paths.items()},
        "hands": reports, "observed_infeasible_qp_recovery": rejected > 0,
        "infeasible_qp_finger_records": rejected,
        "task_success_recorded": row["success"], "held_lift_recorded": row["held_lift"],
        "scope": "Source/metadata/current group and recorded QP linkage only. Repeated retained records are not new solves. No independent LP, MuJoCo Jacobian, historical process identity, native safety or strict-task recomputation; accepted evidence is not task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    receipt = audit_directory(args.directory, args.expected_summary_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted_fresh_contact_evidence": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
