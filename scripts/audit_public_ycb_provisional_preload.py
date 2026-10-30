#!/usr/bin/env python3
"""Independent R21 source, command, target and provisional-policy audit."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import audit_public_ycb_shared_normal as r20
import audit_public_ycb_shared_normal_postrun as postrun


base, r17, r18 = r20.base, r20.r17, r20.r18
FORMAT = "public_ycb_provisional_preload_development_r21"
DECLARATION = "r21_provisional_preload_experiment"
SOURCE = "scripts/develop_public_ycb_provisional_preload_r21.py"
AUDITOR = "scripts/audit_public_ycb_provisional_preload.py"
POLICY = "scripts/provisional_preload_controller_hooks.py"
CLASS = "ProvisionalPreloadBudgetHand"
DEFERRED_REASON = "CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE"
AST_FORMAT = "canonical_ast_json_v1_ignore_empty_type_params"
FROZEN_SOURCE_SHA256 = "e4973c8c2dc55a5a043834a641438008b281a0344447a0fb94caf1ac318e97e4"
FROZEN_POLICY_SHA256 = "f3578add08ac32bdc68789778da82ef32963b51737c433d5cce7f9c34cda5991"
POSTRUN_SOURCE = "scripts/audit_public_ycb_shared_normal_postrun.py"
POSTRUN_TEST = "tests/test_public_ycb_shared_normal_postrun_audit.py"
R20_DIRECTORY = "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414"
R20_JOB = "results/jobs/r20_shared_normal_bias10_close32_seed3414_20261022"
R20_SUMMARY = R20_DIRECTORY + "/summary.json"
R20_PROTOCOL = R20_DIRECTORY + "/protocol.json"
R20_FAILED = "results/r20_shared_normal_audit_20261022/receipt.json"
R20_POSTRUN = "results/r20_shared_normal_postrun_audit_20261022/receipt.json"
PINS = {
    R20_SUMMARY: "a47ab784ed6b0879f149333a08e02928b5ae5430ec93af76af67de02ef63dd7b",
    R20_PROTOCOL: "18f7982dc5e2513e7aa9947af9df5e4086634753858518adaff1f3370349707a",
    R20_DIRECTORY + "/verification_receipt.json": "302af5189b5f475fa2903dae49e69cb088da624b807160f083ff4fb3e1537e30",
    R20_DIRECTORY + "/strict_native_verification.json": "0b81b3c048730ccf4a5d46f857afc28c91745b8ddbb47f4602cc7d2239232897",
    R20_FAILED: "aa456ddbcd40accab48be87b650d2c4deaa71917d6838991b7238b9c8c07f740",
    R20_POSTRUN: "7c7848dfa647d312b555f09041d56b9d1a2bbf712f26f49a2d5dd9a8c025d0d0",
    R20_JOB + "/status.json": "ba01295e2222f262f890a687858869b8510d671eb1d3a41599cec7739e562ba8",
    R20_JOB + "/request.json": "01f4d808fc555a56dadc2efd47bcf2474611a5e22f598cdaba1c55efe8e3f0e9",
    POSTRUN_SOURCE: "a9bde9be7497a7302e4eea71ed1bf33d38dd4945b1b72df3abb1cbcf3231e75b",
    POSTRUN_TEST: "95f32ee4602cccb25aee3765230f5ce2bb4bd0d4ef7d81ef259db496d6861e6f",
    r20.AUDITOR: postrun.FROZEN_AUDITOR_SHA256,
    POLICY: FROZEN_POLICY_SHA256,
}
UNCHANGED_METHODS = ("write", "after_step", "admit_probe", "confirm_probe", "held_revalidated", "abort_probe",
    "update_pressure_relief", "angles", "phase_begin", "regulate", "current_preload_observation")
CHANGE = {"pre_motion_preload_reasons." + DEFERRED_REASON: {"before": "blocks_ready",
    "after": "observed_but_deferred_to_unchanged_probe_confirmation_and_held_policy"}}


def hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def canonical_ast(value):
    if isinstance(value, ast.AST):
        return {"node_type": type(value).__name__, "fields": [[name, canonical_ast(child)]
            for name, child in ast.iter_fields(value) if name != "type_params" or child]}
    if isinstance(value, list):
        return [canonical_ast(child) for child in value]
    return value


def ast_sha(tree):
    return hashlib.sha256(json.dumps(canonical_ast(tree), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def runner_preflight(source):
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PREFLIGHT_HASHES" for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError("R21 runner has no unique literal preflight mapping")
    values = {}
    for key, value in zip(assignments[0].value.keys, assignments[0].value.values):
        path = ast.literal_eval(key)
        digest = base.literal_assignment(source, value.id) if isinstance(value, ast.Name) else ast.literal_eval(value)
        if not isinstance(path, str) or path in values or not hex_digest(digest):
            raise ValueError("Invalid, missing or duplicate R21 runner preflight pin")
        values[path] = digest
    return values


def method_source(archive, before, filename, class_name, method_name):
    source = archive.read(filename).decode()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    segment = "".join(source.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])
    return method, {"module": Path(filename).stem, "qualname": class_name + "." + method_name,
        "method_source_sha256": hashlib.sha256(segment.encode()).hexdigest(),
        "method_source_file": filename, "method_source_file_sha256": before[filename]}


def policy_source_proof(archive, before):
    r14_path = "scripts/develop_public_ycb_constrained_preload_r14.py"
    r16_path = "scripts/develop_public_ycb_bias_recovery_r16.py"
    original14, binding14 = method_source(archive, before, r14_path, "ConstrainedPreloadHand", "update_preload")
    original16, binding16 = method_source(archive, before, r16_path, "BiasRecoveryHand", "update_preload")
    _, wrapper = method_source(archive, before, POLICY, CLASS, "update_preload")
    branch = ast.parse('''if not original["sampled_fixed_normal_1g_feasible"]:
    reasons.append("CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE")
''').body[0]
    transformed14 = copy.deepcopy(original14)
    matches = [node for node in transformed14.body if canonical_ast(node) == canonical_ast(branch)]
    if len(matches) != 1:
        raise ValueError("Original R14 pre-motion actual-1g branch is not unique")
    transformed14.body.remove(matches[0])
    transformed16 = copy.deepcopy(original16)
    parent = ast.parse("super().update_preload(data, admission=admission)", mode="eval").body
    calls = [node for node in ast.walk(transformed16) if isinstance(node, ast.Call)
        and canonical_ast(node) == canonical_ast(parent)]
    if (len(calls) != 1 or not isinstance(transformed16.body[0], ast.Assign)
            or transformed16.body[0].value is not calls[0]):
        raise ValueError("Original R16 preload parent call is not unique at its original position")
    transformed16.body[0].value = ast.parse(
        "_provisional_r14_update_preload(self, data, admission=admission)", mode="eval").body
    paths = {}
    for name, filename, cls, method in (("tracking", "scripts/develop_public_ycb_joint_admittance_r13.py",
            "JointAdmittanceProbeHand", "regulate"), ("gate_only", r14_path, "ConstrainedPreloadHand", "current_preload_observation")):
        _, item = method_source(archive, before, filename, cls, method)
        paths[name] = {key: item[key] for key in ("method_source_sha256", "method_source_file", "method_source_file_sha256")}
        paths[name]["method_name"] = method
    return {"ast_hash_format": AST_FORMAT, "update_preload": wrapper,
        "r14_parent": {"original": binding14, "original_ast_sha256": ast_sha(original14),
            "transformed_ast_sha256": ast_sha(transformed14), "deleted_reason_if_count": 1},
        "r16_recovery_wrapper": {"original": binding16, "original_ast_sha256": ast_sha(original16),
            "transformed_ast_sha256": ast_sha(transformed16), "replaced_parent_call_count": 1},
        "allocator_paths": paths}


def check_protocol(protocol, baseline):
    prior, declaration = baseline["protocol"], protocol[DECLARATION]
    if (protocol["format"] != FORMAT or protocol["runner_revision"] != FORMAT
            or protocol["controller_class"] != CLASS or protocol["admission_policy_revision"] != FORMAT
            or protocol["target_allocator_revision"] != r20.FORMAT
            or declaration["format"] != "r21_provisional_preload_development_protocol_v1"
            or declaration["runner_revision"] != FORMAT or declaration["controller"] != CLASS
            or declaration["admission_policy_revision"] != FORMAT
            or declaration["target_allocator_revision"] != r20.FORMAT or declaration["changed_parameters"] != CHANGE):
        raise ValueError("Unknown, relabeled or nonisolated R21 admission experiment")
    if ("r20_shared_normal_experiment" in protocol
            or protocol["inherited_r20_shared_normal_experiment"] != prior["r20_shared_normal_experiment"]
            or declaration["active_r20_declaration"] != prior["r20_shared_normal_experiment"]
            or declaration["inherited_r20_declaration"] != prior["r20_shared_normal_experiment"]
            or declaration["active_r19_declaration"] != prior["r20_shared_normal_experiment"]["active_r19_declaration"]
            or declaration["inherited_r20_force_control"] != prior["force_control"]
            or protocol["inherited_r20_force_control"] != prior["force_control"]):
        raise ValueError("R21 historical R20 allocator/control declaration differs")
    for key in r20.UNCHANGED_CONTRACTS:
        if protocol[key] != prior[key]:
            raise ValueError("R21 changed an original physical/control/task contract: " + key)
    for key, expected in {"additional_bias_limit_rad": .10, "close_seconds": 32.,
            "arm_interpolation_seconds": 16., "native_timestep_s": .002}.items():
        base.equal(declaration[key], expected, "R21 declared " + key)
    for key in ("actual_fixed_normal_1g_observation_retained", "actual_fixed_normal_1g_confirmation_and_held_policy_unchanged",
            "all_other_preload_reasons_and_recovery_inhibition_unchanged", "target_allocator_and_write_mechanics_unchanged"):
        if declaration[key] is not True:
            raise ValueError("R21 removed an unchanged actual observation/gate: " + key)
    if (declaration["r20_comparison_summary_sha256"] != PINS[R20_SUMMARY]
            or declaration["policy_hook_source_sha256"] != FROZEN_POLICY_SHA256
            or declaration["deferred_pre_motion_reason"] != DEFERRED_REASON
            or declaration["runtime_target_pose_or_qpos_added"] is not False
            or declaration["physics_force_and_strict_task_gates_modified"] is not False
            or declaration["baseline_task_success"] is not False or declaration["baseline_held_lift"] is not False
            or declaration["baseline_original_command_audit_passed"] is not False
            or declaration["baseline_postrun_extension_format"] != postrun.FORMAT
            or declaration["baseline_child_exit_observed"] is not True or declaration["baseline_child_returncode"] != 0):
        raise ValueError("R21 policy, original failure or observed baseline exit provenance differs")
    if (protocol["constrained_preload"]["continuous_seconds"] != .25
            or protocol["load_transfer_probe"]["probe_distance_m"] != .005
            or protocol["load_transfer_probe"]["confirmation_seconds"] != .25
            or protocol["load_transfer_probe"]["maximum_remaining_table_weight_fraction"] != .2):
        raise ValueError("R21 changed the original probe/continuous/table-unloading thresholds")


def frozen_inputs(directory, protocol):
    if (base.digest(r20.__file__) != postrun.FROZEN_AUDITOR_SHA256
            or base.digest(postrun.__file__) != PINS[POSTRUN_SOURCE]
            or base.digest(r17.__file__) != r20.historical.FROZEN_R17_CORE_SHA256
            or base.digest(r18.__file__) != r20.historical.FROZEN_R18_CORE_SHA256):
        raise ValueError("An inherited frozen native/target/initializer audit core changed")
    budget, rates, source = r17.frozen_inputs(directory, protocol)
    base.equal(budget, .10, "R21 immutable instance budget")
    before = protocol["source_hashes_before"]
    pins = dict(PINS, **{SOURCE: FROZEN_SOURCE_SHA256, r20.PURE: r20.FROZEN_PURE_SHA256,
        r20.HOOK: r20.FROZEN_HOOK_SHA256, r20.SOURCE: r20.FROZEN_SOURCE_SHA256})
    if (any(before.get(key) != value for key, value in pins.items())
            or before[AUDITOR] != base.digest(__file__)):
        raise ValueError("R21 execution/allocator/baseline/auditor source identity differs")
    preflight = protocol[DECLARATION]["preflight_input_sha256"]
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(before):
            raise ValueError("R21 source ZIP has missing, duplicated or unmanifested entries")
        for key, expected in before.items():
            if hashlib.sha256(archive.read(key)).hexdigest() != expected:
                raise ValueError("R21 source archive bytes differ: " + key)
        runner = archive.read(SOURCE)
        if (preflight != runner_preflight(runner) or any(before.get(key) != value for key, value in preflight.items())
                or not all(preflight.get(key) == value for key, value in PINS.items())):
            raise ValueError("R21 complete declared preflight closure differs from its frozen runner")
        archived = {key: json.loads(archive.read(key)) for key in PINS if key.endswith(".json")}
        baseline, prior = archived[R20_SUMMARY], archived[R20_PROTOCOL]
        if (baseline["complete"] is not True or baseline["runner_errors"] or baseline["protocol"] != prior
                or prior["format"] != r20.FORMAT or prior["source_changed_during_run"] is not False
                or len(prior["source_hashes_before"]) != 110
                or prior["source_hashes_before"] != prior["source_hashes_after"]
                or any(before.get(key) != value for key, value in prior["source_hashes_before"].items())
                or len(baseline["rows"]) != 1 or len(prior["schedule"]) != 1):
            raise ValueError("R21 does not retain the exact complete 110-source R20 baseline")
        proof = policy_source_proof(archive, before)
    native, strict = (archived[R20_DIRECTORY + "/" + name] for name in
        ("verification_receipt.json", "strict_native_verification.json"))
    extension, failed = archived[R20_POSTRUN], archived[R20_FAILED]
    status, request = archived[R20_JOB + "/status.json"], archived[R20_JOB + "/request.json"]
    initialization = extension["initializer_classification"]
    if (native["complete"] is not True or native["unknowns"] != 0
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or any(value["summary_sha256"] != PINS[R20_SUMMARY] for value in (native, strict, extension))
            or extension["protocol_sha256"] != PINS[R20_PROTOCOL]
            or extension["format"] != postrun.FORMAT or extension["script_sha256"] != PINS[POSTRUN_SOURCE]
            or extension["inherited_frozen_auditor_sha256"] != postrun.FROZEN_AUDITOR_SHA256
            or extension["original_failed_receipt_sha256"] != PINS[R20_FAILED]
            or extension["original_frozen_auditor_succeeded"] is not False
            or extension["extension_inputs_unchanged"] is not True
            or extension["extension_inputs_before"] != extension["extension_inputs_after"]
            or extension["original_failed_audit_overwritten"] is not False
            or extension["original_runtime_source_modified"] is not False
            or any(extension[key] is not True for key in ("all_evidence_consistent", "all_declared_command_bounds_satisfied",
                "all_native_close_spans_within_declared_deadline", "all_recorded_static_target_checks_consistent",
                "all_sampled_hypothetical_target_checks_passed"))
            or initialization["all_contexts_valid"] is not True or initialization["initialization_records"] != 3950
            or initialization["controller_epochs"] != 2 or initialization["original_stream_ordinals_preserved"] is not True
            or failed != {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
                "error": "Unversioned target is not an original caller-gate rejection",
                "script_sha256": postrun.FROZEN_AUDITOR_SHA256}
            or status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or status["returncode"] != 0
            or request["command"][2] != r20.SOURCE):
        raise ValueError("R20 native/strict/postrun/original-failure/normal-exit provenance differs")
    for name, expected in {"BUDGET_RAD": .10, "CLOSE_SECONDS": 32., "ARM_INTERPOLATION_SECONDS": 16.,
            "NATIVE_TIMESTEP_S": .002, "PROTOCOL_FORMAT": FORMAT, "TARGET_ALLOCATOR_REVISION": r20.FORMAT,
            "CONTROLLER_CLASS": CLASS}.items():
        if base.literal_assignment(runner, name) != expected:
            raise ValueError("R21 executed constant differs: " + name)
    if base.digest(r20.allocator.__file__) != before["scripts/tactile_wrench_allocator.py"]:
        raise ValueError("Independent target checker source differs")
    check_protocol(protocol, baseline)
    source.update(r21_source_sha256=before[SOURCE], r21_auditor_sha256=before[AUDITOR],
        provisional_policy_source_sha256=before[POLICY], inherited_r20_source_files=110,
        r20_original_failed_receipt_sha256=PINS[R20_FAILED], r20_postrun_receipt_sha256=PINS[R20_POSTRUN],
        expected_policy_method_bindings=proof)
    return budget, rates, source, baseline


def checked_method_binding(actual, expected):
    if (any(actual[key] != value for key, value in expected.items())
            or not hex_digest(actual["method_bytecode_sha256"])):
        raise ValueError("R21 actual method/source binding differs")


def check_case_metadata(planned, row, protocol, baseline, proof):
    expected = dict(baseline["protocol"]["schedule"][0], runner_revision=FORMAT, controller_class=CLASS,
        controller_revision_scope=protocol["controller_revision_scope"], admission_policy_revision=FORMAT)
    if (planned != expected or any(row[key] != value for key, value in planned.items()) or row["runner_error"]
            or row.get("missing_runtime_controller_binding", False) is not False
            or row.get("missing_runtime_allocator_binding", False) is not False
            or row["target_allocator_revision"] != r20.FORMAT or row["admission_policy_revision"] != FORMAT
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("R21 actual factory, phase, planned case or revision binding differs")
    count, prior = row["effective_hand_instances"], baseline["rows"][0]
    if (type(count) is not int or not 1 <= count <= protocol["maximum_attempts"] or count != len(row["attempts"])
            or row["effective_hand_budgets_rad"] != [.10] * count
            or row["effective_hand_controller_types"] != [CLASS] * count
            or len(row["shared_normal_runtime_bindings"]) != count
            or any(row[key] != prior[key] for key in
                ("derived_controller_sha256", "arm_source_transform", "catalogue_mass_prior_kg"))):
        raise ValueError("R21 actual hand/retry budget, arm source or mass differs")
    for item in row["shared_normal_runtime_bindings"]:
        if (item["controller_class"] != CLASS or item["additional_bias_limit_rad"] != .10
                or item["admission_policy_revision"] != FORMAT
                or any(item[key] is not True for key in
                    ("writer_is_frozen_r16", "after_step_is_frozen_r16", "admit_probe_is_frozen_r16"))
                or not isinstance(item["python_version"], list) or len(item["python_version"]) != 3
                or any(type(value) is not int or value < 0 for value in item["python_version"])
                or set(item["unchanged_methods"]) != set(UNCHANGED_METHODS)
                or any(value is not True for value in item["unchanged_methods"].values())):
            raise ValueError("R21 changed an original writer/confirmation/held/recovery method")
        for path in ("tracking", "gate_only"):
            actual = item[path]
            if (any(actual[key] != value for key, value in proof["allocator_paths"][path].items())
                    or any(actual[key] is not True for key in ("same_code_as_original", "same_closure_as_original",
                        "other_globals_match_except_warning_registry", "allocator_is_shared_normal"))
                    or actual["allocator"] != {"module": "shared_normal_tracking_targets",
                        "qualname": "allocate_tracking_targets", "source_sha256": r20.FROZEN_PURE_SHA256}
                    or not hex_digest(actual["method_bytecode_sha256"])
                    or (path == "gate_only" and actual["r13_proxy_other_members_match"] is not True)):
                raise ValueError("R21 changed a shared-normal allocator path")
        policy = item["admission_policy"]
        if (policy["ast_hash_format"] != AST_FORMAT or policy["deferred_pre_motion_reason"] != DEFERRED_REASON
                or any(policy[key] is not True for key in ("wrapper_is_provisional", "r16_helper_is_provisional",
                    "r14_helper_is_provisional", "original_probe_snapshot_and_actual_fixed_normal_value_retained"))):
            raise ValueError("R21 provisional wrapper/helper identity differs")
        checked_method_binding(policy["update_preload"], proof["update_preload"])
        for name in ("r14_parent", "r16_recovery_wrapper"):
            actual, expected = policy[name], proof[name]
            checked_method_binding(actual["original"], expected["original"])
            count_key = "deleted_reason_if_count" if name == "r14_parent" else "replaced_parent_call_count"
            if (any(actual[key] != value for key, value in expected.items() if key != "original")
                    or type(actual[count_key]) is not int
                    or actual["other_globals_match_except_warning_registry"] is not True):
                raise ValueError("R21 compiled helper AST/one-change/global binding differs")


def audit_policy_rows(preload, probe):
    counts = Counter()
    for row in preload:
        original, policy = row["original_probe_snapshot"], row["provisional_preload_policy"]
        actual = original["sampled_fixed_normal_1g_feasible"]
        ready = base.flag(row, "ready")
        duration = base.finite(row["continuous_seconds"], "preload duration")
        age = None if original["solve_age_s"] is None else base.finite(original["solve_age_s"], "current solve age")
        fresh = base.flag(original, "sampled_solve_current")
        matches = base.flag(original, "cached_measurements_match")
        errors = [abs(base.finite(group["target_n"], "group target")
            - base.finite(group["measured_n"], "group measured normal")) for group in row["current_groups"].values()]
        if (type(actual) is not bool or policy["sampled_fixed_normal_1g_feasible_observed"] is not actual
                or policy["format"] != "provisional_preload_admission_policy_v1"
                or policy["deferred_pre_motion_reason"] != DEFERRED_REASON
                or policy["additional_bias_limit_rad"] != .10 or policy["continuous_preload_seconds"] != .25
                or any(policy[key] is not True for key in ("original_probe_snapshot_unchanged",
                    "recovery_inhibition_unchanged", "actual_fixed_normal_1g_required_for_confirmation",
                    "confirmation_and_held_policy_unchanged", "strict_task_success_unchanged"))
                or not isinstance(row["reasons"], list)
                or any(not isinstance(reason, str) or not reason for reason in row["reasons"])
                or duration < 0. or (age is not None and age < 0.) or DEFERRED_REASON in row["reasons"]):
            raise ValueError("R21 changed/forged the retained actual 1g observation or policy")
        counts["preload_rows"] += 1
        counts["actual_fixed_normal_1g_" + str(actual).lower()] += 1
        if ready:
            if (row["reasons"] or duration < .25 - 1e-9
                    or row.get("r16_recovery_inhibited", False) is not False
                    or not row["current_target_lp"]["feasible"] or not row["current_groups"]
                    or any(error > .35 for error in errors)
                    or original["preprobe_reasons"] or not fresh or not matches or age is None or age > 1e-9):
                raise ValueError("R21 readiness violates an unchanged necessary current gate")
            counts["ready_rows"] += 1
            if not actual:
                counts["ready_rows_with_observed_actual_1g_false"] += 1
    if not counts["preload_rows"]:
        raise ValueError("Missing provisional preload policy rows")
    for row in probe:
        if "confirmation" in row:
            base.flag(row, "confirmation")
        if row.get("confirmation") is True:
            speed = base.finite(row["actual_wrist_speed_m_s"], "probe confirmation speed")
            if (row["sampled_fixed_normal_1g_feasible"] is not True or row["table_unloaded"] is not True
                    or row["actual_probe_endpoint_reached"] is not True or row["accepted_visual_after_probe_start"] is not True
                    or row["motion_safety_reasons"] or not 0. <= speed <= .01):
                raise ValueError("R21 probe confirmation bypassed an unchanged actual-1g/motion/table gate")
            counts["confirmed_probe_records"] += 1
    return {"all_recorded_policy_rows_consistent": True, "counts": dict(counts),
        "scope": "Every preload policy row retains the original Boolean actual-1g observation and .10rad/.25s policy. Readiness receives necessary current-reason/pressure/freshness/duration checks; recovery/abort is independently checked by the inherited native-command core. Positive probe confirmations receive necessary actual-1g/table/endpoint/visual/motion checks. Full pressure-window history, all original eligibility accumulators and raw contact sample comparisons are not independently rebuilt here. Source/AST and unchanged-method bindings establish the declared single admission change, not actual success or complete dynamic gate reconstruction."}


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = (base.checked_path(directory, name) for name in ("summary.json", "protocol.json"))
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    initial = {path: base.digest(path) for path in (summary_path, protocol_path, directory / "frozen_source.zip")}
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or summary["runner_revision"] != FORMAT or summary["controller_class"] != CLASS
            or summary["target_allocator_revision"] != r20.FORMAT or summary["admission_policy_revision"] != FORMAT
            or summary["close_seconds"] != 32. or summary["arm_interpolation_seconds"] != 16.
            or summary["additional_bias_limit_rad"] != .10):
        raise ValueError("Expected complete true-format R21 evidence")
    budget, rates, source, baseline = frozen_inputs(directory, protocol)
    if len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1:
        raise ValueError("R21 requires the isolated same-can same-seed development case")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    check_case_metadata(planned, row, protocol, baseline, source["expected_policy_method_bindings"])
    case = base.checked_path(directory, row["path"])
    paths = {key: base.checked_path(case, filename) for key, filename in (
        ("metadata", "metadata.json"), ("scene", "scene.xml"), ("controller", "derived_controller.py"), ("result", "result.json"))}
    initial.update({path: base.digest(path) for path in paths.values()})
    result = json.loads(paths["result"].read_text())
    if (json.loads(paths["metadata"].read_text()) != row or result["attempts"] != row["attempts"]
            or result["runner_revision"] != FORMAT or result["controller_class"] != CLASS
            or result["target_allocator_revision"] != r20.FORMAT or result["admission_policy_revision"] != FORMAT
            or result["controller_revision"] != "public_ycb_bias_budget_development_r17"
            or result["additional_bias_limit_rad"] != .10 or result["close_seconds"] != 32.
            or result["arm_interpolation_seconds"] != 16. or base.digest(paths["controller"]) != row["derived_controller_sha256"]):
        raise ValueError("R21 metadata, actual class/revision or executed controller binding differs")
    declaration, prior = protocol[DECLARATION], baseline["rows"][0]
    if (row["scene_unchanged_since_load"] is not True or row["scene_sha256_before_load"] != base.digest(paths["scene"])
            or row["scene_sha256_at_finalize"] != base.digest(paths["scene"])
            or row["scene_sha256_before_load"] != prior["scene_sha256_before_load"]
            or declaration["baseline_scene_sha256"] != prior["scene_sha256_before_load"]
            or declaration["baseline_derived_controller_sha256"] != prior["derived_controller_sha256"]):
        raise ValueError("R21 runtime-loaded scene or baseline controller differs")
    model = load_evidence_model(paths["scene"])
    base.equal(model.opt.timestep, .002, "R21 native timestep", 1e-12)
    records = {key: base.checked_stream(case, row["artifacts"][key]) for key in
        ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
    if row["native_audit"]["native_actuation_samples"] != len(records["native"]):
        raise ValueError("R21 native actuation denominator differs")
    spans = r18.native_close_spans(records["native"], protocol["maximum_attempts"])
    receipt = r17.reconstruct_budget_case(records["native"], records["bias_recovery"], records["joint_admittance"],
        base.joint_specs(model, protocol["nominal_profiles"][row["kind"]]), budget_rad=budget, **rates,
        preload_events=records["preload"], probe_events=records["load_transfer_probe"], native_audit=row["native_audit"])
    target_audit, transform = postrun.extended_target_audit()
    targets = target_audit(records["joint_admittance"], records["preload"], row["catalogue_mass_prior_kg"])
    if targets["initializer_classification"]["controller_epochs"] != row["effective_hand_instances"]:
        raise ValueError("R21 initialized hand epochs differ from actual runtime instances")
    receipt.update(case=row["path"], scene_sha256=base.digest(paths["scene"]), metadata_sha256=base.digest(paths["metadata"]),
        result_sha256=base.digest(paths["result"]), derived_controller_sha256=base.digest(paths["controller"]),
        actual_controller_class=CLASS, effective_hand_instances=row["effective_hand_instances"],
        effective_hand_budgets_rad=row["effective_hand_budgets_rad"], native_close_spans=spans,
        all_native_close_spans_within_declared_deadline=True,
        runtime_allocator_path_bindings_checked=2 * row["effective_hand_instances"],
        runtime_method_bytecode_independently_reconstructed=False,
        inputs={key: row["artifacts"][key] for key in records}, static_targets=targets,
        actuation=r17.audit_actuation(records["native"], row["native_audit"]["native_actuation_metadata"], model,
            expected_timing_scope=source["native_actuation_timing_scope"]),
        provisional_policy=audit_policy_rows(records["preload"], records["load_transfer_probe"]),
        task_success=row["success"], held_lift=row["held_lift"],
        task_outcome_scope="Reported actual outcome, not independently rescored by this source/command/target/policy auditor")
    for key, artifact in row["artifacts"].items():
        if key in records:
            path = base.checked_path(case, artifact["path"])
            if path.stat().st_size != artifact["bytes"] or base.digest(path) != artifact["sha256"]:
                raise ValueError("R21 input artifact changed during audit")
    if any(base.digest(path) != expected for path, expected in initial.items()):
        raise ValueError("R21 frozen source or input changed during audit")
    return {"format": "r21_provisional_preload_command_native_target_and_policy_audit_v1", "simulation_only": True,
        "all_evidence_consistent": receipt["accepted_evidence_consistency"],
        "all_declared_command_bounds_satisfied": receipt["original_intersection_satisfied_for_all_audited_writes"],
        "all_native_close_spans_within_declared_deadline": True, "all_recorded_static_target_checks_consistent": True,
        "all_sampled_hypothetical_target_checks_passed": True, "all_recorded_provisional_policy_rows_consistent": True,
        "additional_bias_limit_rad": budget, "close_seconds": 32., "arm_interpolation_seconds": 16.,
        "cases": [receipt], "frozen_source": source, "rates_per_s": rates, "target_audit_transform": transform,
        "summary_sha256": initial[summary_path], "protocol_sha256": initial[protocol_path], "script_sha256": base.digest(__file__),
        "initializer_classification": targets["initializer_classification"],
        "target_qpos_writes_independently_proven": False, "physical_prefix_independently_compared": False,
        "scope": "True R21 provisional pre-motion admission intervention; R20 allocator and R17 adjacent-native command/cache-force plus strict initializer audit reused unchanged. Independently reconstructed source AST removes only one R14 actual-1g pre-motion reason and redirects only the R16 parent helper; recorded live method/source/global bindings are checked. Python bytecode hashes are recorded, not independently rebuilt. Policy telemetry and necessary current readiness/confirmation gates are checked; full dynamic pressure-window/admission history and held outcomes are not rescored. Actual fixed-normal observation is retained. Original 5mm/.25s/table unloading/confirmation/held/strict task gates remain source-bound. Hypothetical target samples are not actual equilibrium or task success; scalar/release/safety exclusions do not prove .10rad biased bounds."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = audit_directory(args.directory)
        status = 0 if receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"] else 1
    except (ValueError, KeyError, OSError, TypeError, StopIteration, zipfile.BadZipFile) as error:
        receipt, status = {"format": "r21_provisional_preload_audit_failure_v1", "accepted_evidence": False,
            "error": str(error), "script_sha256": base.digest(__file__)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": base.digest(args.output), "exit_code": status}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
