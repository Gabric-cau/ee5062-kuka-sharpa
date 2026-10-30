#!/usr/bin/env python3
"""Audit R19's .10rad instance budget against the recovered .04rad R18 run."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import zipfile

import audit_public_ycb_bias_budget as r17
import audit_public_ycb_close_deadline as r18


base = r17.base
FORMAT = "public_ycb_extended_budget_development_r19"
R19_SOURCE = "scripts/develop_public_ycb_extended_budget_r19.py"
R19_AUDITOR = "scripts/audit_public_ycb_extended_budget.py"
R17_CORE = "scripts/audit_public_ycb_bias_budget.py"
R18_CORE = "scripts/audit_public_ycb_close_deadline.py"
R18_SOURCE = "scripts/develop_public_ycb_close_deadline_r18.py"
R18_DIRECTORY = "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered"
R18_SUMMARY = R18_DIRECTORY + "/summary.json"
R18_PROTOCOL = R18_DIRECTORY + "/protocol.json"
R18_RECOVERY = R18_DIRECTORY + "/recovery_receipt.json"
R18_VERIFICATION = R18_DIRECTORY + "/verification_receipt.json"
R18_STRICT = R18_DIRECTORY + "/strict_native_verification.json"
R18_AUDIT = "results/r18_close_deadline_audit_20261022/receipt.json"
R18_RECOVERY_SOURCE = "scripts/recover_public_ycb_close_deadline_finalization.py"
R18_PREFIX = "results/r17_r18_close_deadline_comparison_20261022/receipt.json"
R18_SCENE = R18_DIRECTORY + "/ycb_can_bias_budget_r17_seed3414/scene.xml"
R18_DERIVED = R18_DIRECTORY + "/ycb_can_bias_budget_r17_seed3414/derived_controller.py"
FROZEN_R17_CORE_SHA256 = "4f4cbe14227aa526bb4bb0d1e3ef2707c50dfc184ca539fa185fae7130e97eb7"
FROZEN_R18_CORE_SHA256 = "44f7405cffe1aa87eac98af9764dfd6a0df694c968d4a8f9c7a7703d4b26ef48"
FROZEN_R18_SOURCE_SHA256 = "89e600621af15e198ef5b803acf16d968b5f2b3e4d874f9d0890d23484de8a3e"
FROZEN_R18_SUMMARY_SHA256 = "d3d5dec564810d8c9b6a9cc4cee8354062eb9f99c54960ab49b41ad87a740ff0"
FROZEN_R18_PROTOCOL_SHA256 = "a60d2238665deb40712ad7c66d5b69230a559613f522c9a006faa5b4f6aa0387"
FROZEN_R18_RECOVERY_SHA256 = "a40d4da5f8f0e8e1cbeb471940ce502241da74c69571e9eb442a7db5f17f77fa"
FROZEN_R18_VERIFICATION_SHA256 = "7e97c56ac5b8456cfc1cb93516cfe7dc97fb3f17ab4f94f0e4bd86c159254b21"
FROZEN_R18_STRICT_SHA256 = "98938ea2037f4f32fee56cc18a22d4180e5b17b1bd7d9a9a86d1d4952722b61c"
FROZEN_R18_AUDIT_SHA256 = "ade3adf70fd68f8d5d6f41e28d125b0a0e6778da58e3c2c61661957a3bb99dc4"
FROZEN_R18_RECOVERY_SOURCE_SHA256 = "4d88ef2c34a3c208979a55585180cc1d5ead42eb3142e412a8ebbed7b16f2778"
FROZEN_R18_PREFIX_SHA256 = "76f1cf9f65527d899a0f771c12d397a6c8c45d268c0e4b2055ad391086f321f7"
FROZEN_R18_SCENE_SHA256 = "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e"
FROZEN_R18_DERIVED_SHA256 = "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485"
FROZEN_R19_SOURCE_SHA256 = "619728464296d13ff6476acc16390ef419eb94d2dd8c134b596e40a4d0e417b4"
BUDGET_RAD, BASELINE_BUDGET_RAD = .10, .04
CLOSE_SECONDS, ARM_SECONDS = 32., 16.


def expected_preflight():
    return {R18_SOURCE: FROZEN_R18_SOURCE_SHA256, R18_CORE: FROZEN_R18_CORE_SHA256,
        R18_SUMMARY: FROZEN_R18_SUMMARY_SHA256, R18_PROTOCOL: FROZEN_R18_PROTOCOL_SHA256,
        R18_RECOVERY: FROZEN_R18_RECOVERY_SHA256, R18_VERIFICATION: FROZEN_R18_VERIFICATION_SHA256,
        R18_STRICT: FROZEN_R18_STRICT_SHA256, R18_AUDIT: FROZEN_R18_AUDIT_SHA256,
        R18_RECOVERY_SOURCE: FROZEN_R18_RECOVERY_SOURCE_SHA256, R18_PREFIX: FROZEN_R18_PREFIX_SHA256,
        R18_SCENE: FROZEN_R18_SCENE_SHA256, R18_DERIVED: FROZEN_R18_DERIVED_SHA256}


def frozen_inputs(directory, protocol):
    if (base.digest(r17.__file__) != FROZEN_R17_CORE_SHA256
            or base.digest(r18.__file__) != FROZEN_R18_CORE_SHA256):
        raise ValueError("Frozen R17 command/force or R18 deadline audit core changed")
    budget, rates, source = r17.frozen_inputs(directory, protocol)
    base.equal(budget, BUDGET_RAD, "R19 active budget")
    before, pins = protocol["source_hashes_before"], expected_preflight()
    if any(before.get(path) != expected for path, expected in pins.items()):
        raise ValueError("R19 did not preserve the exact recovered R18 comparison inputs")
    if before[R19_SOURCE] != FROZEN_R19_SOURCE_SHA256:
        raise ValueError("Frozen R19 runner source identity changed")
    if before[R19_AUDITOR] != base.digest(__file__):
        raise ValueError("R19 auditor is not bound to the run snapshot")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        if set(archive.namelist()) != set(before):
            raise ValueError("Unmanifested or missing R19 source archive entries")
        archived = {key: json.loads(archive.read(key)) for key in
            (R18_SUMMARY, R18_PROTOCOL, R18_RECOVERY, R18_VERIFICATION, R18_STRICT, R18_AUDIT, R18_PREFIX)}
        runner_text = archive.read(R19_SOURCE)
    baseline, prior = archived[R18_SUMMARY], archived[R18_PROTOCOL]
    recovery, verification, strict, audit = (archived[key] for key in
        (R18_RECOVERY, R18_VERIFICATION, R18_STRICT, R18_AUDIT))
    if (baseline["complete"] is not True or baseline["protocol"] != prior
            or prior["format"] != r18.FORMAT or prior["source_changed_during_run"] is not False
            or prior["source_hashes_before"] != prior["source_hashes_after"]
            or any(before.get(key) != value for key, value in prior["source_hashes_before"].items())
            or len(baseline["rows"]) != 1 or len(prior["schedule"]) != 1):
        raise ValueError("R19 baseline is not the completed immutable recovered R18 experiment")
    if (baseline["artifact_finalization_recovered"] is not True
            or baseline["original_runner_normal_exit"] is not False
            or prior["finalization_recovery"] != recovery
            or recovery["format"] != "r18_artifact_finalization_recovery_v1"
            or recovery["simulation_rerun"] is not False or recovery["original_complete"] is not False
            or recovery["original_runner_normal_exit_claimed"] is not False
            or recovery["runner_exit_code_observed"] is not None
            or recovery["observed_ssh_client_exit_code"] != 255
            or recovery["script_sha256"] != FROZEN_R18_RECOVERY_SOURCE_SHA256
            or recovery["source_hashes_at_recovery"] != prior["source_hashes_before"]):
        raise ValueError("Recovered R18 provenance was removed or relabeled as normal runner completion")
    if (verification["complete"] is not True or verification["unknowns"]
            or strict["accepted_evidence"] is not True or strict["unknowns"]
            or audit["all_evidence_consistent"] is not True
            or audit["all_declared_command_bounds_satisfied"] is not True
            or audit["all_native_close_spans_within_declared_deadline"] is not True
            or audit["script_sha256"] != FROZEN_R18_CORE_SHA256
            or audit["protocol_sha256"] != FROZEN_R18_PROTOCOL_SHA256
            or archived[R18_PREFIX]["passed"] is not False
            or archived[R18_PREFIX]["candidate_inputs"]["summary"]["sha256"] != FROZEN_R18_SUMMARY_SHA256
            or any(value["summary_sha256"] != FROZEN_R18_SUMMARY_SHA256
                for value in (verification, strict, audit))):
        raise ValueError("R18 native/strict/command receipts are incomplete or not baseline-bound")
    for key, expected in {"additional_bias_limit_rad": BASELINE_BUDGET_RAD,
            "close_seconds": CLOSE_SECONDS, "arm_interpolation_seconds": ARM_SECONDS}.items():
        base.equal(audit[key], expected, "Baseline R18 audit " + key)
    for key, expected in {"BUDGET_RAD": BUDGET_RAD, "BASELINE_BUDGET_RAD": BASELINE_BUDGET_RAD,
            "CLOSE_SECONDS": CLOSE_SECONDS, "NATIVE_TIMESTEP_S": base.STEP_S,
            "ARM_INTERPOLATION_SECONDS": ARM_SECONDS, "PROTOCOL_FORMAT": FORMAT,
            "EXPECTED_R18_SUMMARY_SHA256": FROZEN_R18_SUMMARY_SHA256,
            "EXPECTED_R18_DERIVED_SHA256": FROZEN_R18_DERIVED_SHA256}.items():
        if base.literal_assignment(runner_text, key) != expected:
            raise ValueError("R19 executed constant differs: " + key)
    declaration = protocol["r19_extended_budget_experiment"]
    if (declaration["format"] != "r19_extended_budget_development_protocol_v1"
            or declaration["controller"] != "BiasBudgetHand"
            or declaration["runner_revision"] != FORMAT or protocol["runner_revision"] != FORMAT
            or declaration["changed_parameters"] != {"joint_admittance.additional_bias_limit_rad":
                {"before": BASELINE_BUDGET_RAD, "after": BUDGET_RAD}}
            or declaration["preflight_input_sha256"] != pins
            or declaration["r18_comparison_summary_sha256"] != FROZEN_R18_SUMMARY_SHA256):
        raise ValueError("Unknown or inconsistent R19 isolated budget declaration")
    for key, expected in {"additional_bias_limit_rad": BUDGET_RAD,
            "baseline_additional_bias_limit_rad": BASELINE_BUDGET_RAD, "close_seconds": CLOSE_SECONDS,
            "arm_interpolation_seconds": ARM_SECONDS, "native_timestep_s": base.STEP_S}.items():
        base.equal(declaration[key], expected, "Declared R19 " + key)
    if (base.flag(declaration, "runtime_target_pose_or_qpos_added") is not False
            or base.flag(declaration, "physics_force_and_task_gates_modified") is not False):
        raise ValueError("R19 declares changed physical/task gates or target state inputs")
    expected_joint = copy.deepcopy(prior["joint_admittance"])
    base.equal(expected_joint["additional_bias_limit_rad"], BASELINE_BUDGET_RAD, "Original R18 budget")
    expected_joint["additional_bias_limit_rad"] = BUDGET_RAD
    expected_r17 = copy.deepcopy(prior["r17_bias_budget_experiment"])
    expected_r17["additional_bias_limit_rad"] = BUDGET_RAD
    expected_r17["changed_controller_parameters"]["additional_bias_limit_rad"]["after"] = BUDGET_RAD
    expected_r17["joint_parameters"]["additional_bias_limit_rad"] = BUDGET_RAD
    if (protocol["joint_admittance"] != expected_joint
            or protocol["r17_bias_budget_experiment"] != expected_r17
            or declaration["active_r17_declaration"] != expected_r17
            or declaration["inherited_r18_declaration"] != prior["r18_close_deadline_experiment"]
            or declaration["inherited_r18_r17_declaration"] != prior["r17_bias_budget_experiment"]):
        raise ValueError("Active .10rad and inherited .04rad declarations were confused or changed")
    if ("r18_close_deadline_experiment" in protocol
            or protocol["inherited_r18_close_deadline_experiment"] != prior["r18_close_deadline_experiment"]
            or protocol["inherited_r18_joint_admittance"] != prior["joint_admittance"]
            or protocol["inherited_r18_force_control"] != prior["force_control"]
            or declaration["inherited_r18_joint_admittance"] != prior["joint_admittance"]
            or declaration["inherited_r18_force_control"] != prior["force_control"]
            or declaration["inherited_r18_recovery"] != recovery
            or declaration["effective_phase_seconds"] != prior["phase_seconds"]
            or declaration["effective_constrained_preload"] != prior["constrained_preload"]
            or declaration["baseline_source_after_timing"] != prior["source_hashes_after_timing"]
            or declaration["baseline_scene_sha256"] != FROZEN_R18_SCENE_SHA256
            or declaration["baseline_derived_controller_sha256"] != FROZEN_R18_DERIVED_SHA256
            or declaration["baseline_task_success"] is not False
            or declaration["baseline_held_lift"] is not False
            or declaration["baseline_prefix_comparison_passed"] is not False):
        raise ValueError("R19 historical R18 provenance, failure or effective phase binding differs")
    for key in ("phase_seconds", "constrained_preload", "nominal_profiles", "maximum_attempts",
            "physics_profile", "grip_gate", "scoring", "latched_pressure_escape", "load_transfer_probe",
            "probe_confirmation", "wrench_evidence", "probe_hypothesis", "scalar_acquisition_tracking",
            "tracking_regulator_sha256", "tracking_regulator_source", "r16_bias_recovery_experiment"):
        if protocol[key] != prior[key]:
            raise ValueError("R19 changed an inherited timing/control/physics/task contract: " + key)
    if (prior["phase_seconds"]["CLOSE"] != CLOSE_SECONDS
            or prior["constrained_preload"]["close_seconds"] != CLOSE_SECONDS):
        raise ValueError("R18 comparison is not the original 32s CLOSE experiment")
    for node in ast.walk(ast.parse(runner_text)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        if any(isinstance(part, ast.Attribute) and part.attr == "qpos" for target in targets for part in ast.walk(target)):
            raise ValueError("R19 source contains direct qpos assignment")
    source.update(r19_source_sha256=before[R19_SOURCE], r19_auditor_sha256=before[R19_AUDITOR],
        r17_audit_core_sha256=FROZEN_R17_CORE_SHA256, r18_audit_core_sha256=FROZEN_R18_CORE_SHA256,
        r18_summary_sha256=FROZEN_R18_SUMMARY_SHA256, r18_recovery_sha256=FROZEN_R18_RECOVERY_SHA256,
        inherited_r18_source_files=len(prior["source_hashes_before"]),
        comparison_additional_bias_limit_rad=BASELINE_BUDGET_RAD)
    return budget, rates, source, baseline


def check_case_metadata(planned, row, protocol, baseline):
    if (any(row[key] != value for key, value in planned.items()) or row["runner_error"]
            or row.get("missing_runtime_controller_binding", False) is not False):
        raise ValueError("Mismatching planned R19 case or runner error")
    expected = dict(baseline["protocol"]["schedule"][0], runner_revision=FORMAT,
        additional_bias_limit_rad=BUDGET_RAD, close_seconds=CLOSE_SECONDS,
        arm_interpolation_seconds=ARM_SECONDS)
    if planned != expected:
        raise ValueError("R19 changed the original R18 case or planned budget")
    base.equal(r17.budget_value(row["additional_bias_limit_rad"]), BUDGET_RAD, "Loaded R19 case budget")
    if (row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("Effective R19 task/runner timing differs")
    count = row["effective_hand_instances"]
    if (type(count) is not int or not 1 <= count <= protocol["maximum_attempts"]
            or count != len(row["attempts"])
            or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * count
            or row["effective_hand_controller_types"] != ["BiasBudgetHand"] * count):
        raise ValueError("Actual R19 hand instances/retries did not all use the .10rad budget")
    prior = baseline["rows"][0]
    if (row["derived_controller_sha256"] != prior["derived_controller_sha256"]
            or row["arm_source_transform"] != dict(prior["arm_source_transform"],
                baseline_r18_derived_controller_sha256=prior["derived_controller_sha256"])):
        raise ValueError("R19 changed the R18 derived controller or its original arm interpolation")


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = base.checked_path(directory, "summary.json"), base.checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    if summary["complete"] is not True or summary["protocol"] != protocol or protocol["format"] != FORMAT:
        raise ValueError("Expected a complete consistently bound R19 experiment")
    if (summary["runner_revision"] != FORMAT or summary["close_seconds"] != CLOSE_SECONDS
            or summary["arm_interpolation_seconds"] != ARM_SECONDS
            or summary["additional_bias_limit_rad"] != BUDGET_RAD):
        raise ValueError("R19 summary timing/revision/budget differs")
    budget, rates, source, baseline = frozen_inputs(directory, protocol)
    if len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1:
        raise ValueError("R19 requires one isolated R18 comparison case")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    check_case_metadata(planned, row, protocol, baseline)
    case = base.checked_path(directory, row["path"])
    metadata, scene = base.checked_path(case, "metadata.json"), base.checked_path(case, "scene.xml")
    controller, result_path = base.checked_path(case, "derived_controller.py"), base.checked_path(case, "result.json")
    if json.loads(metadata.read_text()) != row or base.digest(controller) != row["derived_controller_sha256"]:
        raise ValueError("R19 case metadata or derived-controller hash differs")
    result = json.loads(result_path.read_text())
    if (result["attempts"] != row["attempts"] or result["runner_revision"] != FORMAT
            or result["controller_revision"] != "public_ycb_bias_budget_development_r17"
            or result["close_seconds"] != CLOSE_SECONDS or result["arm_interpolation_seconds"] != ARM_SECONDS
            or result["additional_bias_limit_rad"] != BUDGET_RAD):
        raise ValueError("R19 result/retry/controller binding differs")
    if (base.flag(row, "scene_unchanged_since_load") is not True
            or row["scene_sha256_before_load"] != base.digest(scene)
            or row["scene_sha256_at_finalize"] != base.digest(scene)
            or row["scene_sha256_before_load"] != baseline["rows"][0]["scene_sha256_before_load"]):
        raise ValueError("R19 runtime-loaded scene differs from R18")
    model = load_evidence_model(scene)
    base.equal(model.opt.timestep, base.STEP_S, "R19 native timestep", 1e-12)
    records = {key: base.checked_stream(case, row["artifacts"][key]) for key in
        ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
    if row["native_audit"]["native_actuation_samples"] != len(records["native"]):
        raise ValueError("R19 native actuation denominator differs")
    spans = r18.native_close_spans(records["native"], protocol["maximum_attempts"])
    receipt = r17.reconstruct_budget_case(records["native"], records["bias_recovery"], records["joint_admittance"],
        base.joint_specs(model, protocol["nominal_profiles"][row["kind"]]), budget_rad=budget,
        **rates, preload_events=records["preload"], probe_events=records["load_transfer_probe"],
        native_audit=row["native_audit"])
    receipt.update(case=row["path"], scene_sha256=base.digest(scene), metadata_sha256=base.digest(metadata),
        result_sha256=base.digest(result_path), derived_controller_sha256=base.digest(controller),
        effective_hand_instances=row["effective_hand_instances"], effective_hand_budgets_rad=row["effective_hand_budgets_rad"],
        native_close_spans=spans, all_native_close_spans_within_declared_deadline=True,
        inputs={key: row["artifacts"][key] for key in records},
        actuation=r17.audit_actuation(records["native"], row["native_audit"]["native_actuation_metadata"], model,
            expected_timing_scope=source["native_actuation_timing_scope"]),
        task_success=row["success"], held_lift=row["held_lift"],
        task_outcome_scope="Reported original task results; not independently rescored by this budget audit")
    return {"format": "r19_extended_budget_command_and_native_actuation_audit_v1", "simulation_only": True,
        "all_evidence_consistent": receipt["accepted_evidence_consistency"],
        "all_declared_command_bounds_satisfied": receipt["original_intersection_satisfied_for_all_audited_writes"],
        "all_native_close_spans_within_declared_deadline": True, "additional_bias_limit_rad": budget,
        "comparison_additional_bias_limit_rad": BASELINE_BUDGET_RAD, "close_seconds": CLOSE_SECONDS,
        "arm_interpolation_seconds": ARM_SECONDS, "cases": [receipt], "frozen_source": source,
        "rates_per_s": rates, "summary_sha256": base.digest(summary_path), "protocol_sha256": base.digest(protocol_path),
        "script_sha256": base.digest(__file__), "target_qpos_writes_independently_proven": False,
        "physical_prefix_independently_compared": False,
        "scope": "R19 .10rad command-budget sensitivity experiment against an explicitly recovered .04rad R18 baseline. Uses frozen R17 adjacent-native command and same-cache actuator-force/torque algorithms plus the frozen R18 CLOSE-span helper. No global or protocol mutation, physical-prefix equality, strict task rescoring or independent no-target-write proof. Excluded safety/release/scalar modes do not prove .10rad biased bounds; .10rad compliance is not compliance with the .04rad or original .02rad experiments."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = audit_directory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"all_evidence_consistent": receipt["all_evidence_consistent"],
        "all_declared_command_bounds_satisfied": receipt["all_declared_command_bounds_satisfied"],
        "additional_bias_limit_rad": receipt["additional_bias_limit_rad"], "output": str(args.output)}))
    return 0 if receipt["all_declared_command_bounds_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
