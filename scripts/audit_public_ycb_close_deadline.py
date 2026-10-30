#!/usr/bin/env python3
"""Audit the isolated R18 CLOSE deadline using frozen R17 command/force cores."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import zipfile

import audit_public_ycb_bias_budget as r17


base = r17.base
FORMAT = "public_ycb_close_deadline_development_r18"
R18_SOURCE = "scripts/develop_public_ycb_close_deadline_r18.py"
R18_AUDITOR = "scripts/audit_public_ycb_close_deadline.py"
R17_CORE = "scripts/audit_public_ycb_bias_budget.py"
R17_DIRECTORY = "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414"
R17_SUMMARY = R17_DIRECTORY + "/summary.json"
R17_PROTOCOL = R17_DIRECTORY + "/protocol.json"
R17_VERIFICATION = R17_DIRECTORY + "/verification_receipt.json"
R17_STRICT = R17_DIRECTORY + "/strict_native_verification.json"
R17_AUDIT = "results/r17_004_budget_audit_20261022/receipt.json"
FROZEN_R17_CORE_SHA256 = "4f4cbe14227aa526bb4bb0d1e3ef2707c50dfc184ca539fa185fae7130e97eb7"
FROZEN_R17_SOURCE_SHA256 = "c4ce070b16c116677bcf432842885c178d3b790d884c3852d1f9bc1dfbaa1d04"
FROZEN_R17_SUMMARY_SHA256 = "e2883dce787d3c9140edc59c77e415394d904a7029f9f8e4ec5559b3946d61d8"
FROZEN_R17_PROTOCOL_SHA256 = "05ba5d3b4d2f1ee970b148a971f70046067319204f32d34238ee0cbba20760c8"
FROZEN_R17_VERIFICATION_SHA256 = "df8eaa667f52a07c4d58b2c7ce377987f99a096d1e5fdf974b1ac38e82daf0ec"
FROZEN_R17_STRICT_SHA256 = "913b1a2a697d67cdeb4fd57c6c80a7ca0fa6dc41f3265aaf4269b9355479f83f"
FROZEN_R17_AUDIT_SHA256 = "8be63e51f6867a7f8c8ae322ff947bcf9b05e13e961dba8fe8920a5b989060bf"
FROZEN_R18_SOURCE_SHA256 = "89e600621af15e198ef5b803acf16d968b5f2b3e4d874f9d0890d23484de8a3e"
BUDGET_RAD, BEFORE_SECONDS, AFTER_SECONDS = .04, 16., 32.
TRANSFORM = {"arm_step_injections": 1, "arm_fraction_replacements": 2, "hand_fraction_replacements": 0}


def expected_preflight():
    return {r17.R17_SOURCE: FROZEN_R17_SOURCE_SHA256, R17_CORE: FROZEN_R17_CORE_SHA256,
        R17_SUMMARY: FROZEN_R17_SUMMARY_SHA256, R17_PROTOCOL: FROZEN_R17_PROTOCOL_SHA256,
        R17_VERIFICATION: FROZEN_R17_VERIFICATION_SHA256, R17_STRICT: FROZEN_R17_STRICT_SHA256,
        R17_AUDIT: FROZEN_R17_AUDIT_SHA256}


def frozen_inputs(directory, protocol):
    if base.digest(r17.__file__) != FROZEN_R17_CORE_SHA256:
        raise ValueError("Frozen R17 audit core changed")
    budget, rates, source = r17.frozen_inputs(directory, protocol)
    base.equal(budget, BUDGET_RAD, "R18 fixed active budget")
    before = protocol["source_hashes_before"]
    pins = expected_preflight()
    if any(before.get(key) != value for key, value in pins.items()):
        raise ValueError("R18 comparison/preflight identity differs from the exact R17 baseline")
    if before[R18_SOURCE] != FROZEN_R18_SOURCE_SHA256:
        raise ValueError("Frozen R18 runner source identity changed")
    if before[R18_AUDITOR] != base.digest(__file__):
        raise ValueError("R18 auditor is not bound to the run's source snapshot")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        if set(archive.namelist()) != set(before):
            raise ValueError("Unmanifested or missing frozen source archive entries")
        baseline = json.loads(archive.read(R17_SUMMARY))
        prior = json.loads(archive.read(R17_PROTOCOL))
        text = archive.read(R18_SOURCE)
        verification = json.loads(archive.read(R17_VERIFICATION))
        strict = json.loads(archive.read(R17_STRICT))
        audit = json.loads(archive.read(R17_AUDIT))
    if (baseline["complete"] is not True or baseline["protocol"] != prior
            or prior["format"] != "public_ycb_bias_budget_development_r17"
            or prior["source_changed_during_run"] is not False
            or prior["source_hashes_before"] != prior["source_hashes_after"]
            or any(before.get(key) != value for key, value in prior["source_hashes_before"].items())
            or len(baseline["rows"]) != 1 or len(prior["schedule"]) != 1):
        raise ValueError("R18 did not inherit the exact completed R17 source closure")
    if (verification["complete"] is not True or verification["unknowns"]
            or strict["accepted_evidence"] is not True or strict["unknowns"]
            or audit["all_evidence_consistent"] is not True
            or audit["all_declared_command_bounds_satisfied"] is not True
            or audit["script_sha256"] != FROZEN_R17_CORE_SHA256
            or any(item["summary_sha256"] != FROZEN_R17_SUMMARY_SHA256
                for item in (verification, strict, audit))):
        raise ValueError("R17 baseline receipts are incomplete or not comparison-bound")
    for name, expected in {"CLOSE_SECONDS_BEFORE": BEFORE_SECONDS, "CLOSE_SECONDS_AFTER": AFTER_SECONDS,
            "ARM_INTERPOLATION_SECONDS": BEFORE_SECONDS, "NATIVE_TIMESTEP_S": base.STEP_S,
            "BUDGET_RAD": BUDGET_RAD, "PROTOCOL_FORMAT": FORMAT,
            "EXPECTED_R17_SUMMARY_SHA256": FROZEN_R17_SUMMARY_SHA256}.items():
        if base.literal_assignment(text, name) != expected:
            raise ValueError("R18 executed constant differs: " + name)
    declaration = protocol["r18_close_deadline_experiment"]
    if (declaration["format"] != "r18_close_deadline_development_protocol_v1"
            or declaration["controller"] != "BiasBudgetHand" or declaration["runner_revision"] != FORMAT
            or protocol["runner_revision"] != FORMAT
            or declaration["changed_parameters"] != {"phase_seconds.CLOSE": {"before": BEFORE_SECONDS, "after": AFTER_SECONDS}}
            or declaration["preflight_input_sha256"] != pins
            or declaration["r17_comparison_summary_sha256"] != FROZEN_R17_SUMMARY_SHA256):
        raise ValueError("Unknown or inconsistent isolated R18 deadline declaration")
    for key, expected in {"close_seconds": AFTER_SECONDS, "baseline_close_seconds": BEFORE_SECONDS,
            "arm_interpolation_seconds": BEFORE_SECONDS, "native_timestep_s": base.STEP_S,
            "additional_bias_limit_rad": BUDGET_RAD}.items():
        base.equal(declaration[key], expected, "Declared " + key)
    if (base.flag(declaration, "runtime_target_pose_or_qpos_added") is not False
            or base.flag(declaration, "physics_force_and_task_gates_modified") is not False
            or declaration["arm_source_transform"] != {**TRANSFORM, "baseline_close_steps": 8000,
                "extension_holds_arm_fraction_at_one": True}
            or base.flag(declaration["prefix_validation"], "required") is not True
            or declaration["prefix_validation"]["status"] != "NOT_YET_RUN"):
        raise ValueError("R18 transform/scope declaration changed or claims unverified physical prefix")
    if prior["phase_seconds"]["CLOSE"] != BEFORE_SECONDS or prior["constrained_preload"]["close_seconds"] != BEFORE_SECONDS:
        raise ValueError("R17 baseline does not use the original 16s deadline")
    phase = dict(prior["phase_seconds"], CLOSE=AFTER_SECONDS)
    preload = dict(prior["constrained_preload"], close_seconds=AFTER_SECONDS)
    for key in ("joint_admittance", "nominal_profiles", "maximum_attempts", "physics_profile", "grip_gate", "scoring",
            "latched_pressure_escape", "load_transfer_probe", "probe_confirmation", "wrench_evidence",
            "probe_hypothesis", "scalar_acquisition_tracking", "tracking_regulator_sha256",
            "tracking_regulator_source", "r16_bias_recovery_experiment", "r17_bias_budget_experiment"):
        if protocol[key] != prior[key]:
            raise ValueError("R18 changed an inherited control/physics/task contract: " + key)
    if (declaration["inherited_r17_declaration"] != prior["r17_bias_budget_experiment"]
            or declaration["inherited_r17_force_control"] != prior["force_control"]
            or protocol["inherited_r17_force_control"] != prior["force_control"]
            or any(container["inherited_r17_phase_seconds"] != prior["phase_seconds"]
                or container["inherited_r17_constrained_preload"] != prior["constrained_preload"]
                for container in (protocol, declaration))
            or protocol["phase_seconds"] != phase or declaration["effective_phase_seconds"] != phase
            or protocol["constrained_preload"] != preload or declaration["effective_constrained_preload"] != preload):
        raise ValueError("Inherited/effective R18 task or runner phase metadata differs")
    for node in ast.walk(ast.parse(text)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        if any(isinstance(part, ast.Attribute) and part.attr == "qpos" for target in targets for part in ast.walk(target)):
            raise ValueError("R18 source contains direct qpos assignment")
    source.update(r18_source_sha256=before[R18_SOURCE], r18_auditor_sha256=before[R18_AUDITOR],
        r17_audit_core_sha256=FROZEN_R17_CORE_SHA256, r17_summary_sha256=FROZEN_R17_SUMMARY_SHA256,
        baseline_close_seconds=BEFORE_SECONDS, declared_close_seconds=AFTER_SECONDS,
        arm_interpolation_seconds=BEFORE_SECONDS, inherited_r17_source_files=len(prior["source_hashes_before"]))
    return budget, rates, source, baseline


def native_close_spans(native, maximum_attempts):
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise ValueError("Invalid maximum attempts")
    spans, start, end = [], None, None
    for index, sample in enumerate(native, 1):
        if base.tick(sample["timestamp"]) != index:
            raise ValueError("Deadline reconstruction requires contiguous native 2ms samples")
        if sample["state"] == "CLOSE":
            if start is None:
                start = index - 1
            end = index
        elif start is not None:
            spans.append((start, end))
            start = end = None
    if start is not None:
        spans.append((start, end))
    if not spans or len(spans) > maximum_attempts:
        raise ValueError("Missing CLOSE span or too many CLOSE attempts")
    result = []
    for start, end in spans:
        duration = (end - start) * base.STEP_S
        if duration > AFTER_SECONDS + base.STEP_S + 1e-10:
            raise ValueError("Native CLOSE span exceeds the declared 32s deadline plus one 2ms tolerance")
        result.append({"start_s": start * base.STEP_S, "end_s": end * base.STEP_S,
            "native_samples": end - start, "duration_s": duration,
            "exceeds_original_16s": duration > BEFORE_SECONDS + 1e-10})
    return result


def check_case_metadata(planned, row, protocol, baseline):
    if any(row[key] != value for key, value in planned.items()) or row["runner_error"]:
        raise ValueError("Mismatching R18 planned case or runner error")
    prior = baseline["protocol"]["schedule"][0]
    expected = dict(prior, runner_revision=FORMAT, close_seconds=AFTER_SECONDS,
        arm_interpolation_seconds=BEFORE_SECONDS)
    if planned != expected:
        raise ValueError("R18 changed the original R17 case or loaded budget")
    base.equal(r17.budget_value(row["additional_bias_limit_rad"]), BUDGET_RAD, "Loaded R18 case budget")
    if (row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("Actual factory-bound task/runner phases differ from R18 declaration")
    transform = row["arm_source_transform"]
    if (any(transform[key] != value for key, value in TRANSFORM.items())
            or transform["baseline_derived_controller_sha256"] != baseline["rows"][0]["derived_controller_sha256"]
            or transform["derived_controller_sha256"] != row["derived_controller_sha256"]):
        raise ValueError("R18 derived-controller transform is not R17 comparison-bound")


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = base.checked_path(directory, "summary.json"), base.checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    if summary["complete"] is not True or summary["protocol"] != protocol or protocol["format"] != FORMAT:
        raise ValueError("Expected one complete consistently bound R18 experiment")
    if (summary["runner_revision"] != FORMAT or summary["close_seconds"] != AFTER_SECONDS
            or summary["arm_interpolation_seconds"] != BEFORE_SECONDS):
        raise ValueError("R18 summary effective deadline/revision differs")
    budget, rates, source, baseline = frozen_inputs(directory, protocol)
    if len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1:
        raise ValueError("R18 requires one isolated comparison case")
    cases = []
    for planned, row in zip(protocol["schedule"], summary["rows"], strict=True):
        check_case_metadata(planned, row, protocol, baseline)
        case = base.checked_path(directory, row["path"])
        metadata, scene = base.checked_path(case, "metadata.json"), base.checked_path(case, "scene.xml")
        controller = base.checked_path(case, "derived_controller.py")
        if json.loads(metadata.read_text()) != row or base.digest(controller) != row["derived_controller_sha256"]:
            raise ValueError("R18 case metadata or derived-controller hash differs")
        if (base.flag(row, "scene_unchanged_since_load") is not True
                or row["scene_sha256_before_load"] != base.digest(scene)
                or row["scene_sha256_at_finalize"] != base.digest(scene)
                or row["scene_sha256_before_load"] != baseline["rows"][0]["scene_sha256_before_load"]):
            raise ValueError("R18 runtime-loaded scene binding differs")
        model = load_evidence_model(scene)
        base.equal(model.opt.timestep, base.STEP_S, "R18 native timestep", 1e-12)
        records = {key: base.checked_stream(case, row["artifacts"][key]) for key in
            ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
        if row["native_audit"]["native_actuation_samples"] != len(records["native"]):
            raise ValueError("R18 native actuation sample denominator differs")
        spans = native_close_spans(records["native"], protocol["maximum_attempts"])
        receipt = r17.reconstruct_budget_case(records["native"], records["bias_recovery"], records["joint_admittance"],
            base.joint_specs(model, protocol["nominal_profiles"][row["kind"]]), budget_rad=budget,
            **rates, preload_events=records["preload"], probe_events=records["load_transfer_probe"],
            native_audit=row["native_audit"])
        receipt.update(case=row["path"], scene_sha256=base.digest(scene), metadata_sha256=base.digest(metadata),
            derived_controller_sha256=base.digest(controller), native_close_spans=spans,
            all_native_close_spans_within_declared_deadline=True,
            inputs={key: row["artifacts"][key] for key in records},
            actuation=r17.audit_actuation(records["native"], row["native_audit"]["native_actuation_metadata"], model,
                expected_timing_scope=source["native_actuation_timing_scope"]),
            task_success=row["success"], held_lift=row["held_lift"],
            task_outcome_scope="Reported original task results, not independently rescored by this deadline/command/actuation auditor")
        cases.append(receipt)
    return {"format": "r18_close_deadline_command_and_native_actuation_audit_v1", "simulation_only": True,
        "all_evidence_consistent": all(item["accepted_evidence_consistency"] for item in cases),
        "all_declared_command_bounds_satisfied": all(item["original_intersection_satisfied_for_all_audited_writes"] for item in cases),
        "all_native_close_spans_within_declared_deadline": True,
        "close_seconds": AFTER_SECONDS, "baseline_close_seconds": BEFORE_SECONDS,
        "arm_interpolation_seconds": BEFORE_SECONDS, "additional_bias_limit_rad": budget,
        "cases": cases, "frozen_source": source, "rates_per_s": rates,
        "summary_sha256": base.digest(summary_path), "protocol_sha256": base.digest(protocol_path),
        "script_sha256": base.digest(__file__), "target_qpos_writes_independently_proven": False,
        "physical_prefix_independently_compared": False,
        "scope": "Thin R18 timing/provenance wrapper around exact frozen R17 budget and same-cache force algorithms. Native CLOSE spans are measured; a span below the deadline is not an early-success claim. Arm physical-prefix identity, strict task rescoring and no external target writes are not independently established here."}


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
        "close_seconds": receipt["close_seconds"], "output": str(args.output)}))
    return 0 if receipt["all_declared_command_bounds_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
