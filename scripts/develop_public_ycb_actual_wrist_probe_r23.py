#!/usr/bin/env python3
"""Next-run-only bounded actual-wrist probe over the finalized R22 case."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import develop_public_ycb_settled_alignment_r22 as r22
import actual_wrist_probe_controller_hooks as probe


PROTOCOL_FORMAT = "public_ycb_actual_wrist_probe_development_r23"
DECLARATION = "r23_actual_wrist_probe_experiment"
CONTROLLER_CLASS = r22.CONTROLLER_CLASS
TARGET_ALLOCATOR_REVISION = r22.TARGET_ALLOCATOR_REVISION
ADMISSION_POLICY_REVISION = r22.ADMISSION_POLICY_REVISION
CONTROLLER_REVISION_SCOPE = "Inherited R22 open ALIGN and R21 hand; bounded actual-wrist correction during unconfirmed TRIAL_LIFT only"
R22_DIRECTORY = PROJECT / "results/public_ycb_settled_alignment_20261022_r22_bias10_close32_seed3414"
R22_JOB = PROJECT / "results/jobs/r22_settled_alignment_bias10_close32_seed3414_20261022"
R22_SUMMARY_SHA256 = "f67343f1e534694bebf6b4f5c3f50bac619ed5a276d9cc621c374a08a7498618"
R22_ALIGNMENT_RECEIPT = PROJECT / "results/r22_settled_alignment_audit_20261022/receipt.json"
R22_ALIGNMENT_RECEIPT_SHA256 = "66368f6f499b32a3957e676f779759e4e0ec68b84a72bd92978e902ac23f82c3"
ALIGNMENT_AUDITOR = "scripts/audit_public_ycb_settled_alignment.py"
ALIGNMENT_AUDIT_FORMAT = "r22_settled_alignment_native_output_and_cached_wrist_audit_v1"
ALIGNMENT_CHECKS = ("all_native_open_outputs_and_recorded_alignment_rows_consistent",
    "all_recorded_cached_wrist_fk_checks_consistent", "all_alignment_termination_and_close_transitions_consistent")
NEW_SOURCES = ("scripts/actual_wrist_probe_controller_hooks.py", "tests/test_actual_wrist_probe_controller_hooks.py",
    "docs/R23_ACTUAL_WRIST_PROBE_POLICY_20261022.md", "scripts/develop_public_ycb_actual_wrist_probe_r23.py",
    "tests/test_public_ycb_actual_wrist_probe_r23.py", ALIGNMENT_AUDITOR,
    "tests/test_audit_public_ycb_settled_alignment.py", "scripts/public_ycb_evidence.py",
    "scripts/develop_public_ycb_cartesian_probe_r12.py")
sha256 = r22.sha256
checked_sha256 = r22.checked_sha256


def checked_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("R23 preflight path escapes its evidence root")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("R23 preflight does not accept symlink evidence")
    if not path.is_file():
        raise ValueError("Missing finalized R22 preflight file: " + str(path))
    return path


def check_exit(status, request, protocol, profile_path):
    command = request["command"]
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True
            or type(status["returncode"]) is not int or status["returncode"] != 0
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]
            or command.count("scripts/develop_public_ycb_settled_alignment_r22.py") != 1
            or r22._flag_argument(command, "--output") != R22_DIRECTORY.relative_to(PROJECT).as_posix()
            or r22._flag_argument(command, "--profile-json") != profile_path.relative_to(PROJECT).as_posix()
            or r22._flag_argument(command, "--expected-baseline-summary-sha256") !=
                protocol[r22.DECLARATION]["r21_comparison_summary_sha256"]
            or command.count("--cases") != 1 or command[command.index("--cases") + 1:] != ["ycb_can:3414"]):
        raise ValueError("R23 requires observed FINISHED rc0 R22 child exit with the exact case/profile/output")


def check_receipts(summary, hashes, files, native, strict, alignment):
    row = summary["rows"][0]
    count = row["native_audit"]["native_samples"]
    if (any(receipt["summary_sha256"] != hashes[files["summary"]] for receipt in (native, strict, alignment))
            or native["format"] != "public_ycb_development_native_verification_r1"
            or native["complete"] is not True or native["unknowns"] != 0
            or native["source_hashes_before_equal_after"] is not True
            or native["native_samples_contiguous_2ms"] is not True or native["audited_episodes"] != 1
            or native["frozen_source_sha256"] != hashes[files["archive"]]
            or native["native_samples"] != count or len(native["rows"]) != 1
            or native["rows"][0]["unknown"] is not False or native["rows"][0]["path"] != row["path"]
            or native["rows"][0]["outcome"] != row["success"]
            or native["rows"][0]["held_lift"] != row["held_lift"]
            or native["rows"][0]["native_samples"] != count
            or strict["format"] != "public_ycb_strict_native_audit_20261021"
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or len(strict["rows"]) != 1
            or strict["rows"][0]["unknown"] is not False or strict["rows"][0]["path"] != row["path"]
            or strict["rows"][0]["native_samples"] != count
            or strict["rows"][0]["checks"]["legacy_task_success"] != row["success"]
            or strict["rows"][0]["checks"]["held_lift"] != row["held_lift"]
            or strict["rows"][0]["result_sha256"] != hashes[files["result"]]
            or strict["rows"][0]["scene_sha256"] != hashes[files["scene"]]
            or alignment["format"] != ALIGNMENT_AUDIT_FORMAT
            or alignment["accepted_alignment_evidence"] is not True
            or alignment["protocol_sha256"] != hashes[files["protocol"]]
            or alignment["script_sha256"] != hashes[files["alignment_auditor"]]
            or any(alignment[key] is not True for key in ALIGNMENT_CHECKS)
            or alignment["inputs_unchanged_during_audit"] is not True
            or alignment["source"]["source_count"] != 146
            or alignment["source"]["all_archived_input_hashes_match"] is not True
            or alignment["source"]["before_after_manifest_equal"] is not True
            or alignment["source"]["archive_sha256"] != hashes[files["archive"]]
            or alignment["case"] != row["path"] or alignment["native_samples"] != count
            or alignment["native_timestep_s"] != .002
            or alignment["actual_hand_instances"] != row["effective_hand_instances"]
            or alignment["task_success"] != row["success"] or alignment["held_lift"] != row["held_lift"]):
        raise ValueError("Final R22 native/strict/alignment receipts are missing, unknown or inconsistent")
    for name in ("summary", "protocol", "archive", "scene", "controller", "metadata", "result", "native", "trace", "model_loader"):
        path = files["raw_native"] if name == "native" else files[name]
        binding = alignment["inputs"][name]
        if binding["sha256"] != hashes[path] or binding["bytes"] != path.stat().st_size:
            raise ValueError("R22 alignment receipt input binding differs: " + name)
    if (len(alignment["alignments"]) != len(row["alignment_settling"]["attempts"])
            or len(alignment["alignments"]) > row["effective_hand_instances"]
            or any(type(item["settled"]) is not bool for item in alignment["alignments"])):
        raise ValueError("R22 alignment audit attempt/outcome binding differs")


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    expected = checked_sha256(expected_baseline_summary_sha256)
    profile_path = Path(profile_path).resolve(strict=True)
    if not profile_path.is_relative_to(PROJECT):
        raise ValueError("R23 requires the inherited in-project R22 profile")
    files = {"summary": checked_file(R22_DIRECTORY, "summary.json"),
        "protocol": checked_file(R22_DIRECTORY, "protocol.json"),
        "status": checked_file(R22_JOB, "status.json"), "request": checked_file(R22_JOB, "request.json")}
    captured = {name: path.read_bytes() for name, path in files.items()}
    hashes = {files[name]: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
    summary, protocol, status, request = (json.loads(captured[name]) for name in ("summary", "protocol", "status", "request"))
    if (expected != R22_SUMMARY_SHA256 or hashes[files["summary"]] != expected
            or summary["complete"] is not True or summary["runner_errors"]
            or summary["protocol"] != protocol or protocol["format"] != r22.PROTOCOL_FORMAT
            or summary["runner_revision"] != r22.PROTOCOL_FORMAT or len(summary["rows"]) != 1
            or len(protocol["schedule"]) != 1 or protocol["source_changed_during_run"] is not False
            or len(protocol["source_hashes_before"]) != 146
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("R23 requires the explicitly hash-bound complete unchanged 146-source R22 baseline")
    check_exit(status, request, protocol, profile_path)
    inherited = protocol[r22.DECLARATION]
    if protocol["source_hashes_before"] != r22.snapshot(profile_path, inherited):
        raise ValueError("R22 source-bound inherited declaration differs")
    archive_path = checked_file(R22_DIRECTORY, "frozen_source.zip")
    files["archive"], hashes[archive_path] = archive_path, sha256(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names, before = archive.namelist(), protocol["source_hashes_before"]
        if len(names) != len(set(names)) or set(names) != set(before):
            raise ValueError("R22 source ZIP manifest differs")
        for relative, source_sha in before.items():
            if (sha256(checked_file(PROJECT, relative)) != source_sha
                    or hashlib.sha256(archive.read(relative)).hexdigest() != source_sha):
                raise ValueError("R22 archived/current runtime source bytes changed: " + relative)
    row = summary["rows"][0]
    count = row["effective_hand_instances"]
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["runner_error"] is not None
            or any(row[key] != value for key, value in protocol["schedule"][0].items())
            or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or type(count) is not int or not 1 <= count <= protocol["maximum_attempts"] <= 2
            or count != len(row["attempts"]) or row["effective_hand_budgets_rad"] != [.10] * count
            or row["effective_hand_controller_types"] != [CONTROLLER_CLASS] * count
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]
            or protocol["controller_class"] != CONTROLLER_CLASS
            or protocol["target_allocator_revision"] != TARGET_ALLOCATOR_REVISION
            or protocol["admission_policy_revision"] != ADMISSION_POLICY_REVISION
            or protocol["alignment_settling"] != r22.alignment.PARAMETERS
            or row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inherited["baseline_scene_sha256"]
            or row["scene_sha256_at_finalize"] != row["scene_sha256_before_load"]
            or row["derived_controller_sha256"] != probe.INHERITED_SOURCE_SHA256
            or protocol["nominal_profile_file"]["sha256"] != sha256(profile_path)
            or protocol["nominal_profiles"] != json.loads(profile_path.read_text(encoding="utf-8-sig"))):
        raise ValueError("R22 actual scene/profile/case/hand/ALIGN/clock binding differs")
    case = Path(row["path"])
    files.update(scene=checked_file(R22_DIRECTORY, case / "scene.xml"),
        controller=checked_file(R22_DIRECTORY, case / "derived_controller.py"),
        metadata=checked_file(R22_DIRECTORY, case / "metadata.json"),
        result=checked_file(R22_DIRECTORY, case / "result.json"),
        native=checked_file(R22_DIRECTORY, "verification_receipt.json"),
        strict=checked_file(R22_DIRECTORY, "strict_native_verification.json"),
        alignment=checked_file(PROJECT, R22_ALIGNMENT_RECEIPT.relative_to(PROJECT)),
        alignment_auditor=checked_file(PROJECT, ALIGNMENT_AUDITOR),
        model_loader=checked_file(PROJECT, "scripts/public_ycb_evidence.py"))
    for key, name in (("native", "raw_native"), ("trace", "trace")):
        artifact = row["artifacts"][key]
        files[name] = checked_file(R22_DIRECTORY, case / artifact["path"])
        if sha256(files[name]) != artifact["sha256"] or files[name].stat().st_size != artifact["bytes"]:
            raise ValueError("R22 finalized compressed artifact differs: " + key)
    hashes.update({path: sha256(path) for path in files.values() if path not in hashes})
    if (hashes[files["scene"]] != row["scene_sha256_before_load"]
            or hashes[files["controller"]] != row["derived_controller_sha256"]
            or hashes[files["alignment"]] != R22_ALIGNMENT_RECEIPT_SHA256
            or json.loads(files["metadata"].read_text()) != row):
        raise ValueError("Actual R22 scene/controller/metadata bytes differ")
    result = json.loads(files["result"].read_text())
    if (result["alignment_settling"] != row["alignment_settling"]
            or result["alignment_source_transform"] != row["alignment_source_transform"]
            or result["attempts"] != row["attempts"] or result["success"] != row["success"]):
        raise ValueError("Actual R22 result and runtime ALIGN telemetry differ")
    check_receipts(summary, hashes, files,
        *(json.loads(files[name].read_text()) for name in ("native", "strict", "alignment")))
    hashes.update({checked_file(PROJECT, name): sha256(PROJECT / name) for name in NEW_SOURCES})
    if any(sha256(path) != value for path, value in hashes.items()):
        raise ValueError("R23 preflight inputs changed while validating")
    raw_paths = {files["raw_native"], files["trace"]}
    frozen = {path.relative_to(PROJECT).as_posix(): value for path, value in hashes.items() if path not in raw_paths}
    return {"format": "r23_actual_wrist_probe_development_protocol_v1", "runner_revision": PROTOCOL_FORMAT,
        "controller": CONTROLLER_CLASS, "target_allocator_revision": TARGET_ALLOCATOR_REVISION,
        "admission_policy_revision": ADMISSION_POLICY_REVISION,
        "additional_bias_limit_rad": .10, "close_seconds": 32., "arm_interpolation_seconds": 16., "native_timestep_s": .002,
        "changed_parameters": {"unconfirmed_TRIAL_LIFT_actual_wrist_probe": copy.deepcopy(probe.PARAMETERS)},
        "inherited_r22_declaration": copy.deepcopy(inherited),
        "inherited_r22_source_hashes": copy.deepcopy(protocol["source_hashes_before"]),
        "inherited_r22_force_control": protocol["force_control"], "r22_comparison_summary_sha256": expected,
        "baseline_task_success": row["success"], "baseline_held_lift": row["held_lift"],
        "baseline_child_exit_observed": True, "baseline_child_returncode": 0,
        "baseline_r22_alignment_evidence_accepted": True, "r23_command_policy_audit_passed": False,
        "r23_command_policy_audit_status": "not_run_by_this_runner",
        "r21_command_policy_audit_scope": "R21 only; inherited audit is not an R22 or R23 command-policy audit",
        "baseline_scene_sha256": row["scene_sha256_before_load"],
        "baseline_derived_controller_sha256": row["derived_controller_sha256"], "derived_arm_source_changed": True,
        "preflight_input_sha256": frozen,
        "preflight_raw_artifact_bindings": {name: {"path": str(files[name].relative_to(PROJECT)),
            "sha256": hashes[files[name]], "bytes": files[name].stat().st_size} for name in ("raw_native", "trace")},
        "raw_artifact_archive_scope": "Baseline raw native/trace bytes verified at preflight and bound by frozen final summary and alignment receipt; large compressed streams are not recopied into source ZIP",
        "original_hand_class_allocator_admission_and_close_clocks_unchanged": True,
        "r22_alignment_source_and_policy_unchanged": True, "runtime_target_pose_or_qpos_added": False,
        "physics_force_and_strict_task_gates_modified": False,
        "scope": "Finite same-can same-seed actual-wrist probe intervention. R22 ALIGN acceptance does not audit R23 probe behavior, pressure/gravity mapping or dynamic task success."}


def snapshot(profile_path, declaration):
    values = {}
    for manifest in (declaration["inherited_r22_source_hashes"], declaration["preflight_input_sha256"]):
        for relative, expected in manifest.items():
            actual = sha256(checked_file(PROJECT, relative))
            if actual != expected or relative in values and values[relative] != expected:
                raise ValueError("R23 frozen preflight input changed: " + relative)
            values[relative] = actual
    for relative in NEW_SOURCES:
        values[relative] = sha256(checked_file(PROJECT, relative))
    return values


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r22.controller_for(*args, **kwargs)
    task, transformed, telemetry = probe.transform(original, source)
    task.actual_wrist_probe_source_transform = copy.deepcopy(task.r23_actual_wrist_probe_transform)
    task.actual_wrist_probe_telemetry = telemetry
    return task, transformed, hands, estimated


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r22.tagged_protocol(value, declaration["inherited_r22_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": CONTROLLER_REVISION_SCOPE,
        "target_allocator_revision": TARGET_ALLOCATOR_REVISION, "admission_policy_revision": ADMISSION_POLICY_REVISION}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop(r22.DECLARATION, None)
        protocol.update(active, format=PROTOCOL_FORMAT,
            r23_actual_wrist_probe_experiment=copy.deepcopy(declaration),
            inherited_r22_settled_alignment_experiment=copy.deepcopy(declaration["inherited_r22_declaration"]),
            inherited_r22_force_control=declaration["inherited_r22_force_control"],
            actual_wrist_probe=copy.deepcopy(probe.PARAMETERS),
            force_control="R23 restores bounded actual-wrist correction during active unconfirmed TRIAL_LIFT only. Actual cached wrist proprioception and logical commands are separate from gravity-compensated arm motor ctrl. R22 open ALIGN, R21 hand/admission, R20 target allocation, .10rad budget, CLOSE32/arm16, physical scene and strict task gates remain source-bound; R19 prefix, R22 ALIGN and R23 probe transformations and telemetry are separately declared.")
        for planned in protocol["schedule"]:
            planned.update(active)
    for row in value.get("rows", []):
        row.update(active)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(active)
    return value


def isolated_runner(profile_path, declaration):
    bindings = {}

    def bound_controller(*args, **kwargs):
        result = controller_for(*args, **kwargs)
        bindings[Path(args[1]).resolve()] = result[0]
        return result

    def bound_snapshot(profile, inherited):
        if inherited != declaration["inherited_r22_declaration"]:
            raise ValueError("R23 snapshot received a different inherited R22 declaration")
        return snapshot(profile, declaration)

    namespace = dict(r22.isolated_runner.__globals__, controller_for=bound_controller,
        snapshot=bound_snapshot,
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    factory = FunctionType(r22.isolated_runner.__code__, namespace, r22.isolated_runner.__name__,
        r22.isolated_runner.__defaults__, r22.isolated_runner.__closure__)
    original = factory(profile_path, declaration["inherited_r22_declaration"])
    globals_ = dict(original.__globals__)
    previous = globals_["previous"]
    proxy = SimpleNamespace(**vars(previous))

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            task = bindings.get(Path(path).parent.resolve())
            if task is None:
                value["missing_runtime_actual_wrist_probe_binding"] = True
            else:
                value.update(actual_wrist_probe_source_transform=copy.deepcopy(task.actual_wrist_probe_source_transform),
                    actual_wrist_probe=task.actual_wrist_probe_telemetry.as_dict(),
                    inherited_r21_command_audit_does_not_audit_r23=True,
                    inherited_r22_alignment_audit_does_not_audit_r23_probe=True)
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    globals_["previous"] = proxy
    return FunctionType(original.__code__, globals_, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R23 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R23 isolates the frozen ycb_can:3414 development case")
    declaration = validate_inputs(profile_path, expected_baseline_summary_sha256)
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--expected-baseline-summary-sha256", required=True, type=checked_sha256)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases,
        expected_baseline_summary_sha256=args.expected_baseline_summary_sha256)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
