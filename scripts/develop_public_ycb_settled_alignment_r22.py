#!/usr/bin/env python3
"""Next-run-only bounded open ALIGN settling over a finalized R21 baseline."""

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
import develop_public_ycb_provisional_preload_r21 as r21
import settled_alignment_controller_hooks as alignment

PROTOCOL_FORMAT = "public_ycb_settled_alignment_development_r22"
DECLARATION = "r22_settled_alignment_experiment"
CONTROLLER_CLASS = r21.CONTROLLER_CLASS
TARGET_ALLOCATOR_REVISION = r21.TARGET_ALLOCATOR_REVISION
ADMISSION_POLICY_REVISION = r21.PROTOCOL_FORMAT
CONTROLLER_REVISION_SCOPE = "Inherited R21 hand/admission and R19 CLOSE clocks; bounded extra open ALIGN settling only"
R21_DIRECTORY = PROJECT / "results/public_ycb_provisional_preload_20261022_r21_bias10_close32_seed3414"
R21_JOB = PROJECT / "results/jobs/r21_provisional_preload_bias10_close32_seed3414_20261022"
R21_COMMAND_RECEIPT = PROJECT / "results/r21_provisional_preload_audit_20261022/receipt.json"
R21_AUDITOR = "scripts/audit_public_ycb_provisional_preload.py"
R21_AUDIT_FORMAT = "r21_provisional_preload_command_native_target_and_policy_audit_v1"
AUDIT_CHECKS = ("all_evidence_consistent", "all_declared_command_bounds_satisfied",
    "all_native_close_spans_within_declared_deadline", "all_recorded_static_target_checks_consistent",
    "all_sampled_hypothetical_target_checks_passed", "all_recorded_provisional_policy_rows_consistent")
NEW_SOURCES = ("scripts/settled_alignment_controller_hooks.py",
    "tests/test_settled_alignment_controller_hooks.py", "docs/SETTLED_ALIGNMENT_POLICY_20261022.md",
    "scripts/develop_public_ycb_settled_alignment_r22.py", "tests/test_public_ycb_settled_alignment_r22.py")
sha256 = r21.r20.r19.r17.r16.sha256


def checked_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("An explicit lowercase 64-character finalized baseline summary SHA256 is required")
    return value


def checked_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("R22 preflight path escapes its evidence root")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("R22 preflight does not accept symlink evidence")
    if not path.is_file():
        raise ValueError("Missing finalized R21 preflight file: " + str(path))
    return path


def _flag_argument(command, flag):
    if not isinstance(command, list) or command.count(flag) != 1:
        raise ValueError("R21 supervisor command has a missing or repeated " + flag)
    index = command.index(flag)
    if index + 1 == len(command):
        raise ValueError("R21 supervisor command has no value for " + flag)
    return command[index + 1]


def check_exit(status, request, protocol, profile_path):
    command = request["command"]
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True
            or type(status["returncode"]) is not int or status["returncode"] != 0
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]
            or command.count("scripts/develop_public_ycb_provisional_preload_r21.py") != 1
            or _flag_argument(command, "--output") != R21_DIRECTORY.relative_to(PROJECT).as_posix()
            or _flag_argument(command, "--profile-json") != profile_path.relative_to(PROJECT).as_posix()
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]
            or command.count("--cases") != 1):
        raise ValueError("R22 requires observed FINISHED rc0 R21 child exit with the exact case/profile/output")


def check_receipts(summary, protocol, summary_sha, protocol_sha, archive_sha, native, strict, command):
    row = summary["rows"][0]
    if (any(receipt["summary_sha256"] != summary_sha for receipt in (native, strict, command))
            or native["format"] != "public_ycb_development_native_verification_r1"
            or native["complete"] is not True or native["unknowns"] != 0
            or native["source_hashes_before_equal_after"] is not True
            or native["native_samples_contiguous_2ms"] is not True or native["audited_episodes"] != 1
            or native["frozen_source_sha256"] != archive_sha or len(native["rows"]) != 1
            or native["native_samples"] != row["native_audit"]["native_samples"]
            or native["rows"][0]["unknown"] is not False
            or native["rows"][0]["path"] != row["path"]
            or native["rows"][0]["outcome"] != row["success"]
            or native["rows"][0]["held_lift"] != row["held_lift"]
            or strict["format"] != "public_ycb_strict_native_audit_20261021"
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or len(strict["rows"]) != 1
            or strict["rows"][0]["unknown"] is not False or strict["rows"][0]["path"] != row["path"]
            or strict["rows"][0]["checks"]["legacy_task_success"] != row["success"]
            or strict["rows"][0]["checks"]["held_lift"] != row["held_lift"]
            or command["format"] != R21_AUDIT_FORMAT or command["protocol_sha256"] != protocol_sha
            or command["script_sha256"] != protocol["source_hashes_before"][R21_AUDITOR]
            or command["script_sha256"] != sha256(PROJECT / R21_AUDITOR)
            or any(command[key] is not True for key in AUDIT_CHECKS)
            or command["additional_bias_limit_rad"] != .10 or command["close_seconds"] != 32.
            or command["arm_interpolation_seconds"] != 16. or len(command["cases"]) != 1):
        raise ValueError("Final R21 native/strict/command-policy receipts are missing, unknown or inconsistent")
    case = command["cases"][0]
    if (case["case"] != row["path"] or case["actual_controller_class"] != CONTROLLER_CLASS
            or case["effective_hand_instances"] != row["effective_hand_instances"]
            or case["task_success"] != row["success"] or case["held_lift"] != row["held_lift"]
            or case["actuation"]["all_cached_force_and_transmitted_torque_consistent"] is not True
            or case["provisional_policy"]["all_recorded_policy_rows_consistent"] is not True):
        raise ValueError("Final R21 audit case/class/attempt/outcome binding differs")


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    expected = checked_sha256(expected_baseline_summary_sha256)
    profile_path = Path(profile_path).resolve(strict=True)
    if not profile_path.is_relative_to(PROJECT):
        raise ValueError("R22 requires the inherited in-project R21 profile")
    files = {"summary": checked_file(R21_DIRECTORY, "summary.json"),
        "protocol": checked_file(R21_DIRECTORY, "protocol.json"),
        "status": checked_file(R21_JOB, "status.json"), "request": checked_file(R21_JOB, "request.json")}
    captured = {name: path.read_bytes() for name, path in files.items()}
    hashes = {files[name]: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
    summary, protocol, status, request = (json.loads(captured[name]) for name in ("summary", "protocol", "status", "request"))
    if (hashes[files["summary"]] != expected or summary["complete"] is not True or summary["runner_errors"]
            or summary["protocol"] != protocol or protocol["format"] != r21.PROTOCOL_FORMAT
            or summary["runner_revision"] != r21.PROTOCOL_FORMAT or len(summary["rows"]) != 1
            or len(protocol["schedule"]) != 1 or protocol["source_changed_during_run"] is not False
            or len(protocol["source_hashes_before"]) != 130
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("R22 requires the explicitly hash-bound complete unchanged 130-source R21 baseline")
    check_exit(status, request, protocol, profile_path)
    inherited = r21.validate_inputs(profile_path)
    if (protocol["r21_provisional_preload_experiment"] != inherited
            or protocol["source_hashes_before"] != r21.snapshot(profile_path)):
        raise ValueError("R21 inherited R20-pinned declaration differs")
    archive_path = checked_file(R21_DIRECTORY, "frozen_source.zip")
    hashes[archive_path] = sha256(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names, before = archive.namelist(), protocol["source_hashes_before"]
        if len(names) != len(set(names)) or set(names) != set(before):
            raise ValueError("R21 source ZIP manifest differs")
        for relative, source_sha in before.items():
            if (sha256(checked_file(PROJECT, relative)) != source_sha
                    or hashlib.sha256(archive.read(relative)).hexdigest() != source_sha):
                raise ValueError("R21 archived/current runtime source bytes changed: " + relative)
    row = summary["rows"][0]
    count = row["effective_hand_instances"]
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["runner_error"] is not None
            or any(row[key] != value for key, value in protocol["schedule"][0].items())
            or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or type(count) is not int or not 1 <= count <= protocol["maximum_attempts"]
            or count != len(row["attempts"]) or row["effective_hand_budgets_rad"] != [.10] * count
            or row["effective_hand_controller_types"] != [CONTROLLER_CLASS] * count
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]
            or protocol["controller_class"] != CONTROLLER_CLASS
            or protocol["target_allocator_revision"] != TARGET_ALLOCATOR_REVISION
            or protocol["admission_policy_revision"] != ADMISSION_POLICY_REVISION
            or row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inherited["baseline_scene_sha256"]
            or row["scene_sha256_at_finalize"] != row["scene_sha256_before_load"]
            or row["derived_controller_sha256"] != alignment.INHERITED_SOURCE_SHA256
            or protocol["nominal_profile_file"]["sha256"] != sha256(profile_path)
            or protocol["nominal_profiles"] != json.loads(profile_path.read_text(encoding="utf-8-sig"))):
        raise ValueError("R21 actual scene/profile/case/hand/clock binding differs")
    case_relative = Path(row["path"])
    files.update(scene=checked_file(R21_DIRECTORY, case_relative / "scene.xml"),
        controller=checked_file(R21_DIRECTORY, case_relative / "derived_controller.py"),
        metadata=checked_file(R21_DIRECTORY, case_relative / "metadata.json"),
        native=checked_file(R21_DIRECTORY, "verification_receipt.json"),
        strict=checked_file(R21_DIRECTORY, "strict_native_verification.json"),
        command=checked_file(PROJECT, R21_COMMAND_RECEIPT.relative_to(PROJECT)))
    hashes.update({path: sha256(path) for path in files.values() if path not in hashes})
    if (hashes[files["scene"]] != row["scene_sha256_before_load"]
            or hashes[files["controller"]] != row["derived_controller_sha256"]
            or json.loads(files["metadata"].read_text()) != row):
        raise ValueError("Actual R21 scene/controller/metadata bytes differ")
    check_receipts(summary, protocol, expected, hashes[files["protocol"]], hashes[archive_path],
        *(json.loads(files[name].read_text()) for name in ("native", "strict", "command")))
    hashes.update({checked_file(PROJECT, name): sha256(PROJECT / name) for name in NEW_SOURCES})
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("R22 preflight inputs changed while validating")
    return {"format": "r22_settled_alignment_development_protocol_v1", "runner_revision": PROTOCOL_FORMAT,
        "controller": CONTROLLER_CLASS, "target_allocator_revision": TARGET_ALLOCATOR_REVISION,
        "admission_policy_revision": ADMISSION_POLICY_REVISION,
        "additional_bias_limit_rad": .10, "close_seconds": 32., "arm_interpolation_seconds": 16., "native_timestep_s": .002,
        "changed_parameters": {"open_ALIGN_completion": copy.deepcopy(alignment.PARAMETERS)},
        "inherited_r21_declaration": copy.deepcopy(inherited), "inherited_r21_force_control": protocol["force_control"],
        "r21_comparison_summary_sha256": expected, "baseline_task_success": row["success"],
        "baseline_held_lift": row["held_lift"], "baseline_child_exit_observed": True, "baseline_child_returncode": 0,
        "baseline_r21_command_policy_audit_passed": True, "r22_command_policy_audit_passed": False,
        "r22_command_policy_audit_status": "not_run_by_this_runner",
        "baseline_scene_sha256": row["scene_sha256_before_load"],
        "baseline_derived_controller_sha256": row["derived_controller_sha256"], "derived_arm_source_changed": True,
        "preflight_input_sha256": {path.relative_to(PROJECT).as_posix(): digest for path, digest in hashes.items()},
        "original_hand_class_allocator_admission_and_close_clocks_unchanged": True,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_strict_task_gates_modified": False,
        "scope": "Prepared finite same-can same-seed open ALIGN settling intervention. Baseline audit acceptance is not an R22 command audit, dynamic success or a randomized-dataset rate."}


def snapshot(profile_path, declaration):
    values = r21.snapshot(profile_path)
    for relative, expected in declaration["preflight_input_sha256"].items():
        actual = sha256(checked_file(PROJECT, relative))
        if actual != expected:
            raise ValueError("R22 frozen preflight input changed: " + relative)
        values[relative] = actual
    for relative in NEW_SOURCES:
        values[relative] = sha256(checked_file(PROJECT, relative))
    return values


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r21.controller_for(*args, **kwargs)
    task, transformed, telemetry = alignment.transform(original, source)
    task.alignment_source_transform = copy.deepcopy(task.r22_alignment_transform)
    task.alignment_settling_telemetry = telemetry
    return task, transformed, hands, estimated


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r21.tagged_protocol(value, declaration["inherited_r21_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": CONTROLLER_REVISION_SCOPE,
        "target_allocator_revision": TARGET_ALLOCATOR_REVISION, "admission_policy_revision": ADMISSION_POLICY_REVISION}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop("r21_provisional_preload_experiment", None)
        protocol.update(active, format=PROTOCOL_FORMAT,
            r22_settled_alignment_experiment=copy.deepcopy(declaration),
            inherited_r21_provisional_preload_experiment=copy.deepcopy(declaration["inherited_r21_declaration"]),
            inherited_r21_force_control=declaration["inherited_r21_force_control"],
            alignment_settling=copy.deepcopy(alignment.PARAMETERS),
            force_control="R22 adds bounded open ALIGN settling from wrist proprioception, the post-limiter logical arm command and actual open-hand motor controls. Arm command equality is not raw arm motor data.ctrl equality after gravity compensation. R21 hand/admission, R20 targets, .10rad budget, CLOSE32/arm16, physical scene and strict task gates remain source-bound; transformed arm source is declared separately.")
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

    namespace = dict(r21.isolated_runner.__globals__, controller_for=bound_controller,
        snapshot=lambda profile: snapshot(profile, declaration),
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    factory = FunctionType(r21.isolated_runner.__code__, namespace, r21.isolated_runner.__name__,
        r21.isolated_runner.__defaults__, r21.isolated_runner.__closure__)
    original = factory(profile_path, declaration["inherited_r21_declaration"])
    globals_ = dict(original.__globals__)
    previous = globals_["previous"]
    proxy = SimpleNamespace(**vars(previous))

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            task = bindings.get(Path(path).parent.resolve())
            if task is None:
                value["missing_runtime_alignment_binding"] = True
            else:
                value.update(alignment_source_transform=copy.deepcopy(task.alignment_source_transform),
                    alignment_settling=task.alignment_settling_telemetry.as_dict(),
                    arm_source_transform_is_inherited_r19_prefix_only=True,
                    inherited_r21_command_audit_does_not_audit_r22=True)
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    globals_["previous"] = proxy
    return FunctionType(original.__code__, globals_, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R22 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R22 isolates the frozen ycb_can:3414 development case")
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
