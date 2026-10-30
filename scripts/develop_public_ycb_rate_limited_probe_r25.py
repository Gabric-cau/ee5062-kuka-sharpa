#!/usr/bin/env python3
"""Finite R25 reference-governor experiment over the completed R24 case."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import develop_public_ycb_fresh_contact_r24 as r24
import rate_limited_probe_controller_hooks as governor


PROJECT = r24.PROJECT
PROTOCOL_FORMAT = "public_ycb_rate_limited_probe_development_r25"
DECLARATION = "r25_rate_limited_probe_experiment"
CONTROLLER_CLASS = r24.CONTROLLER_CLASS
BASELINE = PROJECT / "results/public_ycb_fresh_contact_20261023_r24_bias10_close32_seed3414"
BASELINE_JOB = PROJECT / "results/jobs/r24_fresh_contact_bias10_close32_seed3414_20261023"
BASELINE_SHA256 = "bea38ee7cb966635eba211d5e7c93a98ef072d8f294877f204e68539665764ba"
HAND_RECEIPT = PROJECT / "results/r24_fresh_contact_audit_20261023/receipt.json"
AUDIT_FORMAT = "r24_fresh_contact_source_reconstruction_and_qp_linkage_audit_v1"
AUDIT_CHECKS = ("all_evidence_consistent", "all_frozen_source_and_recorded_method_bindings_consistent",
    "all_current_contact_reconstruction_and_qp_linkages_consistent")
NEW_SOURCES = ("scripts/rate_limited_probe_controller_hooks.py",
    "tests/test_rate_limited_probe_controller_hooks.py",
    "scripts/develop_public_ycb_rate_limited_probe_r25.py",
    "tests/test_public_ycb_rate_limited_probe_r25.py",
    "docs/R25_RATE_LIMITED_PROBE_POLICY_20261023.md",
    "docs/R24_PROBE_SPEED_DIAGNOSIS_20261023.md",
    "docs/R24_FRESH_CONTACT_AUDIT_ANALYSIS_NOTE_20261023.md",
    "scripts/audit_public_ycb_fresh_contact_r24.py",
    "tests/test_audit_public_ycb_fresh_contact_r24.py")
sha256 = r24.sha256
checked_file = r24.r23.checked_file


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r24.controller_for(*args, **kwargs)
    task, unchanged_source, telemetry = governor.transform(original, source)
    task.actual_wrist_probe_telemetry = telemetry
    return task, unchanged_source, hands, estimated


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    if r24.r23.checked_sha256(expected_baseline_summary_sha256) != BASELINE_SHA256:
        raise ValueError("R25 requires the explicit finalized R24 summary SHA256")
    files = {name: checked_file(BASELINE, filename) for name, filename in
        (("summary", "summary.json"), ("protocol", "protocol.json"), ("archive", "frozen_source.zip"),
         ("native_receipt", "verification_receipt.json"), ("strict_receipt", "strict_native_verification.json"))}
    files.update(status=checked_file(BASELINE_JOB, "status.json"), request=checked_file(BASELINE_JOB, "request.json"),
        hand_receipt=checked_file(PROJECT, HAND_RECEIPT.relative_to(PROJECT)),
        hand_auditor=checked_file(PROJECT, "scripts/audit_public_ycb_fresh_contact_r24.py"))
    hashes = {name: sha256(path) for name, path in files.items()}
    summary, protocol = (json.loads(files[name].read_text()) for name in ("summary", "protocol"))
    before = protocol["source_hashes_before"]
    if (hashes["summary"] != BASELINE_SHA256 or summary["complete"] is not True
            or summary["runner_errors"] != 0 or summary["protocol"] != protocol
            or summary["runner_revision"] != r24.PROTOCOL_FORMAT
            or protocol["format"] != r24.PROTOCOL_FORMAT or len(summary["rows"]) != 1
            or protocol["source_changed_during_run"] is not False or len(before) != 186
            or before != protocol["source_hashes_after"]):
        raise ValueError("R25 baseline is incomplete, changed or not the executed R24 protocol")
    with zipfile.ZipFile(files["archive"]) as archive:
        if len(archive.namelist()) != 186 or set(archive.namelist()) != set(before):
            raise ValueError("R24 source ZIP manifest differs")
        for relative, expected in before.items():
            if (sha256(checked_file(PROJECT, relative)) != expected
                    or hashlib.sha256(archive.read(relative)).hexdigest() != expected):
                raise ValueError("Frozen R24 source differs: " + relative)
    row = summary["rows"][0]
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["runner_error"] is not None
            or row["success"] is not False or row["held_lift"] is not False
            or row["effective_hand_controller_types"] != [CONTROLLER_CLASS] * 2
            or row["effective_hand_budgets_rad"] != [.10] * 2
            or [attempt["failure"] for attempt in row["attempts"]] != ["PROBE_SPEED"] * 2
            or row["derived_controller_sha256"] != "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"):
        raise ValueError("R25 requires the actual two-attempt R24 speed failure case")
    case = Path(row["path"])
    for name, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"),
            ("metadata", "metadata.json"), ("result", "result.json")):
        files[name] = checked_file(BASELINE, case / filename)
    raw = {"native", "trace", "joint_admittance", "bias_recovery"}
    for name in raw:
        artifact = row["artifacts"][name]
        files[name] = checked_file(BASELINE, case / artifact["path"])
        if sha256(files[name]) != artifact["sha256"] or files[name].stat().st_size != artifact["bytes"]:
            raise ValueError("R24 raw artifact differs: " + name)
    hashes.update({name: sha256(path) for name, path in files.items()})
    if (json.loads(files["metadata"].read_text()) != row
            or hashes["controller"] != row["derived_controller_sha256"]
            or hashes["scene"] != row["scene_sha256_before_load"]
            or hashes["scene"] != row["scene_sha256_at_finalize"]):
        raise ValueError("R24 metadata/scene/arm binding differs")
    status, request = (json.loads(files[name].read_text()) for name in ("status", "request"))
    command = request["command"]
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or type(status["returncode"]) is not int
            or status["returncode"] != 0 or command.count("scripts/develop_public_ycb_fresh_contact_r24.py") != 1
            or request["supervisor_sha256"] != before["scripts/run_supervised_experiment.py"]
            or r24.r23.r22._flag_argument(command, "--output") != BASELINE.relative_to(PROJECT).as_posix()
            or r24.r23.r22._flag_argument(command, "--profile-json") != Path(profile_path).relative_to(PROJECT).as_posix()
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]):
        raise ValueError("R24 observed child exit or exact command differs")
    native, strict, hand = (json.loads(files[name].read_text())
        for name in ("native_receipt", "strict_receipt", "hand_receipt"))
    count = row["native_audit"]["native_samples"]
    if (any(value["summary_sha256"] != BASELINE_SHA256 for value in (native, strict, hand))
            or native["complete"] is not True or native["unknowns"] != 0
            or native["native_samples"] != count or native["task_successes"] != 0
            or native["native_samples_contiguous_2ms"] is not True
            or native["source_hashes_before_equal_after"] is not True
            or native["frozen_source_sha256"] != hashes["archive"]
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or strict["strict_successes"] != 0
            or strict["rows"][0]["native_samples"] != count
            or strict["rows"][0]["result_sha256"] != hashes["result"]
            or strict["rows"][0]["scene_sha256"] != hashes["scene"]
            or hand["format"] != AUDIT_FORMAT or hand["accepted_fresh_contact_evidence"] is not True
            or hand["script_sha256"] != hashes["hand_auditor"]
            or hand["source"]["source_count"] != 186
            or any(hand["checks"][key] is not True for key in AUDIT_CHECKS)):
        raise ValueError("R24 final native/strict/current-contact receipts are not accepted and bound")
    for name in ("summary", "protocol", "archive", "scene", "controller", "metadata", "result", "joint_admittance", "bias_recovery"):
        if (hand["inputs"][name]["sha256"] != hashes[name]
                or hand["inputs"][name]["bytes"] != files[name].stat().st_size):
            raise ValueError("R24 current-contact receipt input differs: " + name)
    if any(sha256(path) != hashes[name] for name, path in files.items()):
        raise ValueError("R25 preflight evidence changed during verification")
    return {"format": "r25_rate_limited_probe_protocol_v1", "runner_revision": PROTOCOL_FORMAT,
        "inherited_r24_declaration": protocol[r24.DECLARATION], "inherited_r24_source_hashes": dict(before),
        "r24_comparison_summary_sha256": BASELINE_SHA256,
        "preflight_input_sha256": {path.relative_to(PROJECT).as_posix(): hashes[name]
            for name, path in files.items() if name not in raw},
        "preflight_raw_artifact_bindings": {name: {"path": files[name].relative_to(PROJECT).as_posix(),
            "sha256": hashes[name], "bytes": files[name].stat().st_size} for name in sorted(raw)},
        "inherited_r24_force_control": protocol["force_control"],
        "changed_control": "Filter and vector-rate-limit only the active unconfirmed probe Cartesian IK reference",
        "governor_parameters": copy.deepcopy(governor.PARAMETERS),
        "readable_arm_source_unchanged_from_r24": True,
        "runtime_probe_telemetry_binding_changed": True,
        "hand_physics_and_strict_task_gates_unchanged": True,
        "scope": "One reused can/seed, at most two attempts; no generalization or success claim."}


def snapshot(profile_path, declaration):
    result = {}
    for manifest in (declaration["inherited_r24_source_hashes"], declaration["preflight_input_sha256"]):
        for relative, expected in manifest.items():
            actual = sha256(checked_file(PROJECT, relative))
            if actual != expected or relative in result and result[relative] != actual:
                raise ValueError("R25 frozen input changed: " + relative)
            result[relative] = actual
    for relative in NEW_SOURCES:
        result[relative] = sha256(checked_file(PROJECT, relative))
    return result


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r24.tagged_protocol(value, declaration["inherited_r24_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": declaration["changed_control"]}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop(r24.DECLARATION, None)
        protocol.update(active, format=PROTOCOL_FORMAT,
            r25_rate_limited_probe_experiment=copy.deepcopy(declaration),
            inherited_r24_fresh_contact_tracking_experiment=copy.deepcopy(declaration["inherited_r24_declaration"]),
            inherited_r24_force_control=declaration["inherited_r24_force_control"],
            probe_reference_governor=copy.deepcopy(governor.PARAMETERS),
            force_control="R25 changes the active unconfirmed probe reference through an independently declared filtered vector rate limiter. The readable R24 arm source is unchanged, while its isolated runtime probe telemetry binding changes. R24 hand/current-Jacobian tracking, original actual-wrist speed abort, confirmation, physical parameters and strict task gates are retained.")
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
        if inherited != declaration["inherited_r24_declaration"]:
            raise ValueError("R25 snapshot received a different inherited declaration")
        return snapshot(profile, declaration)

    factory = r24.clone(r24.isolated_runner, controller_for=bound_controller, snapshot=bound_snapshot,
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    original = factory(profile_path, declaration["inherited_r24_declaration"])
    previous = original.__globals__["previous"]
    proxy = copy.copy(previous)

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            task = bindings.get(Path(path).parent.resolve())
            if task is None:
                raise RuntimeError("Missing actual R25 reference governor binding")
            value.update(rate_limited_probe_source_transform=copy.deepcopy(task.r25_rate_limited_probe_transform),
                inherited_r23_probe_audit_does_not_audit_r25_governor=True)
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    return r24.clone(original, previous=proxy)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R25 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R25 isolates ycb_can:3414")
    declaration = validate_inputs(profile_path, expected_baseline_summary_sha256)
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--expected-baseline-summary-sha256", required=True)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1])) for item in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases,
        expected_baseline_summary_sha256=args.expected_baseline_summary_sha256)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
