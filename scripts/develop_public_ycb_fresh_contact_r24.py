#!/usr/bin/env python3
"""Finite current-contact tracking experiment over the completed R23 case."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
import zipfile

import develop_public_ycb_actual_wrist_probe_r23 as r23
import fresh_contact_tracking_controller_hooks as fresh


PROJECT = r23.PROJECT
PROTOCOL_FORMAT = "public_ycb_fresh_contact_tracking_development_r24"
DECLARATION = "r24_fresh_contact_tracking_experiment"
CONTROLLER_CLASS = "FreshContactTrackingHand"
BASELINE = PROJECT / "results/public_ycb_actual_wrist_probe_20261022_r23_bias10_close32_seed3414_retry1"
BASELINE_JOB = PROJECT / "results/jobs/r23_actual_wrist_probe_bias10_close32_seed3414_20261022_retry1"
BASELINE_SHA256 = "6b87f56530c87666b34eb45b177ed4b7732d7927f2a944138bef390ea260a210"
PROBE_RECEIPT = PROJECT / "results/r23_actual_wrist_probe_audit_20261022/receipt_retry1.json"
AUDIT_FORMAT = "r23_actual_wrist_probe_reference_native_and_retained_source_audit_v1"
AUDIT_CHECKS = ("all_evidence_consistent", "all_probe_reference_and_native_clock_checks_consistent",
    "all_probe_termination_and_original_confirmation_checks_consistent",
    "all_retained_arm_source_and_alignment_checks_consistent")
NEW_SOURCES = ("scripts/fresh_contact_tracking_controller_hooks.py",
    "tests/test_fresh_contact_tracking_controller_hooks.py",
    "docs/R24_FRESH_CONTACT_TRACKING_POLICY_20261023.md",
    "scripts/develop_public_ycb_fresh_contact_r24.py",
    "tests/test_public_ycb_fresh_contact_r24.py",
    "scripts/audit_public_ycb_actual_wrist_probe.py",
    "tests/test_audit_public_ycb_actual_wrist_probe.py")
sha256 = r23.sha256


def clone(function, **bindings):
    result = FunctionType(function.__code__, dict(function.__globals__, **bindings),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def controller_for(*args, **kwargs):
    # Change only the innermost hand factory; retain every executed arm extension.
    r21 = r23.r22.r21
    r17_proxy = SimpleNamespace(**{**vars(r21._controller_factory.__globals__["r17"]),
        "controller_for": fresh.controller_for})
    hand_factory = clone(r21._controller_factory, r17=r17_proxy)
    aligned_factory = clone(r23.r22.controller_for,
        r21=SimpleNamespace(**{**vars(r21), "controller_for": hand_factory}))
    return clone(r23.controller_for,
        r22=SimpleNamespace(**{**vars(r23.r22), "controller_for": aligned_factory}))(*args, **kwargs)


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    if r23.checked_sha256(expected_baseline_summary_sha256) != BASELINE_SHA256:
        raise ValueError("R24 requires the explicit finalized R23 summary SHA256")
    inherited = r23.validate_inputs(profile_path, r23.R22_SUMMARY_SHA256)
    files = {name: r23.checked_file(BASELINE, filename) for name, filename in
        (("summary", "summary.json"), ("protocol", "protocol.json"), ("archive", "frozen_source.zip"),
         ("native_receipt", "verification_receipt.json"), ("strict_receipt", "strict_native_verification.json"))}
    files.update(status=r23.checked_file(BASELINE_JOB, "status.json"),
        request=r23.checked_file(BASELINE_JOB, "request.json"),
        probe_receipt=r23.checked_file(PROJECT, PROBE_RECEIPT.relative_to(PROJECT)),
        probe_auditor=r23.checked_file(PROJECT, "scripts/audit_public_ycb_actual_wrist_probe.py"))
    hashes = {name: sha256(path) for name, path in files.items()}
    summary, protocol = (json.loads(files[name].read_text()) for name in ("summary", "protocol"))
    if (hashes["summary"] != BASELINE_SHA256 or summary["complete"] is not True
            or summary["runner_errors"] != 0 or summary["protocol"] != protocol
            or summary["runner_revision"] != r23.PROTOCOL_FORMAT
            or protocol["format"] != r23.PROTOCOL_FORMAT or len(summary["rows"]) != 1
            or protocol[r23.DECLARATION] != inherited
            or protocol["source_changed_during_run"] is not False
            or len(protocol["source_hashes_before"]) != 167
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("R24 baseline is incomplete, changed or not the executed R23 protocol")
    with zipfile.ZipFile(files["archive"]) as archive:
        before = protocol["source_hashes_before"]
        if len(archive.namelist()) != 167 or set(archive.namelist()) != set(before):
            raise ValueError("R23 source ZIP manifest differs")
        for relative, expected in before.items():
            if (sha256(r23.checked_file(PROJECT, relative)) != expected
                    or hashlib.sha256(archive.read(relative)).hexdigest() != expected):
                raise ValueError("Frozen R23 source differs: " + relative)
    row = summary["rows"][0]
    case = Path(row["path"])
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["runner_error"] is not None
            or row["success"] is not False or row["held_lift"] is not False
            or row["effective_hand_controller_types"] != [r23.CONTROLLER_CLASS] * 2
            or row["effective_hand_budgets_rad"] != [.10] * 2
            or [attempt["failure"] for attempt in row["attempts"]] != ["PROBE_BIAS_RECOVERY"] * 2
            or row["derived_controller_sha256"] != "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"):
        raise ValueError("R24 requires the actual two-attempt R23 failure case")
    for name, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"),
            ("metadata", "metadata.json"), ("result", "result.json")):
        files[name] = r23.checked_file(BASELINE, case / filename)
    for name in ("native", "trace", "load_transfer_probe", "bias_recovery"):
        artifact = row["artifacts"][name]
        files[name] = r23.checked_file(BASELINE, case / artifact["path"])
        if sha256(files[name]) != artifact["sha256"] or files[name].stat().st_size != artifact["bytes"]:
            raise ValueError("R23 raw artifact differs: " + name)
    hashes.update({name: sha256(path) for name, path in files.items()})
    if (json.loads(files["metadata"].read_text()) != row
            or hashes["controller"] != row["derived_controller_sha256"]
            or hashes["scene"] != row["scene_sha256_before_load"]
            or row["scene_sha256_at_finalize"] != hashes["scene"]):
        raise ValueError("R23 scene/controller/metadata binding differs")
    status, request = (json.loads(files[name].read_text()) for name in ("status", "request"))
    command = request["command"]
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or type(status["returncode"]) is not int
            or status["returncode"] != 0
            or request["supervisor_sha256"] != before["scripts/run_supervised_experiment.py"]
            or command.count("scripts/develop_public_ycb_actual_wrist_probe_r23.py") != 1
            or r23.r22._flag_argument(command, "--output") != BASELINE.relative_to(PROJECT).as_posix()
            or r23.r22._flag_argument(command, "--profile-json") != Path(profile_path).relative_to(PROJECT).as_posix()
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]):
        raise ValueError("R23 observed child exit or exact command differs")
    native, strict, probe = (json.loads(files[name].read_text())
        for name in ("native_receipt", "strict_receipt", "probe_receipt"))
    count = row["native_audit"]["native_samples"]
    if (any(value["summary_sha256"] != BASELINE_SHA256 for value in (native, strict, probe))
            or native["format"] != "public_ycb_development_native_verification_r1"
            or native["complete"] is not True or native["unknowns"] != 0
            or native["native_samples"] != count or native["task_successes"] != 0
            or native["audited_episodes"] != 1 or native["native_samples_contiguous_2ms"] is not True
            or native["rows"][0]["path"] != row["path"] or native["rows"][0]["unknown"] is not False
            or native["rows"][0]["outcome"] is not False or native["rows"][0]["held_lift"] is not False
            or native["source_hashes_before_equal_after"] is not True
            or native["frozen_source_sha256"] != hashes["archive"]
            or strict["format"] != "public_ycb_strict_native_audit_20261021"
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or strict["strict_successes"] != 0
            or strict["rows"][0]["path"] != row["path"] or strict["rows"][0]["unknown"] is not False
            or strict["rows"][0]["native_samples"] != count
            or strict["rows"][0]["result_sha256"] != hashes["result"]
            or strict["rows"][0]["scene_sha256"] != hashes["scene"]
            or probe["format"] != AUDIT_FORMAT or probe["accepted_probe_evidence"] is not True
            or probe["script_sha256"] != hashes["probe_auditor"]
            or any(probe["checks"][key] is not True for key in AUDIT_CHECKS)
            or probe["source"]["source_count"] != 167 or probe["native_samples"] != count
            or len(probe["probes"]) != 2
            or any(item["final_reason"] != "PROBE_BIAS_RECOVERY" for item in probe["probes"])):
        raise ValueError("R23 final native/strict/probe receipts are not accepted and bound")
    for name in ("summary", "protocol", "archive", "scene", "controller", "metadata", "result",
            "native", "trace", "load_transfer_probe", "bias_recovery"):
        binding = probe["inputs"][name]
        if binding["sha256"] != hashes[name] or binding["bytes"] != files[name].stat().st_size:
            raise ValueError("R23 independent probe receipt input differs: " + name)
    if any(sha256(path) != hashes[name] for name, path in files.items()):
        raise ValueError("R24 preflight evidence changed during verification")
    raw = {"native", "trace", "load_transfer_probe", "bias_recovery"}
    frozen = {path.relative_to(PROJECT).as_posix(): hashes[name]
        for name, path in files.items() if name not in raw}
    return {"format": "r24_fresh_contact_tracking_protocol_v1", "runner_revision": PROTOCOL_FORMAT,
        "inherited_r23_declaration": inherited, "inherited_r23_source_hashes": dict(before),
        "r23_comparison_summary_sha256": BASELINE_SHA256, "preflight_input_sha256": frozen,
        "inherited_r23_force_control": protocol["force_control"],
        "preflight_raw_artifact_bindings": {name: {"path": str(files[name].relative_to(PROJECT)),
            "sha256": hashes[name], "bytes": files[name].stat().st_size} for name in sorted(raw)},
        "changed_control": "Rebuild current contact Jacobians after an accepted current LP instead of invalidating solely on contact-sample change",
        "arm_source_unchanged_from_r23": True, "physics_and_strict_task_gates_unchanged": True,
        "additional_bias_limit_rad": .10, "close_seconds": 32., "arm_interpolation_seconds": 16.,
        "scope": "One reused can/seed, at most two attempts; no generalization or success claim."}


def snapshot(profile_path, declaration):
    result = {}
    for manifest in (declaration["inherited_r23_source_hashes"], declaration["preflight_input_sha256"]):
        for relative, expected in manifest.items():
            actual = sha256(r23.checked_file(PROJECT, relative))
            if actual != expected or relative in result and result[relative] != actual:
                raise ValueError("R24 frozen input changed: " + relative)
            result[relative] = actual
    for relative in NEW_SOURCES:
        result[relative] = sha256(r23.checked_file(PROJECT, relative))
    return result


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r23.tagged_protocol(value, declaration["inherited_r23_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": declaration["changed_control"]}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop(r23.DECLARATION, None)
        protocol.update(active, format=PROTOCOL_FORMAT,
            r24_fresh_contact_tracking_experiment=copy.deepcopy(declaration),
            inherited_r23_actual_wrist_probe_experiment=copy.deepcopy(declaration["inherited_r23_declaration"]),
            inherited_r23_force_control=declaration["inherited_r23_force_control"],
            force_control="R24 rebuilds current contact Jacobians after an accepted current association and LP, replacing only the R13 contact-change early invalidation. Current-box direction QP, R16 writer/recovery, R21 admission, R22 open ALIGN, R23 actual-wrist probe, physical scene and strict task gates are retained.")
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
        bindings[Path(args[1]).resolve()] = result[2]
        return result

    def bound_snapshot(profile, inherited):
        if inherited != declaration["inherited_r23_declaration"]:
            raise ValueError("R24 snapshot received a different inherited declaration")
        return snapshot(profile, declaration)

    factory = clone(r23.isolated_runner, controller_for=bound_controller, snapshot=bound_snapshot,
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    original = factory(profile_path, declaration["inherited_r23_declaration"])
    previous = original.__globals__["previous"]
    proxy = SimpleNamespace(**vars(previous))

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            hands = bindings.get(Path(path).parent.resolve())
            if hands is None:
                raise RuntimeError("Missing actual R24 hand factory binding")
            value.update(fresh_contact_tracking={"format": "r24_current_contact_tracking_v1",
                "hands": [hand.fresh_contact_telemetry() for hand in hands]},
                r23_probe_audit_does_not_audit_r24_hand=True)
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    return clone(original, previous=proxy)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R24 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R24 isolates ycb_can:3414")
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
