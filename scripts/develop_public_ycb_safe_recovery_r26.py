#!/usr/bin/env python3
"""Finite R26 safe-recovery experiment over the finalized R24/R25 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import develop_public_ycb_rate_limited_probe_r25 as r25
import safe_recovery_controller_hooks_r26 as recovery


PROJECT = r25.PROJECT
PROTOCOL_FORMAT = "public_ycb_safe_recovery_development_r26"
DECLARATION = "r26_safe_recovery_experiment"
CONTROLLER_CLASS = r25.CONTROLLER_CLASS
BASELINE = PROJECT / 'results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414'
BASELINE_SHA256 = '8b12661cd1949b6485e398a319cd3407688262205498751f97d0736dea086675'
NEW_SOURCES = (
    "scripts/safe_recovery_controller_hooks_r26.py",
    "scripts/develop_public_ycb_safe_recovery_r26.py",
    "tests/test_safe_recovery_controller_hooks_r26.py",
    "tests/test_public_ycb_safe_recovery_r26.py",
)
sha256 = r25.sha256
checked_file = r25.checked_file


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r25.r24.controller_for(*args, **kwargs)
    wrapped, rendered, telemetry = recovery.transform(original, source)
    return wrapped, rendered, hands, estimated


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    if expected_baseline_summary_sha256 != BASELINE_SHA256:
        raise ValueError('R26 requires the explicit finalized R25 summary SHA256')
    inherited = r25.validate_inputs(profile_path, r25.BASELINE_SHA256)
    files = {name: checked_file(BASELINE, name) for name in (
        'summary.json', 'protocol.json', 'frozen_source.zip', 'verification_receipt.json',
        'strict_native_verification.json')}
    hashes = {name: sha256(path) for name, path in files.items()}
    summary, protocol = (json.loads(files[name].read_text()) for name in ('summary.json', 'protocol.json'))
    before = protocol['source_hashes_before']
    if (hashes['summary.json'] != BASELINE_SHA256 or summary['complete'] is not True
            or summary['runner_errors'] != 0 or summary['protocol'] != protocol
            or protocol['format'] != r25.PROTOCOL_FORMAT
            or protocol['source_changed_during_run'] is not False
            or before != protocol['source_hashes_after'] or len(before) != 207
            or len(summary['rows']) != 1):
        raise ValueError('R26 requires the completed unchanged R25 run')
    row = summary['rows'][0]
    if (row['success'] is not False or row['held_lift'] is not False
            or row['failure'] != 'RECOVERY_CONTACT_GUARD_NOT_RELIEVED'
            or [item['failure'] for item in row['attempts']] != ['HELD_POLICY_FIXED_NORMAL_1G_INFEASIBLE']):
        raise ValueError('R26 baseline is not the observed R25 recovery failure')
    with zipfile.ZipFile(files['frozen_source.zip']) as archive:
        if set(archive.namelist()) != set(before) or len(archive.namelist()) != len(before):
            raise ValueError('R25 source archive manifest differs')
        for relative, expected in before.items():
            if (sha256(checked_file(PROJECT, relative)) != expected
                    or hashlib.sha256(archive.read(relative)).hexdigest() != expected):
                raise ValueError('Frozen R25 source changed: ' + relative)
    native, strict = (json.loads(files[name].read_text()) for name in (
        'verification_receipt.json', 'strict_native_verification.json'))
    if (native['summary_sha256'] != BASELINE_SHA256 or strict['summary_sha256'] != BASELINE_SHA256
            or native['complete'] is not True or native['native_samples_contiguous_2ms'] is not True
            or strict['accepted_evidence'] is not True or strict['strict_successes'] != 0):
        raise ValueError('R25 native/strict receipts are unaccepted or unbound')
    inherited = copy.deepcopy(inherited)
    inherited.update(
        format=PROTOCOL_FORMAT,
        runner_revision=PROTOCOL_FORMAT,
        inherited_r25_runner_revision=r25.PROTOCOL_FORMAT,
        inherited_r25_declaration=copy.deepcopy(inherited),
        changed_control=(
            "R25 filtered probe reference retained; brake, lower to original instrumented "
            "table support and release at a stationary wrist after recovery failures"
        ),
        safe_recovery_parameters=copy.deepcopy(recovery.PARAMETERS),
        safe_recovery_source_transform=True,
        original_task_and_strict_task_gates_unchanged=True,
        r25_baseline_summary_sha256=BASELINE_SHA256,
        r25_frozen_source_hashes=dict(before),
        r25_evidence_hashes={path.relative_to(PROJECT).as_posix(): hashes[name] for name, path in files.items()},
    )
    return inherited


def snapshot(profile_path, declaration):
    result = r25.snapshot(profile_path, declaration)
    for relative, expected in {**declaration.get('r25_frozen_source_hashes', {}),
            **declaration.get('r25_evidence_hashes', {})}.items():
        actual = sha256(checked_file(PROJECT, relative))
        if actual != expected or relative in result and result[relative] != actual:
            raise ValueError('R26 frozen R25 evidence changed: ' + relative)
        result[relative] = actual
    for relative in NEW_SOURCES:
        path = checked_file(PROJECT, relative)
        actual = sha256(path)
        if relative in result and result[relative] != actual:
            raise ValueError("R26 duplicate source hash differs: " + relative)
        result[relative] = actual
    return result


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r25.tagged_protocol(value, declaration)
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and 'nominal_profiles' in protocol:
        protocol["format"] = PROTOCOL_FORMAT
        protocol["runner_revision"] = PROTOCOL_FORMAT
        protocol["controller_revision_scope"] = declaration["changed_control"]
        protocol[DECLARATION] = copy.deepcopy(declaration)
        protocol["inherited_r25_rate_limited_probe_experiment"] = copy.deepcopy(
            declaration["inherited_r25_declaration"])
        protocol["safe_recovery"] = copy.deepcopy(recovery.PARAMETERS)
        protocol["force_control"] = (
            "R25 probe governor retained. R26 brakes while preserving the hand writer, "
            "lowers to original table-support confirmation, then releases at a fixed wrist. "
            "The 12N per-channel threshold is abort-only; it is not a force guarantee."
        )
        for planned in protocol.get("schedule", []):
            planned["runner_revision"] = PROTOCOL_FORMAT
            planned["controller_revision_scope"] = declaration["changed_control"]
    for row in value.get("rows", []):
        row["runner_revision"] = PROTOCOL_FORMAT
        row["controller_revision_scope"] = declaration["changed_control"]
    value["runner_revision"] = PROTOCOL_FORMAT
    value["controller_revision_scope"] = declaration["changed_control"]
    return value


def isolated_runner(profile_path, declaration):
    bindings = {}

    def bound_controller(*args, **kwargs):
        result = controller_for(*args, **kwargs)
        bindings[Path(args[1]).resolve()] = result[0]
        return result

    def bound_snapshot(profile, inherited):
        if inherited != declaration["inherited_r24_declaration"]:
            raise ValueError("R26 snapshot received a different inherited declaration")
        return snapshot(profile, declaration)

    factory = r25.r24.clone(r25.r24.isolated_runner, controller_for=bound_controller,
        snapshot=bound_snapshot,
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    original = factory(profile_path, declaration["inherited_r24_declaration"])
    previous = original.__globals__["previous"]
    proxy = copy.copy(previous)

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            task = bindings.get(Path(path).parent.resolve())
            if task is None:
                raise RuntimeError("Missing actual R26 safe-recovery binding")
            value.update(
                rate_limited_probe_source_transform=copy.deepcopy(task.r25_rate_limited_probe_transform),
                safe_recovery_source_transform=copy.deepcopy(task.r26_safe_recovery_transform),
                safe_recovery=task.safe_recovery_telemetry.as_dict(),
                inherited_r25_probe_audit_does_not_audit_r26_recovery=True,
            )
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    return r25.r24.clone(original, previous=proxy)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R26 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R26 isolates ycb_can:3414")
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

