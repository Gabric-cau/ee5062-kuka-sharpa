#!/usr/bin/env python3
"""Postrun R20 audit extension for the original hand initializer records only."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType
import zipfile

import audit_public_ycb_shared_normal as frozen


FORMAT = "r20_shared_normal_postrun_command_native_and_static_target_audit_v1"
FROZEN_AUDITOR_SHA256 = "c517348408603e70a6432e503ba19b186a4c3773c2336e2830b4ba0e9b34116e"
ORIGINAL_FAILED_RECEIPT_SHA256 = "aa456ddbcd40accab48be87b650d2c4deaa71917d6838991b7238b9c8c07f740"
R20_SUMMARY_SHA256 = "a47ab784ed6b0879f149333a08e02928b5ae5430ec93af76af67de02ef63dd7b"
INITIAL_TARGET = {"feasible": False, "reason": "NOT_EVALUATED"}
PREFIX_PHASES = ("SEARCH", "APPROACH", "ALIGN")
PROJECT = Path(__file__).resolve().parents[1]


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": frozen.base.digest(path)}


def validate_initializer(row):
    target = row["current_target"]
    if (row["event"] != "NOMINAL_REFERENCE_WRITE" or not isinstance(target, dict)
            or target != INITIAL_TARGET or target.get("feasible") is not False
            or row["phase"] not in PREFIX_PHASES or row["mode"] != "SCALAR_ACQUISITION"
            or row["target_age_s"] is not None or row["joint_solves"] != []
            or row["joint_reference_steps"] != {} or row["bounded_motion_predictions"] != []
            or any(row[key] != {} for key in ("candidate_command_rad", "applied_command_rad", "command_rad"))
            or row["safety_override_at_final_write"] is not False):
        raise ValueError("NOT_EVALUATED is not the exact empty pre-CLOSE hand initializer")


def classify_initialization_records(joint_events):
    """Require a complete pre-CLOSE prefix for each initial/new-attempt hand."""
    ordinals, blocks, phases = set(), [], Counter()
    epoch, previous_phase, previous_tick = None, None, None
    for ordinal, row in enumerate(joint_events):
        target = row.get("current_target")
        is_initializer = isinstance(target, dict) and target.get("reason") == "NOT_EVALUATED"
        if row.get("event") != "NOMINAL_REFERENCE_WRITE":
            if is_initializer:
                raise ValueError("Initializer target is attached to a non-write record")
            continue
        tick = frozen.base.tick(row["timestamp_s"])
        dt = frozen.base.finite(row["dt_s"], "initializer local write interval")
        phase = row["phase"]
        new_hand = abs(dt) <= 1e-12
        if epoch is None or new_hand:
            if (not new_hand or phase != "SEARCH" or not is_initializer
                    or (epoch is None and tick != 0)
                    or (epoch is not None and previous_phase != "RECOVER")):
                raise ValueError("Initializer epoch is not the initial or explicit post-RECOVER hand")
            if epoch is not None and epoch["first_close_timestamp_s"] is None:
                raise ValueError("A hand initializer epoch never entered CLOSE")
            epoch = {"controller_epoch": len(blocks) + 1, "first_record_index": ordinal,
                "last_record_index": ordinal, "first_timestamp_s": row["timestamp_s"],
                "last_timestamp_s": row["timestamp_s"], "write_count": 0,
                "first_close_timestamp_s": None}
            blocks.append(epoch)
            stage = 0
        else:
            frozen.base.equal(dt, frozen.base.STEP_S, "Noninitial local write interval", 1e-8)
        if previous_tick is not None and tick <= previous_tick:
            raise ValueError("Inherited write timestamps are not strictly ordered")
        previous_tick, previous_phase = tick, phase
        if epoch["first_close_timestamp_s"] is None:
            if phase == "CLOSE":
                if (not epoch["write_count"]
                        or tick != frozen.base.tick(epoch["last_timestamp_s"]) + 1):
                    raise ValueError("Initializer prefix is not adjacent to its first CLOSE write")
                epoch["first_close_timestamp_s"] = row["timestamp_s"]
            else:
                validate_initializer(row)
                current_stage = PREFIX_PHASES.index(phase)
                if current_stage < stage:
                    raise ValueError("Initializer phases moved backward before CLOSE")
                if epoch["write_count"] and tick != frozen.base.tick(epoch["last_timestamp_s"]) + 1:
                    raise ValueError("Initializer prefix has a physical-write gap")
                stage = current_stage
                ordinals.add(ordinal)
                phases[phase] += 1
                epoch.update(last_record_index=ordinal, last_timestamp_s=row["timestamp_s"],
                    write_count=epoch["write_count"] + 1)
        if is_initializer and ordinal not in ordinals:
            raise ValueError("Initializer target reappeared after the hand entered CLOSE")
    if not blocks or any(block["first_close_timestamp_s"] is None for block in blocks):
        raise ValueError("Missing complete initialized hand/CLOSE epochs")
    return {"record_indices": ordinals, "summary": {"all_contexts_valid": True,
        "initialization_records": len(ordinals), "original_stream_ordinals_preserved": True,
        "controller_epochs": len(blocks), "blocks": blocks, "phase_counts": dict(phases),
        "scope": "Only exact NOT_EVALUATED empty-reference writes in each hand's ordered SEARCH/APPROACH/ALIGN prefix, starting at t=0 or a zero-local-dt SEARCH after RECOVER and ending adjacent to its first CLOSE. The unchanged inherited native-command reconstruction independently validates each retry reset and actual commands. These records are initialization, not allocator solves."}}


def extended_target_audit():
    """Add one strict branch to the frozen target loop without changing ordinals."""
    source = Path(frozen.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_AUDITOR_SHA256:
        raise ValueError("The inherited frozen R20 auditor changed")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "audit_targets"]
    if len(functions) != 1:
        raise ValueError("Missing unique original target-audit function")
    function = copy.deepcopy(functions[0])
    marker = ast.parse('counts[path]["records_with_target"] += 1').body[0]
    matches = []
    for node in ast.walk(function):
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                matches.extend((value, index) for index, child in enumerate(value)
                    if isinstance(child, ast.stmt) and ast.dump(child) == ast.dump(marker))
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(matches) != 1 or len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise ValueError("Frozen target-loop transform anchor differs")
    branch = ast.parse('''if path == "tracking" and ordinal in _initialization["record_indices"]:
    counts[path]["initialization_records"] = counts[path].get("initialization_records", 0) + 1
    continue
''').body[0]
    body, position = matches[0]
    body.insert(position + 1, branch)
    function.body.insert(0, ast.parse('_initialization = _classify_initialization_records(joint_events)').body[0])
    returns[0].value.keys.append(ast.Constant("initializer_classification"))
    returns[0].value.values.append(ast.parse('_initialization["summary"]', mode="eval").body)
    transformed = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(frozen.audit_targets.__globals__, _classify_initialization_records=classify_initialization_records)
    exec(compile(transformed, str(Path(__file__).resolve()) + ":target_extension", "exec"), namespace)
    witness = {"kind": "single_initializer_branch_in_original_target_loop", "branch_insertions": 1,
        "input_records_filtered_or_reordered": False, "original_globals_modified": False,
        "transformed_ast_sha256": hashlib.sha256(ast.dump(transformed, include_attributes=False).encode()).hexdigest(),
        "inherited_function": "audit_public_ycb_shared_normal.audit_targets"}
    return namespace["audit_targets"], witness


def checked_failure(path):
    item = binding(path)
    receipt = json.loads(Path(path).read_text())
    if (item["sha256"] != ORIGINAL_FAILED_RECEIPT_SHA256
            or receipt != {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
                "error": "Unversioned target is not an original caller-gate rejection",
                "script_sha256": FROZEN_AUDITOR_SHA256}):
        raise ValueError("The original frozen-auditor failure receipt differs")
    return item


def audit_directory(directory, original_failed_receipt):
    directory = Path(directory).resolve(strict=True)
    original_failed_receipt = Path(original_failed_receipt).resolve(strict=True)
    paths = (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_shared_normal_postrun_audit.py",
        Path(frozen.__file__).resolve(), original_failed_receipt,
        frozen.base.checked_path(directory, "summary.json"), frozen.base.checked_path(directory, "protocol.json"))
    before = {str(path): binding(path) for path in paths}
    failure = checked_failure(original_failed_receipt)
    if frozen.base.digest(directory / "summary.json") != R20_SUMMARY_SHA256:
        raise ValueError("This postrun extension is bound to the completed original R20 case")
    target_audit, transform = extended_target_audit()
    isolated = FunctionType(frozen.audit_directory.__code__,
        dict(frozen.audit_directory.__globals__, audit_targets=target_audit),
        "postrun_r20_audit_directory", frozen.audit_directory.__defaults__, frozen.audit_directory.__closure__)
    result = isolated(directory)
    classification = result["cases"][0]["static_targets"]["initializer_classification"]
    if (classification["initialization_records"] != 3950 or classification["controller_epochs"] != 2
            or classification["controller_epochs"] != result["cases"][0]["effective_hand_instances"]):
        raise ValueError("The bound R20 initializer count/actual hand epochs differ")
    after = {str(path): binding(path) for path in paths}
    if before != after:
        raise ValueError("Postrun extension source, failed receipt or original inputs changed")
    inherited_format = result["format"]
    if result["script_sha256"] != FROZEN_AUDITOR_SHA256:
        raise ValueError("Inherited audit source binding differs")
    result.update(format=FORMAT, inherited_audit_receipt_format=inherited_format,
        script_sha256=frozen.base.digest(__file__), inherited_frozen_auditor_sha256=FROZEN_AUDITOR_SHA256,
        original_failed_receipt=failure, original_failed_receipt_sha256=failure["sha256"],
        initializer_classification=classification, target_audit_transform=transform,
        extension_inputs_before=before, extension_inputs_after=after, extension_inputs_unchanged=True,
        original_failed_audit_overwritten=False, original_runtime_source_modified=False,
        postrun_extension=True, original_frozen_auditor_succeeded=False)
    result["scope"] += (" This is a separately sourced postrun extension: the frozen auditor originally failed on"
        " NOT_EVALUATED initializer logs and that failure is preserved. Only strictly validated pre-CLOSE"
        " empty-reference initializer records receive a new counted branch. Original stream ordinals and all"
        " remaining target/native/source/command/cache checks are retained; this extension was not a runtime input.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--original-failed-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    failed = args.original_failed_receipt or args.directory.parent / "r20_shared_normal_audit_20261022/receipt.json"
    try:
        receipt = audit_directory(args.directory, failed)
        status = 0 if receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"] else 1
    except (ValueError, KeyError, OSError, TypeError, StopIteration, zipfile.BadZipFile) as error:
        receipt, status = {"format": "r20_shared_normal_postrun_audit_failure_v1", "accepted_evidence": False,
            "error": str(error), "script_sha256": frozen.base.digest(__file__),
            "inherited_frozen_auditor_sha256": FROZEN_AUDITOR_SHA256,
            "original_frozen_auditor_succeeded": False}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": frozen.base.digest(args.output), "exit_code": status}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
