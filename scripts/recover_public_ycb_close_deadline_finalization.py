#!/usr/bin/env python3
"""Recover R18 artifact finalization in a new directory without running physics."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

import develop_public_ycb_close_deadline_r18 as runner
from analyze_public_ycb_vertical_load import digest, json_array
from audit_public_ycb_bias_recovery import checked_path
from evaluate_public_ycb import public_mesh_inventory


def running_controllers():
    matches = []
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if int(proc.parent.name) != os.getpid() and any(
                Path(os.fsdecode(arg)).name == "develop_public_ycb_close_deadline_r18.py"
                for arg in arguments):
            matches.append(int(proc.parent.name))
    return matches


def recover(original, output, profile):
    original, output, profile = original.resolve(strict=True), output.resolve(), profile.resolve(strict=True)
    if output.exists() or output.is_symlink() or output.is_relative_to(original):
        raise ValueError("Recovery requires a new directory outside the original run")
    if not Path("/proc").is_dir() or running_controllers():
        raise ValueError("Linux process inspection must confirm no R18 controller remains")
    summary_path, protocol_path = original / "summary.json", original / "protocol.json"
    original_hashes = {"summary.json": digest(summary_path), "protocol.json": digest(protocol_path)}
    summary, initial_protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    protocol = summary["protocol"]
    protocol_delta = sorted(key for key in initial_protocol.keys() | protocol.keys()
        if initial_protocol.get(key) != protocol.get(key))
    if (protocol_delta not in ([], ["actual_overview_gl"])
            or (protocol_delta and "actual_overview_gl" in initial_protocol)):
        raise ValueError("Initial/live protocols differ beyond the original runtime GL observation")
    if (summary["complete"] is not False
            or protocol["format"] != runner.PROTOCOL_FORMAT
            or protocol.get("source_hashes_after") or len(summary["rows"]) != 1
            or len(protocol["schedule"]) != 1):
        raise ValueError("Expected the unfinalized one-case R18 evidence, not an active or complete run")
    runner.validate_inputs(profile)
    before = protocol["source_hashes_before"]
    current = runner.snapshot(profile)
    if before != current or protocol["public_meshes"] != public_mesh_inventory():
        raise ValueError("Current sources/meshes differ from the original frozen inputs")
    archive_path = original / "frozen_source.zip"
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(before) or set(archive.namelist()) != set(before):
            raise ValueError("Frozen archive inventory differs")
        import hashlib
        if any(hashlib.sha256(archive.read(key)).hexdigest() != value for key, value in before.items()):
            raise ValueError("Frozen archive bytes differ")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    if row["runner_error"] is not None or any(row[key] != value for key, value in planned.items()):
        raise ValueError("The entire scheduled physical case must already be present")
    case = checked_path(original, row["path"])
    if json.loads((case / "metadata.json").read_text()) != row:
        raise ValueError("Original case metadata differs")
    result = json.loads((case / "result.json").read_text())
    if protocol_delta and protocol["actual_overview_gl"] != result["rendering"]:
        raise ValueError("Runtime GL addition differs from the physical result's recorded renderer")
    if (result["success"] != row["success"] or result["attempts"] != row["attempts"]
            or result["actual_held_lift_evidence"]["observed"] != row["held_lift"]
            or not 1 <= len(result["attempts"]) <= result["maximum_attempts"] == 2):
        raise ValueError("Physical result is missing or differs from the completed case row")
    bindings = {"frozen_source.zip": digest(archive_path)}
    for name in ("scene.xml", "derived_controller.py", "result.json", "metadata.json"):
        bindings[row["path"] + "/" + name] = digest(case / name)
    if (digest(case / "scene.xml") != row["scene_sha256_before_load"]
            or digest(case / "scene.xml") != row["scene_sha256_at_finalize"]
            or row["scene_unchanged_since_load"] is not True
            or digest(case / "derived_controller.py") != row["derived_controller_sha256"]):
        raise ValueError("Runtime scene/controller bindings differ")
    for artifact in row["artifacts"].values():
        path = checked_path(case, artifact["path"])
        if path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            raise ValueError("Missing or changed physical artifact")
        bindings[row["path"] + "/" + artifact["path"]] = artifact["sha256"]
    count, end = 0, None
    for count, sample in enumerate(json_array(case / row["artifacts"]["native"]["path"]), 1):
        end = sample["timestamp"]
        if abs(end - count * .002) > 1e-8:
            raise ValueError("Physical native samples are not a complete contiguous 2ms stream")
    audit = row["native_audit"]
    if (not count or count != audit["native_samples"] or count != audit["integration_steps"]
            or audit["external_target_qpos_writes"] != 0
            or abs(end - row["duration_s"]) > 1e-9 or abs(end - result["duration"]) > 1e-9):
        raise ValueError("The completed physical duration/denominator does not close")
    if running_controllers() or runner.snapshot(profile) != before:
        raise ValueError("Controller/source state changed during recovery inspection")
    for name, expected in {**original_hashes, **bindings}.items():
        if digest(original / name) != expected:
            raise ValueError("Original evidence changed during recovery inspection")
    receipt = {"format": "r18_artifact_finalization_recovery_v1", "simulation_rerun": False,
        "original_directory": str(original), "original_complete": False,
        "original_input_hashes": original_hashes, "physical_artifact_hashes": bindings,
        "initial_to_live_protocol_additions": protocol_delta,
        "observed_ssh_client_exit_code": 255, "runner_exit_code_observed": None,
        "original_runner_normal_exit_claimed": False, "physical_case_rows": len(summary["rows"]),
        "native_samples": count, "physical_duration_s": end,
        "source_hashes_at_recovery": current, "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(__file__), "independent_physics_or_task_audit_complete": False,
        "scope": "Original physical artifacts already existed. Only reporting finalization is reconstructed in a separate directory after process absence and source/artifact checks. The SSH disconnection cause and the exact physical end wall time are not independently proven. Complete denotes recovered artifact coverage, not normal runner exit, task success or the project goal. Original directory is preserved; native/strict/command/prefix audits are still required."}
    shutil.copytree(original, output)
    shutil.copy2(summary_path, output / "interrupted_summary.json")
    shutil.copy2(protocol_path, output / "interrupted_protocol.json")
    protocol = copy.deepcopy(protocol)
    protocol.update(source_hashes_after=current, source_changed_during_run=False,
        source_hashes_after_timing="Observed during explicit artifact recovery after runner process absence",
        public_meshes_unchanged=True, ended_at_utc=receipt["recovered_at_utc"],
        ended_at_utc_scope="Recovery finalization time; exact original physical end wall time is unknown",
        finalization_recovery=copy.deepcopy(receipt))
    summary = {**summary, "protocol": protocol, "complete": True, "runner_errors": 0,
        "original_runner_normal_exit": False, "artifact_finalization_recovered": True}
    for name, value in (("protocol.json", protocol), ("summary.json", summary), ("recovery_receipt.json", receipt)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, expected in {**original_hashes, **bindings}.items():
        if digest(original / name) != expected:
            raise ValueError("Original evidence was not preserved")
    print(json.dumps({"output": str(output), "simulation_rerun": False,
        "native_samples": count, "original_runner_normal_exit_claimed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    args = parser.parse_args()
    recover(args.original, args.output, args.profile_json)


if __name__ == "__main__":
    main()
