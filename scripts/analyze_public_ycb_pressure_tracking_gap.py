#!/usr/bin/env python3
"""Offline fixed-normal support checks at three prescribed R19 observations."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from compare_public_ycb_close_deadline import digest, json_array, safe_path
import tactile_wrench_allocator as allocator

PROJECT = Path(__file__).resolve().parents[1]
TIMES = (31., 34., 35.4)
STEP = .002
ACTIVE = ("thumb", "middle", "ring")
EXPECTED_SUMMARY_SHA256 = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
DEPENDENCIES = ("scripts/tactile_wrench_allocator.py",
    "scripts/compare_public_ycb_close_deadline.py")


def binding(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def fixed_checks(patches, com, mass, *, pressures=None, margins=(1., 1.5)):
    result = {}
    for margin in margins:
        solved = allocator.allocate_contact_wrench(patches, com, mass,
            measured_pressures_n=pressures, gravity_margin=margin)
        result[str(margin)] = {key: solved.get(key) for key in (
            "reason", "measured_feasible", "measured_capacity_feasible", "measured_solver_status",
            "measured_capacity_solver_status", "measured_wrench_residual", "measured_channel_pressures_n",
            "measured_within_target_limits", "palm_pressure_excluded_from_allocation_n",
            "solver_time_limit_per_lp_s", "friction_cone_rays", "friction_safety_factor")}
        result[str(margin)]["fixed_normal_scope"] = "All supplied per-patch normals are equality constraints; capacity is a separate upper-bound check"
    return result


def patch_loads(patches):
    return {finger: float(sum(float(p["measured_pressure_n"]) for p in patches
        if p["finger"] == finger)) for finger in allocator.CHANNELS}


def sample_analysis(sample):
    original = sample["original_probe_snapshot"]
    association = original["sampled_public_mesh_association"]
    target = sample["current_target_lp"]
    result = {"timestamp_s": sample["timestamp_s"], "phase": sample["phase"],
        "recorded_target_feasible": target.get("feasible"), "recorded_target_reason": target.get("reason"),
        "ready": sample["ready"], "continuous_seconds": sample["continuous_seconds"],
        "preload_reasons": sample["reasons"], "association_accepted": association["accepted"],
        "association_reason": association["reason"], "current_groups": sample["current_groups"],
        "recorded_measured_fixed_normal_1g_feasible": original["sampled_fixed_normal_1g_feasible"],
        "recorded_measured_capacity_1g_feasible": original["sampled_capacity_1g_feasible"],
        "all_contact_pressures_n": original["pressures_n"],
        "unassociated_pressure_n": association.get("unassociated_pressure_n"),
        "table_load_n": original["table_load_n"], "table_unloaded": original["table_unloaded"],
        "recorded_pressure_stable": original["pressure_stable"]}
    errors = [abs(g["target_n"] - g["measured_n"]) for g in sample["current_groups"].values()]
    result["maximum_group_error_n"] = max(errors) if errors else None
    result["all_recorded_groups_within_035_n"] = bool(errors) and max(errors) <= .35
    if not association["accepted"]:
        return dict(result, analysis_available=False, analysis_reason="RECORDED_ASSOCIATION_INVALID_NO_REPLACEMENT")
    patches = copy.deepcopy(association["patches"])
    com = association["estimated_com_world_m"]
    mass = original["probe_geometry_allocation_1_5g"]["mass_prior_kg"]
    if not math.isfinite(mass) or mass <= 0.:
        raise ValueError("Invalid recorded mass")
    result.update(estimated_com_world_m=com, mass_prior_kg=mass,
        all_associated_patch_count=len(patches), all_associated_channel_loads_n=patch_loads(patches),
        all_associated_actual=fixed_checks(patches, com, mass, pressures=original["pressures_n"], margins=(1.,)))
    actual = result["all_associated_actual"]["1.0"]
    result["recomputed_all_actual_matches_recorded_flags"] = (
        actual["measured_feasible"] == original["sampled_fixed_normal_1g_feasible"]
        and actual["measured_capacity_feasible"] == original["sampled_capacity_1g_feasible"])
    if not target.get("feasible") or not target.get("patches"):
        return dict(result, analysis_available=False,
            analysis_reason="RECORDED_TARGET_INVALID_NO_PREVIOUS_REFERENCE_RECONSTRUCTION")

    indexed = {int(p["input_index"]): p for p in patches}
    if len(indexed) != len(patches):
        raise ValueError("Duplicate association input index")
    raw = [int(p["raw_input_index"]) for p in target["patches"]]
    expected = [int(p["input_index"]) for p in patches
        if p["finger"] in ACTIVE and float(p["measured_pressure_n"]) >= .05]
    if len(raw) != len(set(raw)) or set(raw) != set(expected):
        raise ValueError("Recorded target retained set differs from original active/.05 N rule")
    retained, proposed, pairs = [], [], []
    for record, index in zip(target["patches"], raw):
        measured = indexed[index]
        if (record["finger"] != measured["finger"]
                or not np.allclose(record["position_world_m"], measured["position_world_m"], rtol=0., atol=1e-12)
                or not np.allclose(record["normal_on_object_world"], measured["normal_on_object_world"], rtol=0., atol=1e-12)
                or abs(record["measured_pressure_n"] - measured["measured_pressure_n"]) > 1e-12):
            raise ValueError("Gate target and original association are not the same observation")
        load = float(record["target_normal_n"])
        if not math.isfinite(load) or load < 0.:
            raise ValueError("Invalid recorded target normal")
        retained.append(measured)
        proposed.append(dict(measured, measured_pressure_n=load))
        pairs.append({"raw_input_index": index, "finger": measured["finger"],
            "actual_normal_n": measured["measured_pressure_n"], "target_normal_n": load,
            "target_minus_actual_n": load - measured["measured_pressure_n"]})
    excluded = [p for p in patches if int(p["input_index"]) not in raw]
    result.update(analysis_available=True, retained_patch_count=len(retained),
        excluded_patch_count=len(excluded), excluded_channel_loads_n=patch_loads(excluded),
        excluded_total_load_n=float(sum(p["measured_pressure_n"] for p in excluded)),
        excluded_patches=excluded, retained_patch_normals=pairs,
        retained_actual=fixed_checks(retained, com, mass, margins=(1.,)),
        retained_target=fixed_checks(proposed, com, mass),
        target_plus_excluded_actual=fixed_checks(proposed + excluded, com, mass),
        interpolation_definition="lambda=0 retains current actual normals; lambda=1 uses recorded target normals; excluded actual normals stay fixed in add-back variant",
        interpolation=[])
    for integer in range(11):
        fraction = integer / 10.
        interpolated = [dict(p, measured_pressure_n=(1. - fraction) * p["measured_pressure_n"]
            + fraction * q["measured_pressure_n"]) for p, q in zip(retained, proposed)]
        result["interpolation"].append({"lambda_target": fraction,
            "retained_only": fixed_checks(interpolated, com, mass),
            "with_excluded_actual": fixed_checks(interpolated + excluded, com, mass)})
    for key, endpoint in (("retained_actual", result["interpolation"][0]["retained_only"]),
                          ("retained_target", result["interpolation"][-1]["retained_only"]),
                          ("target_plus_excluded_actual", result["interpolation"][-1]["with_excluded_actual"])):
        for margin, check in result[key].items():
            if check["measured_feasible"] != endpoint[margin]["measured_feasible"]:
                raise ValueError("Repeated endpoint fixed-normal feasibility differs; no stable inference")
    return result


def analyze(directory, output):
    directory, output = directory.resolve(strict=True), output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new offline output directory")
    summary_path = safe_path(directory, "summary.json")
    if digest(summary_path) != EXPECTED_SUMMARY_SHA256:
        raise ValueError("This diagnosis is pinned to the finalized R19 baseline")
    summary = json.loads(summary_path.read_text())
    protocol = summary["protocol"]
    if (not summary["complete"] or summary["runner_errors"] or len(summary["rows"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("Expected one complete unchanged-source R19 case")
    row = summary["rows"][0]
    if row["kind"] != "ycb_can" or row["seed"] != 3414:
        raise ValueError("Unexpected case")
    metadata = safe_path(directory, Path(row["path"]) / "metadata.json")
    protocol_path = safe_path(directory, "protocol.json")
    if json.loads(metadata.read_text()) != row or json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Metadata/protocol differ from finalized summary")
    archive = safe_path(directory, "frozen_source.zip")
    manifest = protocol["source_hashes_before"]
    with zipfile.ZipFile(archive) as frozen:
        if len(frozen.namelist()) != len(set(frozen.namelist())) or set(frozen.namelist()) != set(manifest):
            raise ValueError("Frozen source archive membership differs")
        import hashlib
        if any(hashlib.sha256(frozen.read(name)).hexdigest() != sha for name, sha in manifest.items()):
            raise ValueError("Frozen source archive hash mismatch")
    for dependency in DEPENDENCIES:
        if digest(PROJECT / dependency) != manifest[dependency]:
            raise ValueError("Current offline dependency differs from frozen R19: " + dependency)
    artifact = row["artifacts"]["preload"]
    preload = safe_path(directory, Path(row["path"]) / artifact["path"])
    inputs = {str(path): binding(path) for path in
        (summary_path, protocol_path, metadata, archive, preload, Path(__file__),
         *(PROJECT / name for name in DEPENDENCIES))}
    if inputs[str(preload)] != {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}:
        raise ValueError("Preload artifact differs from finalized metadata")
    wanted = {round(t / STEP): t for t in TIMES}
    selected = []
    for sample in json_array(preload):
        time = float(sample["timestamp_s"])
        tick = round(time / STEP)
        if tick in wanted and abs(time - wanted[tick]) <= 1e-8:
            if sample["phase"] != "CLOSE":
                raise ValueError("Selected A1 observation is not CLOSE")
            selected.append(sample)
            del wanted[tick]
        if not wanted:
            break
    if wanted:
        raise ValueError("Missing exact prescribed timestamps: " + str(wanted))
    analyses = [sample_analysis(sample) for sample in selected]
    if any(binding(Path(path)) != bound for path, bound in inputs.items()):
        raise ValueError("Input/source changed during offline analysis")
    result = {"format": "offline_r19_pressure_tracking_gap_v1", "inputs_before": inputs,
        "inputs_after": copy.deepcopy(inputs), "inputs_unchanged": True,
        "source_hashes_before": copy.deepcopy(manifest), "source_hashes_after": copy.deepcopy(manifest),
        "source_archive_verified": True, "requested_timestamps_s": list(TIMES),
        "preload_decompression_passes": 1, "attempt": 1, "samples": analyses,
        "physics_rerun": False, "controller_changed": False, "previous_references_reconstructed": False,
        "interpolation_grid_only": True,
        "scope": "Three prescribed reused R19 estimated states only, without replacing invalid targets. Fixed-normal static compatibility is not measured tangential equilibrium, dynamic causation, robustness, realizable tracking, a lift, or a randomized success rate. Original .1s LP limits and all geometry/friction/force gates are preserved."}
    output.mkdir(parents=True)
    for name, value in (("selected_observations.json", selected), ("receipt.json", result)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(output), "selected_states": len(analyses),
        "valid_recorded_targets": sum(s["analysis_available"] for s in analyses),
        "all_actual_1g": [s.get("all_associated_actual", {}).get("1.0", {}).get("measured_feasible") for s in analyses]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.output)
