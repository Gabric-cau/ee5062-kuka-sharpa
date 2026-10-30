#!/usr/bin/env python3
"""Offline prospective group-proportional targets at three saved R19 states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import shared_normal_group_targets as grouped


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "results/r19_pressure_tracking_gap_20261022/selected_observations.json"
SELECTED_OBSERVATIONS_SHA256 = "abf67b03b1d3bae7d22fdf8bdcbc3f8d6e7ed64d165cfccd49dfe1ef35f3994e"
PRIOR_RECEIPT_SHA256 = "29bd410bf3c3a448de90d8f77d74c1c6ebb7a71a4012965c12ff226fe1be0d44"
R19_SUMMARY_SHA256 = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
SHARED_SHA256 = "f33434846456e02c063dc3224bd2f03cff5812b218aeab7b7f8de5ad88633101"
R13_SHA256 = "ea37244e99045959c6ad841477b2d275823c1599f1ac9624af9e94c16bf866ed"
TIMES = (31., 34., 35.4)


def binding(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixed_normal_checks(patches, com, mass):
    result = {}
    for margin in (1., 1.5):
        solved = grouped.shared.allocator.allocate_contact_wrench(patches, com, mass, gravity_margin=margin)
        result[str(margin)] = {key: solved.get(key) for key in ("measured_feasible", "measured_solver_status",
            "measured_wrench_residual", "measured_channel_pressures_n", "measured_within_target_limits",
            "normal_pressure_limit_n", "friction_cone_rays", "friction_safety_factor", "solver_time_limit_per_lp_s")}
    return result


def ratio_diagnostic(target, groups):
    if not target["feasible"]:
        return None
    targets = {patch["raw_input_index"]: patch["target_normal_n"] for patch in target["patches"]}
    rows = []
    for group in groups:
        values = [targets[index] for index in group["raw_input_indices"]]
        total = sum(values)
        residual = [value - proportion * total for value, proportion in zip(values, group["measured_proportions"])]
        rows.append({"finger": group["finger"], "hand_body_name": group["hand_body_name"],
            "raw_input_indices": group["raw_input_indices"], "target_patch_normals_n": values,
            "target_group_normal_n": total, "measured_proportions": group["measured_proportions"],
            "proportion_equality_residuals_n": residual})
    return rows


def analyze_observation(observation, prior_sample):
    probe = observation["original_probe_snapshot"]
    association, original_target = probe["sampled_public_mesh_association"], observation["current_target_lp"]
    if (observation["phase"] != "CLOSE" or association["accepted"] is not True
            or original_target["feasible"] is not True or prior_sample["analysis_available"] is not True
            or not np.isclose(observation["timestamp_s"], prior_sample["timestamp_s"], rtol=0., atol=1e-8)
            or probe["sampled_fixed_normal_1g_feasible"] is not False
            or prior_sample["recorded_measured_fixed_normal_1g_feasible"] is not False
            or prior_sample["all_associated_actual"]["1.0"]["measured_feasible"] is not False):
        raise ValueError("Saved R19 state/association/actual-negative evidence differs")
    original = {patch["raw_input_index"]: patch for patch in original_target["patches"]}
    patches = []
    for patch in association["patches"]:
        item = dict(patch)
        if patch["finger"] in grouped.shared.ACTIVE and patch["measured_pressure_n"] >= .05:
            source = original[patch["input_index"]]
            if source["finger"] != patch["finger"] or not source["hand_body_name"]:
                raise ValueError("Original retained hand-link identity is missing")
            for key in ("position_world_m", "normal_on_object_world", "measured_pressure_n"):
                if not np.allclose(source[key], patch[key], rtol=0., atol=1e-10):
                    raise ValueError("Saved target/association patch binding differs")
            item["hand_body_name"] = source["hand_body_name"]
        patches.append(item)
    com, mass = association["estimated_com_world_m"], probe["catalogue_weight_n"] / 9.81
    if (not np.allclose(com, prior_sample["estimated_com_world_m"], rtol=0., atol=1e-10)
            or abs(mass - prior_sample["mass_prior_kg"]) > 1e-12
            or probe["pressures_n"] != prior_sample["all_contact_pressures_n"]):
        raise ValueError("Saved estimated COM, catalogue mass or measured reference differs")
    baseline = grouped.shared.allocate_tracking_targets(patches, com, mass, probe["pressures_n"], None)
    candidate = grouped.allocate_group_proportional_targets(patches, com, mass, probe["pressures_n"])
    groups = candidate["prospective_group_constraints"]["groups"]
    variants = {}
    for name, target in (("shared_normal_previous_none", baseline), ("group_proportional_previous_none", candidate)):
        variant = {"target": target, "group_ratio_diagnostic": ratio_diagnostic(target, groups),
            "retained_hypothetical_fixed_normal_checks": None, "add_back_excluded_actual_fixed_normal_checks": None}
        if target["feasible"]:
            selected = {patch["raw_input_index"]: patch for patch in target["patches"]}
            retained = [dict(patch, measured_pressure_n=patch["target_normal_n"]) for patch in target["patches"]]
            all_patches = [dict(patch, measured_pressure_n=selected[patch["input_index"]]["target_normal_n"])
                if patch["input_index"] in selected else dict(patch) for patch in patches]
            variant["retained_hypothetical_fixed_normal_checks"] = fixed_normal_checks(retained, com, mass)
            variant["add_back_excluded_actual_fixed_normal_checks"] = fixed_normal_checks(all_patches, com, mass)
        variants[name] = variant
    retained_indices = {patch["input_index"] for patch in patches
        if patch["finger"] in grouped.shared.ACTIVE and patch["measured_pressure_n"] >= .05}
    excluded = [patch for patch in patches if patch["input_index"] not in retained_indices]
    return {"timestamp_s": observation["timestamp_s"], "estimated_com_world_m": com, "catalogue_mass_prior_kg": mass,
        "reference_policy": "previous=None for both prospective solves; reference equals current all-contact channel pressures, not the runtime smoothed reference",
        "recorded_runtime_reference_n_not_reconstructed": original_target["reference_n"],
        "actual_recorded_fixed_normal_1g_feasible": False,
        "actual_existing_receipt_fixed_normal_1g_feasible": prior_sample["all_associated_actual"]["1.0"]["measured_feasible"],
        "retained_patch_count": len(retained_indices), "excluded_patch_count": len(excluded),
        "excluded_actual_normals_n": {str(patch["input_index"]): patch["measured_pressure_n"] for patch in excluded},
        "original_r13_groups": groups, "variants": variants}


def run(input_path, prior_path, output):
    paths = (input_path, prior_path, Path(__file__).resolve(), Path(grouped.__file__).resolve(),
        Path(grouped.shared.__file__).resolve(), Path(grouped.r13.__file__).resolve(),
        Path(grouped.shared.allocator.__file__).resolve(), PROJECT / "tests/test_shared_normal_group_targets.py")
    before = {str(Path(path).resolve()): binding(path) for path in paths}
    if (before[str(input_path.resolve())]["sha256"] != SELECTED_OBSERVATIONS_SHA256
            or before[str(prior_path.resolve())]["sha256"] != PRIOR_RECEIPT_SHA256):
        raise ValueError("The exact saved R19 observations or original source receipt changed")
    if (binding(grouped.shared.__file__)["sha256"] != SHARED_SHA256
            or binding(grouped.r13.__file__)["sha256"] != R13_SHA256):
        raise ValueError("The frozen shared-normal LP or original grouping source changed")
    observations, prior = json.loads(input_path.read_text()), json.loads(prior_path.read_text())
    summary_bindings = [entry for path, entry in prior["inputs_before"].items() if path.endswith("/summary.json")]
    if (prior["format"] != "offline_r19_pressure_tracking_gap_v1" or prior["inputs_unchanged"] is not True
            or prior["inputs_before"] != prior["inputs_after"] or prior["physics_rerun"] is not False
            or len(summary_bindings) != 1 or summary_bindings[0]["sha256"] != R19_SUMMARY_SHA256
            or prior["requested_timestamps_s"] != list(TIMES) or len(observations) != 3 or len(prior["samples"]) != 3
            or any(abs(row["timestamp_s"] - timestamp) > 1e-8 for row, timestamp in zip(observations, TIMES))):
        raise ValueError("Expected the exact three saved R19 observations and their completed source receipt")
    samples = [analyze_observation(observation, sample) for observation, sample in zip(observations, prior["samples"])]
    after = {str(Path(path).resolve()): binding(path) for path in paths}
    if before != after:
        raise ValueError("Offline input or analysis source changed")
    result = {"format": "prospective_r19_group_proportional_shared_normal_feasibility_v1", "physics_rerun": False,
        "controller_changed": False, "runtime_frozen_source_modified": False, "previous_references_reconstructed": False,
        "inputs_before": before, "inputs_after": after, "inputs_unchanged": True, "requested_timestamps_s": list(TIMES),
        "samples": samples, "only_added_lp_constraints": "Original R13 retained-group measured proportion equalities on the high-gravity patch normal sums; original shared-patch equalities imply the same low-gravity ratios.",
        "scope": "Prospective static feasibility on three reused estimated R19 states, with previous=None in both comparisons. The exact shared-normal objective/constants and original .05N selection, 1g/1.5g, .75N active floors, 6N channel caps, 16 rays, mu*.85 and .1s per-LP limit are retained. Retained targets and add-back of excluded actual weak contacts are evaluated separately. Recorded actual 1g remains false; no dynamic realizability, robustness, causal benefit, equilibrium, lift, task success, randomized rate or R20 runtime result is claimed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(output), "sha256": binding(output)["sha256"],
        "group_proportional_feasible": [sample["variants"]["group_proportional_previous_none"]["target"]["feasible"] for sample in samples]}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prior-receipt", type=Path, default=DEFAULT_INPUT.parent / "receipt.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.input.resolve(strict=True), args.prior_receipt.resolve(strict=True), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
