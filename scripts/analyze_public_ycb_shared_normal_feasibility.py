#!/usr/bin/env python3
"""Offline existence check for normals shared by 1.5g and 1g wrench solutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from compare_public_ycb_close_deadline import digest, json_array, safe_path
import tactile_wrench_allocator as allocator


TIMES = (19.622, 20.092, 24.188, 24.190, 35.400,
    68.620, 69.774, 79.870, 79.872, 83.900)
ACTIVE = ("thumb", "middle", "ring")


def shared_normals(patches, com, mass):
    retained = [dict(patch) for patch in patches if patch["finger"] in ACTIVE
        and float(patch["measured_pressure_n"]) >= .05]
    if {patch["finger"] for patch in retained} != set(ACTIVE):
        return {"feasible": False, "reason": "THREE_OBSERVED_ACTIVE_FINGERS_REQUIRED"}
    count, rays = len(retained), 16
    variables = count * rays
    fingers = np.zeros((len(allocator.CHANNELS), variables))
    normals = np.zeros((count, variables))
    directions, arms = [], []
    for index, patch in enumerate(retained):
        normal = np.asarray(patch["normal_on_object_world"], dtype=float)
        normal /= np.linalg.norm(normal)
        first, second = allocator._tangents(normal)
        mu = min(.8, float(patch.get("friction_coefficient", .8))) * .85
        theta = 2 * math.pi * np.arange(rays) / rays
        directions.extend(normal + mu * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second))
        arms.extend([np.asarray(patch["position_world_m"]) - np.asarray(com)] * rays)
        section = slice(index * rays, (index + 1) * rays)
        fingers[allocator.CHANNELS.index(patch["finger"]), section] = 1.
        normals[index, section] = 1.
    directions, arms = np.asarray(directions), np.asarray(arms)
    wrench = np.vstack([directions.T, np.cross(arms, directions).T])
    scaled = wrench.copy()
    scaled[3:] /= .05
    zeros = np.zeros_like(scaled)
    # Two friction-cone force fields share each patch's total normal pressure.
    equality = np.vstack([np.hstack([scaled, zeros]), np.hstack([zeros, scaled]),
        np.hstack([normals, -normals])])
    gravity = np.array([0., 0., float(mass) * 9.81, 0., 0., 0.])
    right = np.r_[1.5 * gravity, gravity, np.zeros(count)]
    active_rows = fingers[[allocator.CHANNELS.index(finger) for finger in ACTIVE]]
    inequality = np.vstack([np.hstack([fingers, np.zeros_like(fingers)]),
        np.hstack([-active_rows, np.zeros_like(active_rows)])])
    upper = np.r_[[0. if finger == "palm" else 6. for finger in allocator.CHANNELS],
        np.full(3, -.75)]
    solved = linprog(np.r_[np.ones(variables), np.full(variables, 1e-6)],
        A_ub=inequality, b_ub=upper, A_eq=equality, b_eq=right, bounds=(0., None),
        method="highs", options={"primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8, "time_limit": 2.})
    result = {"feasible": False, "solver_status": int(solved.status),
        "solver_message": solved.message, "patch_count": count,
        "objective": "Minimum shared total normal load, not the controller's tracking-reference objective"}
    if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
        return result
    high, low = solved.x[:variables], solved.x[variables:]
    high_residual, low_residual = wrench @ high - 1.5 * gravity, wrench @ low - gravity
    patch_loads = normals @ high
    normal_residual = normals @ high - normals @ low
    excess = float(np.max(inequality @ solved.x - upper))
    numerical = (max(np.max(abs(high_residual[:3])), np.max(abs(low_residual[:3]))) <= 1e-5
        and max(np.max(abs(high_residual[3:])), np.max(abs(low_residual[3:]))) <= 1e-6
        and np.max(abs(normal_residual)) <= 1e-6 and excess <= 1e-6)
    proposed = [dict(patch, measured_pressure_n=float(load))
        for patch, load in zip(retained, patch_loads, strict=True)]
    independent = {str(margin): allocator.allocate_contact_wrench(proposed, com, mass,
        gravity_margin=margin) for margin in (1., 1.5)}
    checks = {margin: {key: check[key] for key in
        ("measured_feasible", "measured_solver_status", "measured_wrench_residual")}
        for margin, check in independent.items()}
    result.update(feasible=bool(numerical and all(v["measured_feasible"] for v in checks.values())),
        explicit_numerical_checks_pass=bool(numerical), independent_fixed_normal_checks=checks,
        shared_patch_normals_n=patch_loads.tolist(),
        channel_targets_n=dict(zip(allocator.CHANNELS, (fingers @ high).tolist())),
        residual_1_5g=high_residual.tolist(), residual_1g=low_residual.tolist(),
        maximum_shared_normal_residual_n=float(np.max(abs(normal_residual))),
        maximum_inequality_excess_n=excess)
    return result


def analyze(directory, output):
    directory = directory.resolve(strict=True)
    summary_path = safe_path(directory, "summary.json")
    summary = json.loads(summary_path.read_text())
    if not summary["complete"] or len(summary["rows"]) != 1:
        raise ValueError("Expected one finalized artifact set")
    row = summary["rows"][0]
    metadata_path = safe_path(directory, Path(row["path"]) / "metadata.json")
    if json.loads(metadata_path.read_text()) != row:
        raise ValueError("Metadata differs from summary")
    artifact = row["artifacts"]["preload"]
    path = safe_path(directory, Path(row["path"]) / artifact["path"])
    inputs = {str(item): {"bytes": item.stat().st_size, "sha256": digest(item)}
        for item in (summary_path, metadata_path, path)}
    if inputs[str(path)] != {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}:
        raise ValueError("Preload artifact differs from its recorded binding")
    wanted = {round(time / .002): time for time in TIMES}
    samples = []
    for sample in json_array(path):
        tick = round(sample["timestamp_s"] / .002)
        if tick in wanted:
            original = sample["original_probe_snapshot"]
            association = original["sampled_public_mesh_association"]
            if not association["accepted"]:
                raise ValueError("Selected sample association is not accepted")
            mass = original["probe_geometry_allocation_1_5g"]["mass_prior_kg"]
            samples.append({"timestamp_s": sample["timestamp_s"],
                "recorded_target_feasible": sample["current_target_lp"]["feasible"],
                "recorded_target_reason": sample["current_target_lp"]["reason"],
                "shared_normal_existence": shared_normals(association["patches"],
                    association["estimated_com_world_m"], mass)})
            del wanted[tick]
        if not wanted:
            break
    if wanted:
        raise ValueError("Missing selected preload times")
    if any(item.stat().st_size != bound["bytes"] or digest(item) != bound["sha256"]
            for name, bound in inputs.items() for item in (Path(name),)):
        raise ValueError("Analysis input changed during reading")
    result = {"format": "offline_shared_normal_existence_v1", "inputs": inputs,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": samples, "physics_rerun": False, "controller_changed": False,
        "scope": "Selected reused development states only. Existence of hypothetical shared normals "
            "does not establish realizable pressure tracking, measured equilibrium, dynamic retention or task success. "
            "The objective differs from the original controller; previous target references are not reconstructed. "
            "Geometry, mass, friction, 16 cone rays, 0.75 N active floor and 6 N channel caps are retained."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "selected_states": len(samples),
        "shared_feasible": sum(sample["shared_normal_existence"]["feasible"] for sample in samples)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.output)
