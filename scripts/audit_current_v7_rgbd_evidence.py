#!/usr/bin/env python3
"""Audit a complete frozen RGB-D cohort without importing its evaluator.

Success is independently recomputed from the attributed physical trace.
Contact attribution itself is not independently verified without a complete
native per-contact stream. Saved RGB/depth is independently reconstructed
and refit using the declared known-cylinder priors, with no controller run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit_current_v7_evidence import (
    EVIDENCE, PROJECT, close, lift_metrics, read_json, release_metrics,
    rotation, sha256, source_audit, time_audit, verify_artifacts,
)

TARGET = "tool_marker"
METRICS = ("strict_physical_success", "target_held_lift", "collision_free",
           "force_safe", "stable_released_in_slot")


def wilson95(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval needs 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    p, n = successes / total, float(total)
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., centre-half), min(1., centre+half)]


def paired_counts(first, second):
    if set(first) != set(second) or any(type(v) is not bool for v in (*first.values(), *second.values())):
        raise ValueError("Paired comparison requires identical seed keys and known boolean outcomes")
    both = sum(first[k] and second[k] for k in first)
    first_only = sum(first[k] and not second[k] for k in first)
    second_only = sum(not first[k] and second[k] for k in first)
    neither = sum(not first[k] and not second[k] for k in first)
    n = first_only + second_only
    p = 1. if n == 0 else min(1., 2*sum(math.comb(n, i) for i in range(min(first_only, second_only)+1)) / 2**n)
    return {"paired_seeds": len(first), "both_success": both, "first_only": first_only,
            "second_only": second_only, "both_failed": neither, "discordant_pairs": n,
            "exact_two_sided_mcnemar_p": p,
            "interpretation": "Descriptive exploratory paired result; no multiple-comparison adjustment or generalization claim."}


def inventory(directory):
    summary, protocol = read_json(directory/"summary.json"), read_json(directory/"protocol.json")
    expected = {(seed, mode) for seed in protocol["seeds"] for mode in protocol["modes"]}
    observed = [(r["seed"], r["mode"]) for r in summary["results"]]
    missing = []
    for row in summary["results"]:
        case = directory/row["path"]
        paths = [case/"metadata.json", case/"task_result.json", case.parent/"scene.xml",
                 case.parent/"nominal_manifest.json", case.parent/"actual_manifest.json"]
        for path in paths:
            if not path.is_file():
                missing.append(str(path.relative_to(directory)))
        for artifact in row["artifacts"].values():
            path = case/artifact["path"]
            if not path.is_file() or path.stat().st_size != artifact["bytes"]:
                missing.append(str(path.relative_to(directory)))
        if (case/"metadata.json").is_file():
            metadata = read_json(case/"metadata.json")
            for filename in metadata.get("camera_export", {}).get("files", {}).values():
                if not (case/filename).is_file():
                    missing.append(str((case/filename).relative_to(directory)))
    accepted = (summary.get("complete") is True and summary["protocol"] == protocol
                and set(observed) == expected and len(observed) == len(expected) and not missing)
    return {"accepted": accepted, "expected_episodes": len(expected), "summary_episodes": len(observed),
            "complete_flag": summary.get("complete"), "protocol_matches_summary": summary["protocol"] == protocol,
            "missing_or_partial_files": missing}, summary, protocol


def marker(manifest):
    return next(obj for obj in manifest["objects"] if obj["object_id"] == TARGET)


def fit_circle(points_xy, radius, prior):
    points = np.asarray(points_xy, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 20 or not np.isfinite(points).all() or radius <= 0:
        raise ValueError("Circle fit requires finite side points and a positive declared radius")
    fit = least_squares(lambda centre: np.linalg.norm(points-centre, axis=1)-radius,
                        np.asarray(prior, float), ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=200)
    if not fit.success:
        raise ValueError("Independent saved-frame circle fit failed")
    return fit.x, float(np.sqrt(np.mean(fit.fun**2)))


def randomization_audit(seed, nominal, actual, metadata, scene):
    rng = random.Random(0x50620000 + seed)
    expected = {"xy_error_m": [rng.uniform(-.005, .005), rng.uniform(-.005, .005)],
                "mass_factor": rng.uniform(.8, 1.2), "friction_factor": rng.uniform(.8, 1.2)}
    declared = metadata["perturbation"]
    old, new = marker(nominal), marker(actual)
    body = ET.parse(scene).getroot().find(f".//body[@name='v7_obj_{TARGET}']")
    if body is None:
        raise ValueError("Scene lacks target body")
    geom = body.find(f"geom[@name='v7_geom_{TARGET}']")
    if geom is None:
        raise ValueError("Scene lacks target geometry")
    checks = {"seed_draw_matches": all(close(expected[k], declared[k], atol=1e-12) for k in expected),
              "xy_manifest_matches_draw": close(np.asarray(new["pose"][:2])-old["pose"][:2], expected["xy_error_m"], atol=1e-12),
              "mass_manifest_matches_draw": close(new["mass_kg"], old["mass_kg"]*expected["mass_factor"], atol=1e-12),
              "friction_manifest_matches_draw": close(new["friction"], np.asarray(old["friction"])*expected["friction_factor"], atol=1e-12),
              "six_body_three_distractor_scene": len(actual["objects"]) == 6 and sum(o["role"] == "distractor" for o in actual["objects"]) == 3,
              "other_object_specs_unchanged": all(o == next(a for a in actual["objects"] if a["object_id"] == o["object_id"])
                                                  for o in nominal["objects"] if o["object_id"] != TARGET),
              "xml_target_pose_matches_manifest": close(np.fromstring(body.get("pos"), sep=" "), new["pose"]),
              "xml_target_mass_matches_manifest": close(float(geom.get("mass")), new["mass_kg"]),
              "xml_target_friction_matches_manifest": close(np.fromstring(geom.get("friction"), sep=" "), new["friction"]),
              "xml_target_size_matches_manifest": close(np.fromstring(geom.get("size"), sep=" "), new["dimensions"]),
              "xml_target_shape_matches_manifest": geom.get("type") == new["shape"]}
    return {"accepted": all(checks.values()), "checks": checks, "draw": expected,
            "actual_marker_position_m": new["pose"], "actual_marker_mass_kg": new["mass_kg"],
            "actual_marker_friction": new["friction"]}


def refit_saved_rgbd(case, metadata, target):
    export = metadata["camera_export"]
    files = export["files"]
    rgb = np.asarray(Image.open(case/files["rgb"]).convert("RGB"))
    depth = np.load(case/files["depth_metres"], allow_pickle=False)
    valid = np.asarray(Image.open(case/files["valid_mask"])) > 0
    h, w = depth.shape
    if rgb.shape != (h, w, 3) or valid.shape != depth.shape or [h, w] != export["shape"]:
        raise ValueError("Camera export dimensions disagree")
    if not np.array_equal(valid, np.isfinite(depth) & (depth > 0)):
        raise ValueError("Camera validity mask disagrees with metric depth")
    png_depth = np.asarray(Image.open(case/files["depth_png"]))
    expected_png = np.zeros(depth.shape, np.uint16)
    expected_png[valid] = np.clip(np.rint(depth[valid]*1000), 1, 65535).astype(np.uint16)
    if not np.array_equal(png_depth, expected_png) or not close(valid.mean(), export["finite_fraction"]):
        raise ValueError("Saved millimetre depth or finite fraction disagrees with metric depth")
    camera_rotation = np.asarray(export["camera_rotation_world"])
    if (not close(camera_rotation.T @ camera_rotation, np.eye(3)) or not close(np.linalg.det(camera_rotation), 1.)
            or not math.isfinite(export["focal_length_px"]) or export["focal_length_px"] <= 0):
        raise ValueError("Invalid recorded camera calibration")
    v, u = np.indices((h, w))
    local = np.stack(((u-(w-1)/2)*depth/export["focal_length_px"],
                      ((h-1)/2-v)*depth/export["focal_length_px"], -depth), axis=-1)
    cloud = local @ np.asarray(export["camera_rotation_world"]).T + export["camera_position_m"]
    dominant = int(np.argmax(target["rgba"][:3]))
    mask = rgb[..., dominant] > 20
    for c in set(range(3))-{dominant}:
        mask &= rgb[..., dominant] > 1.5*rgb[..., c].astype(float)
    prior = np.asarray(target["pose"])
    half_height, radius = target["dimensions"][1], target["dimensions"][0]
    mask &= np.isfinite(cloud).all(axis=-1)
    mask &= np.linalg.norm(cloud[..., :2]-prior[:2], axis=-1) < .04
    mask &= (cloud[..., 2] > prior[2]-half_height+.008) & (cloud[..., 2] < prior[2]+half_height+.008)
    y, x = np.where(mask)
    if len(y) < 30:
        raise ValueError("Insufficient saved visible RGB-D points")
    ids = np.full((h, w), -1, int)
    ids[y, x] = np.arange(len(y))
    edges_a, edges_b = [], []
    for dy, dx in ((1, 0), (0, 1)):
        yy, xx = y+dy, x+dx
        inside = (yy < h) & (xx < w)
        first = np.flatnonzero(inside)
        yy, xx = yy[inside], xx[inside]
        second = ids[yy, xx]
        joined = (second >= 0) & (np.linalg.norm(cloud[yy, xx]-cloud[y[first], x[first]], axis=1) <= .025)
        edges_a.extend(first[joined]); edges_b.extend(second[joined])
    graph = coo_matrix((np.ones(len(edges_a)), (edges_a, edges_b)), shape=(len(y), len(y)))
    _, labels = connected_components(graph, directed=False)
    biggest = int(np.argmax(np.bincount(labels)))
    visible = cloud[y[labels == biggest], x[labels == biggest]]
    top = prior[2]+half_height
    sides = visible[(visible[:, 2] < top-.012) & (visible[:, 2] > top-2*half_height+.012)]
    if len(sides) < 20:
        raise ValueError("Insufficient saved side points")
    centre, residual = fit_circle(sides[:, :2], radius, prior[:2])
    declared = metadata["detection"]
    delta = float(np.linalg.norm(centre-np.asarray(declared["position_m"][:2])))
    hashes = {key: {"path": filename, "sha256": sha256(case/filename), "bytes": (case/filename).stat().st_size}
              for key, filename in files.items()}
    return {"accepted": delta <= 1e-6 and len(visible) == declared["visible_points"] and len(sides) == declared["side_points"]
                              and abs(residual-declared["fit_residual_m"]) <= 1e-8,
            "independent_fit_xy_m": centre.tolist(), "recorded_fit_difference_m": delta,
            "fit_residual_m": residual, "visible_points": len(visible), "side_points": len(sides),
            "camera_artifact_hashes": hashes,
            "scope": "Saved RGB/depth with recorded camera extrinsics and declared known colour/radius/upright priors; not general 6D pose detection."}


def audit_episode(directory, record, protocol):
    case = directory/record["path"]
    metadata, result = read_json(case/"metadata.json"), read_json(case/"task_result.json")
    trace = read_json(case/"trace.json.gz")
    if not trace or trace[0]["time_s"] != 0 or EVIDENCE in trace[0]:
        raise ValueError("Expected exactly one initial uninstrumented state at t=0")
    rows = trace[1:]
    if not rows or any(EVIDENCE not in row for row in rows):
        raise ValueError("Incomplete integrated-step contact attribution")
    time = time_audit(trace)
    nominal, actual = read_json(case.parent/"nominal_manifest.json"), read_json(case.parent/"actual_manifest.json")
    scene = case.parent/"scene.xml"
    task = protocol["task_spec"]
    slot = next(s["pose"] for s in nominal["slots"] if s["slot_id"] == task["slot_id"])
    lift = lift_metrics(rows, result["phases"], TARGET, result["metrics"]["initial_object_position_m"][2])
    release = release_metrics(rows, result["phases"], task, slot)
    pose_release = release_metrics(rows, result["phases"], task, slot, finite_difference=True)
    peak = max(float(row[EVIDENCE]["maximum_hand_channel_force_n"]) for row in rows)
    collision_samples = sum(bool(row[EVIDENCE]["environment_collisions"]) for row in rows)
    strict = bool(result["success"] and lift["accepted"] and release["accepted"] and peak <= 12 and not collision_samples)
    raw = {"strict_physical_success": strict, "target_held_lift": lift["accepted"],
           "collision_free": collision_samples == 0, "force_safe": peak <= 12,
           "stable_released_in_slot": release["accepted"], "peak_hand_channel_force_n": peak,
           "collision_sample_count": collision_samples, "evidence_samples": len(rows),
           "expected_evidence_samples": len(rows),
           "peak_total_tactile_force_n": max(r["tactile"]["total_normal_force_n"] for r in trace),
           "dock_error_m": float(np.linalg.norm(np.asarray(trace[-1]["evaluation"][TARGET]["position"])-slot)),
           "final_upright_error_deg": math.degrees(math.acos(float(np.clip(rotation(trace[-1]["evaluation"][TARGET]["quaternion"])[2, 2], -1., 1.))))}
    mismatches = [key for key, value in raw.items() if not close(value, record["metrics"][key])]
    for key, value in lift.items():
        expected = record["metrics"]["lift_evidence"][key]
        if (value is None or expected is None) and value is not expected or value is not None and expected is not None and not close(value, expected):
            mismatches.append(f"lift:{key}")
    if not close(release["stable_seconds"], record["metrics"]["release_evidence"]["stable_seconds"]):
        mismatches.append("release:stable_seconds")
    velocity_errors = [max(abs(np.linalg.norm(r[EVIDENCE]["target_linear_velocity_world_m_s"])-r[EVIDENCE]["target_linear_speed_m_s"]),
                           abs(np.linalg.norm(r[EVIDENCE]["target_angular_velocity_world_rad_s"])-r[EVIDENCE]["target_angular_speed_rad_s"])) for r in rows]
    channel_errors = [abs(max(r[EVIDENCE]["hand_channel_forces_n"].values(), default=0)-r[EVIDENCE]["maximum_hand_channel_force_n"]) for r in rows]
    randomization = randomization_audit(record["seed"], nominal, actual, metadata, scene)
    detection = metadata["detection"]
    observation_time = metadata["camera_export"]["timestamp_s"]
    exposed = [r for r in trace if abs(r["time_s"]-observation_time) <= 1e-9]
    if len(exposed) != 1 or detection["position_m"] is None:
        raise ValueError("Cannot match saved RGB-D observation to a unique physical sample")
    physical_xy = np.asarray(exposed[0]["evaluation"][TARGET]["position"][:2])
    visual_error = float(np.linalg.norm(np.asarray(detection["position_m"][:2])-physical_xy))
    map_error = float(np.linalg.norm(np.asarray(marker(nominal)["pose"][:2])-physical_xy))
    refit = refit_saved_rgbd(case, metadata, marker(nominal))
    artifacts = verify_artifacts(case, record["artifacts"])
    checks = {"time_2ms": time["accepted"], "raw_compressed_hashes_match": artifacts["accepted"],
              "metadata_artifacts_match_summary": metadata["artifacts"] == record["artifacts"],
              "source_hashes_match_protocol": metadata["source_hashes_before"] == protocol["source_hashes_before"],
              "scene_hash_matches_metadata": sha256(scene) == metadata["scene_sha256"],
              "all_recomputed_metrics_match_summary": not mismatches,
              "executor_phase_completion_consistent": result["success"] == (len(result["phases"]) == 8 and all(p["status"] == "COMPLETE" for p in result["phases"])),
              "pose_difference_release_agrees": release["accepted"] == pose_release["accepted"],
              "velocity_norms_consistent": bool(max(velocity_errors) <= 1e-10),
              "channel_maxima_consistent": max(channel_errors) <= 1e-10,
              "seed_randomization_verified": randomization["accepted"], "saved_rgbd_independent_fit_matches": refit["accepted"],
              "shared_control_setting_matches_mode": metadata["shared_control"] == (record["mode"] != "rgbd_no_tactile"),
              "tactile_retention_setting_matches_mode": metadata["tactile_retention"] == (record["mode"] != "rgbd_no_tactile")}
    return {"path": record["path"], "seed": record["seed"], "mode": record["mode"],
            "accepted_trace_audit": all(checks.values()), "checks": checks, "time": time,
            "raw_recomputed": raw, "lift": lift, "release": release, "pose_difference_release": pose_release,
            "summary_metric_mismatches": mismatches, "randomization": randomization,
            "rgbd": {"error_at_exposure_xy_m": visual_error, "map_error_at_exposure_xy_m": map_error,
                     "position_estimate_m": detection["position_m"], "exposure_time_s": observation_time,
                     "physical_target_xy_m": physical_xy.tolist(), "independent_saved_frame_refit": refit},
            "scene_sha256": sha256(scene), "native_per_contact_independently_verified": False,
            "runtime_qpos_writes_independently_verified": False,
            "result_sha256": sha256(case/"task_result.json"), "metadata_sha256": sha256(case/"metadata.json")}


def audit_cohort(directory, output):
    available, summary, protocol = inventory(directory)
    if not available["accepted"]:
        raise ValueError(f"Cohort not complete on disk: {json.dumps(available)}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(summary["results"]):
        records.append(audit_episode(directory, row, protocol))
        if (index+1) % 10 == 0:
            print(f"audited {index+1}/{len(summary['results'])} episodes", flush=True)
    source = source_audit(protocol["source_hashes_before"], changed=protocol["source_changed_during_run"])
    source["before_after_identical"] = protocol["source_hashes_before"] == protocol["source_hashes_after"]
    groups, aggregate_mismatches = {}, []
    for mode in protocol["modes"]:
        selected = [r for r in records if r["mode"] == mode]
        success = sum(r["raw_recomputed"]["strict_physical_success"] for r in selected)
        groups[mode] = {"successes": success, "episodes": len(selected), "unknown": 0,
                        "success_rate": success/len(selected), "wilson95": wilson95(success, len(selected)),
                        "maximum_channel_peak_n": max(r["raw_recomputed"]["peak_hand_channel_force_n"] for r in selected),
                        "mean_rgbd_xy_error_m": float(np.mean([r["rgbd"]["error_at_exposure_xy_m"] for r in selected])),
                        "max_rgbd_xy_error_m": max(r["rgbd"]["error_at_exposure_xy_m"] for r in selected),
                        "mean_map_xy_error_m": float(np.mean([r["rgbd"]["map_error_at_exposure_xy_m"] for r in selected])),
                        "max_map_xy_error_m": max(r["rgbd"]["map_error_at_exposure_xy_m"] for r in selected)}
        for metric in METRICS:
            count = sum(r["raw_recomputed"][metric] for r in selected)
            expected = summary["modes"][mode]["rates"][metric]
            if expected["numerator"] != count or expected["denominator"] != len(selected) or expected["unknown"] != 0 or not close(expected["rate"], count/len(selected)):
                aggregate_mismatches.append(f"{mode}:{metric}")
    pair_checks, paired = [], {}
    for seed in protocol["seeds"]:
        cases = [r for r in records if r["seed"] == seed]
        pair_checks.append(len({r["scene_sha256"] for r in cases}) == 1 and all(r["randomization"] == cases[0]["randomization"] for r in cases))
    for first, second in (("map_shared", "rgbd_shared"), ("rgbd_shared", "rgbd_no_tactile"), ("map_shared", "rgbd_no_tactile")):
        if first in protocol["modes"] and second in protocol["modes"]:
            by_mode = {mode: {r["seed"]: r["raw_recomputed"]["strict_physical_success"] for r in records if r["mode"] == mode}
                       for mode in (first, second)}
            paired[f"{first}__vs__{second}"] = paired_counts(by_mode[first], by_mode[second])
    draws = [r["randomization"]["draw"] for r in records if r["mode"] == protocol["modes"][0]]
    unique_draws = len({json.dumps(draw, sort_keys=True) for draw in draws})
    accepted = (all(r["accepted_trace_audit"] for r in records) and source["accepted"] and source["before_after_identical"]
                and all(pair_checks) and unique_draws == len(protocol["seeds"]) and not aggregate_mismatches)
    receipt = {"format": "v7-rgbd-independent-trace-audit-r1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "auditor_sha256": sha256(__file__), "shared_audit_helpers_sha256": sha256(Path(__file__).with_name("audit_current_v7_evidence.py")),
               "cohort": str(directory), "summary_sha256": sha256(directory/"summary.json"), "protocol_sha256": sha256(directory/"protocol.json"),
               "inventory": available, "source": source, "accepted_trace_derived_audit": accepted,
               "full_native_per_contact_audit_passed": False, "simulation_only": True,
               "paired_scene_and_perturbation_identical": all(pair_checks), "unique_randomization_draws": unique_draws,
               "aggregate_summary_mismatches": aggregate_mismatches, "groups": groups, "paired_comparisons": paired,
               "episodes": records,
               "limitations": ["Contact force/collision attribution comes from recorded CalibrationContactReader fields; no full native per-contact stream was recorded.",
                               "Recorded native velocity vectors and norms are internally checked; original velocity computation is not rerun from a full saved dynamical state.",
                               "No runtime object-qpos write counter is present. Matching source hashes do not prove every executed side effect.",
                               "The target is one known upright synthetic cylinder, with known colour/radius/height and small XY/mass/friction perturbations; this is not unseen-tool generalization.",
                               "Same-seed paired scenes do not imply arbitrary environments or hardware transfer. Intervals and p values summarize only this fixed small cohort."]}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    (output/"statistics.json").write_text(json.dumps({"groups": groups, "paired_comparisons": paired}, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"accepted_trace_derived_audit": accepted, "groups": groups, "paired_comparisons": paired}, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT/"results/v7_rgbd_randomized_current_20261022_1600_1629")
    parser.add_argument("--output", type=Path, default=PROJECT/"results/current_rgbd_evidence_audit_20261022")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        state, _, _ = inventory(args.input)
        print(json.dumps(state, indent=2))
        return 0 if state["accepted"] else 2
    return 0 if audit_cohort(args.input, args.output)["accepted_trace_derived_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
