#!/usr/bin/env python3
"""Replay frozen RGB-D observations under declared synthetic sensing faults."""

from dataclasses import replace
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.camera import CameraObservation, save_observation
from kuka_sharpa_grasp.perception import largest_component, point_cloud
from kuka_sharpa_grasp.v7_scene import default_v7_scene_config
from kuka_sharpa_grasp.v7_vision import estimate_upright_cylinder


PROFILES = (
    ("ideal_replay", 0., 0., 0.),
    ("depth_1mm_dropout_5pct", .001, .05, 0.),
    ("depth_3mm_dropout_10pct", .003, .10, 0.),
    ("depth_5mm_dropout_20pct", .005, .20, 0.),
    ("appearance_bbox_occlusion_25pct", 0., 0., .25),
    ("appearance_bbox_occlusion_50pct", 0., 0., .50),
    ("appearance_bbox_occlusion_100pct", 0., 0., 1.),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrupt(observation, spec, seed, profile):
    name, noise, dropout, occlusion = profile
    rng = np.random.default_rng(seed)
    depth = observation.depth.copy()
    rgb = observation.rgb.copy()
    depth += rng.normal(0., noise, depth.shape)
    depth[(rng.random(depth.shape) < dropout) | (depth <= 0.)] = np.nan
    bounds = None
    if occlusion:
        cloud = point_cloud(observation)
        dominant = int(np.argmax(spec.rgba[:3]))
        appearance = rgb[..., dominant] > 20
        for channel in range(3):
            if channel != dominant:
                appearance &= rgb[..., dominant] > 1.5 * rgb[..., channel]
        near = np.linalg.norm(cloud[..., :2] - np.asarray(spec.pose[:2]), axis=-1) < .04
        near &= cloud[..., 2] > spec.pose[2] - spec.dimensions[1] + .008
        near &= cloud[..., 2] < spec.pose[2] + spec.dimensions[1] + .008
        candidate = largest_component(appearance & near & np.isfinite(cloud).all(axis=-1), cloud)
        ys, xs = np.nonzero(candidate)
        if len(xs):
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            width = max(1, round((x1 - x0) * occlusion))
            left = x0 + ((x1 - x0) - width) // 2
            rgb[y0:y1, left:left + width] = 0
            depth[y0:y1, left:left + width] = np.nan
            bounds = [left, y0, left + width, y1]
    return replace(observation, rgb=rgb, depth=depth), bounds


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), PROJECT / "src/kuka_sharpa_grasp/camera.py",
             PROJECT / "src/kuka_sharpa_grasp/perception.py", PROJECT / "src/kuka_sharpa_grasp/v7_vision.py"]
    hashes = {str(path.relative_to(PROJECT)): digest(path) for path in files}
    summary = json.loads((source / "summary.json").read_text())
    if not summary["complete"] or summary["protocol"]["source_changed_during_run"]:
        raise ValueError("source experiment must be complete and frozen")
    records = []
    inputs = []
    for seed in summary["protocol"]["seeds"]:
        directory = source / f"seed_{seed}" / "rgbd_shared"
        meta = json.loads((directory / "wrist_camera.json").read_text())
        observation = CameraObservation(np.asarray(Image.open(directory / meta["files"]["rgb"]).convert("RGB")),
            np.load(directory / meta["files"]["depth_metres"], allow_pickle=False),
            meta["timestamp_s"], meta["timestamp_s"], np.asarray(meta["camera_position_m"]),
            np.asarray(meta["camera_rotation_world"]), meta["focal_length_px"], meta["camera_name"])
        with gzip.open(directory / "trace.json.gz", "rt") as stream:
            trace = json.load(stream)
        frame_state = min(trace, key=lambda row: abs(row["time_s"] - observation.timestamp))
        if abs(frame_state["time_s"] - observation.timestamp) > 1e-8:
            raise ValueError("camera and evaluation timestamps differ")
        actual_xy = np.asarray(frame_state["evaluation"]["tool_marker"]["position"][:2])
        spec = default_v7_scene_config(seed).registry.by_id("tool_marker")
        for path in (directory / "wrist_camera.json", directory / "wrist_rgb.png",
                     directory / "wrist_depth_m.npy", directory / "trace.json.gz"):
            inputs.append({"path": str(path.relative_to(source)), "sha256": digest(path)})
        for index, profile in enumerate(PROFILES):
            noise_seed = seed * 100 + index
            modified, bounds = corrupt(observation, spec, noise_seed, profile)
            detection = estimate_upright_cylinder(modified, spec)
            error = float(np.linalg.norm(np.asarray(detection.position_m[:2]) - actual_xy)) if detection.position_m else None
            records.append({"seed": seed, "profile": profile[0], "noise_seed": noise_seed,
                "detected": detection.position_m is not None, "xy_error_m": error,
                "within_5mm": error is not None and error <= .005,
                "detection": detection.as_dict(), "occlusion_pixels_xyxy": bounds})
            if seed == summary["protocol"]["seeds"][0]:
                save_observation(modified, output / "examples" / profile[0])
    groups = {}
    for profile in PROFILES:
        rows = [row for row in records if row["profile"] == profile[0]]
        errors = [row["xy_error_m"] for row in rows if row["xy_error_m"] is not None]
        groups[profile[0]] = {"trials": len(rows), "detected": sum(row["detected"] for row in rows),
            "within_5mm": sum(row["within_5mm"] for row in rows),
            "accepted_mean_xy_error_m": float(np.mean(errors)) if errors else None,
            "accepted_max_xy_error_m": max(errors) if errors else None,
            "reasons": dict(Counter(row["detection"]["reason"] for row in rows))}
    after = {str(path.relative_to(PROJECT)): digest(path) for path in files}
    result = {"format": "v7_offline_synthetic_sensor_stress_r1", "source_experiment": str(source),
        "source_summary_sha256": digest(source / "summary.json"), "input_files": inputs,
        "profiles": [{"name": name, "depth_noise_std_m": noise, "dropout": drop,
                      "synthetic_bbox_centre_occlusion_fraction": occlusion}
                     for name, noise, drop, occlusion in PROFILES],
        "scope": "Offline perception replay on the same frozen frames; not closed-loop task or hardware sensor accuracy.",
        "input_boundary": "Known appearance/nominal prior define corruption bbox; actual body pose used after detection only to score XY error.",
        "source_before": hashes, "source_after": after, "source_changed_during_run": hashes != after,
        "results": records, "groups": groups}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(groups, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
