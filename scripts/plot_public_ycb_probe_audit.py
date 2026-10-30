#!/usr/bin/env python3
"""Compare recorded wrist motion, evaluated object lift and probe LP evidence."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_v7_robustness import atomic_json, sha256


def checked_stream(case, artifact):
    path = case / artifact["path"]
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"Recorded artifact differs: {path}")
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    if not summary["complete"] or len(summary["rows"]) != 1:
        raise ValueError("Expected one completed probe-development case")
    row = summary["rows"][0]
    case = args.directory / row["path"]
    events = checked_stream(case, row["artifacts"]["load_transfer_probe"])
    native = checked_stream(case, row["artifacts"]["native"])
    starts = sorted({event["probe_start_s"] for event in events
        if event.get("probe_start_s") is not None and event.get("probe_active")})
    if not starts:
        raise ValueError("No admitted probe; do not plot a nonexistent trajectory")
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, len(starts), figsize=(7 * len(starts), 9), squeeze=False)
    native_times = np.asarray([sample["timestamp"] for sample in native])
    target_z = np.asarray([sample["target_position_world_m"][2] for sample in native])
    attempts = []
    for column, start in enumerate(starts):
        selected = [event for event in events if event.get("probe_start_s") == start
            and (event.get("probe_active") or event.get("stop_reason"))]
        times = np.asarray([event["timestamp_s"] for event in selected])
        elapsed = times - start
        wrist_z = np.asarray([event["actual_probe_displacement_m"][2] for event in selected])
        # Target pose is independent evaluation only; never feed it to control.
        object_z = np.interp(times, native_times, target_z) - np.interp(start, native_times, target_z)
        motion, support, feasibility = axes[:, column]
        motion.plot(elapsed, wrist_z * 1000, label="Actual wrist", color="#006e90")
        motion.plot(elapsed, object_z * 1000, label="Object (evaluation only)", color="#bb4430")
        motion.axhline(5., linestyle="--", color="#555555", label="5 mm probe target")
        motion.set_ylabel("Vertical displacement (mm)")
        motion.set_title(f"Attempt {column + 1} | start {start:.3f} s")
        motion.legend(fontsize=8)
        fraction = np.asarray([event["table_load_n"] / event["catalogue_weight_n"] for event in selected])
        support.plot(elapsed, fraction * 100, color="#648442", label="Table / catalogue weight")
        support.axhline(20., linestyle="--", color="#555555", label="Unloading threshold")
        support.set_ylabel("Table support (%)")
        support.set_ylim(0., max(110., fraction.max() * 110))
        support.legend(fontsize=8)
        for field, label, color, offset in (
                ("sampled_capacity_1g_feasible", "Sampled capacity 1g", "#006e90", .025),
                ("sampled_fixed_normal_1g_feasible", "Sampled fixed-normal 1g", "#bb4430", 0.),
                ("original_exact_1_5g_feasible", "Original fixed-normal 1.5g", "#8152a0", -.025)):
            values = np.asarray([float(event[field]) for event in selected])
            feasibility.step(elapsed, values + offset, where="post", label=label, color=color)
        feasibility.set_yticks([0., 1.], ["False", "True"])
        feasibility.set_ylim(-.15, 1.15)
        feasibility.set_ylabel("LP compatibility")
        feasibility.legend(fontsize=8, loc="center right")
        failure = next((event["stop_reason"] for event in selected if event.get("stop_reason")), None)
        for axis in axes[:, column]:
            axis.set_xlabel("Time from actual probe admission (s)")
            axis.grid(axis="y", alpha=.2)
            if failure:
                axis.axvline(elapsed[-1], color="#555555", linestyle=":")
        attempts.append({"start_s": start, "last_s": float(times[-1]), "stop_reason": failure,
            "maximum_wrist_up_mm": float(wrist_z.max() * 1000),
            "maximum_object_up_mm": float(object_z.max() * 1000),
            "minimum_table_weight_fraction": float(fraction.min()),
            "native_motion_samples": len(selected)})
    figure.suptitle(f"Recorded load-transfer probes | task success={row['success']}, held={row['held_lift']}", fontsize=13)
    figure.tight_layout(rect=(0., 0., 1., .96))
    files = []
    for suffix in ("png", "pdf"):
        path = args.output / f"probe_diagnosis.{suffix}"
        figure.savefig(path, dpi=160)
        files.append({"path": path.name, "sha256": sha256(path)})
    plt.close(figure)
    atomic_json(args.output / "plot_metadata.json", {"case": row["path"], "attempts": attempts,
        "input_artifacts": {key: row["artifacts"][key] for key in ("native", "load_transfer_probe")},
        "files": files, "scope": "Recorded native probe trajectories; LP compatibility is not task success"})
    print(json.dumps(attempts, indent=2))


if __name__ == "__main__":
    main()
