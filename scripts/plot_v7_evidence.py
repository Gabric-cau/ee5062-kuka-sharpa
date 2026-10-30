#!/usr/bin/env python3
"""Plot measured V7 outcomes and small-sample uncertainty from saved evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson(successes, total):
    if total <= 0:
        raise ValueError("a confidence interval needs a nonzero denominator")
    z = 1.959963984540054
    proportion = successes / total
    scale = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / scale
    width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / scale
    return max(0., centre - width), min(1., centre + width)


def plot(output, rgbd, sorting, rolling):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"rgbd": rgbd, "sorting": sorting, "rolling": rolling}
    data = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    protocol = data["rgbd"]["protocol"]
    if not data["rgbd"]["complete"] or protocol["source_changed_during_run"]:
        raise ValueError("plot requires a completed source-frozen RGB-D run")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    colours = ["#737C86", "#169578", "#BB5967"]
    modes = ("map_shared", "rgbd_shared", "rgbd_no_tactile")
    stats = [data["rgbd"]["modes"][mode]["rates"]["strict_physical_success"] for mode in modes]
    proportions, lower, upper = [], [], []
    intervals = {}
    for mode, stat in zip(modes, stats):
        if stat["unknown"]:
            raise ValueError("unknown outcomes must be resolved or explicitly plotted before reporting a rate")
        p = stat["numerator"] / stat["total"]
        lo, hi = wilson(stat["numerator"], stat["total"])
        intervals[mode] = {"successes": stat["numerator"], "total": stat["total"],
                           "rate": p, "wilson_95": [lo, hi]}
        proportions.append(100 * p)
        lower.append(max(0., 100 * (p - lo)))
        upper.append(max(0., 100 * (hi - p)))
    bars = axes[0].bar(range(3), proportions, color=colours, width=.6,
                       yerr=np.array([lower, upper]), capsize=4)
    axes[0].set_xticks(range(3), ["Map + tactile", "RGB-D + tactile", "RGB-D, no tactile"], rotation=15)
    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel("Strict physical success (%)")
    axes[0].set_title("Paired held-out marker trials")
    for bar, stat in zip(bars, stats):
        axes[0].text(bar.get_x() + bar.get_width() / 2, 106,
                     f"{stat['numerator']}/{stat['total']}", ha="center", fontsize=10)
    records = data["sorting"]["results"]
    labels = [record["object_id"].removeprefix("tool_").title() for record in records]
    errors = [record["task_result"]["metrics"]["dock_error_m"] * 1000 for record in records]
    axes[1].bar(labels, errors, color=colours)
    axes[1].axhline(25, color="#BB5967", linestyle="--", label="25 mm limit")
    axes[1].set_ylim(0, 29)
    axes[1].set_ylabel("Centre-to-slot error (mm)")
    axes[1].set_title("One continuous three-tool sequence")
    axes[1].legend(frameon=False, fontsize=9)
    forces = [record["peak_channel_force_n"] for record in records]
    axes[2].bar(labels, forces, color=colours)
    axes[2].axhline(12, color="#BB5967", linestyle="--", label="12 N channel limit")
    axes[2].set_ylim(0, 14)
    axes[2].set_ylabel("Peak native tactile channel (N)")
    axes[2].set_title("Measured during the same sequence")
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E0E3E6", linewidth=.6)
    task = protocol["task_spec"]
    fig.suptitle(f"MuJoCo V7 | {task['dock_tolerance_m'] * 1000:g} mm / "
                 f"{task['release_verify_seconds']:g} s | error bars: 95% Wilson intervals", fontsize=12)
    fig.savefig(output / "measured_outcomes.png", dpi=180)
    fig.savefig(output / "measured_outcomes.pdf")
    plt.close(fig)
    episodes = data["rolling"]["episodes"]
    seeds = sorted({row["seed"] for row in episodes})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for mode, colour, label in (("no_override_control", colours[0], "No index override"),
                                ("index_push", colours[1], "Index joint push")):
        selected = {row["seed"]: row for row in episodes if row["arm"] == mode}
        axes[0].plot(seeds, [selected[seed]["regrasp_evidence"]["relative_rotation_deg"]
                            for seed in seeds], "o-", color=colour, label=label)
        axes[1].plot(seeds, [selected[seed]["regrasp_evidence"]["wrist_translation_m"] * 1000
                            for seed in seeds], "o-", color=colour, label=label)
    axes[0].set_ylabel("Object rotation relative to wrist (degrees)")
    axes[1].set_ylabel("Wrist translation (mm)")
    axes[0].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xlabel("Paired seed")
        axis.set_xticks(seeds)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E0E3E6", linewidth=.6)
    fig.suptitle("Table-supported rolling | actuator-driven fingers | five paired trials", fontsize=12)
    fig.savefig(output / "supported_rolling.png", dpi=180)
    fig.savefig(output / "supported_rolling.pdf")
    plt.close(fig)
    manifest = {"format": "v7_measured_figures_r1", "inputs": {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in inputs.items()}, "marker_intervals": intervals,
        "scope": "Intervals describe these finite trials; they do not establish general tool-grasping accuracy.",
        "rolling_scope": data["rolling"]["scope"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgbd", type=Path, required=True)
    parser.add_argument("--sorting", type=Path, required=True)
    parser.add_argument("--rolling", type=Path, required=True)
    args = parser.parse_args()
    plot(args.output, args.rgbd, args.sorting, args.rolling)


if __name__ == "__main__":
    main()
