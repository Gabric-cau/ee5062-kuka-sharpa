#!/usr/bin/env python3
"""Plot peak-preserving native contact histories, including failed attempts."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_v7_robustness import atomic_json, sha256
from public_ycb_evidence import load_evidence_model
from verify_public_ycb_strict import CHANNELS, channel_pressures, hand_geometry


def load_case(directory, row):
    case = directory / row["path"]
    artifact = row["artifacts"]["native"]
    path = case / artifact["path"]
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"Native artifact differs: {path}")
    with gzip.open(path, "rt") as stream:
        native = json.load(stream)
    hand = hand_geometry(load_evidence_model(case / "scene.xml"))
    times = np.asarray([sample["timestamp"] for sample in native])
    values = np.asarray([[pressure[channel] for channel in CHANNELS]
        for sample in native
        for pressure in [channel_pressures(sample["native_contacts"], hand)]])
    if not len(times) or not np.isfinite(values).all():
        raise ValueError("Missing or invalid native samples")
    phases = []
    start = 0
    for end in range(1, len(native) + 1):
        if end == len(native) or native[end]["state"] != native[start]["state"]:
            phases.append((native[start]["state"], times[start], times[end - 1]))
            start = end
    return times, values, phases, {"path": str(path), "sha256": artifact["sha256"],
        "native_samples": len(times), "success": row["success"], "held_lift": row["held_lift"],
        "failure": row.get("failure"), "channel_peaks_n": dict(zip(CHANNELS, values.max(axis=0).tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for directory in args.directories:
        summary = json.loads((directory / "summary.json").read_text())
        if not summary["complete"]:
            raise ValueError("Plot only completed batches")
        cases.extend((directory, row) for row in summary["rows"])
    figure, axes = plt.subplots(len(cases), 1, figsize=(12, 2.7 * len(cases)), squeeze=False)
    records = []
    colors = ("#006e90", "#bb4430", "#648442", "#8152a0", "#b07b00", "#555555")
    for axis, (directory, row) in zip(axes[:, 0], cases, strict=True):
        times, values, phases, record = load_case(directory, row)
        # Keep each channel's maximum in every 50 ms bin, so 2 ms spikes survive.
        bins = np.floor(times / .05).astype(int)
        starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
        peaks = np.maximum.reduceat(values, starts, axis=0)
        for index, (channel, color) in enumerate(zip(CHANNELS, colors, strict=True)):
            axis.plot(times[starts], peaks[:, index], color=color, label=channel, linewidth=.9)
        for phase, start, end in phases:
            if phase == "CLOSE":
                axis.axvspan(start, end, color="#e8e8e8", zorder=-1)
        axis.axhline(12., color="#aa2222", linestyle="--", linewidth=.8)
        axis.set_ylim(0., max(13.5, values.max() * 1.12))
        axis.set_xlim(0., times[-1])
        axis.set_ylabel("Normal force (N)")
        axis.set_xlabel("Simulation time (s); shaded intervals: CLOSE")
        axis.set_title(f"{row['profile']} | {row['kind']} seed {row['seed']} | "
            f"peak {values.max():.3f} N | success={row['success']}, held={row['held_lift']}", fontsize=10)
        axis.grid(axis="y", alpha=.2)
        records.append(record)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    figure.suptitle("Complete native hand-contact histories | dashed line: 12 N assessment reference", fontsize=13)
    figure.tight_layout(rect=(0., .035, 1., .965))
    files = []
    for suffix in ("png", "pdf"):
        path = args.output / f"native_contact_histories.{suffix}"
        figure.savefig(path, dpi=160)
        files.append({"path": path.name, "sha256": sha256(path)})
    plt.close(figure)
    atomic_json(args.output / "plot_metadata.json", {"cases": records, "files": files,
        "display_bin_s": .05, "aggregation": "maximum per channel per bin, never mean",
        "scope": "All completed scheduled cases, attempts and recovery; plotting does not relabel task results"})
    print(json.dumps({"plotted_cases": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
