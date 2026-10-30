#!/usr/bin/env python3
"""Plot saved R19 pressure-gap diagnostics without rerunning any LP or physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


RECEIPT_SHA256 = "29bd410bf3c3a448de90d8f77d74c1c6ebb7a71a4012965c12ff226fe1be0d44"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt_hash = digest(args.receipt)
    if receipt_hash != RECEIPT_SHA256:
        raise ValueError("Expected the exact finalized R19 pressure-gap receipt")
    source_hash = digest(Path(__file__))
    data = json.loads(args.receipt.read_text())
    rows = data["samples"]
    times = np.array([row["timestamp_s"] for row in rows])
    if (data["format"] != "offline_r19_pressure_tracking_gap_v1"
            or len(rows) != 3 or not np.allclose(times, [31., 34., 35.4], atol=1e-8, rtol=0)
            or data["inputs_unchanged"] is not True or data["physics_rerun"] is not False
            or any(row["analysis_available"] is not True for row in rows)
            or any(row["recomputed_all_actual_matches_recorded_flags"] is not True for row in rows)):
        raise ValueError("Expected the three finalized R19 offline diagnostics")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("R19: pressure tracking is distinct from load-bearing", fontsize=17)
    x = np.arange(3)
    colors = {"thumb": "#ad4050", "middle": "#2878a2", "ring": "#338463"}
    for index, finger in enumerate(colors):
        errors = []
        for row in rows:
            groups = [value for name, value in row["current_groups"].items()
                if name.split(":", 1)[0] == finger]
            if not groups:
                raise ValueError("Missing observed finger group")
            errors.append(max(abs(g["target_n"] - g["measured_n"]) for g in groups))
        axes[0, 0].bar(x + (index - 1) * .23, errors, width=.21,
            label=finger, color=colors[finger])
    axes[0, 0].axhline(.35, color="#454545", linestyle="--", label="0.35 N group limit")
    axes[0, 0].set(xticks=x, xticklabels=[f"{time:g} s" for time in times],
        ylabel="Largest absolute group error (N)", ylim=(0., .45),
        title="A. Group error meets tolerance at 34 and 35.4 s")
    axes[0, 0].legend(ncol=2, fontsize=9, loc="upper right")
    loads = [row["table_load_n"] for row in rows]
    axes[0, 1].bar(x, loads, color="#667b86", width=.52, label="Recorded table support")
    weights = np.array([row["mass_prior_kg"] * 9.81 for row in rows])
    if not np.allclose(weights, weights[0], atol=1e-12, rtol=0):
        raise ValueError("Object mass changes across the fixed observations")
    axes[0, 1].axhline(weights[0], color="#bb6a24", linestyle="--", label="Catalogue weight")
    for pos, load in zip(x, loads, strict=True):
        axes[0, 1].text(pos, load + .15, f"{load:.2f} N", ha="center", fontsize=10)
    axes[0, 1].set(xticks=x, xticklabels=[f"{time:g} s" for time in times],
        ylabel="Vertical force (N)", ylim=(0., max(loads) + 1.8),
        title="B. All three actual fixed-normal 1g checks fail")
    axes[0, 1].legend(fontsize=9, loc="upper right")
    cmap = ListedColormap(["#ba5863", "#438675"])
    norm = BoundaryNorm([-.5, .5, 1.5], 2)
    for axis, margin, title in ((axes[1, 0], "1.0", "C. Hypothetical 1g static compatibility"),
            (axes[1, 1], "1.5", "D. Hypothetical 1.5g static compatibility")):
        matrix, labels = [], []
        for row in rows:
            grid = row["interpolation"]
            if not np.allclose([point["lambda_target"] for point in grid], np.linspace(0., 1., 11), atol=1e-12):
                raise ValueError("Unexpected discrete interpolation grid")
            for key, label in (("retained_only", "retained"), ("with_excluded_actual", "+ weak actual")):
                values = [point[key][margin]["measured_feasible"] for point in grid]
                if any(type(value) is not bool for value in values):
                    raise ValueError("Missing static feasibility value")
                matrix.append(values)
                labels.append(f"{row['timestamp_s']:g} s {label}")
        axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        axis.set(xticks=np.arange(11), xticklabels=[f"{value:.1f}" for value in np.linspace(0., 1., 11)],
            yticks=np.arange(6), yticklabels=labels, xlabel="Interpolation fraction: 0 = actual normals, 1 = target normals",
            title=title)
        for i, row in enumerate(matrix):
            for j, feasible in enumerate(row):
                axis.text(j, i, "Y" if feasible else "N", color="white", ha="center", va="center", fontsize=9)
        axis.set_xticks(np.arange(-.5, 11, 1), minor=True)
        axis.set_yticks(np.arange(-.5, 6, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", bottom=False, left=False)
    for axis in axes[0]:
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.supxlabel("Saved offline samples only. Y/N = static LP check; no continuous threshold, dynamic fix or success claim.", fontsize=10)
    outputs = []
    for extension in ("png", "pdf"):
        path = args.output / ("pressure_tracking_gap." + extension)
        fig.savefig(path, dpi=160)
        outputs.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    plt.close(fig)
    if digest(args.receipt) != receipt_hash or digest(Path(__file__)) != source_hash:
        raise ValueError("Plot source or input changed")
    payload = {"format": "r19_pressure_gap_saved_receipt_plot_v1", "receipt_sha256": receipt_hash,
        "analysis_source_sha256": source_hash, "outputs": outputs, "physics_or_lp_rerun": False,
        "scope": "Plot of the saved discrete offline diagnosis. Group pressure error, table load and hypothetical fixed-normal LP compatibility remain separate observations."}
    with (args.output / "plot_receipt.json").open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
