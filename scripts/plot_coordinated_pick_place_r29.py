#!/usr/bin/env python3
"""Plot the complete recorded R29 development trajectory; no simulation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mujoco
import numpy as np

from render_coordinated_pick_place_r29 import CHANNELS, digest, finite, force_channels, load_completed


COLORS = ("#1664a0", "#ba4b62", "#128268", "#c17a16", "#6f55a5", "#505c66")
PHASE_COLORS = ("#dbe4ee", "#e5e4ce", "#e4dced", "#cfe4df", "#dde8ca",
                "#e9dccb", "#d3e7e8", "#ecd8e4", "#dddfe9", "#d9e8d8")
CAPTION = (
    "Recorded native R29 trajectories from one reused development case (seed 3414); the first 22 s are a verified motor-command replay prefix. "
    "Target height and XY position are privileged offline evaluation values, not controller observations. "
    "Six-channel curves sum absolute normal forces across all external hand contacts in the recorded private-forward native samples; "
    "separate native dynamics-cache maxima are shown in the force panel. The 12 N line is the abort gate, not a force invariant. "
    "The last panel shows maximum absolute normal force per target-hand / target-table contact, matching the >0.05 N loaded-contact predicate. "
    "Shaded bands retain every recorded phase. Threshold crossings alone do not establish all held/release predicates; task labels are copied from the completed result. "
    "This figure performs no new physics integration and makes no random-set performance or generalization claim."
)


def contact_peaks(model, native):
    target, table = model.geom("target_geom").id, model.geom("table").id
    hand = {geom for geom in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[geom])).name or "").startswith("left_")}
    hand_peak, table_peak = np.zeros(len(native)), np.zeros(len(native))
    for index, row in enumerate(native):
        for contact in row["native_contacts"]:
            pair = set(contact["geom_ids"])
            normal = abs(float(contact["force_torque_contact_frame"][0]))
            if target in pair and pair & hand:
                hand_peak[index] = max(hand_peak[index], normal)
            if pair == {target, table}:
                table_peak[index] = max(table_peak[index], normal)
    return hand_peak, table_peak


def phase_intervals(native, times):
    starts = [0] + [i for i in range(1, len(native)) if native[i]["state"] != native[i - 1]["state"]]
    return [{"state": native[first]["state"], "start_s": float(times[first] - .002),
        "end_s": float(times[end - 1]), "samples": end - first}
        for first, end in zip(starts, starts[1:] + [len(native)])]


def plot(directory, output):
    paths, hashes, result, native, times = load_completed(directory)
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("Existing directory is not an earlier derived plot output")
    model = mujoco.MjModel.from_binary_path(str(paths["compiled_model.mjb"]))
    qpos = finite([row["target_qpos"] for row in native], (len(native), 7))
    positions = finite([row["target_position_world_m"] for row in native], (len(native), 3))
    np.testing.assert_allclose(qpos[:, :3], positions, rtol=0., atol=1e-10)
    initial = finite(result["native_audit"]["initial_target_qpos"], (7,))
    zone = finite(result["zone"], (3,))
    height = (qpos[:, 2] - initial[2]) * 1000.
    xy_error = np.linalg.norm(positions[:, :2] - zone[:2], axis=1) * 1000.
    forces = force_channels(model, native)
    hand_peak, table_peak = contact_peaks(model, native)
    recorded_peaks = finite([result["strict_monitor"]["channel_peaks_n"][name] for name in CHANNELS], (6,))
    np.testing.assert_allclose(forces.max(axis=0), recorded_peaks, atol=1e-9, rtol=0.)
    cached = finite([result["native_cache_channel_peaks_n"][name] for name in CHANNELS], (6,))
    phases = phase_intervals(native, times)
    if [p["state"] for p in phases] != ["REPLAY"] + [p["state"] for p in result["transitions"]]:
        raise ValueError("Native phase sequence differs from the completed result")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "native_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_s", "phase", "target_rise_mm", "target_xy_zone_error_mm",
            *[name + "_external_normal_sum_n" for name in CHANNELS], "target_hand_contact_peak_n", "target_table_contact_peak_n"])
        for index, row in enumerate(native):
            writer.writerow([times[index], row["state"], height[index], xy_error[index],
                *forces[index], hand_peak[index], table_peak[index]])

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.865, hspace=.35)
    fig.suptitle("R29 | Recorded Lift, Transfer, Placement And Release", fontsize=16, y=.975)
    strict = result["strict_monitor"]
    fig.text(.5, .944, f"One reused development case | {len(native):,} native samples at 2 ms | "
        f"recorded task_success={str(result['success']).lower()} | failure={result['failure']}", ha="center", fontsize=10)
    handles = []
    for index, phase in enumerate(phases):
        color = PHASE_COLORS[index % len(PHASE_COLORS)]
        for axis in axes:
            axis.axvspan(phase["start_s"], phase["end_s"], color=color, alpha=.45, lw=0, zorder=0)
        handles.append(Patch(facecolor=color, label=phase["state"].replace("_", " ")))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .922), ncol=5,
        frameon=False, fontsize=8, handlelength=1.1, columnspacing=1.8)
    axes[0].plot(times, height, color="#1664a0", lw=1.3, label="Actual target freejoint rise")
    axes[0].axhline(70., color="#9d3446", ls="--", lw=1., label="Held-lift height threshold: >70 mm")
    axes[0].set(ylabel="Rise from initial (mm)", title="A  Actual target height")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=.9)
    axes[1].plot(times, xy_error, color="#128268", lw=1.3, label="Actual target-to-zone XY distance")
    axes[1].axhline(25., color="#9d3446", ls="--", lw=1., label="Release XY threshold: <25 mm")
    axes[1].set(ylabel="XY error (mm)", title="B  Actual placement-zone error")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=.9)
    for column, (name, color) in enumerate(zip(CHANNELS, COLORS)):
        axes[2].plot(times, forces[:, column], color=color, lw=.9, label=name, rasterized=True)
    axes[2].axhline(12., color="#9d3446", ls="--", lw=1., label="12 N abort gate")
    axes[2].set(ylabel="External channel sum (N)", title="C  Six-channel force histories (all native samples)", ylim=(-.25, 12.8))
    axes[2].legend(loc="upper right", ncol=7, fontsize=8, framealpha=.9)
    axes[2].text(.01, .77, f"Refreshed native max: {forces.max():.3f} N | "
        f"separate whole-run dynamics-cache max: {cached.max():.3f} N", transform=axes[2].transAxes,
        fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    axes[3].plot(times, hand_peak, color="#6f55a5", lw=1., label="Target-hand contact peak", rasterized=True)
    axes[3].plot(times, table_peak, color="#c17a16", lw=1., label="Target-table contact peak", rasterized=True)
    axes[3].axhline(.05, color="#9d3446", ls="--", lw=1., label="Loaded contact: >0.05 N")
    axes[3].set(ylabel="Per-contact normal (N)", xlabel="Actual simulation time (s)", title="D  Object contact and support evidence")
    axes[3].legend(loc="upper right", fontsize=9, ncol=3, framealpha=.9)
    for axis in axes:
        axis.set_xlim(0., times[-1])
        axis.grid(axis="y", alpha=.25)
        axis.axvline(22., color="#555555", ls=":", lw=1.1)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].text(22. + 1.3, .04, "22 s replay prefix ends", transform=axes[0].get_xaxis_transform(), fontsize=8)
    fig.text(.085, .145, f"Recorded strict monitor: held above 70 mm for {strict['held_seconds']:.3f} s; "
        f"released-stable interval {strict['released_stable_seconds']:.3f} s. "
        f"Final XY error {xy_error[-1]:.3f} mm. Labels are result records, not a new task score.", fontsize=9)
    fig.text(.085, .107, textwrap.fill(CAPTION, width=181), va="top", fontsize=8, linespacing=1.35)
    png, pdf = output / "trajectory_force.png", output / "trajectory_force.pdf"
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)
    pixels = plt.imread(png)
    pixel_std = float(pixels[..., :3].std())
    if pixel_std < .01:
        raise ValueError("Blank scientific plot")
    pdf_check = {"performed": False, "reason": "PyMuPDF unavailable"}
    try:
        import fitz
    except ImportError:
        pass
    else:
        with fitz.open(pdf) as document:
            if len(document) != 1:
                raise ValueError("Expected one scientific figure page")
            page = document[0]
            text = page.get_text()
            expected = ("70 mm", "25 mm", "12 N", "22 s", "reused development case", "recorded task_success=true")
            if any(label not in text for label in expected):
                raise ValueError("Expected threshold/scope labels absent from PDF")
            blocks = page.get_text("blocks")
            if any(block[0] < 0. or block[1] < 0. or block[2] > page.rect.width or block[3] > page.rect.height for block in blocks):
                raise ValueError("PDF text extends outside the page")
            (output / "pdf_text.txt").write_text(text, encoding="utf-8")
            pdf_check = {"performed": True, "tool": "PyMuPDF", "pages": 1, "text_blocks": len(blocks),
                "threshold_and_scope_labels_found": True, "text_inside_page": True}
    metrics = {"native_samples": len(native), "first_timestamp_s": float(times[0]), "final_timestamp_s": float(times[-1]),
        "recorded_task_success": result["success"], "recorded_failure": result["failure"],
        "target_initial_qpos": initial.tolist(), "zone_position_world_m": zone.tolist(),
        "maximum_target_rise_mm": float(height.max()), "final_target_rise_mm": float(height[-1]),
        "final_target_xy_error_mm": float(xy_error[-1]), "minimum_target_xy_error_mm": float(xy_error.min()),
        "native_channel_peaks_n": dict(zip(CHANNELS, forces.max(axis=0).tolist())),
        "recorded_dynamics_cache_peaks_n": dict(zip(CHANNELS, cached.tolist())),
        "native_target_hand_per_contact_peak_n": float(hand_peak.max()),
        "native_target_table_per_contact_peak_n": float(table_peak.max()), "phases": phases,
        "recorded_strict_checks": strict["checks"], "recorded_held_seconds": strict["held_seconds"],
        "recorded_released_stable_seconds": strict["released_stable_seconds"],
        "dynamics_integration_steps": 0, "target_state_scope": "Privileged offline evaluation only", "caption": CAPTION}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    if {name: digest(path) for name, path in paths.items()} != hashes:
        raise ValueError("Raw inputs changed during plotting")
    manifest = {"format": "r29_native_scientific_plot_v1", "input_directory": str(directory), "inputs_sha256": hashes,
        "source_sha256": {Path(__file__).name: digest(Path(__file__)),
            "render_coordinated_pick_place_r29.py": digest(Path(__file__).with_name("render_coordinated_pick_place_r29.py"))},
        "native_samples": len(native), "csv_rows": len(native), "runtime": {"platform": platform.platform(), "mujoco": mujoco.__version__},
        "verification": {"native_2ms_clock": True, "native_peaks_match_recorded_monitor": True,
            "qpos_and_body_position_agree": True, "phase_sequence_matches_result": True,
            "pdf_text_check": pdf_check, "png_pixel_std_0_1": pixel_std, "dynamics_integration_steps": 0},
        "outputs": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "native_samples": len(native),
        "maximum_target_rise_mm": metrics["maximum_target_rise_mm"], "final_xy_error_mm": metrics["final_target_xy_error_mm"],
        "native_peak_n": float(forces.max()), "cached_peak_n": float(cached.max()),
        "manifest_sha256": digest(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input.resolve(), args.output.resolve())
