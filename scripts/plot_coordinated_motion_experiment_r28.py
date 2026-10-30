#!/usr/bin/env python3
"""Plot recorded R28 motion/force evidence without running simulation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
ACTIVE = ("thumb", "middle", "ring")
CONDITIONS = ("STATIC_HOLD", "MOVING_WITHOUT_TRANSPORT", "MOVING_WITH_TRANSPORT")
LABELS = ("Static Hold", "Motion Without Transport", "Motion With Transport")
COLORS = ("#2864ab", "#8c6c3c", "#16835d", "#c45d25", "#6c54a3", "#b33045")
CAPTION = (
    "Three recorded 4 s branches from the same checked 22 s integration checkpoint; 2000 native steps per branch. "
    "Top: integrated actual wrist z and target freejoint qpos z relative to their branch initial values; moving-branch dashed lines mark the 5 mm motion-probe goal. "
    "Middle: per-step envelope of each channel across before, native dynamics cache and after refresh; 12 N is the recorded abort threshold, not a force invariant. "
    "Bottom: RMS force error across thumb, middle and ring, using the fixed R27 targets; solid is after refresh and dashed is the largest RMS among the three sample scopes. "
    "All branches and abort records are retained. task_success=None: this probe does not establish a 70 mm held lift, transfer, release, placement or generalization. "
    "Target state is offline evaluation only; plotting performs no new integration or contact-force reconstruction."
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def finite(values, shape=None):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or shape is not None and values.shape != shape:
        raise ValueError("Invalid recorded numeric shape or nonfinite data")
    return values


def channel_vector(values):
    if set(values) != set(CHANNELS):
        raise ValueError("Incomplete tactile channels")
    result = finite([values[name] for name in CHANNELS], (6,))
    if np.any(result < 0.):
        raise ValueError("Negative recorded channel force")
    return result


def native_channels(model, sample):
    hand = {}
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if body.startswith("left_"):
            hand[geom] = next((f for f in CHANNELS[:-1] if body.startswith("left_" + f + "_")), "palm")
    totals = dict.fromkeys(CHANNELS, 0.)
    for contact in sample["native_contacts"]:
        sides = set(contact["geom_ids"]) & hand.keys()
        if len(sides) == 1:
            totals[hand[next(iter(sides))]] += abs(float(contact["force_torque_contact_frame"][0]))
    return channel_vector(totals)


def collect(directory, condition, model):
    rows = read_gzip(directory / condition["artifacts"]["rows"]["path"])
    native = read_gzip(directory / condition["artifacts"]["native"]["path"])
    if len(native) != condition["native_steps"] or len(rows) < len(native):
        raise ValueError("Recorded branch counts differ")
    if condition["task_success"] is not None:
        raise ValueError("Expected a motion probe without a task-success label")
    start = condition["initial_robot_state"]["timestamp_s"]
    targets = channel_vector(condition["force_targets_n"])
    active = [CHANNELS.index(finger) for finger in ACTIVE]
    initial_wrist = finite(rows[0]["motion_reference"]["wrist_position_world_m"], (3,))
    initial_target = finite(condition["native_audit"]["initial_target_qpos"], (7,))
    times, envelopes, rms_after, rms_envelope, wrist_z, target_z = [], [], [], [], [], []
    peaks = {scope: np.zeros(6) for scope in ("before", "cache", "after")}
    squared_errors_after, aborts = [], []
    cache_count = after_count = 0
    native_by_time = {round(sample["timestamp"], 9): sample for sample in native}
    for index, row in enumerate(rows):
        before = channel_vector(row["before_tactile"]["measured_pressures_n"])
        scopes = [before]
        peaks["before"] = np.maximum(peaks["before"], before)
        end = float(row["before_time_s"])
        cache = row.get("native_dynamics_cache")
        if cache is not None:
            cache_count += 1
            cached = channel_vector(cache["channel_forces_n"])
            np.testing.assert_allclose(native_channels(model, cache["native_sample"]), cached, atol=1e-9, rtol=0.)
            scopes.append(cached)
            peaks["cache"] = np.maximum(peaks["cache"], cached)
            end = float(cache["step_end_s"])
            if abs(end - row["before_time_s"] - .002) > 1e-9:
                raise ValueError("Non-2ms native transition")
        after = row.get("after_tactile")
        if after is not None:
            after_count += 1
            refreshed = channel_vector(after["measured_pressures_n"])
            measured = native_by_time[round(after["timestamp_s"], 9)]
            np.testing.assert_allclose(native_channels(model, measured), refreshed, atol=1e-9, rtol=0.)
            if cache is not None:
                np.testing.assert_array_equal(measured["target_qpos"], cache["native_sample"]["target_qpos"])
            scopes.append(refreshed)
            peaks["after"] = np.maximum(peaks["after"], refreshed)
            error = targets[active] - refreshed[active]
            squared_errors_after.append(error ** 2)
            rms_after.append(float(np.sqrt(np.mean(error ** 2))))
            wrist_z.append(1000. * (row["after_wrist_pose"]["position_world_m"][2] - initial_wrist[2]))
            target_z.append(1000. * (measured["target_qpos"][2] - initial_target[2]))
        else:
            rms_after.append(np.nan)
            wrist_z.append(np.nan)
            target_z.append(np.nan)
        vectors = np.asarray(scopes)
        times.append(end - start)
        envelopes.append(np.max(vectors, axis=0))
        rms_envelope.append(float(np.max(np.sqrt(np.mean((targets[active] - vectors[:, active]) ** 2, axis=1)))))
        if row.get("abort") is not None:
            aborts.append({"row": index, "time_s": end - start, "record": row["abort"]})
    times, envelopes = np.asarray(times), np.asarray(envelopes)
    maximum = np.max(np.asarray(list(peaks.values())), axis=0)
    if abs(float(np.max(maximum)) - condition["peak_channel_force_n"]) > 1e-9:
        raise ValueError("Reported peak differs from the three-scope union")
    if native:
        stamps = np.array([sample["timestamp"] for sample in native])
        np.testing.assert_allclose(stamps, start + .002 * np.arange(1, len(native) + 1), atol=1e-9, rtol=0.)
    if condition["actual_wrist_displacement_m"] is not None and after_count:
        np.testing.assert_allclose(wrist_z[-1], 1000. * condition["actual_wrist_displacement_m"][2], atol=1e-9, rtol=0.)
    tail = np.asarray(squared_errors_after[-100:])
    initial_error = targets[active] - channel_vector(rows[0]["before_tactile"]["measured_pressures_n"])[active]
    metrics = {"condition": condition["condition"], "terminal": condition["terminal"],
        "completed": condition["completed"], "task_success": condition["task_success"],
        "rows": len(rows), "native_steps": len(native), "before_samples": len(rows),
        "native_cache_samples": cache_count, "after_samples": after_count, "aborts": aborts,
        "source_sample_count": len(rows) + cache_count + after_count,
        "initial_rms_force_error_n": float(np.sqrt(np.mean(initial_error ** 2))),
        "last_100_after_samples_rms_error_n": None if not len(tail) else float(np.sqrt(np.mean(tail))),
        "tail_samples": len(tail), "channel_peaks_n": dict(zip(CHANNELS, maximum.tolist())),
        "channel_peaks_by_scope_n": {scope: dict(zip(CHANNELS, values.tolist())) for scope, values in peaks.items()},
        "final_wrist_z_displacement_mm": None if not after_count else wrist_z[-1],
        "final_target_freejoint_z_displacement_mm": None if not after_count else target_z[-1],
        "peak_target_freejoint_z_displacement_mm": None if not after_count else float(np.nanmax(target_z)),
        "final_probe_goal_error_mm": None if not after_count else 1000. * condition["declared_probe_distance_m"] - wrist_z[-1],
        "force_targets_n": condition["force_targets_n"],
        "displacement_scope": "Target is integrated target_free qpos z, not cached target body xpos"}
    return {"times": times, "wrist_z": np.asarray(wrist_z), "target_z": np.asarray(target_z),
        "envelopes": envelopes, "rms_after": np.asarray(rms_after),
        "rms_envelope": np.asarray(rms_envelope), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    summary_path = directory / "summary.json"
    if sha(summary_path) != args.expected_summary_sha256:
        raise ValueError("Summary differs from the requested finalized run")
    summary = json.loads(summary_path.read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if (summary["protocol"] != protocol or summary["task_success"] is not None
            or tuple(c["condition"] for c in summary["conditions"]) != CONDITIONS
            or protocol["source_changed_during_run"] or protocol["inputs_changed_during_run"]):
        raise ValueError("Missing branch, changed inputs or inconsistent motion-probe protocol")
    if sha(directory / "compiled_model.mjb") != protocol["compiled_model_sha256"]:
        raise ValueError("Compiled model differs")
    if sha(directory / "new_integration_checkpoint.npz") != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs")
    for condition in summary["conditions"]:
        for record in condition["artifacts"].values():
            path = directory / record["path"]
            if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError("Incomplete or altered raw branch stream")
    input_paths = sorted(path for path in directory.iterdir() if path.is_file())
    input_hashes = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in input_paths}
    model = mujoco.MjModel.from_binary_path(str(directory / "compiled_model.mjb"))
    branches = [collect(directory, condition, model) for condition in summary["conditions"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and not args.overwrite_derived:
        raise ValueError("Preserve the existing visual-review manifest; choose a new output")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10., "axes.titlesize": 12.,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 10.5), sharex="col")
    fig.subplots_adjust(left=.07, right=.985, top=.87, bottom=.16, hspace=.40, wspace=.23)
    fig.suptitle("R28 | Coordinated Contact During a 5 mm Wrist Motion Probe", fontsize=17, y=.975)
    fig.text(.5, .935, "Same checked 22 s checkpoint | fixed R27 force targets | Recorded native R28 trajectories | task_success=None",
        ha="center", fontsize=10.5, color="#404040")
    force_max = max(12.8, 1.08 * max(float(np.max(b["envelopes"])) for b in branches))
    motion_values = np.r_[0., 5., *[b[k] for b in branches for k in ("wrist_z", "target_z")]]
    lower = min(-.3, float(np.nanmin(motion_values)) - .2)
    upper = max(5.6, float(np.nanmax(motion_values)) + .4)
    for column, (branch, label) in enumerate(zip(branches, LABELS, strict=True)):
        time, metrics = branch["times"], branch["metrics"]
        axes[0, column].set_title(label + "\n" + str(metrics["native_steps"]) + " native steps; " + metrics["terminal"], pad=12)
        axes[0, column].plot(np.r_[0., time], np.r_[0., branch["wrist_z"]], color="#2864ab", lw=1.8, label="Actual wrist z")
        axes[0, column].plot(np.r_[0., time], np.r_[0., branch["target_z"]], color="#b54c24", lw=1.5, label="Target freejoint z")
        if column:
            axes[0, column].axhline(5., color="#555555", ls="--", lw=1.1, label="5 mm probe goal")
        axes[0, column].set_ylim(lower, upper)
        for channel, color, name in zip(range(6), COLORS, CHANNELS, strict=True):
            axes[1, column].plot(time, branch["envelopes"][:, channel], lw=1.2, color=color, label=name)
        axes[1, column].axhline(12., color="#a22f36", ls="--", lw=1.1)
        axes[1, column].set_ylim(-.15, force_max)
        axes[2, column].plot(time, branch["rms_after"], color="#2864ab", lw=1.5, label="After refresh")
        axes[2, column].plot(time, branch["rms_envelope"], color="#b54c24", ls="--", lw=1., label="Worst of three scopes")
        axes[2, column].set_ylim(bottom=0.)
        for axis in axes[:, column]:
            axis.set_xlim(0., max(4., float(np.max(time))))
        axes[2, column].set_xlabel("Elapsed From Branch Start (s)")
        if metrics["aborts"]:
            for abort in metrics["aborts"]:
                for axis in axes[:, column]:
                    axis.axvline(abort["time_s"], color="#a22f36", ls=":")
    axes[0, 0].set_ylabel("Actual Relative z (mm)")
    axes[1, 0].set_ylabel("Channel Normal Load (N)\nBefore / Cache / After Envelope")
    axes[2, 0].set_ylabel("Active-Channel Force-Error RMS (N)")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    axes[0, 1].legend(loc="upper left", fontsize=8.5)
    axes[1, 0].legend(loc="upper right", ncol=2, fontsize=8.5)
    axes[1, 2].text(.98, 12.05 / force_max, "12 N abort threshold", transform=axes[1, 2].transAxes,
        ha="right", va="bottom", fontsize=8.5, color="#a22f36")
    axes[2, 0].legend(loc="upper right", fontsize=8.)
    fig.text(.07, .105,
        "Recorded evidence only. Target state is offline evaluation. A 5 mm wrist probe does not establish a 70 mm held lift or pick-and-place success.\n"
        "All 3 branches retained. Force envelopes include before, native dynamics cache and after refresh. Fixed targets are not a 1g support certificate.",
        fontsize=9., va="top", linespacing=1.6)
    for extension in ("png", "pdf"):
        fig.savefig(output / ("r28_motion_force_comparison." + extension), dpi=220, facecolor="white")
    plt.close(fig)
    (output / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    metrics = {"format": "r28_motion_visual_review_metrics_v1", "task_success": None,
        "conditions": [branch["metrics"] for branch in branches], "caption": CAPTION}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "timeseries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "elapsed_s", "actual_wrist_delta_z_mm", "target_freejoint_delta_z_mm",
            *[name + "_three_scope_envelope_n" for name in CHANNELS], "active_rms_after_n", "active_rms_worst_scope_n"])
        for branch in branches:
            for i, time in enumerate(branch["times"]):
                writer.writerow([branch["metrics"]["condition"], time, branch["wrist_z"][i], branch["target_z"][i],
                    *branch["envelopes"][i], branch["rms_after"][i], branch["rms_envelope"][i]])
    import pymupdf
    with pymupdf.open(output / "r28_motion_force_comparison.pdf") as document:
        text = "\n".join(page.get_text() for page in document)
        pages = len(document)
    expected_text = (*LABELS, "task_success=None", "12 N abort threshold", "5 mm probe goal")
    if not all(value in text for value in expected_text):
        raise ValueError("PDF extracted text is missing an expected label")
    (output / "pdf_extracted_text.txt").write_text(text, encoding="utf-8")
    if input_hashes != {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in input_paths}:
        raise ValueError("Recorded inputs changed while plotting")
    manifest = {"format": "r28_motion_visual_review_manifest_v1", "summary_sha256": sha(summary_path),
        "script_sha256": sha(Path(__file__)), "input_directory": str(directory), "inputs": input_hashes,
        "inputs_unchanged": True, "branch_count": len(branches),
        "native_steps": sum(branch["metrics"]["native_steps"] for branch in branches),
        "force_source_samples": sum(branch["metrics"]["source_sample_count"] for branch in branches),
        "aborted_branches": sum(bool(branch["metrics"]["aborts"]) for branch in branches),
        "task_success": None, "simulation_steps_performed": 0, "caption": CAPTION,
        "pdf_text_check": {"tool": "PyMuPDF", "pdftotext_binary_available": False, "pages": pages,
            "required_labels": list(expected_text), "all_present": True},
        "python_version": platform.python_version(), "matplotlib_version": matplotlib.__version__,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output / "manifest.json"), "conditions": metrics["conditions"]}, allow_nan=False))


if __name__ == "__main__":
    main()
