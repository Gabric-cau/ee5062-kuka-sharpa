#!/usr/bin/env python3
"""Plot complete, already-audited R25/R26 loads; no dynamics or score recomputation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from audit_public_ycb_settled_alignment import json_array


ROOT = Path(__file__).resolve().parents[1]
CASE = "ycb_can_bias_budget_r17_seed3414"
CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
RUNS = {
    "R25": "public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414",
    "R26": "public_ycb_brake_supported_recovery_20261024_r26_seed3414",
}
TRACE_PINS = {
    "R25": "6b3f34d687c680e353cd628a0ef6ed31a3e2d22a8506fcdbeb2c4516864ce40c",
    "R26": "707291b019ab29615c9497b3fcf0020c4dce1b16800521e9edef0c799a8a796b",
}
RECOVERY_RECEIPT = ROOT / "results/r26_safe_recovery_audit_20261024/receipt_retry1.json"
RECOVERY_SHA256 = "31863354dc495298706bfe2a2ca393e332983214ea6ee23e2fd725e7adbe0986"
STRICT_PINS = {
    "R25": "768eda8d5d2f711d6c94c278263128ad9afc160f2d0651879e57e9912feeed67",
    "R26": "cdccebe9c5eb601bf43a2449badea908e1a57001e2e73bd04ac58b4e4a159932",
}
STAGE_COLORS = {"BRAKE": "#9199a2", "LOWER": "#4d9ca9", "RELEASE": "#d1ad53"}
CAPTION = (
    "One reused development case: public-YCB can, seed 3414. Attempt counts differ: R25 = 1; R26 = 2.\n"
    "Both full tasks failed; neither run achieved held lift (0 s). This is not a randomized success rate or a force invariant.\n"
    "Max channel = max(thumb, index, middle, ring, pinky, palm); every saved trace row is retained.\n"
    "R26 shading follows the audited BRAKE -> LOWER -> BRAKE -> RELEASE intervals."
)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def file_record(path):
    return {"path": str(Path(path).relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def require(value, message):
    if not value:
        raise ValueError(message)


def tick(value):
    require(type(value) in (int, float) and np.isfinite(value), "Nonfinite/nonnumeric timestamp")
    index = round(value / .002)
    require(index > 0 and abs(value - .002 * index) < 1e-7, "Timestamp is off the positive native grid")
    return index


def load_trace(label, stages):
    directory = ROOT / "results" / RUNS[label]
    trace_path = directory / CASE / "trace.json.gz"
    strict_path = directory / "strict_native_verification.json"
    result_path = directory / CASE / "result.json"
    require(digest(trace_path) == TRACE_PINS[label], label + " trace differs from the reviewed input")
    require(digest(strict_path) == STRICT_PINS[label], label + " strict receipt differs")
    strict = json.loads(strict_path.read_text())
    require(strict["accepted_evidence"] is True and len(strict["rows"]) == 1,
        label + " strict evidence is not accepted")
    evidence = strict["rows"][0]
    require(digest(result_path) == evidence["result_sha256"], label + " result binding differs")
    require(evidence["strict_success"] is False and evidence["checks"]["held_lift"] is False
        and evidence["held_seconds"] == 0., label + " no longer describes the expected failed task")
    result = json.loads(result_path.read_text())
    attempts = len(result["attempts"])
    require(result["success"] is False and result["actual_held_lift_evidence"]["observed"] is False
        and attempts == {"R25": 1, "R26": 2}[label], label + " task/attempt boundary differs")
    duration = float(result["duration"])
    del result
    rows, seen, previous = [], set(), 0
    for raw in json_array(trace_path):
        index = tick(raw["timestamp"])
        require(index >= previous and index <= evidence["native_samples"], "Trace clock is not monotonic/in range")
        previous = index
        seen.add(index)
        force = raw["hand_segment_forces"]
        require(set(force) == set(CHANNELS), "Unexpected/missing hand channel")
        loads = np.array([force[channel] for channel in CHANNELS], dtype=float)
        require(np.isfinite(loads).all() and np.all(loads >= 0.), "Invalid normal load")
        mode = ""
        if label == "R26":
            # Stage telemetry integrates samples after start and through its endpoint.
            matches = [stage["mode"] for stage in stages
                if round(stage["start_s"] / .002) < index <= round(stage["end_s"] / .002)]
            require(len(matches) <= 1, "Overlapping audited recovery stages")
            mode = matches[0] if matches else ""
        rows.append((float(raw["timestamp"]), index, raw["state"], *loads.tolist(), float(max(loads)), mode))
    require(seen == set(range(1, evidence["native_samples"] + 1)), label + " native coverage is incomplete")
    array = np.array([row[3:10] for row in rows], dtype=float)
    peaks = array[:, :6].max(axis=0)
    expected = [evidence["channel_peaks_n"][channel] for channel in CHANNELS]
    require(np.allclose(peaks, expected, rtol=0., atol=1e-9), label + " trace/strict channel peaks differ")
    require(abs(rows[-1][0] - duration) < 1e-7, label + " episode is truncated")
    maximum_row = int(np.argmax(array[:, 6]))
    return {"rows": rows, "times": np.array([row[0] for row in rows]), "forces": array[:, 6],
        "attempts": attempts, "duration_s": duration, "native_samples": len(seen),
        "trace_rows": len(rows), "duplicate_time_rows_retained": len(rows) - len(seen),
        "channel_peaks_n": dict(zip(CHANNELS, peaks.tolist(), strict=True)),
        "peak_n": float(peaks.max()), "peak_channel": CHANNELS[int(np.argmax(peaks))],
        "peak_timestamp_s": rows[maximum_row][0],
        "input_paths": [trace_path, strict_path, result_path]}


def draw(data, stages, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "path.simplify": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    figure.subplots_adjust(left=.085, right=.975, top=.83, bottom=.175, hspace=.48)
    figure.suptitle("Recorded contact loads during failed grasp episodes", x=.085, y=.975,
        ha="left", fontsize=17, fontweight="bold")
    figure.text(.085, .943, "R25 immediate release vs R26 brake-supported release | YCB can, development seed 3414",
        ha="left", fontsize=10, color="#41464d")
    figure.legend(handles=[Patch(facecolor=color, alpha=.3, label=mode)
        for mode, color in STAGE_COLORS.items()], loc="upper left", bbox_to_anchor=(.078, .925),
        frameon=False, ncol=3, title="Audited R26 recovery stages", title_fontsize=9, fontsize=9)
    for axis, label, panel in ((axes[0], "R25", "A"), (axes[1], "R26", "B"), (axes[2], "R26", "C")):
        item = data[label]
        if label == "R26":
            for stage in stages:
                axis.axvspan(stage["start_s"], stage["end_s"], color=STAGE_COLORS[stage["mode"]], alpha=.28, lw=0.)
        axis.plot(item["times"], item["forces"], color="#1b6685" if label == "R26" else "#8e3743",
            linewidth=.8, label="Max across thumb, index, middle, ring, pinky, palm")
        axis.axhline(12., color="#9c202e", linewidth=1., linestyle=(0, (5, 3)))
        axis.set_xlim(0., 100.)
        axis.set_ylim(0., 12. if panel == "C" else 60.)
        axis.set_ylabel("Max channel load (N)")
        axis.set_yticks(np.arange(0., 13., 3.) if panel == "C" else np.arange(0., 61., 12.))
        axis.grid(axis="y", color="#d7dce0", linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
        if panel != "C":
            axis.set_title(f"{panel}. {label} | {item['attempts']} attempt(s) | complete {item['duration_s']:.3f} s"
                f" | peak {item['peak_n']:.3f} N ({item['peak_channel']})", loc="left", pad=9)
        else:
            axis.set_title("C. R26 magnified vertically (0-12 N); complete episode retained", loc="left", pad=9)
        axis.scatter([item["peak_timestamp_s"]], [item["peak_n"]], s=15, color="#24282c", zorder=5)
        axis.text(99., 13.8 if panel != "C" else 11.4, "12 N gate", color="#9c202e", fontsize=9,
            ha="right", va="bottom" if panel != "C" else "top")
    axes[0].annotate("Episode ends at 33.338 s", xy=(data["R25"]["duration_s"], data["R25"]["forces"][-1]),
        xytext=(46., 25.), fontsize=9, color="#41464d",
        arrowprops={"arrowstyle": "-", "color": "#616971", "linewidth": .7})
    axes[2].text(35.675, 9.8, "Recovery 1", ha="center", fontsize=10)
    axes[2].text(84.496, 9.8, "Recovery 2", ha="center", fontsize=10)
    axes[2].annotate("Peak 5.285 N", xy=(data["R26"]["peak_timestamp_s"], data["R26"]["peak_n"]),
        xytext=(63., 7.7), fontsize=9, arrowprops={"arrowstyle": "-", "color": "#616971", "linewidth": .7})
    axes[2].set_xticks(np.arange(0., 101., 10.))
    axes[2].set_xlabel("Simulation time (s)")
    figure.text(.085, .103, CAPTION, ha="left", va="top", fontsize=9, linespacing=1.55, color="#31373c")
    figure.savefig(output / "whole_episode_contact_loads.png", dpi=180,
        metadata={"Description": CAPTION, "Software": "matplotlib"})
    figure.savefig(output / "whole_episode_contact_loads.pdf", metadata={
        "Title": "R25/R26 complete failed-episode contact loads", "Subject": CAPTION,
        "Creator": "scripts/plot_r26_supported_recovery.py"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/r26_recovery_visual_review_20261024")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "results/r26_recovery_visual_review_20261024"), "Output must stay in the review directory")
    require(digest(RECOVERY_RECEIPT) == RECOVERY_SHA256, "Expected the finalized R26 recovery receipt")
    recovery = json.loads(RECOVERY_RECEIPT.read_text())
    require(recovery["accepted_recovery_evidence"] is True and recovery["inputs_unchanged_during_audit"] is True
        and recovery["task_success"] is False and recovery["held_lift"] is False, "Unexpected recovery audit boundary")
    stages = recovery["stages"]
    require([stage["mode"] for stage in stages] == ["BRAKE", "LOWER", "BRAKE", "RELEASE"] * 2
        and all(stage["finished"] for stage in stages), "Expected two complete audited recovery sequences")
    data = {label: load_trace(label, stages) for label in RUNS}
    require(abs(data["R26"]["peak_n"] - recovery["whole_episode_peak_channel_n"]) <= 1e-9,
        "R26 whole-episode peak differs from recovery receipt")
    for stage in stages:
        times = data["R26"]["times"]
        native_ticks = np.rint(times / .002).astype(int)
        mask = (native_ticks > round(stage["start_s"] / .002)) & (native_ticks <= round(stage["end_s"] / .002))
        require(mask.any() and abs(float(data["R26"]["forces"][mask].max())
            - stage["reconstructed_peak_channel_n"]) <= 1e-9, "Recovery-stage peak differs from audit")
    input_paths = [Path(__file__).resolve(), Path(json_array.__code__.co_filename).resolve(), RECOVERY_RECEIPT]
    input_paths += [path for item in data.values() for path in item["input_paths"]]
    before = [file_record(path) for path in input_paths]
    output.mkdir(parents=True, exist_ok=False)
    for label, item in data.items():
        with (output / f"{label.lower()}_whole_episode_channels.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp_s", "native_tick", "state", *[channel + "_n" for channel in CHANNELS],
                "max_channel_n", "audited_recovery_stage"])
            writer.writerows(item["rows"])
    with (output / "r26_audited_recovery_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence", "mode", "start_s", "end_s", "reconstructed_peak_channel_n"])
        writer.writeheader()
        writer.writerows({"sequence": index // 4 + 1, **{key: stage[key] for key in writer.fieldnames if key != "sequence"}}
            for index, stage in enumerate(stages))
    draw(data, stages, output)
    require(before == [file_record(path) for path in input_paths], "Input changed while plotting")
    summaries = {label: {key: value for key, value in item.items()
        if key not in ("rows", "times", "forces", "input_paths")} for label, item in data.items()}
    manifest = {"format": "r26_supported_recovery_visual_review_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": CAPTION, "physics_rerun": False, "score_recomputed_or_rewritten": False,
        "inputs_unchanged": True, "all_raw_trace_rows_plotted_and_exported": True,
        "raw_trace_native_correspondence_previously_audited_not_recomputed_here": True,
        "channel_peaks_match_strict_receipts": True, "r26_episode_and_stage_peaks_match_recovery_receipt": True,
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "environment": {"PYTHONPATH": "src:scripts"},
        "reproduce": ["python", "scripts/plot_r26_supported_recovery.py", "--output",
            "results/r26_recovery_visual_review_20261024/new_review"],
        "inputs": before, "runs": summaries, "outputs": [file_record(path) for path in sorted(output.iterdir())]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "r25_peak_n": data["R25"]["peak_n"],
        "r26_peak_n": data["R26"]["peak_n"], "all_raw_rows": {label: item["trace_rows"] for label, item in data.items()},
        "manifest_sha256": digest(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
