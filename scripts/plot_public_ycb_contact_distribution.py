#!/usr/bin/env python3
"""Plot recorded contact distribution and middle-joint tracking, without dynamics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_v7_robustness import atomic_json, sha256
from plot_public_ycb_probe_audit import checked_stream
from public_ycb_evidence import load_evidence_model
from verify_public_ycb_strict import hand_geometry


FINGERS = ("thumb", "middle", "ring")
COLORS = ("#006e90", "#bb4430", "#648442")


def recorded_probe(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["complete"] or len(summary["rows"]) != 1:
        raise ValueError("Expected one completed development case")
    row = summary["rows"][0]
    case = directory / row["path"]
    native = checked_stream(case, row["artifacts"]["native"])
    events = checked_stream(case, row["artifacts"]["load_transfer_probe"])
    starts = sorted({event["probe_start_s"] for event in events
        if event.get("probe_start_s") is not None and event.get("probe_active")})
    if not starts:
        raise ValueError("No admitted probe; no trajectory to plot")
    start = starts[0]
    probe = [event for event in events if event.get("probe_start_s") == start
        and (event.get("probe_active") or event.get("stop_reason"))]
    stop = max(event["timestamp_s"] for event in probe)
    selected = [sample for sample in native if start - 1e-8 <= sample["timestamp"] <= stop + 1e-8]
    model = load_evidence_model(case / "scene.xml")
    hand = hand_geometry(model)
    target = model.geom("target_geom").id
    pair_mu = {frozenset((int(model.pair_geom1[index]), int(model.pair_geom2[index]))):
        float(model.pair_friction[index, 0]) for index in range(model.npair)}
    pressures, middle_parts, utilization = [], [], []
    for sample in selected:
        channels = dict.fromkeys(FINGERS, 0.)
        parts = {"MP": 0., "DP": 0., "other": 0.}
        ratios = []
        for contact in sample["native_contacts"]:
            sides = set(contact["geom_ids"]) & hand.keys()
            if len(sides) != 1:
                continue
            gid = next(iter(sides))
            finger = hand[gid]
            force = np.asarray(contact["force_torque_contact_frame"], dtype=float)
            normal = abs(float(force[0]))
            if finger in channels:
                channels[finger] += normal
            if finger == "middle":
                body = model.body(int(model.geom_bodyid[gid])).name
                part = next((part for part in ("MP", "DP") if body.endswith("_" + part)), "other")
                parts[part] += normal
            if target in contact["geom_ids"] and normal >= .05:
                key = frozenset(contact["geom_ids"])
                if key not in pair_mu:
                    raise ValueError("Expected explicit physical hand-target friction pair")
                coefficient = min(.8, pair_mu[key]) * .85
                ratios.append(float(np.linalg.norm(force[1:3])) / (normal * coefficient))
        pressures.append([channels[finger] for finger in FINGERS])
        middle_parts.append([parts[part] for part in ("MP", "DP", "other")])
        utilization.append(max(ratios) if ratios else np.nan)
    times = np.asarray([sample["timestamp"] - start for sample in selected])
    return row, case, model, selected, times, np.asarray(pressures), np.asarray(middle_parts), np.asarray(utilization)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(4, len(args.directories), figsize=(7 * len(args.directories), 12), squeeze=False)
    metadata = []
    for column, directory in enumerate(args.directories):
        row, case, model, samples, times, pressures, parts, utilization = recorded_probe(directory)
        pressure_axis, parts_axis, friction_axis, joint_axis = axes[:, column]
        for index, (finger, color) in enumerate(zip(FINGERS, COLORS, strict=True)):
            pressure_axis.plot(times, pressures[:, index], label=finger, color=color)
        pressure_axis.axhline(6., linestyle="--", color="#555555", label="Relief trigger")
        pressure_axis.set_title(directory.name.replace("public_ycb_", ""), fontsize=10)
        pressure_axis.set_ylabel("External channel normal (N)")
        pressure_axis.legend(fontsize=8, ncol=2)
        for index, (part, color) in enumerate(zip(("MP", "DP", "other"), COLORS, strict=True)):
            parts_axis.plot(times, parts[:, index], label=part, color=color)
        parts_axis.set_ylabel("Middle link normal (N)")
        parts_axis.legend(fontsize=8)
        friction_axis.plot(times, utilization, color="#8152a0", label="Max across patches >=0.05 N")
        friction_axis.axhline(1., linestyle="--", color="#555555", label="Conservative LP coefficient")
        friction_axis.set_ylabel("|Tangential| / (normal x LP mu)")
        friction_axis.legend(fontsize=8)
        for suffix, color in zip(("MCP_FE", "PIP", "DIP"), COLORS, strict=True):
            name = "left_middle_" + suffix
            joint = model.joint(name).id
            aids = np.flatnonzero((model.actuator_trntype == 0) & (model.actuator_trnid[:, 0] == joint))
            if len(aids) != 1:
                raise ValueError("Expected one joint position actuator")
            actual = np.asarray([sample["actual_hand_joints"][name]["position_rad"] for sample in samples])
            command = np.asarray([sample["motor_commands"][int(aids[0])] for sample in samples])
            joint_axis.plot(times, np.degrees(actual - actual[0]), color=color, label=suffix + " actual")
            joint_axis.plot(times, np.degrees(command - command[0]), color=color, linestyle="--", label=suffix + " command")
        joint_axis.set_ylabel("Angle change from admission (deg)")
        joint_axis.legend(fontsize=7, ncol=2)
        for axis in axes[:, column]:
            axis.set_xlabel("Time from first actual probe admission (s)")
            axis.grid(axis="y", alpha=.2)
        metadata.append({"case": row["path"], "task_success": row["success"], "held_lift": row["held_lift"],
            "native_samples": len(samples), "probe_duration_s": float(times[-1]),
            "peak_plotted_channel_n": float(pressures.max()),
            "friction_ratio_max": float(np.nanmax(utilization)),
            "inputs": {key: row["artifacts"][key] for key in ("native", "load_transfer_probe")},
            "scene_sha256": sha256(case / "scene.xml")})
    figure.suptitle("Recorded contact distribution | LP friction margin is not a task-success test", fontsize=12)
    figure.tight_layout(rect=(0., 0., 1., .97))
    files = []
    for suffix in ("png", "pdf"):
        output = args.output / ("contact_distribution." + suffix)
        figure.savefig(output, dpi=160)
        files.append({"path": output.name, "sha256": sha256(output)})
    plt.close(figure)
    atomic_json(args.output / "plot_metadata.json", {"cases": metadata, "files": files,
        "script_sha256": sha256(Path(__file__)), "scope": "Read-only native evidence; first probe only; weak patches are omitted only from the plotted friction ratio; actual joint and contact caches may straddle one native integration step"})
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
