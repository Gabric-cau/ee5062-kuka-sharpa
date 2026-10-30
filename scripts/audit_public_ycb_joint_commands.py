#!/usr/bin/env python3
"""Reconstruct active joint-command bounds from saved native observations."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np

from evaluate_v7_robustness import atomic_json, sha256
from plot_public_ycb_probe_audit import checked_stream
from public_ycb_evidence import load_evidence_model


def frozen_rates(directory, protocol):
    relative = "scripts/develop_public_ycb_force_band.py"
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        source = archive.read(relative)
    import hashlib
    if hashlib.sha256(source).hexdigest() != protocol["source_hashes_before"][relative]:
        raise ValueError("Frozen rate declaration changed")
    candidates = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        if all(isinstance(keywords.get(name), ast.Constant) for name in ("close_rate", "relief_rate")):
            candidates.append({name: float(keywords[name].value) for name in ("close_rate", "relief_rate")})
    if len(candidates) != 1:
        raise ValueError("Expected one frozen acquisition rate declaration")
    return candidates[0], relative


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Expected completed evidence")
    protocol = summary["protocol"]
    rates, declaration = frozen_rates(directory, protocol)
    bias_limit = float(protocol["joint_admittance"]["additional_bias_limit_rad"])
    cases = []
    tolerance = 1e-10
    for row in summary["rows"]:
        if row.get("runner_error"):
            cases.append({"case": row["path"], "unknown": True, "reason": row["runner_error"]})
            continue
        case = directory / row["path"]
        native = checked_stream(case, row["artifacts"]["native"])
        events = checked_stream(case, row["artifacts"]["joint_admittance"])
        model = load_evidence_model(case / "scene.xml")
        by_time = {round(item["timestamp"], 9): item for item in native}
        profile = protocol["nominal_profiles"][row["kind"]]
        captured = {}
        counts = Counter()
        peaks = Counter()
        examples = []
        violation_rows = []
        for event in events:
            if event.get("event") == "ACTIVATION_CAPTURE":
                captured = event["captured_command_minus_actual_rad"].copy()
                counts["activation_captures"] += 1
                continue
            if event.get("event") != "NOMINAL_REFERENCE_WRITE" or event.get("mode") != "JOINT_CONTROL":
                continue
            if event.get("safety_override_at_final_write"):
                counts["emergency_overridden_writes"] += 1
                continue
            timestamp, dt = float(event["timestamp_s"]), float(event["dt_s"])
            before = by_time.get(round(timestamp, 9))
            after = by_time.get(round(timestamp + float(model.opt.timestep), 9))
            if before is None or after is None or dt <= 0.:
                raise ValueError("Missing native state around actual command write")
            counts["active_writes"] += 1
            names = event["applied_command_rad"]
            for name, logged in names.items():
                if name not in captured:
                    raise ValueError("Active write without captured loaded bias")
                joint = model.joint(name).id
                actuator = np.flatnonzero((model.actuator_trntype == 0) &
                    (model.actuator_trnid[:, 0] == joint))
                if len(actuator) != 1:
                    raise ValueError("Expected one joint position actuator")
                actuator = int(actuator[0])
                actual = float(before["actual_hand_joints"][name]["position_rad"])
                previous = float(before["motor_commands"][actuator])
                applied = float(after["motor_commands"][actuator])
                if abs(applied - logged) > tolerance:
                    raise ValueError("Logged applied command differs from next native motor input")
                parts = name.split("_")
                finger, suffix = parts[1], "_".join(parts[2:])
                if finger == "thumb":
                    key = "thumb_mcp_flexion_ratio" if suffix == "MCP_FE" else "thumb_ip_flexion_ratio"
                    ratio = abs(float(profile[key]))
                else:
                    key, default = {"MCP_FE": ("nonthumb_mcp_ratio", 1.),
                        "PIP": ("nonthumb_pip_ratio", 1.4), "DIP": ("nonthumb_dip_ratio", .8)}[suffix]
                    ratio = abs(float(profile.get(key, default)))
                lower = max(float(model.jnt_range[joint, 0]),
                    previous - rates["relief_rate"] * ratio * dt,
                    actual + captured[name] - bias_limit)
                upper = min(float(model.jnt_range[joint, 1]),
                    previous + rates["close_rate"] * ratio * dt,
                    actual + captured[name] + bias_limit)
                bias = applied - actual - captured[name]
                violations = {
                    "added_bias": max(0., abs(bias) - bias_limit),
                    "closing_rate": max(0., applied - previous - rates["close_rate"] * ratio * dt),
                    "relief_rate": max(0., previous - applied - rates["relief_rate"] * ratio * dt),
                    "joint_range": max(0., model.jnt_range[joint, 0] - applied,
                        applied - model.jnt_range[joint, 1]),
                    "combined_interval": max(0., lower - applied, applied - upper),
                }
                counts["joint_writes"] += 1
                peaks["absolute_added_bias_rad"] = max(peaks["absolute_added_bias_rad"], abs(bias))
                flags = []
                for kind, excess in violations.items():
                    peaks[kind + "_excess_rad"] = max(peaks[kind + "_excess_rad"], float(excess))
                    if excess > tolerance:
                        counts[kind + "_violating_joint_writes"] += 1
                        flags.append(kind)
                if flags:
                    record = {"timestamp_s": timestamp, "joint": name, "violations": flags,
                        "actual_native_rad": actual, "previous_native_command_rad": previous,
                        "applied_next_native_command_rad": applied, "added_bias_rad": bias,
                        "reconstructed_interval_rad": [lower, upper], "excess_rad": violations}
                    violation_rows.append(record)
                    if len(examples) < 5:
                        examples.append(record)
        cases.append({"case": row["path"], "unknown": False, "counts": dict(counts),
            "maxima": dict(peaks), "first_violation_examples": examples,
            "violating_timestamps": sorted({item["timestamp_s"] for item in violation_rows}),
            "active_command_bounds_satisfied": bool(counts["joint_writes"] and not violation_rows),
            "task_success": row["success"], "held_lift": row["held_lift"],
            "inputs": {key: row["artifacts"][key] for key in ("native", "joint_admittance")},
            "scene_sha256": sha256(case / "scene.xml")})
    return {"format": "native_joint_command_bounds_r1", "rates_per_s": rates,
        "rate_declaration_in_frozen_source": declaration, "additional_bias_limit_rad": bias_limit,
        "numerical_tolerance_rad": tolerance, "cases": cases,
        "summary_sha256": sha256(directory / "summary.json"), "script_sha256": sha256(Path(__file__)),
        "scope": "Read-only active nominal commands only. Actual joint positions and previous/next actuator commands are taken from native records, not the controller's step diagnostics. Captured bias is a declared activation offset. Safety overrides and inactive modes are excluded explicitly. This does not evaluate force, friction or task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = audit(args.directory.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, receipt)
    for case in receipt["cases"]:
        print(json.dumps({key: case.get(key) for key in
            ("case", "unknown", "counts", "maxima", "active_command_bounds_satisfied")}), flush=True)


if __name__ == "__main__":
    main()
