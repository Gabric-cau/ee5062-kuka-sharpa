#!/usr/bin/env python3
"""Independent offline R17/R18 plots; never run or modify a controller."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import compare_public_ycb_bias_budget as metrics
from compare_public_ycb_close_deadline import (
    EXPECTED_R17_SUMMARY_SHA256, digest, json_array, safe_path,
)


PROJECT = Path(__file__).resolve().parents[1]
FORMATS = {"baseline": "public_ycb_bias_budget_development_r17",
    "candidate": "public_ycb_close_deadline_development_r18"}
PROBE_KEYS = ("timestamp_s", "phase", "probe_active", "actual_probe_displacement_m",
    "actual_probe_endpoint_reached", "table_load_n", "catalogue_weight_n", "table_unloaded",
    "probe_eligible_seconds", "probe_confirmed_seconds", "sampled_fixed_normal_1g_feasible",
    "sampled_capacity_1g_feasible", "geometry_1_5g_feasible", "preprobe_reasons",
    "admission", "confirmation", "stop_reason", "pressure_stable", "actual_wrist_speed_m_s")


def captured_json(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()}


def source_bindings():
    paths = (Path(__file__).resolve(), PROJECT / "scripts/compare_public_ycb_bias_budget.py",
        PROJECT / "scripts/compare_public_ycb_close_deadline.py",
        PROJECT / "scripts/audit_public_ycb_bias_recovery.py")
    return {str(path.relative_to(PROJECT)): digest(path) for path in paths}


def bind_artifact(case, artifact):
    path = safe_path(case, artifact["path"])
    bound = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if bound["bytes"] != artifact["bytes"] or bound["sha256"] != artifact["sha256"]:
        raise ValueError("Recorded compressed artifact hash/size differs: " + str(path))
    return path, bound


def reduced_rows(path, key):
    """Retain curve inputs rather than repeated LP/association/contact payloads."""
    result, previous = [], -1
    for row in json_array(path):
        index = metrics.tick(row["timestamp_s"])
        if index < previous:
            raise ValueError("Plot input is not ordered")
        previous = index
        if key == "load_transfer_probe":
            result.append({name: row[name] for name in PROBE_KEYS if name in row})
        elif key == "preload":
            result.append({"timestamp_s": row["timestamp_s"], "phase": row["phase"],
                "current_target_lp": {"feasible": row["current_target_lp"].get("feasible", False)},
                "current_groups": row["current_groups"], "ready": row["ready"],
                "continuous_seconds": row["continuous_seconds"], "reasons": row["reasons"]})
        else:
            result.append({"timestamp_s": row["timestamp_s"], "phase": row["phase"], "mode": row["mode"],
                "additional_bias_limit_rad": row["additional_bias_limit_rad"],
                "fresh_target": row["fresh_target"], "failure_latched": row["failure_latched"],
                "per_finger": [{"mode": finger["mode"], "joints": {name: {key: joint[key] for key in
                    ("applied_command_rad", "previous_reference_rad", "before_actual_rad", "fixed_capture_bias_rad", "feasible")}
                    for name, joint in finger["joints"].items()}} for finger in row["per_finger"]]})
    return result


def stream_torque(path, metadata, guards):
    if (metadata["force_evaluation_state"] != "last_completed_dynamics_evaluation_cache"
            or metadata["observation_time_semantics"] != "post_integration_native_sample"
            or metadata["postintegration_qpos_recomputed"] is not False or metadata["evaluation_only"] is not True):
        raise ValueError("Unsupported cached native actuation timing")
    parameters = metadata["actuator_parameters"]
    positions = {item["joint_name"]: index for index, item in enumerate(parameters)}
    if len(positions) != len(parameters):
        raise ValueError("Duplicate native actuator joint mapping")
    series = [{name: {"time": [], "torque": [], "clamped": []} for name in metrics.THUMB} for _ in guards]
    sample_count = 0
    for sample_count, row in enumerate(json_array(path), 1):
        index = metrics.tick(row["timestamp"])
        if index != sample_count:
            raise ValueError("Native plot input is not contiguous from the first timestep")
        write_index = index - 1
        selected = [ordinal for ordinal, guarded in enumerate(guards) if write_index in guarded]
        if not selected:
            continue
        cache = row["native_actuation"]
        if cache["actuator_ids"] != metadata["actuator_ids"] or cache["joint_dof_ids"] != metadata["joint_dof_ids"]:
            raise ValueError("Native actuation mapping differs from metadata")
        for name in metrics.THUMB:
            offset = positions[name]
            item = parameters[offset]
            torque = float(cache["qfrc_actuator"][offset])
            raw = float(cache["actuator_force"][offset]) * item["gear"][0]
            if not np.isfinite([torque, raw]).all():
                raise ValueError("Nonfinite cached torque")
            for ordinal in selected:
                values = series[ordinal][name]
                values["time"].append(float(row["timestamp"]))
                values["torque"].append(torque)
                values["clamped"].append(abs(torque - raw) > 1e-9)
    summaries = []
    for curves in series:
        summary = {}
        for name, values in curves.items():
            parameter = parameters[positions[name]]
            limits = parameter["jointactfrcrange"] if parameter["jointactfrclimited"] else None
            summary[name] = {"scope": "Guarded CLOSE writes only; native cached actuator torque, not net/contact torque",
                "torque_nm": metrics.statistics(values["torque"]),
                "absolute_torque_nm": metrics.statistics(abs(value) for value in values["torque"]),
                "joint_actuator_force_limits_nm": limits,
                "joint_limit_clamping": metrics.flag_summary({metrics.tick(time) - 1: flag
                    for time, flag in zip(values["time"], values["clamped"], strict=True)})}
            values["limits"] = limits
            del values["clamped"]
        summaries.append(summary)
    return summaries, series, sample_count


def progress_plot(output, name, label, attempt, preload, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = metrics.unique_close(preload, attempt["close_start_tick"], attempt["close_end_tick"])
    selected = [selected[index] for index in sorted(selected)]
    original = metrics.unique_close(probe, attempt["close_start_tick"], attempt["close_end_tick"])
    original = [original[index] for index in sorted(original)]
    origin = attempt["close_start_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    groups = sorted({key for row in selected for key in row["current_groups"]})
    colors = {"thumb": "#bd3535", "middle": "#247c49", "ring": "#315ba5"}
    for key in groups:
        finger = key.split(":", 1)[0]
        values = [abs(row["current_groups"][key]["target_n"] - row["current_groups"][key]["measured_n"])
            if key in row["current_groups"] else np.nan for row in selected]
        axes[0].plot([row["timestamp_s"] - origin for row in selected], values,
            color=colors.get(finger), linewidth=.8, label=key)
    axes[0].axhline(.35, linestyle="--", color="black", label="Required group error <=0.35 N")
    axes[0].set_ylabel("Absolute group error (N)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("Each associated group is checked separately; gaps are absent/invalid groups")
    axes[1].plot([row["timestamp_s"] - origin for row in selected],
        [row["continuous_seconds"] for row in selected], color="#bd3535", label="Continuous preload")
    axes[1].axhline(.25, color="black", linestyle="--", label="Required preload 0.25 s")
    axes[1].set_ylim(-.015, max(.3, max(row["continuous_seconds"] for row in selected) * 1.1))
    axes[1].set_ylabel("Preload continuous (s)")
    axes[1].legend(loc="upper left", fontsize=8)
    eligible = axes[1].twinx()
    eligible.plot([row["timestamp_s"] - origin for row in original],
        [row["probe_eligible_seconds"] for row in original], color="#986b20", label="Original probe eligibility")
    eligible.set_ylabel("Original eligibility continuous (s)")
    eligible.legend(loc="upper right", fontsize=8)
    times = [row["timestamp_s"] - origin for row in original]
    for field, label_text, color in (("sampled_fixed_normal_1g_feasible", "Measured fixed-normal 1g", "#247c49"),
            ("probe_active", "Probe active", "#315ba5"), ("pressure_stable", "Pressure stable", "#986b20")):
        axes[2].step(times, [int(row[field]) for row in original], where="post", label=label_text, color=color, linewidth=.9)
    axes[2].set_ylim(-.1, 1.2)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylabel("Recorded flags")
    axes[2].set_xlabel("Seconds after CLOSE begins")
    axes[2].legend(fontsize=8, ncol=3)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axvline(16., color="#777777", linestyle=":", linewidth=.8)
    fig.suptitle(f"{label} | attempt {attempt['attempt']} | original deadline at 16 s", fontsize=11)
    stem = f"{name}_attempt{attempt['attempt']}_gate_progress"
    files = []
    for extension in ("png", "pdf"):
        filename = stem + "." + extension
        fig.savefig(output / filename, dpi=150)
        files.append(filename)
    plt.close(fig)
    return files


def plot_run(directory, role, output, baseline_sha):
    directory = directory.resolve(strict=True)
    summary, summary_bound = captured_json(safe_path(directory, "summary.json"))
    protocol, protocol_bound = captured_json(safe_path(directory, "protocol.json"))
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or protocol["format"] != FORMATS[role] or len(summary["rows"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("Expected one completed, unchanged case with its true role-specific protocol")
    if role == "baseline" and summary_bound["sha256"] != EXPECTED_R17_SUMMARY_SHA256:
        raise ValueError("Plot baseline differs from the exact frozen R17 summary")
    if role == "candidate" and protocol["r18_close_deadline_experiment"]["r17_comparison_summary_sha256"] != baseline_sha:
        raise ValueError("R18 plot inputs are bound to a different R17 baseline")
    expected_close = 16. if role == "baseline" else 32.
    budget = float(protocol["joint_admittance"]["additional_bias_limit_rad"])
    if budget != .04 or protocol["phase_seconds"]["CLOSE"] != expected_close:
        raise ValueError("Expected the fixed .04 rad, 16/32 second comparison")
    row = summary["rows"][0]
    case = directory / row["path"]
    metadata, metadata_bound = captured_json(safe_path(directory, Path(row["path"]) / "metadata.json"))
    scene = safe_path(directory, Path(row["path"]) / "scene.xml")
    scene_bytes = scene.read_bytes()
    scene_bound = {"path": str(scene), "bytes": len(scene_bytes), "sha256": hashlib.sha256(scene_bytes).hexdigest()}
    if (metadata != row or row.get("runner_error") or row["scene_unchanged_since_load"] is not True
            or scene_bound["sha256"] != row["scene_sha256_before_load"]
            or float(ET.fromstring(scene_bytes).find("option").attrib["timestep"]) != metrics.STEP):
        raise ValueError("Case metadata, loaded scene or native timestep differs")
    inputs = {"summary": summary_bound, "protocol": protocol_bound, "metadata": metadata_bound, "scene": scene_bound}
    paths = {}
    for key in ("load_transfer_probe", "preload", "bias_recovery", "native"):
        paths[key], inputs[key] = bind_artifact(case, row["artifacts"][key])
    probe = reduced_rows(paths["load_transfer_probe"], "load_transfer_probe")
    windows = metrics.close_windows(probe, row["attempts"])
    if not windows:
        raise ValueError("No recorded CLOSE window")
    preload = reduced_rows(paths["preload"], "preload")
    bias = reduced_rows(paths["bias_recovery"], "bias_recovery")
    attempts, curves, guards = [], [], []
    for window in windows:
        probe_metric, probe_series = metrics.probe_metrics(probe, window)
        preload_metric, pressure_series = metrics.preload_metrics(preload, window)
        bias_metric, bias_series, guarded = metrics.bias_metrics(bias, window, budget)
        attempts.append({**window, "probe": probe_metric, "preload": preload_metric, "bias": bias_metric,
            "whole_attempt_probe": metrics.attempt_probe_metrics(probe, window)})
        curves.append({"probe": probe_series, "pressure": pressure_series, "bias": bias_series})
        guards.append(guarded)
    del bias
    gc.collect()
    torque_summary, torque_curves, native_count = stream_torque(paths["native"], row["native_audit"]["native_actuation_metadata"], guards)
    if native_count != row["native_audit"]["native_samples"] or native_count != row["native_audit"]["native_actuation_samples"]:
        raise ValueError("Native sample counts differ from metadata")
    figures = []
    label = f"{role.upper()} {protocol['format']} | CLOSE {expected_close:g}s"
    for index, (attempt, series) in enumerate(zip(attempts, curves, strict=True)):
        attempt["direct_native_thumb_torque"] = torque_summary[index]
        series["torque"] = torque_curves[index]
        figures.extend(metrics.plot_attempt(output, role, label, attempt, budget, series))
        figures.extend(progress_plot(output, role, label, attempt, preload, probe))
    for bound in inputs.values():
        path = Path(bound["path"])
        if path.stat().st_size != bound["bytes"] or digest(path) != bound["sha256"]:
            raise ValueError("Plot input changed during analysis")
    return {"role": role, "directory": str(directory), "protocol_format": protocol["format"],
        "declared_close_seconds": expected_close, "budget_rad": budget,
        "case": {"kind": row["kind"], "seed": row["seed"], "path": row["path"],
            "reported_success": row["success"], "reported_held_lift": row["held_lift"], "reported_failure": row.get("failure")},
        "inputs": inputs, "native_samples_streamed": native_count, "attempts": attempts,
        "figures": [{"path": name, "bytes": (output / name).stat().st_size, "sha256": digest(output / name)} for name in figures]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_bindings()
    baseline = plot_run(args.baseline, "baseline", args.output, EXPECTED_R17_SUMMARY_SHA256)
    candidate = plot_run(args.candidate, "candidate", args.output, baseline["inputs"]["summary"]["sha256"])
    if (any(baseline["case"][key] != candidate["case"][key] for key in ("kind", "seed"))
            or baseline["inputs"]["scene"]["sha256"] != candidate["inputs"]["scene"]["sha256"]):
        raise ValueError("Plot pair differs in case identity or loaded scene")
    if sources != source_bindings():
        raise ValueError("Offline analysis source changed during plotting")
    result = {"format": "independent_r17_r18_close_deadline_plots_v1", "independent_postrun_analysis": True,
        "part_of_runtime_frozen_manifest": False, "analysis_source_sha256": sources,
        "runs": [baseline, candidate], "scope": "Descriptive plots of completed recorded R17/R18 runs using their actual protocol formats. "
            "Pure metrics and the four-panel plot are reused from the frozen R16/R17 comparison helper; its protocol loader is not used or relabelled. "
            "Native contacts are streamed and discarded; only guarded CLOSE thumb actuation values are retained. Cached torque at t+dt belongs to write t dynamics. "
            "Plots do not establish physical-prefix identity, command consistency, measured equilibrium, success, or independent repetitions; separate auditors remain required. "
            "Table load and simulator-derived target values are evaluation-only, never controller inputs. Supported shear may reverse during lift; no geometry conclusion follows from its sign. "
            "Pressure curves sum associated groups per finger; the companion gate-progress figure shows every group error separately and the unchanged .35 N/.25 s thresholds. "
            "Original probe eligibility is distinct from continuous preload. One passing native sample is not sustained admission."}
    destination = args.output / "comparison.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"comparison": str(destination), "sha256": digest(destination),
        "figures": sum(len(run["figures"]) for run in result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
