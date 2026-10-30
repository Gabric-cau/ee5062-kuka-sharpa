#!/usr/bin/env python3
"""Independent offline R18/R19 budget plots using their actual recorded formats."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import plot_public_ycb_close_deadline as common


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_R18_SUMMARY_SHA256 = "d3d5dec564810d8c9b6a9cc4cee8354062eb9f99c54960ab49b41ad87a740ff0"
EXPECTED_R18_RECOVERY_SHA256 = "a40d4da5f8f0e8e1cbeb471940ce502241da74c69571e9eb442a7db5f17f77fa"
FORMATS = {"baseline": "public_ycb_close_deadline_development_r18",
    "candidate": "public_ycb_extended_budget_development_r19"}
BUDGETS = {"baseline": .04, "candidate": .10}
metrics = common.metrics
digest, safe_path, json_array = common.digest, common.safe_path, common.json_array


def source_bindings():
    paths = (Path(__file__).resolve(), Path(common.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_bias_budget.py",
        PROJECT / "scripts/compare_public_ycb_close_deadline.py",
        PROJECT / "scripts/audit_public_ycb_bias_recovery.py")
    return {str(path.relative_to(PROJECT)): digest(path) for path in paths}


def reduced_preload(path):
    result, previous = [], -1
    for row in json_array(path):
        index = metrics.tick(row["timestamp_s"])
        if index < previous:
            raise ValueError("Preload input is not ordered")
        previous = index
        observation = row["original_probe_snapshot"]
        result.append({"timestamp_s": row["timestamp_s"], "phase": row["phase"],
            "current_target_lp": {key: row["current_target_lp"].get(key)
                for key in ("feasible", "reason")},
            "current_groups": row["current_groups"], "ready": row["ready"],
            "continuous_seconds": row["continuous_seconds"], "reasons": row["reasons"],
            "window_reset_reason": row.get("window_reset_reason"),
            "r16_recovery_inhibited": row.get("r16_recovery_inhibited", False),
            "pressure_window_s": observation["pressure_window_s"],
            "pressure_spans_n": observation["pressure_spans_n"],
            "pressure_stable": observation["pressure_stable"]})
    return result


def preload_context(rows, attempt):
    selected = metrics.unique_close(rows, attempt["close_start_tick"], attempt["close_end_tick"])
    return {"unique_samples": len(selected),
        "target_valid": metrics.flag_summary({index: bool(row["current_target_lp"]["feasible"]
            and row["current_groups"]) for index, row in selected.items()}),
        "target_reasons": dict(Counter(row["current_target_lp"]["reason"] for row in selected.values())),
        "recovery_inhibited": metrics.flag_summary({index: row["r16_recovery_inhibited"]
            for index, row in selected.items()}),
        "pressure_window_seconds": metrics.statistics(row["pressure_window_s"] for row in selected.values()),
        "full_pressure_window": metrics.flag_summary({index: row["pressure_window_s"] >= .25 - 1e-9
            for index, row in selected.items()}),
        "pressure_span_above_03n": metrics.flag_summary({index: max(row["pressure_spans_n"].values()) > .3
            for index, row in selected.items()}),
        "scope": "Unique post-integration CLOSE samples; zero span after clearing a short pressure window is not stability evidence. "
            "Error statistics from the shared helper retain only available targets; invalid samples are not zero error."}


def progress_plot(output, name, label, attempt, preload, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = metrics.unique_close(preload, attempt["close_start_tick"], attempt["close_end_tick"])
    selected = [selected[index] for index in sorted(selected)]
    original = metrics.unique_close(probe, attempt["close_start_tick"], attempt["close_end_tick"])
    original = [original[index] for index in sorted(original)]
    origin = attempt["close_start_s"]
    times = [row["timestamp_s"] - origin for row in selected]
    original_times = [row["timestamp_s"] - origin for row in original]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    colors = {"thumb": "#bd3535", "middle": "#247c49", "ring": "#315ba5"}
    for key in sorted({key for row in selected for key in row["current_groups"]}):
        values = [abs(row["current_groups"][key]["target_n"] - row["current_groups"][key]["measured_n"])
            if key in row["current_groups"] else np.nan for row in selected]
        axes[0].plot(times, values, color=colors.get(key.split(":", 1)[0]), linewidth=.8, label=key)
    axes[0].axhline(.35, linestyle="--", color="black", label="Required group error <=0.35 N")
    axes[0].set_ylabel("Absolute group error (N)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("Associated groups are checked separately; absent targets remain gaps")
    axes[1].plot(times, [row["continuous_seconds"] for row in selected], color="#bd3535", label="Continuous preload")
    axes[1].axhline(.25, color="black", linestyle="--", label="Required preload 0.25 s")
    axes[1].set_ylim(-.015, max(.3, max(row["continuous_seconds"] for row in selected) * 1.1))
    axes[1].set_ylabel("Preload continuous (s)")
    axes[1].legend(loc="upper left", fontsize=8)
    eligible = axes[1].twinx()
    eligible.plot(original_times, [row["probe_eligible_seconds"] for row in original],
        color="#986b20", label="Original probe eligibility")
    eligible.set_ylabel("Original eligibility (s)")
    eligible.legend(loc="upper right", fontsize=8)
    axes[2].plot(times, [row["pressure_window_s"] for row in selected], color="#315ba5", label="Pressure window age")
    axes[2].axhline(.25, linestyle="--", color="#315ba5", label="Required window 0.25 s")
    axes[2].set_ylabel("Pressure window (s)")
    axes[2].legend(loc="upper left", fontsize=8)
    spans = axes[2].twinx()
    spans.plot(times, [max(row["pressure_spans_n"].values()) for row in selected], color="#bd3535", label="Maximum channel span")
    spans.axhline(.3, linestyle="--", color="#bd3535", label="Maximum span 0.3 N")
    spans.set_ylabel("Pressure span (N)")
    spans.legend(loc="upper right", fontsize=8)
    for field, text, color in (("sampled_fixed_normal_1g_feasible", "Measured fixed-normal 1g", "#247c49"),
            ("probe_active", "Probe active", "#315ba5"), ("pressure_stable", "Pressure stable", "#986b20")):
        axes[3].step(original_times, [int(row[field]) for row in original], where="post", label=text, color=color, linewidth=.9)
    axes[3].step(times, [int(row["r16_recovery_inhibited"]) for row in selected],
        where="post", label="Recovery inhibits preload", color="#bd3535", linewidth=.8)
    axes[3].set_ylim(-.1, 1.2)
    axes[3].set_yticks([0, 1])
    axes[3].set_ylabel("Recorded flags")
    axes[3].set_xlabel("Seconds after CLOSE begins")
    axes[3].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axvline(16., color="#777777", linestyle=":", linewidth=.8)
        axis.axvline(32., color="black", linestyle=":", linewidth=.8)
    fig.suptitle(f"{label} | attempt {attempt['attempt']}\n16 s arm-clock reference; 32 s CLOSE limit", fontsize=11)
    files = []
    for extension in ("png", "pdf"):
        filename = f"{name}_attempt{attempt['attempt']}_gate_progress.{extension}"
        fig.savefig(output / filename, dpi=150)
        files.append(filename)
    plt.close(fig)
    return files


def plot_run(directory, role, output):
    directory = directory.resolve(strict=True)
    summary, summary_bound = common.captured_json(safe_path(directory, "summary.json"))
    protocol, protocol_bound = common.captured_json(safe_path(directory, "protocol.json"))
    budget = BUDGETS[role]
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or protocol["format"] != FORMATS[role] or len(summary["rows"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != budget
            or protocol["phase_seconds"]["CLOSE"] != 32.
            or protocol["constrained_preload"]["close_seconds"] != 32.):
        raise ValueError("Expected one finalized case with its actual role-specific budget and 32s CLOSE")
    inputs = {"summary": summary_bound, "protocol": protocol_bound}
    if role == "baseline":
        recovery, inputs["recovery"] = common.captured_json(safe_path(directory, "recovery_receipt.json"))
        if (summary_bound["sha256"] != EXPECTED_R18_SUMMARY_SHA256
                or inputs["recovery"]["sha256"] != EXPECTED_R18_RECOVERY_SHA256
                or summary["artifact_finalization_recovered"] is not True
                or summary["original_runner_normal_exit"] is not False
                or protocol["finalization_recovery"] != recovery
                or recovery["simulation_rerun"] is not False
                or recovery["runner_exit_code_observed"] is not None):
            raise ValueError("Plot baseline must preserve the exact recovered R18 provenance")
        declaration = protocol["r18_close_deadline_experiment"]
    else:
        declaration = protocol["r19_extended_budget_experiment"]
        if (declaration["r18_comparison_summary_sha256"] != EXPECTED_R18_SUMMARY_SHA256
                or declaration["additional_bias_limit_rad"] != budget
                or declaration["changed_parameters"] != {"joint_admittance.additional_bias_limit_rad":
                    {"before": .04, "after": .10}}):
            raise ValueError("R19 plot input is bound to a different baseline or experiment")
    if declaration["arm_interpolation_seconds"] != 16. or declaration["native_timestep_s"] != metrics.STEP:
        raise ValueError("The original arm clock or native timestep differs")
    row = summary["rows"][0]
    metadata, inputs["metadata"] = common.captured_json(safe_path(directory, Path(row["path"]) / "metadata.json"))
    scene = safe_path(directory, Path(row["path"]) / "scene.xml")
    controller = safe_path(directory, Path(row["path"]) / "derived_controller.py")
    for key, path in (("scene", scene), ("controller", controller)):
        inputs[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if (metadata != row or row.get("runner_error") or row["additional_bias_limit_rad"] != budget
            or row["close_seconds"] != 32. or row["arm_interpolation_seconds"] != 16.
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]
            or row["scene_unchanged_since_load"] is not True
            or inputs["scene"]["sha256"] != row["scene_sha256_before_load"]
            or inputs["scene"]["sha256"] != row["scene_sha256_at_finalize"]
            or inputs["controller"]["sha256"] != row["derived_controller_sha256"]
            or float(ET.fromstring(scene.read_bytes()).find("option").attrib["timestep"]) != metrics.STEP):
        raise ValueError("Case metadata, clocks, loaded scene, controller or timestep differs")
    if role == "candidate" and (row.get("missing_runtime_controller_binding", False) is not False
            or not row["effective_hand_budgets_rad"]
            or row["effective_hand_budgets_rad"] != [budget] * len(row["attempts"])
            or row["effective_hand_instances"] != len(row["attempts"])):
        raise ValueError("R19 actual hand/retry budgets are missing or inconsistent")
    case, paths = scene.parent, {}
    for key in ("load_transfer_probe", "preload", "bias_recovery", "native"):
        paths[key], inputs[key] = common.bind_artifact(case, row["artifacts"][key])
    probe = common.reduced_rows(paths["load_transfer_probe"], "load_transfer_probe")
    windows = metrics.close_windows(probe, row["attempts"])
    if not windows:
        raise ValueError("No recorded CLOSE window")
    preload, bias = reduced_preload(paths["preload"]), common.reduced_rows(paths["bias_recovery"], "bias_recovery")
    attempts, curves, guards = [], [], []
    for window in windows:
        probe_metric, probe_series = metrics.probe_metrics(probe, window)
        preload_metric, pressure_series = metrics.preload_metrics(preload, window)
        bias_metric, bias_series, guarded = metrics.bias_metrics(bias, window, budget)
        attempts.append({**window, "probe": probe_metric, "preload": preload_metric, "bias": bias_metric,
            "preload_context": preload_context(preload, window),
            "whole_attempt_probe": metrics.attempt_probe_metrics(probe, window)})
        curves.append({"probe": probe_series, "pressure": pressure_series, "bias": bias_series})
        guards.append(guarded)
    del bias
    gc.collect()
    torque_summary, torque_curves, native_count = common.stream_torque(paths["native"],
        row["native_audit"]["native_actuation_metadata"], guards)
    if native_count != row["native_audit"]["native_samples"] or native_count != row["native_audit"]["native_actuation_samples"]:
        raise ValueError("Native sample counts differ from metadata")
    figures = []
    label = f"{role.upper()} {'R18 recovered' if role == 'baseline' else 'R19'} | budget {budget:g} rad | CLOSE 32s"
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
        "declared_close_seconds": 32., "arm_interpolation_seconds": 16., "budget_rad": budget,
        "source_hashes_after_timing": protocol.get("source_hashes_after_timing", "Original runner finalization"),
        "artifact_finalization_recovered": summary.get("artifact_finalization_recovered", False),
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
    baseline = plot_run(args.baseline, "baseline", args.output)
    candidate = plot_run(args.candidate, "candidate", args.output)
    if (any(baseline["case"][key] != candidate["case"][key] for key in ("kind", "seed"))
            or any(baseline["inputs"][key]["sha256"] != candidate["inputs"][key]["sha256"]
                for key in ("scene", "controller"))):
        raise ValueError("Plot pair differs in case identity, loaded scene or executed controller source")
    if sources != source_bindings():
        raise ValueError("Offline analysis source changed during plotting")
    result = {"format": "independent_r18_r19_extended_budget_plots_v1", "independent_postrun_analysis": True,
        "part_of_runtime_frozen_manifest": False, "analysis_source_sha256": sources,
        "runs": [baseline, candidate], "physical_prefix_independently_compared": False,
        "scope": "Descriptive actual-format plots of the recovered .04rad R18 and .10rad R19 cases at unchanged 32s CLOSE/16s arm clocks. "
            "The existing pure metrics/four-panel plot, reduced stream reader and cached-torque sampler are reused; old format-specific loaders are not used. "
            "Native contacts are streamed and discarded, retaining guarded thumb actuation only; cache observed at t+dt belongs to write t dynamics. "
            "The separate progress figure distinguishes pressure-window age and span from controller-cleared windows, original eligibility and continuous preload. "
            "Invalid targets are gaps, not zero error; group errors are conditional on available targets. Attempt 2 is a history-dependent retry, not another matched initial condition. "
            "Figures do not verify command safety, normal process exit, physical-prefix identity, measured equilibrium, task success or independent repetitions. "
            "Separate native/strict/command audits and supervisor receipts remain required. Table loads and target-derived values remain evaluation-only, never controller inputs. "
            "Budget expansion does not imply improvement; supported shear may change during lift and does not by itself establish a geometric cause."}
    destination = args.output / "comparison.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"comparison": str(destination), "sha256": digest(destination),
        "figures": sum(len(run["figures"]) for run in result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
