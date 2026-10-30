#!/usr/bin/env python3
"""Offline, hash-bound R16/R17 physical comparison; never run a controller."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import numpy as np

from audit_public_ycb_bias_recovery import checked_path, checked_stream, digest, tick

STEP = .002
FINGERS = ("thumb", "middle", "ring")
THUMB = ("left_thumb_MCP_FE", "left_thumb_IP")
PLOT_JOINTS = (*THUMB, "left_middle_PIP")
TOL = 1e-10


def statistics(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite recorded measurement")
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def flag_summary(values):
    ordered = sorted(values.items())
    count = sum(bool(value) for _, value in ordered)
    longest = current = 0
    previous = None
    for index, value in ordered:
        current = (current + 1 if previous is not None and index == previous + 1 else 1) if value else 0
        longest = max(longest, current)
        previous = index
    return {"unique_samples": len(ordered), "true_samples": count,
        "fraction": count / len(ordered) if ordered else None,
        "first_time_s": next((index * STEP for index, value in ordered if value), None),
        "sampled_contiguous_seconds": longest * STEP}


def unique_close(rows, start, end):
    return {tick(row["timestamp_s"]): row for row in rows
        if row["phase"] == "CLOSE" and start <= tick(row["timestamp_s"]) <= end}


def close_windows(rows, attempts):
    windows, active, search_starts, previous_phase = [], None, [], None
    for row in rows:
        index = tick(row["timestamp_s"])
        if row["phase"] == "SEARCH" and previous_phase != "SEARCH":
            search_starts.append(index - 1)
        previous_phase = row["phase"]
        if row["phase"] == "CLOSE":
            if active is None:
                active = [index - 1, index]
            active[1] = index
        elif active is not None:
            windows.append(active)
            active = None
    if active is not None:
        windows.append(active)
    output = []
    for ordinal, (start, end) in enumerate(windows, 1):
        prior = [row["attempt"] for row in attempts if row.get("belief_timestamp") is not None
            and float(row["belief_timestamp"]) <= (start + 1) * STEP + 1e-8]
        attempt_start = max((index for index in search_starts if index <= start), default=0)
        attempt_end = next((index for index in search_starts if index > start), tick(rows[-1]["timestamp_s"]))
        output.append({"attempt": max(prior) if prior else ordinal,
            "close_start_s": start * STEP, "close_end_s": end * STEP,
            "close_start_tick": start, "close_end_tick": end,
            "attempt_start_s": attempt_start * STEP, "attempt_end_s": attempt_end * STEP,
            "attempt_start_tick": attempt_start, "attempt_end_tick": attempt_end})
    return output


def attempt_probe_metrics(rows, window):
    selected = [row for row in rows
        if window["attempt_start_tick"] < tick(row["timestamp_s"]) <= window["attempt_end_tick"]]
    samples = {tick(row["timestamp_s"]): row for row in selected}
    active = {index: row for index, row in samples.items() if row["probe_active"]}
    displacements = [row["actual_probe_displacement_m"][2] for row in active.values()
        if row["actual_probe_displacement_m"] is not None]
    result = {"scope": "Whole attempt, all phases; unloaded before contact or after release is not lift evidence",
        "raw_observations": len(selected), "unique_observations": len(samples),
        "phases": dict(Counter(row["phase"] for row in samples.values())),
        "admission_true_events": sum(row.get("admission") is True for row in selected),
        "confirmation_true_events": sum(row.get("confirmation") is True for row in selected),
        "maximum_reported_eligible_seconds": max((row["probe_eligible_seconds"] for row in selected), default=0.),
        "maximum_reported_confirmed_seconds": max((row["probe_confirmed_seconds"] for row in selected), default=0.),
        "probe_active_vertical_displacement_m": statistics(displacements),
        "probe_active_table_load_n": statistics(row["table_load_n"] for row in active.values()),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in selected if row.get("stop_reason")))}
    for key in ("probe_active", "actual_probe_endpoint_reached", "sampled_fixed_normal_1g_feasible",
            "sampled_capacity_1g_feasible", "table_unloaded"):
        result[key] = flag_summary({index: row[key] for index, row in samples.items()})
    result["active_measured_1g_and_unloaded"] = flag_summary({index: bool(row["probe_active"]
        and row["sampled_fixed_normal_1g_feasible"] and row["table_unloaded"])
        for index, row in samples.items()})
    return result


def probe_metrics(rows, window):
    samples = unique_close(rows, window["close_start_tick"], window["close_end_tick"])
    ordered = [samples[index] for index in sorted(samples)]
    result = {"unique_samples": len(samples), "catalogue_weight_n": ordered[0]["catalogue_weight_n"],
        "table_load_n": statistics(row["table_load_n"] for row in ordered),
        "maximum_reported_eligible_seconds": max(row["probe_eligible_seconds"] for row in ordered),
        "maximum_reported_confirmed_seconds": max(row["probe_confirmed_seconds"] for row in ordered),
        "admissions": sum(row.get("admission") is True for row in ordered),
        "confirmations": sum(row.get("confirmation") is True for row in ordered),
        "preprobe_reasons": dict(Counter(reason for row in ordered for reason in row["preprobe_reasons"]))}
    for key in ("sampled_fixed_normal_1g_feasible", "sampled_capacity_1g_feasible",
            "geometry_1_5g_feasible", "table_unloaded", "probe_active"):
        result[key] = flag_summary({index: row[key] for index, row in samples.items()})
    series = {"time": [row["timestamp_s"] for row in ordered],
        "table_load": [row["table_load_n"] for row in ordered],
        "fixed_1g": [int(row["sampled_fixed_normal_1g_feasible"]) for row in ordered],
        "unloaded": [int(row["table_unloaded"]) for row in ordered]}
    return result, series


def preload_metrics(rows, window):
    start, end = window["close_start_tick"], window["close_end_tick"]
    selected = [row for row in rows if row["phase"] == "CLOSE" and start <= tick(row["timestamp_s"]) <= end]
    unique = unique_close(selected, start, end)
    feasible = [row for row in selected if row["current_target_lp"].get("feasible") and row["current_groups"]]
    errors = {finger: [] for finger in FINGERS}
    series = {"time": [], **{finger: {"target": [], "actual": []} for finger in FINGERS}}
    for index in sorted(unique):
        row = unique[index]
        valid = bool(row["current_target_lp"].get("feasible") and row["current_groups"])
        series["time"].append(row["timestamp_s"])
        for finger in FINGERS:
            groups = [group for key, group in row["current_groups"].items() if key.startswith(finger + ":")]
            series[finger]["target"].append(sum(group["target_n"] for group in groups) if valid and groups else np.nan)
            series[finger]["actual"].append(sum(group["measured_n"] for group in groups) if valid and groups else np.nan)
    for row in feasible:
        for key, group in row["current_groups"].items():
            errors[key.split(":", 1)[0]].append(abs(group["target_n"] - group["measured_n"]))
    within = lambda row: bool(row["current_groups"]) and all(
        abs(group["target_n"] - group["measured_n"]) <= .35 for group in row["current_groups"].values())
    return {"raw_observations": len(selected), "unique_observations": len(unique),
        "feasible_current_lp_observations": len(feasible),
        "all_group_errors_within_035n_raw_observations": sum(within(row) for row in selected),
        "all_group_errors_within_035n": flag_summary({index: within(row) for index, row in unique.items()}),
        "per_finger_group_absolute_error_n": {finger: statistics(values) for finger, values in errors.items()},
        "ready": flag_summary({index: row["ready"] for index, row in unique.items()}),
        "maximum_reported_continuous_seconds": max(row["continuous_seconds"] for row in selected),
        "reasons": dict(Counter(reason for row in selected for reason in row["reasons"]))}, series


def bias_metrics(rows, window, budget):
    start, end = window["close_start_tick"], window["close_end_tick"]
    selected = [row for row in rows if row["phase"] == "CLOSE" and start <= tick(row["timestamp_s"]) < end]
    per_joint, modes, guarded = {}, Counter(), set()
    recoveries = moving = mixed = invalid = empty = failed = 0
    first_recovery = first_moving = None
    for event in selected:
        modes[event["mode"]] += 1
        if event.get("additional_bias_limit_rad", budget) != budget:
            raise ValueError("Per-write budget differs from declared run")
        fingers = event["per_finger"]
        if fingers:
            guarded.add(tick(event["timestamp_s"]))
        recovery = [finger for finger in fingers if finger["mode"] == "BIAS_RECOVERY"]
        changed = any(abs(joint["applied_command_rad"] - joint["previous_reference_rad"]) > 1e-15
            for finger in recovery for joint in finger["joints"].values())
        recoveries += bool(recovery)
        moving += bool(recovery) and changed
        mixed += bool(recovery) and any(finger["mode"] == "NOMINAL" for finger in fingers)
        invalid += bool(recovery) and not event["fresh_target"]
        failed += bool(event["failure_latched"])
        if recovery and first_recovery is None:
            first_recovery = event["timestamp_s"]
        if changed and first_moving is None:
            first_moving = event["timestamp_s"]
        for finger in fingers:
            for name, joint in finger["joints"].items():
                value = joint["applied_command_rad"] - joint["before_actual_rad"] - joint["fixed_capture_bias_rad"]
                if not np.isfinite(value):
                    raise ValueError("Nonfinite added command bias")
                empty += not joint["feasible"]
                item = per_joint.setdefault(name, {"ticks": [], "added": [], "modes": []})
                item["ticks"].append(tick(event["timestamp_s"]))
                item["added"].append(value)
                item["modes"].append(finger["mode"])
    summary, series = {}, {}
    for name, item in per_joint.items():
        pairs = list(zip(item["ticks"], item["added"], strict=True))
        summary[name] = {"added_bias_rad": statistics(item["added"]), "modes": dict(Counter(item["modes"])),
            "positive_budget_boundary": flag_summary({index: budget - TOL <= value <= budget + TOL for index, value in pairs}),
            "negative_budget_boundary": flag_summary({index: -budget - TOL <= value <= -budget + TOL for index, value in pairs}),
            "outside_declared_budget_samples": sum(abs(value) > budget + TOL for _, value in pairs),
            "outside_old_002_budget_samples": sum(abs(value) > .02 + TOL for _, value in pairs)}
        if name in PLOT_JOINTS:
            series[name] = {"time": [index * STEP for index in item["ticks"]], "added": item["added"]}
    return {"write_modes": dict(modes), "guarded_write_events": len(guarded), "recovery_events": recoveries,
        "moving_recovery_events": moving, "pure_hold_recovery_events": recoveries - moving,
        "mixed_nominal_recovery_events": mixed, "invalid_target_recovery_events": invalid,
        "first_recovery_s": first_recovery, "first_moving_recovery_s": first_moving,
        "empty_joint_boxes": empty, "latched_failure_events": failed, "per_joint": summary}, series, guarded


def torque_metrics(native, metadata, window, guarded):
    if (metadata.get("force_evaluation_state") != "last_completed_dynamics_evaluation_cache"
            or metadata.get("observation_time_semantics") != "post_integration_native_sample"
            or metadata.get("postintegration_qpos_recomputed") is not False
            or metadata.get("evaluation_only") is not True):
        raise ValueError("Unsupported native force-cache timing contract")
    parameters = metadata["actuator_parameters"]
    offsets = {item["joint_name"]: index for index, item in enumerate(parameters)}
    summary, series = {}, {}
    for name in THUMB:
        offset, item = offsets[name], parameters[offsets[name]]
        values, indices, saturation = [], [], []
        for sample in native:
            # A write at t uses the dynamics cache observed after integration at t + dt.
            write_tick = tick(sample["timestamp"]) - 1
            if write_tick not in guarded:
                continue
            record = sample["native_actuation"]
            if record["actuator_ids"] != metadata["actuator_ids"] or record["joint_dof_ids"] != metadata["joint_dof_ids"]:
                raise ValueError("Native actuation mapping differs from metadata")
            value = float(record["qfrc_actuator"][offset])
            raw = float(record["actuator_force"][offset]) * item["gear"][0]
            values.append(value)
            indices.append(write_tick)
            saturation.append(abs(value - raw) > 1e-9)
        summary[name] = {"scope": "Guarded CLOSE writes only; cached native actuator torque, not net/contact torque",
            "torque_nm": statistics(values), "absolute_torque_nm": statistics(abs(value) for value in values),
            "joint_actuator_force_limits_nm": item["jointactfrcrange"] if item["jointactfrclimited"] else None,
            "joint_limit_clamping": flag_summary(dict(zip(indices, saturation, strict=True)))}
        series[name] = {"time": [(index + 1) * STEP for index in indices], "torque": values,
            "limits": summary[name]["joint_actuator_force_limits_nm"]}
    return summary, series


def plot_attempt(output, name, label, attempt, budget, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    origin = attempt["close_start_s"]
    relative = lambda values: np.asarray(values) - origin
    pressure = series["pressure"]
    for finger, color in zip(FINGERS, ("#bd3535", "#247c49", "#315ba5"), strict=True):
        axes[0].plot(relative(pressure["time"]), pressure[finger]["actual"], color=color, label=finger + " measured")
        axes[0].plot(relative(pressure["time"]), pressure[finger]["target"], color=color, linestyle="--", label=finger + " target")
    axes[0].set_ylabel("Associated group sum (N)")
    axes[0].set_title("Feasible current gate LP only; gaps are invalid targets")
    axes[0].legend(ncol=3, fontsize=8)
    for joint, values in series["bias"].items():
        axes[1].plot(relative(values["time"]), values["added"], label=joint.removeprefix("left_"))
    for value in (-budget, budget):
        axes[1].axhline(value, color="black", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Added command bias (rad)")
    axes[1].set_title(f"Declared budget +/-{budget:g} rad; guarded writes only")
    axes[1].legend(fontsize=8)
    probe = series["probe"]
    axes[2].plot(relative(probe["time"]), probe["table_load"], color="#545454", label="Table load")
    axes[2].axhline(attempt["probe"]["catalogue_weight_n"], color="#986b20", linestyle="--", label="Object weight")
    axes[2].set_ylabel("Virtual table load (N)")
    axes[2].legend(loc="upper left", fontsize=8)
    binary = axes[2].twinx()
    binary.step(relative(probe["time"]), probe["fixed_1g"], color="#247c49", where="post", label="Measured 1g feasible")
    binary.step(relative(probe["time"]), probe["unloaded"], color="#315ba5", linestyle=":", where="post", label="Table unloaded")
    binary.set_ylim(-.1, 1.2)
    binary.set_yticks([0, 1])
    binary.set_ylabel("Recorded gate")
    binary.legend(loc="upper right", fontsize=8)
    if series["torque"] is None:
        axes[3].text(.5, .5, "No direct native actuation cache recorded (R16: null)", ha="center", transform=axes[3].transAxes)
    else:
        for joint, values in series["torque"].items():
            line, = axes[3].plot(relative(values["time"]), values["torque"], label=joint.removeprefix("left_"))
            if values["limits"]:
                for limit in values["limits"]:
                    axes[3].axhline(limit, color=line.get_color(), linestyle=":", linewidth=.6)
        axes[3].legend(fontsize=8)
    axes[3].set_ylabel("Cached actuator torque (Nm)")
    axes[3].set_xlabel("Seconds after CLOSE begins")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(f"{label} | {name} | attempt {attempt['attempt']}", fontsize=12)
    stem = f"{name}_attempt{attempt['attempt']}"
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=150)
    plt.close(fig)
    return [f"{stem}.png", f"{stem}.pdf"]


def compare_run(directory, output, ordinal):
    summary_path = checked_path(directory, "summary.json")
    summary = json.loads(summary_path.read_text())
    protocol_path = checked_path(directory, "protocol.json")
    if not summary["complete"] or summary["protocol"] != json.loads(protocol_path.read_text()):
        raise ValueError("Run must be complete with matching summary/protocol")
    protocol = summary["protocol"]
    if protocol["format"] not in ("public_ycb_bias_recovery_development_r16", "public_ycb_bias_budget_development_r17"):
        raise ValueError("Only the explicit R16/R17 recorded protocols are supported")
    if protocol["source_changed_during_run"] or protocol["source_hashes_before"] != protocol["source_hashes_after"]:
        raise ValueError("Runtime source changed")
    budget = float(protocol["joint_admittance"]["additional_bias_limit_rad"])
    label = directory.name
    result = {"directory": str(directory), "summary_sha256": digest(summary_path),
        "protocol_sha256": digest(protocol_path), "budget_rad": budget, "cases": []}
    strict_path = directory / "strict_native_verification.json"
    strict_rows = {}
    result["strict_receipt"] = None
    if strict_path.exists():
        strict_path = checked_path(directory, "strict_native_verification.json")
        strict = json.loads(strict_path.read_text())
        if strict["summary_sha256"] != result["summary_sha256"]:
            raise ValueError("Strict receipt is bound to a different summary")
        strict_rows = {row["path"]: row for row in strict["rows"]}
        if len(strict_rows) != len(strict["rows"]) or set(strict_rows) != {row["path"] for row in summary["rows"]}:
            raise ValueError("Strict receipt case inventory mismatch")
        result["strict_receipt"] = {"path": str(strict_path), "sha256": digest(strict_path),
            "accepted_evidence": strict["accepted_evidence"], "strict_successes": strict["strict_successes"],
            "unknowns": strict["unknowns"], "thresholds": strict["thresholds"]}
    for case_index, row in enumerate(summary["rows"]):
        case = checked_path(directory, row["path"])
        if json.loads(checked_path(case, "metadata.json").read_text()) != row or row["runner_error"]:
            raise ValueError("Case metadata mismatch or incomplete runner")
        inputs = {}
        def stream(key):
            values = checked_stream(case, row["artifacts"][key])
            inputs[key] = row["artifacts"][key]
            return values
        rows = stream("load_transfer_probe")
        windows = close_windows(rows, row["attempts"])
        attempts, plots = [], []
        for window in windows:
            metrics, series = probe_metrics(rows, window)
            attempts.append({**window, "probe": metrics, "whole_attempt_probe": attempt_probe_metrics(rows, window)})
            plots.append({"probe": series})
        del rows
        gc.collect()
        rows = stream("preload")
        for attempt, series in zip(attempts, plots, strict=True):
            attempt["preload"], series["pressure"] = preload_metrics(rows, attempt)
        del rows
        gc.collect()
        rows = stream("bias_recovery")
        guards = []
        for attempt, series in zip(attempts, plots, strict=True):
            attempt["bias"], series["bias"], guarded = bias_metrics(rows, attempt, budget)
            guards.append(guarded)
        del rows
        gc.collect()
        actuation = row["native_audit"].get("native_actuation_metadata")
        native = stream("native") if actuation else None
        for index, (attempt, series) in enumerate(zip(attempts, plots, strict=True)):
            if native is None:
                attempt["direct_native_thumb_torque"] = series["torque"] = None
            else:
                attempt["direct_native_thumb_torque"], series["torque"] = torque_metrics(native, actuation, attempt, guards[index])
        del native
        gc.collect()
        figures = []
        name = f"run{ordinal:02d}_case{case_index:02d}"
        for attempt, series in zip(attempts, plots, strict=True):
            figures.extend(plot_attempt(output, name, label, attempt, budget, series))
        result["cases"].append({"kind": row["kind"], "seed": row["seed"], "path": row["path"],
            "task_success": row["success"], "held_lift": row["held_lift"], "failure": row.get("failure"),
            "reported_attempts": row["attempts"], "duration_s": row["duration_s"],
            "strict_native_verification": strict_rows.get(row["path"]),
            "native_audit": row["native_audit"], "hash_checked_streams": inputs,
            "attempts": attempts, "figures": figures})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    output = {"format": "offline_public_ycb_bias_budget_comparison_v1", "script_sha256": digest(__file__),
        "scope": "Descriptive hash-checked recorded streams, not a replacement for independent command/force/strict audits. "
            "Samples are correlated, not independent trials. R16 has no direct cached-torque measurement. "
            "Bias uses write-time actual encoders; cached torque observed at t+dt belongs to write t dynamics. "
            "CLOSE curves and pressure/bias metrics are separate from whole-attempt probe workflow. "
            "All-phase unloaded flags alone are not lift evidence. Optional strict receipt is summary-hash bound; "
            "task success remains the whole episode outcome.", "runs": []}
    for ordinal, directory in enumerate(args.runs, 1):
        output["runs"].append(compare_run(directory.resolve(strict=True), args.output, ordinal))
    destination = args.output / "comparison.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparison": str(destination), "runs": len(output["runs"])}))


if __name__ == "__main__":
    main()
