#!/usr/bin/env python3
"""Offline R20/R21 curves requiring finalized native, strict and command audits."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

import numpy as np

import plot_public_ycb_shared_normal as previous

common, metrics = previous.common, previous.metrics
digest, safe_path = common.digest, common.safe_path
PROJECT = Path(__file__).resolve().parents[1]
R20_SUMMARY_SHA256 = "a47ab784ed6b0879f149333a08e02928b5ae5430ec93af76af67de02ef63dd7b"
R20_RECEIPTS = {
    "native_verification": "302af5189b5f475fa2903dae49e69cb088da624b807160f083ff4fb3e1537e30",
    "strict_verification": "0b81b3c048730ccf4a5d46f857afc28c91745b8ddbb47f4602cc7d2239232897",
    "command_verification": "7c7848dfa647d312b555f09041d56b9d1a2bbf712f26f49a2d5dd9a8c025d0d0"}
FORMATS = {"baseline": "public_ycb_shared_normal_development_r20",
    "candidate": "public_ycb_provisional_preload_development_r21"}
CLASSES = {"baseline": "SharedNormalBudgetHand", "candidate": "ProvisionalPreloadBudgetHand"}
CANDIDATE_AUDIT_FORMAT = "r21_provisional_preload_command_native_target_and_policy_audit_v1"
CANDIDATE_AUDITOR = "scripts/audit_public_ycb_provisional_preload.py"
COMMAND_PATHS = {"baseline": PROJECT / "results/r20_shared_normal_postrun_audit_20261022/receipt.json",
    "candidate": PROJECT / "results/r21_provisional_preload_audit_20261022/receipt.json"}
CHECKS = ("all_evidence_consistent", "all_declared_command_bounds_satisfied",
    "all_native_close_spans_within_declared_deadline", "all_recorded_static_target_checks_consistent",
    "all_sampled_hypothetical_target_checks_passed", "all_recorded_provisional_policy_rows_consistent")
BUDGET_RAD, CLOSE_LIMIT_S, ARM_SECONDS = .10, 32., 16.
OBJECT_MOTION_SCOPE = (
    "Native target_qpos[2] is the postintegration free-joint-origin world height, evaluation only. "
    "Each actual probe_start_s baseline and active observation require an exact recorded native 2ms tick; "
    "there is no interpolation or extrapolation. Curves use the final probe-active state at each clock. "
    "Wrist motion uses cached world poses whose timestamps may differ by one dynamics evaluation from "
    "the integrated object state. Comparing these curves is not exact slip, clearance or task-success evidence.")


def case_directory(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Evidence case path escapes run directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink evidence is not accepted")
    path = path.resolve(strict=True)
    if not path.is_dir() or not path.is_relative_to(root):
        raise ValueError("Missing or escaping evidence case directory")
    return path


def source_bindings():
    paths = (Path(__file__).resolve(), Path(previous.__file__).resolve(), Path(previous.previous.__file__).resolve(),
        Path(common.__file__).resolve(), Path(metrics.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_close_deadline.py",
        PROJECT / "scripts/audit_public_ycb_bias_recovery.py",
        PROJECT / previous.POSTRUN_AUDITOR, PROJECT / CANDIDATE_AUDITOR)
    return {path.relative_to(PROJECT).as_posix(): digest(path) for path in paths}


def audit_receipts(directory, role, summary_bound, protocol_bound, row, protocol, command_path):
    if role == "baseline":
        inputs, state = previous.audit_receipts(directory, "candidate", summary_bound["sha256"],
            protocol_bound["sha256"], row, protocol, command_path)
        if any(inputs[name]["sha256"] != expected for name, expected in R20_RECEIPTS.items()):
            raise ValueError("Exact R20 native/strict/postrun comparison receipt differs")
        return inputs, state
    inputs, receipts = {}, {}
    for name, path in (("native_verification", safe_path(directory, "verification_receipt.json")),
            ("strict_verification", safe_path(directory, "strict_native_verification.json")),
            ("command_verification", command_path.resolve(strict=True))):
        receipts[name], inputs[name] = common.captured_json(path)
        if receipts[name]["summary_sha256"] != summary_bound["sha256"]:
            raise ValueError("Candidate receipt binds a different finalized summary: " + name)
    native, strict, command = (receipts[name] for name in
        ("native_verification", "strict_verification", "command_verification"))
    if (native["format"] != "public_ycb_development_native_verification_r1" or native["complete"] is not True
            or native["unknowns"] != 0 or native["source_hashes_before_equal_after"] is not True
            or native["native_samples_contiguous_2ms"] is not True or native["audited_episodes"] != 1
            or len(native["rows"]) != 1 or native["rows"][0]["path"] != row["path"]
            or native["rows"][0]["outcome"] != row["success"] or native["rows"][0]["held_lift"] != row["held_lift"]
            or strict["format"] != "public_ycb_strict_native_audit_20261021"
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or len(strict["rows"]) != 1 or strict["rows"][0]["path"] != row["path"]
            or strict["rows"][0]["checks"]["legacy_task_success"] != row["success"]
            or strict["rows"][0]["checks"]["held_lift"] != row["held_lift"]):
        raise ValueError("Candidate native/strict receipt is incomplete or disagrees with the outcome")
    if (command["format"] != CANDIDATE_AUDIT_FORMAT or command["protocol_sha256"] != protocol_bound["sha256"]
            or command["script_sha256"] != protocol["source_hashes_before"][CANDIDATE_AUDITOR]
            or command["script_sha256"] != digest(PROJECT / CANDIDATE_AUDITOR)
            or any(command[key] is not True for key in CHECKS)
            or command["additional_bias_limit_rad"] != BUDGET_RAD or command["close_seconds"] != CLOSE_LIMIT_S
            or command["arm_interpolation_seconds"] != ARM_SECONDS or len(command["cases"]) != 1):
        raise ValueError("Required R21 source/command/cache/target/policy audit is absent or inconsistent")
    case = command["cases"][0]
    if (case["case"] != row["path"] or case["actual_controller_class"] != CLASSES[role]
            or case["task_success"] != row["success"] or case["held_lift"] != row["held_lift"]
            or case["effective_hand_instances"] != row["effective_hand_instances"]
            or case["actuation"]["all_cached_force_and_transmitted_torque_consistent"] is not True
            or case["provisional_policy"]["all_recorded_policy_rows_consistent"] is not True):
        raise ValueError("R21 audit case, actual class/attempts or policy/actuation binding differs")
    return inputs, {"native_evidence_complete": True, "strict_evidence_accepted": True,
        "command_cache_target_and_policy_consistent": True, "reported_task_success": row["success"],
        "reported_held_lift": row["held_lift"], "strict_task_success": strict["rows"][0]["strict_success"],
        "scope": "Required receipts accept evidence consistency; they may retain task failure. This plotter does not establish normal process exit or independently rebuild all dynamic gates."}


def reduced_probe(path):
    fields = (*common.PROBE_KEYS, "stage", "probe_start_s", "decisive_current_solve",
        "sampled_solve_current", "cached_measurements_match", "solve_age_s",
        "accepted_visual_after_probe_start", "motion_safety_reasons", "preload_admission",
        "preload_reasons", "preload_continuous_seconds", "r16_recovery_inhibited")
    rows, previous_tick = [], -1
    for ordinal, row in enumerate(common.json_array(path)):
        index = metrics.tick(row["timestamp_s"])
        if index < previous_tick:
            raise ValueError("Probe observations are not chronological")
        previous_tick = index
        rows.append({**{key: row[key] for key in fields if key in row}, "stream_ordinal": ordinal})
    return rows


def reduced_preload(path):
    rows, previous_tick = [], -1
    for row in common.json_array(path):
        index = metrics.tick(row["timestamp_s"])
        if index < previous_tick:
            raise ValueError("Preload observations are not chronological")
        previous_tick = index
        observation = row["original_probe_snapshot"]
        rows.append({**{key: row[key] for key in ("timestamp_s", "phase", "current_groups", "ready",
                "continuous_seconds", "reasons", "admission_recheck")},
            "current_target_lp": {key: row["current_target_lp"].get(key) for key in ("feasible", "reason")},
            "r16_recovery_inhibited": row.get("r16_recovery_inhibited", False),
            **{key: observation[key] for key in ("pressure_window_s", "pressure_spans_n", "pressure_stable")}})
    return rows


def reduced_bias(path):
    rows, previous_tick = [], -1
    for row in common.json_array(path):
        index = metrics.tick(row["timestamp_s"])
        if index <= previous_tick:
            raise ValueError("Bias writes are not strictly chronological")
        previous_tick = index
        rows.append({**{key: row[key] for key in ("timestamp_s", "phase", "mode", "additional_bias_limit_rad",
                "fresh_target", "failure_latched", "post_step_timestamp_s", "probe_abort_applied",
                "resolved_abort_reason", "post_step_returned_failure", "original_post_step_failure")},
            "per_finger": [{"mode": finger["mode"], "joints": {name: {key: joint[key] for key in
                ("applied_command_rad", "previous_reference_rad", "before_actual_rad", "fixed_capture_bias_rad", "feasible")}
                for name, joint in finger["joints"].items()}} for finger in row["per_finger"]]})
    return rows


def attempt_windows(probe, attempts):
    starts, previous_phase = [], None
    for row in probe:
        if row["phase"] == "SEARCH" and previous_phase != "SEARCH":
            starts.append(metrics.tick(row["timestamp_s"]) - 1)
        previous_phase = row["phase"]
    if len(starts) != len(attempts) or not starts or len(set(starts)) != len(starts):
        raise ValueError("Actual SEARCH epochs do not match recorded attempt count")
    close = metrics.close_windows(probe, attempts)
    known = {attempt["attempt"] for attempt in attempts}
    if any(window["attempt"] not in known for window in close):
        raise ValueError("CLOSE interval is not associated with a recorded attempt")
    result = []
    for index, attempt in enumerate(attempts):
        windows = [window for window in close if window["attempt"] == attempt["attempt"]]
        if len(windows) > 1:
            raise ValueError("Unexpected multiple distinct CLOSE intervals within one attempt")
        end = starts[index + 1] if index + 1 < len(starts) else metrics.tick(probe[-1]["timestamp_s"])
        result.append({"attempt": attempt["attempt"], "attempt_start_tick": starts[index], "attempt_end_tick": end,
            "attempt_start_s": starts[index] * metrics.STEP, "attempt_end_s": end * metrics.STEP,
            "close": windows[0] if windows else None, "recorded_attempt_failure": attempt.get("failure")})
    return result


def decision_metrics(probe, bias, attempt):
    inside = lambda time: attempt["attempt_start_tick"] < metrics.tick(time) <= attempt["attempt_end_tick"]
    selected = [row for row in probe if inside(row["timestamp_s"])]
    events = [{key: row.get(key) for key in ("stream_ordinal", "timestamp_s", "phase", "admission", "confirmation",
        "decisive_current_solve", "sampled_solve_current", "cached_measurements_match", "solve_age_s",
        "sampled_fixed_normal_1g_feasible", "probe_active", "actual_probe_endpoint_reached", "table_unloaded",
        "probe_confirmed_seconds", "preload_reasons", "stop_reason")}
        for row in selected if "admission" in row or "confirmation" in row or row.get("stop_reason")]
    admissions = [row for row in events if row["admission"] is True]
    if any(row["decisive_current_solve"] is not True or row["sampled_solve_current"] is not True
            or row["cached_measurements_match"] is not True or row["solve_age_s"] is None
            or abs(row["solve_age_s"]) > 1e-9 for row in admissions):
        raise ValueError("A reported admission does not retain an explicit fresh decisive-solve record")
    aborts = [{key: row[key] for key in ("timestamp_s", "post_step_timestamp_s", "phase",
        "probe_abort_applied", "resolved_abort_reason", "post_step_returned_failure", "original_post_step_failure")}
        for row in bias if inside(row["post_step_timestamp_s"])
        and (row["probe_abort_applied"] or row["post_step_returned_failure"])]
    unique = {metrics.tick(row["timestamp_s"]): row for row in selected}
    active = {index: row for index, row in unique.items() if row["probe_active"]}
    return {"whole_attempt": metrics.attempt_probe_metrics(probe, attempt),
        "first_fresh_admission_s": admissions[0]["timestamp_s"] if admissions else None,
        "fresh_admission_records": len(admissions), "raw_decision_events": events,
        "r16_post_step_failure_or_abort_events": aborts,
        "r16_resolved_abort_reasons": dict(Counter(row["resolved_abort_reason"] for row in aborts if row["resolved_abort_reason"])),
        "active_endpoint_and_actual_1g_and_unloaded": metrics.flag_summary({index: bool(
            row["probe_active"] and row["actual_probe_endpoint_reached"]
            and row["sampled_fixed_normal_1g_feasible"] and row["table_unloaded"])
            for index, row in unique.items()}),
        "probe_active_horizontal_displacement_m": metrics.statistics(
            np.linalg.norm(row["actual_probe_displacement_m"][:2]) for row in active.values()
            if row["actual_probe_displacement_m"] is not None),
        "scope": "Raw same-clock decision records preserve stream order; sampled flags use the final observation at each clock. Admission occurs before probe activation. Original probe snapshots can precede a queued R16 abort, so final post-step failure/abort events are reported separately. Probe-active wrist displacement is not object lift, endpoint/1g/table samples are not a rebuilt continuous confirmation, and release/table unloading outside active probe is not held evidence."}


def stream_object_height(path):
    heights = {}
    for ordinal, row in enumerate(common.json_array(path), 1):
        index = metrics.tick(row["timestamp"])
        if index != ordinal:
            raise ValueError("Object-height native samples are not contiguous from the first timestep")
        height = float(row["target_qpos"][2])
        if not np.isfinite(height):
            raise ValueError("Nonfinite native free-joint height")
        heights[index] = (float(row["timestamp"]), height)
    return heights


def object_probe_motion(probe, attempt, heights):
    selected = {metrics.tick(row["timestamp_s"]): row for row in probe
        if attempt["attempt_start_tick"] < metrics.tick(row["timestamp_s"]) <= attempt["attempt_end_tick"]}
    series, segments = {"time": [], "displacement_m": []}, {}
    for index, row in selected.items():
        series["time"].append(row["timestamp_s"])
        if not row["probe_active"]:
            series["displacement_m"].append(np.nan)
            continue
        if row.get("probe_start_s") is None:
            raise ValueError("Active object-motion observation has no actual probe start")
        start_tick = metrics.tick(row["probe_start_s"])
        if start_tick > index or start_tick not in heights or index not in heights:
            raise ValueError("Object motion requires exact native start and observation ticks")
        baseline_time, baseline_height = heights[start_tick]
        timestamp, height = heights[index]
        if (abs(baseline_time - row["probe_start_s"]) > 1e-8
                or abs(timestamp - row["timestamp_s"]) > 1e-8):
            raise ValueError("Object-motion native and probe timestamps differ")
        displacement = height - baseline_height
        series["displacement_m"].append(displacement)
        segment = segments.setdefault(start_tick, {"probe_start_s": row["probe_start_s"],
            "baseline_native_timestamp_s": baseline_time, "baseline_native_tick": start_tick,
            "baseline_free_joint_height_m": baseline_height, "displacements": [], "last_s": timestamp})
        segment["displacements"].append(displacement)
        segment["last_s"] = timestamp
    result = {"native_timestamp_field": "timestamp", "native_height_field": "target_qpos[2]",
        "probe_start_field": "probe_start_s", "exact_native_tick_matching": True,
        "interpolated_or_extrapolated": False, "recorded_probe_starts": len(segments),
        "active_unique_observations": sum(len(value["displacements"]) for value in segments.values()),
        "probe_segments": [], "scope": OBJECT_MOTION_SCOPE}
    for segment in segments.values():
        values = segment.pop("displacements")
        result["probe_segments"].append({**segment, "object_vertical_displacement_m": metrics.statistics(values)})
    return result, series


def save_figure(fig, output, stem):
    import matplotlib.pyplot as plt
    names = [stem + ".png", stem + ".pdf"]
    for name in names:
        fig.savefig(output / name, dpi=150)
    plt.close(fig)
    return names


def progress_plot(output, role, label, window, preload, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(metrics.unique_close(preload, window["close_start_tick"], window["close_end_tick"]).values())
    observations = list(metrics.unique_close(probe, window["close_start_tick"], window["close_end_tick"]).values())
    origin = window["close_start_s"]
    times = [row["timestamp_s"] - origin for row in rows]
    other_times = [row["timestamp_s"] - origin for row in observations]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for key in sorted({key for row in rows for key in row["current_groups"]}):
        axes[0].plot(times, [abs(row["current_groups"][key]["target_n"] - row["current_groups"][key]["measured_n"])
            if key in row["current_groups"] else np.nan for row in rows], linewidth=.8, label=key)
    axes[0].axhline(.35, color="black", linestyle="--", label="Required group error <=0.35 N")
    axes[0].set_ylabel("Group error (N)")
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].plot(times, [row["continuous_seconds"] for row in rows], label="Continuous preload", color="#bd3535")
    axes[1].axhline(.25, color="black", linestyle="--", label="Required 0.25 s")
    axes[1].set_ylabel("Preload (s)")
    axes[1].legend(loc="upper left", fontsize=8)
    eligible = axes[1].twinx()
    eligible.plot(other_times, [row["probe_eligible_seconds"] for row in observations], color="#986b20", label="Original eligibility")
    eligible.set_ylabel("Original eligibility (s)")
    eligible.legend(loc="upper right", fontsize=8)
    axes[2].plot(times, [row["pressure_window_s"] for row in rows], label="Pressure-window age", color="#315ba5")
    axes[2].axhline(.25, color="#315ba5", linestyle="--")
    axes[2].set_ylabel("Window age (s)")
    axes[2].legend(loc="upper left", fontsize=8)
    spans = axes[2].twinx()
    spans.plot(times, [max(row["pressure_spans_n"].values()) for row in rows], color="#bd3535", label="Maximum channel span")
    spans.axhline(.3, color="#bd3535", linestyle="--", label="Maximum 0.3 N")
    spans.set_ylabel("Pressure span (N)")
    spans.legend(loc="upper right", fontsize=8)
    for key, text in (("sampled_fixed_normal_1g_feasible", "Observed actual fixed-normal 1g"),
            ("pressure_stable", "Pressure stable"), ("probe_active", "Probe active")):
        axes[3].step(other_times, [int(row[key]) for row in observations], where="post", label=text)
    axes[3].step(times, [int(row["r16_recovery_inhibited"]) for row in rows], where="post", label="Recovery inhibits preload")
    axes[3].set(ylim=(-.1, 1.2), yticks=[0, 1], ylabel="Recorded flags", xlabel="Seconds after actual CLOSE begins")
    axes[3].legend(ncol=2, fontsize=8)
    duration = window["close_end_s"] - origin
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, duration)
        if ARM_SECONDS <= duration:
            axis.axvline(ARM_SECONDS, color="#777777", linestyle=":", linewidth=.8)
    fig.suptitle(f"{label} | attempt {window['attempt']} | observed CLOSE {duration:.3f} s (limit 32 s)", fontsize=11)
    return save_figure(fig, output, f"{role}_attempt{window['attempt']}_gate_progress")


def probe_plot(output, role, label, attempt, probe, decisions, parameters, object_series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in probe if attempt["attempt_start_tick"] < metrics.tick(row["timestamp_s"]) <= attempt["attempt_end_tick"]]
    rows = list({metrics.tick(row["timestamp_s"]): row for row in selected}.values())
    origin = attempt["attempt_start_s"]
    times = np.asarray([row["timestamp_s"] - origin for row in rows])
    displacement = np.asarray([row["actual_probe_displacement_m"] if row["probe_active"]
        and row["actual_probe_displacement_m"] is not None else [np.nan] * 3 for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    axes[0].plot(times, 1000 * displacement[:, 2], label="Probe-active wrist vertical motion")
    axes[0].plot(np.asarray(object_series["time"]) - origin,
        1000 * np.asarray(object_series["displacement_m"]),
        label="Postintegration free-joint height, evaluation only", color="#bb4430")
    axes[0].plot(times, 1000 * np.linalg.norm(displacement[:, :2], axis=1), label="Probe-active wrist horizontal drift")
    axes[0].axhline(parameters["probe_distance_m"] * 1000, color="black", linestyle="--", label="Declared 5 mm endpoint")
    axes[0].axhline(parameters["maximum_horizontal_probe_drift_m"] * 1000, color="#bd3535", linestyle=":", label="Horizontal limit")
    axes[0].set_ylabel("Vertical motion / drift (mm)")
    axes[0].legend(fontsize=8)
    axes[1].plot(times, [row["table_load_n"] for row in rows], color="#545454", label="Actual virtual table load")
    weight = rows[0]["catalogue_weight_n"]
    axes[1].axhline(weight, color="#986b20", linestyle="--", label="Catalogue weight")
    axes[1].axhline(weight * parameters["maximum_remaining_table_weight_fraction"], color="#247c49", linestyle=":", label="Confirmation table-load ceiling")
    axes[1].set_ylabel("Table load (N)")
    axes[1].legend(fontsize=8)
    axes[2].plot(times, [row["probe_confirmed_seconds"] for row in rows], color="#247c49", label="Reported continuous confirmation")
    axes[2].axhline(parameters["confirmation_seconds"], color="black", linestyle="--", label="Required 0.25 s")
    axes[2].set_ylabel("Confirmation (s)")
    axes[2].legend(fontsize=8)
    for field, name in (("probe_active", "Probe active"), ("sampled_fixed_normal_1g_feasible", "Observed fixed-normal 1g"),
            ("table_unloaded", "Table unloaded"), ("actual_probe_endpoint_reached", "Actual endpoint reached")):
        axes[3].step(times, [int(row[field]) for row in rows], where="post", label=name, linewidth=.9)
    event_times = []
    for event in decisions["raw_decision_events"]:
        if event["admission"] is not None or event["confirmation"] is not None:
            when = event["timestamp_s"] - origin
            positive = event["admission"] is True or event["confirmation"] is True
            name = ("Fresh admission" if event["admission"] is True else "Admission rejected") if event["admission"] is not None else (
                "Confirmation accepted" if event["confirmation"] else "Confirmation rejected")
            axes[3].scatter([when], [1.08 if positive else -.08], marker="^" if positive else "x", label=name, s=30)
            event_times.append(event["timestamp_s"])
    reasons = set()
    for event in decisions["r16_post_step_failure_or_abort_events"]:
        reason = event["resolved_abort_reason"] or event["post_step_returned_failure"]
        if reason:
            when = event["post_step_timestamp_s"] - origin
            axes[3].scatter([when], [.5], marker="x", color="#bd3535", s=40, label=reason if reason not in reasons else None)
            reasons.add(reason)
            event_times.append(event["post_step_timestamp_s"])
    for event in decisions["raw_decision_events"]:
        if event["stop_reason"] and event["stop_reason"] not in reasons:
            axes[3].scatter([event["timestamp_s"] - origin], [.5], marker="x", color="#bd3535", label=event["stop_reason"])
            reasons.add(event["stop_reason"])
    active_times = [row["timestamp_s"] for row in rows if row["probe_active"]]
    admitted = [row["timestamp_s"] for row in decisions["raw_decision_events"] if row["admission"] is True]
    if active_times or admitted:
        first = min([*active_times, *admitted])
        last = max([*active_times, *admitted, *event_times])
        limits = (max(origin, first - .3) - origin, min(attempt["attempt_end_s"], last + .3) - origin)
    else:
        limits = ((attempt["close"]["close_start_s"] if attempt["close"] else origin) - origin,
            (attempt["close"]["close_end_s"] if attempt["close"] else attempt["attempt_end_s"]) - origin)
    axes[3].set(ylim=(-.18, 1.22), yticks=[0, 1], ylabel="Recorded flags / events", xlabel="Seconds after actual attempt begins")
    handles, labels = axes[3].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[3].legend(unique.values(), [textwrap.fill(name, 42) for name in unique], ncol=2, fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(*limits)
    state = "actual fresh admission recorded" if admitted else "no actual admission recorded"
    fig.suptitle(f"{label} | attempt {attempt['attempt']} | {state}\nCached wrist and integrated object states may differ by one dynamics evaluation; not exact slip or clearance", fontsize=10)
    return save_figure(fig, output, f"{role}_attempt{attempt['attempt']}_probe_detail")


def plot_run(directory, role, output, command_path, expected_summary_sha256):
    directory = directory.resolve(strict=True)
    summary, summary_bound = common.captured_json(safe_path(directory, "summary.json"))
    protocol, protocol_bound = common.captured_json(safe_path(directory, "protocol.json"))
    if (summary_bound["sha256"] != expected_summary_sha256 or summary["complete"] is not True
            or summary["protocol"] != protocol or protocol["format"] != FORMATS[role]
            or summary["runner_revision"] != FORMATS[role] or protocol["runner_revision"] != FORMATS[role]
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1
            or protocol["controller_class"] != CLASSES[role] or protocol["target_allocator_revision"] != FORMATS["baseline"]
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_LIMIT_S
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_LIMIT_S):
        raise ValueError("Expected an exact finalized actual-format R20/R21 .10rad/CLOSE-limit32 case")
    if role == "candidate":
        declaration = protocol["r21_provisional_preload_experiment"]
        if (protocol["admission_policy_revision"] != FORMATS[role]
                or declaration["r20_comparison_summary_sha256"] != R20_SUMMARY_SHA256
                or declaration["additional_bias_limit_rad"] != BUDGET_RAD or declaration["close_seconds"] != CLOSE_LIMIT_S
                or declaration["arm_interpolation_seconds"] != ARM_SECONDS or declaration["native_timestep_s"] != metrics.STEP):
            raise ValueError("R21 admission revision, baseline or declared clocks differ")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    case = case_directory(directory, row["path"])
    metadata, metadata_bound = common.captured_json(safe_path(case, "metadata.json"))
    count = row["effective_hand_instances"]
    if (metadata != row or row["runner_error"] or any(row[key] != value for key, value in planned.items())
            or row.get("missing_runtime_controller_binding", False) is not False
            or row.get("missing_runtime_allocator_binding", False) is not False
            or type(count) is not int or not 1 <= count <= protocol["maximum_attempts"]
            or count != len(row["attempts"]) or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * count
            or row["effective_hand_controller_types"] != [CLASSES[role]] * count
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("Actual hand/attempt/clock binding differs")
    inputs = {"summary": summary_bound, "protocol": protocol_bound, "metadata": metadata_bound}
    for name, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"), ("source_archive", "frozen_source.zip")):
        path = safe_path(directory if name == "source_archive" else case, filename)
        inputs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inputs["scene"]["sha256"]
            or row["scene_sha256_at_finalize"] != inputs["scene"]["sha256"]
            or row["derived_controller_sha256"] != inputs["controller"]["sha256"]
            or float(ET.fromstring(Path(inputs["scene"]["path"]).read_bytes()).find("option").attrib["timestep"]) != metrics.STEP):
        raise ValueError("Runtime-loaded scene, timestep or derived arm source differs")
    receipt_inputs, audit_state = audit_receipts(directory, role, summary_bound, protocol_bound, row, protocol, command_path)
    inputs.update(receipt_inputs)
    native_receipt = json.loads(Path(inputs["native_verification"]["path"]).read_text())
    if native_receipt["frozen_source_sha256"] != inputs["source_archive"]["sha256"]:
        raise ValueError("Frozen source archive is not bound to the native receipt")
    paths = {}
    for name in ("load_transfer_probe", "preload", "bias_recovery", "native"):
        paths[name], inputs[name] = common.bind_artifact(case, row["artifacts"][name])
    probe, preload, bias = reduced_probe(paths["load_transfer_probe"]), reduced_preload(paths["preload"]), reduced_bias(paths["bias_recovery"])
    object_heights = stream_object_height(paths["native"])
    object_native_count = len(object_heights)
    attempts, curves, guards, object_curves = [], [], [], []
    for attempt in attempt_windows(probe, row["attempts"]):
        decisions = decision_metrics(probe, bias, attempt)
        object_stats, object_series = object_probe_motion(probe, attempt, object_heights)
        value = {**attempt, "decisions": decisions, "object_motion": object_stats}
        object_curves.append(object_series)
        if attempt["close"]:
            window = attempt["close"]
            probe_stats, probe_series = metrics.probe_metrics(probe, window)
            preload_stats, pressure_series = metrics.preload_metrics(preload, window)
            bias_stats, bias_series, guarded = metrics.bias_metrics(bias, window, BUDGET_RAD)
            value["close_metrics"] = {**window, "probe": probe_stats, "preload": preload_stats,
                "bias": bias_stats, "preload_context": previous.previous.preload_context(preload, window)}
            curves.append({"probe": probe_series, "pressure": pressure_series, "bias": bias_series})
            guards.append(guarded)
        else:
            value["close_metrics"] = None
            curves.append(None)
            guards.append(set())
        attempts.append(value)
    del bias, object_heights
    gc.collect()
    torque_stats, torque_curves, native_count = common.stream_torque(paths["native"], row["native_audit"]["native_actuation_metadata"], guards)
    if (native_count != row["native_audit"]["native_samples"]
            or native_count != row["native_audit"]["native_actuation_samples"]
            or native_count != native_receipt["native_samples"] or native_count != object_native_count):
        raise ValueError("Native observation denominator differs")
    label = f"{role.upper()} {'R20 original admission' if role == 'baseline' else 'R21 provisional admission'} | .10rad | arm16"
    figures = []
    for index, attempt in enumerate(attempts):
        attempt["direct_native_close_thumb_torque"] = torque_stats[index]
        if attempt["close_metrics"]:
            curves[index]["torque"] = torque_curves[index]
            figures.extend(metrics.plot_attempt(output, role, label, attempt["close_metrics"], BUDGET_RAD, curves[index]))
            figures.extend(progress_plot(output, role, label, attempt["close_metrics"], preload, probe))
        figures.extend(probe_plot(output, role, label, attempt, probe, attempt["decisions"],
            protocol["load_transfer_probe"], object_curves[index]))
    for bound in inputs.values():
        path = Path(bound["path"])
        if path.stat().st_size != bound["bytes"] or digest(path) != bound["sha256"]:
            raise ValueError("Bound plotting input changed during analysis")
    return {"role": role, "directory": str(directory), "protocol_format": protocol["format"],
        "controller_class": CLASSES[role], "additional_bias_limit_rad": BUDGET_RAD,
        "declared_close_limit_seconds": CLOSE_LIMIT_S, "arm_interpolation_seconds": ARM_SECONDS,
        "case": {"kind": row["kind"], "seed": row["seed"], "path": row["path"], "reported_success": row["success"],
            "reported_held_lift": row["held_lift"], "reported_failure": row.get("failure")},
        "required_audit_receipts": audit_state, "inputs": inputs, "native_samples_streamed": native_count,
        "actual_hand_instances": count, "attempts": attempts,
        "figures": [{"path": name, "bytes": (output / name).stat().st_size, "sha256": digest(output / name)} for name in figures]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-candidate-summary-sha256", required=True)
    parser.add_argument("--baseline-command-receipt", type=Path, default=COMMAND_PATHS["baseline"])
    parser.add_argument("--candidate-command-receipt", type=Path, default=COMMAND_PATHS["candidate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.expected_candidate_summary_sha256) != 64 or any(char not in "0123456789abcdef" for char in args.expected_candidate_summary_sha256):
        parser.error("Candidate summary SHA256 must be explicit lowercase hexadecimal")
    sources = source_bindings()
    candidate_directory = args.candidate.resolve(strict=True)
    candidate_summary, candidate_bound = common.captured_json(safe_path(candidate_directory, "summary.json"))
    candidate_protocol, candidate_protocol_bound = common.captured_json(safe_path(candidate_directory, "protocol.json"))
    if (candidate_bound["sha256"] != args.expected_candidate_summary_sha256
            or candidate_summary["complete"] is not True or candidate_summary["protocol"] != candidate_protocol
            or len(candidate_summary["rows"]) != 1):
        raise ValueError("Final candidate summary is absent, incomplete or not explicitly hash-bound")
    # Refuse absent/failed candidate audits before reading either run's large streams.
    audit_receipts(candidate_directory, "candidate", candidate_bound, candidate_protocol_bound,
        candidate_summary["rows"][0], candidate_protocol, args.candidate_command_receipt)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = plot_run(args.baseline, "baseline", args.output, args.baseline_command_receipt, R20_SUMMARY_SHA256)
    candidate = plot_run(args.candidate, "candidate", args.output, args.candidate_command_receipt, args.expected_candidate_summary_sha256)
    if (any(baseline["case"][key] != candidate["case"][key] for key in ("kind", "seed"))
            or any(baseline["inputs"][key]["sha256"] != candidate["inputs"][key]["sha256"] for key in ("scene", "controller"))):
        raise ValueError("Comparison case, loaded scene or actual derived arm source differs")
    if source_bindings() != sources:
        raise ValueError("Offline plotting source changed during analysis")
    result = {"format": "independent_r20_r21_provisional_preload_plots_v1", "independent_postrun_analysis": True,
        "part_of_runtime_frozen_manifest": False, "analysis_source_sha256": sources, "runs": [baseline, candidate],
        "physical_prefix_independently_compared": False,
        "object_motion_scope": OBJECT_MOTION_SCOPE,
        "scope": "Finalized receipt-bound R20/R21 actual admission comparison at unchanged .10rad, 32s maximum CLOSE and 16s arm clock. Actual attempt count and observed CLOSE endpoints are derived from recorded phases; successful early admission need not use the full deadline. Raw same-clock decisions retain their sequence, while state curves use the final observation at each clock. Native thumb torque is the cached actuator quantity for guarded CLOSE writes, not net/contact torque. Probe detail distinguishes cached wrist displacement from postintegration free-joint-origin object height, evaluation only; neither is a rebuilt held-lift outcome. Final R16 abort events remain separate from pre-abort original probe snapshots. Table unloading outside active probe and isolated 1g/endpoint samples are not confirmed held evidence. Original native/strict/command/policy audits remain mandatory; plots do not independently rebuild every continuous gate, establish normal process exit, certify safety, compare physical prefixes or produce a public randomized success rate. Retries are history-dependent, not independent repetitions."}
    destination = args.output / "comparison.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"comparison": str(destination), "sha256": digest(destination),
        "figures": sum(len(run["figures"]) for run in result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
