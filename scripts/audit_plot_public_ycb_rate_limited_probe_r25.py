#!/usr/bin/env python3
"""Narrow finalized R25 reference-governor audit and recorded-cache speed plots."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import plot_public_ycb_probe_speed as speed


PROJECT = Path(__file__).resolve().parents[1]
FORMAT = "r25_recorded_reference_governor_audit_and_speed_diagnostics_v1"
REVISION = "public_ycb_rate_limited_probe_development_r25"
ENTRY = "scripts/develop_public_ycb_rate_limited_probe_r25.py"
CLASS = "FreshContactTrackingHand"
TELEMETRY_FORMAT = "rate_limited_actual_wrist_probe_policy_v1"
ARM_SHA256 = "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"
RAW_SHA256 = "8729e049bae598a397933d69e8bf40f1e74534025c81c297effdfe04ec454ddf"
INHERITED_HOOK_SHA256 = "a60164c9c2538693ed89dc5855dad44944b85ba0ee83b60396f838b368aeb39e"
FROZEN_PINS = {
    ENTRY: "2d23d580102dab48718e94246e84fce9deb00dda8a2ed81cd3272548799dd63e",
    "scripts/rate_limited_probe_controller_hooks.py": "27691d998994be32b640d1155f01fa3152342f0c81441b4aa16b21450835616d",
    "scripts/develop_public_ycb_cartesian_probe_r12.py": RAW_SHA256,
    "scripts/actual_wrist_probe_controller_hooks.py": INHERITED_HOOK_SHA256,
}
RAW_PARAMETERS = {"path_seconds": 3., "settle_seconds": 3., "maximum_probe_seconds": 6.,
    "distance_m": .005, "actual_error_gain": 2., "reference_correction_limit_m": .002,
    "cartesian_ik_sample_seconds": .02}
GOVERNOR = {"filter_time_constant_s": .1, "maximum_target_speed_m_s": .004,
    "reference_interval_s": .02, "clock_tolerance_s": 1e-7,
    "filter_discretization": "exact_first_order_zero_order_hold",
    "initialization": "admission_actual_wrist_once"}
SAMPLE_GOVERNOR_FIELDS = ("raw_cartesian_ik_target_world_m", "filtered_cartesian_ik_target_world_m",
    "applied_cartesian_ik_target_world_m", "reference_endpoint_elapsed_s", "reference_dt_s",
    "applied_target_delta_world_m", "applied_target_speed_m_s")
SCOPE = (
    "Independent arithmetic recomputation of every recorded R25 IK raw R12 cubic/bounded-error "
    "reference, admission initialization, exact first-order tau=0.1s filter, vector 0.004m/s "
    "limiter, and 20ms endpoint/lookahead clocks. Every probe native sample must carry its most "
    "recent already-issued applied IK target; unchanged legacy extraction independently binds "
    "these exact samples and their cached wrist positions to trace/native records. Final source "
    "ZIP, supervisor exit, artifacts and native/strict receipts are bound. This does not independently "
    "recompute IK solutions, LP, Jacobians, QP, contact physics, instantaneous integrated-state "
    "velocity, or task success, and does not prove historical process identity. Reference speed "
    "limiting does not imply actual wrist speed compliance or task success. No simulation runs.")


def equal_number(value, expected, label, tolerance=2e-11):
    if abs(speed.number(value) - expected) > tolerance:
        raise ValueError(label + " differs")


def equal_vector(value, expected, label):
    residual = float(np.max(np.abs(speed.vector(value) - expected)))
    if residual > 2e-12:
        raise ValueError(label + " differs")
    return residual


def source_contract(run):
    protocol = json.loads(Path(run["inputs"]["protocol"]["path"]).read_bytes())
    result = json.loads(Path(run["inputs"]["result"]["path"]).read_bytes())
    manifest, row = protocol["source_hashes_before"], run["row"]
    declaration = protocol["r25_rate_limited_probe_experiment"]
    count = row["effective_hand_instances"]
    if (len(manifest) != 207 or any(manifest.get(name) != digest for name, digest in FROZEN_PINS.items())
            or protocol["probe_reference_governor"] != GOVERNOR
            or declaration["governor_parameters"] != GOVERNOR
            or row["derived_controller_sha256"] != ARM_SHA256
            or type(count) is not int or not 1 <= count <= 2 or count != len(row["attempts"])
            or row["effective_hand_controller_types"] != [CLASS] * count
            or row["effective_hand_budgets_rad"] != [.1] * count):
        raise ValueError("Frozen R25 source, hand or declared governor binding differs")
    expected = {"format": "r25_rate_limited_probe_global_binding_v1",
        "inherited_source_sha256": ARM_SHA256, "readable_source_sha256": ARM_SHA256,
        "readable_source_unchanged": True, "compiled_arm_code_unchanged": True,
        "replacement_global": "_r23_probe_telemetry", "inherited_telemetry_class": "ProbeTelemetry",
        "replacement_telemetry_class": "RateLimitedProbeTelemetry",
        "replacement_tracker_class": "RateLimitedProbeTracker",
        "inherited_probe_hook_sha256": INHERITED_HOOK_SHA256,
        "raw_reference_source_sha256": RAW_SHA256, "raw_reference_parameters": RAW_PARAMETERS,
        "reference_governor": GOVERNOR, "original_decisive_confirmation_called": True,
        "all_original_native_safety_checks_retained": True,
        "runtime_target_state_used": False, "target_qpos_writes_added": False}
    for value in (row, result):
        if (value["rate_limited_probe_source_transform"] != expected
                or value["inherited_r23_probe_audit_does_not_audit_r25_governor"] is not True):
            raise ValueError("Recorded R25 replacement binding differs")
    return {"source_count": len(manifest), "frozen_source_pins": FROZEN_PINS,
        "actual_hand_instances": count,
        "derived_arm_sha256": ARM_SHA256, "recorded_transform": expected,
        "historical_runtime_identity_independently_proven": False}


def check_governor(telemetry):
    """Recompute from observations and constants, without runtime reference helpers."""
    if (telemetry["format"] != TELEMETRY_FORMAT
            or telemetry["inherited_format"] != "actual_wrist_probe_policy_v1"
            or telemetry["reference_governor"] != GOVERNOR
            or telemetry["parameters"] != RAW_PARAMETERS
            or telemetry["reference_source_sha256"] != RAW_SHA256
            or not telemetry["attempts"]):
        raise ValueError("Unsupported R25 reference telemetry contract")
    reports = []
    for ordinal, record in enumerate(telemetry["attempts"], 1):
        start_s = speed.number(record["start_s"])
        start = speed.vector(record["actual_admission_start_world_m"])
        filtered, applied, previous_actual = start.copy(), start.copy(), start.copy()
        previous_endpoint, previous_clock = 0., start_s
        rows, samples = record["ik_samples"], record["samples"]
        if not rows or not samples or record["outcome"] is None:
            raise ValueError("Incomplete governed probe")
        residual, limited = 0., 0
        rate_rows = []
        previous_raw = start.copy()
        for index, row in enumerate(rows):
            endpoint = speed.number(row["reference_endpoint_elapsed_s"])
            clock = speed.number(row["timestamp_s"])
            dt = endpoint - previous_endpoint
            equal_number(endpoint, (index + 1) * .02, "20ms endpoint", 1e-7)
            equal_number(dt, .02, "20ms interval", 1e-7)
            equal_number(clock, start_s + endpoint - .02, "20ms lookahead clock", 1e-7)
            if endpoint > 6. + 1e-7 or row["parameters"] != RAW_PARAMETERS or row["reference_governor_parameters"] != GOVERNOR:
                raise ValueError("Reference duration or row parameters differ")
            actual = speed.vector(row["actual_wrist_at_ik_sample_world_m"])
            if index == 0:
                equal_vector(actual, start, "Admission actual-wrist initialization")
            fraction = min(1., max(0., endpoint / 3.))
            progress = 3. * fraction ** 2 - 2. * fraction ** 3
            desired = start + [0., 0., .005 * progress]
            error = desired - actual
            correction = 2. * error
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > .002:
                correction *= .002 / correction_norm
            raw = desired + correction
            alpha = float(-np.expm1(-dt / .1))
            next_filtered = filtered + alpha * (raw - filtered)
            requested = next_filtered - applied
            distance = float(np.linalg.norm(requested))
            is_limited = distance > .004 * dt
            delta = requested * (.004 * dt / distance) if is_limited else requested
            next_applied = applied + delta
            actual_dt = clock - previous_clock
            vectors = {"actual_admission_start_world_m": start,
                "desired_actual_path_world_m": desired, "actual_path_error_world_m": error,
                "bounded_reference_correction_world_m": correction,
                "raw_cartesian_ik_target_world_m": raw,
                "filtered_cartesian_ik_target_world_m": next_filtered,
                "applied_cartesian_ik_target_world_m": next_applied,
                "cartesian_ik_target_world_m": next_applied,
                "filtered_target_delta_world_m": next_filtered - filtered,
                "applied_target_delta_world_m": delta,
                "actual_wrist_delta_since_previous_ik_world_m": actual - previous_actual}
            for field, expected in vectors.items():
                residual = max(residual, equal_vector(row[field], expected, "IK " + field))
            for field, expected in (("path_elapsed_s", endpoint), ("actual_elapsed_s", clock - start_s),
                    ("cubic_progress", progress), ("reference_dt_s", dt), ("filter_alpha", alpha),
                    ("applied_target_speed_m_s", float(np.linalg.norm(delta)) / dt),
                    ("actual_wrist_sample_dt_s", actual_dt)):
                equal_number(row[field], expected, "IK " + field)
            if index == 0:
                equal_number(actual_dt, 0., "First IK admission clock", 1e-7)
                if row["actual_wrist_speed_since_previous_ik_m_s"] is not None:
                    raise ValueError("Admission zero-time speed must be null")
            else:
                if actual_dt <= 0.:
                    raise ValueError("IK observation clocks must increase")
                equal_number(row["actual_wrist_speed_since_previous_ik_m_s"],
                    float(np.linalg.norm(actual - previous_actual)) / actual_dt, "IK cached actual speed")
            if row["target_pose_used"] is not False or np.linalg.norm(delta) / dt > .004 + 2e-11:
                raise ValueError("Forbidden target-state use or reference speed excess")
            rate_rows.append({"timestamp_s": clock, "endpoint_elapsed_s": endpoint,
                "raw_difference_speed_m_s": float(np.linalg.norm(raw - previous_raw)) / dt,
                "filtered_difference_speed_m_s": float(np.linalg.norm(next_filtered - filtered)) / dt,
                "applied_difference_speed_m_s": float(np.linalg.norm(delta)) / dt,
                "limiter_active": bool(is_limited)})
            limited += int(is_limited)
            filtered, applied, previous_raw = next_filtered, next_applied, raw
            previous_endpoint, previous_clock, previous_actual = endpoint, clock, actual
        latest, ik_ticks = -1, [speed.evidence.tick(row["timestamp_s"], .002) for row in rows]
        for index, sample in enumerate(samples, 1):
            clock = speed.number(sample["timestamp_s"])
            tick = speed.evidence.tick(clock, .002)
            equal_number(clock, start_s + index * .002, "Native sample clock", 1e-7)
            while latest + 1 < len(rows) and ik_ticks[latest + 1] < tick:
                latest += 1
            if latest < 0:
                raise ValueError("Native sample has no already-issued IK reference")
            row = rows[latest]
            age = clock - row["timestamp_s"]
            equal_number(sample["elapsed_s"], clock - start_s, "Sample elapsed")
            equal_number(sample["latest_ik_timestamp_s"], row["timestamp_s"], "Latest IK clock")
            equal_number(sample["ik_sample_age_s"], age, "IK sample age")
            if (age <= 0. or age > .02 + 1e-7 or sample["parameters"] != RAW_PARAMETERS
                    or sample["runtime_target_state_used"] is not False
                    or sample["reference_governor"] != {field: row[field] for field in SAMPLE_GOVERNOR_FIELDS}):
                raise ValueError("Native sample stale or mismatched governed reference")
            equal_vector(sample["actual_admission_start_world_m"], start, "Sample admission")
            for field in ("desired_actual_path_world_m", "bounded_reference_correction_world_m", "cartesian_ik_target_world_m"):
                equal_vector(sample[field], speed.vector(row[field]), "Sample " + field)
        equal_number(record["outcome"]["timestamp_s"], samples[-1]["timestamp_s"], "Final sample clock")
        reports.append({"probe": ordinal, "ik_rows_recomputed": len(rows), "native_samples_bound": len(samples),
            "limiter_active_ik_rows": limited, "maximum_target_vector_residual_m": residual,
            "maximum_applied_reference_speed_m_s": max(row["applied_difference_speed_m_s"] for row in rate_rows),
            "reference_difference_rates": rate_rows})
    return {"all_recorded_reference_arithmetic_and_sample_bindings_consistent": True,
        "ik_rows_recomputed": sum(row["ik_rows_recomputed"] for row in reports),
        "native_samples_bound": sum(row["native_samples_bound"] for row in reports), "probes": reports}


def analysis_adapter(run):
    """Only adapt a copied top-level format; exact native sample dictionaries survive."""
    adapted = dict(run)
    adapted["row"] = dict(run["row"])
    telemetry = run["row"]["actual_wrist_probe"]
    if telemetry["format"] != TELEMETRY_FORMAT:
        raise ValueError("Unexpected analysis adapter input")
    adapted["row"]["actual_wrist_probe"] = dict(telemetry, format="actual_wrist_probe_policy_v1")
    return adapted


def governor_figures(run, audit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    for probe, report in zip(run["probes"], audit["probes"]):
        rows = probe["ik_samples"]
        times = [row["reference_endpoint_elapsed_s"] for row in rows]
        admission = speed.vector(probe["actual_admission_start_world_m"])
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for axis, k in zip(axes[:3], range(3)):
            for field, label, color in (("raw_cartesian_ik_target_world_m", "Raw R12", "#b34444"),
                    ("filtered_cartesian_ik_target_world_m", "Filtered", "#4d7c42"),
                    ("applied_cartesian_ik_target_world_m", "Applied IK", "#176ca4")):
                axis.plot(times, [(row[field][k] - admission[k]) * 1000. for row in rows],
                    label=label, color=color, marker=".")
            axis.set_ylabel("Reference d" + "xyz"[k] + " (mm)")
        for field, label, color in (("raw_difference_speed_m_s", "Raw difference", "#b34444"),
                ("filtered_difference_speed_m_s", "Filtered difference", "#4d7c42"),
                ("applied_difference_speed_m_s", "Applied difference", "#176ca4")):
            axes[3].plot(times, [row[field] * 1000. for row in report["reference_difference_rates"]],
                label=label, color=color, marker=".")
        axes[3].axhline(4., label="Applied norm limit 4mm/s", color="#333333", linestyle="--")
        axes[3].set_ylabel("Reference norm speed (mm/s)")
        axes[3].set_xlabel("20ms reference endpoint elapsed (s); wrist observation is 20ms earlier")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(loc="best", fontsize=8)
        fig.suptitle(f"R25 probe {probe['probe']} | independently recomputed reference governor\n"
            f"task={run['row']['success']} | held={run['row']['held_lift']} | {probe['outcome']['final_reason']}", fontsize=10)
        fig.text(.5, .012, "Reference differences use each 20ms endpoint interval, including the admission-initialized first update.\n"
            "Applied reference compliance does not imply actual wrist speed compliance or task success.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0., .04, 1., .95))
        for suffix in ("png", "pdf"):
            path = output / f"probe{probe['probe']}_reference_governor.{suffix}"
            fig.savefig(path, dpi=150)
            artifacts.append(speed.evidence.capture(path))
        plt.close(fig)
    return artifacts


def source_bindings():
    sources = speed.source_bindings()
    for relative in ("scripts/audit_plot_public_ycb_rate_limited_probe_r25.py",
            "tests/test_audit_plot_public_ycb_rate_limited_probe_r25.py"):
        sources[relative] = speed.evidence.digest(PROJECT / relative)
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True, type=speed.evidence.checked_sha)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    sources = source_bindings()
    run = speed.read_run(args.directory, args.expected_summary_sha256, args.job, REVISION, CLASS, ENTRY)
    source = source_contract(run)
    audit = check_governor(run["row"]["actual_wrist_probe"])
    run = speed.extract(analysis_adapter(run))
    speed.evidence.unchanged(run["inputs"])
    args.output.mkdir(parents=True, exist_ok=False)
    plots = speed.figures(run, args.output) + governor_figures(run, audit, args.output)
    reduced = args.output / "recorded_probes_and_reference_audit.json.gz"
    with gzip.open(reduced, "wt", encoding="utf-8") as stream:
        json.dump({"probes": run["probes"], "reference_audit": audit}, stream, separators=(",", ":"), allow_nan=False)
    for probe in run["probes"]:
        del probe["series"], probe["ik_samples"], probe["guard_observations"]
    for probe in audit["probes"]:
        del probe["reference_difference_rates"]
    row = run.pop("row")
    run["case"] = {key: row[key] for key in ("path", "kind", "seed", "success", "held_lift", "failure", "attempts")}
    speed.evidence.unchanged(run["inputs"])
    if source_bindings() != sources:
        raise ValueError("Analysis source changed during verification")
    receipt = {"format": FORMAT, "scope": SCOPE, "summary_sha256": args.expected_summary_sha256,
        "accepted_reference_governor_evidence": True,
        "checks": {"final_frozen_sources_and_recorded_governor_binding_consistent": True,
            "all_recorded_reference_arithmetic_and_sample_bindings_consistent": True,
            "all_recorded_probe_samples_and_ik_observations_match_native_trace": True},
        "independent_postrun_analysis": True, "part_of_runtime_frozen_manifest": False,
        "inputs_unchanged_during_analysis": True, "analysis_source_sha256": sources,
        "source": source, "reference_audit": audit, "run": run,
        "analysis_only_adapter": {"input_format": TELEMETRY_FORMAT,
            "adapted_format": "actual_wrist_probe_policy_v1", "input_files_changed": False,
            "changed_fields": ["copied_row.actual_wrist_probe.format"],
            "sample_dictionaries_preserved_for_exact_native_trace_matching": True},
        "figures": plots, "reduced_recording": speed.evidence.capture(reduced)}
    output = args.output / "diagnostics.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"diagnostics": str(output), "sha256": speed.evidence.digest(output), "figures": len(plots),
        "reference_ik_rows": audit["ik_rows_recomputed"], "task_success": run["case"]["success"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
