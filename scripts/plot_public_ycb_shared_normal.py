#!/usr/bin/env python3
"""Post-run R19/R20 allocator plots, bound to actual formats and audit receipts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import plot_public_ycb_extended_budget as previous


common, metrics = previous.common, previous.metrics
digest, safe_path = common.digest, common.safe_path
PROJECT = Path(__file__).resolve().parents[1]
R19_SUMMARY_SHA256 = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
FORMATS = {"baseline": "public_ycb_extended_budget_development_r19",
    "candidate": "public_ycb_shared_normal_development_r20"}
CLASSES = {"baseline": "BiasBudgetHand", "candidate": "SharedNormalBudgetHand"}
COMMAND_FORMATS = {"baseline": "r19_extended_budget_command_and_native_actuation_audit_v1",
    "candidate": "r20_shared_normal_postrun_command_native_and_static_target_audit_v1"}
COMMAND_AUDITORS = {"baseline": "scripts/audit_public_ycb_extended_budget.py",
    "candidate": "scripts/audit_public_ycb_shared_normal.py"}
POSTRUN_AUDITOR = "scripts/audit_public_ycb_shared_normal_postrun.py"
ORIGINAL_FAILURE_SHA256 = "aa456ddbcd40accab48be87b650d2c4deaa71917d6838991b7238b9c8c07f740"
COMMAND_PATHS = {"baseline": PROJECT / "results/r19_extended_budget_audit_20261022/receipt.json",
    "candidate": PROJECT / "results/r20_shared_normal_postrun_audit_20261022/receipt.json"}
R19_RECEIPT_SHA256 = {
    "native_verification": "2b6261e9978c27dffd824617d5df4ffa1edca9392b6da9fee9eed179a4ab9863",
    "strict_verification": "afac02d33fc31e4d38168c8a07099049ef961e91028f9d83745ad40fe6c4b1a3",
    "command_verification": "bc440c0e45f2aa9f612263cbe243af3c74e6213d06668cdb14db30b27c8c3822",
}
BUDGET_RAD, CLOSE_SECONDS, ARM_SECONDS = .10, 32., 16.


def source_bindings():
    paths = (Path(__file__).resolve(), Path(previous.__file__).resolve(), Path(common.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_bias_budget.py",
        PROJECT / "scripts/compare_public_ycb_close_deadline.py",
        PROJECT / "scripts/audit_public_ycb_bias_recovery.py", PROJECT / POSTRUN_AUDITOR)
    return {str(path.relative_to(PROJECT)): digest(path) for path in paths}


def audit_receipts(directory, role, summary_sha, protocol_sha, row, protocol, command_path):
    inputs, receipts = {}, {}
    for name, path in (("native_verification", safe_path(directory, "verification_receipt.json")),
            ("strict_verification", safe_path(directory, "strict_native_verification.json")),
            ("command_verification", command_path.resolve(strict=True))):
        receipts[name], inputs[name] = common.captured_json(path)
        if receipts[name]["summary_sha256"] != summary_sha:
            raise ValueError("Independent receipt does not bind this finalized summary: " + name)
        if role == "baseline" and inputs[name]["sha256"] != R19_RECEIPT_SHA256[name]:
            raise ValueError("The exact frozen R19 baseline receipt differs: " + name)
    native, strict, command = (receipts[key] for key in
        ("native_verification", "strict_verification", "command_verification"))
    if (native["format"] != "public_ycb_development_native_verification_r1" or native["complete"] is not True
            or native["unknowns"] != 0 or native["source_hashes_before_equal_after"] is not True
            or native["native_samples_contiguous_2ms"] is not True or native["audited_episodes"] != 1
            or len(native["rows"]) != 1 or native["rows"][0]["path"] != row["path"]
            or native["rows"][0]["outcome"] != row["success"] or native["rows"][0]["held_lift"] != row["held_lift"]):
        raise ValueError("Native evidence receipt is incomplete or disagrees with the actual outcome")
    if (strict["format"] != "public_ycb_strict_native_audit_20261021" or strict["accepted_evidence"] is not True
            or strict["unknowns"] != 0 or strict["denominator"] != 1 or len(strict["rows"]) != 1
            or strict["rows"][0]["path"] != row["path"]
            or strict["rows"][0]["checks"]["legacy_task_success"] != row["success"]
            or strict["rows"][0]["checks"]["held_lift"] != row["held_lift"]):
        raise ValueError("Strict evidence receipt is incomplete or disagrees with the actual outcome")
    expected_auditor = protocol["source_hashes_before"][COMMAND_AUDITORS[role]]
    if role == "candidate":
        extension_path = PROJECT / POSTRUN_AUDITOR
        expected_auditor = digest(extension_path)
        failed_path = PROJECT / "results/r20_shared_normal_audit_20261022/receipt.json"
        failed, failed_bound = common.captured_json(failed_path)
        classification = command["initializer_classification"]
        if (command["inherited_frozen_auditor_sha256"] != protocol["source_hashes_before"][COMMAND_AUDITORS[role]]
                or command["inherited_audit_receipt_format"] != "r20_shared_normal_command_native_and_static_target_audit_v1"
                or command["original_failed_receipt_sha256"] != ORIGINAL_FAILURE_SHA256
                or command["original_failed_receipt"]["sha256"] != ORIGINAL_FAILURE_SHA256
                or command["original_failed_receipt"]["bytes"] != failed_bound["bytes"]
                or failed_bound["sha256"] != ORIGINAL_FAILURE_SHA256
                or failed != {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
                    "error": "Unversioned target is not an original caller-gate rejection",
                    "script_sha256": protocol["source_hashes_before"][COMMAND_AUDITORS[role]]}
                or classification["all_contexts_valid"] is not True
                or classification["initialization_records"] != 3950 or classification["controller_epochs"] != 2
                or classification["original_stream_ordinals_preserved"] is not True
                or classification != command["cases"][0]["static_targets"]["initializer_classification"]
                or command["extension_inputs_unchanged"] is not True
                or command["extension_inputs_before"] != command["extension_inputs_after"]
                or command["postrun_extension"] is not True
                or command["original_failed_audit_overwritten"] is not False
                or command["original_runtime_source_modified"] is not False
                or command["original_frozen_auditor_succeeded"] is not False
                or command["target_audit_transform"]["input_records_filtered_or_reordered"] is not False
                or command["target_audit_transform"]["branch_insertions"] != 1
                or command["target_audit_transform"]["original_globals_modified"] is not False):
            raise ValueError("R20 postrun initializer extension or preserved original failure differs")
        inputs["original_failed_command_verification"] = failed_bound
        inputs["postrun_command_auditor"] = {"path": str(extension_path),
            "bytes": extension_path.stat().st_size, "sha256": expected_auditor}
    if (command["format"] != COMMAND_FORMATS[role] or command["protocol_sha256"] != protocol_sha
            or command["script_sha256"] != expected_auditor
            or any(command[key] is not True for key in ("all_evidence_consistent",
                "all_declared_command_bounds_satisfied", "all_native_close_spans_within_declared_deadline"))
            or command["additional_bias_limit_rad"] != BUDGET_RAD or command["close_seconds"] != CLOSE_SECONDS
            or command["arm_interpolation_seconds"] != ARM_SECONDS or len(command["cases"]) != 1
            or command["cases"][0]["case"] != row["path"]
            or command["cases"][0]["task_success"] != row["success"]
            or command["cases"][0]["held_lift"] != row["held_lift"]
            or command["cases"][0]["actuation"]["all_cached_force_and_transmitted_torque_consistent"] is not True):
        raise ValueError("Command/cache evidence receipt is incomplete, altered or not case/source-bound")
    if role == "candidate" and any(command[key] is not True for key in
            ("all_recorded_static_target_checks_consistent", "all_sampled_hypothetical_target_checks_passed")):
        raise ValueError("R20 static-target evidence receipt is not consistent")
    return inputs, {"native_evidence_complete": True, "strict_evidence_accepted": True,
        "command_and_cache_evidence_consistent": True, "reported_task_success": row["success"],
        "reported_held_lift": row["held_lift"], "strict_task_success": strict["rows"][0]["strict_success"],
        "scope": "Receipt consistency is required; task failure is allowed. A process-normal-exit claim is not established by this plotter."}


def plot_run(directory, role, output, command_path):
    directory = directory.resolve(strict=True)
    summary, summary_bound = common.captured_json(safe_path(directory, "summary.json"))
    protocol, protocol_bound = common.captured_json(safe_path(directory, "protocol.json"))
    if (summary["complete"] is not True or summary["protocol"] != protocol or protocol["format"] != FORMATS[role]
            or summary["runner_revision"] != FORMATS[role] or protocol["runner_revision"] != FORMATS[role]
            or len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_SECONDS
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_SECONDS
            or summary["additional_bias_limit_rad"] != BUDGET_RAD or summary["close_seconds"] != CLOSE_SECONDS
            or summary["arm_interpolation_seconds"] != ARM_SECONDS):
        raise ValueError("Expected one finalized actual-format .10rad/CLOSE32/arm16 case")
    if role == "baseline":
        if summary_bound["sha256"] != R19_SUMMARY_SHA256:
            raise ValueError("Plot baseline must be the exact frozen R19 result")
    else:
        declaration = protocol["r20_shared_normal_experiment"]
        if (declaration["r19_comparison_summary_sha256"] != R19_SUMMARY_SHA256
                or declaration["additional_bias_limit_rad"] != BUDGET_RAD
                or declaration["close_seconds"] != CLOSE_SECONDS or declaration["arm_interpolation_seconds"] != ARM_SECONDS
                or declaration["native_timestep_s"] != metrics.STEP
                or declaration["changed_parameters"] != {"tracking_and_gate_allocator": {
                    "before": "r13_single_1_5g_then_fixed_1g_postcheck", "after": "shared_normal_tracking_target_v1"}}
                or protocol["controller_class"] != CLASSES[role] or protocol["target_allocator_revision"] != FORMATS[role]):
            raise ValueError("R20 plot input has a different baseline, allocator or inherited settings")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    metadata, metadata_bound = common.captured_json(safe_path(directory, Path(row["path"]) / "metadata.json"))
    if (metadata != row or row["runner_error"] or any(row[key] != value for key, value in planned.items())
            or row.get("missing_runtime_controller_binding", False) is not False
            or row["additional_bias_limit_rad"] != BUDGET_RAD or row["close_seconds"] != CLOSE_SECONDS
            or row["arm_interpolation_seconds"] != ARM_SECONDS
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("Actual planned case, hand budget or clock binding differs")
    count = row["effective_hand_instances"]
    if (type(count) is not int or not 1 <= count <= protocol["maximum_attempts"] or count != len(row["attempts"])
            or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * count
            or row["effective_hand_controller_types"] != [CLASSES[role]] * count):
        raise ValueError("Live controller classes or immutable per-retry budgets differ")
    if role == "candidate" and (row.get("missing_runtime_allocator_binding", False) is not False
            or row["controller_class"] != CLASSES[role] or row["target_allocator_revision"] != FORMATS[role]
            or len(row["shared_normal_runtime_bindings"]) != count):
        raise ValueError("R20 live two-path allocator evidence is missing")
    case = (directory / row["path"]).resolve(strict=True)
    if not case.is_dir() or not case.is_relative_to(directory):
        raise ValueError("Missing or escaping evidence case directory")
    inputs = {"summary": summary_bound, "protocol": protocol_bound, "metadata": metadata_bound}
    for key, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py")):
        path = safe_path(case, filename)
        inputs[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inputs["scene"]["sha256"]
            or row["scene_sha256_at_finalize"] != inputs["scene"]["sha256"]
            or row["derived_controller_sha256"] != inputs["controller"]["sha256"]
            or float(ET.fromstring(Path(inputs["scene"]["path"]).read_bytes()).find("option").attrib["timestep"]) != metrics.STEP):
        raise ValueError("Runtime-loaded scene, timestep or executed source differs")
    receipt_inputs, audit_state = audit_receipts(directory, role, summary_bound["sha256"], protocol_bound["sha256"],
        row, protocol, command_path)
    inputs.update(receipt_inputs)
    paths = {}
    for key in ("load_transfer_probe", "preload", "bias_recovery", "native"):
        paths[key], inputs[key] = common.bind_artifact(case, row["artifacts"][key])
    probe = common.reduced_rows(paths["load_transfer_probe"], "load_transfer_probe")
    windows = metrics.close_windows(probe, row["attempts"])
    if not windows or len(windows) != len(row["attempts"]):
        raise ValueError("Recorded CLOSE windows do not match actual attempt count")
    preload, bias = previous.reduced_preload(paths["preload"]), common.reduced_rows(paths["bias_recovery"], "bias_recovery")
    attempts, curves, guards = [], [], []
    for window in windows:
        probe_stats, probe_series = metrics.probe_metrics(probe, window)
        preload_stats, pressure_series = metrics.preload_metrics(preload, window)
        bias_stats, bias_series, guarded = metrics.bias_metrics(bias, window, BUDGET_RAD)
        context = previous.preload_context(preload, window)
        attempts.append({**window, "probe": probe_stats, "preload": preload_stats, "bias": bias_stats,
            "preload_context": context, "whole_attempt_probe": metrics.attempt_probe_metrics(probe, window),
            "gate_only_target_valid": context["target_valid"], "gate_only_target_reasons": context["target_reasons"],
            "target_statistics_scope": "Unique current post-integration gate-only LP records. Caller/solver rejection reasons are preserved; absent targets do not become zero error. These are observations, not independent LP trials."})
        curves.append({"probe": probe_series, "pressure": pressure_series, "bias": bias_series})
        guards.append(guarded)
    del bias
    gc.collect()
    torque_stats, torque_curves, native_count = common.stream_torque(paths["native"],
        row["native_audit"]["native_actuation_metadata"], guards)
    if native_count != row["native_audit"]["native_samples"] or native_count != row["native_audit"]["native_actuation_samples"]:
        raise ValueError("Actual native sample denominator differs")
    label = f"{role.upper()} {'R19 original target' if role == 'baseline' else 'R20 shared-normal target'} | .10rad | CLOSE32/arm16"
    figures = []
    for index, (attempt, series) in enumerate(zip(attempts, curves, strict=True)):
        attempt["direct_native_thumb_torque"] = torque_stats[index]
        series["torque"] = torque_curves[index]
        figures.extend(metrics.plot_attempt(output, role, label, attempt, BUDGET_RAD, series))
        figures.extend(previous.progress_plot(output, role, label, attempt, preload, probe))
    for bound in inputs.values():
        path = Path(bound["path"])
        if path.stat().st_size != bound["bytes"] or digest(path) != bound["sha256"]:
            raise ValueError("Plot input changed during analysis")
    return {"role": role, "directory": str(directory), "protocol_format": protocol["format"],
        "controller_class": CLASSES[role], "additional_bias_limit_rad": BUDGET_RAD,
        "close_seconds": CLOSE_SECONDS, "arm_interpolation_seconds": ARM_SECONDS,
        "case": {"kind": row["kind"], "seed": row["seed"], "path": row["path"],
            "reported_success": row["success"], "reported_held_lift": row["held_lift"], "reported_failure": row.get("failure")},
        "required_audit_receipts": audit_state, "inputs": inputs, "native_samples_streamed": native_count,
        "attempts": attempts, "figures": [{"path": name, "bytes": (output / name).stat().st_size,
            "sha256": digest(output / name)} for name in figures]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-command-receipt", type=Path, default=COMMAND_PATHS["baseline"])
    parser.add_argument("--candidate-command-receipt", type=Path, default=COMMAND_PATHS["candidate"])
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_bindings()
    baseline = plot_run(args.baseline, "baseline", args.output, args.baseline_command_receipt)
    candidate = plot_run(args.candidate, "candidate", args.output, args.candidate_command_receipt)
    if (any(baseline["case"][key] != candidate["case"][key] for key in ("kind", "seed"))
            or any(baseline["inputs"][key]["sha256"] != candidate["inputs"][key]["sha256"] for key in ("scene", "controller"))):
        raise ValueError("Comparison case identity, actual scene or executed arm source differs")
    if source_bindings() != sources:
        raise ValueError("Offline plotting source changed during analysis")
    result = {"format": "independent_r19_r20_shared_normal_plots_v1", "independent_postrun_analysis": True,
        "part_of_runtime_frozen_manifest": False, "analysis_source_sha256": sources, "runs": [baseline, candidate],
        "physical_prefix_independently_compared": False,
        "scope": "Actual-format, receipt-bound R19/R20 allocator comparison at unchanged .10rad/CLOSE32/arm16. Each observed attempt produces the existing pressure/bias/table/torque figure plus gate-progress figure in PNG/PDF; two attempts in both runs produce 16 files. Invalid targets are gaps and errors remain conditional on available groups. Gate-only target validity/reasons count unique observations, not independent solver experiments. Pressure-window age/span, recovery inhibition, original eligibility and continuous preload remain separate. Thumb torque is last-evaluation native cached actuation observed at t+dt for write t, not post-qpos-recomputed PD or contact torque. Attempt 2 is a history-dependent retry, not a matched independent initial condition. Native/strict/command receipt consistency is required, while failed task outcomes are retained. Static feasible targets, clean commands and plots do not prove actual measured equilibrium, grasp success, normal process exit, physical-prefix identity, causal attribution or randomized-dataset performance. Table/target ground truth remains evaluation-only. No missing ray solution or optimality proof is claimed."}
    destination = args.output / "comparison.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"comparison": str(destination), "sha256": digest(destination),
        "figures": sum(len(run["figures"]) for run in result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
