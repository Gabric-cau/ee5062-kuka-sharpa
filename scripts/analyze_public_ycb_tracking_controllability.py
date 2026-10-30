#!/usr/bin/env python3
"""Offline recorded R19 group Jacobian/command diagnostic, without simulation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from compare_public_ycb_close_deadline import digest, json_array, safe_path

PROJECT = Path(__file__).resolve().parents[1]
TIMES = (31., 34., 35.4)
STEP = .002
FINGERS = ("thumb", "middle", "ring")
SUMMARY_SHA = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
GAP_RECEIPT_SHA = "29bd410bf3c3a448de90d8f77d74c1c6ebb7a71a4012965c12ff226fe1be0d44"
OBSERVATIONS_SHA = "abf67b03b1d3bae7d22fdf8bdcbc3f8d6e7ed64d165cfccd49dfe1ef35f3994e"
GAP_DIRECTORY = PROJECT / "results/r19_pressure_tracking_gap_20261022"
EXTRACTED_EVENTS_SHA = "627f099c4b400aa23f03efd055f7d76be14f22015f9019db9e6e5313e0459372"
EXTRACTION_RECEIPT_SHA = "c72fd63643e4583c2bb344971ad0a4013462b10d518364a2318e8a36b6660c7d"


def binding(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def tick(time):
    value = float(time)
    index = round(value / STEP)
    if not np.isfinite(value) or abs(value - index * STEP) > 1e-8:
        raise ValueError("Recorded observation is not on the 2 ms grid")
    return index


def group_key(group):
    return f"{group['finger']}:{group['hand_body_name']}:" + ",".join(map(str, sorted(group["raw_input_indices"])))


def active_bounds(joint):
    applied = float(joint["applied_command_rad"])
    result = {"actual_rad": joint["before_actual_rad"], "reference_before_rad": joint["previous_reference_rad"],
        "applied_reference_rad": applied, "fixed_capture_bias_rad": joint["fixed_capture_bias_rad"],
        "added_bias_before_rad": joint["previous_reference_rad"] - joint["before_actual_rad"] - joint["fixed_capture_bias_rad"],
        "added_bias_after_write_rad": applied - joint["before_actual_rad"] - joint["fixed_capture_bias_rad"],
        "budget_rad": joint["additional_bias_limit_rad"], "box_feasible": joint["feasible"],
        "intervals_rad": {}, "active_faces_tolerance_rad": 1e-9, "active_faces": []}
    for label, key in (("range", "range_rad"), ("rate", "rate_interval_rad"),
                       ("bias", "bias_interval_rad"), ("intersection", "intersection_rad")):
        low, high = map(float, joint[key])
        result["intervals_rad"][label] = [low, high]
        for face, distance in (("lower", applied - low), ("upper", high - applied)):
            if abs(distance) <= 1e-9:
                result["active_faces"].append(label + ":" + face)
    return result


def finger_analysis(solve, recovery, joint_event, previous_event, observation):
    names = solve["joint_names"]
    groups = solve["groups"]
    matrix = np.asarray(solve["compression_jacobian_m_rad"], dtype=float)
    desired = np.asarray(solve["desired_normal_velocity_m_s"], dtype=float)
    error = np.asarray([g["target_normal_n"] - g["measured_normal_n"] for g in groups])
    dt = float(joint_event["dt_s"])
    if matrix.shape != (len(groups), len(names)) or dt <= 0. or not np.isfinite(matrix).all():
        raise ValueError("Invalid recorded current Jacobian/write interval")
    if not np.allclose(error, solve["pressure_errors_n"], atol=1e-12, rtol=0.):
        raise ValueError("Recorded group errors disagree")
    if not np.allclose(np.clip(error * .0005, -.001, .001), desired, atol=1e-14, rtol=0.):
        raise ValueError("Desired velocity differs from the frozen pressure-to-motion heuristic")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not np.allclose(singular, solve["singular_values_m_rad"], atol=1e-12, rtol=0.):
        raise ValueError("Recorded and independently recomputed singular values disagree")
    joint_velocity = np.array([(recovery["joints"][name]["applied_command_rad"]
        - recovery["joints"][name]["previous_reference_rad"]) / dt for name in names])
    applied_prediction = matrix @ joint_velocity
    if (recovery["predicted_applied_normal_velocity_m_s"] is not None
            and not np.allclose(applied_prediction, recovery["predicted_applied_normal_velocity_m_s"], atol=1e-12, rtol=0.)):
        raise ValueError("Recorded and reconstructed applied command prediction disagree")
    unbounded = np.asarray(solve["unlimited_joint_velocity_rad_s"], dtype=float)
    linear_solution = np.linalg.lstsq(matrix, desired, rcond=None)[0]
    ordered_groups = []
    for group in groups:
        key = group_key(group)
        gate = observation["current_groups"].get(key)
        ordered_groups.append({"key": key, "raw_patch_indices": group["raw_input_indices"],
            "observed_patch_count": len(group["raw_input_indices"]),
            "target_n": group["target_normal_n"], "measured_n": group["measured_normal_n"],
            "target_minus_measured_n": group["target_normal_n"] - group["measured_normal_n"],
            "error_sign": int(np.sign(group["target_normal_n"] - group["measured_normal_n"])),
            "gate_group_with_same_key": gate,
            "gate_minus_tracking_target_n": None if gate is None else gate["target_n"] - group["target_normal_n"],
            "gate_minus_tracking_measured_n": None if gate is None else gate["measured_n"] - group["measured_normal_n"]})
    encoder_proxy = None
    if previous_event is not None and tick(joint_event["timestamp_s"]) - tick(previous_event["timestamp_s"]) == 1:
        before = previous_event.get("joint_reference_steps", {})
        if all(name in before and "before_actual_rad" in before[name] for name in names):
            encoder_velocity = np.array([(recovery["joints"][name]["before_actual_rad"]
                - before[name]["before_actual_rad"]) / STEP for name in names])
            encoder_proxy = {"interval_start_s": previous_event["timestamp_s"],
                "interval_end_s": joint_event["timestamp_s"], "actual_joint_angle_difference_velocity_rad_s": encoder_velocity.tolist(),
                "current_J_times_encoder_difference_m_s": (matrix @ encoder_velocity).tolist(),
                "scope": "Backward finite-difference encoder proxy under the current J; not measured contact compression or pressure derivative, and excludes object motion"}
    qp = recovery.get("nominal_qp")
    return {"finger": solve["finger"], "fresh_jacobian_available": True,
        "group_count": len(groups), "joint_count": len(names),
        "groups": ordered_groups, "joint_names": names, "compression_jacobian_m_rad": matrix.tolist(),
        "recomputed_rank": int(np.linalg.matrix_rank(matrix)),
        "rank_tolerance": float(singular.max() * max(matrix.shape) * np.finfo(float).eps) if len(singular) else 0.,
        "singular_values_m_rad": singular.tolist(),
        "unbounded_linear_least_squares_desired_residual_m_s": (matrix @ linear_solution - desired).tolist(),
        "target_minus_measured_group_error_n": error.tolist(), "desired_normal_velocity_m_s": desired.tolist(),
        "recorded_r13_unbounded_prediction_before_direction_hold_m_s": solve["predicted_unbounded_normal_velocity_m_s"],
        "recorded_r13_joint_velocity_after_direction_hold_rad_s": unbounded.tolist(),
        "J_times_r13_velocity_after_direction_hold_m_s": (matrix @ unbounded).tolist(),
        "r13_direction_hold": solve["incompatible_direction_hold"],
        "per_finger_write_mode": recovery["mode"], "per_finger_write_reason": recovery["reason"],
        "nominal_qp": qp, "applied_reference_joint_velocity_rad_s": joint_velocity.tolist(),
        "predicted_applied_command_normal_velocity_m_s": applied_prediction.tolist(),
        "desired_minus_predicted_applied_command_velocity_m_s": (desired - applied_prediction).tolist(),
        "applied_reference_joints": {name: active_bounds(recovery["joints"][name]) for name in names},
        "encoder_difference_proxy": encoder_proxy,
        "scope": "Instantaneous group kinematics/recorded commands, not independently actuated per-patch forces or dynamic controllability"}


def invalid_finger_analysis(finger, recovery, event, observation):
    names = recovery["joint_names"]
    dt = float(event["dt_s"])
    patches = [patch for patch in event["current_target"].get("patches", []) if patch["finger"] == finger]
    gate = {key: value for key, value in observation["current_groups"].items() if key.startswith(finger + ":")}
    return {"finger": finger, "fresh_jacobian_available": False,
        "group_count": None, "joint_count": len(names), "joint_names": names,
        "group_count_scope": "No fresh tracking groups or J were constructed after the invalid-target gate; gate-only groups below are a separate observation/path",
        "compression_jacobian_m_rad": None, "recomputed_rank": None, "singular_values_m_rad": None,
        "target_minus_measured_group_error_n": None, "desired_normal_velocity_m_s": None,
        "predicted_applied_command_normal_velocity_m_s": None,
        "per_finger_write_mode": recovery["mode"], "per_finger_write_reason": recovery["reason"],
        "nominal_qp": recovery["nominal_qp"],
        "applied_reference_joint_velocity_rad_s": [(recovery["joints"][name]["applied_command_rad"]
            - recovery["joints"][name]["previous_reference_rad"]) / dt for name in names],
        "applied_reference_joints": {name: active_bounds(recovery["joints"][name]) for name in names},
        "gate_only_group_count_for_context": len(gate), "gate_only_groups_for_context": gate,
        "unadmitted_target_patch_count": len(patches),
        "unadmitted_per_patch_error_n_for_context": [patch["target_normal_n"] - patch["measured_pressure_n"] for patch in patches],
        "scope": "Invalid tracking target, not a failed-rank measurement. No stale J, group error or desired normal velocity is fabricated; per-patch candidate errors are not group-control errors"}


def analyze(directory, output, selected_events=None):
    directory, output = directory.resolve(strict=True), output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new offline diagnostic directory")
    summary_path = safe_path(directory, "summary.json")
    gap_path, observations_path = GAP_DIRECTORY / "receipt.json", GAP_DIRECTORY / "selected_observations.json"
    if (digest(summary_path) != SUMMARY_SHA or digest(gap_path) != GAP_RECEIPT_SHA
            or digest(observations_path) != OBSERVATIONS_SHA):
        raise ValueError("Pinned baseline/selected-observation hashes differ")
    summary, gap = json.loads(summary_path.read_text()), json.loads(gap_path.read_text())
    observations = json.loads(observations_path.read_text())
    protocol, case_row = summary["protocol"], summary["rows"][0]
    if (not summary["complete"] or summary["runner_errors"] or not gap["inputs_unchanged"]
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("Baseline/source is not finalized")
    helper = PROJECT / "scripts/compare_public_ycb_close_deadline.py"
    if digest(helper) != protocol["source_hashes_before"][helper.relative_to(PROJECT).as_posix()]:
        raise ValueError("Imported parser differs from frozen source")
    case = directory / case_row["path"]
    paths = {}
    for name in ("joint_admittance", "bias_recovery"):
        record = case_row["artifacts"][name]
        path = safe_path(case, record["path"])
        if binding(path) != {"bytes": record["bytes"], "sha256": record["sha256"]}:
            raise ValueError("Recorded stream hash differs: " + name)
        paths[name] = path
    inputs = {str(path): binding(path) for path in
        (summary_path, gap_path, observations_path, helper, Path(__file__), *paths.values())}
    wanted = {tick(time): time for time in TIMES}
    selected, previous = {}, None
    recoveries = {}
    if selected_events is None:
        for event in json_array(paths["joint_admittance"]):
            index = tick(event["timestamp_s"])
            if index > max(wanted) + 1:
                break
            if event.get("event") != "NOMINAL_REFERENCE_WRITE":
                continue
            if event["phase"] == "CLOSE":
                for target in wanted:
                    if index in (target - 1, target):
                        selected[target] = (event, previous)
            previous = event
        selected_ticks = {tick(event["timestamp_s"]) for event, _ in selected.values()}
        for event in json_array(paths["bias_recovery"]):
            index = tick(event["timestamp_s"])
            if index > max(selected_ticks):
                break
            if index in selected_ticks:
                if index in recoveries:
                    raise ValueError("Ambiguous recovery event at selected write")
                recoveries[index] = event
    else:
        selected_events = selected_events.resolve(strict=True)
        prior_receipt = selected_events.parent / "receipt.json"
        if digest(selected_events) != EXTRACTED_EVENTS_SHA or digest(prior_receipt) != EXTRACTION_RECEIPT_SHA:
            raise ValueError("Only the exact previously extracted selection is accepted for reuse")
        prior = json.loads(prior_receipt.read_text())
        if not prior["inputs_unchanged"] or any(prior["inputs_before"][str(path)] != inputs[str(path)] for path in paths.values()):
            raise ValueError("Reused selection is not bound to these original streams")
        for record in json.loads(selected_events.read_text()):
            event = record["joint_event"]
            selected[tick(record["requested_time_s"])] = (event, record["previous_joint_write"])
            recoveries[tick(event["timestamp_s"])] = record["bias_recovery_event"]
        inputs[str(selected_events)] = binding(selected_events)
        inputs[str(prior_receipt)] = binding(prior_receipt)
        selected_ticks = {tick(event["timestamp_s"]) for event, _ in selected.values()}
    if set(selected) != set(wanted):
        raise ValueError("No exact or preceding-2ms CLOSE write for a prescribed point")
    if set(recoveries) != selected_ticks:
        raise ValueError("Selected write lacks same-time per-finger recovery evidence")
    results, raw = [], []
    for requested, observation in zip(TIMES, observations):
        event, previous_event = selected[tick(requested)]
        recovery = recoveries[tick(event["timestamp_s"])]
        if tick(observation["timestamp_s"]) != tick(requested):
            raise ValueError("Existing selected observation order differs")
        if tick(recovery["post_step_timestamp_s"]) != tick(event["timestamp_s"]) + 1:
            raise ValueError("Recorded command/post-step timeline differs")
        by_finger = {finger["finger"]: finger for finger in recovery["per_finger"]}
        solves = event["joint_solves"]
        diagnostics = {solve["finger"]: finger_analysis(solve, by_finger[solve["finger"]], event,
            previous_event, observation) for solve in solves}
        for finger in FINGERS:
            if finger not in diagnostics:
                diagnostics[finger] = invalid_finger_analysis(finger, by_finger[finger], event, observation)
        result = {"requested_post_step_preload_time_s": requested,
            "selected_write_time_s": event["timestamp_s"],
            "write_selection": "EXACT_CLOCK_CLOSE_WRITE" if tick(event["timestamp_s"]) == tick(requested) else "PRECEDING_2MS_CLOSE_WRITE",
            "write_post_step_observation_time_s": recovery["post_step_timestamp_s"],
            "timeline_scope": "Regulation/Jacobian and reference write occur before mj_step at selected_write_time. Preload at the same clock is the previous integration observation; a same-clock write predicts the next 2ms step, not an independently simultaneous gate target. A preceding write generated the requested observation.",
            "global_mode": event["mode"], "fresh_target": recovery["fresh_target"],
            "target_reason": recovery["target_reason"], "target_age_s": event["target_age_s"],
            "recovery_active": recovery["recovery_active"], "final_safety_override": recovery["final_safety_override"],
            "finger_diagnostics": [diagnostics[finger] for finger in FINGERS],
            "missing_fresh_finger_solves": [finger for finger in FINGERS if finger not in {s["finger"] for s in solves}]}
        results.append(result)
        raw.append({"requested_time_s": requested, "joint_event": event,
            "previous_joint_write": previous_event, "bias_recovery_event": recovery})
    if any(binding(Path(path)) != bound for path, bound in inputs.items()):
        raise ValueError("Analysis input/source changed")
    receipt = {"format": "offline_r19_tracking_controllability_v1", "inputs_before": inputs,
        "inputs_after": copy.deepcopy(inputs), "inputs_unchanged": True, "numpy_version": np.__version__,
        "original_source_hashes_before": protocol["source_hashes_before"],
        "original_source_hashes_after": protocol["source_hashes_after"],
        "stream_decompression_passes": {name: int(selected_events is None) for name in paths},
        "prior_extraction_stream_decompression_passes": None if selected_events is None else {name: 1 for name in paths},
        "samples": results,
        "physics_rerun": False, "controller_changed": False, "native_stream_loaded": False,
        "scope": "Recorded three-state group Jacobians and applied reference commands only. Predicted command normal velocity is not measured contact velocity, pressure response, actuator feasibility over time or a dynamic causal explanation. Gate and tracking outputs retain their own timestamps/target histories and group identities."}
    output.mkdir(parents=True)
    for name, value in (("receipt.json", receipt), ("selected_joint_events.json", raw)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(output), "selected_samples": len(results),
        "write_times_s": [r["selected_write_time_s"] for r in results],
        "finger_diagnostic_counts": [len(r["finger_diagnostics"]) for r in results]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-events", type=Path,
        help="Reuse the exact hash-bound initial three-state extraction, without another gzip pass")
    args = parser.parse_args()
    analyze(args.directory, args.output, args.selected_events)
