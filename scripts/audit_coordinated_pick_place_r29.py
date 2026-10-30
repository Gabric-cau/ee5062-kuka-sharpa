#!/usr/bin/env python3
"""Read-only R29 evidence audit; failed tasks can be valid negative evidence.

No frozen controller is imported or executed and no physics is stepped. Native
target truth is used only by the original independent strict task audit.
Recorded write counters remain runner assertions, not a replay proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from audit_coordinated_motion_experiment_r28 import (
    array, artifact, close, load_json, require, servo_binding, sha256, verify_source_archive,
)
from verify_public_ycb_strict import audit_case

DT = .002
ARRIVAL_TRANSITIONS = {("PRELOAD", "TRIAL_LIFT"), ("TRIAL_LIFT", "LIFT"),
    ("LIFT", "TRANSFER"), ("TRANSFER", "PLACE"), ("SUPPORT_SETTLE", "RELEASE"),
    ("RETRACT", "VERIFY")}


def json_number(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Unsupported JSON value: " + type(value).__name__)


def tick(timestamp):
    value = float(array(timestamp, ()))
    require(abs(value / DT - round(value / DT)) < 1e-6, "timestamp off native grid")
    return round(value / DT)


def check_command(model, row, native, bias):
    state, solved = row["before_robot_state"], row["solver"]
    names = state["joint_names"]
    joints, actuators = servo_binding(model, names)
    q, velocity = array(state["qpos_rad"], (len(names),)), array(state["qvel_rad_s"], (len(names),))
    previous = array(state["actual_ctrl"], (model.nu,))
    command, speed = array(solved["command_rad"], q.shape), array(solved["command_velocity_rad_s"], q.shape)
    opening = row["opening"]
    require(type(opening) is bool and solved["accepted"] is True and row["command_applied"] is True,
        "rejected solver command was applied")
    phase = native["state"]
    rates = np.array([(.15 if phase == "RELEASE" else .05) if name.startswith("left_") else .025 for name in names])
    require(np.all(abs(speed) <= rates + 1e-9), "applied command exceeds declared rate")
    close(command, previous[actuators] + DT * speed, "command/rate increment differs")
    lower, upper = model.jnt_range[joints].T.copy()
    for index, actuator in enumerate(actuators):
        if model.actuator_ctrllimited[actuator]:
            lower[index] = max(lower[index], model.actuator_ctrlrange[actuator, 0])
            upper[index] = min(upper[index], model.actuator_ctrlrange[actuator, 1])
    deflection_low, deflection_high = bias - .05, bias + .05
    if opening:
        hand = np.array([name.startswith("left_") for name in names])
        deflection_low[hand], deflection_high[hand] = -.25, .25
    torque_low, torque_high = [], []
    for joint, actuator in zip(joints, actuators, strict=True):
        limits = []
        if model.actuator_forcelimited[actuator]:
            limits.append(model.actuator_forcerange[actuator])
        if model.jnt_actfrclimited[joint]:
            limits.append(model.jnt_actfrcrange[joint])
        require(bool(limits), "missing actual actuator torque range")
        torque_low.append(max(value[0] for value in limits))
        torque_high.append(min(value[1] for value in limits))
    torque_low, torque_high = np.array(torque_low), np.array(torque_high)
    kp, kd = model.actuator_gainprm[actuators, 0], -model.actuator_biasprm[actuators, 2]
    predicted = kp * (command - q) - kd * velocity
    expected_lower = np.maximum.reduce([lower, previous[actuators] - DT * rates,
        q + deflection_low, q + (torque_low + kd * velocity) / kp])
    expected_upper = np.minimum.reduce([upper, previous[actuators] + DT * rates,
        q + deflection_high, q + (torque_high + kd * velocity) / kp])
    close(solved["bounds"]["command_lower_rad"], expected_lower, "lower command intersection differs", 2e-12)
    close(solved["bounds"]["command_upper_rad"], expected_upper, "upper command intersection differs", 2e-12)
    require(np.all(command >= expected_lower - 2e-12) and np.all(command <= expected_upper + 2e-12),
        "applied command outside independent reference box")
    require(np.all(predicted >= torque_low - 1e-9) and np.all(predicted <= torque_high + 1e-9),
        "applied frozen-state predicted torque outside motor range")
    close(solved["predicted_motor_torque_nm"], predicted, "motor prediction formula differs")
    actual = previous.copy()
    actual[actuators] = command
    close(row["actual_ctrl_for_step"], actual, "unexpected actuator writer after solver")
    close(native["motor_commands"], actual, "native actual command differs from applied candidate")
    cached = array(row["cached_motor_torque_nm"], q.shape)
    close(cached, predicted, "actual step-start motor torque differs from frozen prediction", 1e-8)
    residual = array(solved["contact_residual_m_s"])
    require(np.max(abs(residual)) <= .0001 + 1e-12, "accepted contact residual exceeds unchanged gate")
    return float(np.max(abs(cached - predicted)))


def audit_arrivals(transitions, rows, config):
    pose_rows = [row for row in rows if "motion_reference" in row]
    by_tick = {tick(row["timestamp_s"]): index for index, row in enumerate(pose_rows)}
    reports = []
    for previous, current in zip(transitions[:-1], transitions[1:], strict=True):
        pair = previous["state"], current["state"]
        at = tick(current["timestamp_s"])
        if pair in ARRIVAL_TRANSITIONS:
            require(at in by_tick, "arrival transition lacks current robot pose")
            index = by_tick[at]
            require(index > 0, "arrival transition lacks preceding trajectory")
            old_goal = array(pose_rows[index - 1]["actual_goal_world_m"], (3,))
            end_time = float(current["timestamp_s"])
            start_index = index
            for candidate in range(index, -1, -1):
                row = pose_rows[candidate]
                timestamp = float(row["timestamp_s"])
                if timestamp < float(previous["timestamp_s"]) - 1e-9:
                    break
                position = array(row["motion_reference"]["wrist_position_world_m"], (3,))
                if candidate:
                    before = pose_rows[candidate - 1]
                    elapsed = timestamp - float(before["timestamp_s"])
                    speed = np.linalg.norm(position - array(before["motion_reference"]["wrist_position_world_m"], (3,))) / max(DT, elapsed)
                else:
                    speed = 0.
                if np.linalg.norm(old_goal - position) >= config["arrival_position_tolerance_m"] or speed >= config["arrival_linear_speed_m_s"]:
                    break
                start_index = candidate
            continuous = end_time - float(pose_rows[start_index]["timestamp_s"])
            require(continuous + 1e-9 >= config["arrival_continuous_s"], "phase advanced before continuous actual arrival")
            if pair == ("PRELOAD", "TRIAL_LIFT"):
                require(end_time - previous["timestamp_s"] >= 2. - 1e-9, "preload skipped minimum duration")
            if pair == ("SUPPORT_SETTLE", "RELEASE"):
                require(pose_rows[index]["support_gate"]["confirmed"] is True, "release lacked fresh filtered support")
            reports.append({"from": pair[0], "to": pair[1], "continuous_arrival_seconds": continuous})
        elif pair[1] == "SUPPORT_SETTLE":
            require(at in by_tick and pose_rows[by_tick[at]]["support_gate"]["confirmed"] is True,
                "support settling began without confirmed support")
        elif pair[1] == "RECOVERY_LOWER":
            require(any(tick(row["timestamp_s"]) == at and "abort" in row for row in rows),
                "recovery transition lacks retained triggering failure")
    return reports


def audit(directory):
    summary = load_json(directory / "summary.json")
    result, protocol = load_json(directory / "result.json"), load_json(directory / "protocol.json")
    require(summary["complete"] is True and summary["protocol"] == protocol, "incomplete or divergent protocol")
    source_count = verify_source_archive(directory, protocol)
    require(protocol["input_hashes"] == protocol["input_hashes_after"] and not protocol["inputs_changed_during_run"],
        "inputs changed during run")
    for relative, expected in protocol["input_hashes"].items():
        path = (PROJECT / relative).resolve()
        require(path.is_relative_to(PROJECT) and sha256(path) == expected, "actual input hash differs: " + relative)
    require(sha256(directory / "compiled_model.mjb") == protocol["compiled_model_sha256"], "compiled model hash differs")
    model = mujoco.MjModel.from_binary_path(str(directory / "compiled_model.mjb"))
    require(abs(float(model.opt.timestep) - DT) < 1e-12, "native timestep changed")
    require(len(summary["rows"]) == 1, "expected one finite R29 case")
    case = summary["rows"][0]
    native, trace = artifact(directory, case["artifacts"]["native"]), artifact(directory, case["artifacts"]["trace"])
    audit_counts = result["native_audit"]
    require(audit_counts["external_target_qpos_writes"] == audit_counts["external_target_qvel_writes"] == 0,
        "journal reported external object state writes")
    require(audit_counts["integration_steps"] == audit_counts["native_samples"] == len(native) == len(trace),
        "native denominator differs from actual integration counts")
    require(len({tick(row["timestamp"]) for row in native}) == len(native), "duplicate native samples")
    by_tick = {tick(row["timestamp"]): row for row in native}
    by_trace = {tick(row["timestamp"]): row for row in trace}
    expected_mesh = (PROJECT / "vendor/ycb/005_tomato_soup_can/005_tomato_soup_can/poisson/nontextured.stl").resolve()
    strict = audit_case(directory, dict(case, runner_error=None), {"ycb_can": expected_mesh})
    require(strict["unknown"] is False, "strict raw native audit incomplete")
    require(strict["checks"]["native_contiguous_2ms"] is True, "native evidence has missing or irregular steps")
    controller = artifact(directory, protocol["controller_rows"]) if "controller_rows" in protocol else []
    first = next((row for row in controller if "before_robot_state" in row), None)
    bias = None
    if first is not None:
        state = first["before_robot_state"]
        _, actuators = servo_binding(model, state["joint_names"])
        bias = array(state["actual_ctrl"], (model.nu,))[actuators] - array(state["qpos_rad"])
    checked, torque_error, applied_ticks = 0, 0., []
    for row in controller:
        if row["command_applied"]:
            start = tick(row["timestamp_s"])
            require(start in by_tick and start + 1 in by_tick, "applied command lacks before/after native record")
            state, previous = row["before_robot_state"], by_tick[start]
            for name, value, speed in zip(state["joint_names"], state["qpos_rad"], state["qvel_rad_s"], strict=True):
                close(value, previous["robot_joint_position_rad"][name], "controller actual position differs from native")
                if name.startswith("left_"):
                    close(speed, previous["actual_hand_joints"][name]["velocity_rad_s"], "hand velocity differs from native")
                else:
                    close(speed, by_trace[start]["arm_joint_velocity_rad_s"][int(name.removeprefix("joint")) - 1],
                        "arm velocity differs from native trace")
            close(state["actual_ctrl"], previous["motor_commands"], "before actual command differs from native")
            torque_error = max(torque_error, check_command(model, row, by_tick[start + 1], bias))
            applied_ticks.append(start + 1)
            checked += 1
        elif "solver" in row and not row["solver"]["accepted"]:
            require("abort" in row, "rejected candidate lacks retained abort")
    continuation_native = [value for value in by_tick if value > 11000]
    require(applied_ticks == continuation_native, "unlogged continuation step or discarded failure step")
    arrivals = audit_arrivals(result.get("transitions", []), controller, protocol["configuration"])
    cached_peaks = array(list(result["native_cache_channel_peaks_n"].values()), (6,))
    require(np.all(cached_peaks >= 0.), "invalid recorded cached contact-force peaks")
    independent_task_success = bool(result["success"] and strict["strict_success"] and np.max(cached_peaks) <= 12.)
    require(result["success"] is False or independent_task_success, "reported task success failed independent strict gates")
    return {"format": "independent_r29_native_command_audit_v1", "accepted_evidence": True,
        "task_success": independent_task_success, "recorded_task_success": result["success"],
        "failure": result["failure"], "strict_audit": strict, "checked_applied_commands": checked,
        "maximum_cached_motor_torque_prediction_error_nm": torque_error, "arrival_transitions": arrivals,
        "recorded_cache_force_peaks_n": result["native_cache_channel_peaks_n"],
        "source_archive_files": source_count, "native_samples": len(native),
        "summary_sha256": sha256(directory / "summary.json"), "audit_script_sha256": sha256(Path(__file__)),
        "checks": {"frozen_source_and_inputs": True, "native_write_assertions_and_denominator": True,
            "independent_strict_raw_task_audit": True, "actual_single_writer_commands": True,
            "joint_rate_deflection_torque_bounds": True, "actual_arrival_evidence": True},
        "scope": "Valid negative experiments may pass evidence audit; fixed-state motor torque bounds are not dynamic contact-force invariants. Journal write counters and aggregate step-start contact-force peaks are runner assertions, not independently replayed physics; integrated-state contacts and commands are independently recomputed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.directory.resolve(), args.output.resolve()
    require(output != directory / "summary.json" and not output.exists(), "new independent receipt path required")
    try:
        report = audit(directory)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        report = {"format": "independent_r29_native_command_audit_v1", "accepted_evidence": False,
            "task_success": None, "error": str(error), "audit_script_sha256": sha256(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=json_number)
    output.write_text(encoded + "\n")
    print(encoded)
    return int(not report["accepted_evidence"])


if __name__ == "__main__":
    raise SystemExit(main())
