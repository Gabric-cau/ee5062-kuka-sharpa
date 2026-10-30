#!/usr/bin/env python3
"""Read-only identity and native trace arithmetic audit for finite R27 trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
FORMAT = "r27_coordinated_contact_identity_and_trace_arithmetic_audit_v1"
RUNNER = "scripts/run_coordinated_contact_experiment_r27.py"
RUNNER_SHA = "f47e32454c7a172af143ee3d9de2a3c1852ddce7cdb18cd63d4573df6e595e12"
CONDITIONS = ("HOLD", "LOCKED_ARM", "FULL_JOINT")
ACTIVE = ("thumb", "middle", "ring")
CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
ARM = tuple("joint" + str(i) for i in range(1, 8))
DT = .002


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def near(actual, expected, label, tolerance=1e-12):
    first, second = np.asarray(actual, float), np.asarray(expected, float)
    require(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()
        and np.all(np.abs(first - second) <= tolerance), label + " differs")


def replay_audit(replay, configuration):
    limits = {"qpos_error_rad": 1e-6, "qvel_error_rad_s": 1e-5, "force_error_n": .01,
        "patch_position_error_m": 1e-5, "patch_normal_error": 1e-4, "patch_force_error_n": .01}
    config_keys = ("replay_position_tolerance_rad", "replay_velocity_tolerance_rad_s",
        "replay_force_tolerance_n", "replay_patch_position_tolerance_m", "replay_patch_normal_tolerance",
        "replay_force_tolerance_n")
    for (name, limit), key in zip(limits.items(), config_keys):
        near(configuration[key], limit, "unrelaxed replay tolerance " + name, 0.)
    checks = replay["checks"]
    require(replay["accepted"] is True and replay["terminal"] == "replay_matched"
        and len(checks) == round(configuration["replay_until_s"] / DT), "Replay is not fully matched")
    maxima = dict.fromkeys(limits, 0.)
    for ordinal, row in enumerate(checks, 1):
        near(row["timestamp_s"], ordinal * DT, "replay clock", 1e-7)
        require(row["accepted"] is True and row["patch_count_matches"] is True
            and row["patch_mask_matches"] is True, "Replay sample/mask rejected")
        for name, limit in limits.items():
            value = float(row[name])
            require(np.isfinite(value) and 0. <= value <= limit, "Replay mismatch: " + name)
            maxima[name] = max(maxima[name], value)
    return {"native_steps": len(checks), "accepted": True, "maximum_errors": maxima,
        "tolerances_unchanged": True}


def condition_audit(result, configuration, checkpoint_time):
    condition, rows = result["condition"], result["trace"]
    required = round(configuration["response_seconds"] / DT)
    tail = min(required, round(.2 / DT))
    require(condition in CONDITIONS and result["terminal"] == "completed" and result["completed"] is True
        and result["full_response_completed"] is True and result["last_window_complete"] is True
        and result["native_steps"] == result["required_native_steps"] == len(rows) == required,
        "Condition aborted/incomplete and cannot establish a paired improvement")
    require(rows[0]["before"] == result["initial_robot_state"], "Initial robot state differs from first trace")
    require(result["task_success"] is None, "Contact trial was relabeled as grasp success")
    errors, peak, torque_residual, aborts = [], 0., None, []
    for index, row in enumerate(rows):
        require("abort" not in row and row["after"] is not None, "Unexpected condition abort")
        near(row["before"]["timestamp_s"], checkpoint_time + index * DT, "before native clock", 1e-7)
        near(row["after"]["timestamp_s"], checkpoint_time + (index + 1) * DT, "after native clock", 1e-7)
        cache = row["step_dynamics_cache"]
        near(cache["step_start_s"], row["before"]["timestamp_s"], "cache start clock")
        near(cache["step_end_s"], row["after"]["timestamp_s"], "cache end clock")
        near(row["actual_ctrl_for_step"], row["after"]["actual_ctrl"], "actual integrated motor commands")
        solver = row["solver"]
        if condition == "HOLD":
            require(solver is None and row["command_applied"] is False, "HOLD unexpectedly issued a solver command")
            near(row["actual_ctrl_for_step"], rows[0]["before"]["actual_ctrl"], "fixed HOLD motor commands")
        else:
            require(solver["accepted"] is True and row["command_applied"] is True, "Unaccepted solver command")
            difference = np.asarray(cache["actuator_force_nm"]) - np.asarray(solver["predicted_motor_torque_nm"])
            near(cache["actual_minus_frozen_state_predicted_motor_torque_nm"], difference, "motor prediction residual")
            torque_residual = max(torque_residual or 0., float(np.max(np.abs(difference))))
        for channels in (row["before_tactile"]["measured_pressures_n"],
                row["after_tactile"]["measured_pressures_n"], cache["channel_forces_n"]):
            require(set(channels) == set(CHANNELS), "Incomplete external hand channel coverage")
            values = np.asarray(list(channels.values()), float)
            require(np.isfinite(values).all() and np.all(values >= 0.), "Invalid native channel pressure")
            peak = max(peak, float(np.max(values)))
        error = [result["targets_n"][finger] - row["after_tactile"]["measured_pressures_n"][finger]
            for finger in ACTIVE]
        near([row["force_error_n"][finger] for finger in ACTIVE], error, "native force error")
        errors.append(error)
    initial_error = [result["targets_n"][finger] - rows[0]["before_tactile"]["measured_pressures_n"][finger]
        for finger in ACTIVE]
    initial_rms = float(np.sqrt(np.mean(np.square(initial_error))))
    tail_rms = float(np.sqrt(np.mean(np.square(errors[-tail:]))))
    near(result["initial_rms_force_error_n"], initial_rms, "initial RMS")
    near(result["last_window_rms_force_error_n"], tail_rms, "last-window RMS")
    near(result["peak_channel_force_n"], peak, "whole-condition peak")
    require(peak <= configuration["force_abort_n"] and result["last_window_actual_samples"] == tail
        and result["last_window_requested_samples"] == tail, "Force gate/window differs")
    names = rows[0]["before"]["joint_names"]
    require(names == rows[-1]["after"]["joint_names"], "Robot joint coverage changed")
    delta = np.asarray(rows[-1]["after"]["qpos_rad"]) - np.asarray(rows[0]["before"]["qpos_rad"])
    arm_delta = [float(delta[names.index(name)]) for name in ARM]
    ctrl_delta = np.asarray(rows[-1]["after"]["actual_ctrl"]) - np.asarray(rows[0]["before"]["actual_ctrl"])
    return {"condition": condition, "terminal": "completed", "native_steps": required, "abort": None,
        "targets_n": result["targets_n"], "initial_rms_force_error_n": initial_rms,
        "last_window_rms_force_error_n": tail_rms, "peak_channel_force_n": peak,
        "end_robot_qpos_delta_rad": {name: float(value) for name, value in zip(names, delta)},
        "arm_actual_qpos_delta_rad": arm_delta, "arm_actual_max_abs_delta_rad": max(map(abs, arm_delta)),
        "arm_actual_l2_delta_rad": float(np.linalg.norm(arm_delta)),
        "arm_motor_command_delta_rad": ctrl_delta[:7].tolist(),
        "maximum_absolute_prediction_vs_native_motor_torque_residual_nm": torque_residual,
        "motor_torque_scope": "Frozen-state motor prediction versus the following native dynamics cache, not future contact-force prediction"}


def audit_directory(directory, expected_summary_sha):
    directory = Path(directory).resolve()
    files = sorted(path for path in directory.iterdir() if path.is_file())
    hashes = {path.name: sha(path) for path in files}
    require(hashes["summary.json"] == expected_summary_sha, "Summary SHA differs")
    summary, protocol = (json.loads((directory / name).read_text()) for name in ("summary.json", "protocol.json"))
    require(summary["protocol"] == protocol and summary["terminal"] == "paired_conditions_recorded"
        and summary["paired_comparison_eligible"] is True and summary["task_success"] is None,
        "Run does not contain an eligible paired contact comparison")
    require(protocol["source_hashes_before"] == protocol["source_hashes_after"]
        and protocol["source_changed_during_run"] is False
        and protocol["input_hashes"] == protocol["input_hashes_after"]
        and protocol["inputs_changed_during_run"] is False, "Experiment identity changed")
    require(protocol["source_hashes_before"][RUNNER] == RUNNER_SHA, "Runner identity differs")
    for relative, expected in protocol["source_hashes_before"].items():
        path = PROJECT / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and sha(path) == expected,
            "Current source differs: " + relative)
    baseline = PROJECT / "results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414"
    for relative, expected in protocol["input_hashes"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
            and sha(baseline / relative) == expected, "Baseline input differs: " + relative)
    config = protocol["configuration"]
    near([config["replay_until_s"], config["response_seconds"], config["force_abort_n"]], [22., 2., 12.], "finite protocol")
    replay = replay_audit(json.loads((directory / "replay_verification.json").read_text()), config)
    checkpoint = json.loads((directory / "checkpoint_manifest.json").read_text())
    require(checkpoint["compiled_model"] == protocol["compiled_model"]
        and checkpoint["input_hashes"] == protocol["input_hashes"], "Checkpoint identity differs")
    require(protocol["compiled_model"]["file"] == "compiled_model.mjb"
        and hashes["compiled_model.mjb"] == protocol["compiled_model"]["sha256"]
        and hashes["new_integration_checkpoint.npz"] == checkpoint["sha256"], "Binary model/checkpoint SHA differs")
    model = mujoco.MjModel.from_binary_path(str(directory / "compiled_model.mjb"))
    require((model.nq, model.nv, model.nu) == (checkpoint["model_nq"], checkpoint["model_nv"], checkpoint["model_nu"]),
        "Compiled model dimensions differ")
    with np.load(directory / "new_integration_checkpoint.npz", allow_pickle=False) as archive:
        state, spec = archive["state"], int(archive["state_spec"])
    require(spec == checkpoint["state_spec"] == int(mujoco.mjtState.mjSTATE_INTEGRATION)
        and len(state) == checkpoint["state_size"] == mujoco.mj_stateSize(model, spec)
        and np.isfinite(state).all(), "Invalid integration checkpoint")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    near(data.time, checkpoint["time_s"], "checkpoint time")
    reports, initial = [], []
    require(len(summary["conditions"]) == len(CONDITIONS), "Paired denominator differs")
    for name, row in zip(CONDITIONS, summary["conditions"]):
        result = json.loads((directory / (name.lower() + ".json")).read_text())
        require(result["condition"] == name and {k: v for k, v in result.items() if k != "trace"} == row,
            "Condition result/summary differs")
        observed = result["initial_robot_state"]
        initial.append(observed)
        joints = [model.joint(name).id for name in observed["joint_names"]]
        near(data.qpos[model.jnt_qposadr[joints]], observed["qpos_rad"], "checkpoint robot qpos", 0.)
        near(data.qvel[model.jnt_dofadr[joints]], observed["qvel_rad_s"], "checkpoint robot qvel", 0.)
        near(data.ctrl, observed["actual_ctrl"], "checkpoint actual motor inputs", 0.)
        reports.append(condition_audit(result, config, float(data.time)))
    require(initial[0] == initial[1] == initial[2], "Conditions did not start from identical robot state")
    hold, locked, full = [row["last_window_rms_force_error_n"] for row in reports]
    differences = {"full_joint_minus_hold_rms_n": full - hold,
        "full_joint_minus_locked_arm_rms_n": full - locked,
        "full_joint_minus_initial_rms_n": full - reports[2]["initial_rms_force_error_n"]}
    for key, value in differences.items():
        near(summary[key], value, "paired difference " + key)
    require(all(sha(directory / name) == expected for name, expected in hashes.items()), "Input changed during audit")
    return {"format": FORMAT, "accepted_trace_arithmetic": True, "summary_sha256": expected_summary_sha,
        "script_sha256": sha(__file__), "artifact_hashes": hashes, "source_hashes": protocol["source_hashes_before"],
        "baseline_input_hashes": protocol["input_hashes"], "compiled_model_sha256": hashes["compiled_model.mjb"],
        "checkpoint_sha256": hashes["new_integration_checkpoint.npz"], "checkpoint_time_s": float(data.time),
        "identical_initial_robot_state": True, "replay": replay, "conditions": reports,
        "paired_comparison_eligible": True, **differences, "inputs_unchanged_during_audit": True,
        "task_success": None, "force_invariant_proven": False, "simulation_integration_performed_by_audit": False,
        "scope": "Single checked 22s contact checkpoint, three 2s responses. Robot/tactile whitelist and recorded force/RMS/torque arithmetic only. No independent contact-physics re-integration, grasp/retention result, robustness claim or future contact-force prediction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "New audit receipt required")
    try:
        receipt, status = audit_directory(args.directory, args.expected_summary_sha256), 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        receipt, status = {"format": FORMAT, "accepted_trace_arithmetic": False,
            "error": str(error), "script_sha256": sha(__file__)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"accepted_trace_arithmetic": receipt["accepted_trace_arithmetic"], "receipt": str(args.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
