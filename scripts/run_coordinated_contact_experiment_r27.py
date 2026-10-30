#!/usr/bin/env python3
"""A finite contact-regulation experiment, not a grasp or force-safety claim.

First replay saved actual motor inputs into the archived scene and compare a
robot/tactile whitelist at every native step. Only a matching replay can create
a NEW integration-state checkpoint. Three conditions restore that checkpoint.
No historical controller, AST transformation, object pose, or material lookup
participates in regulation. No rendering or GPU is required.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_policy import CONTACT_CHANNELS, FingerContactPolicy
from kuka_sharpa_grasp.pick_place import Scenario, reset_task
from public_ycb_evidence import load_evidence_model


BASELINE = PROJECT / "results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
CONDITIONS = ("HOLD", "LOCKED_ARM", "FULL_JOINT")


@dataclass(frozen=True)
class Configuration:
    replay_until_s: float = 22.
    average_window_s: float = .2
    response_seconds: float = 2.
    force_abort_n: float = 12.
    contact_floor_n: float = .05
    uncontrolled_contact_tolerance_n: float = 1e-5
    force_target_limit_n: float = 6.
    pressure_velocity_gain_m_n_s: float = .0005
    maximum_normal_velocity_m_s: float = .001
    arm_command_rate_rad_s: float = .025
    finger_command_rate_rad_s: float = .05
    added_deflection_limit_rad: float = .05
    task_regularizer_m_rad: float = .001
    maximum_contact_residual_m_s: float = .0001
    replay_position_tolerance_rad: float = 1e-6
    replay_velocity_tolerance_rad_s: float = 1e-5
    replay_force_tolerance_n: float = .01
    replay_patch_position_tolerance_m: float = 1e-5
    replay_patch_normal_tolerance: float = 1e-4

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(item.name + " must be finite and positive")
        if self.force_target_limit_n > 6. or self.force_abort_n > 12.:
            raise ValueError("This experiment does not relax the 6N target or 12N abort limits")
        if self.average_window_s > self.replay_until_s:
            raise ValueError("The averaging window must fit within replay")
        for value in (self.replay_until_s, self.average_window_s, self.response_seconds):
            if abs(value / .002 - round(value / .002)) > 1e-7:
                raise ValueError("All durations must be integral native 2ms intervals")


class ExperimentAbort(RuntimeError):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail
        super().__init__(reason)


def json_value(value):
    if is_dataclass(value):
        return {item.name: json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_policy(config):
    return FingerContactPolicy(("thumb", "middle", "ring"), {
        "thumb": ("left_thumb_MCP_FE", "left_thumb_IP"),
        "middle": ("left_middle_MCP_FE", "left_middle_PIP", "left_middle_DIP"),
        "ring": ("left_ring_MCP_FE", "left_ring_PIP", "left_ring_DIP")},
        normal_target_limit_n=config.force_target_limit_n, contact_floor_n=config.contact_floor_n,
        uncontrolled_contact_tolerance_n=config.uncontrolled_contact_tolerance_n)


def robot_state(model, data, joint_names):
    joints = [model.joint(name).id for name in joint_names]
    return {"joint_names": list(joint_names), "qpos_rad": data.qpos[model.jnt_qposadr[joints]].copy(),
        "qvel_rad_s": data.qvel[model.jnt_dofadr[joints]].copy(), "actual_ctrl": data.ctrl.copy(),
        "timestamp_s": float(data.time)}


def canonical_patches(patches):
    return sorted([dict(patch) for patch in patches], key=lambda patch: (
        patch["finger"], patch["hand_body_name"], *patch["position_world_m"], patch["input_index"]))


def archive_array(value, shape, label):
    try:
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("invalid shape or nonfinite value")
        return array
    except (TypeError, ValueError) as error:
        raise ExperimentAbort("replay_invalid_archive", label + ": " + str(error)) from error


def expected_robot_state(native_row, trace_row, joint_names, arm_names):
    arm_q = archive_array(trace_row["arm_joint_position"], (len(arm_names),), "arm_joint_position")
    arm_v = archive_array(trace_row["arm_joint_velocity_rad_s"], (len(arm_names),), "arm_joint_velocity_rad_s")
    arm_indices = {name: index for index, name in enumerate(arm_names)}
    positions, velocities = [], []
    for name in joint_names:
        if name in arm_indices:
            positions.append(arm_q[arm_indices[name]])
            velocities.append(arm_v[arm_indices[name]])
        else:
            row = native_row["actual_hand_joints"][name]
            positions.append(float(archive_array(row["position_rad"], (), name + ".position_rad")))
            velocities.append(float(archive_array(row["velocity_rad_s"], (), name + ".velocity_rad_s")))
    pressures = archive_array([trace_row["hand_segment_forces"][name] for name in CONTACT_CHANNELS],
        (len(CONTACT_CHANNELS),), "hand_segment_forces")
    if np.any(pressures < 0.):
        raise ExperimentAbort("replay_invalid_archive", "Negative recorded channel pressure")
    return np.array(positions), np.array(velocities), pressures


def expected_tactile_patches(model, native_row):
    """Evaluation-only archive conversion; inspect only the hand side of contacts."""
    channels = {}
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if body.startswith("left_"):
            channels[geom] = (next((finger for finger in CONTACT_CHANNELS[:-1]
                if body.startswith(f"left_{finger}_")), "palm"), body)
    patches = []
    for index, contact in enumerate(native_row["native_contacts"]):
        geoms = archive_array(contact["geom_ids"], (2,), "contact geom_ids")
        if np.any(geoms != np.floor(geoms)) or np.any(geoms < 0) or np.any(geoms >= model.ngeom):
            raise ExperimentAbort("replay_invalid_archive", "Invalid recorded contact geom ids")
        first, second = map(int, geoms)
        if (first in channels) == (second in channels):
            continue
        hand = first if first in channels else second
        finger, body = channels[hand]
        frame = archive_array(contact["contact_frame_world_rows"], (3, 3), "contact frame")
        point = archive_array(contact["position_world_m"], (3,), "contact position")
        force = archive_array(contact["force_torque_contact_frame"], (6,), "contact force")
        normal = frame[0] * (1. if hand == first else -1.)
        patches.append(dict(input_index=index, finger=finger, hand_body_name=body,
            position_world_m=point.tolist(), normal_on_object_world=normal.tolist(),
            measured_pressure_n=abs(float(force[0]))))
    return canonical_patches(patches)


def compare_replay_sample(model, data, native_row, trace_row, joint_names, arm_names, config, expected_archive=None):
    """The archive's object pose, spatial velocity and material fields are unused."""
    actual = robot_state(model, data, joint_names)
    if expected_archive is None:
        expected_archive = (*expected_robot_state(native_row, trace_row, joint_names, arm_names),
            expected_tactile_patches(model, native_row))
    expected_q, expected_v, expected_pressures, expected = expected_archive
    sample = read_native_contact_sample(model, data)
    patches = canonical_patches(sample.patches)
    metrics = {"timestamp_s": float(data.time),
        "qpos_error_rad": float(np.max(abs(actual["qpos_rad"] - expected_q))),
        "qvel_error_rad_s": float(np.max(abs(actual["qvel_rad_s"] - expected_v))),
        "force_error_n": float(np.max(abs(np.array([sample.measured_pressures_n[name]
            for name in CONTACT_CHANNELS]) - expected_pressures))),
        "patch_position_error_m": 0., "patch_normal_error": 0., "patch_force_error_n": 0.,
        "patch_count_matches": len(patches) == len(expected)}
    masks_match = metrics["patch_count_matches"]
    if masks_match:
        for observed, recorded in zip(patches, expected, strict=True):
            masks_match &= (observed["finger"], observed["hand_body_name"]) == (recorded["finger"], recorded["hand_body_name"])
            metrics["patch_position_error_m"] = max(metrics["patch_position_error_m"],
                float(np.max(abs(np.asarray(observed["position_world_m"]) - recorded["position_world_m"]))))
            metrics["patch_normal_error"] = max(metrics["patch_normal_error"],
                float(np.max(abs(np.asarray(observed["normal_on_object_world"]) - recorded["normal_on_object_world"]))))
            metrics["patch_force_error_n"] = max(metrics["patch_force_error_n"],
                abs(observed["measured_pressure_n"] - recorded["measured_pressure_n"]))
    metrics["patch_mask_matches"] = masks_match
    limits = {"qpos_error_rad": config.replay_position_tolerance_rad,
        "qvel_error_rad_s": config.replay_velocity_tolerance_rad_s,
        "force_error_n": config.replay_force_tolerance_n,
        "patch_position_error_m": config.replay_patch_position_tolerance_m,
        "patch_normal_error": config.replay_patch_normal_tolerance,
        "patch_force_error_n": config.replay_force_tolerance_n}
    metrics["accepted"] = bool(masks_match and all(np.isfinite(metrics[key])
        and metrics[key] <= limit for key, limit in limits.items()))
    return metrics, sample


def replay_prefix(model, data, native_rows, trace_rows, joint_names, arm_names, config):
    if abs(float(model.opt.timestep) - .002) > 1e-12 or abs(float(data.time)) > 1e-12:
        raise ExperimentAbort("replay_invalid_initial_clock")
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ExperimentAbort("replay_unrecorded_applied_forces")
    required = round(config.replay_until_s / .002)
    if len(native_rows) < required:
        raise ExperimentAbort("replay_missing_native_rows")
    try:
        traces = {round(float(archive_array(row["timestamp"], (), "trace timestamp")) / .002): row
            for row in trace_rows}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
        raise ExperimentAbort("replay_invalid_archive", str(error)) from error
    checks, window = [], []
    for index in range(required):
        row = native_rows[index]
        tick = index + 1
        try:
            timestamp = float(archive_array(row["timestamp"], (), "native timestamp"))
            if abs(timestamp - tick * .002) > 1e-7 or tick not in traces:
                raise ExperimentAbort("replay_noncontiguous_archive", {"tick": tick})
            command = archive_array(row["motor_commands"], (model.nu,), "motor_commands")
            expected = (*expected_robot_state(row, traces[tick], joint_names, arm_names),
                expected_tactile_patches(model, row))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
            raise ExperimentAbort("replay_invalid_archive", {"tick": tick, "error": str(error)}) from error
        data.ctrl[:] = command  # Archived values already include the old feedforward offset.
        mujoco.mj_step(model, data)
        checked, sample = compare_replay_sample(model, data, row, traces[tick], joint_names, arm_names, config, expected)
        checks.append(checked)
        if not checked["accepted"]:
            return {"accepted": False, "terminal": "replay_mismatch", "checks": checks}, []
        if data.time > config.replay_until_s - config.average_window_s + 1e-9:
            window.append(dict(sample.measured_pressures_n))
    return {"accepted": True, "terminal": "replay_matched", "checks": checks,
        "scope": "New replay validated against robot/tactile archive whitelist, not an old exact checkpoint"}, window


def integration_state(model, data):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    if not np.isfinite(state).all():
        raise ExperimentAbort("nonfinite_checkpoint")
    return spec, state


def restore_checkpoint(model, spec, state):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state.copy(), spec)
    mujoco.mj_forward(model, data)
    return data


def declared_targets(window, policy):
    if not window:
        raise ExperimentAbort("missing_force_average_window")
    targets = dict.fromkeys(CONTACT_CHANNELS, 0.)
    deltas = {"thumb": -.5, "middle": -.5, "ring": .5}
    for finger in policy.active_fingers:
        targets[finger] = float(np.mean([row[finger] for row in window])) + deltas[finger]
    try:
        return dict(policy.validate_targets_n(targets))
    except ValueError as error:
        raise ExperimentAbort("invalid_declared_force_targets", str(error)) from error


def check_contact_sample(sample, policy, config):
    pressures = sample.measured_pressures_n
    if max(pressures.values()) > config.force_abort_n:
        raise ExperimentAbort("native_force_abort", dict(pressures))
    try:
        return policy.validate_contacts(sample.patches, pressures,
            controlled_joint_names=policy.controlled_joint_names, timestamp_s=sample.timestamp_s)
    except ValueError as error:
        raise ExperimentAbort("contact_contract_abort", str(error)) from error


def run_condition(model, spec, checkpoint, adapter, targets, condition, config):
    if condition not in CONDITIONS:
        raise ValueError("Unknown paired condition")
    data = restore_checkpoint(model, spec, checkpoint)
    policy = adapter.policy
    servos = adapter.servos
    actuator_ids = [servo.actuator_id for servo in servos]
    qpos_ids = [servo.qpos_index for servo in servos]
    bias = data.ctrl[actuator_ids].copy() - data.qpos[qpos_ids]
    arm_count = len(servos) - len(policy.controlled_joint_names)
    rates = np.r_[np.full(arm_count, config.arm_command_rate_rad_s),
        np.full(len(servos) - arm_count, config.finger_command_rate_rad_s)]
    if condition == "LOCKED_ARM":
        rates[:arm_count] = 0.
    rows, terminal, peak = [], "completed", 0.
    for _ in range(round(config.response_seconds / .002)):
        row = {"before": robot_state(model, data, adapter.joint_names), "solver": None,
            "command_applied": False, "after": None}
        rows.append(row)
        try:
            before = read_native_contact_sample(model, data)
            row["before_tactile"] = json_value(before)
            row["before_canonical_patches"] = canonical_patches(before.patches)
            row["before_force_error_n"] = {finger: targets[finger] - before.measured_pressures_n[finger]
                for finger in policy.active_fingers}
            peak = max(peak, *before.measured_pressures_n.values())
            observation = check_contact_sample(before, policy, config)
            if condition != "HOLD":
                # A declared pressure-to-motion heuristic, not estimated stiffness.
                velocities = [float(np.clip(config.pressure_velocity_gain_m_n_s * (
                    targets[patch.finger] - before.measured_pressures_n[patch.finger]),
                    -config.maximum_normal_velocity_m_s, config.maximum_normal_velocity_m_s))
                    for patch in observation.patches]
                counts = {finger: sum(patch.finger == finger for patch in observation.patches)
                    for finger in policy.active_fingers}
                weights = [1. / counts[patch.finger] for patch in observation.patches]
                solved = adapter.solve(data, observation, velocities,
                    command_rate_lower_rad_s=-rates, command_rate_upper_rad_s=rates,
                    deflection_lower_rad=bias - config.added_deflection_limit_rad,
                    deflection_upper_rad=bias + config.added_deflection_limit_rad,
                    task_reference_velocity_rad_s=np.zeros(len(servos)),
                    task_weights=np.full(len(servos), config.task_regularizer_m_rad ** 2),
                    contact_weights=weights, maximum_contact_residual_m_s=config.maximum_contact_residual_m_s)
                row["solver"] = json_value(solved)
                row["target_normal_velocity_m_s"] = velocities
                if not solved.accepted:
                    raise ExperimentAbort("solver_rejected", solved.reason)
                data.ctrl[actuator_ids] = solved.command_rad
                row["command_applied"] = True
            row["actual_ctrl_for_step"] = data.ctrl.copy()
            start = float(data.time)
            mujoco.mj_step(model, data)
            cached = read_native_contact_sample(model, data)
            peak = max(peak, *cached.measured_pressures_n.values())
            row["step_dynamics_cache"] = {"step_start_s": start, "step_end_s": float(data.time),
                "force_evaluation_scope": "Native dynamics cache for step; not recomputed at integrated qpos",
                "canonical_patches": canonical_patches(cached.patches),
                "channel_forces_n": dict(cached.measured_pressures_n),
                "actuator_force_nm": data.actuator_force[actuator_ids].copy(),
                "actuator_length_rad": data.actuator_length[actuator_ids].copy(),
                "actuator_velocity_rad_s": data.actuator_velocity[actuator_ids].copy()}
            if row["solver"] is not None:
                row["step_dynamics_cache"]["actual_minus_frozen_state_predicted_motor_torque_nm"] = (
                    data.actuator_force[actuator_ids] - np.asarray(row["solver"]["predicted_motor_torque_nm"]))
            cache_abort = None
            try:
                check_contact_sample(cached, policy, config)
            except ExperimentAbort as error:
                cache_abort = error
            # Explicitly refresh at integrated qpos before using these forces as
            # the next controller sample. This zero-time call is part of protocol.
            mujoco.mj_forward(model, data)
            after = read_native_contact_sample(model, data)
            peak = max(peak, *after.measured_pressures_n.values())
            row["after"] = robot_state(model, data, adapter.joint_names)
            row["after_tactile"] = json_value(after)
            row["after_canonical_patches"] = canonical_patches(after.patches)
            row["force_error_n"] = {finger: targets[finger] - after.measured_pressures_n[finger]
                for finger in policy.active_fingers}
            if cache_abort is not None:
                raise cache_abort
            check_contact_sample(after, policy, config)
        except (ExperimentAbort, ValueError, FloatingPointError) as error:
            terminal = error.reason if isinstance(error, ExperimentAbort) else "execution_error"
            row["abort"] = {"reason": terminal, "detail": getattr(error, "detail", str(error))}
            break
    errors = [list(row["force_error_n"].values()) for row in rows if "force_error_n" in row]
    required = round(config.response_seconds / .002)
    tail = min(required, round(.2 / .002))
    native_steps = sum("step_dynamics_cache" in row for row in rows)
    initial_error = list(rows[0].get("before_force_error_n", {}).values()) if rows else []
    return {"condition": condition, "terminal": terminal, "completed": terminal == "completed",
        "native_steps": native_steps, "required_native_steps": required,
        "full_response_completed": terminal == "completed" and native_steps == required,
        "peak_channel_force_n": peak,
        "targets_n": dict(targets), "fixed_capture_deflection_rad": bias,
        "initial_robot_state": rows[0]["before"] if rows else None,
        "initial_rms_force_error_n": None if not initial_error else float(np.sqrt(np.mean(np.square(initial_error)))),
        "last_window_rms_force_error_n": None if not errors else float(np.sqrt(np.mean(np.square(errors[-tail:])))),
        "last_window_requested_samples": tail, "last_window_actual_samples": min(len(errors), tail),
        "last_window_complete": len(errors) >= tail,
        "trace": rows, "task_success": None,
        "claim_scope": "Contact regulation only; no grasp, retention or dynamic force invariant claim"}


def paired_comparison(conditions):
    """Aborted trajectories retain raw RMS, but cannot establish improvement."""
    eligible = (len(conditions) == len(CONDITIONS)
        and tuple(row["condition"] for row in conditions) == CONDITIONS
        and all(row["completed"] and row["full_response_completed"] and row["last_window_complete"]
            for row in conditions)
        and len({row["native_steps"] for row in conditions}) == 1)
    comparison = {"paired_comparison_eligible": bool(eligible), "full_joint_minus_hold_rms_n": None,
        "full_joint_minus_locked_arm_rms_n": None, "full_joint_minus_initial_rms_n": None,
        "interpretation": "Native force-error comparison only, not task success"}
    if eligible:
        hold, locked, full = conditions
        comparison.update(full_joint_minus_hold_rms_n=full["last_window_rms_force_error_n"] - hold["last_window_rms_force_error_n"],
            full_joint_minus_locked_arm_rms_n=full["last_window_rms_force_error_n"] - locked["last_window_rms_force_error_n"],
            full_joint_minus_initial_rms_n=full["last_window_rms_force_error_n"] - full["initial_rms_force_error_n"])
    return comparison


def source_snapshot():
    paths = [Path(__file__).resolve(), PROJECT / "scripts/public_ycb_evidence.py"]
    paths.extend(sorted((PROJECT / "src/kuka_sharpa_grasp").glob("*.py")))
    return {path.relative_to(PROJECT).as_posix(): file_hash(path) for path in paths}


def run(output, baseline, config):
    if output.exists() or output.is_symlink():
        raise FileExistsError("A new output directory is required")
    output.mkdir(parents=True)
    source_before = source_snapshot()
    inputs = {name: file_hash(baseline / name) for name in ("scene.xml", "native.json.gz", "trace.json.gz")}
    protocol = {"format": "r27_coordinated_native_contact_experiment_v1", "configuration": json_value(config),
        "conditions": CONDITIONS, "force_delta_n": {"thumb": -.5, "middle": -.5, "ring": .5},
        "mujoco_version": mujoco.__version__, "timestep_s": .002,
        "input_hashes": inputs, "source_hashes_before": source_before,
        "sampling": "restore+forward; before refreshed sample; one direct command write; mj_step; native cache; mj_forward; after refreshed sample",
        "feedforward": "Initial actual motor references include archived offsets; no extra gravity writer or external force",
        "force_gate": "12N abort threshold with violating samples retained, not a post-step guarantee",
        "controller_inputs": "Robot proprioception, declared servo limits, all external-hand tactile patches only",
        "checkpoint_scope": "New mjSTATE_INTEGRATION after checked motor replay; archive is not an exact checkpoint",
        "task_success": None}
    write_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "terminal": "preparing", "conditions": [], "task_success": None,
        "paired_comparison_eligible": False}
    try:
        model = load_evidence_model(baseline / "scene.xml")
        compiled_path = output / "compiled_model.mjb"
        mujoco.mj_saveModel(model, str(compiled_path), None)
        protocol["compiled_model"] = {"file": compiled_path.name, "sha256": file_hash(compiled_path),
            "scope": "Actual compiled MuJoCo model including loaded mesh and texture data"}
        data = mujoco.MjData(model)
        if abs(float(model.opt.timestep) - .002) > 1e-12:
            raise ExperimentAbort("invalid_native_timestep")
        reset_task(model, data, Scenario(3414, "ycb_can"))
        policy = make_policy(config)
        adapter = ContactExecutionAdapter(model, policy, arm_joint_names=ARM_JOINTS)
        robot_joints = tuple(model.joint(joint).name for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE)
        with gzip.open(baseline / "native.json.gz", "rt") as stream:
            native_rows = json.load(stream)
        with gzip.open(baseline / "trace.json.gz", "rt") as stream:
            trace_rows = json.load(stream)
        replay, window = replay_prefix(model, data, native_rows, trace_rows, robot_joints, ARM_JOINTS, config)
        write_json(output / "replay_verification.json", replay)
        if not replay["accepted"]:
            summary["terminal"] = "replay_mismatch"
        else:
            targets = declared_targets(window, policy)
            # Canonical refresh is explicit and identical to each restored condition.
            mujoco.mj_forward(model, data)
            check_contact_sample(read_native_contact_sample(model, data), policy, config)
            spec, checkpoint = integration_state(model, data)
            np.savez(output / "new_integration_checkpoint.npz", state=checkpoint, state_spec=spec)
            write_json(output / "checkpoint_manifest.json", {"sha256": file_hash(output / "new_integration_checkpoint.npz"),
                "state_spec": spec, "state_size": len(checkpoint), "time_s": float(data.time),
                "model_nq": model.nq, "model_nv": model.nv, "model_nu": model.nu,
                "force_average_window": window, "targets_n": targets, "input_hashes": inputs,
                "compiled_model": protocol["compiled_model"]})
            for condition in CONDITIONS:
                result = run_condition(model, spec, checkpoint, adapter, targets, condition, config)
                write_json(output / (condition.lower() + ".json"), result)
                summary["conditions"].append({key: value for key, value in result.items() if key != "trace"})
            summary["terminal"] = "paired_conditions_recorded"
    except (ExperimentAbort, ValueError, RuntimeError) as error:
        summary["terminal"] = error.reason if isinstance(error, ExperimentAbort) else "runner_error"
        summary["error"] = str(error)
        summary["detail"] = getattr(error, "detail", None)
    summary.update(paired_comparison(summary["conditions"]))
    protocol["source_hashes_after"] = source_snapshot()
    protocol["source_changed_during_run"] = source_before != protocol["source_hashes_after"]
    protocol["input_hashes_after"] = {name: file_hash(baseline / name) for name in inputs}
    protocol["inputs_changed_during_run"] = inputs != protocol["input_hashes_after"]
    write_json(output / "protocol.json", protocol)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--arm-command-rate", type=float, default=.025)
    parser.add_argument("--finger-command-rate", type=float, default=.05)
    args = parser.parse_args()
    config = Configuration(arm_command_rate_rad_s=args.arm_command_rate,
        finger_command_rate_rad_s=args.finger_command_rate)
    summary = run(args.output.resolve(), args.baseline.resolve(), config)
    print(json.dumps({"terminal": summary["terminal"], "paired_comparison_eligible": summary["paired_comparison_eligible"], "conditions": [
        {key: value for key, value in row.items() if key in ("condition", "terminal", "native_steps", "peak_channel_force_n")}
        for row in summary["conditions"]]}, indent=2))
    return int(not summary["paired_comparison_eligible"]
        or summary["protocol"]["source_changed_during_run"] or summary["protocol"]["inputs_changed_during_run"])


if __name__ == "__main__":
    raise SystemExit(main())
