#!/usr/bin/env python3
"""Matched 5mm motion probes; this experiment cannot establish task success."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import shutil
import socket
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_policy import ContactObservation
from kuka_sharpa_grasp.coordinated_motion_reference import CoordinatedMotionReference
from kuka_sharpa_grasp.pick_place import Scenario, reset_task
from public_ycb_evidence import load_evidence_model
import run_coordinated_contact_experiment_r27 as r27

CONDITIONS = ("STATIC_HOLD", "MOVING_WITHOUT_TRANSPORT", "MOVING_WITH_TRANSPORT")


@dataclass(frozen=True)
class Configuration:
    preload_seconds: float = 2.
    probe_distance_m: float = .005
    probe_speed_m_s: float = .004
    settle_seconds: float = .75
    arm_task_weight_m_rad: float = .02
    finger_task_weight_m_rad: float = .001
    maximum_tracking_error_m: float = .006

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(name + " must be finite and positive")
        if self.probe_distance_m > .005 or self.probe_speed_m_s > .004:
            raise ValueError("First motion probe is limited to 5mm at 4mm/s")
        for seconds in (self.preload_seconds, self.settle_seconds, self.probe_distance_m / self.probe_speed_m_s):
            if abs(seconds / .002 - round(seconds / .002)) > 1e-7:
                raise ValueError("Durations must be integral native 2ms intervals")

    @property
    def duration_s(self):
        return self.preload_seconds + self.probe_distance_m / self.probe_speed_m_s + self.settle_seconds


def trajectory(elapsed_s, condition, config):
    if condition not in CONDITIONS or not np.isfinite(elapsed_s) or elapsed_s < -1e-9:
        raise ValueError("Invalid trajectory condition or time")
    progress = max(0., elapsed_s - config.preload_seconds)
    if condition == "STATIC_HOLD":
        return np.zeros(3), np.zeros(3), "PRELOAD" if elapsed_s < config.preload_seconds else "HOLD"
    distance = min(config.probe_distance_m, config.probe_speed_m_s * progress)
    moving = config.preload_seconds <= elapsed_s < config.preload_seconds + config.probe_distance_m / config.probe_speed_m_s
    phase = "PRELOAD" if elapsed_s < config.preload_seconds else ("PROBE" if moving else "SETTLE")
    return np.array([0., 0., distance]), np.array([0., 0., config.probe_speed_m_s if moving else 0.]), phase


def normal_velocity_targets(reference, observation, pressures, force_targets, condition, contact_config):
    """Cap only the pressure heuristic, never the rigid transport term."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    if tuple(reference.patch_input_indices) != tuple(patch.input_index for patch in observation.patches):
        raise ValueError("Motion and contact row identities must match")
    declared = np.array([(force_targets[patch.finger], pressures[patch.finger]) for patch in observation.patches])
    if not np.isfinite(declared).all() or np.any(declared < 0.):
        raise ValueError("Native pressures and declared targets must be finite and nonnegative")
    correction = np.array([np.clip(contact_config.pressure_velocity_gain_m_n_s * (
        force_targets[patch.finger] - pressures[patch.finger]),
        -contact_config.maximum_normal_velocity_m_s, contact_config.maximum_normal_velocity_m_s)
        for patch in observation.patches])
    transport = np.asarray(reference.rigid_transport_normal_velocity_m_s, dtype=float)
    if transport.shape != correction.shape or not np.isfinite(transport).all():
        raise ValueError("Motion and contact rows must match")
    used = transport if condition == "MOVING_WITH_TRANSPORT" else np.zeros_like(transport)
    return correction + used, correction, transport


class NativeJournal:
    """Evaluation recorder; privileged fields never enter motion or force control."""

    def __init__(self, model, data):
        self.model = model
        self.target_body = model.body("target").id
        self.target_geom = model.geom("target_geom").id
        self.table = model.geom("table").id
        address = int(model.jnt_qposadr[model.joint("target_free").id])
        velocity_address = int(model.jnt_dofadr[model.joint("target_free").id])
        self.target_slice = slice(address, address + 7)
        self.target_velocity_slice = slice(velocity_address, velocity_address + 6)
        self.previous_target_qpos = data.qpos[self.target_slice].copy()
        self.previous_target_qvel = data.qvel[self.target_velocity_slice].copy()
        self.initial_target_qpos = self.previous_target_qpos.copy()
        self.external_qpos_writes = []
        self.external_qvel_writes = []
        self.integration_steps = 0
        self.replay_checks, self.replay_native = [], []
        self.hand_joints = [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and (model.joint(j).name or "").startswith("left_")]

    def step(self, data):
        current = data.qpos[self.target_slice].copy()
        if not np.array_equal(current, self.previous_target_qpos):
            self.external_qpos_writes.append({"time_s": float(data.time),
                "before": self.previous_target_qpos.tolist(), "after": current.tolist()})
            raise r27.ExperimentAbort("external_target_qpos_write")
        current_velocity = data.qvel[self.target_velocity_slice].copy()
        if not np.array_equal(current_velocity, self.previous_target_qvel):
            self.external_qvel_writes.append({"time_s": float(data.time),
                "before": self.previous_target_qvel.tolist(), "after": current_velocity.tolist()})
            raise r27.ExperimentAbort("external_target_qvel_write")
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise r27.ExperimentAbort("unrecorded_applied_force")
        mujoco.mj_step(self.model, data)
        self.integration_steps += 1
        self.previous_target_qpos = data.qpos[self.target_slice].copy()
        self.previous_target_qvel = data.qvel[self.target_velocity_slice].copy()

    def sample(self, data, phase, *, force_cache_scope):
        contacts, wrench = [], np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            mujoco.mj_contactForce(self.model, data, index, wrench)
            contacts.append({"geom_ids": [int(contact.geom1), int(contact.geom2)],
                "distance_m": float(contact.dist), "position_world_m": contact.pos.tolist(),
                "contact_frame_world_rows": contact.frame.reshape(3, 3).tolist(),
                "force_torque_contact_frame": wrench.copy().tolist()})
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.target_body, velocity, 0)
        return {"timestamp": float(data.time), "state": phase,
            "target_qpos": data.qpos[self.target_slice].tolist(),
            "target_position_world_m": data.xpos[self.target_body].tolist(),
            "target_spatial_velocity_world": velocity.tolist(),
            "motor_commands": data.ctrl.copy().tolist(), "native_contacts": contacts,
            "force_cache_scope": force_cache_scope,
            "actual_hand_joints": {self.model.joint(j).name: {
                "position_rad": float(data.qpos[self.model.jnt_qposadr[j]]),
                "velocity_rad_s": float(data.qvel[self.model.jnt_dofadr[j]])} for j in self.hand_joints}}

    def audit(self):
        return {"integration_steps": self.integration_steps,
            "external_target_qpos_writes": len(self.external_qpos_writes),
            "external_target_qvel_writes": len(self.external_qvel_writes),
            "write_events": self.external_qpos_writes,
            "velocity_write_events": self.external_qvel_writes,
            "initial_target_qpos": self.initial_target_qpos.tolist()}


def replay(model, data, native_rows, trace_rows, joint_names, contact_config, journal):
    if abs(float(model.opt.timestep) - .002) > 1e-12 or abs(float(data.time)) > 1e-12:
        raise r27.ExperimentAbort("replay_invalid_initial_clock")
    required = round(contact_config.replay_until_s / .002)
    if len(native_rows) < required:
        raise r27.ExperimentAbort("replay_missing_native_rows")
    try:
        by_tick = {round(float(r27.archive_array(row["timestamp"], (), "trace timestamp")) / .002): row
            for row in trace_rows}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
        raise r27.ExperimentAbort("replay_invalid_archive", str(error)) from error
    checks, window, actual_native = journal.replay_checks, [], journal.replay_native
    for tick in range(1, required + 1):
        archived = native_rows[tick - 1]
        try:
            timestamp = float(r27.archive_array(archived["timestamp"], (), "native timestamp"))
            if abs(timestamp - tick * .002) > 1e-7 or tick not in by_tick:
                raise r27.ExperimentAbort("replay_noncontiguous_archive")
            trace = by_tick[tick]
            phase = "REPLAY_" + archived["state"]
            command = r27.archive_array(archived["motor_commands"], (model.nu,), "motor_commands")
            expected = (*r27.expected_robot_state(archived, trace, joint_names, r27.ARM_JOINTS),
                r27.expected_tactile_patches(model, archived))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
            raise r27.ExperimentAbort("replay_invalid_archive", {"tick": tick, "error": str(error)}) from error
        data.ctrl[:] = command
        journal.step(data)
        actual_native.append(journal.sample(data, phase,
            force_cache_scope="Native dynamics cache; kinematics and force were evaluated at step start"))
        checked, sample = r27.compare_replay_sample(model, data, archived, trace, joint_names,
            r27.ARM_JOINTS, contact_config, expected)
        checks.append(checked)
        if not checked["accepted"]:
            raise r27.ExperimentAbort("replay_mismatch", checked)
        if data.time > contact_config.replay_until_s - contact_config.average_window_s + 1e-9:
            window.append(dict(sample.measured_pressures_n))
    return {"accepted": True, "checks": checks}, window, actual_native


def run_condition(model, spec, checkpoint, adapter, force_targets, condition, config, contact_config):
    data = r27.restore_checkpoint(model, spec, checkpoint)
    journal = NativeJournal(model, data)
    motion = CoordinatedMotionReference(model, joint_names=adapter.joint_names)
    servos, policy = adapter.servos, adapter.policy
    act = [servo.actuator_id for servo in servos]
    qids = [servo.qpos_index for servo in servos]
    arm_count = len(r27.ARM_JOINTS)
    rates = np.r_[np.full(arm_count, contact_config.arm_command_rate_rad_s),
        np.full(len(servos) - arm_count, contact_config.finger_command_rate_rad_s)]
    weights = np.r_[np.full(arm_count, config.arm_task_weight_m_rad ** 2),
        np.full(len(servos) - arm_count, config.finger_task_weight_m_rad ** 2)]
    bias = data.ctrl[act].copy() - data.qpos[qids]
    start = float(data.time)
    initial = None
    rows, native, terminal = [], [], "completed"
    peak = 0.
    for _ in range(round(config.duration_s / .002)):
        row = {"before_time_s": float(data.time), "command_applied": False,
            "before_robot_state": r27.robot_state(model, data, adapter.joint_names)}
        rows.append(row)
        try:
            sample = read_native_contact_sample(model, data)
            row["before_tactile"] = r27.json_value(sample)
            row["before_canonical_patches"] = r27.canonical_patches(sample.patches)
            peak = max(peak, *sample.measured_pressures_n.values())
            observation = r27.check_contact_sample(sample, policy, contact_config)
            offset, velocity, phase = trajectory(float(data.time) - start, condition, config)
            row["phase"] = phase
            reference = motion.prepare(data, observation, desired_linear_velocity_world_m_s=velocity,
                joint_velocity_limits_rad_s=rates, desired_position_world_m=None if initial is None else initial[0] + offset,
                desired_rotation_world=None if initial is None else initial[1], max_linear_speed_m_s=.004)
            if initial is None:
                initial = (np.array(reference.wrist_position_world_m), np.array(reference.wrist_rotation_world_rows))
            row["motion_reference"] = r27.json_value(reference)
            if np.linalg.norm(reference.position_error_world_m) > config.maximum_tracking_error_m:
                raise r27.ExperimentAbort("wrist_tracking_error", reference.position_error_world_m)
            normal, correction, transport = normal_velocity_targets(reference, observation,
                sample.measured_pressures_n, force_targets, condition, contact_config)
            counts = {finger: sum(patch.finger == finger for patch in observation.patches)
                for finger in policy.active_fingers}
            solved = adapter.solve(data, observation, normal,
                command_rate_lower_rad_s=-rates, command_rate_upper_rad_s=rates,
                deflection_lower_rad=bias - contact_config.added_deflection_limit_rad,
                deflection_upper_rad=bias + contact_config.added_deflection_limit_rad,
                task_reference_velocity_rad_s=reference.task_reference_velocity_rad_s,
                task_weights=weights, contact_weights=[1. / counts[p.finger] for p in observation.patches],
                maximum_contact_residual_m_s=contact_config.maximum_contact_residual_m_s)
            row.update(solver=r27.json_value(solved), target_normal_velocity_m_s=normal,
                pressure_correction_m_s=correction, rigid_transport_normal_velocity_m_s=transport)
            if not solved.accepted:
                raise r27.ExperimentAbort("solver_rejected", solved.reason)
            data.ctrl[act] = solved.command_rad  # The only response-phase actuator writer.
            row["command_applied"] = True
            row["actual_ctrl_for_step"] = data.ctrl.copy()
            journal.step(data)
            cached = read_native_contact_sample(model, data)
            peak = max(peak, *cached.measured_pressures_n.values())
            row["native_dynamics_cache"] = {"timestamp_s": float(data.time),
                "step_start_s": row["before_time_s"], "step_end_s": float(data.time),
                "force_evaluation_scope": "Native dynamics cache for step; not recomputed at integrated qpos",
                "canonical_patches": r27.canonical_patches(cached.patches),
                "channel_forces_n": dict(cached.measured_pressures_n),
                "actuator_force_nm": data.actuator_force[act].copy(),
                "actuator_length_rad": data.actuator_length[act].copy(),
                "actuator_velocity_rad_s": data.actuator_velocity[act].copy(),
                "native_sample": journal.sample(data, phase,
                    force_cache_scope="Native dynamics cache during [step_start, step_end]; generalized qpos already integrated"),
                "motor_torque_prediction_residual_nm": data.actuator_force[act] - solved.predicted_motor_torque_nm}
            cache_abort = None
            try:
                r27.check_contact_sample(cached, policy, contact_config)
            except r27.ExperimentAbort as error:
                cache_abort = error
            mujoco.mj_forward(model, data)
            after = read_native_contact_sample(model, data)
            peak = max(peak, *after.measured_pressures_n.values())
            row["after_tactile"] = r27.json_value(after)
            row["after_canonical_patches"] = r27.canonical_patches(after.patches)
            row["after_robot_state"] = r27.robot_state(model, data, adapter.joint_names)
            # Query integrated robot pose without reusing any pre-step patch.
            after_motion = motion.prepare(data, ContactObservation((), after.measured_pressures_n, (), (), after.timestamp_s),
                desired_linear_velocity_world_m_s=(0., 0., 0.), joint_velocity_limits_rad_s=rates)
            row["after_wrist_pose"] = {"position_world_m": after_motion.wrist_position_world_m,
                "rotation_world_rows": after_motion.wrist_rotation_world_rows, "timestamp_s": float(data.time)}
            native.append(journal.sample(data, phase,
                force_cache_scope="Zero-time mj_forward at integrated qpos after the native step"))
            if cache_abort is not None:
                raise cache_abort
            r27.check_contact_sample(after, policy, contact_config)
        except (r27.ExperimentAbort, ValueError, FloatingPointError) as error:
            terminal = error.reason if isinstance(error, r27.ExperimentAbort) else "execution_error"
            row["abort"] = {"reason": terminal, "detail": getattr(error, "detail", str(error))}
            break
    positions = [np.array(row["motion_reference"]["wrist_position_world_m"]) for row in rows if "motion_reference" in row]
    after_positions = [np.array(row["after_wrist_pose"]["position_world_m"]) for row in rows if "after_wrist_pose" in row]
    actual_displacement = None if not positions else (after_positions[-1] if after_positions else positions[-1]) - positions[0]
    completed = terminal == "completed" and journal.integration_steps == round(config.duration_s / .002)
    return {"condition": condition, "terminal": terminal, "completed": completed,
        "native_steps": journal.integration_steps, "required_native_steps": round(config.duration_s / .002),
        "peak_channel_force_n": peak, "force_targets_n": force_targets,
        "initial_robot_state": rows[0]["before_robot_state"] if rows else None,
        "fixed_capture_deflection_rad": bias,
        "actual_wrist_displacement_m": actual_displacement,
        "declared_probe_distance_m": 0. if condition == "STATIC_HOLD" else config.probe_distance_m,
        "native_audit": journal.audit(), "rows": rows, "native": native,
        "task_success": None,
        "scope": "Development motion probe from replayed contact state; not 70mm held lift, placement or generalization"}


def write_gzip(path, rows):
    with gzip.open(path, "wt") as stream:
        json.dump(r27.json_value(rows), stream, allow_nan=False, separators=(",", ":"))
    return {"path": path.name, "sha256": r27.file_hash(path), "bytes": path.stat().st_size}


def snapshot():
    paths = [Path(__file__).resolve(), PROJECT / "scripts/run_coordinated_contact_experiment_r27.py",
        PROJECT / "scripts/public_ycb_evidence.py"]
    paths += sorted((PROJECT / "src/kuka_sharpa_grasp").glob("*.py"))
    return {p.relative_to(PROJECT).as_posix(): r27.file_hash(p) for p in paths}


def run(output, baseline, config):
    output.mkdir(parents=True, exist_ok=False)
    contact_config = r27.Configuration()
    before = snapshot()
    inputs = {name: r27.file_hash(baseline / name) for name in ("scene.xml", "native.json.gz", "trace.json.gz")}
    protocol = {"format": "r28_coordinated_5mm_motion_probe_v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config), "contact_configuration": asdict(contact_config),
        "conditions": CONDITIONS, "source_hashes_before": before, "input_hashes": inputs,
        "baseline": str(baseline), "mujoco_version": mujoco.__version__,
        "runtime_environment": {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname()},
        "controller_inputs": "Robot proprioception, declared servo model, external-hand tactile patches and fixed R27 force targets",
        "evaluation_only": "NativeJournal reads target pose/velocity/contact identities; these outputs never enter the controller",
        "branching": "One newly integrated 22s prefix, verified each native step, followed by three restored new integration-state branches",
        "force_targets": "Same R27 +/-0.5N channel perturbation in each branch; not a 1g support certificate",
        "force_gate": "12N abort, violating cached/refreshed samples retained; no force invariant",
        "task_success": None}
    r27.write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    shutil.copy2(baseline / "scene.xml", output / "scene.xml")
    summary = {"conditions": [], "terminal": "preparing", "task_success": None}
    replay_journal, replay_accepted = None, False
    try:
        model = load_evidence_model(baseline / "scene.xml")
        if abs(float(model.opt.timestep) - .002) > 1e-12:
            raise r27.ExperimentAbort("invalid_native_timestep")
        mujoco.mj_saveModel(model, str(output / "compiled_model.mjb"), None)
        protocol["compiled_model_sha256"] = r27.file_hash(output / "compiled_model.mjb")
        data = mujoco.MjData(model)
        reset_task(model, data, Scenario(3414, "ycb_can"))
        policy = r27.make_policy(contact_config)
        adapter = ContactExecutionAdapter(model, policy, arm_joint_names=r27.ARM_JOINTS)
        joints = tuple(model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
        with gzip.open(baseline / "native.json.gz", "rt") as stream:
            old_native = json.load(stream)
        with gzip.open(baseline / "trace.json.gz", "rt") as stream:
            old_trace = json.load(stream)
        replay_journal = NativeJournal(model, data)
        checked, window, prefix = replay(model, data, old_native, old_trace, joints, contact_config, replay_journal)
        replay_accepted = checked["accepted"]
        force_targets = r27.declared_targets(window, policy)
        mujoco.mj_forward(model, data)
        r27.check_contact_sample(read_native_contact_sample(model, data), policy, contact_config)
        spec, checkpoint = r27.integration_state(model, data)
        np.savez(output / "new_integration_checkpoint.npz", state=checkpoint, state_spec=spec)
        protocol["checkpoint_sha256"] = r27.file_hash(output / "new_integration_checkpoint.npz")
        for condition in CONDITIONS:
            result = run_condition(model, spec, checkpoint, adapter, force_targets, condition, config, contact_config)
            result["artifacts"] = {
                key: write_gzip(output / (condition.lower() + "_" + key + ".json.gz"), result.pop(key))
                for key in ("rows", "native")}
            r27.write_json(output / (condition.lower() + ".json"), result)
            summary["conditions"].append(result)
            print(json.dumps(r27.json_value(result)), flush=True)
        summary["terminal"] = "all_branches_recorded"
    except (r27.ExperimentAbort, ValueError, RuntimeError) as error:
        summary.update(terminal=error.reason if isinstance(error, r27.ExperimentAbort) else "runner_error",
            error=str(error), detail=getattr(error, "detail", None))
    if replay_journal is not None:
        r27.write_json(output / "replay_verification.json", {"accepted": replay_accepted,
            "terminal": "replay_matched" if replay_accepted else summary["terminal"],
            "detail": None if replay_accepted else summary.get("detail"), "checks": replay_journal.replay_checks})
        protocol["prefix_native"] = write_gzip(output / "prefix_native.json.gz", replay_journal.replay_native)
        protocol["prefix_native_audit"] = replay_journal.audit()
    protocol.update(source_hashes_after=snapshot(),
        input_hashes_after={name: r27.file_hash(baseline / name) for name in inputs},
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    protocol["source_changed_during_run"] = before != protocol["source_hashes_after"]
    protocol["inputs_changed_during_run"] = inputs != protocol["input_hashes_after"]
    summary["protocol"] = protocol
    summary["paired_comparison_eligible"] = (len(summary["conditions"]) == len(CONDITIONS)
        and tuple(row["condition"] for row in summary["conditions"]) == CONDITIONS
        and all(row["completed"] and row["native_steps"] == row["required_native_steps"]
            for row in summary["conditions"])
        and len({row["native_steps"] for row in summary["conditions"]}) == 1)
    r27.write_json(output / "protocol.json", protocol)
    r27.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", default=r27.BASELINE, type=Path)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.baseline.resolve(), Configuration())
    print(json.dumps({"terminal": result["terminal"], "paired_comparison_eligible": result["paired_comparison_eligible"]}))
    return int(not result["paired_comparison_eligible"] or result["protocol"]["source_changed_during_run"]
        or result["protocol"]["inputs_changed_during_run"])


if __name__ == "__main__":
    raise SystemExit(main())
