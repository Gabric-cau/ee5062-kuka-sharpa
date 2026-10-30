#!/usr/bin/env python3
"""Finite full-task continuation with event-based arrival and physical release.

This reuses a development seed and a verified motor replay, not a new random
grasp generator. Target ground truth is isolated in the evaluation journal.
The explicitly instrumented virtual table supplies only a scalar support load.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "scripts")]
from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_load_targets import ContactLoadTargetConfig, allocate_contact_load_targets
from kuka_sharpa_grasp.contact_policy import ContactObservation
from kuka_sharpa_grasp.coordinated_admittance import solve_coordinated_admittance
from kuka_sharpa_grasp.coordinated_motion_reference import CoordinatedMotionReference
from kuka_sharpa_grasp.native_motion_evaluation import NativeMotionEvaluator
from kuka_sharpa_grasp.pick_place import Scenario, reset_task
from kuka_sharpa_grasp.strict_motion_monitor import StrictMotionMonitor
from kuka_sharpa_grasp.support import SupportConfig, TableSupportGate
from kuka_sharpa_grasp.vision_anchored_load_reference import load_vision_anchored_load_reference
from public_ycb_evidence import load_evidence_model
import run_coordinated_contact_experiment_r27 as r27
import run_coordinated_motion_experiment_r28 as r28

ZONE = np.array([.65, -.18, .38])
PRIOR = PROJECT / "results/r28_load_target_diagnostic_20261024/diagnostic.json"
MESH = PROJECT / "vendor/ycb/005_tomato_soup_can/005_tomato_soup_can/poisson/nontextured.stl"


@dataclass(frozen=True)
class Configuration:
    maximum_continuation_s: float = 180.
    maximum_linear_speed_m_s: float = .004
    arrival_position_tolerance_m: float = .0005
    arrival_linear_speed_m_s: float = .0005
    arrival_continuous_s: float = .15
    lift_wrist_distance_m: float = .085
    channel_abort_n: float = 12.
    release_rate_rad_s: float = .15
    force_target_margin: float = 1.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(name + " must be finite and positive")
        if self.channel_abort_n > 12. or self.maximum_linear_speed_m_s > .004 or self.release_rate_rad_s > .15:
            raise ValueError("Declared force, wrist speed and release-rate maxima must not be relaxed")
        if self.force_target_margin < 1.:
            raise ValueError("Static gravity margin must be at least one")
        if abs(self.maximum_continuation_s / .002 - round(self.maximum_continuation_s / .002)) > 1e-7:
            raise ValueError("Continuation duration must use integral 2ms steps")


class Arrival:
    def __init__(self):
        self.since = None

    def update(self, timestamp, error, speed, config):
        if error < config.arrival_position_tolerance_m and speed < config.arrival_linear_speed_m_s:
            if self.since is None:
                self.since = timestamp
            return timestamp - self.since + 1e-9 >= config.arrival_continuous_s
        self.since = None
        return False


class FullJournal(r28.NativeJournal):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.scratch = mujoco.MjData(model)
        self.evaluator = NativeMotionEvaluator(model)
        self.monitor = StrictMotionMonitor(initial_target_height_m=float(self.initial_target_qpos[2]),
            goal_position_world_m=ZONE)
        self.native, self.trace, self.cache_peaks = [], [], dict.fromkeys(r27.CONTACT_CHANNELS, 0.)
        self.phase, self.release_commanded = "REPLAY", False
        self.last_sample = None
        self.robot_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

    def step(self, data):
        super().step(data)
        cached = read_native_contact_sample(self.model, data)
        for name, value in cached.measured_pressures_n.items():
            self.cache_peaks[name] = max(self.cache_peaks[name], value)
        # Refresh a private full-state evaluator without changing the live
        # integrator's warm-start state or its replay force cache.
        spec, state = r27.integration_state(self.model, data)
        mujoco.mj_setState(self.model, self.scratch, state, spec)
        mujoco.mj_forward(self.model, self.scratch)
        live = self.scratch
        record = self.sample(live, self.phase,
            force_cache_scope="Private full-state mj_forward at integrated qpos; evaluation only")
        record["arm_joint_position"] = data.qpos[:7].copy().tolist()
        record["robot_joint_position_rad"] = {self.model.joint(j).name: float(data.qpos[self.model.jnt_qposadr[j]])
            for j in self.robot_joints}
        self.native.append(record)
        self.last_sample = self.evaluator.sample_after_forward(live, expected_timestamp_s=float(data.time),
            release_commanded=self.release_commanded, phase=self.phase)
        self.monitor.observe(self.last_sample)
        self.trace.append({"timestamp": float(data.time), "state": self.phase,
            "arm_joint_position": record["arm_joint_position"],
            "arm_joint_velocity_rad_s": data.qvel[:7].copy().tolist(),
            "oracle": {"release_commanded": self.release_commanded},
            "hand_segment_forces": dict(zip(r27.CONTACT_CHANNELS, self.last_sample.hand_channel_forces_n))})

    def assert_safety(self):
        report = self.monitor.result()
        if max(self.cache_peaks.values()) > 12. or not report["checks"]["all_channels_within_12n"]:
            raise r27.ExperimentAbort("whole_episode_force_limit")
        if not report["checks"]["no_robot_table_penetration"]:
            raise r27.ExperimentAbort("robot_table_penetration")


def solve_command(model, data, adapter, observation, reference, targets, bias, opening,
                  *, holding_without_contacts=False, release_rate_rad_s=.15):
    servos = adapter.servos
    finger_rate = release_rate_rad_s if opening else .05
    rates = np.r_[np.full(7, .025), np.full(len(servos) - 7, finger_rate)]
    task = np.asarray(reference.task_reference_velocity_rad_s).copy()
    lower, upper = bias - .05, bias + .05
    if opening:
        # Explicit release policy: same joint/rate/torque bounds; allow a
        # negative flexion servo deflection instead of keeping grasp preload.
        lower[7:] = -.25
        upper[7:] = .25
        task[7:] = np.clip(-data.ctrl[[s.actuator_id for s in servos]][7:] / .002, -finger_rate, finger_rate)
        jac, normal, weights = np.zeros((1, len(servos))), np.zeros(1), np.zeros(1)
    elif holding_without_contacts:
        task[7:] = 0.
        jac, normal, weights = np.zeros((1, len(servos))), np.zeros(1), np.zeros(1)
    else:
        jac = adapter.compression_jacobian(data, observation)
        correction = np.array([np.clip(.0005 * (targets[p.finger] - observation.measured_pressures_n[p.finger]),
            -.001, .001) for p in observation.patches])
        normal = np.asarray(reference.rigid_transport_normal_velocity_m_s) + correction
        counts = {f: sum(p.finger == f for p in observation.patches) for f in adapter.policy.active_fingers}
        weights = np.array([1. / counts[p.finger] for p in observation.patches])
    solved = solve_coordinated_admittance(jac, normal,
        actual_position_rad=data.qpos[[s.qpos_index for s in servos]],
        actual_velocity_rad_s=data.qvel[[s.dof_index for s in servos]],
        previous_command_rad=data.ctrl[[s.actuator_id for s in servos]], timestep_s=.002,
        joint_lower_rad=[s.lower_rad for s in servos], joint_upper_rad=[s.upper_rad for s in servos],
        command_rate_lower_rad_s=-rates, command_rate_upper_rad_s=rates,
        deflection_lower_rad=lower, deflection_upper_rad=upper,
        servo_kp_nm_rad=[s.kp_nm_rad for s in servos], servo_kd_nm_s_rad=[s.kd_nm_s_rad for s in servos],
        torque_lower_nm=[s.lower_torque_nm for s in servos], torque_upper_nm=[s.upper_torque_nm for s in servos],
        task_reference_velocity_rad_s=task, contact_weights=weights,
        task_weights=np.r_[np.full(7, .02 ** 2), np.full(len(servos) - 7, .001 ** 2)],
        maximum_contact_residual_m_s=.0001)
    return solved


def run_motion(model, data, adapter, journal, reference_prior, targets, config):
    motion = CoordinatedMotionReference(model, joint_names=adapter.joint_names)
    ids = [s.actuator_id for s in adapter.servos]
    bias = data.ctrl[ids].copy() - data.qpos[[s.qpos_index for s in adapter.servos]]
    rates = np.r_[np.full(7, .025), np.full(len(ids) - 7, .05)]
    empty = lambda sample: ContactObservation((), sample.measured_pressures_n, (), (), sample.timestamp_s)
    sample = read_native_contact_sample(model, data)
    first = motion.prepare(data, empty(sample), desired_linear_velocity_world_m_s=(0., 0., 0.),
        joint_velocity_limits_rad_s=rates)
    initial = np.asarray(first.wrist_position_world_m)
    rotation = np.asarray(first.wrist_rotation_world_rows)
    estimate = reference_prior.evaluate(initial, rotation, timestamp_s=float(data.time))
    xy = ZONE[:2] - np.asarray(estimate["estimated_origin_world_m"])[:2]
    offset = np.r_[xy, 0.]
    goals = {"PRELOAD": initial, "TRIAL_LIFT": initial + [0., 0., .005],
        "LIFT": initial + [0., 0., config.lift_wrist_distance_m],
        "TRANSFER": initial + offset + [0., 0., config.lift_wrist_distance_m],
        "PLACE": initial + offset + [0., 0., -.015]}
    phase, entered, started = "PRELOAD", float(data.time), float(data.time)
    desired = initial.copy()
    phase_start = initial.copy()
    goal = goals[phase].copy()
    arrival = Arrival()
    transitions, rows, failure = [{"state": phase, "timestamp_s": entered}], [], None
    support_config = SupportConfig(minimum_load_n=.7 * reference_prior.catalogue_mass_kg * 9.81,
        confirmation_seconds=.12)
    support_gate = TableSupportGate(support_config)
    last_position, last_time = initial.copy(), float(data.time)
    opening = False
    timeouts = {"PRELOAD": 8., "TRIAL_LIFT": 8., "LIFT": 32., "TRANSFER": 65.,
        "PLACE": 35., "SUPPORT_SETTLE": 4., "RELEASE": 18., "RETRACT": 60., "VERIFY": 5., "RECOVERY_LOWER": 40.}

    def enter(next_phase, current_position):
        nonlocal phase, entered, phase_start, goal, arrival, support_gate, opening
        phase, entered = next_phase, float(data.time)
        phase_start = current_position.copy()
        if phase in goals:
            goal = goals[phase].copy()
        elif phase == "RECOVERY_LOWER":
            goal = current_position + [0., 0., -.15]
        elif phase == "RETRACT":
            goal = current_position + [0., 0., .18]
        else:
            goal = current_position.copy()
        opening = phase in ("RELEASE", "RETRACT", "VERIFY")
        journal.release_commanded = opening
        if phase in ("PLACE", "RECOVERY_LOWER"):
            support_gate = TableSupportGate(support_config)
        arrival = Arrival()
        transitions.append({"state": phase, "timestamp_s": entered})
        print(json.dumps({"phase": phase, "time_s": entered, "failure": failure}), flush=True)

    for _ in range(round(config.maximum_continuation_s / .002)):
        row = {"timestamp_s": float(data.time), "state": phase, "command_applied": False}
        rows.append(row)
        try:
            journal.assert_safety()
            sample = read_native_contact_sample(model, data)
            if max(sample.measured_pressures_n.values()) > config.channel_abort_n:
                raise r27.ExperimentAbort("native_force_abort")
            recovering = phase == "RECOVERY_LOWER" or (phase == "SUPPORT_SETTLE" and failure is not None)
            if recovering:
                adapter.policy.validate_contacts(sample.patches, sample.measured_pressures_n,
                    controlled_joint_names=adapter.policy.controlled_joint_names, require_all_active=False,
                    timestamp_s=sample.timestamp_s)
                observation = empty(sample)
            else:
                observation = empty(sample) if opening else r27.check_contact_sample(sample, adapter.policy, r27.Configuration())
            pose = motion.prepare(data, empty(sample), desired_linear_velocity_world_m_s=(0., 0., 0.),
                joint_velocity_limits_rad_s=rates)
            position = np.asarray(pose.wrist_position_world_m)
            speed = np.linalg.norm(position - last_position) / max(.002, float(data.time) - last_time)
            last_position, last_time = position.copy(), float(data.time)
            support = table_support_load(model, data)
            row.update(table_sensor_load_n=support, before_tactile=r27.json_value(sample),
                before_robot_state=r27.robot_state(model, data, adapter.joint_names))
            row["support_gate"] = support_gate.update(float(data.time), support)
            fresh_support = support_gate.fresh_confirmation(float(data.time))
            if phase == "RELEASE" and not fresh_support:
                raise r27.ExperimentAbort("release_support_lost")
            arrived = arrival.update(float(data.time), np.linalg.norm(goal - position), speed, config)
            age = float(data.time) - entered
            if phase == "PRELOAD" and age >= 2. and arrived:
                enter("TRIAL_LIFT", position)
            elif phase == "TRIAL_LIFT" and arrived:
                enter("LIFT", position)
            elif phase == "LIFT" and arrived and age >= 1.:
                enter("TRANSFER", position)
            elif phase == "TRANSFER" and arrived:
                enter("PLACE", position)
            elif phase in ("PLACE", "RECOVERY_LOWER") and fresh_support:
                enter("SUPPORT_SETTLE", position)
            elif phase == "SUPPORT_SETTLE" and fresh_support and arrived:
                enter("RELEASE", position)
            elif phase == "RELEASE" and age >= .5 and max(sample.measured_pressures_n.values()) < .05:
                enter("RETRACT", position)
            elif phase == "RETRACT" and arrived:
                enter("VERIFY", position)
            elif phase == "VERIFY" and age >= 3.5:
                break
            elif age > timeouts[phase]:
                raise r27.ExperimentAbort("stage_timeout_" + phase)
            # A finite constant-speed reference followed by event-based pose
            # holding; transition is based on actual pose and velocity only.
            delta = goal - phase_start
            length = np.linalg.norm(delta)
            travel = config.maximum_linear_speed_m_s * max(0., float(data.time) - entered)
            desired = goal.copy() if length < 1e-12 else phase_start + delta * min(1., travel / length)
            velocity = np.zeros(3) if travel >= length else delta / length * config.maximum_linear_speed_m_s
            reference = motion.prepare(data, observation, desired_linear_velocity_world_m_s=velocity,
                desired_position_world_m=desired, desired_rotation_world=rotation,
                joint_velocity_limits_rad_s=rates, max_linear_speed_m_s=config.maximum_linear_speed_m_s)
            solved = solve_command(model, data, adapter, observation, reference, targets, bias,
                opening, holding_without_contacts=phase == "RECOVERY_LOWER" or (phase == "SUPPORT_SETTLE" and failure is not None),
                release_rate_rad_s=config.release_rate_rad_s if phase == "RELEASE" else .05)
            row.update(motion_reference=r27.json_value(reference), solver=r27.json_value(solved),
                actual_goal_world_m=goal, declared_trajectory_world_m=desired, opening=opening)
            if not solved.accepted:
                raise r27.ExperimentAbort("solver_rejected", solved.reason)
            data.ctrl[ids] = solved.command_rad
            row["command_applied"] = True
            row["actual_ctrl_for_step"] = data.ctrl.copy()
            journal.phase = phase
            journal.step(data)
            row["cached_motor_torque_nm"] = data.actuator_force[ids].copy()
            mujoco.mj_forward(model, data)
            journal.assert_safety()
            if phase == "RECOVERY_LOWER" or (phase == "SUPPORT_SETTLE" and failure is not None):
                recovery_sample = read_native_contact_sample(model, data)
                adapter.policy.validate_contacts(recovery_sample.patches, recovery_sample.measured_pressures_n,
                    controlled_joint_names=adapter.policy.controlled_joint_names, require_all_active=False,
                    timestamp_s=recovery_sample.timestamp_s)
            elif not opening:
                r27.check_contact_sample(read_native_contact_sample(model, data), adapter.policy, r27.Configuration())
        except (r27.ExperimentAbort, ValueError, FloatingPointError) as error:
            reason = error.reason if isinstance(error, r27.ExperimentAbort) else "execution_error"
            row["abort"] = {"reason": reason, "detail": getattr(error, "detail", str(error))}
            if failure is None:
                failure = reason
                if reason == "release_support_lost":
                    break
                # No opening in the air. Recovery uses a declared scalar
                # table load, with all original safety violations retained.
                enter("RECOVERY_LOWER", last_position)
                if "force" in reason or "penetration" in reason:
                    break
            else:
                failure += ";recovery:" + reason
                break
    else:
        failure = failure or "continuation_timeout"
    return {"transitions": transitions, "failure": failure, "rows": rows,
        "final_phase": phase, "duration_s": float(data.time) - started, "targets_n": targets}


def table_support_load(model, data):
    """Declared simulated instrumented-table signal; no object pose is returned."""
    target, table = model.geom("target_geom").id, model.geom("table").id
    wrench, load = np.zeros(6), 0.
    for index, contact in enumerate(data.contact[:data.ncon]):
        if set(map(int, contact.geom)) == {target, table}:
            mujoco.mj_contactForce(model, data, index, wrench)
            load += abs(float(wrench[0]))
    return load


def run(output, baseline):
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), PROJECT / "scripts/run_coordinated_motion_experiment_r28.py",
        PROJECT / "scripts/run_coordinated_contact_experiment_r27.py", PROJECT / "scripts/public_ycb_evidence.py"]
    sources += sorted((PROJECT / "src/kuka_sharpa_grasp").glob("*.py"))
    hashes = lambda: {p.relative_to(PROJECT).as_posix(): r27.file_hash(p) for p in sources}
    inputs = {str(p.relative_to(PROJECT)): r27.file_hash(p) for p in
        [baseline / n for n in ("scene.xml", "native.json.gz", "trace.json.gz", "result.json")] + [PRIOR, MESH]}
    config = Configuration()
    protocol = {"format": "r29_event_arrival_full_task_development_v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname(),
            "mujoco": mujoco.__version__}, "configuration": asdict(config), "source_hashes_before": hashes(),
        "input_hashes": inputs, "controller_boundary": "Robot kinematics, native tactile contacts, old RGBD pose/public priors, scalar simulated table support only",
        "scope": "Reused development seed, validated 22s command prefix, event-based continuation; no random-set performance claim"}
    r27.write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["source_hashes_before"]:
            archive.write(PROJECT / name, name)
    shutil.copy2(baseline / "scene.xml", output / "scene.xml")
    result, journal = {"success": False, "failure": "NOT_STARTED", "zone": ZONE}, None
    try:
        model = load_evidence_model(baseline / "scene.xml")
        mujoco.mj_saveModel(model, str(output / "compiled_model.mjb"), None)
        protocol["compiled_model_sha256"] = r27.file_hash(output / "compiled_model.mjb")
        data = mujoco.MjData(model)
        reset_task(model, data, Scenario(3414, "ycb_can"))
        adapter = ContactExecutionAdapter(model, r27.make_policy(r27.Configuration()), arm_joint_names=r27.ARM_JOINTS)
        journal = FullJournal(model, data)
        with gzip.open(baseline / "native.json.gz", "rt") as stream:
            old_native = json.load(stream)
        with gzip.open(baseline / "trace.json.gz", "rt") as stream:
            old_trace = json.load(stream)
        joints = tuple(model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
        checked, _, _ = r28.replay(model, data, old_native, old_trace, joints, r27.Configuration(), journal)
        r27.write_json(output / "replay_verification.json", checked)
        mujoco.mj_forward(model, data)
        ref, receipt = load_vision_anchored_load_reference(baseline / "result.json", baseline / "trace.json.gz", PRIOR,
            mesh_path=MESH, expected_hashes={"result": r27.file_hash(baseline / "result.json"),
                "trace": r27.file_hash(baseline / "trace.json.gz"), "prior": r27.file_hash(PRIOR), "mesh": r27.file_hash(MESH)})
        r27.write_json(output / "vision_prior_sources.json", receipt)
        wrist = model.body("link7").id
        est = ref.evaluate(data.xpos[wrist], data.xmat[wrist].reshape(3, 3), timestamp_s=float(data.time))
        sample = read_native_contact_sample(model, data)
        obs = r27.check_contact_sample(sample, adapter.policy, r27.Configuration())
        allocated = allocate_contact_load_targets(obs.as_allocator_patches(), est["estimated_com_world_m"],
            ref.catalogue_mass_kg, sample.measured_pressures_n,
            declared_friction_coefficient=ref.declared_friction_coefficient,
            config=ContactLoadTargetConfig(gravity_margin=config.force_target_margin),
            observation_timestamp_s=float(data.time), now_s=float(data.time))
        r27.write_json(output / "initial_static_targets.json", allocated)
        if not allocated["feasible"]:
            raise r27.ExperimentAbort("initial_static_targets_infeasible", allocated["reason"])
        continuation = run_motion(model, data, adapter, journal, ref, allocated["targets_n"], config)
        rows = continuation.pop("rows")
        protocol["controller_rows"] = r28.write_gzip(output / "controller.json.gz", rows)
        result.update(continuation)
        result["success"] = continuation["failure"] is None and continuation["final_phase"] == "VERIFY"
        result["failure"] = continuation["failure"]
    except (r27.ExperimentAbort, ValueError, RuntimeError, KeyError) as error:
        result["failure"] = error.reason if isinstance(error, r27.ExperimentAbort) else "runner_error"
        result["error"] = str(error)
        result["detail"] = getattr(error, "detail", None)
    if journal is not None:
        audit = journal.audit()
        audit["native_samples"] = len(journal.native)
        strict = journal.monitor.result(external_target_qpos_writes=audit["external_target_qpos_writes"],
            integration_steps=audit["integration_steps"], native_samples=len(journal.native), original_task_success=result["success"])
        result.update(native_audit=audit, strict_monitor=strict, native_cache_channel_peaks_n=journal.cache_peaks)
        result["success"] = bool(result["success"] and strict["task_success"]
            and max(journal.cache_peaks.values()) <= 12. and audit["external_target_qvel_writes"] == 0)
        artifacts = {"native": r28.write_gzip(output / "native.json.gz", journal.native),
            "trace": r28.write_gzip(output / "trace.json.gz", journal.trace)}
    else:
        artifacts = {}
    protocol.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), source_hashes_after=hashes(),
        input_hashes_after={name: r27.file_hash(PROJECT / name) for name in inputs})
    protocol["source_changed_during_run"] = protocol["source_hashes_before"] != protocol["source_hashes_after"]
    protocol["inputs_changed_during_run"] = inputs != protocol["input_hashes_after"]
    row = {"path": ".", "kind": "ycb_can", "success": result["success"], "failure": result["failure"],
        "runner_error": result.get("error"), "native_audit": result.get("native_audit"), "artifacts": artifacts}
    r27.write_json(output / "result.json", result)
    r27.write_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "rows": [row], "complete": True}
    r27.write_json(output / "summary.json", summary)
    print(json.dumps(r27.json_value(row)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=r27.BASELINE)
    args = parser.parse_args()
    summary = run(args.output.resolve(), args.baseline.resolve())
    raise SystemExit(int(summary["protocol"]["source_changed_during_run"] or summary["protocol"]["inputs_changed_during_run"]
        or not summary["rows"][0]["success"]))
