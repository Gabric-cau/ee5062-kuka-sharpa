#!/usr/bin/env python3
"""Physical screwdriver pick, OMPL waypoint transfer over a guard, and docking.

Only the original session reset initializes object joints. Declared priors
define grasp localization and joint-path targets; native contact attribution
and object state also supply supervisory phase, safety and acceptance checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_calibrate import CalibrationContactReader, EVIDENCE_KEY, score_lift_evidence
from kuka_sharpa_grasp.v7_motion import CollisionChecker, HeldToolProxy, RRTConnectPlanner
from kuka_sharpa_grasp.v7_scene import WorkbenchObstacle, build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import PhaseRecord, TaskSpec
from run_v7_fixture_task import FixtureExecutor

GUARD = WorkbenchObstacle("carried_transfer_guard", (.630, -.140, .535),
                          (.035, .009, .155), (.80, .15, .10, 1.),
                          "Declared 310 mm vertical guard coupon; engineering-assumption obstacle.")


def hashes():
    files = [Path(__file__).resolve(), PROJECT / "scripts/run_v7_fixture_task.py",
             PROJECT / "scripts/run_v7_tasks.py", PROJECT / "scripts/run_v7_workbench.py"]
    files += sorted((PROJECT / "src").rglob("*.py"))
    return {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class IntegratedRecorder:
    def __init__(self, session, output, video=False):
        self.session, self.phase = session, "INITIALIZE"
        self.trace = gzip.open(output / "physical_trace.jsonl.gz", "wt")
        self.contacts = gzip.open(output / "native_contacts.jsonl.gz", "wt")
        self.target = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_GEOM, "v7_geom_tool_screwdriver")
        self.guard = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_GEOM, "v7_obstacle_carried_transfer_guard")
        self.peaks, self.target_force = {}, 0.
        self.reader = CalibrationContactReader(session, "tool_screwdriver")
        self.samples = self.environment_collision_samples = self.released_target_contact_samples = 0
        self.hand_self_events = self.non_hand_self_events = 0
        self.peak_hand_self_force = self.maximum_hand_self_penetration = 0.
        self.transfer_rows, self.transfer_violations, self.containment_failures = 0, 0, 0
        self.maximum_proxy_bound = np.zeros(3)
        self.minimum_guard_clearance = math.inf
        self.proxy = None
        self.video = None
        if video:
            from run_v7_tasks import TaskVideoRecorder
            self.video = TaskVideoRecorder(session.model, output, 20)

    def __call__(self, snapshot):
        session, data, model = self.session, self.session.data, self.session.model
        snapshot[EVIDENCE_KEY] = self.reader.read()
        self.samples += 1
        self.environment_collision_samples += bool(snapshot[EVIDENCE_KEY]["environment_collisions"])
        snapshot["arm_goal"] = session.arm_goal.tolist()
        snapshot["requested_finger_closure"] = dict(session.requested_closure)
        wrench, contacts, target_force = np.zeros(6), [], 0.
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            mujoco.mj_contactForce(model, data, i, wrench)
            normal = abs(float(wrench[0]))
            both_hand = a in session.hand_geoms and b in session.hand_geoms
            both_robot = a in session.robot_geoms and b in session.robot_geoms
            target_hand = self.target in (a, b) and ((a in session.hand_geoms) or (b in session.hand_geoms))
            kind = ("hand_self" if both_hand else "non_hand_robot_self" if both_robot else
                    "target_hand" if target_hand else "tool_environment" if self.target in (a, b) else
                    "robot_environment" if (a in session.robot_geoms) != (b in session.robot_geoms)
                    else "environment_environment")
            if both_hand and normal > .01:
                self.hand_self_events += 1
                self.peak_hand_self_force = max(self.peak_hand_self_force, normal)
                self.maximum_hand_self_penetration = max(self.maximum_hand_self_penetration, -float(c.dist))
            elif both_robot and normal > .01:
                self.non_hand_self_events += 1
            contacts.append({"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a) or f"geom_{a}",
                             "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b) or f"geom_{b}",
                             "kind": kind, "distance_m": float(c.dist), "force_local_n_nm": wrench.tolist()})
            if self.target in (a, b):
                other = b if a == self.target else a
                if other in session.hand_geoms:
                    target_force += normal
            if self.phase == "OMPL_TRANSFER" and normal > .01:
                robot_env = (a in session.robot_geoms) != (b in session.robot_geoms)
                held_contact = self.target in (a, b) and ((a in session.hand_geoms) or (b in session.hand_geoms))
                tool_environment = self.target in (a, b) and not held_contact
                if (robot_env and not held_contact) or tool_environment:
                    self.transfer_violations += 1
        self.target_force = target_force
        if self.phase == "VERIFY" and target_force > .05:
            self.released_target_contact_samples += 1
        for name, reading in snapshot["tactile"]["fingers"].items():
            self.peaks[name] = max(self.peaks.get(name, 0.), float(reading["normal_force_n"]))
        snapshot["phase"] = self.phase
        snapshot["target_hand_force_n"] = target_force
        if self.phase == "OMPL_TRANSFER":
            self.transfer_rows += 1
            gids = session.robot_geoms | {self.target}
            clearance = min(float(mujoco.mj_geomDistance(model, data, gid, self.guard, 1., None)) for gid in gids)
            self.minimum_guard_clearance = min(self.minimum_guard_clearance, clearance)
            snapshot["actual_guard_clearance_m"] = clearance
            if self.proxy is not None:
                wrist_r = data.xmat[session.wrist_body].reshape(3, 3)
                center = data.xpos[session.wrist_body] + wrist_r @ np.asarray(self.proxy.local_position)
                local_r = np.empty(9)
                mujoco.mju_quat2Mat(local_r, np.asarray(self.proxy.local_quaternion))
                proxy_r = wrist_r @ local_r.reshape(3, 3)
                error = proxy_r.T @ (data.geom_xpos[self.target] - center)
                axis = proxy_r.T @ data.geom_xmat[self.target].reshape(3, 3)[:, 2]
                bound = np.abs(error) + .100 * np.abs(axis) + .014
                self.maximum_proxy_bound = np.maximum(self.maximum_proxy_bound, bound)
                contained = bool(np.all(bound <= np.asarray(self.proxy.half_extents)))
                self.containment_failures += int(not contained)
                snapshot["physical_tool_inside_declared_proxy"] = contained
                snapshot["tool_proxy_bound_m"] = bound.tolist()
        self.trace.write(json.dumps(snapshot) + "\n")
        self.contacts.write(json.dumps({"time_s": float(data.time), "phase": self.phase, "contacts": contacts}) + "\n")
        if self.video:
            self.video(data, snapshot)

    def close(self):
        self.trace.close()
        self.contacts.close()
        if self.video:
            self.video.close()


class CarriedGuardExecutor(FixtureExecutor):
    def __init__(self, session, recorder):
        super().__init__(session, speed=.03, dock_xy=(.010, 0.), dock_clearance=.014)
        self.recorder, self.planner_records, self.joint_commands = recorder, [], []
        self.direct_blocked_states = 0
        self.held_lift_position = None
        self.extra_phase_records = []

    def _phase(self, name, target, closure, **kwargs):
        self.recorder.phase = name
        if name == "DOCK":
            roll = .10
            delta, goal_quaternion = np.empty(4), np.empty(4)
            mujoco.mju_axisAngle2Quat(delta, np.array([1., 0., 0.]), roll)
            mujoco.mju_mulQuat(goal_quaternion, delta, self.session.quaternion_goal)
            self.recorder.phase = "TOOL_ALIGN"
            self.current_phase = "TOOL_ALIGN"
            start = float(self.session.data.time)
            peak_force = 0.
            peak_contacts = 0
            failure = None
            steps = round(2.6 / self.session.model.opt.timestep)
            interval = round(.02 / self.session.model.opt.timestep)
            for index in range(steps):
                if index < round(2. / self.session.model.opt.timestep) and index % interval == 0:
                    self._command(self.session.position_goal.copy(), (.001, 0., 0.))
                row = self._step(closure, kwargs["trace"])
                peak_force = max(peak_force, row["tactile"]["total_normal_force_n"])
                peak_contacts = max(peak_contacts, row["tactile"]["contact_count"])
                if self.session.stopped:
                    failure = self.session.status
                    break
            dot = abs(float(np.dot(self.session.snapshot()["wrist_quaternion"], goal_quaternion)))
            if failure is None and 2*math.acos(min(1., dot)) > .008:
                failure = "TOOL_ALIGN_ORIENTATION_NOT_REACHED"
            alignment = PhaseRecord("TOOL_ALIGN", start, float(self.session.data.time),
                "COMPLETE" if failure is None else "FAILED", index+1, peak_force, peak_contacts,
                () if failure is None else (failure,))
            self.extra_phase_records.append(alignment)
            self.recorder.phase = name
            if failure:
                return alignment, failure
        if name != "TRANSFER":
            return super()._phase(name, target, closure, **kwargs)
        self.held_lift_position = self.session.snapshot()["evaluation"]["tool_screwdriver"]["position"]
        elevated_goal = np.asarray(target) + np.array([*self.dock_xy, 0.])
        elevated_goal[2] = max(elevated_goal[2], .900)
        return self._planned_transfer(elevated_goal, closure, kwargs["trace"])

    def _planned_transfer(self, goal, closure, trace):
        session, model = self.session, self.session.model
        start_time, start_q = float(session.data.time), session.data.qpos[:7].copy()
        start_pos = session.data.xpos[session.wrist_body].copy()
        quaternion = session.quaternion_goal.copy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        inverse = quaternion.copy()
        inverse[1:] *= -1
        # Declared tool centre offset, not a fitted live object pose.
        offset = rotation.T @ np.array([.065, -.037, -.210])
        proxy = HeldToolProxy((.045, .045, .145), tuple(offset), tuple(inverse), object_id="tool_screwdriver")
        self.recorder.proxy = proxy
        checker = CollisionChecker(model, session.config, safety_margin=.004)
        planner = RRTConnectPlanner(checker)
        hand = session.snapshot()["hand_joint_position"]
        goal_q = session.kinematics.solve(goal, quaternion, start_q)
        failure = None
        if goal_q is None:
            failure = "TRANSFER_GOAL_IK_FAILED"
        if failure is None:
            direct = tuple(start_q*(1-t) + goal_q*t for t in np.linspace(0., 1., 301))
            self.direct_blocked_states = sum(not checker.check(q, hand, "tool_screwdriver", approach=False,
                                                                heldtool=proxy).collision_free for q in direct)
        safe_height = max(float(start_pos[2]), float(goal[2])) + .170
        coarse = (np.array([*start_pos[:2], safe_height]), np.array([*goal[:2], safe_height]), goal)
        waypoints, origin = [], start_pos
        for endpoint in coarse:
            segments = max(1, math.ceil(float(np.linalg.norm(endpoint-origin))/.030))
            waypoints.extend(origin + (endpoint-origin)*i/segments for i in range(1, segments+1))
            origin = endpoint
        path, seed = [], start_q
        for index, point in enumerate(waypoints):
            if failure:
                break
            end = session.kinematics.solve(point, quaternion, seed)
            if end is None:
                failure = f"WAYPOINT_{index}_IK_FAILED"
                break
            # A small local joint-space box constrains each OMPL query around
            # nearby fixed-orientation IK states; model limits remain binding.
            planner.kinematics.lower = np.maximum(session.kinematics.lower, np.minimum(seed, end)-.020)
            planner.kinematics.upper = np.minimum(session.kinematics.upper, np.maximum(seed, end)+.020)
            planned = planner.plan(seed, end, hand, "tool_screwdriver", timebudget=6., approach=False, heldtool=proxy)
            entry = {"waypoint": point.tolist(), "status": planned.status, "clearance_m": planned.clearance,
                     "original_states": len(planned.path), "held_proxy": asdict(proxy)}
            self.planner_records.append(entry)
            if not planned.success:
                failure = f"WAYPOINT_{index}_{planned.status.upper()}"
                break
            # A dense validated shortcut removes unnecessary random excursions
            # while preserving the real OMPL exact route acceptance boundary.
            n = max(1, math.ceil(float(np.max(abs(end-seed))) / .003))
            shortcut = tuple(seed + (end-seed)*t for t in np.linspace(0., 1., n+1))
            shortcut_ok = all(checker.check(q, hand, "tool_screwdriver", approach=False,
                                            heldtool=proxy).collision_free for q in shortcut)
            chosen = shortcut if shortcut_ok else planned.path
            entry["shortcut_validated"] = shortcut_ok
            entry["executed_states"] = len(chosen)
            path.extend(chosen if not path else chosen[1:])
            seed = end
        steps, peak_force, peak_contacts = 0, 0., 0
        original = session.hand.config
        session.hand.config = replace(original, maximum_force_n=8., relief_rate=.8)
        self.recorder.phase = "OMPL_TRANSFER"
        try:
            for first, second in zip(path, path[1:]):
                if failure:
                    break
                count = max(1, math.ceil(float(np.max(abs(second-first))) / (.08 * model.opt.timestep)))
                for i in range(1, count+1):
                    command = first + (second-first)*i/count
                    session.arm_goal = command
                    row = self._step(closure, trace)
                    steps += 1
                    if i == count:
                        self.joint_commands.append({"time_s": float(session.data.time), "arm_goal": command.tolist()})
                    tactile = row["tactile"]
                    peak_force = max(peak_force, float(tactile["total_normal_force_n"]))
                    peak_contacts = max(peak_contacts, int(tactile["contact_count"]))
                    channel_peak = max(float(v["normal_force_n"]) for v in tactile["fingers"].values())
                    if session.stopped:
                        failure = session.status
                    elif channel_peak > 12.:
                        failure = "CHANNEL_FORCE_LIMIT_EXCEEDED"
                    elif self.recorder.transfer_violations:
                        failure = "NATIVE_TRANSFER_ENVIRONMENT_CONTACT"
                    elif self.recorder.non_hand_self_events:
                        failure = "NATIVE_NON_HAND_SELF_CONTACT"
                    elif self.recorder.containment_failures:
                        failure = "PHYSICAL_TOOL_OUTSIDE_PROXY"
                    elif self.recorder.target_force < .2:
                        failure = "TARGET_HAND_CONTACT_LOST"
                    if failure:
                        break
            if failure is None:
                for _ in range(round(.4/model.opt.timestep)):
                    row = self._step(closure, trace)
                    if (session.stopped or self.recorder.transfer_violations or self.recorder.containment_failures
                            or self.recorder.target_force < .2
                            or max(float(v["normal_force_n"]) for v in row["tactile"]["fingers"].values()) > 12.):
                        failure = "TRANSFER_SETTLE_VERIFICATION_FAILED"
                        break
                session.position_goal = goal.copy()
                session.quaternion_goal = quaternion
        finally:
            session.hand.config = original
        return PhaseRecord("OMPL_TRANSFER", start_time, float(session.data.time),
                           "COMPLETE" if failure is None else "FAILED", steps,
                           peak_force, peak_contacts, () if failure is None else (failure,)), failure


def run(output, seed=0, video=False):
    output.mkdir(parents=True, exist_ok=False)
    source = hashes()
    base = upright_tool_fixture(seed)
    config = replace(base, obstacles=base.obstacles+(GUARD,), version="v7_carried_guard_r2")
    scene = build_v7_workbench_scene(output / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config)
    session.hand.config = replace(session.hand.config, retention_minimum_force_n=3.)
    recorder = IntegratedRecorder(session, output, video)
    original_step = session.step
    def monitored(command=None):
        row = original_step(command)
        recorder(row)
        return row
    session.step = monitored
    executor = CarriedGuardExecutor(session, recorder)
    spec = TaskSpec("tool_screwdriver", slot_id="slot_screwdriver", grasp_offset_m=(-.065, .037, .210),
                    approach_axis_world=(0., 0., -1.), grasp_closure=.65, lift_height_m=.10,
                    phase_timeout_s=12., hold_seconds=.1, tactile_force_min_n=.15,
                    release_verify_seconds=3., dock_tolerance_m=.025)
    try:
        result = executor.run(spec)
    finally:
        recorder.close()
    force_safe = max(recorder.peaks.values(), default=0.) <= 12.
    lift = score_lift_evidence(result, "tool_screwdriver")
    body = session.model.body("v7_obj_tool_screwdriver").id
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(session.model, session.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    linear_speed, angular_speed = float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))
    source_changed = source != hashes()
    checks = {"task_complete": bool(result.success), "target_held_lift": lift.accepted,
              "channel_force_at_most_12n": force_safe, "transfer_environment_contact_free": recorder.transfer_violations == 0,
              "whole_task_environment_collision_free": recorder.environment_collision_samples == 0,
              "non_hand_self_contact_free": recorder.non_hand_self_events == 0,
              "held_tool_inside_proxy": recorder.containment_failures == 0,
              "straight_path_blocked": executor.direct_blocked_states > 0,
              "released_without_target_hand_contact": recorder.released_target_contact_samples == 0,
              "final_linear_speed_below_20mm_s": linear_speed < .02,
              "final_angular_speed_below_025rad_s": angular_speed < .25,
              "complete_poststep_contact_instrumentation": recorder.samples == len(result.trace)-1,
              "frozen_source": not source_changed}
    payload = {"scope": "physical_pick_carried_ompl_transfer_dock_release", "seed": seed,
               "success": all(checks.values()), "acceptance_checks": checks,
               "task_success": result.success, "failure_reason": result.failure_reason,
               "force_safe_12n": force_safe, "channel_peak_force_n": recorder.peaks,
               "dock_error_m": result.metrics["dock_error_m"], "phases": [p.as_dict() for p in sorted(
                   (*result.phases, *executor.extra_phase_records), key=lambda p:p.start_time_s)],
               "direct_blocked_states": executor.direct_blocked_states, "direct_states": 301,
               "planner_records": executor.planner_records, "native_transfer_collision_events": recorder.transfer_violations,
               "whole_task_environment_collision_samples": recorder.environment_collision_samples,
               "native_non_hand_self_contact_events": recorder.non_hand_self_events,
               "hand_self_contacts": {"events": recorder.hand_self_events, "peak_force_n": recorder.peak_hand_self_force,
                   "maximum_penetration_m": recorder.maximum_hand_self_penetration,
                   "scope": "recorded neighboring finger contact; not omitted from native logs"},
               "physical_tool_proxy_containment_failures": recorder.containment_failures,
               "maximum_physical_tool_proxy_bound_m": recorder.maximum_proxy_bound.tolist(),
               "minimum_actual_guard_clearance_m": recorder.minimum_guard_clearance if math.isfinite(recorder.minimum_guard_clearance) else None,
               "held_lift_position_m": executor.held_lift_position, "initial_object_position_m": result.metrics["initial_object_position_m"],
               "final_object_position_m": result.metrics["final_object_position_m"], "native_contacts_sampling_period_s": float(session.model.opt.timestep),
               "live_object_qpos_writes": 0, "source_sha256": source, "source_changed_during_run": source_changed,
               "native_contact_samples": recorder.samples, "lift_evidence": lift.as_dict(),
               "final_linear_speed_m_s": linear_speed, "final_angular_speed_rad_s": angular_speed,
               "controller": {"dock_xy_m": executor.dock_xy.tolist(), "dock_clearance_m": executor.dock_clearance,
                   "pre_dock_world_x_roll_rad": .10, "roll_duration_s": 2., "roll_settle_s": .6,
                   "development_seed": 0, "grasp_localization": "declared_map_prior",
                   "evaluation_object_pose_used_for_command_targets": False,
                   "object_state_and_native_contacts_used_for_supervisory_guards": True},
               "scene_manifest": config.manifest(), "task_spec": asdict(spec)}
    (output / "result.json").write_text(json.dumps(payload, indent=2)+"\n")
    (output / "joint_path_commands.json").write_text(json.dumps(executor.joint_commands)+"\n")
    (output / "action_log.json").write_text(json.dumps(result.metrics["action_log"])+"\n")
    (output / "events.json").write_text(json.dumps(session.events)+"\n")
    print(json.dumps({k:payload[k] for k in ("seed", "success", "failure_reason", "dock_error_m",
        "channel_peak_force_n", "acceptance_checks", "minimum_actual_guard_clearance_m")}, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    return 0 if run(args.output, args.seed, args.video)["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
