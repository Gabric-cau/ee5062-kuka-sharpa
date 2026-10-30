#!/usr/bin/env python3
"""Execute an obstacle route with MuJoCo arm/hand actuators and native contacts.

The hand is empty. Six free workbench objects remain physical. A hypothetical
carried-tool proxy is checked separately and is never called a real grasp.
Only reset initializes robot joints; execution writes ctrl and calls mj_step.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from kuka_sharpa_grasp.pick_place import RobotGravityCompensation
from kuka_sharpa_grasp.v7_motion import CollisionChecker, HeldToolProxy, RRTConnectPlanner
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from run_v7_obstacle_gate import START_Q, GOAL_Q, TRANSFER_BARRIER


HAND_Q = {
    "left_thumb_CMC_FE": 1.08, "left_thumb_MCP_FE": .15, "left_thumb_IP": .51,
    "left_index_MCP_FE": .57, "left_index_PIP": .67, "left_index_DIP": .43,
    "left_middle_MCP_FE": .65, "left_middle_PIP": .91, "left_middle_DIP": .52,
    "left_ring_MCP_FE": .65, "left_ring_PIP": .90, "left_ring_DIP": .52,
    "left_pinky_MCP_FE": .65, "left_pinky_PIP": .91, "left_pinky_DIP": .52,
}


def _name(model, obj, index):
    return mujoco.mj_id2name(model, obj, index) or f"id_{index}"


def execute(model, checker, path, output, *, speed=.12, stop_force=.5):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    arm_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)])
    data.qpos[arm_adrs] = path[0]
    for name, angle in HAND_Q.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = angle
    for aid in range(7, model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
    data.ctrl[:7] = path[0]
    mujoco.mj_forward(model, data)
    gravity = RobotGravityCompensation(model)
    robot = set(checker.robot_geom_ids)
    environment = set(checker.fixed_geom_ids) | set(checker.object_geom_ids)
    dt = float(model.opt.timestep)
    summary = {"steps": 0, "environment_contact_steps": 0, "self_contact_steps": 0,
               "peak_environment_force_n": 0., "peak_self_force_n": 0.,
               "minimum_actual_clearance_m": math.inf,
               "maximum_arm_tracking_error_rad": 0., "stopped": False,
               "maximum_arm_velocity_rad_s": 0., "maximum_arm_torque_nm": 0.,
               "clearance_sampling_period_s": .02,
               "clearance_scope": "actual robot versus all declared fixed/free geoms, sampled at 50 Hz",
               "contacts_sampling_period_s": dt,
               "initial_robot_qpos": path[0].tolist(), "scope": "physical_empty_hand_route",
               "live_object_qpos_writes": 0, "contacts_file": "contacts.jsonl.gz",
               "trace_file": "physical_trace.jsonl.gz"}
    output.mkdir()
    native_wrench = np.zeros(6)
    frame_step = max(1, round(.02 / dt))
    with gzip.open(output / "contacts.jsonl.gz", "wt") as contacts, gzip.open(
            output / "physical_trace.jsonl.gz", "wt") as trace:
        def step(command):
            gravity.apply(data, command)
            mujoco.mj_step(model, data)
            summary["steps"] += 1
            error = float(np.max(np.abs(data.qpos[arm_adrs] - command)))
            summary["maximum_arm_tracking_error_rad"] = max(summary["maximum_arm_tracking_error_rad"], error)
            summary["maximum_arm_velocity_rad_s"] = max(summary["maximum_arm_velocity_rad_s"],
                                                           float(np.max(np.abs(data.qvel[:7]))))
            summary["maximum_arm_torque_nm"] = max(summary["maximum_arm_torque_nm"],
                                                     float(np.max(np.abs(data.qfrc_actuator[:7]))))
            rows = []
            env_force = self_force = 0.
            unsafe = False
            for index in range(data.ncon):
                c = data.contact[index]
                a, b = int(c.geom1), int(c.geom2)
                mujoco.mj_contactForce(model, data, index, native_wrench)
                force = abs(float(native_wrench[0]))
                kind = ("robot_environment" if (a in robot) != (b in robot) else
                        "robot_self" if a in robot and b in robot else "environment_environment")
                rows.append({"geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                             "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, b),
                             "kind": kind, "distance_m": float(c.dist),
                             "force_local_n_nm": native_wrench.tolist()})
                if kind == "robot_environment":
                    env_force = max(env_force, force)
                    unsafe |= force > stop_force and c.dist < -.0002
                elif kind == "robot_self":
                    self_force = max(self_force, force)
            summary["environment_contact_steps"] += int(env_force > .01)
            summary["self_contact_steps"] += int(self_force > .01)
            summary["peak_environment_force_n"] = max(summary["peak_environment_force_n"], env_force)
            summary["peak_self_force_n"] = max(summary["peak_self_force_n"], self_force)
            contacts.write(json.dumps({"time_s": float(data.time), "contacts": rows}) + "\n")
            if summary["steps"] % frame_step == 0:
                clearance = min(float(mujoco.mj_geomDistance(model, data, a, b, 1., None))
                                for a in robot for b in environment)
                summary["minimum_actual_clearance_m"] = min(summary["minimum_actual_clearance_m"], clearance)
                trace.write(json.dumps({"time_s": float(data.time), "command": command.tolist(),
                                        "arm_qpos": data.qpos[arm_adrs].tolist(),
                                        "arm_qvel": data.qvel[:7].tolist(),
                                        "arm_torque_nm": data.qfrc_actuator[:7].tolist(),
                                        "actual_clearance_m": clearance,
                                        "tracking_error_rad": error}) + "\n")
            if unsafe:
                summary["stopped"] = True
            return not unsafe

        # A slow command follows every dense planner segment. Physics is
        # integrated continuously and arm velocities are never overwritten.
        for first, second in zip(path, path[1:]):
            steps = max(1, math.ceil(float(np.max(np.abs(second - first))) / (speed * dt)))
            for index in range(1, steps + 1):
                if not step(first + (second - first) * index / steps):
                    break
            if summary["stopped"]:
                break
        if not summary["stopped"]:
            for _ in range(round(.5 / dt)):
                if not step(path[-1]):
                    break
    summary["final_arm_qpos"] = data.qpos[arm_adrs].tolist()
    summary["goal_error_rad"] = float(np.max(np.abs(data.qpos[arm_adrs] - path[-1])))
    summary["elapsed_simulation_s"] = float(data.time)
    summary["success"] = bool(not summary["stopped"] and summary["environment_contact_steps"] == 0
                               and summary["goal_error_rad"] < .01)
    (output / "execution_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timebudget", type=float, default=6.)
    parser.add_argument("--speed", type=float, default=.12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.speed <= 0 or not math.isfinite(args.speed):
        parser.error("speed must be finite and positive")
    args.output.mkdir(parents=True)
    base = default_v7_scene_config(args.seed)
    cfg = replace(base, obstacles=base.obstacles + (TRANSFER_BARRIER,), version="v7_obstacle_execution_r1")
    scene = build_v7_workbench_scene(args.output / "scene.xml", config=cfg)
    model = mujoco.MjModel.from_xml_path(str(scene))
    checker = CollisionChecker(model, cfg, safety_margin=.004)
    proxy = HeldToolProxy((.014, .014, .047), object_id="hypothetical_screwdriver")
    result = RRTConnectPlanner(checker).plan(START_Q, GOAL_Q, HAND_Q,
                                           timebudget=args.timebudget, approach=False, heldtool=proxy)
    payload = {"scope": "physical_empty_hand_obstacle_route", "physical_carried_tool": False,
               "proxy_scope": "hypothetical_declared_geometry_at_wrist; no tool is grasped",
               "planner_status": result.status, "planner_clearance_m": result.clearance,
               "planned_states": len(result.path), "scene_manifest": cfg.manifest(),
               "source_sha256": {
                   path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [Path(__file__).resolve(), PROJECT / "scripts" / "run_v7_obstacle_gate.py",
                                PROJECT / "src" / "kuka_sharpa_grasp" / "v7_motion.py",
                                PROJECT / "src" / "kuka_sharpa_grasp" / "pick_place.py",
                                PROJECT / "src" / "kuka_sharpa_grasp" / "v7_scene.py"]}}
    if result.success:
        (args.output / "planned_path.json").write_text(json.dumps([q.tolist() for q in result.path]) + "\n")
        payload["planned_execution"] = execute(model, checker, result.path, args.output / "planned", speed=args.speed)
        direct = tuple(START_Q * (1-t) + GOAL_Q*t for t in np.linspace(0., 1., 301))
        payload["direct_execution"] = execute(model, checker, direct, args.output / "direct", speed=args.speed)
        payload["acceptance"] = bool(payload["planned_execution"]["success"]
                                     and payload["direct_execution"]["stopped"])
    else:
        payload["acceptance"] = False
    (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "scene_manifest"}, indent=2))
    return 0 if payload["acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
