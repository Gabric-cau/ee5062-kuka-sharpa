#!/usr/bin/env python3
"""Run the marker task with a wrist RGB-D XY correction."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver, save_observation
from kuka_sharpa_grasp.object_registry import ToolObjectRegistry
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import V7TaskExecutor, calibrated_task_spec
from kuka_sharpa_grasp.v7_vision import estimate_upright_cylinder


def perturbation(seed):
    import random
    rng = random.Random(0x50620000 + int(seed))
    return rng.uniform(-.005, .005), rng.uniform(-.005, .005), rng.uniform(.8, 1.2), rng.uniform(.8, 1.2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    nominal = default_v7_scene_config(args.seed)
    target = nominal.registry.by_id("tool_marker")
    dx, dy, mass_factor, friction_factor = perturbation(args.seed)
    actual_target = replace(target, pose=(target.pose[0] + dx, target.pose[1] + dy, target.pose[2]),
                            mass_kg=target.mass_kg * mass_factor,
                            friction=tuple(value * friction_factor for value in target.friction))
    actual = replace(nominal, registry=ToolObjectRegistry(
        actual_target if item.object_id == target.object_id else item for item in nominal.registry.objects))
    scene = build_v7_workbench_scene(args.output / "scene.xml", config=actual)
    model = mujoco.MjModel.from_xml_path(str(scene))
    session = WorkbenchSession(model, nominal)
    task = calibrated_task_spec(target.object_id, slot_id=target.dock_slot)
    executor = V7TaskExecutor(session, reset_each_episode=False)
    grasp = np.asarray(target.pose) + np.asarray(task.grasp_offset_m)
    pregrasp = grasp - np.asarray(task.approach_axis_world) * task.approach_distance_m
    approach_trace = []
    approach, failure = executor._phase("APPROACH", pregrasp, task.pregrasp_closure,
                                        trace=approach_trace, timeout_s=task.phase_timeout_s)
    if failure is not None:
        payload = {"success": False, "failure_reason": failure, "approach": approach.as_dict()}
        (args.output / "result.json").write_text(json.dumps(payload, indent=2))
        return 2
    with CameraObserver(model, CameraConfig(width=320, height=240)) as camera:
        observation = camera.poll(session.data)
        if observation is None:
            raise RuntimeError("wrist RGB-D frame was not available")
        detection = estimate_upright_cylinder(observation, target)
        camera_export = save_observation(observation, args.output)
    if detection.position_m is None:
        payload = {"success": False, "failure_reason": detection.reason,
                   "detection": detection.as_dict()}
        (args.output / "result.json").write_text(json.dumps(payload, indent=2))
        return 2
    detected_registry = ToolObjectRegistry(
        replace(item, pose=detection.position_m) if item.object_id == target.object_id else item
        for item in nominal.registry.objects)
    session.config = replace(nominal, registry=detected_registry)
    result = executor.run(task)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "v7_obj_tool_marker")
    payload = {"success": result.success, "failure_reason": result.failure_reason,
               "task_result": result.as_dict(), "approach": approach.as_dict(),
               "detection": detection.as_dict(),
               "camera_export": camera_export,
               "physical_final_position_m": session.data.xpos[body].tolist(),
               "controller_input_policy": "wrist_rgbd_known_radius_fit; no live object qpos or body pose"}
    (args.output / "result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"success": result.success, "failure_reason": result.failure_reason,
                      "rgbd_fit_residual_m": detection.fit_residual_m,
                      "dock_error_m": result.metrics["dock_error_m"]}))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
