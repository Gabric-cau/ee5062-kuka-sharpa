#!/usr/bin/env python3
"""Sort all three tools in one continuously integrated cluttered workcell."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_v7_robustness import atomic_json, write_gzip
from evaluate_v7_tool_feedback import Instrumentation, episode_metrics
from run_v7_fixture_task import FixtureExecutor
from kuka_sharpa_grasp.v7_calibrate import CalibrationContactReader, EVIDENCE_KEY, score_lift_evidence
from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import TaskSpec, V7TaskExecutor, calibrated_task_spec


class ContinuousVideo:
    def __init__(self, model, output, fps):
        import imageio.v2 as imageio
        from run_v7_workbench import WorkbenchRenderer
        self.renderer = WorkbenchRenderer(model)
        self.writer = imageio.get_writer(str(output / "tool_sorting.mp4"), fps=fps,
                                         codec="libx264", macro_block_size=None)
        self.fps, self.next_sample, self.frames = fps, 0., 0

    def __call__(self, data, snapshot):
        if float(data.time) + 1e-12 >= self.next_sample:
            self.writer.append_data(self.renderer.frame(data, snapshot))
            self.next_sample += 1. / self.fps
            self.frames += 1

    def close(self):
        self.writer.close()
        self.renderer.close()


def run(output, seed=0, *, video=False, fps=20):
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((PROJECT / "src").rglob("*.py")) + [Path(__file__).resolve(),
        PROJECT / "scripts" / "run_v7_fixture_task.py", PROJECT / "scripts" / "run_v7_workbench.py",
        PROJECT / "scripts" / "evaluate_v7_tool_feedback.py", PROJECT / "scripts" / "evaluate_v7_robustness.py"]
    hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config = upright_tool_fixture(seed)
    scene = build_v7_workbench_scene(output / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config)
    recorder = ContinuousVideo(session.model, output, fps) if video else None
    results, rows, actions = [], [], []
    try:
        for object_id in ("tool_marker", "tool_screwdriver", "tool_wrench"):
            session.hand.engaged.clear()
            session.hand.hold_reference = None
            instrument = Instrumentation(session, object_id)

            def post_step(data, snapshot):
                instrument(data, snapshot)
                if recorder:
                    recorder(data, snapshot)

            if object_id == "tool_marker":
                session.hand.config = replace(session.hand.config, retention_minimum_force_n=1.)
                spec = replace(calibrated_task_spec(object_id, slot_id="slot_marker"),
                               release_verify_seconds=3., phase_timeout_s=12., dock_tolerance_m=.025)
                executor = V7TaskExecutor(session, reset_each_episode=False, step_callback=post_step)
            else:
                wrench = object_id == "tool_wrench"
                session.hand.config = replace(session.hand.config, retention_minimum_force_n=3.)
                spec = TaskSpec(object_id, slot_id=config.registry.by_id(object_id).dock_slot,
                    grasp_offset_m=(-.065, .037, .233 if wrench else .210),
                    approach_axis_world=(0., 0., -1.), grasp_closure=.75 if wrench else .65,
                    lift_height_m=.10, phase_timeout_s=12., hold_seconds=.1,
                    tactile_force_min_n=.15, release_verify_seconds=3., dock_tolerance_m=.025)
                executor = FixtureExecutor(session, dock_xy=(-.030, .010) if wrench else (.010, -.012),
                                           dock_clearance=.006 if wrench else .014)
                executor.reset_each_episode = False
                executor.step_callback = post_step
            result = executor.run(spec)
            lift = score_lift_evidence(result, object_id)
            measured = episode_metrics(result, spec, config)
            peak = max(row[EVIDENCE_KEY]["maximum_hand_channel_force_n"] for row in result.trace[1:])
            collision_free = not any(row[EVIDENCE_KEY]["environment_collisions"] for row in result.trace[1:])
            accepted = measured["strict_physical_success"] is True
            compact = result.as_dict()
            compact.pop("trace")
            compact["metrics"] = {k: v for k, v in compact["metrics"].items() if k not in {"events", "action_log"}}
            record = {"object_id": object_id, "task_spec": asdict(spec), "accepted": accepted,
                      "physical_metrics": measured,
                      "lift_evidence": lift.as_dict(), "peak_channel_force_n": peak,
                      "collision_free": collision_free, "task_result": compact}
            results.append(record)
            rows.extend(result.trace[1:])
            actions.extend(result.metrics["action_log"])
            atomic_json(output / "progress.json", results)
            print(json.dumps({"object_id": object_id, "accepted": accepted,
                              "failure_reason": result.failure_reason, "peak_channel_force_n": peak,
                              "dock_error_m": result.metrics["dock_error_m"]}), flush=True)
            if not accepted:
                break
    finally:
        if recorder:
            recorder.close()
    final_tools = {}
    for record in results:
        object_id = record["object_id"]
        body = session.model.body(f"v7_obj_{object_id}").id
        slot = next(item for item in config.slots if item.accepted_object_id == object_id)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(session.model, session.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        error = float(np.linalg.norm(session.data.xpos[body] - slot.pose))
        linear_speed, angular_speed = float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))
        final_tools[object_id] = {"position_m": session.data.xpos[body].tolist(),
            "dock_error_m": error, "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "still_docked": error <= .025 and linear_speed < .02 and angular_speed < .25}
    summary = {
        "format": "v7_continuous_tool_sorting_precise_r2", "seed": seed,
        "expected_tools": 3, "completed_tools": len(results),
        "success": len(results) == 3 and all(result["accepted"] for result in results)
                   and all(item["still_docked"] for item in final_tools.values()),
        "final_tools": final_tools,
        "results": results, "scene_manifest": config.manifest(), "source_sha256": hashes,
        "source_sha256_after": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "duration_s": float(session.data.time), "resets_between_tools": 0,
        "qpos_policy": "reset at construction only; no live arm/hand/object qpos writes",
        "video": {"path": "tool_sorting.mp4", "frames": recorder.frames, "fps": fps,
                  "synchronized_post_step_physics": True} if recorder else None,
        "artifacts": {"trace": write_gzip(output / "trace.json.gz", rows),
                      "actions": write_gzip(output / "action_log.json.gz", actions),
                      "events": write_gzip(output / "events.json.gz", session.events)},
    }
    summary["source_changed_during_run"] = summary["source_sha256"] != summary["source_sha256_after"]
    summary["success"] = summary["success"] and not summary["source_changed_during_run"]
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    if args.output is None:
        args.output = PROJECT / "results" / ("v7_tool_sorting_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    summary = run(args.output.resolve(), args.seed, video=args.video, fps=args.fps)
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
