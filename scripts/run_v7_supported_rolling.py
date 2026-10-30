#!/usr/bin/env python3
"""Paired, actuator-only table-supported marker rolling and no-override control.

The declared marker grasp lowers the wrist 6 mm onto table support. The wrist
command then stays fixed for 1.5 s while index MCP/PIP targets push the marker.
Simulation poses are recorded only for evaluation and never drive commands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from kuka_sharpa_grasp.v7_evaluator import (
    PoseSnapshot, RegraspMeasurement, RegraspSpec, RegraspTraceSample, evaluate_regrasp,
)
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import V7TaskExecutor, calibrated_task_spec

INDEX_PUSH = {"left_index_MCP_FE": .95, "left_index_PIP": 1.15}


def source_hashes():
    paths = [Path(__file__).resolve()] + sorted((PROJECT / "src" / "kuka_sharpa_grasp").glob("*.py"))
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def measure(session, snapshot):
    """Native contacts and relative poses are evaluation-only records."""
    model, data = session.model, session.data
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "v7_geom_tool_marker")
    force, fingers, support = 0., set(), False
    native, wrench = [], np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        mujoco.mj_contactForce(model, data, index, wrench)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                 for gid in (a, b)]
        native.append({"geom1": names[0], "geom2": names[1],
                       "distance_m": float(contact.dist), "position_m": contact.pos.tolist(),
                       "frame": contact.frame.tolist(), "force_local_n_nm": wrench.tolist()})
        if target not in (a, b):
            continue
        other = b if a == target else a
        normal = abs(float(wrench[0]))
        channel = session.tactile.geometry_channels.get(other)
        if channel:
            force += normal
            if channel != "palm" and normal > .01:
                fingers.add(channel)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        if (name == "table" or name.startswith("v7_rack")) and normal > .01:
            support = True
    thumb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_thumb_fingertip")
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_index_fingertip")
    aperture = float(np.linalg.norm(data.site_xpos[thumb] - data.site_xpos[index]))
    obj = snapshot["evaluation"]["tool_marker"]
    pose = PoseSnapshot(float(data.time), tuple(snapshot["wrist_position"]), tuple(obj["position"]),
                        tuple(snapshot["wrist_quaternion"]), tuple(obj["quaternion"]))
    sample = RegraspTraceSample(pose, force, aperture, len(fingers), support, pose.object_position[2])
    relpos, relquat = RegraspMeasurement._relative_pose(pose)
    record = {"hand_contact_force_n": force, "finger_aperture_m": aperture,
              "finger_contact_count": len(fingers), "finger_contact_channels": sorted(fingers),
              "support_contact": support, "object_height_m": pose.object_position[2],
              "relative_object_position_m": list(relpos), "relative_object_quaternion": list(relquat)}
    return sample, record, native


class Recorder:
    def __init__(self, session, output, *, video=False, fps=20):
        self.session, self.output = session, output
        self.phase = "INITIALIZE"
        self.trace = gzip.open(output / "physical_trace.jsonl.gz", "wt")
        self.contacts = gzip.open(output / "native_contacts.jsonl.gz", "wt")
        self.samples = []
        self.frames, self.next_frame, self.fps = 0, 0., fps
        self.renderer = self.writer = None
        self.peak_channel = {finger: 0. for finger in ("thumb", "index", "middle", "ring", "pinky", "palm")}
        if video:
            import imageio.v2 as imageio
            from run_v7_workbench import WorkbenchRenderer
            self.renderer = WorkbenchRenderer(session.model)
            self.writer = imageio.get_writer(str(output / "supported_rolling.mp4"), fps=fps,
                                            codec="libx264", macro_block_size=None)

    def record(self, snapshot):
        sample, measurement, contacts = measure(self.session, snapshot)
        snapshot["phase"], snapshot["supported_rolling"] = self.phase, measurement
        snapshot["joint_override_targets_rad"] = dict(self.session.finger_overrides)
        snapshot["joint_override_motor_commands_rad"] = dict(self.session.finger_override_commands)
        self.trace.write(json.dumps(snapshot) + "\n")
        self.contacts.write(json.dumps({"time_s": sample.pose.timestamp, "phase": self.phase,
                                       "contacts": contacts}) + "\n")
        for name, reading in snapshot["tactile"]["fingers"].items():
            self.peak_channel[name] = max(self.peak_channel.get(name, 0.), float(reading["normal_force_n"]))
        if self.phase in {"INDEX_PUSH", "NO_OVERRIDE_CONTROL"}:
            self.samples.append(sample)
        if self.writer is not None and sample.pose.timestamp + 1e-12 >= self.next_frame:
            from PIL import Image, ImageDraw
            canvas = Image.fromarray(self.renderer.frame(self.session.data, snapshot))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 738, 32), fill=(23, 29, 35))
            draw.text((18, 10), f"Table-supported rolling | {self.phase} | t={sample.pose.timestamp:.2f}s",
                      fill=(235, 240, 245))
            self.writer.append_data(np.asarray(canvas))
            self.frames += 1
            self.next_frame += 1. / self.fps

    def close(self):
        self.trace.close()
        self.contacts.close()
        if self.writer is not None:
            self.writer.close()
            self.renderer.close()


def run_arm(output, *, seed=0, override=True, lower_m=.006, seconds=1.5, video=False, fps=20):
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    config = default_v7_scene_config(seed)
    scene = build_v7_workbench_scene(output / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config)
    session.select_target("tool_marker")
    task = calibrated_task_spec("tool_marker")
    recorder = Recorder(session, output, video=video, fps=fps)
    original_step = session.step
    def recorded_step(command=None):
        snapshot = original_step(command)
        recorder.record(snapshot)
        return snapshot
    session.step = recorded_step
    executor = V7TaskExecutor(session)
    grasp = np.asarray(config.registry.by_id("tool_marker").pose) + task.grasp_offset_m
    approach = grasp - np.asarray(task.approach_axis_world) * task.approach_distance_m
    phases, failure, setup_trace = [], None, []
    commands = []
    initial_rolling = fixed_arm_command = None
    try:
        for name, position, closure in (("APPROACH", approach, 0.), ("ALIGN", grasp, 0.),
                                         ("CLOSE", grasp, task.grasp_closure)):
            recorder.phase = name
            start = float(session.data.time)
            done = None if name != "CLOSE" else lambda row, reached: reached and float(row["time_s"]) - start >= 2.
            phase, failure = executor._phase(name, position, closure, trace=setup_trace,
                                             timeout_s=6., done=done)
            phases.append(phase.as_dict())
            if failure:
                break
        if failure is None:
            recorder.phase = "POST_CLOSE_SETTLE"
            for _ in range(50):
                session.step()
            recorder.phase = "SUPPORT_CONTACT"
            phase, failure = executor._phase("SUPPORT_CONTACT", grasp - (0., 0., lower_m),
                                             task.grasp_closure, trace=setup_trace, timeout_s=6.)
            phases.append(phase.as_dict())
        if failure is None:
            recorder.phase = "SUPPORT_SETTLE"
            for _ in range(100):
                session.step()
            initial_rolling = session.snapshot()
            fixed_arm_command = session.arm_goal.copy()
            recorder.phase = "INDEX_PUSH" if override else "NO_OVERRIDE_CONTROL"
            recorder.record(initial_rolling)
            if override:
                for name, angle in INDEX_PUSH.items():
                    session.set_joint_target(name, angle)
                    commands.append({"time_s": float(session.data.time), "kind": "joint_target",
                                     "joint": name, "target_rad": angle})
            for _ in range(round(seconds / float(session.model.opt.timestep))):
                session.step()
                if session.stopped:
                    failure = session.status
                    break
    finally:
        recorder.close()
    evidence = None
    if recorder.samples:
        evidence = evaluate_regrasp(RegraspMeasurement(recorder.samples[0].pose,
                                   recorder.samples[-1].pose, tuple(recorder.samples))).as_dict()
        evidence["relative_rotation_deg"] = math.degrees(evidence["relative_rotation_rad"])
    peak = max(recorder.peak_channel.values())
    payload = {"format": "kuka-sharpa-v7-supported-rolling-r1", "seed": seed,
               "arm": "index_push" if override else "no_override_control", "scope": "table_supported_rolling",
               "in_air_regrasp_claim": False, "simulation_only": True,
               "task_spec": asdict(task), "lower_wrist_m": lower_m, "requested_rolling_seconds": seconds,
               "fixed_wrist_arm_goal": None if fixed_arm_command is None else fixed_arm_command.tolist(),
               "final_wrist_arm_goal": session.arm_goal.tolist(), "joint_target_commands": commands,
               "prepare_phases": phases, "failure_reason": failure, "regrasp_evidence": evidence,
               "channel_peak_force_n": recorder.peak_channel, "force_safe_12n": peak <= 12.,
               "success": bool(failure is None and evidence and evidence["accepted"] and peak <= 12.),
               "support_fraction": None if not recorder.samples else sum(s.support_contact for s in recorder.samples) / len(recorder.samples),
               "min_target_hand_force_n": None if not recorder.samples else min(s.hand_contact_force_n for s in recorder.samples),
               "collision_events": session.events, "initial_rolling_snapshot": initial_rolling,
               "sample_count": len(recorder.samples), "sampling_period_s": float(session.model.opt.timestep),
               "live_object_qpos_writes": 0, "source_sha256": hashes, "source_changed_during_run": source_hashes() != hashes,
               "scene_manifest": config.manifest(), "video": "supported_rolling.mp4" if video else None,
               "video_frames": recorder.frames, "evaluator_thresholds": asdict(RegraspSpec())}
    (output / "joint_targets.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "cartesian_commands.json").write_text(json.dumps(executor.actions, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.fps <= 60:
        parser.error("count must be positive and fps must be 1-60")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.count):
        pair = args.output / f"seed_{seed}"
        pair.mkdir()
        for override in (False, True):
            label = "index_push" if override else "no_override_control"
            row = run_arm(pair / label, seed=seed, override=override, video=args.video, fps=args.fps)
            compact = {key: row[key] for key in ("seed", "arm", "success", "failure_reason", "regrasp_evidence",
                                                  "force_safe_12n", "source_changed_during_run")}
            rows.append(compact)
            print(json.dumps(compact), flush=True)
    summary = {"format": "kuka-sharpa-v7-supported-rolling-paired-r1", "episodes": rows,
               "seeds": args.count, "index_push_successes": sum(r["success"] for r in rows if r["arm"] == "index_push"),
               "control_successes": sum(r["success"] for r in rows if r["arm"] == "no_override_control"),
               "scope": "table_supported_rolling; no in-air or hardware claim"}
    summary["paired_acceptance"] = bool(all(
        r["failure_reason"] is None and not r["source_changed_during_run"]
        and r["success"] == (r["arm"] == "index_push") for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["paired_acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
