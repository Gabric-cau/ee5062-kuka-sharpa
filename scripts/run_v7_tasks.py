#!/usr/bin/env python3
"""Run physical V7 multi-object task episodes and write inspectable evidence.

Example::

    PYTHONPATH=src python scripts/run_v7_tasks.py --object tool_marker \
      --phase-timeout 2 --output results/v7_task_marker

The output is an experiment trace.  ``success`` is only true after a measured
lift and (when a slot is declared) a measured dock error; contact alone is not
reported as a successful grasp.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import TaskSpec, V7TaskExecutor
from kuka_sharpa_grasp.v7_episode import TeleopEpisodeRecorder
from kuka_sharpa_grasp.teleop import TeleopCommand


class TaskVideoRecorder:
    """Sample the live post-step state; never reconstruct or advance physics."""

    PANES = {
        "overview": (36, 576, 16, 736),
        "hand": (36, 411, 744, 1244),
        "wrist_rgb": (425, 605, 744, 984),
        "wrist_depth": (425, 605, 1000, 1240),
    }

    def __init__(self, model, output: Path, fps: int):
        import imageio.v2 as imageio
        from run_v7_workbench import WorkbenchRenderer

        self.imageio = imageio
        self.output = output / "video"
        self.output.mkdir()
        self.fps = fps
        self.renderer = WorkbenchRenderer(model)
        self.target = None
        self.writer = None
        self.next_sample = 0.
        self.previous_time = -1.
        self.reports = []
        self.report = None
        self.last_frame = None
        self.closed = False

    def _finish_episode(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        if self.last_frame is not None:
            self.imageio.imwrite(self.output / f"{self.target}_final.png", self.last_frame)
        self.reports.append(self.report)

    def __call__(self, data, snapshot):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        target, timestamp = snapshot["selected_target"], float(data.time)
        if target != self.target or timestamp < self.previous_time:
            self._finish_episode()
            self.target = target
            self.next_sample = 0.
            self.renderer.camera.reset()
            self.writer = self.imageio.get_writer(
                str(self.output / f"{target}.mp4"), fps=self.fps,
                codec="libx264", macro_block_size=None)
            self.report = {"object_id": target, "video": f"video/{target}.mp4",
                           "frames": 0, "fps": self.fps, "time_start_s": timestamp,
                           "time_end_s": timestamp, "peak_tactile_force_n": -1.,
                           "panes": {name: {"min_pixel_std": None, "max_pixel_std": 0.,
                                              "max_nonzero_fraction": 0.}
                                     for name in self.PANES}}
        self.previous_time = timestamp
        if timestamp + 1e-12 < self.next_sample:
            return
        frame = self.renderer.frame(data, snapshot)
        canvas = Image.fromarray(frame)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 735, 32), fill=(23, 29, 35))
        draw.text((18, 10), f"V7 physical task | {snapshot.get('phase', '-')} | t={timestamp:.2f}s | synchronized physics",
                  fill=(235, 240, 245), font=ImageFont.load_default())
        frame = np.asarray(canvas)
        self.writer.append_data(frame)
        self.last_frame = frame.copy()
        self.report["frames"] += 1
        self.report["time_end_s"] = timestamp
        if self.report["frames"] == 1:
            self.imageio.imwrite(self.output / f"{target}_first.png", frame)
        force = float(snapshot.get("tactile", {}).get("total_normal_force_n", 0.))
        if force > self.report["peak_tactile_force_n"]:
            self.report["peak_tactile_force_n"] = force
            self.imageio.imwrite(self.output / f"{target}_peak_tactile.png", frame)
        for name, (y0, y1, x0, x1) in self.PANES.items():
            pane = frame[y0:y1, x0:x1]
            stats = self.report["panes"][name]
            std = float(pane.std())
            stats["min_pixel_std"] = std if stats["min_pixel_std"] is None else min(stats["min_pixel_std"], std)
            stats["max_pixel_std"] = max(stats["max_pixel_std"], std)
            stats["max_nonzero_fraction"] = max(stats["max_nonzero_fraction"], float(np.mean(pane != 0)))
        self.next_sample += 1. / self.fps

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._finish_episode()
        finally:
            self.renderer.close()
        report = {"format": "kuka-sharpa-v7-live-task-video-r1",
                  "recording_mode": "synchronized_recorded_physics",
                  "reconstruction_from_recorded_physics": False,
                  "physics_advanced_by_recorder": False,
                  "task_success_computed_from_video": False,
                  "episodes": self.reports}
        (self.output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def source_snapshot() -> dict[str, str]:
    files = sorted((PROJECT / "src").rglob("*.py")) + [
        Path(__file__).resolve(), PROJECT / "scripts" / "run_v7_workbench.py"]
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if not path.name.startswith("._")}


def write_episode_artifacts(output, result, *, seed, shared_control, hashes, video):
    episode = output / result.object_id
    recorder = TeleopEpisodeRecorder(
        episode, seed=seed, mode="physical_task", shared_control=shared_control,
        initial_commands=[TeleopCommand.from_dict(item) for item in result.metrics.get("action_log", [])],
        metadata={"entry_point": "scripts/run_v7_tasks.py", "target_id": result.object_id,
                  "object_id": result.object_id, "slot_id": result.slot_id,
                  "physics_backend": "classic_mujoco_cpu", "mujoco": mujoco.__version__,
                  "source_sha256": hashes, "scene_manifest": "../scene_manifest.json",
                  "video": f"../video/{result.object_id}.mp4" if video else None,
                  "video_recording_mode": "synchronized_recorded_physics" if video else None})
    for snapshot in result.trace:
        recorder.record_snapshot(snapshot)
    recorder.finalize(
        task_success=result.success,
        task_success_note="Physical task executor decision; not inferred from video.",
        events=result.metrics.get("events", []),
        extra_metadata={"failure_reason": result.failure_reason,
                        "task_metrics": {key: value for key, value in result.metrics.items()
                                         if key not in {"action_log", "events"}}})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", dest="object_id", action="append", help="actionable object ID; repeat for a batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase-timeout", type=float, default=6.0)
    parser.add_argument("--hold-seconds", type=float, default=.1)
    parser.add_argument("--profile", choices=("baseline", "calibrated"), default="calibrated")
    parser.add_argument("--manual", action="store_true", help="disable tactile shared-control stops")
    parser.add_argument("--video", action="store_true", help="record synchronized live MuJoCo views for each tool")
    parser.add_argument("--fps", type=int, default=20, help="video sampling rate (1-60)")
    parser.add_argument("--output", type=Path, help="new output directory")
    args = parser.parse_args()
    if args.phase_timeout <= 0 or args.hold_seconds < 0:
        parser.error("phase timeout must be positive and hold seconds non-negative")
    if not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    if args.output is None:
        args.output = PROJECT / "results" / "v7_tasks" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True)
    config = default_v7_scene_config(args.seed)
    scene = build_v7_workbench_scene(args.output / "scene.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(scene))
    session = WorkbenchSession(model, config, shared_control=not args.manual)
    executor = V7TaskExecutor(session)
    requested = tuple(args.object_id) if args.object_id else None
    selected = executor.default_tasks(requested)
    tasks = [replace(item if args.profile == "calibrated" else TaskSpec(item.object_id, slot_id=item.slot_id),
                     phase_timeout_s=args.phase_timeout, hold_seconds=args.hold_seconds)
             for item in selected]
    hashes = source_snapshot()
    (args.output / "scene_manifest.json").write_text(
        json.dumps({"layout": config.manifest(), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                    "source_sha256": hashes}, indent=2) + "\n", encoding="utf-8")
    video = TaskVideoRecorder(model, args.output, args.fps) if args.video else None
    executor.step_callback = video
    try:
        results = executor.run_batch(tasks)
    finally:
        if video is not None:
            video.close()
    payload = {"format": "kuka-sharpa-v7-physical-task-r1", "seed": args.seed,
               "shared_control": session.shared_control,
               "profile": args.profile, "task_specs": [asdict(item) for item in tasks],
               "task_success": all(item.success for item in results),
               "results": [item.as_dict() for item in results]}
    for result in results:
        write_episode_artifacts(args.output, result, seed=args.seed,
                                shared_control=session.shared_control, hashes=hashes, video=args.video)
    (args.output / "task_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    all_events = [event for item in results for event in item.metrics.get("events", [])]
    # Keep events.json as the same list-of-records contract used by the
    # interactive recorder and batch evaluator. Target IDs belong in metadata.
    (args.output / "events.json").write_text(json.dumps(all_events, indent=2) + "\n", encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps({
        "entry_point": "scripts/run_v7_tasks.py", "seed": args.seed,
        "physics_backend": "classic_mujoco_cpu", "mujoco": mujoco.__version__,
        "shared_control": session.shared_control,
        "video": "video/metadata.json" if args.video else None,
        "video_recording_mode": "synchronized_recorded_physics" if args.video else None,
        "scene_manifest": "scene_manifest.json", "source_sha256": hashes,
        "target_ids": [item.object_id for item in results],
        "task_success": payload["task_success"],
        "task_success_note": "Success requires measured lift and dock tolerance; tactile contact alone is insufficient.",
    }, indent=2) + "\n", encoding="utf-8")
    summary = {"output": str(args.output), "task_success": payload["task_success"],
               "episodes": [{"object_id": item.object_id, "success": item.success,
                             "failure_reason": item.failure_reason,
                             "phases": [phase.status for phase in item.phases]}
                            for item in results]}
    print(json.dumps(summary, indent=2))
    return 0 if payload["task_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
