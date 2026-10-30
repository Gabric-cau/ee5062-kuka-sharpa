#!/usr/bin/env python3
"""Inspect an unchanged CPU grasp trial in the native MuJoCo viewer.

Space/P pauses, R replays identical inputs, C toggles contact points,
F toggles force vectors, and V switches the overview/hand camera.
The final physical state stays frozen until the native window is closed.
"""

from __future__ import annotations

import argparse
import csv
import time
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.visualization import (
    PlaybackControls, ReplayRequested, ViewerClosed, add_trial_arguments,
    configure_camera, configure_visuals, create_run_directory,
    load_trial_record, load_validation_events, presentation_manifest,
    resolve_trial, stage_at, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_trial_arguments(parser)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--validation-events", type=Path, help="JSON list of {wall_s, key}; same dispatcher as native keys.")
    parser.add_argument("--close-after", type=float, help="Validation-only wall seconds before closing the window.")
    args = parser.parse_args()
    if args.close_after is not None and args.close_after <= 0:
        parser.error("--close-after must be positive")
    record = load_trial_record(args)
    schedule = load_validation_events(args.validation_events)
    run_dir = create_run_directory("viewer_runs", args.run_dir)
    model = mujoco.MjModel.from_xml_path(str(build_scene(run_dir / "scene.xml", object_kind=args.object)))
    data = mujoco.MjData(model)
    rollout = GraspRollout(model, data)
    trial = resolve_trial(args, rollout, record)
    rollout.configure_object(trial.object_mass, trial.object_friction)
    rollout.reset(trial.seed, trial.object_pos)
    configure_visuals(model)
    manifest = presentation_manifest(trial, "native_mujoco_glfw_wgl", run_dir.name)
    manifest.update({"realtime": args.realtime, "validation_events": str(args.validation_events) if args.validation_events else None,
                     "close_after_wall_s": args.close_after})
    write_json(run_dir / "manifest.json", manifest)
    controls = PlaybackControls()
    replay_number = 0
    next_sync = 0.0
    schedule_index = 0
    completed = 0
    final_result = None
    with mujoco.viewer.launch_passive(model, data, key_callback=controls.dispatch_key,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            configure_camera(viewer.cam, model, data)
        controls.session_start = controls.wall_anchor = time.perf_counter()

        def service() -> float:
            nonlocal schedule_index, next_sync
            now = time.perf_counter()
            elapsed = now - controls.session_start
            while schedule_index < len(schedule) and schedule[schedule_index][0] <= elapsed:
                controls.dispatch_key(schedule[schedule_index][1], source="validation_schedule")
                schedule_index += 1
            actions = controls.apply_pending(float(data.time), now)
            if args.close_after is not None and elapsed >= args.close_after:
                viewer.close()
            if not viewer.is_running():
                raise ViewerClosed()
            if now >= next_sync or actions:
                if "camera" in actions or controls.detail:
                    with viewer.lock():
                        configure_camera(viewer.cam, model, data, controls.detail)
                total = rollout.reader.total(rollout.reader.read(data))
                state = final_result.status if final_result else stage_at(float(data.time))
                state = f"paused / {state}" if controls.paused else state
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  f"{trial.object_kind} / {trial.controller}\ntrial {replay_number + 1}\n{state}\ncontacts {total.contacts}",
                                  f"\nseed {trial.seed}\nt={data.time:.2f}s\n{total.normal:.1f} N"))
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(controls.contacts)
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(controls.forces)
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
                viewer.sync()
                if actions:
                    for event in controls.log[-len(actions):]:
                        event["native_contact_points"] = int(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
                        event["native_force_vectors"] = int(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
                next_sync = now + 1.0 / 60.0
            if controls.replay:
                raise ReplayRequested()
            return now

        def frame_callback(step: int, frame_data: mujoco.MjData, readings: dict) -> None:
            del step, readings
            while True:
                now = service()
                if not controls.paused and (not args.realtime or now >= controls.deadline(float(frame_data.time))):
                    return
                time.sleep(0.002)
        try:
            while True:
                final_result = None
                trial_dir = run_dir / f"replay_{replay_number:03d}"
                trial_dir.mkdir(exist_ok=False)
                write_json(trial_dir / "inputs.json", asdict(trial))
                try:
                    frame_callback(0, data, {})
                    result, rows = rollout.run_trace(**trial.rollout_arguments(), frame_callback=frame_callback)
                except ReplayRequested:
                    write_json(trial_dir / "result.json", {"status": "interrupted_for_replay", "sim_time": float(data.time),
                                                          "trial": asdict(trial)})
                else:
                    final_result = result
                    completed += 1
                    write_json(trial_dir / "result.json", {**asdict(result), "trial": asdict(trial)})
                    with (trial_dir / "trace.csv").open("x", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    controls.log.append({"action": "trial_complete", "replay_number": replay_number,
                                         "wall_s": time.perf_counter() - controls.session_start,
                                         "sim_time_before": float(data.time), "sim_time_after": float(data.time),
                                         "status": result.status, "failure_reason": result.failure_reason})
                    print(result, flush=True)
                    print("Trial ended. Window stays open; Space/P pause, R replay, C contacts, F forces, V camera.", flush=True)
                    try:
                        while True:
                            service()
                            time.sleep(0.005)
                    except ReplayRequested:
                        pass
                before = float(data.time)
                rollout.configure_object(trial.object_mass, trial.object_friction)
                rollout.reset(trial.seed, trial.object_pos)
                controls.begin_replay(before, time.perf_counter())
                replay_number += 1
                if controls.detail:
                    with viewer.lock():
                        configure_camera(viewer.cam, model, data, True)
        except (ViewerClosed, KeyboardInterrupt):
            interrupted_path = run_dir / f"replay_{replay_number:03d}" / "result.json"
            if not interrupted_path.exists():
                write_json(interrupted_path, {"status": "interrupted_window_closed", "sim_time": float(data.time),
                                              "trial": asdict(trial)})
            controls.log.append({"action": "window_closed", "wall_s": time.perf_counter() - controls.session_start,
                                 "sim_time_before": float(data.time), "sim_time_after": float(data.time),
                                 "completed_replays": completed})
        finally:
            write_json(run_dir / "viewer_events.json", {"events": controls.log, "completed_replays": completed})
    print(f"Viewer evidence: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
