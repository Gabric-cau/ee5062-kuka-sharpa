#!/usr/bin/env python3
"""Verify native-step GUI/replay equivalence and decode the recorded video."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


NUMERIC = ("time_s", "wrist_position", "wrist_quaternion", "arm_joint_position",
           "arm_joint_velocity", "arm_command", "finger_closure", "hand_joint_position")
STATE = ("mode", "selected_target", "status")


def read(directory, name):
    return json.loads((directory / name).read_text())


def numeric(value):
    if isinstance(value, dict):
        return np.asarray([value[key] for key in sorted(value)], dtype=float)
    return np.asarray(value, dtype=float)


def verify(gui, replay, output):
    if output.exists():
        raise FileExistsError(output)
    traces = [read(directory, "trace.json") for directory in (gui, replay)]
    metadata = [read(directory, "metadata.json") for directory in (gui, replay)]
    if not traces[0] or len(traces[0]) != len(traces[1]):
        raise ValueError("GUI and replay require equal, nonempty native traces")
    if any(item["task_success"] is not None for item in metadata):
        raise ValueError("GUI recording cannot supply an inferred task outcome")
    if metadata[0].get("input_provenance") != "automated_gui_input":
        raise ValueError("This verifier expects the labeled automated GUI acceptance run")
    events = read(gui, "operator_events.json")
    events = events["events"] if isinstance(events, dict) else events
    event_types = {event["event"] for event in events}
    down_keys = {event.get("key") for event in events if event["event"] == "KEY_DOWN"}
    fingers = {event.get("finger") for event in events if event["event"] == "FINGER_TARGET"}
    if not {"TARGET_SELECT", "MODE_CHANGE", "FINGER_TARGET", "KEY_DOWN", "KEY_UP",
            "FOCUS_LOST"} <= event_types or not {"w", "e", "]", "escape", "r", "d"} <= down_keys:
        raise ValueError("Automated GUI input did not cover every required interaction")
    if not {"thumb", "index"} <= fingers:
        raise ValueError("Individual-finger widget coverage is incomplete")
    errors = dict.fromkeys(NUMERIC, 0.)
    state_equal = True
    for first, second in zip(*traces):
        for key in NUMERIC:
            a, b = numeric(first[key]), numeric(second[key])
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Invalid replay field: {key}")
            errors[key] = max(errors[key], float(np.max(np.abs(a - b))))
        state_equal &= all(first.get(key) == second.get(key) for key in STATE)
    times = np.asarray([row["time_s"] for row in traces[0]])
    intervals = np.diff(times)
    if not len(intervals) or not np.allclose(intervals, .002, atol=1e-10, rtol=0):
        raise ValueError("Expected uninterrupted 2 ms physical observations")
    source_equal = metadata[0]["source_sha256"] == metadata[1]["source_sha256"]
    if not source_equal or not state_equal or any(error > 1e-10 for error in errors.values()):
        raise ValueError("GUI/replay source or state equivalence failed")
    videos = {}
    for directory in (gui, replay):
        path = directory / "workbench.mp4"
        frames, first, last, minimum_std = 0, None, None, float("inf")
        with imageio.get_reader(path) as reader:
            fps = float(reader.get_meta_data()["fps"])
            for frame in reader:
                if frame.shape != (720, 1280, 3):
                    raise ValueError("Unexpected video dimensions")
                minimum_std = min(minimum_std, float(frame.std()))
                if first is None:
                    first = frame
                last = frame
                frames += 1
        if frames != round(times[-1] * fps) or minimum_std < 5:
            raise ValueError("Incomplete or blank recording")
        difference = float(np.abs(last.astype(float) - first.astype(float)).mean())
        if difference < .1:
            raise ValueError("No visible scene movement in acceptance recording")
        Image.fromarray(first).save(directory / "acceptance_first.png")
        Image.fromarray(last).save(directory / "acceptance_last.png")
        videos[str(directory)] = {"frames": frames, "fps": fps,
            "minimum_frame_pixel_std": minimum_std,
            "mean_first_last_pixel_difference": difference,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {"status": "accepted_automated_gui_and_replay_equivalence",
        "human_operator": False, "task_success": None,
        "gui_run": str(gui), "replay_run": str(replay),
        "physics_frames": {"gui": len(traces[0]), "replay": len(traces[1])},
        "timestep_range_s": [float(intervals.min()), float(intervals.max())],
        "max_same_time_absolute_error": errors, "mode_target_status_identical": state_equal,
        "operator_event_count": len(events), "source_hashes_match_between_runs": source_equal,
        "required_input_coverage": True,
        "source_sha256": metadata[0]["source_sha256"], "videos": videos,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.gui, args.replay, args.output)
    print(json.dumps({"status": result["status"], "physics_frames": result["physics_frames"],
                      "operator_event_count": result["operator_event_count"]}))


if __name__ == "__main__":
    main()
