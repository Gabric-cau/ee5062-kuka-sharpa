#!/usr/bin/env python3
"""Audit V7 Tk recordings and replay without inferring a human operator."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


TIMESTEP = .002
TOLERANCE = 1e-10
ARTIFACTS = ("metadata.json", "trace.json", "action_log.json", "events.json", "scene.xml", "workbench.mp4")
REQUIRED_TRACE = {"time_s", "status", "shared_control", "selected_target", "wrist_position",
                  "wrist_quaternion", "arm_command", "arm_joint_position", "arm_joint_velocity",
                  "hand_joint_position", "finger_closure", "task_success"}
EVENT_TYPES = {"KEY_DOWN", "KEY_UP", "FOCUS_LOST", "MODE_CHANGE", "TARGET_SELECT", "FINGER_TARGET"}
ATTESTATION_FORMAT = "kuka-sharpa-v7-operator-attestation"


def reject_nonfinite_constant(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite_constant)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(directory, *, gui=False):
    return {name: sha256(directory / name)
            for name in (*ARTIFACTS, *(("operator_events.json",) if gui else ()))}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite number: {label}")
    return float(value)


def compare(first, second, label, errors):
    """Compare every serialized observation, including nested tactile/object data."""
    if isinstance(first, dict):
        if not isinstance(second, dict) or first.keys() != second.keys():
            raise ValueError(f"Different object fields: {label}")
        for key in first:
            compare(first[key], second[key], f"{label}.{key}", errors)
    elif isinstance(first, list):
        if not isinstance(second, list) or len(first) != len(second):
            raise ValueError(f"Different array size: {label}")
        for index, (a, b) in enumerate(zip(first, second)):
            compare(a, b, f"{label}[{index}]", errors)
    elif isinstance(first, (int, float)) and not isinstance(first, bool):
        difference = abs(finite(first, label) - finite(second, label))
        field = label.split(".", 1)[-1].split("[", 1)[0]
        errors[field] = max(errors.get(field, 0.), difference)
        if difference > TOLERANCE:
            raise ValueError(f"Numeric replay mismatch: {label}")
    elif type(first) is not type(second) or first != second:
        raise ValueError(f"Replay state mismatch: {label}")


def validate_metadata(metadata, trace, *, gui):
    expected_mode = "interactive" if gui else "replay"
    if metadata.get("format") != "kuka-sharpa-v7-teleop-episode" or metadata.get("format_version") != 1:
        raise ValueError("Unsupported episode metadata")
    if metadata.get("mode") != expected_mode or metadata.get("task_success", "missing") is not None:
        raise ValueError("Expected interactive/replay recording with task_success null")
    if metadata.get("frames") != len(trace) or not trace:
        raise ValueError("Invalid metadata trace count")
    if not math.isclose(finite(metadata.get("duration_s"), "duration_s"), trace[-1]["time_s"], abs_tol=TOLERANCE):
        raise ValueError("Metadata duration differs from native trace")
    fps = metadata.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Invalid recording fps")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing source provenance hashes")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value) for value in sources.values()):
        raise ValueError("Invalid source provenance hash")
    return fps


def keyboard_command(held, timestamp):
    """Independent V7 default KeyboardTeleop contract, not controller execution."""
    return {"timestamp": timestamp,
        "translation_m": [.01 * (int("d" in held) - int("a" in held)),
                          .01 * (int("w" in held) - int("s" in held)),
                          .01 * (int("e" in held) - int("q" in held))],
        "rotation_rad": [.08 * (int("j" in held) - int("l" in held)),
                         .08 * (int("i" in held) - int("k" in held)),
                         .08 * (int("o" in held) - int("u" in held))],
        "grip_delta": .08 * (int("]" in held) - int("[" in held)),
        "buttons": sorted(held & {"space", "r", "escape"}),
        "selected_target": next((key[7:] for key in sorted(held) if key.startswith("target:")), None),
        "source": "keyboard"}


def validate_inputs(gui, replay, metadata, duration):
    payload = read_json(gui / "operator_events.json")
    if payload.get("format") != "kuka-sharpa-v7-operator-events" or payload.get("version") != 1:
        raise ValueError("Unsupported operator event format")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Recording requires nonempty Tk input events")
    previous = -1.
    for event in events:
        timestamp = finite(event.get("time_s"), "operator event time_s")
        if timestamp < 0 or timestamp < previous or timestamp > duration + TOLERANCE:
            raise ValueError("Operator event timestamps must be ordered within the recording")
        if not math.isclose(timestamp / TIMESTEP, round(timestamp / TIMESTEP), abs_tol=1e-7):
            raise ValueError("Operator event is not on a native step boundary")
        previous = timestamp
        kind = event.get("event")
        if kind not in EVENT_TYPES:
            raise ValueError("Unsupported Tk operator event")
        if kind in {"KEY_DOWN", "KEY_UP"} and (not isinstance(event.get("key"), str) or not event["key"]):
            raise ValueError("Invalid Tk key")
        if kind == "MODE_CHANGE" and event.get("mode") not in {"shared", "manual"}:
            raise ValueError("Invalid Tk mode")
        if kind == "TARGET_SELECT" and (not isinstance(event.get("target"), str) or not event["target"]):
            raise ValueError("Invalid target selection")
        if kind == "FINGER_TARGET":
            if event.get("finger") not in {"thumb", "index", "middle", "ring", "pinky"}:
                raise ValueError("Invalid finger selection")
            if not 0 <= finite(event.get("value"), "finger target") <= 1:
                raise ValueError("Invalid finger target")
    logs = [read_json(directory / "action_log.json") for directory in (gui, replay)]
    if any(log.get("format") != "kuka-sharpa-v7-teleop" for log in logs):
        raise ValueError("Unsupported command log format")
    if logs[0] != logs[1]:
        raise ValueError("Replay action log differs from GUI input")
    commands = logs[0].get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("Missing GUI commands")
    for item in metadata:
        if item.get("commands") != len(commands):
            raise ValueError("Metadata command count differs from action log")
    if metadata[0].get("operator_event_count") != len(events):
        raise ValueError("GUI operator event count is inconsistent")
    if metadata[1].get("operator_events_replayed") != len(events):
        raise ValueError("Replay did not declare the complete operator event stream")
    tick_steps = max(1, round((1. / metadata[0]["fps"]) / TIMESTEP))
    expected_count = math.ceil((duration - 1e-12) / (tick_steps * TIMESTEP))
    if len(commands) != expected_count:
        raise ValueError("Incomplete GUI tick command coverage")
    held, cursor = set(), 0

    def consume_event(event):
        kind = event["event"]
        if kind == "KEY_DOWN":
            held.add(event["key"])
            if event.get("held_keys") != sorted(held):
                raise ValueError("KEY_DOWN held-key evidence is inconsistent")
            if event["key"] == "escape":
                held.clear()
        elif kind == "KEY_UP":
            held.discard(event["key"])
            if event.get("held_keys") != sorted(held):
                raise ValueError("KEY_UP held-key evidence is inconsistent")
        elif kind == "FOCUS_LOST":
            if event.get("held_keys") != sorted(held) or not held:
                raise ValueError("FOCUS_LOST held-key evidence is inconsistent")
            held.clear()

    for index, command in enumerate(commands):
        timestamp = finite(command.get("timestamp"), "command timestamp")
        if not math.isclose(timestamp, index * tick_steps * TIMESTEP, abs_tol=TOLERANCE):
            raise ValueError("GUI command cadence differs from native tick cadence")
        while cursor < len(events) and events[cursor]["time_s"] <= timestamp + 1e-12:
            consume_event(events[cursor])
            cursor += 1
        compare(keyboard_command(held, timestamp), command, "command", {})
    for event in events[cursor:]:
        consume_event(event)
    return events, commands


def video_audit(path, *, expected_frames, fps):
    frames, minimum_std, first, last = 0, math.inf, None, None
    with imageio.get_reader(path) as reader:
        actual_fps = float(reader.get_meta_data()["fps"])
        if not math.isclose(actual_fps, fps, abs_tol=1e-6, rel_tol=0):
            raise ValueError("Video fps differs from recording metadata")
        for frame in reader:
            if frame.shape != (720, 1280, 3):
                raise ValueError("Unexpected video dimensions")
            minimum_std = min(minimum_std, float(frame.std()))
            if first is None:
                first = frame
            last = frame
            frames += 1
    if frames != expected_frames or minimum_std < 5:
        raise ValueError("Incomplete or blank recording video")
    difference = float(np.abs(first.astype(float) - last.astype(float)).mean())
    return {"decoded_frames": frames, "fps": actual_fps,
            "minimum_frame_pixel_std": minimum_std,
            "mean_first_last_pixel_difference": difference,
            "visible_first_last_change": difference >= .1}


def attestation_template(hashes):
    return {"format": ATTESTATION_FORMAT, "version": 1, "operator_name": "",
        "signed_at_utc": "", "statement": "",
        "artifact_sha256": hashes,
        "notice": "Self-report only. This does not technically authenticate identity or human input."}


def validate_attestation(path, hashes, provenance):
    if path is None:
        return {"status": "not_supplied", "technically_authenticated": False}
    if provenance != "unverified_gui_input":
        raise ValueError("An automated GUI recording cannot receive a human-operation attestation")
    item = read_json(path)
    if item.get("format") != ATTESTATION_FORMAT or item.get("version") != 1:
        raise ValueError("Unsupported attestation format")
    if any(not isinstance(item.get(key), str) or not item[key].strip()
           for key in ("operator_name", "signed_at_utc", "statement")):
        raise ValueError("Attestation requires an operator name, statement and signing time")
    try:
        signed = datetime.fromisoformat(item["signed_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid attestation signing time") from error
    if signed.utcoffset() is None:
        raise ValueError("Attestation signing time requires a timezone")
    if item.get("artifact_sha256") != hashes:
        raise ValueError("Attestation does not bind the audited GUI artifacts")
    return {"status": "self_attested_human_operation", "technically_authenticated": False,
            "operator_name": item["operator_name"], "statement": item["statement"],
            "signed_at_utc": item["signed_at_utc"], "path": str(Path(path).resolve()),
            "sha256": sha256(path)}


def verify(gui, replay, output, *, attestation=None, template_output=None):
    gui, replay, output = Path(gui).resolve(), Path(replay).resolve(), Path(output).resolve()
    if template_output is not None:
        template_output = Path(template_output).resolve()
    if output.exists() or (template_output is not None and Path(template_output).exists()):
        raise FileExistsError("Audit output and attestation template must be new files")
    if output == template_output:
        raise ValueError("Audit and template outputs must be different files")
    if any(path.is_relative_to(directory) for path in (output, template_output) if path is not None
           for directory in (gui, replay)):
        raise ValueError("Write the audit outside the immutable recording directories")
    hashes = {"gui": artifact_hashes(gui, gui=True), "replay": artifact_hashes(replay)}
    metadata = [read_json(directory / "metadata.json") for directory in (gui, replay)]
    traces = [read_json(directory / "trace.json") for directory in (gui, replay)]
    fps = [validate_metadata(item, trace, gui=index == 0)
           for index, (item, trace) in enumerate(zip(metadata, traces))]
    provenance = metadata[0].get("input_provenance")
    if provenance not in {"unverified_gui_input", "automated_gui_input"}:
        raise ValueError("Unsupported or missing GUI input provenance")
    for key in ("seed", "shared_control", "scene", "source_sha256", "mujoco", "physics_backend"):
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f"GUI/replay metadata differs: {key}")
    if fps[0] != fps[1] or len(traces[0]) != len(traces[1]):
        raise ValueError("GUI/replay fps or native sample counts differ")
    errors = {}
    for first, second in zip(*traces):
        if not REQUIRED_TRACE <= first.keys() or first.get("task_success", "missing") is not None:
            raise ValueError("Incomplete native snapshot or unsupported task-success claim")
        compare(first, second, "trace", errors)
    times = np.asarray([finite(row["time_s"], "trace.time_s") for row in traces[0]])
    if not np.allclose(times, np.arange(1, len(times) + 1) * TIMESTEP, atol=TOLERANCE, rtol=0):
        raise ValueError("Expected uninterrupted post-step observations starting at 2 ms")
    session_events = [read_json(directory / "events.json") for directory in (gui, replay)]
    if any(not isinstance(items, list) for items in session_events):
        raise ValueError("Session events must be arrays")
    compare(session_events[0], session_events[1], "session_events", {})
    if any(item.get("events") != len(session_events[0]) for item in metadata):
        raise ValueError("Metadata session event counts differ from events.json")
    events, commands = validate_inputs(gui, replay, metadata, float(times[-1]))
    cursor, expected_shared, selected = 0, metadata[0]["shared_control"], None
    for row in traces[0]:
        while cursor < len(events) and events[cursor]["time_s"] < row["time_s"] - 1e-12:
            event = events[cursor]
            if event["event"] == "MODE_CHANGE":
                expected_shared = event["mode"] == "shared"
            elif event["event"] == "TARGET_SELECT":
                selected = event["target"]
            cursor += 1
        if row["shared_control"] != expected_shared:
            raise ValueError("Tk mode events differ from physical snapshot control mode")
        if selected is not None and row["selected_target"] != selected:
            raise ValueError("Tk target events differ from physical snapshot target")
    replay_frames = math.floor((times[-1] - TIMESTEP + 1e-12) * fps[1]) + 1
    videos = {"gui": video_audit(gui / "workbench.mp4", expected_frames=len(commands), fps=fps[0]),
              "replay": video_audit(replay / "workbench.mp4", expected_frames=replay_frames, fps=fps[1])}
    declaration = validate_attestation(attestation, hashes["gui"], provenance)
    if hashes != {"gui": artifact_hashes(gui, gui=True), "replay": artifact_hashes(replay)}:
        raise ValueError("Recording artifacts changed during audit")
    result = {"format": "kuka-sharpa-v7-operator-recording-audit", "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_recording_and_replay_consistency", "input_provenance": provenance,
        "human_operator": False, "human_identity_or_input_technically_verified": False,
        "operator_attestation": declaration, "task_success": None,
        "gui_run": str(gui), "replay_run": str(replay),
        "physics_frames": len(times), "duration_s": float(times[-1]), "timestep_s": TIMESTEP,
        "all_serialized_trace_fields_compared": True, "max_same_time_absolute_error": errors,
        "commands_checked_against_raw_tk_keys": len(commands),
        "operator_event_count": len(events), "operator_event_types": dict(Counter(e["event"] for e in events)),
        "session_events_identical": True, "source_hashes_match_between_runs": True,
        "scene_file_hashes_equal": hashes["gui"]["scene.xml"] == hashes["replay"]["scene.xml"],
        "artifact_sha256": hashes, "source_sha256": metadata[0]["source_sha256"], "videos": videos,
        "video_claim": "Decoded nonblank videos, dimensions, fps and frame counts checked. GUI and replay sample at different physics phases; paired pixel equality is not asserted.",
        "limitations": ["Recorded files and source hash labels do not authenticate who generated the input.",
            "Optional attestation is a self-report bound to artifact hashes, not identity verification.",
            "This is a simulation recording audit, not task success, hardware force feedback or human usability evidence."],
        "verifier_sha256": sha256(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if template_output is not None:
        template = Path(template_output)
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(json.dumps(attestation_template(hashes["gui"]), indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attestation", type=Path, help="Optional operator self-report bound to GUI artifact hashes")
    parser.add_argument("--attestation-template", type=Path, help="Write a blank self-report template after a passing audit")
    args = parser.parse_args()
    result = verify(args.gui, args.replay, args.output, attestation=args.attestation,
                    template_output=args.attestation_template)
    print(json.dumps({key: result[key] for key in ("status", "input_provenance", "human_operator", "operator_attestation")}))


if __name__ == "__main__":
    main()
