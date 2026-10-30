#!/usr/bin/env python3
"""Hash-bound offline vertical-load diagnostics for recorded CLOSE tail windows."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_v7_robustness import atomic_json
from public_ycb_evidence import load_evidence_model


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_array(path):
    decoder, buffer, position, opened = json.JSONDecoder(), "", 0, False
    with gzip.open(path, "rt") as stream:
        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if not opened:
                    if position == len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("Native stream is not a JSON array")
                    opened, position = True, position + 1
                    continue
                if position == len(buffer):
                    break
                if buffer[position] == "]":
                    if (buffer[position + 1:] + stream.read()).strip():
                        raise ValueError("Trailing native JSON content")
                    return
                try:
                    record, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                position = end
                yield record
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise ValueError("Incomplete native JSON array")
            buffer, position = buffer[position:] + chunk, 0


def vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("Invalid native " + label)
    return result


def contained(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Input must be a regular file within its evidence directory")
    return path


def force_record(sample, model, target_geom, geom_names, body_names, previous):
    totals = {name: np.zeros(3) for name in ("hand_world", "hand_normal", "hand_tangent", "table_world", "other_world")}
    fingers, counts = {}, Counter({"excluded_non_target_contact_observations": 0,
        "included_target_contact_observations": 0, "included_target_hand_contact_observations": 0,
        "included_target_table_contact_observations": 0, "included_target_other_contact_observations": 0,
        "included_zero_normal_contact_observations": 0})
    for contact in sample["native_contacts"]:
        ids = contact["geom_ids"]
        if len(ids) != 2 or any(type(item) is not int or not 0 <= item < model.ngeom for item in ids):
            raise ValueError("Invalid native geom IDs")
        if target_geom not in ids:
            counts["excluded_non_target_contact_observations"] += 1
            continue
        if ids[0] == ids[1]:
            raise ValueError("Target self-contact cannot define a force-on-target sign")
        counts["included_target_contact_observations"] += 1
        peer = ids[1] if ids[0] == target_geom else ids[0]
        sign = 1. if ids[1] == target_geom else -1.
        basis = np.asarray(contact["contact_frame_world_rows"], dtype=float)
        if (basis.shape != (3, 3) or not np.isfinite(basis).all()
                or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-8, rtol=0.)):
            raise ValueError("Invalid native contact frame")
        local = vector(contact["force_torque_contact_frame"], 6, "contact wrench")
        world = sign * basis.T @ local[:3]
        normal = sign * basis[0] * local[0]
        tangent = world - normal
        if abs(local[0]) <= 1e-12:
            counts["included_zero_normal_contact_observations"] += 1
        if body_names[peer].startswith("left_"):
            counts["included_target_hand_contact_observations"] += 1
            for key, value in (("hand_world", world), ("hand_normal", normal), ("hand_tangent", tangent)):
                totals[key] += value
            finger = body_names[peer].split("_")[1]
            row = fingers.setdefault(finger, {"world": np.zeros(3), "normal": np.zeros(3),
                "tangent": np.zeros(3), "normal_load_n": 0.})
            for key, value in (("world", world), ("normal", normal), ("tangent", tangent)):
                row[key] += value
            row["normal_load_n"] += float(local[0])
        elif geom_names[peer] == "table":
            counts["included_target_table_contact_observations"] += 1
            totals["table_world"] += world
        else:
            counts["included_target_other_contact_observations"] += 1
            totals["other_world"] += world
    spatial = vector(sample["target_spatial_velocity_world"], 6, "target spatial velocity")
    target_body = model.body("target").id
    # NativeAudit uses mjOBJ_BODY, whose velocity origin is the inertial COM.
    com_velocity = spatial[3:]
    acceleration = None if previous is None else (com_velocity - previous[1]) / (sample["timestamp"] - previous[0])
    mass = float(model.body_mass[target_body])
    balance = sum((totals[name] for name in ("hand_world", "table_world", "other_world")), np.zeros(3)) + mass * model.opt.gravity
    result = {"timestamp_s": float(sample["timestamp"]), "forces_xyz_n": {key: value.tolist() for key, value in totals.items()},
        "finger_forces": {finger: {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in row.items()} for finger, row in fingers.items()},
        "target_com_velocity_world_m_s": com_velocity.tolist(),
        "target_com_acceleration_fd_m_s2": None if acceleration is None else acceleration.tolist(),
        "contact_plus_gravity_residual_xyz_n": balance.tolist(),
        "contact_plus_gravity_minus_mass_acceleration_xyz_n": None if acceleration is None else (balance - mass * acceleration).tolist(),
        "contact_observation_counts": dict(counts)}
    return result, (float(sample["timestamp"]), com_velocity)


def summarize_window(records, start, count, ordinal, window_seconds):
    end = records[-1]["timestamp_s"]
    selected = [row for row in records if row["timestamp_s"] >= end - window_seconds - 1e-9]
    def stats(values):
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {"mean": array.mean(axis=0).tolist(), "minimum": array.min(axis=0).tolist(), "maximum": array.max(axis=0).tolist()}
    names = sorted({finger for row in selected for finger in row["finger_forces"]})
    fingers = {}
    for finger in names:
        fingers[finger] = {key: stats([row["finger_forces"].get(finger, {}).get(key, [0., 0., 0.]) for row in selected])
            for key in ("world", "normal", "tangent")}
        fingers[finger]["normal_load_n"] = stats([row["finger_forces"].get(finger, {}).get("normal_load_n", 0.) for row in selected])
    counts = Counter()
    for row in selected:
        counts.update(row["contact_observation_counts"])
    return {"close_segment_ordinal": ordinal, "close_segment_start_s": start,
        "close_segment_end_s": end, "close_segment_native_frames": count,
        "window_requested_seconds": window_seconds, "window_start_s": selected[0]["timestamp_s"],
        "window_end_s": end, "window_frames": len(selected), "window_boundary_inclusive": True,
        "force_xyz_n": {key: stats([row["forces_xyz_n"][key] for row in selected]) for key in selected[0]["forces_xyz_n"]},
        "finger_force_xyz_n": fingers,
        "com_velocity_world_m_s": stats([row["target_com_velocity_world_m_s"] for row in selected]),
        "com_acceleration_fd_m_s2": stats([row["target_com_acceleration_fd_m_s2"] for row in selected if row["target_com_acceleration_fd_m_s2"] is not None]),
        "contact_plus_gravity_residual_xyz_n": stats([row["contact_plus_gravity_residual_xyz_n"] for row in selected]),
        "contact_plus_gravity_minus_mass_acceleration_xyz_n": stats([row["contact_plus_gravity_minus_mass_acceleration_xyz_n"] for row in selected if row["contact_plus_gravity_minus_mass_acceleration_xyz_n"] is not None]),
        "contact_observation_counts": dict(counts), "frames": selected}


def analyze(directory, window_seconds=.25):
    directory = directory.resolve()
    if not np.isfinite(window_seconds) or window_seconds <= 0.:
        raise ValueError("A positive finite diagnostic window is required")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not summary["complete"] or len(summary["rows"]) != 1 or summary["protocol"]["source_changed_during_run"]:
        raise ValueError("Expected one completed case with unchanged recorded sources")
    row = summary["rows"][0]
    if row.get("runner_error"):
        raise ValueError("Cannot diagnose a case with a runner error")
    case = (directory / row["path"]).resolve()
    if not case.is_relative_to(directory):
        raise ValueError("Case escapes evidence directory")
    artifact = row["artifacts"]["native"]
    native_path, scene_path = contained(case, artifact["path"]), contained(case, "scene.xml")
    inputs = {"summary": summary_path, "native": native_path, "scene": scene_path,
        "script": Path(__file__).resolve(), "portable_model_loader": PROJECT / "scripts/public_ycb_evidence.py",
        "native_recorder": PROJECT / "scripts/evaluate_public_ycb.py"}
    before = {key: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for key, path in inputs.items()}
    if before["native"]["sha256"] != artifact["sha256"] or before["native"]["bytes"] != artifact["bytes"]:
        raise ValueError("Native compressed artifact differs from recorded hash/size")
    if row.get("scene_sha256_before_load") != before["scene"]["sha256"] or not row.get("scene_unchanged_since_load"):
        raise ValueError("Native scene must match its original before-load binding")
    model = load_evidence_model(scene_path)
    dt = float(model.opt.timestep)
    target_geom, target_body = model.geom("target_geom").id, model.body("target").id
    geom_names = [model.geom(g).name or "" for g in range(model.ngeom)]
    body_names = [model.body(int(model.geom_bodyid[g])).name or "" for g in range(model.ngeom)]
    records = deque(maxlen=int(np.ceil(window_seconds / dt)) + 3)
    windows, native_count, close_count, start, previous, last_time = [], 0, 0, None, None, 0.
    for sample in json_array(native_path):
        native_count += 1
        timestamp = float(sample["timestamp"])
        if (not np.isfinite(timestamp) or abs(timestamp - native_count * dt) > 1e-7
                or abs(timestamp - last_time - dt) > 1e-7):
            raise ValueError("Native samples must be contiguous from the first timestep")
        last_time = timestamp
        # Keep previous cached velocity across phase boundaries for the first derivative.
        value, previous = force_record(sample, model, target_geom, geom_names, body_names, previous)
        if sample["state"] == "CLOSE":
            if start is None:
                start = timestamp
            close_count += 1
            records.append(value)
        elif records:
            windows.append(summarize_window(records, start, close_count, len(windows) + 1, window_seconds))
            records.clear()
            close_count, start = 0, None
    if records:
        windows.append(summarize_window(records, start, close_count, len(windows) + 1, window_seconds))
    if not windows:
        raise ValueError("No recorded CLOSE segment")
    if any(digest(path) != before[key]["sha256"] for key, path in inputs.items()):
        raise ValueError("A diagnostic input changed during analysis")
    return {"format": "public_ycb_vertical_load_window_diagnostic_v1", "inputs": before,
        "case": row["path"], "native_samples": native_count, "timestep_s": dt,
        "target": {"body_id": target_body, "geom_id": target_geom, "mass_kg": float(model.body_mass[target_body]),
            "gravity_world_m_s2": model.opt.gravity.tolist(),
            "gravity_force_xyz_n": (model.body_mass[target_body] * model.opt.gravity).tolist(),
            "weight_n": float(model.body_mass[target_body] * np.linalg.norm(model.opt.gravity))},
        "field_definitions": {
            "force_on_target": "Let B=contact_frame_world_rows, f=force_torque_contact_frame[:3]. Force on geom2 is B.T@f; force on geom1 is its negative. Select the sign for target_geom.",
            "normal_tangent": "normal_on_target=sign*B[0]*f[0]; tangent=force_on_target-normal_on_target; all vector outputs use world x,y,z axes and newtons.",
            "hand_classification": "Peer geom body name starts with left_; finger is body-name component 1 after splitting at underscore. This is evaluation-only simulator identity.",
            "table_classification": "Peer geom name equals table; every remaining target contact is separately retained as other_world.",
            "counts": "Contact observations counted per native frame, not unique physical contacts; zero-normal contacts remain included. Contacts without target are excluded. Missing/invalid required target-contact geom IDs, frame or wrench reject analysis instead of silently dropping force.",
            "acceleration": "Finite difference of cached native target_spatial_velocity_world[3:6]; dt uses adjacent native timestamps. NativeAudit records mj_objectVelocity(mjOBJ_BODY, world orientation), which is already centered on the body inertial COM, not the regular body origin. No second COM-offset correction is applied. This is a sampled diagnostic, not an exact simultaneous qacc measurement.",
            "timing": "Native forces, contact bases, target xquat and spatial velocity are cached derived quantities after the original integration step. Observation timestamps label completed transitions; no forward, state update or additional simulation is performed here.",
            "residuals": "contact_plus_gravity=sum(hand_world,table_world,other_world)+mass*gravity; acceleration residual subtracts mass*finite-difference COM acceleration. Contact rotational torques are irrelevant to this linear-force balance and no moment balance is claimed."},
        "windows": windows, "inputs_unchanged_during_analysis": True,
        "missing_or_invalid_required_target_contact_records": 0,
        "velocity_origin_reference": "https://github.com/google-deepmind/mujoco/blob/3.3.0/src/engine/engine_support.c#L1163",
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "scope": "Offline sampled CLOSE tail-window force decomposition only. No task-success, grasp-stability, causal control diagnosis, physical-sensor validation or permission to change control/gates is established. Downward shear while table-supported can reverse during lifting; it alone does not prove wrong geometry. Evaluation-only target/peer simulator identities and state must not be supplied to the controller."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.directory, args.window_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    for window in result["windows"]:
        print(json.dumps({"segment": window["close_segment_ordinal"],
            "window_s": [window["window_start_s"], window["window_end_s"]], "frames": window["window_frames"],
            "hand_mean_xyz_n": window["force_xyz_n"]["hand_world"]["mean"],
            "table_mean_xyz_n": window["force_xyz_n"]["table_world"]["mean"],
            "hand_normal_mean_xyz_n": window["force_xyz_n"]["hand_normal"]["mean"],
            "hand_tangent_mean_xyz_n": window["force_xyz_n"]["hand_tangent"]["mean"]}))


if __name__ == "__main__":
    main()
