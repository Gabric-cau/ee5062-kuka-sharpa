#!/usr/bin/env python3
"""Independent offline audit of the three current V7 evidence collections.

No controller or existing success evaluator is imported. Native contact logs
are authoritative where present; attributed trace fields remain explicitly a
different evidence tier. This does not replay dynamics or prove qpos writes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
EVIDENCE = "calibration_evaluation"
PERIOD = .002


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def read_jsonl(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(a, b, atol=1e-8):
    return bool(np.allclose(a, b, atol=atol, rtol=0, equal_nan=False))


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def unit(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid quaternion")
    return q / np.linalg.norm(q)


def rotation(q):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, unit(q))
    return matrix.reshape(3, 3)


def angle(a, b):
    first, second = unit(a), unit(b)
    # atan2 is stable near zero, where acos(dot) amplifies rounding noise.
    inverse = first * np.array([1., -1., -1., -1.])
    relative = quat_multiply(inverse, second)
    return 2 * math.atan2(float(np.linalg.norm(relative[1:])), abs(float(relative[0])))


def relative_pose(row, object_id):
    obj = row["evaluation"][object_id]
    wrist = unit(row["wrist_quaternion"])
    inverse = wrist * np.array([1., -1., -1., -1.])
    return (rotation(wrist).T @ (np.asarray(obj["position"]) - row["wrist_position"]),
            quat_multiply(inverse, unit(obj["quaternion"])))


def time_audit(rows, *, allow_boundary_duplicates=False):
    if not rows:
        return {"accepted": False, "samples": 0, "reason": "EMPTY_TRACE"}
    times = np.asarray([row["time_s"] for row in rows], float)
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite timestamps")
    duplicate_indices = np.flatnonzero(abs(np.diff(times)) < 1e-12) + 1
    valid_duplicates = []
    physics = ("wrist_position", "wrist_quaternion", "arm_joint_position",
               "hand_joint_position", "evaluation", "tactile")
    for index in duplicate_indices:
        same = all(rows[index].get(key) == rows[index-1].get(key) for key in physics)
        boundary = rows[index].get("phase") != rows[index-1].get("phase")
        if allow_boundary_duplicates and same and boundary:
            valid_duplicates.append(int(index))
    invalid_duplicates = sorted(set(map(int, duplicate_indices)) - set(valid_duplicates))
    delta = np.diff(times)
    physical_delta = delta[abs(delta) >= 1e-12]
    accepted = (not invalid_duplicates and bool(len(physical_delta))
                and bool(np.all(abs(physical_delta-PERIOD) <= 1e-9)))
    return {"accepted": bool(accepted), "samples": len(rows), "first_time_s": float(times[0]),
            "last_time_s": float(times[-1]), "boundary_duplicate_indices": valid_duplicates,
            "invalid_duplicate_indices": invalid_duplicates,
            "minimum_positive_step_s": float(min(physical_delta)) if len(physical_delta) else None,
            "maximum_positive_step_s": float(max(physical_delta)) if len(physical_delta) else None,
            "non_2ms_steps": int(sum(abs(physical_delta-PERIOD) > 1e-9))}


def source_audit(declared, *, changed=False):
    mismatched, missing = [], []
    for relative, expected in declared.items():
        path = PROJECT / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            mismatched.append(relative)
    return {"accepted": bool(declared) and not changed and not mismatched and not missing,
            "declared_files": len(declared), "declared_source_changed_during_run": changed,
            "current_source_mismatches": mismatched, "missing_source_files": missing,
            "scope": "current files match recorded hashes; hashes alone do not prove executed code"}


class Geometry:
    def __init__(self, scene):
        root = ET.parse(scene).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            raise ValueError("Missing compiler")
        compiler.set("meshdir", str(PROJECT / "vendor/mujoco_menagerie"))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.names, self.robot, self.hand = {}, set(), {}
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name or f"geom_{gid}"
            body = self.model.body(int(self.model.geom_bodyid[gid])).name or ""
            self.names[name] = gid
            if body in {"base", *(f"link{i}" for i in range(1, 8))} or body.startswith("left_"):
                self.robot.add(name)
            if body.startswith("left_"):
                self.hand[name] = next((c for c in CHANNELS[:-1] if body.startswith(f"left_{c}_")), "palm")
        self.wrist = self.model.body("link7").id

    def contacts(self, native, target_id, *, support_mode="all_environment"):
        target = f"v7_geom_{target_id}"
        channels, fingers, target_contacts, target_force, support_force = dict.fromkeys(CHANNELS, 0.), set(), 0, 0., 0.
        collisions, transfer_violations, self_contacts = [], 0, []
        non_hand_self = 0
        rolling_fingers, rolling_support = set(), False
        for contact in native["contacts"]:
            a, b = contact["geom1"], contact["geom2"]
            if a not in self.names or b not in self.names:
                raise ValueError(f"Unknown native geometry {a}, {b}")
            wrench = np.asarray(contact["force_local_n_nm"], float)
            distance = float(contact["distance_m"])
            if wrench.shape != (6,) or not np.isfinite(wrench).all() or not math.isfinite(distance):
                raise ValueError("Invalid native contact")
            normal = abs(float(wrench[0]))
            one_hand = (a in self.hand) != (b in self.hand)
            one_robot = (a in self.robot) != (b in self.robot)
            if one_hand:
                hand, other = (a, b) if a in self.hand else (b, a)
                channel = self.hand[hand]
                channels[channel] += normal
                if other == target:
                    target_force += normal
                    target_contacts += int(normal > .05)
                    if channel != "palm" and normal > .05:
                        fingers.add(channel)
                    if channel != "palm" and normal > .01:
                        rolling_fingers.add(channel)
            if target in (a, b):
                other = b if a == target else a
                if other not in self.robot:
                    support_force += normal
                if (other == "table" or other.startswith("v7_rack")) and normal > .01:
                    rolling_support = True
            allowed = target in (a, b) and (a in self.hand or b in self.hand)
            if one_robot and not allowed and distance < -.002 and normal > .5:
                collisions.append({"geom1": a, "geom2": b})
            if a in self.hand and b in self.hand and normal > .01:
                self_contacts.append((normal, -distance))
            elif a in self.robot and b in self.robot and normal > .01:
                non_hand_self += 1
            if normal > .01 and ((one_robot and not allowed) or (target in (a, b) and not allowed)):
                transfer_violations += 1
        return {"target_hand_force_n": target_force, "target_hand_contacts": target_contacts,
                "target_contact_fingers": sorted(fingers), "target_support_force_n": support_force,
                "maximum_hand_channel_force_n": max(channels.values()), "hand_channel_forces_n": channels,
                "environment_collisions": collisions, "rolling_finger_count": len(rolling_fingers),
                "rolling_support": rolling_support, "transfer_violations": transfer_violations,
                "self_contacts": self_contacts, "non_hand_self": non_hand_self}

    def reconstruct(self, row, previous):
        # mj_step integrates qpos after computing xpos. Recorded xpos/contact
        # fields therefore correspond to the preceding poststep joint qpos.
        model, data = self.model, self.data
        data.qpos[:7] = previous["arm_joint_position"]
        for name, value in previous["hand_joint_position"].items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        for object_id, pose in row["evaluation"].items():
            body = model.body(f"v7_obj_{object_id}").id
            joint = int(model.body_jntadr[body])
            address = int(model.jnt_qposadr[joint])
            data.qpos[address:address+7] = (*pose["position"], *pose["quaternion"])
        mujoco.mj_kinematics(model, data)
        error = float(np.linalg.norm(data.xpos[self.wrist] - row["wrist_position"]))
        if error > 1e-7:
            raise ValueError(f"Recorded joint and wrist poses disagree: {error}")
        return error


def lift_metrics(rows, phases, target, initial_height):
    complete = all(EVIDENCE in row for row in rows)
    collisions = sum(bool(row[EVIDENCE]["environment_collisions"]) for row in rows if EVIDENCE in row)
    peaks = [float(row[EVIDENCE]["maximum_hand_channel_force_n"]) for row in rows if EVIDENCE in row]
    best, best_height, endpoint_height, endpoint_held, lift_complete = 0., 0., None, False, False
    for phase in phases:
        if phase["name"] != "LIFT":
            continue
        selected = [row for row in rows if phase["start_time_s"] < row["time_s"] <= phase["end_time_s"]+1e-9]
        lift_complete |= phase["status"] == "COMPLETE"
        start, origin, low, previous = None, None, math.inf, None
        for row in selected:
            e = row.get(EVIDENCE, {})
            height = row["evaluation"][target]["position"][2] - initial_height
            rel, _ = relative_pose(row, target)
            held = (e.get("target_hand_force_n", -1) >= .25 and e.get("target_hand_contacts", -1) >= 2
                    and len(e.get("target_contact_fingers", [])) >= 2
                    and e.get("target_support_force_n", math.inf) <= .10
                    and e.get("environment_collisions") == []
                    and e.get("maximum_hand_channel_force_n", math.inf) <= 30.)
            valid = held and height >= .04 and (previous is None or abs(row["time_s"]-previous-PERIOD) <= 1e-9)
            if valid and start is not None and np.linalg.norm(rel-origin) > .03:
                valid = False
            if valid:
                if start is None:
                    start, origin, low = row["time_s"], rel, height
                low = min(low, height)
                if row["time_s"]-start > best:
                    best, best_height = row["time_s"]-start, low
            else:
                start, origin, low = None, None, math.inf
            previous = row["time_s"]
            endpoint_height, endpoint_held = height, held
    accepted = (complete and lift_complete and best >= .10-1e-9 and endpoint_height is not None
                and endpoint_height >= .04 and endpoint_held and not collisions and max(peaks, default=math.inf) <= 30.)
    return {"accepted": bool(accepted), "sustained_lift_seconds": best, "sustained_height_change_m": best_height,
            "lift_endpoint_height_change_m": endpoint_height, "lift_endpoint_target_held": endpoint_held,
            "maximum_hand_channel_force_n": max(peaks, default=None), "collision_samples": collisions}


def release_metrics(rows, phases, task, slot, *, finite_difference=False):
    verify = next((p for p in phases if p["name"] == "VERIFY"), None)
    release = next((p for p in phases if p["name"] == "RELEASE"), None)
    if verify is None or release is None:
        return {"accepted": False, "stable_seconds": 0., "reason": "RELEASE_OR_VERIFY_NOT_REACHED"}
    selected = [r for r in rows if verify["start_time_s"] < r["time_s"] <= verify["end_time_s"]+1e-9]
    preceding = [r for r in rows if r["time_s"] <= verify["start_time_s"]+1e-9]
    previous = preceding[-1] if finite_difference and preceding else None
    start, best, max_linear, max_angular = None, 0., 0., 0.
    for row in selected:
        e = row[EVIDENCE]
        pose = row["evaluation"][task["object_id"]]
        if finite_difference:
            if previous is None:
                previous = row
                continue
            dt = row["time_s"]-previous["time_s"]
            last = previous["evaluation"][task["object_id"]]
            linear = float(np.linalg.norm(np.asarray(pose["position"])-last["position"]) / dt)
            angular = angle(last["quaternion"], pose["quaternion"]) / dt
        else:
            linear, angular = float(e["target_linear_speed_m_s"]), float(e["target_angular_speed_rad_s"])
        max_linear, max_angular = max(max_linear, linear), max(max_angular, angular)
        valid = (np.linalg.norm(np.asarray(pose["position"])-slot) <= task["dock_tolerance_m"]
                 and linear < .02 and angular < .25 and e["target_hand_force_n"] <= .05
                 and max(row["finger_closure"].values()) <= .03
                 and (previous is None or abs(row["time_s"]-previous["time_s"]-PERIOD) <= 1e-9))
        if valid:
            start = row["time_s"] if start is None else start
            best = max(best, row["time_s"]-start)
        else:
            start = None
        previous = row
    ending = 0. if start is None or not selected else selected[-1]["time_s"]-start
    required = task["release_verify_seconds"]
    accepted = release["status"] == verify["status"] == "COMPLETE" and ending >= required-1e-9
    return {"accepted": bool(accepted), "stable_seconds": ending, "maximum_stable_seconds": best,
            "maximum_linear_speed_m_s": max_linear, "maximum_angular_speed_rad_s": max_angular,
            "speed_source": "pose_finite_difference_2ms" if finite_difference else "recorded_native_body_velocity",
            "required_covered_seconds": required, "native_velocity_independently_verified": False}


def verify_artifacts(directory, artifacts):
    mismatches = []
    for key, record in artifacts.items():
        path = directory / record["path"]
        if (not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            mismatches.append(key)
    return {"accepted": not mismatches and bool(artifacts), "mismatches": mismatches}


def audit_three_tools(directory):
    summary, protocol = read_json(directory / "summary.json"), read_json(directory / "protocol.json")
    records, paired_scenes = [], {}
    for item in summary["results"]:
        case = directory / item["path"]
        metadata, result = read_json(case / "metadata.json"), read_json(case / "task_result.json")
        trace = read_json(case / "trace.json.gz")
        rows = trace[1:] if trace and trace[0]["time_s"] == 0 else trace
        task = protocol["frozen_task_specs"][item["object_id"]]
        manifest = read_json(case / "scene_manifest.json")
        slot = next(s["pose"] for s in manifest["slots"] if s["slot_id"] == task["slot_id"])
        lift = lift_metrics(rows, result["phases"], item["object_id"], result["metrics"]["initial_object_position_m"][2])
        released = release_metrics(rows, result["phases"], task, slot)
        peak = max(r[EVIDENCE]["maximum_hand_channel_force_n"] for r in rows)
        collision_free = not any(r[EVIDENCE]["environment_collisions"] for r in rows)
        strict = bool(result["success"] and lift["accepted"] and released["accepted"] and peak <= 12 and collision_free)
        raw = {"strict_physical_success": strict, "target_held_lift": lift["accepted"],
               "stable_released_in_slot": released["accepted"], "peak_hand_channel_force_n": peak,
               "collision_free": collision_free, "force_safe": peak <= 12}
        mismatches = [k for k, value in raw.items() if (not close(value, item["metrics"][k])
                       if isinstance(value, float) else value != item["metrics"][k])]
        for key, value in lift.items():
            if not close(value, item["metrics"]["lift_evidence"][key]):
                mismatches.append(f"lift:{key}")
        if not close(released["stable_seconds"], item["metrics"]["release_evidence"]["stable_seconds"]):
            mismatches.append("release:stable_seconds")
        velocity_errors = [max(abs(np.linalg.norm(row[EVIDENCE]["target_linear_velocity_world_m_s"])
                                   - row[EVIDENCE]["target_linear_speed_m_s"]),
                               abs(np.linalg.norm(row[EVIDENCE]["target_angular_velocity_world_rad_s"])
                                   - row[EVIDENCE]["target_angular_speed_rad_s"])) for row in rows]
        scene_sha = sha256(case / "scene.xml")
        key = (item["seed"], item["object_id"])
        scene_match = scene_sha == metadata["scene_sha256"]
        pair_match = paired_scenes.setdefault(key, scene_sha) == scene_sha
        time = time_audit(trace)
        artifacts = verify_artifacts(case, item["artifacts"])
        records.append({"path": item["path"], "seed": item["seed"], "object_id": item["object_id"],
                        "mode": item["mode"], "time": time, "artifacts": artifacts,
                        "scene_sha256": scene_sha, "scene_hash_matches_metadata": scene_match,
                        "paired_scene_identical": pair_match, "raw_recomputed": raw,
                        "lift": lift, "release": released, "summary_metric_mismatches": mismatches,
                        "maximum_velocity_norm_telemetry_error": max(velocity_errors, default=math.inf),
                        "trace_derived_metrics_consistent": not mismatches and time["accepted"] and artifacts["accepted"] and scene_match,
                        "native_contact_independently_verified": False,
                        "missing_evidence": ["FULL_NATIVE_CONTACT_STREAM", "RUNTIME_OBJECT_QPOS_WRITE_COUNTER"]})
    source = source_audit(protocol["source_hashes_before"], changed=protocol["source_changed_during_run"])
    source["before_after_identical"] = protocol["source_hashes_before"] == protocol["source_hashes_after"]
    counts = dict(Counter(f"{r['object_id']}:{r['mode']}:success={r['raw_recomputed']['strict_physical_success']}" for r in records))
    group_mismatches = []
    for object_id in protocol["tools"]:
        for mode in protocol["modes"]:
            selected = [r for r in records if r["object_id"] == object_id and r["mode"] == mode]
            expected = summary["tools"][object_id][mode]
            successes = sum(r["raw_recomputed"]["strict_physical_success"] for r in selected)
            rate = expected["rates"]["strict_physical_success"]
            if (expected["episodes"] != len(selected) or rate["numerator"] != successes
                    or rate["denominator"] != len(selected) or rate["unknown"] != 0
                    or not close(rate["rate"], successes/len(selected))):
                group_mismatches.append(f"{object_id}:{mode}")
    return {"scope": "known-tool fresh-fixture paired experiment; trace-derived contact attribution",
            "source": source, "episodes": records, "counts": counts,
            "aggregate_summary_mismatches": group_mismatches,
            "trace_summary_consistent": all(r["trace_derived_metrics_consistent"] for r in records) and source["accepted"] and not group_mismatches,
            "full_native_audit_passed": False,
            "limitations": ["No full native contact stream was recorded; original task outcomes are retained as trace-derived results.",
                            "No independent runtime qpos-write instrumentation is present."]}


def rolling_summary_audit(directory, records):
    summary = read_json(directory / "summary.json")
    entries = {(row["seed"], row["arm"]): row for row in summary["episodes"]}
    mismatches = []
    push_successes = control_successes = 0
    for record in records:
        case = PROJECT / record["path"]
        original = read_json(case / "result.json")
        key = (original["seed"], original["arm"])
        entry = entries.get(key)
        if entry is None or entry["success"] != record["task_success"]:
            mismatches.append(str(key))
        if original["arm"] == "index_push":
            push_successes += record["task_success"]
        else:
            control_successes += record["task_success"]
    paired = bool(records) and all(r["raw_summary_consistent"] for r in records)
    paired &= all(r["task_success"] == (Path(r["path"]).name == "index_push") for r in records)
    checks = {"episode_count": len(entries) == len(records) == summary["seeds"]*2,
              "index_push_count": push_successes == summary["index_push_successes"],
              "control_count": control_successes == summary["control_successes"],
              "paired_acceptance": paired == summary["paired_acceptance"],
              "all_episode_summaries": not mismatches}
    return {"checks": checks, "accepted": all(checks.values()), "index_push_successes": push_successes,
            "control_successes": control_successes, "mismatches": mismatches,
            "maximum_boundary_markers_per_trace": max(len(r["time"]["boundary_duplicate_indices"]) for r in records),
            "boundary_marker_policy": "Same-time phase marker allowed only if all physical state and tactile fields are identical; all positive time steps remain 2ms."}


def audit_native_case(directory, *, rolling):
    result = read_json(directory / "result.json")
    rows, native = read_jsonl(directory / "physical_trace.jsonl.gz"), read_jsonl(directory / "native_contacts.jsonl.gz")
    if len(rows) != len(native):
        raise ValueError("Native contact count differs from physical trace")
    geometry = Geometry(directory / "scene.xml")
    object_id = "tool_marker" if rolling else "tool_screwdriver"
    time = time_audit(rows, allow_boundary_duplicates=rolling)
    source = source_audit(result["source_sha256"], changed=result["source_changed_during_run"])
    peak, calibration_errors, derived = dict.fromkeys(CHANNELS, 0.), [], []
    native_tactile_error, native_target_error = 0., 0.
    transfer_violations, hand_self_events, non_hand_self, containment_failures = 0, 0, 0, 0
    self_peak, self_penetration, min_clearance, clearance_error, max_pose_error = 0., 0., math.inf, 0., 0.
    max_bound, bound_error, selected = np.zeros(3), 0., []
    proxy = None if rolling else result["planner_records"][0]["held_proxy"]
    for index, (row, contacts) in enumerate(zip(rows, native, strict=True)):
        if abs(row["time_s"]-contacts["time_s"]) > 1e-12 or row["phase"] != contacts["phase"]:
            raise ValueError("Native contact timestamps/phases differ from physical trace")
        e = geometry.contacts(contacts, object_id)
        for name in CHANNELS:
            value = e["hand_channel_forces_n"][name]
            peak[name] = max(peak[name], value)
            native_tactile_error = max(native_tactile_error, abs(value-row["tactile"]["fingers"][name]["normal_force_n"]))
        if not rolling:
            native_target_error = max(native_target_error, abs(e["target_hand_force_n"]-row["target_hand_force_n"]))
            old = row[EVIDENCE]
            for key in ("target_hand_force_n", "target_hand_contacts", "target_support_force_n", "maximum_hand_channel_force_n"):
                if not close(old[key], e[key]):
                    calibration_errors.append([index, key])
            if old["target_contact_fingers"] != e["target_contact_fingers"] or bool(old["environment_collisions"]) != bool(e["environment_collisions"]):
                calibration_errors.append([index, "fingers_or_collision"])
        clean = dict(row)
        clean[EVIDENCE] = e
        derived.append(clean)
        hand_self_events += len(e["self_contacts"])
        non_hand_self += e["non_hand_self"]
        for force, penetration in e["self_contacts"]:
            self_peak, self_penetration = max(self_peak, force), max(self_penetration, penetration)
        active = row["phase"] in ({"INDEX_PUSH", "NO_OVERRIDE_CONTROL"} if rolling else {"OMPL_TRANSFER"})
        if not active:
            continue
        if index == 0:
            raise ValueError("Missing prior joint state for kinematic reconstruction")
        previous_index = index-1
        if abs(row["time_s"]-rows[previous_index]["time_s"]) < 1e-12:
            previous_index -= 1
        if previous_index < 0:
            raise ValueError("Missing physical predecessor for boundary marker")
        max_pose_error = max(max_pose_error, geometry.reconstruct(row, rows[previous_index]))
        if rolling:
            thumb, tip = geometry.model.site("left_thumb_fingertip").id, geometry.model.site("left_index_fingertip").id
            aperture = float(np.linalg.norm(geometry.data.site_xpos[thumb]-geometry.data.site_xpos[tip]))
            old = row["supported_rolling"]
            if not close(aperture, old["finger_aperture_m"]):
                calibration_errors.append([index, "aperture"])
            if (not close(e["target_hand_force_n"], old["hand_contact_force_n"])
                    or e["rolling_finger_count"] != old["finger_contact_count"]
                    or e["rolling_support"] != old["support_contact"]):
                calibration_errors.append([index, "rolling_contact"])
            pos, quat = relative_pose(row, object_id)
            if not close(pos, old["relative_object_position_m"]) or angle(quat, old["relative_object_quaternion"]) > 1e-7:
                calibration_errors.append([index, "relative_pose"])
            selected.append({"row": row, "relative_position": pos, "relative_quaternion": quat,
                             "aperture": aperture, "contact_count": e["rolling_finger_count"],
                             "force": e["target_hand_force_n"], "support": e["rolling_support"]})
        else:
            transfer_violations += e["transfer_violations"]
            target, guard = geometry.names["v7_geom_tool_screwdriver"], geometry.names["v7_obstacle_carried_transfer_guard"]
            distance = min(float(mujoco.mj_geomDistance(geometry.model, geometry.data, geometry.names[name], guard, 1., None))
                           for name in geometry.robot | {"v7_geom_tool_screwdriver"})
            clearance_error = max(clearance_error, abs(distance-row["actual_guard_clearance_m"]))
            min_clearance = min(min_clearance, distance)
            wrist_r = geometry.data.xmat[geometry.wrist].reshape(3, 3)
            proxy_center = geometry.data.xpos[geometry.wrist] + wrist_r @ proxy["local_position"]
            proxy_r = wrist_r @ rotation(proxy["local_quaternion"])
            error = proxy_r.T @ (geometry.data.geom_xpos[target]-proxy_center)
            axis = proxy_r.T @ geometry.data.geom_xmat[target].reshape(3, 3)[:, 2]
            bound = abs(error) + .100*abs(axis) + .014
            contained = bool(np.all(bound <= proxy["half_extents"]))
            containment_failures += int(not contained)
            max_bound = np.maximum(max_bound, bound)
            bound_error = max(bound_error, float(np.max(abs(bound-row["tool_proxy_bound_m"]))))
            if contained != row["physical_tool_inside_declared_proxy"]:
                calibration_errors.append([index, "proxy_containment"])
    checks = {"native_and_trace_2ms": time["accepted"], "current_source_matches_recorded_hashes": source["accepted"],
              "native_channel_aggregation_matches": native_tactile_error <= 1e-8,
              "native_derived_telemetry_matches": not calibration_errors,
              "channel_peak_summary_matches": all(close(peak[k], result["channel_peak_force_n"][k]) for k in CHANNELS),
              "scene_timestep_2ms": abs(geometry.model.opt.timestep-PERIOD) <= 1e-12}
    common = {"path": directory.relative_to(PROJECT).as_posix(), "time": time, "source": source,
              "scene_sha256": sha256(directory / "scene.xml"), "scene_hash_was_recorded_at_run": False,
              "raw_artifacts": {name: {"sha256": sha256(directory/name), "bytes": (directory/name).stat().st_size}
                                for name in ("result.json", "scene.xml", "physical_trace.jsonl.gz", "native_contacts.jsonl.gz")},
              "native_channel_peak_n": peak, "maximum_native_tactile_error_n": native_tactile_error,
              "maximum_reconstructed_wrist_error_m": max_pose_error, "telemetry_mismatch_examples": calibration_errors[:20],
              "telemetry_mismatch_count": len(calibration_errors), "runtime_object_qpos_writes_independently_verified": False,
              "checks": checks}
    if rolling:
        if not selected:
            raise ValueError("No rolling/control samples")
        first, last = selected[0], selected[-1]
        apertures = [s["aperture"] for s in selected]
        metric = {"relative_translation_m": float(np.linalg.norm(last["relative_position"]-first["relative_position"])),
                  "relative_rotation_rad": angle(first["relative_quaternion"], last["relative_quaternion"]),
                  "wrist_translation_m": max(float(np.linalg.norm(np.asarray(s["row"]["wrist_position"])-first["row"]["wrist_position"])) for s in selected),
                  "wrist_rotation_rad": max(angle(first["row"]["wrist_quaternion"], s["row"]["wrist_quaternion"]) for s in selected),
                  "duration_s": last["row"]["time_s"]-first["row"]["time_s"], "finger_motion_m": max(apertures)-min(apertures),
                  "contact_transitions": sum(a["contact_count"] != b["contact_count"] for a, b in zip(selected, selected[1:])),
                  "max_object_drop_m": max(0., first["row"]["evaluation"][object_id]["position"][2]-min(s["row"]["evaluation"][object_id]["position"][2] for s in selected)),
                  "continuous_hand_contact": all(s["force"] > 0 for s in selected),
                  "continuous_support_contact": all(s["support"] for s in selected)}
        accepted = ((metric["relative_translation_m"] >= .015 or metric["relative_rotation_rad"] >= math.radians(15))
                    and metric["wrist_translation_m"] <= .020 and metric["wrist_rotation_rad"] <= math.radians(10)
                    and all(s["force"] >= .5 for s in selected) and metric["continuous_support_contact"]
                    and metric["finger_motion_m"] >= .003 and metric["contact_transitions"] >= 1
                    and metric["max_object_drop_m"] <= .005)
        mismatches = [key for key, value in metric.items() if not close(value, result["regrasp_evidence"][key])]
        checks.update(rolling_metrics_match_summary=not mismatches,
                      acceptance_matches_summary=accepted == result["regrasp_evidence"]["accepted"],
                      sample_count_matches=len(selected) == result["sample_count"])
        common.update(scope="native_contact_and_pose_supported_rolling", raw_recomputed=metric,
                      recomputed_rolling_accepted=bool(accepted), summary_metric_mismatches=mismatches,
                      task_success=bool(accepted and max(peak.values()) <= 12 and result["failure_reason"] is None))
    else:
        task = result["task_spec"]
        lift = lift_metrics(derived, result["phases"], object_id, result["initial_object_position_m"][2])
        slot = next(s["pose"] for s in result["scene_manifest"]["slots"] if s["slot_id"] == task["slot_id"])
        released = release_metrics(derived, result["phases"], task, slot, finite_difference=True)
        checks.update(native_target_force_matches=native_target_error <= 1e-8,
                      transfer_contact_count_matches=transfer_violations == result["native_transfer_collision_events"],
                      non_hand_self_count_matches=non_hand_self == result["native_non_hand_self_contact_events"],
                      hand_self_count_matches=hand_self_events == result["hand_self_contacts"]["events"],
                      hand_self_peak_matches=close(self_peak, result["hand_self_contacts"]["peak_force_n"]),
                      hand_self_penetration_matches=close(self_penetration, result["hand_self_contacts"]["maximum_penetration_m"]),
                      proxy_containment_count_matches=containment_failures == result["physical_tool_proxy_containment_failures"],
                      guard_clearance_matches=close(min_clearance, result["minimum_actual_guard_clearance_m"]),
                      all_guard_clearance_samples_match=clearance_error <= 1e-8,
                      all_proxy_bounds_match=bound_error <= 1e-8,
                      lift_acceptance_matches=lift["accepted"] == result["lift_evidence"]["accepted"],
                      pose_derived_stable_release=released["accepted"])
        common.update(scope="native_contact_and_reconstructed_geometry_carried_transfer", lift=lift, release=released,
                      transfer_environment_collision_events=transfer_violations,
                      whole_task_environment_collision_samples=sum(bool(r[EVIDENCE]["environment_collisions"]) for r in derived),
                      non_hand_robot_self_contact_events=non_hand_self,
                      hand_self_events=hand_self_events, peak_hand_self_force_n=self_peak,
                      maximum_hand_self_penetration_m=self_penetration,
                      collision_scope="Hand-internal self contacts are recorded and allowed; they are not zero. Non-hand robot self contacts and environment contacts are separately audited.",
                      minimum_guard_clearance_m=min_clearance, maximum_guard_clearance_error_m=clearance_error,
                      maximum_proxy_bound_m=max_bound.tolist(), maximum_proxy_bound_error_m=bound_error,
                      native_final_velocity_independently_verified=False,
                      task_success=bool(lift["accepted"] and released["accepted"] and transfer_violations == 0
                                        and containment_failures == 0 and non_hand_self == 0 and max(peak.values()) <= 12))
    common["raw_summary_consistent"] = all(checks.values())
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--three-tool", type=Path, default=PROJECT/"results/v7_three_tool_feedback_current_20261021_40_44")
    parser.add_argument("--rolling", type=Path, default=PROJECT/"results/v7_supported_rolling_current_20261022")
    parser.add_argument("--carried", type=Path, default=PROJECT/"results/v7_carried_obstacle_current_20261022_seed0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"format": "current-v7-independent-evidence-audit-r1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "auditor_sha256": sha256(__file__), "simulation_only": True, "human_teleoperation_verified": False,
               "policy": "fail closed for missing evidence at each named evidence tier; preserve trace-derived task outcomes"}
    for name, audit in (("three_tools", lambda: audit_three_tools(args.three_tool)),
                        ("rolling", lambda: [audit_native_case(p.parent, rolling=True) for p in sorted(args.rolling.glob("seed_*/*/result.json"))]),
                        ("carried", lambda: audit_native_case(args.carried, rolling=False))):
        try:
            payload[name] = audit()
        except Exception as error:
            payload[name] = {"audit_completed": False, "error": f"{type(error).__name__}: {error}"}
        (args.output/f"{name}.json").write_text(json.dumps(payload[name], indent=2, allow_nan=False)+"\n")
    payload["all_requested_audits_completed"] = all(not isinstance(payload[k], dict) or payload[k].get("audit_completed") is not False
                                                    for k in ("three_tools", "rolling", "carried"))
    payload["rolling_summary"] = (rolling_summary_audit(args.rolling, payload["rolling"])
                                  if isinstance(payload["rolling"], list) else {"accepted": False})
    (args.output/"rolling_summary.json").write_text(json.dumps(payload["rolling_summary"], indent=2, allow_nan=False)+"\n")
    payload["all_available_evidence_consistent"] = bool(payload["all_requested_audits_completed"]
        and payload["three_tools"].get("trace_summary_consistent")
        and isinstance(payload["rolling"], list) and payload["rolling"]
        and all(r["raw_summary_consistent"] for r in payload["rolling"])
        and payload["rolling_summary"]["accepted"]
        and payload["carried"].get("raw_summary_consistent"))
    (args.output/"receipt.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"completed": payload["all_requested_audits_completed"],
                      "three_tool_consistent": payload["three_tools"].get("trace_summary_consistent"),
                      "rolling_consistent": all(r["raw_summary_consistent"] for r in payload["rolling"]) if isinstance(payload["rolling"], list) else False,
                      "carried_consistent": payload["carried"].get("raw_summary_consistent")}, indent=2))
    return 0 if payload["all_available_evidence_consistent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
