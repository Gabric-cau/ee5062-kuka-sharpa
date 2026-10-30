#!/usr/bin/env python3
"""Independently audit rich native records under the additional 25mm/3s/12N gates."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

from evaluate_v7_robustness import atomic_json, sha256
from public_ycb_evidence import load_evidence_model

PROJECT = Path(__file__).resolve().parents[1]
CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
THRESHOLDS = {"xy_error_m": .025, "height_error_m": .025, "released_seconds": 3.,
              "channel_peak_n": 12., "linear_speed_m_s": .04, "angular_speed_rad_s": .2,
              "hand_clearance_m": .10, "held_rise_m": .07, "held_seconds": .1,
              "robot_table_penetration_m": .002, "held_contact_normal_minimum_n": .05}


def longest_interval(times, flags):
    start, previous, longest = None, None, 0.
    for timestamp, valid in zip(times, flags, strict=True):
        if previous is not None and not 0 < timestamp - previous <= .002 + 1e-9:
            start = None
        if valid:
            start = timestamp if start is None else start
            longest = max(longest, timestamp - start)
        else:
            start = None
        previous = timestamp
    return longest


def hand_geometry(model):
    hand = {}
    for gid in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[gid])).name or ""
        if body.startswith("left_") and model.geom_contype[gid]:
            hand[gid] = next((finger for finger in CHANNELS[:-1]
                              if body.startswith("left_" + finger + "_")), "palm")
    return hand


def channel_pressures(contacts, hand):
    pressure = dict.fromkeys(CHANNELS, 0.)
    for contact in contacts:
        force = np.asarray(contact["force_torque_contact_frame"], float)
        if force.shape != (6,) or not np.isfinite(force).all():
            raise ValueError("Invalid native wrench")
        sides = set(contact["geom_ids"]) & hand.keys()
        if len(sides) == 1:
            pressure[hand[next(iter(sides))]] += abs(float(force[0]))
    return pressure


def audit_case(case, row, public_mesh_paths):
    if row.get("runner_error"):
        return {"path": row["path"], "strict_success": None, "unknown": True, "reason": row["runner_error"]}
    streams = {}
    for key in ("native", "trace"):
        record = row["artifacts"][key]
        path = case / record["path"]
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Changed raw stream: {path}")
        with gzip.open(path, "rt") as handle:
            streams[key] = json.load(handle)
    native, trace = streams["native"], streams["trace"]
    if not native or any("actual_hand_joints" not in sample for sample in native):
        return {"path": row["path"], "strict_success": None, "unknown": True, "reason": "ACTUAL_HAND_STATE_MISSING"}
    result = json.loads((case / "result.json").read_text())
    xml = ET.parse(case / "scene.xml").getroot()
    target_xml = xml.find(".//geom[@name='target_geom']")
    if target_xml is None:
        raise ValueError("Missing public target mesh")
    target_asset = xml.find(f"./asset/mesh[@name='{target_xml.get('mesh')}']")
    if target_asset is None:
        raise ValueError("Public target geometry does not reference a mesh asset")
    mesh_path = (PROJECT / "vendor" / "mujoco_menagerie" / target_asset.get("file")).resolve()
    if mesh_path != public_mesh_paths[row["kind"]]:
        raise ValueError("Recorded scene references a different public object asset")
    model = load_evidence_model(case / "scene.xml")
    if abs(model.opt.timestep - .002) > 1e-12:
        raise ValueError("Expected declared 2ms physics")
    hand, target, table = hand_geometry(model), model.geom("target_geom").id, model.geom("table").id
    joints = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(model.njnt)
              if (model.joint(j).name or "").startswith("left_") and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    address = int(model.jnt_qposadr[model.joint("target_free").id])
    data = mujoco.MjData(model)
    by_time = {round(sample["timestamp"], 9): sample for sample in trace}
    times = np.asarray([sample["timestamp"] for sample in native])
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite native timestamps")
    peaks, phases, held, released, collisions = dict.fromkeys(CHANNELS, 0.), {}, [], [], []
    first_height, zone = native[0]["target_position_world_m"][2], np.asarray(result["zone"])
    for sample in native:
        observed = by_time[round(sample["timestamp"], 9)]
        contacts = sample["native_contacts"]
        pressures = channel_pressures(contacts, hand)
        for name, value in pressures.items():
            peaks[name] = max(peaks[name], value)
        phases[sample["state"]] = max(phases.get(sample["state"], 0.), max(pressures.values()))
        hand_contact = any(target in contact["geom_ids"] and bool(set(contact["geom_ids"]) & hand.keys()) for contact in contacts)
        loaded_hand_contact = any(target in contact["geom_ids"] and bool(set(contact["geom_ids"]) & hand.keys())
                                  and abs(contact["force_torque_contact_frame"][0]) > .05 for contact in contacts)
        support = any(set(contact["geom_ids"]) == {target, table}
                      and abs(contact["force_torque_contact_frame"][0]) > .05 for contact in contacts)
        collisions.append(any(table in contact["geom_ids"] and contact["distance_m"] < -.002
                              and any((model.body(int(model.geom_bodyid[g])).name or "").startswith(("link", "left_"))
                                      for g in contact["geom_ids"] if g != table) for contact in contacts))
        position, velocity = np.asarray(sample["target_position_world_m"]), np.asarray(sample["target_spatial_velocity_world"])
        if position.shape != (3,) or velocity.shape != (6,) or not np.isfinite(np.r_[position, velocity]).all():
            raise ValueError("Invalid measured state")
        held.append(position[2] > first_height + .07 and loaded_hand_contact and not support)
        if set(sample["actual_hand_joints"]) != set(joints):
            raise ValueError("Incomplete actual hand joint coverage")
        data.qpos[:7] = observed["arm_joint_position"]
        data.qpos[address:address+7] = sample["target_qpos"]
        for name, qaddress in joints.items():
            data.qpos[qaddress] = sample["actual_hand_joints"][name]["position_rad"]
        if not np.isfinite(data.qpos).all():
            raise ValueError("Nonfinite actual qpos")
        mujoco.mj_kinematics(model, data)
        clearance = min(float(mujoco.mj_geomDistance(model, data, target, gid, 1., None)) for gid in hand)
        released.append(observed["oracle"]["release_commanded"] and not hand_contact and support
                        and np.linalg.norm(position[:2] - zone[:2]) < .025 and abs(position[2] - zone[2]) < .025
                        and np.linalg.norm(velocity[3:]) < .04 and np.linalg.norm(velocity[:3]) < .2 and clearance > .10)
    contiguous = bool(abs(times[0] - .002) < 1e-9 and np.allclose(np.diff(times), .002, atol=1e-9, rtol=0))
    audit = row["native_audit"]
    held_seconds, released_seconds = longest_interval(times, held), longest_interval(times, released)
    checks = {"legacy_task_success": result["success"] is True, "native_contiguous_2ms": contiguous,
              "no_external_target_qpos_writes": audit["external_target_qpos_writes"] == 0
                  and audit["integration_steps"] == audit["native_samples"] == len(native),
              "held_lift": held_seconds + 1e-9 >= .1, "released_stable_3s": released_seconds + 1e-9 >= 3.,
              "all_channels_within_12n": max(peaks.values()) <= 12., "no_robot_table_penetration": not any(collisions)}
    return {"path": row["path"], "strict_success": all(checks.values()), "unknown": False, "checks": checks,
            "channel_peaks_n": peaks, "phase_peaks_n": phases, "native_samples": len(native),
            "held_seconds": held_seconds, "released_stable_seconds": released_seconds,
            "legacy_failure": result["failure"], "result_sha256": sha256(case / "result.json"),
            "scene_sha256": sha256(case / "scene.xml")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary = json.loads((directory / "summary.json").read_text())
    protocol = summary["protocol"]
    if not summary["complete"] or protocol["source_changed_during_run"] or protocol["source_hashes_before"] != protocol["source_hashes_after"]:
        raise ValueError("Incomplete or mutated experiment")
    if not protocol["public_meshes_unchanged"]:
        raise ValueError("Mesh inventory changed")
    public_mesh_paths = {}
    for kind, record in protocol["public_meshes"].items():
        path = (PROJECT / record["path"]).resolve()
        if not path.is_relative_to(PROJECT) or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Actual public mesh differs from the frozen inventory")
        public_mesh_paths[kind] = path
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        for path, expected in protocol["source_hashes_before"].items():
            if hashlib.sha256(archive.read(path)).hexdigest() != expected:
                raise ValueError("Frozen source archive differs")
    if len(summary["rows"]) != len(protocol["schedule"]):
        raise ValueError("Scheduled denominator differs")
    rows = []
    for planned, row in zip(protocol["schedule"], summary["rows"], strict=True):
        if any(row[key] != value for key, value in planned.items()):
            raise ValueError("Schedule mismatch")
        audited = audit_case(directory / row["path"], row, public_mesh_paths)
        rows.append(audited)
        print(json.dumps(audited), flush=True)
    receipt = {"format": "public_ycb_strict_native_audit_20261021", "accepted_evidence": True,
               "thresholds": THRESHOLDS, "denominator": len(rows), "rows": rows,
               "strict_successes": sum(row["strict_success"] is True for row in rows),
               "unknowns": sum(row["unknown"] for row in rows), "summary_sha256": sha256(directory / "summary.json"),
               "verification_script_sha256": sha256(Path(__file__)),
               "public_mesh_inventory_checked": True,
               "portable_scene_loader_sha256": sha256(PROJECT / "scripts" / "public_ycb_evidence.py"),
               "scope": "Additional whole-episode 25mm/3s/12N gates. Original V6 labels retained. Native ground truth only for independent evaluation. Clearance recomputed in scratch data from integrated actual qpos. Robot-table gate does not certify general clutter collision avoidance."}
    atomic_json(args.output.resolve(), receipt)


if __name__ == "__main__":
    main()
