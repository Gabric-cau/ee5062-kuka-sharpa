#!/usr/bin/env python3
"""Offline nominal hand/mesh collision envelope, not a dynamic grasp trial."""

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from develop_public_ycb import source_snapshot
from evaluate_public_ycb import public_mesh_inventory
from evaluate_public_ycb_improved import BOX_HAND, BOX_SPEC
from evaluate_v7_robustness import atomic_json, sha256
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.hand_grasp import HandConfig, HandContactReader, HandGraspController
from kuka_sharpa_grasp.pick_place import RobotKinematics
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


def snapshot():
    result = source_snapshot()
    for name in ("check_public_ycb_aperture.py", "evaluate_public_ycb_improved.py"):
        path = PROJECT / "scripts" / name
        result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return result


def distances(model, data, reader):
    rows = []
    for gid, finger in reader.fingers.items():
        if not model.geom_contype[gid]:
            continue
        distance = float(mujoco.mj_geomDistance(model, data, reader.target, gid, .2, None))
        rows.append({"finger": finger, "geom": model.geom(gid).name,
            "body": model.body(int(model.geom_bodyid[gid])).name,
            "distance_mm": 1000 * distance})
    return rows


def measure(model, kin, aperture, config):
    q = kin.solve(np.asarray(aperture), tolerance=.0001)
    if q is None:
        return {"ik": False}
    data = mujoco.MjData(model)
    data.qpos[:7] = q
    target = model.body("target").id
    joint = int(model.body_jntadr[target])
    address = int(model.jnt_qposadr[joint])
    data.qpos[address:address + 7] = [.72, 0., .38, 1., 0., 0., 0.]
    reader = HandContactReader(model)
    hand = HandGraspController(model, config)
    hand.initialize(data)
    mujoco.mj_forward(model, data)
    opened = distances(model, data, reader)
    first, closure_rows = {}, []
    for closure in np.linspace(0., .7, 15):
        hand.closure = dict.fromkeys(hand.closure, float(closure))
        hand.initialize(data)
        mujoco.mj_forward(model, data)
        measured = distances(model, data, reader)
        contacts = [row for row in measured if row["distance_mm"] <= 0.]
        for row in sorted(contacts, key=lambda row: row["distance_mm"]):
            first.setdefault(row["finger"], {"closure": float(closure), **row})
        proximal = [row for row in contacts if row["finger"] == "palm" or "_MP" in row["body"]]
        fingers = sorted({row["finger"] for row in contacts if row["finger"] != "palm"})
        closure_rows.append({"closure": float(closure), "contact_fingers": fingers,
            "minimum_distance_mm": min(row["distance_mm"] for row in measured),
            "maximum_proximal_penetration_mm": max([0.] + [-row["distance_mm"] for row in proximal]),
            "contacts": contacts})
    balanced = next((row for row in closure_rows if "thumb" in row["contact_fingers"]
        and len(row["contact_fingers"]) >= 3), None)
    return {"ik": True, "wrist_position_m": kin.position(q).tolist(),
        "open_minimum_clearance_mm": min(row["distance_mm"] for row in opened),
        "open_distances": opened, "first_contact": first, "closure_envelope": closure_rows,
        "first_three_finger_with_thumb": balanced,
        "runtime_target_pose_input": False,
        "offline_scratch_qpos_writes": "Nominal authored mesh geometry only; no task trajectory or success claim"}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    rows = []
    for kind in ("ycb_potted_meat_can", "ycb_cracker_box"):
        scene = build_pick_place_scene(output / f"{kind}.xml", kind, physics_profile="physical-v6")
        model = mujoco.MjModel.from_xml_path(str(scene))
        kin = RobotKinematics(model)
        base = BOX_SPEC if kind == "ycb_cracker_box" else GRASP_SPECIFICATIONS[kind]
        heights = (.16, .18, .20, .22, .24) if kind == "ycb_cracker_box" else (.035, .055, .075, .095)
        for dx, dy, z, opposition in product((-.02, 0., .02), (-.015, 0., .015), heights, (1., 1.2, 1.4)):
            aperture = (base.aperture_offset[0] + dx, base.aperture_offset[1] + dy, z)
            config = replace(BOX_HAND if kind == "ycb_cracker_box" else HandConfig(),
                             thumb_opposition_rad=opposition)
            rows.append({"kind": kind, "aperture_offset_m": aperture,
                         "thumb_opposition_rad": opposition, **measure(model, kin, aperture, config)})
        eligible = [row for row in rows if row["kind"] == kind and row["ik"]
            and row["open_minimum_clearance_mm"] >= 1.
            and row["first_three_finger_with_thumb"] is not None]
        eligible.sort(key=lambda row: (row["first_three_finger_with_thumb"]["maximum_proximal_penetration_mm"],
            -row["first_three_finger_with_thumb"]["minimum_distance_mm"],
            -row["open_minimum_clearance_mm"]))
        atomic_json(output / f"{kind}_ranked.json", eligible)
        print(kind, "grid", sum(row["kind"] == kind for row in rows), "eligible", len(eligible), flush=True)
        for row in eligible[:6]:
            print("  ", row["aperture_offset_m"], row["thumb_opposition_rad"],
                "open mm", round(row["open_minimum_clearance_mm"], 3),
                "three-finger", row["first_three_finger_with_thumb"]["closure"],
                "proximal mm", round(row["first_three_finger_with_thumb"]["maximum_proximal_penetration_mm"], 3), flush=True)
    after = snapshot()
    summary = {"format": "public_ycb_nominal_collision_envelope_r1", "complete": True,
        "source_hashes_before": before, "source_hashes_after": after,
        "source_changed_during_run": before != after, "public_meshes": meshes,
        "public_meshes_unchanged": meshes == public_mesh_inventory(),
        "scope": "Known upright mesh nominal body position (0.72,0,0.38); offline static hand geometry, not measured contact forces or task success",
        "rows": rows, "runtime": {"mujoco": mujoco.__version__, "python": sys.version}}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.output.resolve())
    return int(summary["source_changed_during_run"] or not summary["public_meshes_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
