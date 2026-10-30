#!/usr/bin/env python3
"""Nominal static contact-wrench analysis, not a manipulation-success test."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np
from scipy.optimize import linprog

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from check_public_ycb_aperture import distances, snapshot as envelope_snapshot
from evaluate_public_ycb_improved import BOX_HAND
from evaluate_v7_robustness import atomic_json, sha256
from kuka_sharpa_grasp.hand_grasp import HandConfig, HandContactReader, HandGraspController
from kuka_sharpa_grasp.pick_place import RobotKinematics
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def contact_rows(model, data, reader):
    rows = []
    for contact in data.contact[:data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if reader.target not in (a, b):
            continue
        other = b if a == reader.target else a
        finger = reader.fingers.get(other)
        if finger is None or not model.geom_contype[other]:
            continue
        frame = np.asarray(contact.frame).reshape(3, 3)
        # MuJoCo contact forces act on geom2; orient all forces onto the object.
        basis = frame.T * (1. if b == reader.target else -1.)
        rows.append({"finger": finger, "body": model.body(int(model.geom_bodyid[other])).name,
                     "geom": model.geom(other).name, "distance_m": float(contact.dist),
                     "position_world_m": contact.pos.tolist(), "basis_on_target": basis.tolist(),
                     "sliding_friction": float(contact.friction[0])})
    return rows


def support_lp(contacts, center, mass, gravity, channel_limit=8.):
    if not contacts:
        return {"feasible": False, "reason": "NO_CONTACTS"}
    count = len(contacts)
    matrix = np.zeros((6, 3 * count))
    inequalities, bounds = [], []
    for i, contact in enumerate(contacts):
        basis = np.asarray(contact["basis_on_target"])
        arm = np.asarray(contact["position_world_m"]) - center
        matrix[:3, 3*i:3*i+3] = basis
        matrix[3:, 3*i:3*i+3] = np.column_stack([np.cross(arm, vector) for vector in basis.T])
        for sx, sy in ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.)):
            row = np.zeros(3 * count)
            row[3*i:3*i+3] = (-contact["sliding_friction"], sx, sy)
            inequalities.append(row)
            bounds.append(0.)
    for finger in (*FINGERS, "palm"):
        row = np.zeros(3 * count)
        for i, contact in enumerate(contacts):
            if contact["finger"] == finger:
                row[3*i] = 1.
        inequalities.append(row)
        bounds.append(channel_limit)
    load = np.r_[-mass * gravity, np.zeros(3)]
    objective = np.zeros(3 * count)
    objective[::3] = 1.
    variable_bounds = [(0., None) if i % 3 == 0 else (None, None) for i in range(3 * count)]
    solution = linprog(objective, A_ub=np.asarray(inequalities), b_ub=np.asarray(bounds),
                       A_eq=matrix, b_eq=load, bounds=variable_bounds, method="highs")
    if not solution.success:
        return {"feasible": False, "reason": solution.message, "status": int(solution.status)}
    capacities = {}
    for finger in (*FINGERS, "palm"):
        capacities[finger] = sum(float(solution.x[3*i]) for i, c in enumerate(contacts) if c["finger"] == finger)
    # This is only static gravity support, with a conservative friction pyramid.
    expanded = np.column_stack((matrix, -load))
    margin = linprog(np.r_[np.zeros(3*count), -1.],
                     A_ub=np.column_stack((inequalities, np.zeros(len(inequalities)))), b_ub=bounds,
                     A_eq=expanded, b_eq=np.zeros(6), bounds=variable_bounds + [(0., 5.)], method="highs")
    return {"feasible": True, "required_total_normal_n": float(solution.fun),
            "required_channel_normal_n": capacities,
            "gravity_load_multiplier": None if not margin.success else float(margin.x[-1]),
            "wrench_residual": float(np.linalg.norm(matrix @ solution.x - load)),
            "contact_forces_contact_frame_n": solution.x.reshape(-1, 3).tolist()}


def set_posture(model, data, hand, closures):
    hand.closure = closures.copy()
    hand.initialize(data)
    mujoco.mj_forward(model, data)


def shallow_closures(model, data, hand, reader, penetration):
    result = dict.fromkeys(FINGERS, 0.)
    for finger in FINGERS:
        def clearance(value):
            trial = result.copy()
            trial[finger] = value
            set_posture(model, data, hand, trial)
            return min(r["distance_mm"] / 1000 for r in distances(model, data, reader) if r["finger"] == finger)
        if clearance(0.) <= -penetration:
            return None, "OPEN_INTERFERENCE"
        previous = 0.
        for value in np.linspace(.025, 1., 40):
            if clearance(float(value)) <= -penetration:
                low, high = previous, float(value)
                for _ in range(14):
                    middle = (low + high) / 2.
                    if clearance(middle) <= -penetration:
                        high = middle
                    else:
                        low = middle
                result[finger] = high
                break
            previous = float(value)
    set_posture(model, data, hand, result)
    return result, None


def run(envelope, output):
    output.mkdir(parents=True, exist_ok=False)
    before = envelope_snapshot()
    before[Path(__file__).relative_to(PROJECT).as_posix()] = sha256(Path(__file__))
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    original = json.loads(envelope.read_text())
    rows = []
    for kind in ("ycb_potted_meat_can", "ycb_cracker_box"):
        model = mujoco.MjModel.from_xml_path(str(build_pick_place_scene(output / f"{kind}.xml", kind,
                                                                  physics_profile="physical-v6")))
        kin, reader = RobotKinematics(model), HandContactReader(model)
        target = model.body("target").id
        address = int(model.jnt_qposadr[model.joint("target_free").id])
        candidates = [row for row in original["rows"] if row["kind"] == kind and row["ik"]
                      and row["open_minimum_clearance_mm"] >= 1.]
        for index, candidate in enumerate(candidates):
            config = replace(BOX_HAND if kind == "ycb_cracker_box" else HandConfig(),
                             thumb_opposition_rad=candidate["thumb_opposition_rad"])
            q = kin.solve(np.asarray(candidate["aperture_offset_m"]), tolerance=.0001)
            data, hand = mujoco.MjData(model), HandGraspController(model, config)
            data.qpos[:7] = q
            data.qpos[address:address+7] = [.72, 0., .38, 1., 0., 0., 0.]
            for penetration in (.00025, .0005, .001):
                closures, error = shallow_closures(model, data, hand, reader, penetration)
                contacts = [] if closures is None else contact_rows(model, data, reader)
                active = sorted({c["finger"] for c in contacts if c["finger"] != "palm"})
                lp = support_lp(contacts, data.xipos[target].copy(), float(model.body_mass[target]),
                                model.opt.gravity.copy())
                row = {"kind": kind, "aperture_offset_m": candidate["aperture_offset_m"],
                       "thumb_opposition_rad": candidate["thumb_opposition_rad"],
                       "nominal_shallow_penetration_m": penetration, "closure": closures,
                       "open_clearance_mm": candidate["open_minimum_clearance_mm"],
                       "contact_fingers": active, "contacts": contacts, "geometry_error": error,
                       "gravity_support_lp": lp, "nominal_mass_kg": float(model.body_mass[target]),
                       "candidate_admissible": (error is None and "thumb" in active and len(active) >= 3
                                                and lp["feasible"])}
                rows.append(row)
            if index % 10 == 0:
                print(kind, "candidate", index+1, "/", len(candidates), flush=True)
        ranked = sorted([row for row in rows if row["kind"] == kind and row["candidate_admissible"]],
                        key=lambda row: (-row["gravity_support_lp"]["gravity_load_multiplier"],
                                         row["gravity_support_lp"]["required_total_normal_n"],
                                         row["nominal_shallow_penetration_m"]))
        atomic_json(output / f"{kind}_ranked.json", ranked)
        print(kind, "admissible", len(ranked), "top", [{"aperture": r["aperture_offset_m"],
            "opposition": r["thumb_opposition_rad"], "closures": r["closure"],
            "normal_n": r["gravity_support_lp"]["required_total_normal_n"],
            "load_multiplier": r["gravity_support_lp"]["gravity_load_multiplier"]} for r in ranked[:3]], flush=True)
    after = envelope_snapshot()
    after[Path(__file__).relative_to(PROJECT).as_posix()] = sha256(Path(__file__))
    result = {"format": "nominal_public_ycb_static_gravity_wrench_r1", "complete": True,
              "envelope_input": str(envelope), "envelope_sha256": sha256(envelope),
              "source_hashes_before": before, "source_hashes_after": after,
              "source_changed_during_run": before != after, "rows": rows,
              "scope": "Scratch nominal geometry only. Per-channel 8N conservative sliding-friction LP balances gravity/torque; no actuator reachability, disturbances, dynamic retention, or task-success claim. Reset-like qpos writes never occur in a task trajectory."}
    atomic_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.envelope.resolve(), args.output.resolve())
    return int(result["source_changed_during_run"])


if __name__ == "__main__":
    raise SystemExit(main())
