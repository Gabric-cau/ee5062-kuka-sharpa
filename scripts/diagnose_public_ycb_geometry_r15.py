#!/usr/bin/env python3
"""Private nominal YCB geometry screen, never a dynamic task or success test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
import zipfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from estimated_tactile_patches import PublicMeshSurface
from score_public_ycb_grasp_envelope import contact_rows, set_posture, shallow_closures
from tactile_wrench_allocator import allocate_contact_wrench
from kuka_sharpa_grasp.hand_grasp import FINGERS, HandConfig, HandContactReader, HandGraspController
from kuka_sharpa_grasp.pick_place import RobotKinematics
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene
from kuka_sharpa_grasp.v7_motion import V7Kinematics


SCOPE = ("Offline authored nominal public-mesh geometry and private robot-hinge scratch states only. "
    "No target qpos writes, integration, actual RGB-D estimate, measured preload, actuator-force "
    "convergence, physical retention, probe admission, or dynamic task-success evidence.")
ORIGIN = np.array([.72, 0., .38])


@dataclass(frozen=True)
class Candidate:
    opposition_rad: float
    roll_deg: float
    yaw_deg: float

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Candidate angles must be finite")
        if not .8 <= self.opposition_rad <= 1.4:
            raise ValueError("Diagnostic opposition must be in [0.8, 1.4] rad")
        if abs(self.roll_deg) > 12. or abs(self.yaw_deg) > 12.:
            raise ValueError("Diagnostic wrist rotations must be within +/-12 degrees")


@dataclass(frozen=True)
class Limits:
    open_target_m: float = .001
    table_m: float = .005
    pressure_per_finger_n: float = 6.
    gravity_margin: float = 1.5
    friction_safety_factor: float = .85
    normal_cosine: float = .85
    maximum_abs_normal_z: float = .2
    weighted_abs_normal_z: float = .12
    surface_band_m: float = .006
    allocated_contact_floor_n: float = .1


LIMITS = Limits()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_snapshot(profile_path):
    paths = sorted((PROJECT / "src").rglob("*.py"))
    paths += [PROJECT / "scripts" / name for name in (
        "diagnose_public_ycb_geometry_r15.py", "estimated_tactile_patches.py",
        "score_public_ycb_grasp_envelope.py", "check_public_ycb_aperture.py",
        "tactile_wrench_allocator.py")]
    paths += [PROJECT / "tests/test_public_ycb_geometry_r15.py"]
    paths += [Path(profile_path)]
    return {str(path.relative_to(PROJECT)): sha256(path) for path in paths}


def candidate_rotation(nominal_rotation, candidate):
    """World-frame yaw after world-frame roll; neither rotates the object."""
    roll = Rotation.from_euler("x", candidate.roll_deg, degrees=True).as_matrix()
    yaw = Rotation.from_euler("z", candidate.yaw_deg, degrees=True).as_matrix()
    return yaw @ roll @ np.asarray(nominal_rotation)


def quaternion(rotation):
    value = np.empty(4)
    mujoco.mju_mat2Quat(value, np.asarray(rotation).reshape(9))
    return value


def minimum_clearance(model, data, fixed_geom, robot_geoms, maximum=1.):
    return min(float(mujoco.mj_geomDistance(model, data, fixed_geom, gid, maximum, None))
        for gid in robot_geoms)


def target_unchanged(model, data):
    address = int(model.jnt_qposadr[model.joint("target_free").id])
    return bool(np.array_equal(data.qpos[address:address + 7], model.qpos0[address:address + 7]))


def assert_private_state(model, data):
    if data.time != 0. or not target_unchanged(model, data):
        raise RuntimeError("Static analysis must leave time and authored target qpos unchanged")


def estimated_contacts(model, data, reader, surface):
    annotated, patches = [], []
    friction = float(np.sqrt(surface.friction_coefficient))
    for contact in contact_rows(model, data, reader):
        point = np.asarray(contact["position_world_m"])
        normal = np.asarray(contact["basis_on_target"])[:, 0]
        query = surface.query_surface(point - ORIGIN)
        cosine = None if not query["accepted"] else float(-query["outward_normal_local"] @ normal)
        annotated.append({**contact, "normal_on_object_world": normal.tolist(),
            "surface_signed_distance_m": query.get("signed_distance_m"),
            "surface_normal_cosine": cosine})
        patches.append({"finger": contact["finger"], "position_world_m": point.tolist(),
            "normal_on_object_world": normal.tolist(), "measured_pressure_n": 0.,
            "friction_coefficient": friction})
    return annotated, patches


def allocation(patches, surface, margin):
    return allocate_contact_wrench(patches, ORIGIN + surface.com_local_m, surface.mass_kg,
        pressure_limit_n=LIMITS.pressure_per_finger_n, gravity_margin=margin,
        friction_safety_factor=LIMITS.friction_safety_factor)


def hypothetical_fixed_normals(patches, proposed, surface):
    hypothetical = [dict(patch) for patch in patches]
    for row in proposed.get("patch_allocations", []):
        hypothetical[row["input_index"]]["measured_pressure_n"] = row["target_normal_pressure_n"]
    result = allocation(hypothetical, surface, 1.)
    result["scope"] = "Proposed 1.5g normal loads held fixed under 1g; hypothetical, never measured force"
    return result


def compression_diagnostics(model, data, hand, contacts, proposed):
    values = []
    for row in proposed.get("patch_allocations", []):
        load = row.get("target_normal_pressure_n")
        if load is None or load < LIMITS.allocated_contact_floor_n:
            continue
        contact = contacts[row["input_index"]]
        body = model.body(contact["body"]).id
        linear = np.zeros((3, model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jac(model, data, linear, angular, np.asarray(contact["position_world_m"]), body)
        selected = []
        for name in hand.actuators:
            if not name.startswith("left_" + contact["finger"] + "_") or "AA" in name or "CMC" in name:
                continue
            joint = model.joint(name).id
            qadr = int(model.jnt_qposadr[joint])
            dof = int(model.jnt_dofadr[joint])
            ratio = hand.angles().get(name, 0.) / max(hand.closure[contact["finger"]], 1e-12)
            selected.append((name, dof, ratio, qadr))
        normal = np.asarray(contact["normal_on_object_world"])
        jacobian = np.array([normal @ linear[:, dof] for _, dof, _, _ in selected])
        gain = float(jacobian @ [ratio for _, _, ratio, _ in selected])
        values.append({"finger": contact["finger"], "body": contact["body"],
            "allocated_normal_n": load, "joint_names": [name for name, _, _, _ in selected],
            "normal_jacobian_m_rad": jacobian.tolist(),
            "closure_synergy_compression_gain_m": gain,
            "scope": "Local kinematic hand-point derivative with object fixed; not force or relative-motion prediction"})
    return values


def geometry_reasons(row):
    reasons = []
    scalars = [row["closed_hand_table_clearance_m"], row["maximum_contact_penetration_m"]]
    scalars += [value for value in (row["maximum_allocated_abs_normal_z"],
        row["weighted_allocated_abs_normal_z"]) if value is not None]
    scalars += [contact[key] for contact in row["allocated_contacts"]
        for key in ("surface_normal_cosine", "surface_signed_distance_m") if contact[key] is not None]
    scalars += list(row["allocation_1_5g"].get("target_pressures_n", {}).values())
    if not np.isfinite(scalars).all():
        reasons.append("NONFINITE_STATIC_METRIC")
    if row.get("geometry_error"):
        reasons.append(row["geometry_error"])
    lp = row["allocation_1_5g"]
    if not lp.get("feasible"):
        reasons.append("STATIC_1_5G_LP_INFEASIBLE")
    if not row["allocation_1g"].get("feasible"):
        reasons.append("STATIC_1G_LP_INFEASIBLE")
    if not row["hypothetical_fixed_normal_1g"].get("measured_feasible"):
        reasons.append("HYPOTHETICAL_FIXED_NORMAL_1G_INFEASIBLE")
    if "thumb" not in row["allocated_fingers"] or len(row["allocated_fingers"]) < 3:
        reasons.append("INSUFFICIENT_ALLOCATED_FINGERS_INCLUDING_THUMB")
    if row["maximum_allocated_abs_normal_z"] is None or row["maximum_allocated_abs_normal_z"] > LIMITS.maximum_abs_normal_z:
        reasons.append("ALLOCATED_NORMAL_NOT_SIDEWALL")
    if row["weighted_allocated_abs_normal_z"] is None or row["weighted_allocated_abs_normal_z"] > LIMITS.weighted_abs_normal_z:
        reasons.append("WEIGHTED_NORMAL_NOT_SIDEWALL")
    for contact in row["allocated_contacts"]:
        if (contact["surface_normal_cosine"] is None
                or contact["surface_normal_cosine"] < LIMITS.normal_cosine
                or contact["surface_signed_distance_m"] is None
                or abs(contact["surface_signed_distance_m"]) > LIMITS.surface_band_m):
            reasons.append("PUBLIC_SURFACE_GEOMETRY_DISAGREES")
            break
    if any(contact["finger"] == "palm" for contact in row["contacts"]):
        reasons.append("UNALLOCATED_PALM_CONTACT")
    if row["closed_hand_table_clearance_m"] < LIMITS.table_m:
        reasons.append("CLOSED_TABLE_CLEARANCE")
    if row["maximum_contact_penetration_m"] > row["penetration_m"] + 1e-5:
        reasons.append("DEEPER_THAN_REQUESTED_SHALLOW_CONTACT")
    if any(item["closure_synergy_compression_gain_m"] <= 0.
            for item in row["compression_diagnostics"] if item["finger"] == "thumb"):
        reasons.append("THUMB_FLEXION_NOT_COMPRESSING")
    return list(dict.fromkeys(reasons))


def evaluate_candidate(model, nominal_kin, pose_kin, reader, surface, profile, candidate, penetration):
    row = {"candidate": asdict(candidate), "penetration_m": float(penetration),
        "aperture_offset_m": profile["aperture_offset"], "static_screen_pass": False,
        "failure_reasons": [], "task_success": None}
    config = replace(HandConfig(), thumb_opposition_rad=candidate.opposition_rad,
        thumb_mcp_flexion_ratio=profile["thumb_mcp_flexion_ratio"],
        thumb_ip_flexion_ratio=profile["thumb_ip_flexion_ratio"])
    data, hand = mujoco.MjData(model), HandGraspController(model, config)
    set_posture(model, data, hand, dict.fromkeys(FINGERS, 0.))
    nominal_kin.copy_open_hand(data)
    aperture = np.asarray(profile["aperture_offset"])
    nominal = nominal_kin.solve(aperture, tolerance=.0001)
    rotation = candidate_rotation(nominal_kin.rotation, candidate)
    position = nominal_kin.origin + aperture
    aligned = None if nominal is None else pose_kin.solve(position, quaternion(rotation), seed=nominal)
    approached = None if aligned is None else pose_kin.solve(position + [0., 0., .18], quaternion(rotation), seed=aligned)
    row.update(wrist_rotation_world=rotation.tolist(), wrist_position_world_m=position.tolist(),
        aligned_arm_joints_rad=None if aligned is None else aligned.tolist(),
        approach_arm_joints_rad=None if approached is None else approached.tolist())
    if aligned is None or approached is None:
        row["failure_reasons"] = ["ALIGNED_IK" if aligned is None else "APPROACH_IK"]
        assert_private_state(model, data)
        return row
    hand_geoms = [gid for gid in reader.fingers if model.geom_contype[gid]]
    robot_geoms = [gid for gid in range(model.ngeom) if model.geom_contype[gid]
        and (model.body(int(model.geom_bodyid[gid])).name or "").startswith(("left_", "link", "base"))]
    table = model.geom("table").id
    clearances = []
    for fraction in np.linspace(0., 1., 21):
        data.qpos[pose_kin.qpos_adrs] = (1. - fraction) * approached + fraction * aligned
        mujoco.mj_forward(model, data)
        clearances.append({"fraction": float(fraction),
            "robot_table_m": minimum_clearance(model, data, table, robot_geoms),
            "hand_target_m": minimum_clearance(model, data, reader.target, hand_geoms, .2)})
    row.update(open_approach_samples=clearances,
        open_approach_robot_table_clearance_m=min(item["robot_table_m"] for item in clearances),
        open_approach_hand_target_clearance_m=min(item["hand_target_m"] for item in clearances))
    if row["open_approach_robot_table_clearance_m"] < LIMITS.table_m:
        row["failure_reasons"].append("OPEN_APPROACH_TABLE_CLEARANCE")
    if row["open_approach_hand_target_clearance_m"] < LIMITS.open_target_m:
        row["failure_reasons"].append("OPEN_APPROACH_TARGET_CLEARANCE")
    closures, error = shallow_closures(model, data, hand, reader, penetration)
    annotated, patches = ([], []) if closures is None else estimated_contacts(model, data, reader, surface)
    proposed = allocation(patches, surface, LIMITS.gravity_margin)
    row["unfiltered_closure"] = closures
    row["unfiltered_contacts"] = annotated
    if proposed["feasible"] and closures is not None:
        active = {finger for finger, value in proposed["target_pressures_n"].items()
            if value >= LIMITS.allocated_contact_floor_n}
        closures = {finger: value if finger in active else 0. for finger, value in closures.items()}
        set_posture(model, data, hand, closures)
        annotated, patches = estimated_contacts(model, data, reader, surface)
        proposed = allocation(patches, surface, LIMITS.gravity_margin)
    one_g = allocation(patches, surface, 1.)
    fixed = hypothetical_fixed_normals(patches, proposed, surface) if proposed["feasible"] else {
        "measured_feasible": False, "reason": "NO_PROPOSED_1_5G_LOADS"}
    active = [item for item in proposed.get("patch_allocations", [])
        if item["target_normal_pressure_n"] is not None
        and item["target_normal_pressure_n"] >= LIMITS.allocated_contact_floor_n]
    load = sum(item["target_normal_pressure_n"] for item in active)
    normal_z = [abs(item["normal_on_object_world"][2]) for item in active]
    row.update(geometry_error=error, closure=closures, contacts=annotated,
        allocation_1_5g=proposed, allocation_1g=one_g, hypothetical_fixed_normal_1g=fixed,
        allocated_fingers=sorted({item["finger"] for item in active}),
        allocated_contacts=[annotated[item["input_index"]] for item in active],
        maximum_allocated_abs_normal_z=max(normal_z, default=None),
        weighted_allocated_abs_normal_z=None if load == 0. else sum(
            item["target_normal_pressure_n"] * abs(item["normal_on_object_world"][2]) for item in active) / load,
        closed_hand_table_clearance_m=minimum_clearance(model, data, table, hand_geoms),
        maximum_contact_penetration_m=max([-item["distance_m"] for item in annotated] + [0.]),
        compression_diagnostics=compression_diagnostics(model, data, hand, annotated, proposed))
    row["failure_reasons"].extend(geometry_reasons(row))
    row["failure_reasons"] = list(dict.fromkeys(row["failure_reasons"]))
    row["static_screen_pass"] = not row["failure_reasons"]
    assert_private_state(model, data)
    assert_private_state(model, nominal_kin.scratch)
    assert_private_state(model, pose_kin.scratch)
    row["authored_target_qpos_unchanged"] = True
    return row


def rank_key(row):
    loads = row["allocation_1_5g"]["target_pressures_n"]
    return (max(loads.values()), loads["thumb"], row["allocation_1_5g"]["minimum_total_normal_load_n"],
        -row["closed_hand_table_clearance_m"], row["penetration_m"])


def proposed_profile(row, original, ranked_path):
    candidate = row["candidate"]
    rotated = bool(candidate["roll_deg"] or candidate["yaw_deg"])
    profile = {**original, "thumb_opposition_rad": candidate["opposition_rad"],
        "contact_closure": row["closure"],
        "force_targets_n": {finger: row["allocation_1_5g"]["target_pressures_n"][finger] for finger in FINGERS},
        "nominal_shallow_penetration_m": row["penetration_m"],
        "geometry_evidence": str(ranked_path.relative_to(PROJECT)),
        "geometry_evidence_sha256": sha256(ranked_path), "geometry_scope": SCOPE,
        "static_sidewall_metrics": {
            "allocated_max_abs_normal_z": row["maximum_allocated_abs_normal_z"],
            "allocated_weighted_abs_normal_z": row["weighted_allocated_abs_normal_z"],
            "closed_hand_table_clearance_m": row["closed_hand_table_clearance_m"],
            "open_approach_robot_table_clearance_m": row["open_approach_robot_table_clearance_m"],
            "open_approach_hand_mesh_clearance_m": row["open_approach_hand_target_clearance_m"]}}
    return {"candidate": candidate, "scope": SCOPE, "static_profile_only": True,
        "r14_fixed_orientation_schema_compatible": not rotated,
        "requires_explicit_wrist_orientation_controller": rotated,
        "desired_world_wrist_rotation": row["wrist_rotation_world"],
        "warning": "Do not run a rotated profile with a fixed-orientation controller; that would test a different geometry.",
        "ycb_can": profile}


def run(output, profile_path, oppositions, rolls, yaws, penetrations):
    candidates = [Candidate(*values) for values in product(oppositions, rolls, yaws)]
    if not candidates or len(candidates) > 200:
        raise ValueError("Require 1 to 200 nominal candidates")
    if not penetrations or any(not np.isfinite(value) or not .0001 <= value <= .0005 for value in penetrations):
        raise ValueError("Shallow penetration must remain in [0.1, 0.5] mm")
    profile = json.loads(Path(profile_path).read_text())["ycb_can"]
    baseline = Candidate(profile["thumb_opposition_rad"], 0., 0.)
    if baseline not in candidates:
        candidates.insert(0, baseline)
    before = source_snapshot(profile_path)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "frozen_geometry_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    model = mujoco.MjModel.from_xml_path(str(build_pick_place_scene(
        output / "nominal_scene.xml", "ycb_can", physics_profile="physical-v6")))
    nominal_kin = RobotKinematics(model)
    pose_kin = V7Kinematics(model, position_tolerance=.0001, orientation_tolerance=.00075)
    reader, surface = HandContactReader(model), PublicMeshSurface.from_public_kind("ycb_can")
    rows, started = [], time.perf_counter()
    for index, candidate in enumerate(candidates):
        for penetration in penetrations:
            row = evaluate_candidate(model, nominal_kin, pose_kin, reader, surface, profile, candidate, penetration)
            row.update(candidate_index=index, is_r14_nominal_orientation=(candidate == baseline))
            rows.append(row)
        if index % 10 == 0:
            print(json.dumps({"candidate": index + 1, "total": len(candidates),
                "rows": len(rows), "static_passes": sum(row["static_screen_pass"] for row in rows)}), flush=True)
    after = source_snapshot(profile_path)
    if before != after:
        raise RuntimeError("Static geometry dependencies changed during analysis")
    baselines = {row["penetration_m"]: row for row in rows if row["is_r14_nominal_orientation"]}
    for row in rows:
        reference = baselines[row["penetration_m"]]
        if row.get("allocation_1_5g", {}).get("feasible") and reference.get("allocation_1_5g", {}).get("feasible"):
            current = row["allocation_1_5g"]["target_pressures_n"]
            original = reference["allocation_1_5g"]["target_pressures_n"]
            row["same_penetration_r14_load_difference_n"] = {
                "thumb": current["thumb"] - original["thumb"],
                "maximum_channel": max(current.values()) - max(original.values())}
    ranked = sorted((row for row in rows if row["static_screen_pass"]), key=rank_key)
    summary = {"format": "public_ycb_nominal_geometry_r15", "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
        "integration_steps": 0, "target_qpos_writes": 0, "dynamic_success": None,
        "nominal_authored_origin_m": ORIGIN.tolist(), "public_mesh_prior": surface.source,
        "mass_prior_kg": surface.mass_kg, "nominal_pair_friction_prior": float(np.sqrt(surface.friction_coefficient)),
        "limits": asdict(LIMITS), "strict_task_channel_limit_n_unchanged": 12.,
        "probe_and_hold_gates_modified": False,
        "orientation_convention": "R_world = Rz_world(yaw) @ Rx_world(roll) @ R14_nominal_rotation",
        "profile_path": str(Path(profile_path).resolve()), "profile_sha256": sha256(profile_path),
        "source_hashes_before": before, "source_hashes_after": after, "source_unchanged": True,
        "candidate_count": len(candidates), "row_count": len(rows), "static_pass_count": len(ranked),
        "failure_reason_counts": dict(Counter(reason for row in rows for reason in row["failure_reasons"])),
        "ranking": "minimize max 1.5g finger load, then thumb load, then total load; table clearance tie-break",
        "wall_seconds": time.perf_counter() - started, "rows": rows}
    write_json(output / "summary.json", summary)
    write_json(output / "ranked.json", ranked)
    if ranked:
        selected = [("lowest_peak_load", ranked[0])]
        fixed = [row for row in ranked if not row["candidate"]["roll_deg"] and not row["candidate"]["yaw_deg"]]
        if fixed:
            selected.append(("fixed_orientation_candidate", fixed[0]))
        for label, row in selected:
            write_json(output / (label + "_profile.json"), proposed_profile(row, profile, output / "ranked.json"))
    print(json.dumps({"summary": str(output / "summary.json"), "summary_sha256": sha256(output / "summary.json"),
        "candidate_count": len(candidates), "row_count": len(rows), "static_pass_count": len(ranked),
        "failure_reason_counts": summary["failure_reason_counts"],
        "top": [{"candidate": row["candidate"], "penetration_m": row["penetration_m"],
            "loads_1_5g_n": row["allocation_1_5g"]["target_pressures_n"],
            "closed_table_clearance_m": row["closed_hand_table_clearance_m"]} for row in ranked[:5]]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROJECT / "config/public_ycb_sidewall_20261021.json")
    parser.add_argument("--oppositions", nargs="+", type=float, default=[.9, 1., 1.1, 1.2, 1.3])
    parser.add_argument("--roll-deg", nargs="+", type=float, default=[-6., 0., 6.])
    parser.add_argument("--yaw-deg", nargs="+", type=float, default=[-6., 0., 6.])
    parser.add_argument("--penetrations", nargs="+", type=float, default=[.00025, .0005])
    args = parser.parse_args()
    run(args.output.resolve(), args.profile.resolve(), args.oppositions, args.roll_deg, args.yaw_deg, args.penetrations)


if __name__ == "__main__":
    main()
