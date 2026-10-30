#!/usr/bin/env python3
"""Independent raw R29 task ordering and final release audit, no integration.

Loads the frozen compiled model for geometry identity and kinematic distance
queries only. It neither imports a controller nor steps/forwards dynamics.
All target state/contact identity is privileged, post-run evaluation truth.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np


EXPECTED = ("PRELOAD", "TRIAL_LIFT", "LIFT", "TRANSFER", "PLACE", "SUPPORT_SETTLE",
            "RELEASE", "RETRACT", "VERIFY")
DT = .002


def require(flag, message):
    if not flag:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_interval(records):
    start, previous, longest, first_qualified = None, None, 0., None
    for timestamp, qualifies in records:
        if previous is not None and abs(timestamp - previous - DT) > 1e-9:
            start = None
        if qualifies:
            start = timestamp if start is None else start
            first_qualified = timestamp if first_qualified is None else first_qualified
            longest = max(longest, timestamp - start)
        else:
            start = None
        previous = timestamp
    return dict(longest_seconds=longest, first_qualified_timestamp_s=first_qualified)


def audit(directory):
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in ("summary.json", "protocol.json", "result.json",
        "native.json.gz", "compiled_model.mjb", "frozen_source.zip")}
    before = {name: sha256(path) for name, path in paths.items()}
    result, protocol, summary = (json.loads(paths[name].read_text()) for name in
                                 ("result.json", "protocol.json", "summary.json"))
    require(summary["protocol"] == protocol and summary["complete"] is True, "incomplete metadata")
    require(protocol["source_hashes_before"] == protocol["source_hashes_after"]
            and protocol["source_changed_during_run"] is False, "source mutation")
    require(protocol["input_hashes"] == protocol["input_hashes_after"]
            and protocol["inputs_changed_during_run"] is False, "input mutation")
    with zipfile.ZipFile(paths["frozen_source.zip"]) as archive:
        require(set(archive.namelist()) == set(protocol["source_hashes_before"])
                and len(archive.namelist()) == len(set(archive.namelist())), "source members mismatch")
        for name, digest in protocol["source_hashes_before"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "source bytes mismatch")
    native_record = summary["rows"][0]["artifacts"]["native"]
    require(native_record["path"] == "native.json.gz" and native_record["sha256"] == before["native.json.gz"]
            and native_record["bytes"] == paths["native.json.gz"].stat().st_size, "native identity mismatch")
    require(protocol["compiled_model_sha256"] == before["compiled_model.mjb"], "compiled model identity mismatch")
    transitions = result["transitions"]
    require(tuple(row["state"] for row in transitions) == EXPECTED, "unexpected task order/recovery")
    times = {row["state"]: float(row["timestamp_s"]) for row in transitions}
    require(all(second["timestamp_s"] > first["timestamp_s"] for first, second in zip(transitions, transitions[1:])),
            "reversed task transitions")
    model = mujoco.MjModel.from_binary_path(str(paths["compiled_model.mjb"]))
    require(abs(float(model.opt.timestep) - DT) < 1e-12, "wrong native timestep")
    target, table = model.geom("target_geom").id, model.geom("table").id
    target_joint = model.joint("target_free").id
    target_address = int(model.jnt_qposadr[target_joint])
    hand = {g for g in range(model.ngeom) if (model.body(int(model.geom_bodyid[g])).name or "").startswith("left_")}
    clearance_geoms = [g for g in hand if model.geom_contype[g] or model.geom_conaffinity[g]]
    robot_joints = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(model.njnt)
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    scratch = mujoco.MjData(model)
    with gzip.open(paths["native.json.gz"], "rt") as stream:
        native = json.load(stream)
    require(len(native) == result["native_audit"]["integration_steps"] == result["native_audit"]["native_samples"],
            "native denominator mismatch")
    initial_height = float(result["native_audit"]["initial_target_qpos"][2])
    zone = np.asarray(result["zone"], dtype=float)
    held_before, held_lift_phase, released_verify = [], [], []
    phase_heights, transfer = {}, []
    last_release, minimum_verify_clearance, maximum_phase_label_error = None, float("inf"), 0
    for index, row in enumerate(native):
        timestamp = float(row["timestamp"])
        require(abs(timestamp - (index + 1) * DT) < 1e-8, "native time gap/duplicate")
        qpos = np.asarray(row["target_qpos"], dtype=float)
        position = np.asarray(row["target_position_world_m"], dtype=float)
        spatial = np.asarray(row["target_spatial_velocity_world"], dtype=float)
        require(qpos.shape == (7,) and position.shape == (3,) and spatial.shape == (6,)
                and np.isfinite(np.r_[qpos, position, spatial]).all(), "invalid target state")
        require(np.allclose(qpos[:3], position, atol=1e-10, rtol=0.), "free body position and refreshed position disagree")
        actual_phase = next((phase["state"] for phase in reversed(transitions)
                             if timestamp > phase["timestamp_s"] + 1e-9), "REPLAY")
        require(row["state"] == actual_phase, "native phase differs from actual transition/step interval")
        hand_normals, table_normals = [], []
        hand_contact = False
        for contact in row["native_contacts"]:
            pair = set(contact["geom_ids"])
            force = np.asarray(contact["force_torque_contact_frame"], dtype=float)
            require(force.shape == (6,) and np.isfinite(force).all(), "invalid contact wrench")
            if target in pair and pair & hand:
                hand_contact = True
                hand_normals.append(abs(float(force[0])))
            if pair == {target, table}:
                table_normals.append(abs(float(force[0])))
        loaded_hand = max(hand_normals, default=0.) > .05
        loaded_table = max(table_normals, default=0.) > .05
        held = position[2] > initial_height + .07 and loaded_hand and not loaded_table
        phase_heights.setdefault(actual_phase, []).append(float(position[2] - initial_height))
        if timestamp <= times["TRANSFER"] + 1e-9:
            held_before.append((timestamp, held))
        if actual_phase == "LIFT":
            held_lift_phase.append((timestamp, held))
        if actual_phase == "TRANSFER":
            transfer.append(dict(timestamp_s=timestamp, position_world_m=position.tolist(), held=held,
                                 hand_loaded=loaded_hand, table_loaded=loaded_table))
        if actual_phase == "VERIFY":
            require(set(row["robot_joint_position_rad"]) == set(robot_joints), "incomplete robot geometry state")
            for name, address in robot_joints.items():
                scratch.qpos[address] = row["robot_joint_position_rad"][name]
            scratch.qpos[target_address:target_address + 7] = qpos
            require(np.isfinite(scratch.qpos).all(), "nonfinite scratch geometry state")
            mujoco.mj_kinematics(model, scratch)
            clearance = min(float(mujoco.mj_geomDistance(model, scratch, target, g, 1., None)) for g in clearance_geoms)
            minimum_verify_clearance = min(minimum_verify_clearance, clearance)
            xy_error = float(np.linalg.norm(position[:2] - zone[:2]))
            height_error = abs(float(position[2] - zone[2]))
            linear, angular = float(np.linalg.norm(spatial[3:])), float(np.linalg.norm(spatial[:3]))
            qualified = (timestamp > times["RELEASE"] and not hand_contact and loaded_table
                         and xy_error < .025 and height_error < .025 and linear < .04 and angular < .2 and clearance > .1)
            released_verify.append((timestamp, qualified))
            last_release = dict(timestamp_s=timestamp, target_position_world_m=position.tolist(),
                xy_error_m=xy_error, height_error_m=height_error, linear_speed_m_s=linear,
                angular_speed_rad_s=angular, hand_clearance_m=clearance,
                hand_contact=hand_contact, loaded_table=loaded_table, qualifies=bool(qualified))
    before_transfer, during_lift, after_verify = (longest_interval(rows) for rows in
                                                (held_before, held_lift_phase, released_verify))
    checks = dict(actual_held_before_transfer=before_transfer["longest_seconds"] + 1e-9 >= .1,
                  actual_held_during_lift_phase=during_lift["longest_seconds"] + 1e-9 >= .1,
                  all_transfer_samples_held=bool(transfer) and all(row["held"] for row in transfer),
                  released_stable_after_verify_3s=after_verify["longest_seconds"] + 1e-9 >= 3.)
    after = {name: sha256(path) for name, path in paths.items()}
    require(before == after, "input changed during audit")
    return dict(format="independent_r29_task_sequence_audit_v1", accepted_evidence=True,
        sequence_checks_passed=all(checks.values()), checks=checks, recorded_task_success=result["success"],
        task_success_scope="Sequence supplement only; combine with independent native/command/strict audit",
        native_samples=len(native), transitions=transitions, held_before_transfer=before_transfer,
        held_during_lift_phase=during_lift, released_after_verify=after_verify,
        maximum_object_rise_by_phase_m={name: max(values) for name, values in phase_heights.items()},
        transfer=dict(samples=len(transfer), held_samples=sum(row["held"] for row in transfer),
            first=transfer[0] if transfer else None, last=transfer[-1] if transfer else None,
            end_xy_error_m=None if not transfer else float(np.linalg.norm(np.asarray(transfer[-1]["position_world_m"])[:2] - zone[:2]))),
        final_release=last_release, minimum_verify_hand_clearance_m=minimum_verify_clearance,
        verified_source_files=len(protocol["source_hashes_before"]), input_hashes_before=before,
        input_hashes_after=after, audit_script_sha256=sha256(Path(__file__)),
        limitations=["One reused seed and 22s validated motor prefix; no randomized success rate or new grasp generator.",
            "Recorded RGB-D anchor with robot-only rigid propagation; no new visual measurement during continuation.",
            "Target-specific virtual instrumented table signal is explicitly privileged sensing.",
            "Post-run target state is used only for evaluation; this script does not test causal control isolation.",
            "No dynamics stepping/forwarding; clearance uses scratch kinematics and native actual qpos.",
            "Controller row state is entry label before any same-row transition; native state denotes applied phase."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False))
    return int(not report["sequence_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
