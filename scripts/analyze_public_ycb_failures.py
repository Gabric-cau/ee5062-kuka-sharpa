#!/usr/bin/env python3
"""Post-hoc public-YCB phase diagnosis; never send commands or alter a scene."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_v7_robustness import atomic_json, sha256
from kuka_sharpa_grasp.pick_place import RobotKinematics


def load_gzip(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def diagnose(directory):
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["complete"]
    receipt = json.loads((directory / "verification_receipt.json").read_text())
    assert receipt["complete"] and receipt["all_known_metrics_recomputed_equal"]
    episodes, failures = [], Counter()
    for row in summary["rows"]:
        if row["runner_error"]:
            episodes.append({"path": row["path"], "runner_error": row["runner_error"]})
            continue
        case = directory / row["path"]
        result = json.loads((case / "result.json").read_text())
        native = load_gzip(case / "native.json.gz")
        trace = load_gzip(case / "trace.json.gz")
        unique_trace = {round(sample["timestamp"], 9): sample for sample in trace}
        model = mujoco.MjModel.from_xml_path(str(case / "scene.xml"))
        kin = RobotKinematics(model)
        hand = {}
        for gid in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if body.startswith("left_") and model.geom_contype[gid] != 0:
                hand[gid] = next((finger for finger in ("thumb", "index", "middle", "ring", "pinky")
                    if body.startswith("left_" + finger + "_")), "palm")
        segments = []
        for sample in native:
            measured = unique_trace[round(sample["timestamp"], 9)]
            if not segments or segments[-1]["state"] != sample["state"]:
                segments.append({"state": sample["state"], "samples": [], "trace": []})
            segments[-1]["samples"].append(sample)
            segments[-1]["trace"].append(measured)
        phases = []
        for segment in segments:
            samples, traces = segment["samples"], segment["trace"]
            peak_channel, peak_time, peak_pairs = 0., None, []
            for sample in samples:
                forces, pairs = defaultdict(float), []
                for contact in sample["native_contacts"]:
                    side = set(contact["geom_ids"]) & set(hand)
                    if len(side) == 1:
                        finger = hand[next(iter(side))]
                        force = abs(contact["force_torque_contact_frame"][0])
                        forces[finger] += force
                        pairs.append({"finger": finger, "normal_force_n": force,
                            "geoms": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                                      for gid in contact["geom_ids"]],
                            "distance_m": contact["distance_m"]})
                current = max(forces.values(), default=0.)
                if current > peak_channel:
                    peak_channel, peak_time, peak_pairs = current, sample["timestamp"], pairs
            final = traces[-1]
            phases.append({"state": segment["state"], "start_s": samples[0]["timestamp"],
                "end_s": samples[-1]["timestamp"], "native_samples": len(samples),
                "peak_channel_n": peak_channel, "peak_time_s": peak_time, "peak_contact_pairs": peak_pairs,
                "end_limited_command_tracking_error_m": final["arm_tracking_error"],
                "end_wrist_position_m": final["wrist_position"],
                "end_target_position_m": final["oracle"]["position"],
                "end_xy_error_m": final["oracle"]["placement_error"],
                "end_hand_clearance_m": final["oracle"]["hand_clearance_m"],
                "end_target_angular_speed_rad_s": final["oracle"]["angular_speed_rad_s"],
                "end_target_linear_speed_m_s": final["oracle"]["linear_speed_m_s"],
                "end_table_load_n": final["table_load_sensor"]["raw_load_n"],
                "support_fraction": sum(sample["oracle"]["table_support"] for sample in traces) / len(traces),
                "hand_contact_fraction": sum(sample["oracle"]["hand_contact"] for sample in traces) / len(traces)})
        attempts = []
        for attempt in result["attempts"]:
            timestamp = attempt["belief_timestamp"]
            same_time = min(native, key=lambda sample: abs(sample["timestamp"] - timestamp))
            assert abs(same_time["timestamp"] - timestamp) < 1e-9
            candidate = attempt["selected_candidate"]
            failure_transition = next((item for item in result["transitions"]
                if item["timestamp"] >= timestamp and item["reason"] == attempt["failure"]
                and item["state"] == "RECOVER"), None)
            failed_phase = None if failure_transition is None else failure_transition["previous"]
            if attempt["failure"] is not None:
                failures[(row["kind"], failed_phase, attempt["failure"])] += 1
            attempts.append({**attempt, "failure_phase": failed_phase,
                "same_time_true_body_position_m": same_time["target_position_world_m"],
                "candidate_origin_xy_error_mm": None if candidate is None else 1000 * float(np.linalg.norm(
                    np.asarray(candidate[:2]) - same_time["target_position_world_m"][:2]))})
        observations = []
        for observation in result["vision_updates"]:
            same_time = min(native, key=lambda sample: abs(sample["timestamp"] - observation["timestamp"]))
            assert abs(same_time["timestamp"] - observation["timestamp"]) < 1e-9
            centroid = observation["centroid"]
            observations.append({**observation,
                "same_time_true_body_position_m": same_time["target_position_world_m"],
                "centroid_minus_body_xy_mm": None if centroid is None else (
                    1000 * (np.asarray(centroid[:2]) - same_time["target_position_world_m"][:2])).tolist(),
                "metric_scope": "Visible surface centroid versus body origin; not a localization-origin error",
                "held_candidate_origin_xy_error_mm": None})
        episodes.append({"path": row["path"], "kind": row["kind"], "seed": row["pose_seed"],
            "success": row["success"], "held_lift": row["held_lift"], "failure": result["failure"],
            "attempts": attempts, "phases": phases, "camera_observations": observations})
    return {"format": "public_ycb_readonly_phase_diagnosis_r1",
        "input_summary_sha256": sha256(directory / "summary.json"),
        "input_verified_receipt_sha256": sha256(directory / "verification_receipt.json"),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "all_scheduled_episodes_retained": len(episodes) == summary["scheduled"],
        "failure_counts": [{"kind": kind, "phase": phase, "failure": failure, "count": count}
            for (kind, phase, failure), count in failures.items()],
        "unknown_fields": ["Held RGB-D candidate origin is not present in the original V6 result log; do not infer it from the surface centroid"],
        "scope": "Evaluation-only native pose/velocity/contact readback; no live command, physics, mesh, or score change",
        "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.directory.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result["failure_counts"], indent=2))


if __name__ == "__main__":
    main()
