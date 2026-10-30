#!/usr/bin/env python3
"""Recompute the public baseline's metrics and audit its complete physical evidence."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_public_ycb import schedule_for, summarize
from evaluate_v7_robustness import atomic_json, sha256
from public_ycb_evidence import load_evidence_model
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_controller import completed_success
from kuka_sharpa_grasp.pick_place_evaluator import held_lift_evidence


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text())
    protocol = summary["protocol"]
    assert summary["complete"] and not protocol["source_changed_during_run"]
    assert protocol["public_meshes_unchanged"]
    assert protocol["source_hashes_before"] == protocol["source_hashes_after"]
    schedule = protocol["schedule"]
    assert schedule == schedule_for(protocol["selection_seed"], schedule[0]["pose_seed"], len(schedule))
    assert [(row["index"], row["kind"], row["pose_seed"]) for row in summary["rows"]] == [
        (case["index"], case["kind"], case["pose_seed"]) for case in schedule]
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        assert archive.testzip() is None
        for relative, expected in protocol["source_hashes_before"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == expected
    for name, expected in protocol.get("derived_controller_sources", {}).items():
        assert Path(name).name == name and sha256(directory / name) == expected
    for mesh in protocol["public_mesh_inventory"].values():
        path = PROJECT / mesh["path"]
        assert sha256(path) == mesh["sha256"] and path.stat().st_size == mesh["bytes"]
    totals = {"gzip_artifacts": 0, "native_samples": 0, "png_frames": 0}
    audited = []
    for row in summary["rows"]:
        case_directory = directory / row["path"]
        blobs = {}
        for name, artifact in row["artifacts"].items():
            path = case_directory / artifact["path"]
            assert path.stat().st_size == artifact["bytes"] and sha256(path) == artifact["sha256"]
            with gzip.open(path, "rt") as handle:
                blobs[name] = json.load(handle)
            totals["gzip_artifacts"] += 1
        if row["runner_error"] is not None:
            assert row["success"] is row["held_lift"] is None
            audited.append({"path": row["path"], "outcome": "unknown_runtime_error"})
            continue
        result = json.loads((case_directory / "result.json").read_text())
        trace, native = blobs["trace"], blobs["native"]
        assert native and trace
        assert result["maximum_attempts"] == protocol["maximum_attempts_per_episode"] == 2
        assert len(result["attempts"]) <= 2
        audit = row["native_audit"]
        assert audit == result["native_audit"] and audit["external_target_qpos_writes"] == 0
        assert audit["integration_steps"] == audit["native_samples"] == len(native)
        scenario = Scenario(row["pose_seed"], row["kind"])
        assert np.allclose(audit["initial_target_qpos"], [*scenario.position, 1., 0., 0., 0.], atol=1e-12, rtol=0)
        times = np.asarray([sample["timestamp"] for sample in native])
        assert abs(times[0] - .002) < 1e-9
        assert np.allclose(np.diff(times), .002, atol=1e-9, rtol=0)
        assert abs(times[-1] - result["duration"]) < 1e-9
        physical = json.loads((case_directory / "physical_model.json").read_text())
        assert sha256(case_directory / "scene.xml") == physical["scene_sha256"]
        parameters = next(spec for spec in protocol["physics_profile"]["objects"] if spec["kind"] == row["kind"])
        assert abs(physical["mass_kg"] - parameters["mass_kg"]) < 1e-12
        assert np.allclose(physical["friction"], parameters["friction"], rtol=0, atol=1e-12)
        model = load_evidence_model(case_directory / "scene.xml")
        target, table = model.geom("target_geom").id, model.geom("table").id
        hand = {index for index in range(model.ngeom) if (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])) or "").startswith("left_")
            and model.geom_contype[index] != 0}
        hand_channel = {}
        for index in hand:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])) or ""
            hand_channel[index] = next((finger for finger in ("thumb", "index", "middle", "ring", "pinky")
                if body_name.startswith("left_" + finger + "_")), "palm")
        peaks = {finger: 0. for finger in ("thumb", "index", "middle", "ring", "pinky", "palm")}
        for sample in native:
            channels = dict.fromkeys(peaks, 0.)
            for contact in sample["native_contacts"]:
                pair = set(contact["geom_ids"])
                side = pair & hand
                if len(side) == 1:
                    channels[hand_channel[next(iter(side))]] += abs(contact["force_torque_contact_frame"][0])
            for finger, force in channels.items():
                peaks[finger] = max(peaks[finger], force)
        indexed = {round(sample["timestamp"], 9): sample for sample in native}
        final_stable_since = None
        for sample in trace:
            timestamp = sample["timestamp"]
            measured = indexed[round(timestamp, 9)]
            labels = sample["oracle"]
            assert np.allclose(labels["position"], measured["target_position_world_m"], atol=1e-12, rtol=0)
            velocity = np.asarray(measured["target_spatial_velocity_world"])
            assert abs(np.linalg.norm(velocity[3:]) - labels["linear_speed_m_s"]) < 1e-12
            assert abs(np.linalg.norm(velocity[:3]) - labels["angular_speed_rad_s"]) < 1e-12
            contacts = measured["native_contacts"]
            hand_contact = any(target in contact["geom_ids"] and bool(set(contact["geom_ids"]) & hand)
                               for contact in contacts)
            support = any(set(contact["geom_ids"]) == {target, table}
                          and abs(contact["force_torque_contact_frame"][0]) > .05 for contact in contacts)
            assert hand_contact == labels["hand_contact"] and support == labels["table_support"]
            valid = (labels["release_commanded"] and labels["placement_error"] < .045
                and abs(labels["position"][2] - .38) < .025 and labels["linear_speed_m_s"] < .04
                and labels["angular_speed_rad_s"] < .2 and labels["hand_clearance_m"] > .10
                and not hand_contact and support)
            final_stable_since = (timestamp if final_stable_since is None else final_stable_since) if valid else None
        lift = held_lift_evidence(trace)
        assert lift == result["actual_held_lift_evidence"] and lift["observed"] == row["held_lift"]
        stable_seconds = 0. if final_stable_since is None else times[-1] - final_stable_since
        final_physical_success = stable_seconds >= .5 and lift["observed"]
        recomputed = completed_success(result["transitions"], final_physical_success, result["failure"])
        assert recomputed == row["success"] == result["success"]
        for frame in result["frames"]:
            pixels = np.asarray(Image.open(case_directory / frame["file"]))
            assert pixels.shape == (480, 640, 3) and pixels.std() > 1
            assert 0 <= frame["timestamp"] <= result["duration"] + 1e-9
            totals["png_frames"] += 1
        totals["native_samples"] += len(native)
        audited.append({"path": row["path"], "outcome": bool(recomputed),
            "held_lift": lift["observed"], "stable_released_seconds": stable_seconds,
            "native_samples": len(native), "external_target_qpos_writes": 0,
            "metric_recompute_equal": True,
            "peak_native_hand_channel_force_n": max(peaks.values()),
            "peak_native_hand_channels_n": peaks,
            "within_additional_12n_channel_reference": max(peaks.values()) <= 12.,
            "robot_table_contact_samples": sum(bool(sample["robot_table_contacts"]) for sample in trace)})
        print(row["path"], recomputed, "stable_release", stable_seconds, flush=True)
    recomputed_summary = summarize(summary["rows"], len(schedule))
    for key, value in recomputed_summary.items():
        assert summary[key] == value
    receipt = {"format": "current_random_ycb_baseline_physical_evidence_audit",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_sha256": sha256(directory / "summary.json"),
        "verification_script_sha256": sha256(Path(__file__)),
        "portable_model_loader_sha256": sha256(PROJECT / "scripts" / "public_ycb_evidence.py"),
        "archived_scene_policy": "Original XML bytes preserved; only compiler meshdir relocated in memory to packaged vendor assets",
        "complete": True, "scheduled_episodes": len(schedule), **totals,
        "frozen_source_archive_sha256": sha256(directory / "frozen_source.zip"),
        "source_hashes_before_equal_after": True, "public_mesh_hashes_and_sizes_equal": True,
        "native_samples_contiguous_2ms": True, "all_known_metrics_recomputed_equal": True,
        "additional_12n_force_reference": {
            "within_reference_episodes": sum(item.get("within_additional_12n_channel_reference", False) for item in audited),
            "task_success_and_within_reference": sum(item.get("outcome") is True and item.get("within_additional_12n_channel_reference", False) for item in audited),
            "denominator": len(schedule),
            "scope": "Independent native hand-side normal-force sum against all non-hand geometry; comparison only, not the original V6 task acceptance gate"},
        "episodes": audited, "scope": "Recorded simulation evidence and source/archive audits; failed objects remain in the denominator, with no unseen-object or hardware claim."}
    atomic_json(directory / "verification_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    receipt = verify(args.directory.resolve())
    print(json.dumps({key: value for key, value in receipt.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
