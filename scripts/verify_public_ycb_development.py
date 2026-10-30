#!/usr/bin/env python3
"""Verify frozen public-object development evidence, including negative trials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_v7_robustness import atomic_json, sha256
from public_ycb_evidence import load_evidence_model
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_controller import completed_success
from kuka_sharpa_grasp.pick_place_evaluator import held_lift_evidence


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text())
    protocol = summary["protocol"]
    assert summary["complete"] and not protocol["source_changed_during_run"]
    assert protocol["source_hashes_before"] == protocol["source_hashes_after"]
    assert len(summary["rows"]) == len(protocol["schedule"])
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        for relative, expected in protocol["source_hashes_before"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == expected
    for mesh in protocol["public_meshes"].values():
        path = PROJECT / mesh["path"]
        assert path.stat().st_size == mesh["bytes"] and sha256(path) == mesh["sha256"]
    audited, native_count, artifacts = [], 0, 0
    for planned, row in zip(protocol["schedule"], summary["rows"], strict=True):
        assert all(row[key] == value for key, value in planned.items())
        case = directory / row["path"]
        blobs = {}
        for name, artifact in row["artifacts"].items():
            path = case / artifact["path"]
            assert path.stat().st_size == artifact["bytes"] and sha256(path) == artifact["sha256"]
            with gzip.open(path, "rt") as handle:
                blobs[name] = json.load(handle)
            artifacts += 1
        if row["runner_error"]:
            assert row["success"] is None and row["held_lift"] is None
            audited.append({"path": row["path"], "outcome": None, "unknown": True})
            continue
        result = json.loads((case / "result.json").read_text())
        assert len(result["attempts"]) <= result["maximum_attempts"] == protocol["maximum_attempts"] == 2
        if row.get("derived_controller_sha256"):
            assert sha256(case / "derived_controller.py") == row["derived_controller_sha256"]
        trace, native, audit = blobs["trace"], blobs["native"], row["native_audit"]
        assert trace and native and audit["external_target_qpos_writes"] == 0
        assert len(native) == audit["integration_steps"] == audit["native_samples"]
        assert np.allclose(audit["initial_target_qpos"], [*Scenario(row["seed"], row["kind"]).position, 1., 0., 0., 0.], atol=1e-12, rtol=0)
        times = np.asarray([sample["timestamp"] for sample in native])
        assert abs(times[0] - .002) < 1e-9 and np.allclose(np.diff(times), .002, atol=1e-9, rtol=0)
        assert abs(times[-1] - result["duration"]) < 1e-9
        model = load_evidence_model(case / "scene.xml")
        target, table = model.geom("target_geom").id, model.geom("table").id
        hand = {}
        for gid in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[gid])).name
            if body.startswith("left_") and model.geom_contype[gid]:
                hand[gid] = next((finger for finger in ("thumb", "index", "middle", "ring", "pinky")
                    if body.startswith("left_" + finger + "_")), "palm")
        peaks = dict.fromkeys(("thumb", "index", "middle", "ring", "pinky", "palm"), 0.)
        by_phase, maximum_proximal, indexed = {}, 0., {}
        for sample in native:
            channels = dict.fromkeys(peaks, 0.)
            for contact in sample["native_contacts"]:
                side = set(contact["geom_ids"]) & hand.keys()
                if len(side) != 1:
                    continue
                gid = next(iter(side))
                channels[hand[gid]] += abs(contact["force_torque_contact_frame"][0])
                if "_MP" in model.body(int(model.geom_bodyid[gid])).name:
                    maximum_proximal = max(maximum_proximal, -contact["distance_m"])
            for finger, pressure in channels.items():
                peaks[finger] = max(peaks[finger], pressure)
            by_phase[sample["state"]] = max(by_phase.get(sample["state"], 0.), max(channels.values()))
            indexed[round(sample["timestamp"], 9)] = sample
        stable_since = None
        for sample in trace:
            measured = indexed[round(sample["timestamp"], 9)]
            labels = sample["oracle"]
            assert np.allclose(labels["position"], measured["target_position_world_m"], atol=1e-12, rtol=0)
            velocity = measured["target_spatial_velocity_world"]
            assert abs(np.linalg.norm(velocity[:3]) - labels["angular_speed_rad_s"]) < 1e-12
            assert abs(np.linalg.norm(velocity[3:]) - labels["linear_speed_m_s"]) < 1e-12
            hand_contact = any(target in contact["geom_ids"] and bool(set(contact["geom_ids"]) & hand.keys())
                for contact in measured["native_contacts"])
            supported = any(set(contact["geom_ids"]) == {target, table}
                and abs(contact["force_torque_contact_frame"][0]) > .05 for contact in measured["native_contacts"])
            assert labels["hand_contact"] == hand_contact and labels["table_support"] == supported
            assert labels["lifted"] == (measured["target_position_world_m"][2]
                > native[0]["target_position_world_m"][2] + .07)
            valid = (labels["release_commanded"] and labels["placement_error"] < .045
                and abs(labels["position"][2] - .38) < .025 and labels["linear_speed_m_s"] < .04
                and labels["angular_speed_rad_s"] < .2 and labels["hand_clearance_m"] > .10
                and not hand_contact and supported)
            stable_since = (sample["timestamp"] if stable_since is None else stable_since) if valid else None
        lift = held_lift_evidence(trace)
        assert lift == result["actual_held_lift_evidence"] and lift["observed"] == row["held_lift"]
        stable = 0. if stable_since is None else times[-1] - stable_since
        success = completed_success(result["transitions"], stable >= .5 and lift["observed"], result["failure"])
        assert success == row["success"] == result["success"]
        native_count += len(native)
        audited.append({"path": row["path"], "outcome": success, "held_lift": lift["observed"],
            "unknown": False, "native_samples": len(native), "stable_release_s": stable,
            "peak_native_hand_channel_n": max(peaks.values()), "native_channel_peaks_n": peaks,
            "native_phase_peaks_n": by_phase, "maximum_middle_link_penetration_mm": maximum_proximal * 1000,
            "additional_12n_reference": max(peaks.values()) <= 12.,
            "no_external_target_qpos_writes": True})
        print(row["path"], "task", success, "held", lift["observed"],
            "peak channel N", round(max(peaks.values()), 3), "phase", by_phase, flush=True)
    receipt = {"format": "public_ycb_development_native_verification_r1", "complete": True,
        "summary_sha256": sha256(directory / "summary.json"),
        "verification_script_sha256": sha256(Path(__file__)),
        "portable_model_loader_sha256": sha256(PROJECT / "scripts" / "public_ycb_evidence.py"),
        "archived_scene_policy": "Original XML bytes preserved; only compiler meshdir relocated in memory to packaged vendor assets",
        "frozen_source_sha256": sha256(directory / "frozen_source.zip"),
        "source_hashes_before_equal_after": True, "native_samples_contiguous_2ms": True,
        "scheduled_denominator": len(protocol["schedule"]), "audited_episodes": len(audited),
        "task_successes": sum(row["outcome"] is True for row in audited),
        "held_lifts": sum(row.get("held_lift") is True for row in audited),
        "unknowns": sum(row["unknown"] for row in audited),
        "within_additional_12n_reference": sum(row.get("additional_12n_reference", False) for row in audited),
        "native_samples": native_count, "gzip_artifacts": artifacts, "rows": audited,
        "scope": "Development poses reused; additional 12N reference does not replace legacy V6 task scoring"}
    atomic_json(directory / "verification_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.directory.resolve())


if __name__ == "__main__":
    main()
