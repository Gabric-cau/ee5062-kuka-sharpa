import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.visualization import (
    PlaybackControls, PresentationTrial, add_trial_arguments, configure_camera, configure_visuals,
    create_run_directory, load_trial_record, load_validation_events, resolve_trial, stage_at,
    presentation_manifest,
)


def test_pause_offsets_realtime_deadline_without_changing_sim_time():
    controls = PlaybackControls(now=10.0)
    controls.dispatch_key(32)
    controls.apply_pending(0.5, 10.5)
    assert controls.paused
    controls.dispatch_key(80)
    controls.apply_pending(0.5, 11.0)
    assert not controls.paused
    assert controls.deadline(0.6) == pytest.approx(11.1)
    assert controls.log[0]["sim_time_before"] == controls.log[1]["sim_time_before"] == 0.5
    assert controls.log[1]["paused_wall_s"] == 0.5


def test_force_and_contact_switches_are_independent_and_replay_resets_clock():
    controls = PlaybackControls(now=10.0)
    controls.dispatch_key(70, "validation_schedule")
    assert controls.apply_pending(0.4, 10.4) == ["forces"]
    assert controls.contacts and controls.forces
    controls.dispatch_key(67)
    controls.apply_pending(0.4, 10.4)
    assert not controls.contacts and controls.forces
    controls.dispatch_key(82)
    controls.apply_pending(0.4, 10.4)
    assert controls.replay
    controls.begin_replay(0.4, 10.5)
    assert not controls.replay and not controls.paused
    assert controls.deadline(0.0) == 10.5
    assert controls.log[-1]["sim_time_before"] == 0.4
    assert controls.log[-1]["sim_time_after"] == 0.0


def test_saved_trial_replays_exact_inputs_and_unmodified_physics(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "isolated" / "scene.xml", object_kind="ycb_can")))
    rollout = GraspRollout(model, mujoco.MjData(model))
    parser = argparse.ArgumentParser()
    add_trial_arguments(parser)
    saved = {
        "object_kind": "ycb_can", "seed": 7, "controller": "fixed", "duration_s": 0.05,
        "sampled_object_pos": [0.719, 0.003, 0.38], "object_mass": 0.17,
        "object_friction": 0.62, "trial_id": "saved-trial",
    }
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    args = parser.parse_args(["--trial-json", str(path), "--seed", "99", "--mass", "1.0"])
    trial = resolve_trial(args, rollout, load_trial_record(args))
    assert trial.seed == 7 and trial.controller == "fixed"
    assert trial.object_pos == (0.719, 0.003, 0.38)
    assert trial.object_mass == 0.17 and trial.object_friction == 0.62
    assert trial.source_trial_id == "saved-trial"
    assert trial.source_provenance == "unverified_missing_provenance"
    first, first_rows = rollout.run_trace(**trial.rollout_arguments())
    configure_visuals(model)
    second, second_rows = rollout.run_trace(**trial.rollout_arguments())
    assert asdict(first) == asdict(second)
    assert first_rows == second_rows


def test_reset_camera_and_run_directory_preserve_existing_outputs(tmp_path):
    run_dir = create_run_directory("unused", tmp_path / "viewer")
    with pytest.raises(FileExistsError):
        create_run_directory("unused", run_dir)
    model = mujoco.MjModel.from_xml_path(str(build_scene(run_dir / "scene.xml")))
    data = mujoco.MjData(model)
    rollout = GraspRollout(model, data)
    rollout.reset()
    camera = mujoco.MjvCamera()
    configure_camera(camera, model, data)
    assert camera.distance == 1.65
    assert np.all(np.isfinite(camera.lookat))
    configure_camera(camera, model, data, detail=True)
    assert 0.0 < camera.distance < 1.0
    assert np.linalg.norm(camera.lookat) > 0.5
    assert stage_at(0.0) == "open" and stage_at(1.31) == "lift"


def test_validation_events_use_native_key_codes_and_reject_unknown_keys(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([{"wall_s": 0.7, "key": "P"}, {"wall_s": 0.2, "key": "SPACE"}]), encoding="utf-8")
    assert load_validation_events(path) == [(0.2, 32), (0.7, 80)]
    path.write_text(json.dumps([{"wall_s": 0.2, "key": "X"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_validation_events(path)


def test_saved_trial_checks_record_hash_source_and_physics_versions(tmp_path, monkeypatch):
    from kuka_sharpa_grasp import experiments

    versions = {"python": "test-python", "mujoco": "test-mujoco", "numpy": "test-numpy"}
    monkeypatch.setattr(experiments, "software_versions", lambda: versions)
    monkeypatch.setattr(experiments, "source_identity", lambda: {"sha256": "test-source"})
    record = {"object_kind": "ycb_can", "controller": "force", "source_sha256": "test-source", "versions": versions}
    record["record_sha256"] = experiments.content_hash(record)
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    args = argparse.Namespace(trial_json=path)
    assert load_trial_record(args)["_replay_provenance"] == "verified_record_source_versions"
    record["controller"] = "fixed"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="record SHA256 mismatch"):
        load_trial_record(args)
    record.pop("record_sha256")
    record["source_sha256"] = "different-source"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="physical source/assets"):
        load_trial_record(args)
    record["source_sha256"] = "test-source"
    record["versions"] = {**versions, "mujoco": "different-mujoco"}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="mujoco version differs"):
        load_trial_record(args)


def test_presentation_run_ids_are_unique_but_physics_configuration_is_stable(monkeypatch):
    from kuka_sharpa_grasp import experiments

    monkeypatch.setattr(experiments, "source_identity", lambda: {"sha256": "same-source"})
    monkeypatch.setattr(experiments, "machine_identity", lambda: {"host": "same-host", "machine_id": "same-machine"})
    trial = PresentationTrial("ycb_can", 7, "force", 2.5, (0.72, 0.0, 0.38), 0.15, 0.8)
    first = presentation_manifest(trial, "mujoco_offscreen_opengl", "rollout_evidence")
    second = presentation_manifest(trial, "mujoco_offscreen_opengl", "rollout_evidence")
    assert first["run_id"] != second["run_id"]
    assert first["run_label"] == second["run_label"]
    assert first["config_hash"] == second["config_hash"]
    assert first["trial_id"] == second["trial_id"]
