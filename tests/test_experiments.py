import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp import experiments as exp
from kuka_sharpa_grasp.rollout import GraspRollout, TrialResult
from kuka_sharpa_grasp.scenarios import ScenarioSampler
from kuka_sharpa_grasp.scene import OBJECT_KINDS, build_scene


@pytest.fixture
def local_metadata(monkeypatch):
    files = {"test-source": {"sha256": "verified-for-unit-test", "bytes": 1}}
    source = {"files": files, "sha256": exp.content_hash(files)}
    versions = {"python": "test", "mujoco": "test", "numpy": "test", "system": "test"}
    machine = {"host": "test-local-host", "machine_id": "test-local-machine"}
    monkeypatch.setattr(exp, "source_identity", lambda: source)
    monkeypatch.setattr(exp, "software_versions", lambda: versions)
    monkeypatch.setattr(exp, "machine_identity", lambda: machine)
    return source, versions, machine


def fake_result(**kwargs):
    return TrialResult(kwargs["seed"], kwargs["controller"], 10, 2, 1, 3.0, 3.0, 2.0, 0.6,
                       kwargs["object_mass"], True, "success", object_friction=kwargs["object_friction"],
                       object_x=kwargs["object_pos"][0], object_y=kwargs["object_pos"][1], object_z=kwargs["object_pos"][2])


def test_paired_pose_mass_friction_and_identity_are_reproducible():
    config = exp.ExperimentConfig(object_kind="ycb_can", seeds=(7, 8), xy_jitter=0.005,
                                  friction_range=(0.5, 1.0), mass_range=(0.1, 0.3))
    plan = exp.make_trial_plan(config, exp.content_hash(config.serialized()))
    assert plan == exp.make_trial_plan(exp.ExperimentConfig.from_dict(config.serialized()), exp.content_hash(config.serialized()))
    for fixed, force in zip(plan[::2], plan[1::2]):
        assert (fixed["controller"], force["controller"]) == ("fixed", "force")
        assert fixed["trial_id"] != force["trial_id"]
        assert all(fixed[field] == force[field] for field in ("object_pos", "object_mass", "object_friction", "scenario_id"))
        assert fixed["object_pos"][2] == 0.38
    sampler = ScenarioSampler((0.7, 0, 0.43), 0.005, (0.5, 1.0), (0.1, 0.3))
    assert replace(sampler.sample(7, "fixed"), controller="force") == sampler.sample(7, "force")


def test_resume_skips_completed_ids_and_preserves_json_bytes(tmp_path, local_metadata):
    config = exp.ExperimentConfig(seeds=(0, 1), duration_s=0.02, xy_jitter=0.005)
    run = tmp_path / "run"
    calls = []
    def runner(**kwargs):
        calls.append((kwargs["seed"], kwargs["controller"]))
        return fake_result(**kwargs)
    exp.run_experiment(config, run_dir=run, max_new_trials=1, runner=runner)
    existing = {path: path.read_bytes() for path in (run / "trials").glob("*.json")}
    summary, coverage = exp.aggregate_runs([run])
    assert coverage["incomplete_trials"] == 3
    assert coverage["paired_outcomes"][0]["incomplete_pairs"] == 2
    assert sum(group["planned_trials"] for group in summary) == 4
    assert sum(group["incomplete_trials"] for group in summary) == 3
    assert summary[0]["success_rate_planned"] == 0.5
    exp.run_experiment(run_dir=run, resume=True, runner=runner)
    assert calls == [(0, "fixed"), (0, "force"), (1, "fixed"), (1, "force")]
    assert all(path.read_bytes() == data for path, data in existing.items())
    exp.run_experiment(run_dir=run, resume=True, runner=runner)
    assert len(calls) == 4
    completion = json.loads((run / "completion.json").read_text())
    assert completion["complete"] and completion["completed"] == completion["planned"] == 4
    assert completion["total_recorded_invocation_wall_s"] >= completion["total_trial_elapsed_s"]
    assert len(exp.read_trial_records(run, exp.load_manifest(run))) == 4


def test_keyboard_interrupt_retains_finished_trial_for_resume(tmp_path, local_metadata):
    calls = 0
    def interrupted_runner(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return fake_result(**kwargs)
    run = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        exp.run_experiment(exp.ExperimentConfig(seeds=(0,), duration_s=0.02), run_dir=run, runner=interrupted_runner)
    assert len(list((run / "trials").glob("*.json"))) == 1
    exp.run_experiment(run_dir=run, resume=True, runner=fake_result)
    assert len(exp.read_trial_records(run, exp.load_manifest(run))) == 2


@pytest.mark.parametrize("field", ("configuration", "source", "versions", "machine"))
def test_resume_rejects_mismatched_identity(tmp_path, local_metadata, monkeypatch, field):
    run = tmp_path / "mismatch"
    config = exp.ExperimentConfig(seeds=(0,), duration_s=0.02)
    exp.run_experiment(config, run_dir=run, max_new_trials=0, runner=fake_result)
    changed_config = config
    if field == "configuration":
        changed_config = replace(config, duration_s=0.04)
    else:
        index, function = {"source": (0, "source_identity"), "versions": (1, "software_versions"),
                           "machine": (2, "machine_identity")}[field]
        changed = {**local_metadata[index], "changed": True}
        monkeypatch.setattr(exp, function, lambda: changed)
    with pytest.raises(ValueError, match="resume .* mismatch"):
        exp.run_experiment(changed_config, run_dir=run, resume=True, runner=fake_result)


def test_json_publication_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / "trial.json"
    exp._write_json(path, {"status": "dropped"}, immutable=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        exp._write_json(path, {"status": "success"}, immutable=True)
    assert path.read_bytes() == before


def test_aggregate_rejects_duplicate_runs_and_repeated_trials(tmp_path, local_metadata):
    config = exp.ExperimentConfig(seeds=(0,), duration_s=0.02)
    first = exp.run_experiment(config, run_dir=tmp_path / "first", runner=fake_result)
    second = exp.run_experiment(config, run_dir=tmp_path / "second", runner=fake_result)
    with pytest.raises(ValueError, match="duplicate run"):
        exp.aggregate_runs([first, first / "results.csv"])
    with pytest.raises(ValueError, match="duplicate trial"):
        exp.aggregate_runs([first, second])


@pytest.mark.parametrize("field", ("configuration", "versions", "source"))
def test_aggregation_rejects_mixed_protocols(tmp_path, local_metadata, monkeypatch, field):
    config = exp.ExperimentConfig(seeds=(0,), duration_s=0.02)
    first = exp.run_experiment(config, run_dir=tmp_path / "first", runner=fake_result)
    if field == "configuration":
        config = replace(config, object_kind="ycb_can")
    elif field == "versions":
        monkeypatch.setattr(exp, "software_versions", lambda: {**local_metadata[1], "mujoco": "changed"})
    else:
        files = {"changed": {"sha256": "changed", "bytes": 1}}
        monkeypatch.setattr(exp, "source_identity", lambda: {"files": files, "sha256": exp.content_hash(files)})
    second = exp.run_experiment(config, run_dir=tmp_path / "second", runner=fake_result)
    with pytest.raises(ValueError, match="aggregation .* mismatch"):
        exp.aggregate_runs([first, second])


@pytest.mark.parametrize("failure", ("exception", "nonfinite", "drop"))
def test_failed_trials_remain_in_denominator(tmp_path, local_metadata, failure):
    def runner(**kwargs):
        if kwargs["controller"] == "fixed":
            if failure == "exception":
                raise RuntimeError("test physics failure")
            if failure == "nonfinite":
                return replace(fake_result(**kwargs), peak_force=float("nan"))
            return replace(fake_result(**kwargs), status="dropped", lifted=False, final_contacts=0)
        return fake_result(**kwargs)
    run = exp.run_experiment(exp.ExperimentConfig(seeds=(0,), duration_s=0.02),
                             run_dir=tmp_path / "run", runner=runner)
    summary, coverage = exp.aggregate_runs([run])
    assert coverage["completed_trials"] == coverage["planned_trials"] == 2
    assert sum(group["successes"] for group in summary) == 1
    assert sum(group["dropped"] + group["errors"] for group in summary) == 1
    assert coverage["paired_outcomes"][0]["force_only"] == 1


def test_modified_trial_evidence_is_rejected(tmp_path, local_metadata):
    run = exp.run_experiment(exp.ExperimentConfig(seeds=(0,), duration_s=0.02),
                             run_dir=tmp_path / "run", runner=fake_result)
    path = next((run / "trials").glob("*.json"))
    row = json.loads(path.read_text())
    row["status"] = "dropped"
    exp._write_json(path, row)
    with pytest.raises(ValueError, match="trial record hash mismatch"):
        exp.aggregate_runs([run])


def test_concurrent_mixed_object_scene_builds_match_serial_coverage(tmp_path):
    root_scene = PROJECT / "scene.xml"
    original = root_scene.read_bytes() if root_scene.exists() else None
    def compile_and_run(object_kind, directory):
        scene = build_scene(directory / object_kind / "scene.xml", object_kind)
        model = mujoco.MjModel.from_xml_path(str(scene))
        rollout = GraspRollout(model, mujoco.MjData(model))
        result = rollout.run(seed=0, controller="fixed", duration=0.02)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        return object_kind, (model.nq, model.nu, int(model.geom_type[geom]), tuple(rollout.object_initial_pos),
                             result.status, result.object_height, result.peak_force)
    serial = dict(compile_and_run(kind, tmp_path / "serial") for kind in OBJECT_KINDS)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compile_and_run, kind, tmp_path / "concurrent") for kind in OBJECT_KINDS]
        concurrent = dict(future.result() for future in futures)
    assert concurrent == serial
    if original is None:
        assert not root_scene.exists()
    else:
        assert root_scene.read_bytes() == original


def test_source_manifest_hashes_assets_and_excludes_presentation_files():
    source = exp.source_identity()
    assert source["sha256"] == exp.content_hash(source["files"])
    assert "vendor/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml" in source["files"]
    assert "vendor/ycb/005_tomato_soup_can/005_tomato_soup_can/poisson/nontextured.stl" in source["files"]
    assert "src/kuka_sharpa_grasp/rollout.py" in source["files"]
    assert not any("results/" in name or "visualization.py" in name or "hardware.py" in name for name in source["files"])


def test_run_lock_rejects_a_second_local_writer(tmp_path):
    with exp._run_lock(tmp_path):
        with pytest.raises(ValueError, match="another local process"):
            with exp._run_lock(tmp_path):
                pytest.fail("second writer acquired an existing run lock")


def test_cli_exclusive_seed_stop_randomize_default_and_checkpoint(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exp, "run_experiment", lambda config, **kwargs: calls.append((config, kwargs)) or tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--object", "ycb_can", "--seed-start", "4", "--seed-stop", "7",
                                     "--randomize", "--mass-range", "0.1", "0.2", "--max-new-trials", "1"])
    exp.experiment_cli()
    config, arguments = calls[0]
    assert config.seeds == (4, 5, 6)
    assert config.xy_jitter == 0.005
    assert config.mass_range == (0.1, 0.2)
    assert arguments["max_new_trials"] == 1


def test_cli_legacy_seed_count_and_mass_sweep_groups(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exp, "run_experiment", lambda config, **kwargs: calls.append(config) or tmp_path)
    monkeypatch.setattr(sys, "argv", ["sweep", "--seeds", "2", "--seed-start", "3", "--masses", "0.15", "0.30"])
    exp.experiment_cli(mass_sweep=True)
    assert calls[0].seeds == (3, 4)
    assert calls[0].masses == (0.15, 0.30)
    plan = exp.make_trial_plan(calls[0], exp.content_hash(calls[0].serialized()))
    assert {trial["mass_group"] for trial in plan} == {"0.15", "0.3"}


def test_unrandomized_seed_repeats_count_one_physical_scenario(tmp_path, local_metadata):
    run = exp.run_experiment(exp.ExperimentConfig(seeds=(0, 1, 2), duration_s=0.02),
                             run_dir=tmp_path / "repeats", runner=fake_result)
    assert exp.load_manifest(run)["unique_physical_scenarios"] == 1
    _, coverage = exp.aggregate_runs([run])
    assert coverage["unique_physical_scenarios"] == 1
    assert coverage["paired_outcomes"][0]["both_succeed"] == 3
