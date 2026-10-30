"""Runner mechanics on millisecond fixtures, never the archived 22s experiment."""

from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_coordinated_contact_experiment_r27 as runner
from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_policy import CONTACT_CHANNELS, FingerContactPolicy
from kuka_sharpa_grasp.coordinated_admittance import CoordinatedAdmittanceResult


XML = '''<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>
<worldbody><body name="arm"><inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
<joint name="joint1" axis="0 0 1" range="-1 1"/>
<body name="left_thumb_distal" pos=".1 0 0"><joint name="left_thumb_IP" axis="0 0 1" range="-1 1"/>
<geom name="hand" type="sphere" size=".03" pos=".1 0 0" mass=".1"/></body></body>
<geom name="unlabelled_surface" type="sphere" size=".03" pos=".2 .055 0"/></worldbody>
<actuator><position name="joint1" joint="joint1" kp="10" kv="1" forcelimited="true" forcerange="-1 1"/>
<position name="left_thumb_IP" joint="left_thumb_IP" kp="10" kv="1" forcelimited="true" forcerange="-1 1"/>
</actuator></mujoco>'''


@pytest.fixture
def fixture():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = FingerContactPolicy(("thumb",), {"thumb": ("left_thumb_IP",)})
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("joint1",))
    config = runner.Configuration(replay_until_s=.004, average_window_s=.002, response_seconds=.002)
    return model, data, policy, adapter, config


def small_archive(model, data, steps=2):
    native, trace = [], []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        contacts = []
        for contact_id, contact in enumerate(data.contact[:data.ncon]):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            contacts.append({"geom_ids": contact.geom.tolist(), "distance_m": float(contact.dist),
                "position_world_m": contact.pos.tolist(), "force_torque_contact_frame": force.tolist(),
                "contact_frame_world_rows": contact.frame.reshape(3, 3).tolist()})
        sample = read_native_contact_sample(model, data)
        native.append({"timestamp": float(data.time), "motor_commands": data.ctrl.tolist(),
            "actual_hand_joints": {"left_thumb_IP": {"position_rad": float(data.qpos[1]),
                "velocity_rad_s": float(data.qvel[1])}}, "native_contacts": contacts,
            "target_qpos": [999.], "target_spatial_velocity_world": [999.]})
        trace.append({"timestamp": float(data.time), "arm_joint_position": [float(data.qpos[0])],
            "arm_joint_velocity_rad_s": [float(data.qvel[0])],
            "hand_segment_forces": dict(sample.measured_pressures_n)})
    return native, trace


@pytest.mark.parametrize("changes", [{"response_seconds": 0.}, {"force_abort_n": 12.01},
    {"force_target_limit_n": 6.01}, {"arm_command_rate_rad_s": float("nan")},
    {"response_seconds": .003}, {"average_window_s": 23.}, {"finger_command_rate_rad_s": True}])
def test_configuration_rejects_invalid_or_relaxed_protocol(changes):
    with pytest.raises(ValueError):
        runner.Configuration(**changes)


def test_replay_matches_robot_tactile_whitelist_and_never_reads_target_fields(fixture):
    model, data, policy, adapter, config = fixture
    native, trace = small_archive(model, data)
    for row in native:
        # These cannot be serialized or numerically interpreted; replay ignores them.
        row["target_qpos"], row["target_spatial_velocity_world"] = object(), object()
    replay_data = mujoco.MjData(model)
    mujoco.mj_forward(model, replay_data)
    result, window = runner.replay_prefix(model, replay_data, native, trace,
        adapter.joint_names, ("joint1",), config)
    assert result["accepted"] and result["terminal"] == "replay_matched"
    assert len(result["checks"]) == 2 and len(window) == 1
    assert all(row["qpos_error_rad"] == 0. and row["force_error_n"] == 0. for row in result["checks"])


@pytest.mark.parametrize("mutation", ["position", "force", "contact_mask", "contact_position"])
def test_replay_mismatch_stops_at_first_native_sample(fixture, mutation):
    model, data, policy, adapter, config = fixture
    native, trace = small_archive(model, data)
    if mutation == "position":
        trace[0]["arm_joint_position"][0] += .1
    elif mutation == "force":
        trace[0]["hand_segment_forces"]["thumb"] += .1
    elif mutation == "contact_mask":
        native[0]["native_contacts"] = []
    else:
        native[0]["native_contacts"][0]["position_world_m"][0] += .1
    replay_data = mujoco.MjData(model)
    mujoco.mj_forward(model, replay_data)
    result, window = runner.replay_prefix(model, replay_data, native, trace,
        adapter.joint_names, ("joint1",), config)
    assert not result["accepted"] and result["terminal"] == "replay_mismatch"
    assert len(result["checks"]) == 1 and not window
    assert replay_data.time == .002


def test_missing_or_noncontiguous_archive_cannot_create_a_checkpoint(fixture):
    model, data, policy, adapter, config = fixture
    with pytest.raises(runner.ExperimentAbort, match="missing_native_rows"):
        runner.replay_prefix(model, data, [], [], adapter.joint_names, ("joint1",), config)


@pytest.mark.parametrize("mutation", ["missing_arm", "nan_arm_velocity", "nan_normal", "nan_force",
    "wrong_force_shape", "invalid_geom", "missing_hand"])
def test_invalid_whitelisted_archive_is_rejected_before_replaying_motor_input(fixture, mutation):
    model, data, policy, adapter, config = fixture
    native, trace = small_archive(model, data)
    if mutation == "missing_arm":
        del trace[0]["arm_joint_position"]
    elif mutation == "nan_arm_velocity":
        trace[0]["arm_joint_velocity_rad_s"][0] = float("nan")
    elif mutation == "nan_normal":
        native[0]["native_contacts"][0]["contact_frame_world_rows"][0][0] = float("nan")
    elif mutation == "nan_force":
        native[0]["native_contacts"][0]["force_torque_contact_frame"][0] = float("nan")
    elif mutation == "wrong_force_shape":
        native[0]["native_contacts"][0]["force_torque_contact_frame"] = [1.]
    elif mutation == "invalid_geom":
        native[0]["native_contacts"][0]["geom_ids"][0] = model.ngeom
    else:
        del native[0]["actual_hand_joints"]["left_thumb_IP"]
    replay_data = mujoco.MjData(model)
    mujoco.mj_forward(model, replay_data)
    with pytest.raises(runner.ExperimentAbort, match="replay_invalid_archive"):
        runner.replay_prefix(model, replay_data, native, trace, adapter.joint_names, ("joint1",), config)
    assert replay_data.time == 0.


def test_declared_targets_use_window_mean_and_fixed_delta_without_force_clipping():
    policy = runner.make_policy(runner.Configuration())
    sample = dict.fromkeys(CONTACT_CHANNELS, 0.) | {"thumb": 4., "middle": 3., "ring": 1.}
    targets = runner.declared_targets([sample, sample | {"thumb": 5.}], policy)
    assert targets == dict.fromkeys(CONTACT_CHANNELS, 0.) | {"thumb": 4., "middle": 2.5, "ring": 1.5}
    sample["ring"] = 5.9
    with pytest.raises(runner.ExperimentAbort, match="invalid_declared_force_targets"):
        runner.declared_targets([sample], policy)
    assert sample["ring"] == 5.9


def test_three_conditions_restore_identical_new_checkpoint_and_label_cache_times(fixture):
    model, data, policy, adapter, config = fixture
    spec, checkpoint = runner.integration_state(model, data)
    checkpoint_before = checkpoint.copy()
    initial = read_native_contact_sample(model, data)
    targets = dict.fromkeys(CONTACT_CHANNELS, 0.) | {"thumb": initial.measured_pressures_n["thumb"] + .001}
    results = [runner.run_condition(model, spec, checkpoint, adapter, targets, condition, config)
        for condition in runner.CONDITIONS]
    for result in results:
        assert result["completed"], result
        assert result["full_response_completed"] and result["last_window_complete"]
        assert result["last_window_actual_samples"] == result["last_window_requested_samples"] == 1
        assert result["native_steps"] == 1 and result["task_success"] is None
        row = result["trace"][0]
        assert row["before"]["timestamp_s"] == 0.
        assert row["step_dynamics_cache"]["step_start_s"] == 0.
        assert row["step_dynamics_cache"]["step_end_s"] == .002
        assert row["after"]["timestamp_s"] == .002
        assert row["before_canonical_patches"] and row["after_canonical_patches"]
        if result["condition"] == "HOLD":
            assert row["solver"] is None and not row["command_applied"]
        else:
            assert row["solver"]["accepted"] and row["command_applied"]
            np.testing.assert_allclose(row["step_dynamics_cache"]["actuator_force_nm"],
                row["solver"]["predicted_motor_torque_nm"], atol=1e-12)
            np.testing.assert_allclose(row["step_dynamics_cache"]["actual_minus_frozen_state_predicted_motor_torque_nm"], 0., atol=1e-12)
        assert result["initial_robot_state"]["timestamp_s"] == 0.
        np.testing.assert_array_equal(result["initial_robot_state"]["qpos_rad"],
            results[0]["initial_robot_state"]["qpos_rad"])
        np.testing.assert_array_equal(result["initial_robot_state"]["actual_ctrl"],
            results[0]["initial_robot_state"]["actual_ctrl"])
    locked = results[1]["trace"][0]
    np.testing.assert_allclose(locked["solver"]["command_velocity_rad_s"][0], 0.)
    np.testing.assert_array_equal(checkpoint, checkpoint_before)
    assert data.time == 0.  # Each condition owns separate data.
    assert runner.paired_comparison(results)["paired_comparison_eligible"]
    json.dumps(runner.json_value(results), allow_nan=False)


def test_rejected_solver_logs_candidate_and_does_not_take_an_extra_step(fixture, monkeypatch):
    model, data, policy, adapter, config = fixture
    spec, checkpoint = runner.integration_state(model, data)
    monkeypatch.setattr(adapter, "solve", lambda *args, **kwargs:
        CoordinatedAdmittanceResult(False, "CONTACT_TRACKING_RESIDUAL"))
    initial = read_native_contact_sample(model, data)
    targets = dict.fromkeys(CONTACT_CHANNELS, 0.) | {"thumb": initial.measured_pressures_n["thumb"]}
    result = runner.run_condition(model, spec, checkpoint, adapter, targets, "FULL_JOINT", config)
    assert result["terminal"] == "solver_rejected" and result["native_steps"] == 0
    assert not result["full_response_completed"] and not result["last_window_complete"]
    row = result["trace"][0]
    assert not row["command_applied"] and row["after"] is None
    assert row["solver"]["reason"] == "CONTACT_TRACKING_RESIDUAL"


def test_native_force_abort_retains_violating_measurement(fixture):
    model, data, policy, adapter, config = fixture
    spec, checkpoint = runner.integration_state(model, data)
    initial = read_native_contact_sample(model, data)
    assert initial.measured_pressures_n["thumb"] > 0.
    config = replace(config, force_abort_n=initial.measured_pressures_n["thumb"] / 2.)
    result = runner.run_condition(model, spec, checkpoint, adapter,
        dict(initial.measured_pressures_n), "HOLD", config)
    assert result["terminal"] == "native_force_abort" and result["native_steps"] == 0
    assert result["peak_channel_force_n"] > config.force_abort_n
    assert result["trace"][0]["before_tactile"]["measured_pressures_n"]["thumb"] > config.force_abort_n


def test_early_abort_raw_rms_cannot_count_as_paired_force_improvement():
    rows = [{"condition": condition, "completed": True, "full_response_completed": True,
        "last_window_complete": True, "native_steps": 1000,
        "initial_rms_force_error_n": .5, "last_window_rms_force_error_n": .4}
        for condition in runner.CONDITIONS]
    result = runner.paired_comparison(rows)
    assert result["paired_comparison_eligible"]
    assert result["full_joint_minus_initial_rms_n"] == pytest.approx(-.1)
    rows[2].update(completed=False, full_response_completed=False, native_steps=1,
        last_window_rms_force_error_n=0.)
    result = runner.paired_comparison(rows)
    assert not result["paired_comparison_eligible"]
    assert result["full_joint_minus_hold_rms_n"] is None
    assert result["full_joint_minus_locked_arm_rms_n"] is None
    assert result["full_joint_minus_initial_rms_n"] is None
    assert rows[2]["last_window_rms_force_error_n"] == 0.
    rows[2].update(completed=True, full_response_completed=True, native_steps=1000,
        last_window_complete=False)
    assert not runner.paired_comparison(rows)["paired_comparison_eligible"]


def test_top_level_replay_failure_never_runs_paired_conditions_or_saves_checkpoint(fixture, monkeypatch, tmp_path):
    model, data, policy, adapter, config = fixture
    native, trace = small_archive(model, data)
    trace[0]["arm_joint_position"][0] += .1
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "scene.xml").write_text(XML)
    for name, rows in (("native.json.gz", native), ("trace.json.gz", trace)):
        with gzip.open(baseline / name, "wt") as stream:
            json.dump(rows, stream)
    monkeypatch.setattr(runner, "load_evidence_model", lambda path: model)
    monkeypatch.setattr(runner, "reset_task", lambda model, data, scenario: mujoco.mj_forward(model, data))
    monkeypatch.setattr(runner, "make_policy", lambda config: policy)
    monkeypatch.setattr(runner, "ARM_JOINTS", ("joint1",))
    monkeypatch.setattr(runner, "run_condition", lambda *args: pytest.fail("Replay failure escaped its gate"))
    output = tmp_path / "new_run"
    summary = runner.run(output, baseline, config)
    assert summary["terminal"] == "replay_mismatch" and summary["conditions"] == []
    assert not (output / "new_integration_checkpoint.npz").exists()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["terminal"] == "replay_mismatch"
    assert saved["protocol"]["inputs_changed_during_run"] is False
    assert saved["protocol"]["source_changed_during_run"] is False
    assert saved["paired_comparison_eligible"] is False
    compiled = output / saved["protocol"]["compiled_model"]["file"]
    assert compiled.is_file() and runner.file_hash(compiled) == saved["protocol"]["compiled_model"]["sha256"]
    restored_model = mujoco.MjModel.from_binary_path(str(compiled))
    np.testing.assert_array_equal(restored_model.geom_pos, model.geom_pos)


def test_top_level_new_checkpoint_is_bound_to_actual_compiled_model(fixture, monkeypatch, tmp_path):
    model, data, policy, adapter, config = fixture
    native, trace = small_archive(model, data)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "scene.xml").write_text(XML)
    for name, rows in (("native.json.gz", native), ("trace.json.gz", trace)):
        with gzip.open(baseline / name, "wt") as stream:
            json.dump(rows, stream)
    monkeypatch.setattr(runner, "load_evidence_model", lambda path: model)
    monkeypatch.setattr(runner, "reset_task", lambda model, data, scenario: mujoco.mj_forward(model, data))
    monkeypatch.setattr(runner, "make_policy", lambda config: policy)
    monkeypatch.setattr(runner, "ARM_JOINTS", ("joint1",))
    output = tmp_path / "new_run"
    summary = runner.run(output, baseline, config)
    assert summary["terminal"] == "paired_conditions_recorded", summary
    assert len(summary["conditions"]) == 3
    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    assert manifest["compiled_model"] == summary["protocol"]["compiled_model"]
    assert manifest["sha256"] == runner.file_hash(output / "new_integration_checkpoint.npz")
    compiled = output / manifest["compiled_model"]["file"]
    assert runner.file_hash(compiled) == manifest["compiled_model"]["sha256"]
    restored_model = mujoco.MjModel.from_binary_path(str(compiled))
    with np.load(output / "new_integration_checkpoint.npz") as checkpoint:
        restored = runner.restore_checkpoint(restored_model, int(checkpoint["state_spec"]), checkpoint["state"])
    assert restored.time == .004
    np.testing.assert_allclose(restored.qpos, data.qpos, atol=1e-15)
    np.testing.assert_allclose(restored.qvel, data.qvel, atol=1e-15)
    assert manifest["time_s"] == .004 and len(manifest["force_average_window"]) == 1
