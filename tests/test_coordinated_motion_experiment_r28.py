"""Matched runner checks on native millisecond fixtures, never the 22s replay."""

from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_coordinated_motion_experiment_r28 as runner
from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_policy import CONTACT_CHANNELS, ContactObservation, ContactPatch, FingerContactPolicy
from kuka_sharpa_grasp.coordinated_admittance import CoordinatedAdmittanceResult


XML = '''<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>
<worldbody>
 <geom name="table" type="plane" size="1 1 .01" pos="0 0 -1"/>
 <body name="target" pos=".4 0 .055"><freejoint name="target_free"/>
  <geom name="target_geom" type="sphere" size=".03" mass=".1"/></body>
 <body name="arm"><joint name="joint1" axis="0 1 0" range="-1 1"/>
  <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
  <body name="link7" pos=".2 0 0">
   <body name="left_thumb_distal" pos=".1 0 0"><joint name="left_thumb_IP" axis="0 0 1" range="-1 1"/>
    <geom name="hand" type="sphere" size=".03" pos=".1 0 0" mass=".1"/>
   </body>
  </body>
 </body>
</worldbody>
<actuator>
 <position name="joint1" joint="joint1" kp="10" kv="1" forcelimited="true" forcerange="-1 1"/>
 <position name="left_thumb_IP" joint="left_thumb_IP" kp="10" kv="1" forcelimited="true" forcerange="-1 1"/>
</actuator></mujoco>'''


@pytest.fixture
def fixture(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = FingerContactPolicy(("thumb",), {"thumb": ("left_thumb_IP",)})
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("joint1",))
    contact_config = runner.r27.Configuration(replay_until_s=.004, average_window_s=.002)
    config = runner.Configuration(preload_seconds=.002, probe_distance_m=.000008,
        probe_speed_m_s=.004, settle_seconds=.002)
    monkeypatch.setattr(runner.r27, "ARM_JOINTS", ("joint1",))
    return model, data, policy, adapter, config, contact_config


@pytest.mark.parametrize("elapsed,phase,distance,speed", [
    (0., "PRELOAD", 0., 0.), (1.998, "PRELOAD", 0., 0.),
    (2., "PROBE", 0., .004), (2.625, "PROBE", .0025, .004),
    (3.25, "SETTLE", .005, 0.), (4., "SETTLE", .005, 0.)])
def test_declared_motion_trajectory_has_native_phase_boundaries_and_metre_units(elapsed, phase, distance, speed):
    config = runner.Configuration()
    for condition in runner.CONDITIONS[1:]:
        offset, velocity, actual_phase = runner.trajectory(elapsed, condition, config)
        assert actual_phase == phase
        np.testing.assert_allclose(offset, [0., 0., distance], atol=1e-15)
        np.testing.assert_allclose(velocity, [0., 0., speed], atol=1e-15)
    assert config.duration_s == 4.
    offset, velocity, phase = runner.trajectory(elapsed, "STATIC_HOLD", config)
    np.testing.assert_array_equal(offset, np.zeros(3))
    np.testing.assert_array_equal(velocity, np.zeros(3))
    assert phase == ("PRELOAD" if elapsed < 2. else "HOLD")


@pytest.mark.parametrize("changes", [{"probe_distance_m": .0051}, {"probe_speed_m_s": .0041},
    {"preload_seconds": .003}, {"settle_seconds": 0.}, {"probe_speed_m_s": float("nan")},
    {"preload_seconds": np.bool_(True)}])
def test_invalid_or_expanded_probe_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        runner.Configuration(**changes)


def test_transport_is_added_after_pressure_clamp_without_clipping_measured_load():
    patch = ContactPatch(8, "thumb", "left_thumb_distal", (0., 0., 0.), (0., 0., 1.), 8., None)
    observed = ContactObservation((patch,), {}, ("thumb",), (), 0.)
    reference = SimpleNamespace(patch_input_indices=(8,), rigid_transport_normal_velocity_m_s=(.004,))
    pressures, targets = {"thumb": 8.}, {"thumb": 2.}
    config = runner.r27.Configuration()
    target, correction, transport = runner.normal_velocity_targets(reference, observed, pressures,
        targets, "MOVING_WITH_TRANSPORT", config)
    np.testing.assert_allclose(correction, [-.001])
    np.testing.assert_allclose(transport, [.004])
    np.testing.assert_allclose(target, [.003])
    assert pressures["thumb"] == 8.
    target, correction, transport = runner.normal_velocity_targets(reference, observed, pressures,
        targets, "MOVING_WITHOUT_TRANSPORT", config)
    np.testing.assert_allclose(target, [-.001])
    np.testing.assert_allclose(transport, [.004])  # Diagnostic even when not applied.


@pytest.mark.parametrize("mutation", ["wrong_index", "wrong_shape", "nan_transport", "nan_measured"])
def test_transport_row_identity_and_nonfinite_feedback_are_rejected(mutation):
    patch = ContactPatch(8, "thumb", "left_thumb_distal", (0., 0., 0.), (0., 0., 1.), 1., None)
    observed = ContactObservation((patch,), {}, ("thumb",), (), 0.)
    reference = SimpleNamespace(patch_input_indices=(8,), rigid_transport_normal_velocity_m_s=(.004,))
    pressures = {"thumb": 1.}
    if mutation == "wrong_index":
        reference.patch_input_indices = (9,)
    elif mutation == "wrong_shape":
        reference.rigid_transport_normal_velocity_m_s = ()
    elif mutation == "nan_transport":
        reference.rigid_transport_normal_velocity_m_s = (float("nan"),)
    else:
        pressures["thumb"] = float("nan")
    with pytest.raises(ValueError):
        runner.normal_velocity_targets(reference, observed, pressures, {"thumb": 2.},
            "MOVING_WITH_TRANSPORT", runner.r27.Configuration())


@pytest.mark.parametrize("write", ["qpos", "qvel", "qfrc_applied", "xfrc_applied"])
def test_native_journal_blocks_external_state_or_force_writes_before_integration(fixture, write):
    model, data, policy, adapter, config, contact_config = fixture
    journal = runner.NativeJournal(model, data)
    if write == "qpos":
        data.qpos[journal.target_slice.start] += .001
        reason = "external_target_qpos_write"
    elif write == "qvel":
        data.qvel[journal.target_velocity_slice.start] += .1
        reason = "external_target_qvel_write"
    elif write == "qfrc_applied":
        data.qfrc_applied[0] = 1.
        reason = "unrecorded_applied_force"
    else:
        data.xfrc_applied[journal.target_body, 0] = 1.
        reason = "unrecorded_applied_force"
    with pytest.raises(runner.r27.ExperimentAbort, match=reason):
        journal.step(data)
    assert journal.integration_steps == 0 and data.time == 0.
    assert journal.audit()["external_target_qpos_writes"] == int(write == "qpos")
    assert journal.audit()["external_target_qvel_writes"] == int(write == "qvel")


def test_native_journal_allows_real_target_dynamics_and_does_not_write_it(fixture):
    model, data, policy, adapter, config, contact_config = fixture
    journal = runner.NativeJournal(model, data)
    initial = data.qpos[journal.target_slice].copy()
    for _ in range(3):
        journal.step(data)
        mujoco.mj_forward(model, data)
    assert journal.integration_steps == 3 and data.time == .006
    assert not np.array_equal(data.qpos[journal.target_slice], initial)
    assert journal.audit()["external_target_qpos_writes"] == journal.audit()["external_target_qvel_writes"] == 0
    snapshot = runner.r27.integration_state(model, data)[1]
    sample = journal.sample(data, "TEST", force_cache_scope="Fresh mj_forward")
    np.testing.assert_array_equal(runner.r27.integration_state(model, data)[1], snapshot)
    assert sample["native_contacts"] and sample["motor_commands"] == data.ctrl.tolist()


def test_matched_native_conditions_single_write_torque_cache_and_integrated_after_pose(fixture, monkeypatch):
    model, data, policy, adapter, config, contact_config = fixture
    spec, checkpoint = runner.r27.integration_state(model, data)
    saved = checkpoint.copy()
    initial = read_native_contact_sample(model, data)
    targets = dict(initial.measured_pressures_n)
    original_step = runner.NativeJournal.step
    original_solve = adapter.solve
    predicted_commands, stepped_commands = [], []
    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        if result.accepted:
            predicted_commands.append(result.command_rad.copy())
        return result
    def step(journal, current):
        stepped_commands.append(current.ctrl[[servo.actuator_id for servo in adapter.servos]].copy())
        assert len(stepped_commands) == len(predicted_commands)
        np.testing.assert_array_equal(stepped_commands[-1], predicted_commands[-1])
        original_step(journal, current)
    monkeypatch.setattr(adapter, "solve", solve)
    monkeypatch.setattr(runner.NativeJournal, "step", step)
    results = [runner.run_condition(model, spec, checkpoint, adapter, targets, condition, config, contact_config)
        for condition in runner.CONDITIONS]
    for result in results:
        assert result["completed"], result
        assert result["native_steps"] == result["required_native_steps"] == 3
        assert result["native_audit"]["external_target_qpos_writes"] == 0
        np.testing.assert_array_equal(result["initial_robot_state"]["qpos_rad"], results[0]["initial_robot_state"]["qpos_rad"])
        np.testing.assert_array_equal(result["initial_robot_state"]["actual_ctrl"], results[0]["initial_robot_state"]["actual_ctrl"])
        for index, row in enumerate(result["rows"]):
            assert row["command_applied"] and row["solver"]["accepted"]
            assert row["before_robot_state"]["timestamp_s"] == pytest.approx(index * .002)
            cache = row["native_dynamics_cache"]
            assert cache["step_start_s"] == pytest.approx(index * .002)
            assert cache["step_end_s"] == pytest.approx((index + 1) * .002)
            assert row["after_robot_state"]["timestamp_s"] == cache["step_end_s"]
            assert row["after_wrist_pose"]["timestamp_s"] == cache["step_end_s"]
            assert row["before_canonical_patches"] and cache["canonical_patches"] and row["after_canonical_patches"]
            assert cache["native_sample"]["native_contacts"]
            np.testing.assert_allclose(cache["motor_torque_prediction_residual_nm"], 0., atol=1e-12)
            np.testing.assert_array_equal(row["actual_ctrl_for_step"], row["after_robot_state"]["actual_ctrl"])
        end = np.array(result["rows"][-1]["after_wrist_pose"]["position_world_m"])
        start = np.array(result["rows"][0]["motion_reference"]["wrist_position_world_m"])
        np.testing.assert_allclose(result["actual_wrist_displacement_m"], end - start, atol=1e-15)
        assert result["task_success"] is None
    assert len(predicted_commands) == len(stepped_commands) == 9
    assert results[1]["rows"][1]["motion_reference"]["requested_linear_velocity_world_m_s"][2] > 0.
    assert results[2]["rows"][1]["rigid_transport_normal_velocity_m_s"][0] > 0.
    np.testing.assert_array_equal(checkpoint, saved)
    assert data.time == 0.
    json.dumps(runner.r27.json_value(results), allow_nan=False)


def test_solver_rejection_retains_bounds_candidate_and_never_writes_or_steps(fixture, monkeypatch):
    model, data, policy, adapter, config, contact_config = fixture
    spec, checkpoint = runner.r27.integration_state(model, data)
    monkeypatch.setattr(adapter, "solve", lambda *args, **kwargs:
        CoordinatedAdmittanceResult(False, "CONTACT_TRACKING_RESIDUAL"))
    result = runner.run_condition(model, spec, checkpoint, adapter,
        dict(read_native_contact_sample(model, data).measured_pressures_n), "MOVING_WITH_TRANSPORT", config, contact_config)
    assert not result["completed"] and result["native_steps"] == 0
    assert result["terminal"] == "solver_rejected"
    assert result["rows"][0]["solver"]["reason"] == "CONTACT_TRACKING_RESIDUAL"
    assert not result["rows"][0]["command_applied"]
    assert "after_robot_state" not in result["rows"][0]


@pytest.mark.parametrize("failure", ["mismatch", "second_tick_invalid_archive"])
def test_failed_replay_preserves_all_completed_native_steps_and_prevents_branches(fixture, monkeypatch, tmp_path, failure):
    model, data, policy, adapter, config, contact_config = fixture
    archive_journal = runner.NativeJournal(model, data)
    native, trace = [], []
    for _ in range(2):
        archive_journal.step(data)
        native.append(archive_journal.sample(data, "ARCHIVE", force_cache_scope="Native step cache"))
        trace.append({"timestamp": float(data.time),
            "arm_joint_position": [float(data.qpos[model.jnt_qposadr[model.joint("joint1").id]])],
            "arm_joint_velocity_rad_s": [float(data.qvel[model.jnt_dofadr[model.joint("joint1").id]])],
            "hand_segment_forces": dict(read_native_contact_sample(model, data).measured_pressures_n)})
    if failure == "mismatch":
        trace[0]["arm_joint_position"][0] += .1
        expected_terminal = "replay_mismatch"
    else:
        del native[1]["motor_commands"]
        expected_terminal = "replay_invalid_archive"
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "scene.xml").write_text(XML)
    for name, rows in (("native.json.gz", native), ("trace.json.gz", trace)):
        with gzip.open(baseline / name, "wt") as stream:
            json.dump(rows, stream)
    monkeypatch.setattr(runner, "load_evidence_model", lambda path: model)
    monkeypatch.setattr(runner, "reset_task", lambda model, data, scenario: mujoco.mj_forward(model, data))
    monkeypatch.setattr(runner.r27, "Configuration", lambda: contact_config)
    monkeypatch.setattr(runner.r27, "make_policy", lambda config: policy)
    monkeypatch.setattr(runner, "run_condition", lambda *args: pytest.fail("Failed replay must not enter response"))
    output = tmp_path / "failed_run"
    result = runner.run(output, baseline, config)
    assert result["terminal"] == expected_terminal
    assert result["conditions"] == [] and not result["paired_comparison_eligible"]
    assert not (output / "new_integration_checkpoint.npz").exists()
    verification = json.loads((output / "replay_verification.json").read_text())
    assert not verification["accepted"] and len(verification["checks"]) == 1
    with gzip.open(output / "prefix_native.json.gz", "rt") as stream:
        actual_native = json.load(stream)
    assert len(actual_native) == result["protocol"]["prefix_native_audit"]["integration_steps"] == 1
    assert actual_native[0]["timestamp"] == .002 and actual_native[0]["native_contacts"]
    assert result["protocol"]["prefix_native"]["sha256"] == runner.r27.file_hash(output / "prefix_native.json.gz")
    environment = result["protocol"]["runtime_environment"]
    assert environment["python"] and environment["platform"] and environment["hostname"]
