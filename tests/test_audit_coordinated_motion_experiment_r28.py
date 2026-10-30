"""Independent audit rejects changed evidence and accepts real negative probes."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_coordinated_motion_experiment_r28 import (
    AuditError, CHANNELS, audit_branch, channels_from_native, check_command,
    check_normal_targets, check_tactile, finite_tree, privileged_motion_metrics, verify_source_archive)


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
      <worldbody>
        <body name="link1"><joint name="joint1" range="-2 2"/>
          <geom name="robot" type="sphere" size=".01" mass="1"/>
          <body name="left_thumb_distal"><geom name="hand" type="sphere" size=".01" mass=".1"/>
            <geom name="hand_internal" type="sphere" size=".01" mass=".1"/></body>
        </body>
        <body name="other"><geom name="external" type="sphere" size=".01" mass=".1"/></body>
        <geom name="table" type="box" size="1 1 .01"/>
      </worldbody>
      <actuator><position name="arm" joint="joint1" kp="100" kv="2" ctrlrange="-1 1" forcerange="-10 10"/></actuator>
    </mujoco>''')


def contact(first, second, force):
    return dict(geom_ids=[first, second], distance_m=0., position_world_m=[0., 0., 0.],
                contact_frame_world_rows=np.eye(3).tolist(), force_torque_contact_frame=[force, 0., 0., 0., 0., 0.])


def raw(model, forces=(2., 3., 50.)):
    hand, internal, external = (model.geom(name).id for name in ("hand", "hand_internal", "external"))
    return dict(timestamp=22.002, motor_commands=[.00002], target_qpos=[0.]*7,
        target_position_world_m=[0., 0., 0.], target_spatial_velocity_world=[0.]*6,
        actual_hand_joints={},
        native_contacts=[contact(hand, external, forces[0]), contact(external, hand, forces[1]),
                         contact(hand, internal, forces[2])])


def tactile(pressure=2.):
    return dict(timestamp_s=22., measured_pressures_n=dict.fromkeys(CHANNELS, 0.) | {"thumb": pressure},
        patches=[dict(input_index=0, finger="thumb", hand_body_name="left_thumb_distal",
            position_world_m=[1., 0., 0.], normal_on_object_world=[0., 1., 0.], measured_pressure_n=pressure)])


def normal_row(condition="MOVING_WITH_TRANSPORT"):
    observed = tactile()
    # omega z x point x gives +y, plus y translation: 0.003m/s rigid motion.
    reference = dict(patch_input_indices=[0], wrist_position_world_m=[0., 0., 0.],
        achieved_linear_velocity_world_m_s=[0., .001, 0.], achieved_angular_velocity_world_rad_s=[0., 0., .002],
        rigid_transport_normal_velocity_m_s=[.003], joint_names=["joint1", "left_thumb_IP"],
        wrist_jacobian_world=[[0., 0.], [.001, 0.], [0., 0.], [0., 0.], [0., 0.], [.002, 0.]],
        task_reference_velocity_rad_s=[1., 0.])
    return dict(before_tactile=observed, motion_reference=reference,
        rigid_transport_normal_velocity_m_s=[.003], pressure_correction_m_s=[.001],
        target_normal_velocity_m_s=[.004 if condition == "MOVING_WITH_TRANSPORT" else .001])


CONTACT_CONFIG = dict(pressure_velocity_gain_m_n_s=.0005, maximum_normal_velocity_m_s=.001,
    arm_command_rate_rad_s=.025, finger_command_rate_rad_s=.05, added_deflection_limit_rad=.05)


def command_row():
    command = .00002
    return dict(command_applied=True, before_robot_state=dict(joint_names=["joint1"], qpos_rad=[0.],
        qvel_rad_s=[.001], actual_ctrl=[0.], timestamp_s=22.), actual_ctrl_for_step=[command],
        solver=dict(accepted=True, command_rad=[command], command_velocity_rad_s=[.01],
            predicted_motor_torque_nm=[0.], bounds=dict(command_lower_rad=[-.00005], command_upper_rad=[.00005])),
        native_dynamics_cache=dict(actuator_length_rad=[0.], actuator_velocity_rad_s=[.001],
            actuator_force_nm=[0.], motor_torque_prediction_residual_nm=[0.]))


def test_raw_contacts_independently_sum_six_channels_and_handle_geometry_order(model):
    totals, patches = channels_from_native(model, raw(model))
    assert totals == dict.fromkeys(CHANNELS, 0.) | {"thumb": 5.}
    assert len(patches) == 2
    assert patches[0]["normal_on_object_world"] == [1., 0., 0.]
    assert patches[1]["normal_on_object_world"] == [-1., 0., 0.]


@pytest.mark.parametrize("mutate", [
    lambda value: value["native_contacts"][0].update(force_torque_contact_frame=[float("nan")]*6),
    lambda value: value["native_contacts"][0].update(geom_ids=[True, 0]),
    lambda value: value.update(motor_commands=[False]),
])
def test_nonfinite_or_ambiguous_native_fields_are_rejected(model, mutate):
    value = raw(model)
    mutate(value)
    with pytest.raises(AuditError):
        channels_from_native(model, value)


def test_tactile_channel_sum_is_checked_independently():
    value = tactile()
    assert check_tactile(value)[0] == 2.
    value["measured_pressures_n"]["thumb"] = 1.
    with pytest.raises(AuditError, match="totals disagree"):
        check_tactile(value)


@pytest.mark.parametrize("condition", ["STATIC_HOLD", "MOVING_WITHOUT_TRANSPORT", "MOVING_WITH_TRANSPORT"])
def test_correct_transport_is_added_only_for_treatment_and_not_force_clamped(condition):
    row = normal_row(condition)
    targets = dict.fromkeys(CHANNELS, 0.) | {"thumb": 8.}
    check_normal_targets(row, targets, condition, CONTACT_CONFIG)
    assert row["pressure_correction_m_s"] == [.001]
    if condition == "MOVING_WITH_TRANSPORT":
        assert row["target_normal_velocity_m_s"] == [.004]


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(target_normal_velocity_m_s=[-.002]),
    lambda row: row.update(target_normal_velocity_m_s=[.001]),
    lambda row: row["motion_reference"].update(patch_input_indices=[1]),
    lambda row: row["motion_reference"].update(task_reference_velocity_rad_s=[1., .1]),
    lambda row: row["motion_reference"].update(achieved_angular_velocity_world_rad_s=[0., 0., -.002]),
])
def test_wrong_transport_sign_missing_feedforward_and_finger_leak_are_rejected(mutate):
    row = normal_row()
    mutate(row)
    with pytest.raises(AuditError):
        check_normal_targets(row, dict.fromkeys(CHANNELS, 0.) | {"thumb": 8.},
                             "MOVING_WITH_TRANSPORT", CONTACT_CONFIG)


def test_actual_position_command_and_frozen_servo_torque_are_recomputed(model):
    assert check_command(model, command_row(), np.zeros(1), CONTACT_CONFIG) == 0.


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(actual_ctrl_for_step=[.00003]),
    lambda row: row["solver"].update(predicted_motor_torque_nm=[.1]),
    lambda row: row["solver"].update(command_velocity_rad_s=[.03]),
    lambda row: row["solver"]["bounds"].update(command_upper_rad=[.01]),
    lambda row: row["native_dynamics_cache"].update(actuator_force_nm=[.1]),
    lambda row: row["native_dynamics_cache"].update(actuator_velocity_rad_s=[0.]),
    lambda row: row["solver"].update(accepted=False),
])
def test_post_solver_changes_wrong_bounds_or_stale_motor_state_are_rejected(model, mutate):
    row = command_row()
    mutate(row)
    with pytest.raises(AuditError):
        check_command(model, row, np.zeros(1), CONTACT_CONFIG)


def aborted_fixture(model):
    state = dict(joint_names=["joint1"], qpos_rad=[0.], qvel_rad_s=[0.], actual_ctrl=[0.], timestamp_s=22.)
    observed = tactile(13.)
    row = dict(before_time_s=22., before_robot_state=deepcopy(state), before_tactile=observed,
        before_canonical_patches=deepcopy(observed["patches"]), command_applied=False,
        abort=dict(reason="native_force_abort", detail="retained failure"))
    result = dict(condition="STATIC_HOLD", task_success=None, initial_robot_state=deepcopy(state),
        fixed_capture_deflection_rad=[0.], force_targets_n=dict.fromkeys(CHANNELS, 0.) | {"thumb": 2.},
        native_steps=0, required_native_steps=2, terminal="native_force_abort", completed=False,
        peak_channel_force_n=13., native_audit=dict(integration_steps=0, external_target_qpos_writes=0,
            external_target_qvel_writes=0, write_events=[], velocity_write_events=[]))
    return result, [row], [], state


def test_zero_step_force_abort_is_real_negative_evidence_with_retained_peak(model):
    result, rows, native, state = aborted_fixture(model)
    audit = audit_branch(model, result, rows, native, state, CONTACT_CONFIG)
    assert audit["accepted_negative_evidence"] is True
    assert audit["completed"] is False and audit["native_steps"] == 0
    assert audit["peak_channel_force_n"] == 13.
    assert audit["peaks_by_scope_n"]["before"]["thumb"] == 13.
    assert audit["task_success"] is None


def stepped_fixture(model):
    row = command_row()
    before = tactile(2.)
    cached, refreshed = raw(model, (3., 4., 50.)), raw(model, (4., 5., 50.))
    cache_forces, cache_patches = channels_from_native(model, cached)
    after_forces, after_patches = channels_from_native(model, refreshed)
    after = dict(timestamp_s=22.002, measured_pressures_n=after_forces, patches=after_patches)
    row.update(before_time_s=22., before_tactile=before, before_canonical_patches=before["patches"],
        after_tactile=after, after_canonical_patches=after_patches,
        after_robot_state=dict(joint_names=["joint1"], qpos_rad=[.000001], qvel_rad_s=[.001],
            actual_ctrl=row["actual_ctrl_for_step"], timestamp_s=22.002),
        after_wrist_pose=dict(timestamp_s=22.002))
    row["native_dynamics_cache"].update(step_start_s=22., step_end_s=22.002, timestamp_s=22.002,
        native_sample=cached, channel_forces_n=cache_forces, canonical_patches=cache_patches)
    result = dict(condition="STATIC_HOLD", task_success=None, initial_robot_state=deepcopy(row["before_robot_state"]),
        fixed_capture_deflection_rad=[0.], force_targets_n=dict.fromkeys(CHANNELS, 0.) | {"thumb": 2.},
        native_steps=1, required_native_steps=1, terminal="completed", completed=True, peak_channel_force_n=9.,
        native_audit=dict(integration_steps=1, external_target_qpos_writes=0, external_target_qvel_writes=0,
                          write_events=[], velocity_write_events=[]))
    return result, [row], [refreshed], deepcopy(row["before_robot_state"])


def test_executed_step_includes_before_cache_and_refreshed_native_scopes(model):
    args = stepped_fixture(model)
    audit = audit_branch(model, *args, CONTACT_CONFIG)
    assert audit["completed"] is True
    assert audit["peak_channel_force_n"] == 9.
    assert audit["peaks_by_scope_n"]["before"]["thumb"] == 2.
    assert audit["peaks_by_scope_n"]["cache"]["thumb"] == 7.
    assert audit["peaks_by_scope_n"]["after"]["thumb"] == 9.


@pytest.mark.parametrize("mutation", ["cache_command", "native_command", "cache_force", "time", "peak", "count"])
def test_executed_step_rejects_cache_native_command_force_timing_and_peak_changes(model, mutation):
    result, rows, native, state = stepped_fixture(model)
    if mutation == "cache_command":
        rows[0]["native_dynamics_cache"]["native_sample"]["motor_commands"] = [.1]
    elif mutation == "native_command":
        native[0]["motor_commands"] = [.1]
    elif mutation == "cache_force":
        rows[0]["native_dynamics_cache"]["channel_forces_n"]["thumb"] = 1.
    elif mutation == "time":
        rows[0]["native_dynamics_cache"]["step_start_s"] = 22.002
    elif mutation == "peak":
        result["peak_channel_force_n"] = 7.
    elif mutation == "count":
        result["native_audit"]["integration_steps"] = 2
    with pytest.raises(AuditError):
        audit_branch(model, result, rows, native, state, CONTACT_CONFIG)


@pytest.mark.parametrize("mutate", [
    lambda result, rows: result.update(peak_channel_force_n=2.),
    lambda result, rows: result.update(completed=True),
    lambda result, rows: result.update(task_success=True),
    lambda result, rows: result["initial_robot_state"].update(qpos_rad=[.1]),
    lambda result, rows: result["native_audit"].update(external_target_qvel_writes=1),
    lambda result, rows: rows[0].update(before_time_s=22.002),
])
def test_negative_evidence_cannot_hide_peak_claim_success_change_start_or_inject_state(model, mutate):
    result, rows, native, state = aborted_fixture(model)
    mutate(result, rows)
    with pytest.raises(AuditError):
        audit_branch(model, result, rows, native, state, CONTACT_CONFIG)


def test_archive_checks_recorded_members_bytes_and_before_after_not_current_glob(tmp_path):
    source = b"print('frozen source')\n"
    manifest = {"scripts/runner.py": hashlib.sha256(source).hexdigest()}
    protocol = dict(source_hashes_before=manifest, source_hashes_after=manifest.copy(), source_changed_during_run=False)
    with zipfile.ZipFile(tmp_path / "frozen_source.zip", "x") as archive:
        archive.writestr("scripts/runner.py", source)
    (tmp_path / "unrelated_new_source.py").write_text("unrelated current work\n")
    assert verify_source_archive(tmp_path, protocol) == 1
    protocol["source_hashes_after"]["src/new_file.py"] = "different"
    with pytest.raises(AuditError, match="source changed"):
        verify_source_archive(tmp_path, protocol)


@pytest.mark.parametrize("mutation", ["bytes", "extra", "duplicate"])
def test_archive_rejects_changed_extra_or_duplicate_source_members(tmp_path, mutation):
    source = b"frozen"
    manifest = {"scripts/runner.py": hashlib.sha256(source).hexdigest()}
    protocol = dict(source_hashes_before=manifest, source_hashes_after=manifest.copy(), source_changed_during_run=False)
    with zipfile.ZipFile(tmp_path / "frozen_source.zip", "x") as archive:
        archive.writestr("scripts/runner.py", b"changed" if mutation == "bytes" else source)
        if mutation == "extra":
            archive.writestr("extra.py", b"extra")
        elif mutation == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("scripts/runner.py", source)
    with pytest.raises(AuditError):
        verify_source_archive(tmp_path, protocol)


def test_nonfinite_values_are_rejected_even_inside_unused_abort_diagnostics():
    with pytest.raises(AuditError):
        finite_tree(dict(rows=[dict(abort=dict(detail=[float("inf")]))]))


def test_privileged_height_uses_integrated_free_qpos_not_cached_body_position(model):
    records = []
    target, table, hand = (model.geom(name).id for name in ("external", "table", "hand"))
    for step in range(1, 52):
        record = raw(model)
        record.update(timestamp=22. + step * .002, target_qpos=[0., 0., .071, 1., 0., 0., 0.],
                      target_position_world_m=[0., 0., 99.], native_contacts=[contact(hand, target, .051)])
        records.append(record)
    metrics = privileged_motion_metrics(model, records, initial_checkpoint_height_m=.01,
        initial_episode_height_m=0., target_geom_id=target, table_geom_id=table)
    assert metrics["maximum_free_joint_rise_from_checkpoint_m"] == pytest.approx(.061)
    assert metrics["maximum_free_joint_rise_from_episode_start_m"] == pytest.approx(.071)
    assert metrics["branch_observed_held_lift"] is True
    assert metrics["loaded_hand_without_loaded_table_samples"] == 51
    assert metrics["full_task_evaluated"] is False
    records[25]["native_contacts"].append(contact(target, table, .051))
    metrics = privileged_motion_metrics(model, records, initial_checkpoint_height_m=.01,
        initial_episode_height_m=0., target_geom_id=target, table_geom_id=table)
    assert not metrics["branch_observed_held_lift"]
    assert metrics["loaded_table_samples"] == 1
    assert metrics["maximum_target_table_normal_sum_n"] == .051


def test_privileged_loaded_contact_threshold_is_per_contact_not_force_sum(model):
    target, table, hand = (model.geom(name).id for name in ("external", "table", "hand"))
    record = raw(model)
    record["native_contacts"] = [contact(hand, target, .03), contact(hand, target, .03), contact(target, table, .05)]
    metrics = privileged_motion_metrics(model, [record], initial_checkpoint_height_m=0.,
        initial_episode_height_m=0., target_geom_id=target, table_geom_id=table)
    assert metrics["maximum_target_hand_normal_sum_n"] == .06
    assert metrics["loaded_hand_samples"] == 0
    assert metrics["loaded_table_samples"] == 0
