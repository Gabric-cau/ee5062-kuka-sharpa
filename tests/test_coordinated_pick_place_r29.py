"""Critical R29 transitions and native command paths, without a full rollout."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "scripts")]
import run_coordinated_pick_place_r29 as runner
from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter, read_native_contact_sample
from kuka_sharpa_grasp.contact_policy import ContactObservation, FingerContactPolicy
from kuka_sharpa_grasp.coordinated_motion_reference import CoordinatedMotionReference


@pytest.fixture
def fixture():
    links = ''.join(f'<body name="arm{i}"><joint name="joint{i}" axis="0 0 1" range="-1 1"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>' for i in range(1, 8))
    actuators = ''.join(f'<position name="joint{i}" joint="joint{i}" kp="50" kv="1" '
        'forcelimited="true" forcerange="-80 80"/>' for i in range(1, 8))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
    <option timestep=".002" gravity="0 0 0"/><worldbody>
    <geom name="table" type="plane" size="1 1 .01" pos="0 0 -1"/>
    <body name="target" pos=".4 0 .055"><freejoint name="target_free"/>
    <geom name="target_geom" type="sphere" size=".03" mass=".1"/></body>
    {links}<body name="link7" pos=".2 0 0"><body name="left_thumb_distal" pos=".1 0 0">
    <joint name="left_thumb_IP" axis="0 1 0" range="-1 1"/>
    <geom type="sphere" size=".03" pos=".1 0 0" mass=".1"/></body></body>{'</body>' * 7}
    </worldbody><actuator>{actuators}<position name="left_thumb_IP" joint="left_thumb_IP"
    kp="10" kv="1" forcelimited="true" forcerange="-1 1"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = FingerContactPolicy(("thumb",), {"thumb": ("left_thumb_IP",)})
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=tuple(f"joint{i}" for i in range(1, 8)))
    return model, data, adapter


def test_arrival_requires_continuous_position_and_velocity_and_resets_after_violation():
    gate, config = runner.Arrival(), runner.Configuration()
    assert not gate.update(1., .0004, .0004, config)
    assert not gate.update(1.14, .0004, .0004, config)
    assert not gate.update(1.15, .0004, .0006, config)
    assert not gate.update(1.2, .0004, .0004, config)
    assert gate.update(1.35, .0004, .0004, config)
    assert not gate.update(1.36, .0006, .0004, config)


@pytest.mark.parametrize("changes", [{"maximum_continuation_s": float("nan")}, {"channel_abort_n": 12.1},
    {"release_rate_rad_s": .151}, {"maximum_linear_speed_m_s": .0041},
    {"force_target_margin": .9}, {"arrival_continuous_s": True}])
def test_invalid_configuration_does_not_relax_original_limits(changes):
    with pytest.raises(ValueError):
        runner.Configuration(**changes)


def test_empty_contact_release_and_recovery_use_valid_zero_jacobian_bounded_commands(fixture):
    model, data, adapter = fixture
    sample = read_native_contact_sample(model, data)
    empty = ContactObservation((), sample.measured_pressures_n, (), (), float(data.time))
    motion = CoordinatedMotionReference(model, joint_names=adapter.joint_names)
    ref = motion.prepare(data, empty, desired_linear_velocity_world_m_s=(0., 0., 0.),
        joint_velocity_limits_rad_s=(.025,) * 7 + (.15,))
    data.ctrl[-1] = .02
    bias = data.ctrl.copy() - data.qpos[[servo.qpos_index for servo in adapter.servos]]
    before = runner.r27.integration_state(model, data)[1]
    release = runner.solve_command(model, data, adapter, empty, ref, {}, bias, True)
    recovery = runner.solve_command(model, data, adapter, empty, ref, {}, bias, False,
        holding_without_contacts=True)
    assert release.accepted and recovery.accepted
    assert release.command_velocity_rad_s[-1] == pytest.approx(-.15)
    assert release.command_rad[-1] == pytest.approx(.0197)
    assert recovery.command_velocity_rad_s[-1] == pytest.approx(0.)
    assert recovery.command_rad[-1] == pytest.approx(.02)
    np.testing.assert_array_equal(release.predicted_normal_velocity_m_s, [0.])
    np.testing.assert_array_equal(recovery.predicted_normal_velocity_m_s, [0.])
    np.testing.assert_array_equal(runner.r27.integration_state(model, data)[1], before)


def prior():
    return SimpleNamespace(catalogue_mass_kg=.1, evaluate=lambda *args, **kwargs:
        {"estimated_origin_world_m": (.65, -.18, .38)})


def test_actual_native_steps_record_strict_inputs_and_do_not_skip_preload(fixture):
    model, data, adapter = fixture
    journal = runner.FullJournal(model, data)
    targets = dict(read_native_contact_sample(model, data).measured_pressures_n)
    result = runner.run_motion(model, data, adapter, journal, prior(), targets,
        runner.Configuration(maximum_continuation_s=.006))
    assert result["failure"] == "continuation_timeout"
    assert [row["state"] for row in result["transitions"]] == ["PRELOAD"]
    assert journal.integration_steps == len(journal.native) == len(journal.trace) == 3
    assert journal.monitor.result()["accepted_samples"] == 3
    for row, native in zip(result["rows"], journal.native, strict=True):
        assert row["command_applied"] and row["solver"]["accepted"]
        assert native["force_cache_scope"].startswith("Private full-state")
        assert native["native_contacts"] and native["target_qpos"]
        np.testing.assert_array_equal(row["actual_ctrl_for_step"], native["motor_commands"])
        np.testing.assert_allclose(row["cached_motor_torque_nm"], row["solver"]["predicted_motor_torque_nm"], atol=1e-12)
        assert row["support_gate"]["raw_load_n"] == row["table_sensor_load_n"]
    audit = journal.audit()
    assert audit["external_target_qpos_writes"] == audit["external_target_qvel_writes"] == 0


def test_poststep_safety_failure_preserves_actual_native_step_and_solver_record(fixture, monkeypatch):
    model, data, adapter = fixture
    journal = runner.FullJournal(model, data)
    original = journal.assert_safety
    def safety():
        original()
        if journal.integration_steps:
            raise runner.r27.ExperimentAbort("whole_episode_force_limit")
    monkeypatch.setattr(journal, "assert_safety", safety)
    result = runner.run_motion(model, data, adapter, journal, prior(),
        dict(read_native_contact_sample(model, data).measured_pressures_n), runner.Configuration(maximum_continuation_s=.006))
    assert result["failure"] == "whole_episode_force_limit"
    assert journal.integration_steps == len(journal.native) == len(journal.trace) == 1
    assert journal.monitor.result()["observed_samples"] == 1
    row = result["rows"][0]
    assert row["command_applied"] and row["solver"]["accepted"]
    assert row["abort"]["reason"] == "whole_episode_force_limit"
    assert "cached_motor_torque_nm" in row


def test_lost_contact_recovery_takes_a_bounded_physical_step_without_opening(fixture):
    model, data, adapter = fixture
    # Remove contact only in initial fixture state, before any journal starts.
    address = int(model.jnt_qposadr[model.joint("target_free").id])
    data.qpos[address:address + 3] = [1., 1., 1.]
    mujoco.mj_forward(model, data)
    journal = runner.FullJournal(model, data)
    result = runner.run_motion(model, data, adapter, journal, prior(), {},
        runner.Configuration(maximum_continuation_s=.006))
    assert result["failure"].startswith("contact_contract_abort") or result["failure"] == "execution_error"
    assert [row["state"] for row in result["transitions"]][:2] == ["PRELOAD", "RECOVERY_LOWER"]
    assert journal.integration_steps >= 1
    recovery_rows = [row for row in result["rows"] if row["state"] == "RECOVERY_LOWER"]
    assert recovery_rows and recovery_rows[0]["command_applied"]
    assert not recovery_rows[0]["opening"]
    assert recovery_rows[0]["solver"]["command_velocity_rad_s"][-1] == pytest.approx(0.)
