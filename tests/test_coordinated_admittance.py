"""Reference optimization contracts; no contact retention or rollout claims."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.coordinated_admittance import (
    position_servo_velocity_box, solve_coordinated_admittance,
)
import kuka_sharpa_grasp.coordinated_admittance as coordinated


def state(n=2, **changes):
    values = dict(actual_position_rad=np.zeros(n), actual_velocity_rad_s=np.zeros(n),
        previous_command_rad=np.zeros(n), timestep_s=.1,
        joint_lower_rad=np.full(n, -2.), joint_upper_rad=np.full(n, 2.),
        command_rate_lower_rad_s=np.full(n, -.5), command_rate_upper_rad_s=np.full(n, .5),
        deflection_lower_rad=np.full(n, -.2), deflection_upper_rad=np.full(n, .2),
        servo_kp_nm_rad=np.full(n, 10.), servo_kd_nm_s_rad=np.ones(n),
        torque_lower_nm=np.full(n, -2.), torque_upper_nm=np.full(n, 2.))
    values.update(changes)
    return values


def test_shared_arm_motion_solves_contacts_that_finger_only_motion_cannot():
    # Wrist compression helps one patch and unloads the other. Finger 1 cannot
    # close further; coordinated wrist motion plus finger 2 still solves both.
    matrix = np.array([[.01, .01, 0.], [-.01, 0., .01]])
    inputs = state(3, command_rate_lower_rad_s=np.array([-.5, 0., -.5]),
        command_rate_upper_rad_s=np.array([.5, 0., .5]))
    target = np.array([.001, .001])
    result = solve_coordinated_admittance(matrix, target,
        maximum_contact_residual_m_s=1e-10, **inputs)
    assert result.accepted and result.solver_converged
    np.testing.assert_allclose(result.command_velocity_rad_s, [.1, 0., .2], atol=1e-10)
    np.testing.assert_allclose(result.predicted_normal_velocity_m_s, target, atol=1e-12)
    assert result.active_constraints["fixed_velocity"] == (1,)
    inputs["command_rate_lower_rad_s"][0] = 0.
    inputs["command_rate_upper_rad_s"][0] = 0.
    independent = solve_coordinated_admittance(matrix, target,
        maximum_contact_residual_m_s=1e-10, **inputs)
    assert independent.solver_converged and not independent.accepted
    assert independent.reason == "CONTACT_TRACKING_RESIDUAL"
    assert independent.predicted_normal_velocity_m_s[0] == 0.


def test_explicit_soft_task_weights_trade_off_without_claiming_exact_tracking():
    result = solve_coordinated_admittance([[1.]], [1.], **state(1),
        task_reference_velocity_rad_s=[0.], task_weights=[3.], contact_weights=[1.],
        maximum_contact_residual_m_s=.8)
    assert result.accepted
    assert result.command_velocity_rad_s[0] == pytest.approx(.25)
    assert result.contact_residual_m_s[0] == pytest.approx(-.75)
    assert result.task_residual_rad_s[0] == pytest.approx(.25)
    strict = solve_coordinated_admittance([[1.]], [1.], **state(1),
        task_reference_velocity_rad_s=[0.], task_weights=[3.], maximum_contact_residual_m_s=.01)
    assert not strict.accepted and strict.solver_converged
    assert strict.reason == "CONTACT_TRACKING_RESIDUAL"


def test_torque_and_command_rate_conflict_is_rejected_without_zero_fallback():
    inputs = state(1, previous_command_rad=[.2],
        command_rate_lower_rad_s=[-.01], command_rate_upper_rad_s=[.01],
        torque_lower_nm=[-.5], torque_upper_nm=[.5])
    box = position_servo_velocity_box(**inputs)
    assert not box.accepted and box.reason == "EMPTY_REFERENCE_BOX"
    assert box.conflicting_joint_indices == (0,)
    assert box.component_lower_rad["command_rate"][0] == pytest.approx(.199)
    assert box.component_upper_rad["predicted_servo_torque"][0] == pytest.approx(.05)
    result = solve_coordinated_admittance([[1.]], [0.], **inputs)
    assert not result.accepted and not result.solver_converged
    assert result.command_velocity_rad_s is None and result.command_rad is None


def test_damping_shifts_torque_box_and_reports_active_torque_limit():
    inputs = state(1, actual_position_rad=[.3], actual_velocity_rad_s=[.4],
        previous_command_rad=[.3], timestep_s=.1, servo_kp_nm_rad=[10.], servo_kd_nm_s_rad=[2.],
        torque_lower_nm=[-.1], torque_upper_nm=[.1],
        command_rate_lower_rad_s=[-2.], command_rate_upper_rad_s=[2.])
    box = position_servo_velocity_box(**inputs)
    np.testing.assert_allclose(box.lower_rad_s, [.7])
    np.testing.assert_allclose(box.upper_rad_s, [.9])
    result = solve_coordinated_admittance([[1.]], [1.], **inputs,
        maximum_contact_residual_m_s=.11)
    assert result.accepted
    assert result.command_velocity_rad_s[0] == pytest.approx(.9, abs=1e-9)
    assert result.predicted_motor_torque_nm[0] == pytest.approx(.1, abs=1e-9)
    assert result.active_constraints["predicted_servo_torque_upper"] == (0,)
    assert result.diagnostics["clipping_used"] is False
    assert result.diagnostics["dynamic_force_guarantee"] is False


def test_equal_bounds_are_solved_exactly_without_expanding_or_clipping():
    result = solve_coordinated_admittance([[1., 1.]], [.125],
        **state(2, timestep_s=.125, command_rate_lower_rad_s=[.125, 0.], command_rate_upper_rad_s=[.125, 0.]),
        maximum_contact_residual_m_s=0.)
    assert result.accepted
    np.testing.assert_array_equal(result.command_velocity_rad_s, [.125, 0.])
    assert result.active_constraints["fixed_velocity"] == (0, 1)
    assert result.diagnostics["iterations"] == 0


def test_conflicting_bounds_are_not_tolerantly_clipped_into_success():
    inputs = state(1, previous_command_rad=[.1], timestep_s=1.,
        command_rate_lower_rad_s=[0.], command_rate_upper_rad_s=[0.],
        torque_lower_nm=[-1.], torque_upper_nm=[1. - 1e-12])
    result = solve_coordinated_admittance([[1.]], [0.], **inputs, bound_tolerance_rad_s=1.)
    assert not result.accepted and result.reason == "EMPTY_REFERENCE_BOX"
    assert result.command_rad is None


@pytest.mark.parametrize("changes", [
    {"timestep_s": 0.}, {"timestep_s": float("nan")}, {"timestep_s": True},
    {"actual_position_rad": [0., float("inf")]}, {"actual_velocity_rad_s": [0.]},
    {"servo_kp_nm_rad": [0., 1.]}, {"servo_kd_nm_s_rad": [-1., 1.]},
    {"joint_lower_rad": [1., 0.], "joint_upper_rad": [0., 1.]},
    {"command_rate_lower_rad_s": [1., 0.], "command_rate_upper_rad_s": [0., 1.]},
    {"deflection_lower_rad": [1., 0.], "deflection_upper_rad": [0., 1.]},
    {"torque_lower_nm": [1., 0.], "torque_upper_nm": [0., 1.]},
    {"previous_command_rad": [False, .1]},
])
def test_invalid_joint_state_or_bounds_are_rejected(changes):
    result = solve_coordinated_admittance(np.eye(2), [0., 0.], **state(**changes))
    assert not result.accepted and result.command_rad is None
    assert result.reason == "INVALID_INPUT"


@pytest.mark.parametrize("matrix,target,options", [
    ([[1.]], [0.], {}), ([[float("nan"), 0.]], [0.], {}),
    ([[1., 0.]], [float("inf")], {}), ([[1., 0.]], [0., 0.], {}),
    ([[1., 0.]], [0.], {"task_weights": [-1., 0.]}),
    ([[1., 0.]], [0.], {"contact_weights": [0.], "task_weights": [0., 0.]}),
    ([[1., 0.]], [0.], {"contact_weights": [True]}),
    ([[1., 0.]], [0.], {"task_reference_velocity_rad_s": [0.]}),
    ([[1., 0.]], [0.], {"maximum_contact_residual_m_s": -1.}),
    ([[1., 0.]], [0.], {"max_iterations": True}),
])
def test_invalid_objective_is_rejected(matrix, target, options):
    result = solve_coordinated_admittance(matrix, target, **state(), **options)
    assert not result.accepted and result.command_rad is None
    assert result.reason == "INVALID_INPUT_OR_NUMERICAL_FAILURE"


def test_input_arrays_are_not_modified():
    inputs = state()
    before = {name: value.copy() for name, value in inputs.items() if isinstance(value, np.ndarray)}
    matrix, target = np.eye(2), np.array([.1, -.1])
    result = solve_coordinated_admittance(matrix, target, **inputs)
    assert result.accepted
    for name, value in before.items():
        np.testing.assert_array_equal(inputs[name], value)
    np.testing.assert_array_equal(matrix, np.eye(2))
    np.testing.assert_array_equal(target, [.1, -.1])


@pytest.mark.parametrize("candidate,success,reason", [
    ([.6], True, "REFERENCE_BOUND_VIOLATION"),
    ([0.], False, "SOLVER_DID_NOT_CONVERGE"),
    ([float("nan")], True, "NONFINITE_SOLVER_CANDIDATE"),
])
def test_solver_candidates_are_independently_checked_without_clipping(monkeypatch, candidate, success, reason):
    monkeypatch.setattr(coordinated, "lsq_linear", lambda *args, **kwargs: SimpleNamespace(
        x=np.asarray(candidate), success=success, status=1, message="Synthetic solver candidate",
        nit=1, optimality=0.))
    result = solve_coordinated_admittance([[1.]], [0.], **state(1), maximum_contact_residual_m_s=1.)
    assert not result.accepted and result.reason == reason
    if np.isfinite(candidate).all():
        np.testing.assert_array_equal(result.command_velocity_rad_s, candidate)
        assert result.diagnostics["clipping_used"] is False
    else:
        assert result.command_rad is None


def test_torque_is_rechecked_after_command_reconstruction(monkeypatch):
    monkeypatch.setattr(coordinated, "lsq_linear", lambda *args, **kwargs: SimpleNamespace(
        x=np.array([.5000000001]), success=True, status=1, message="Synthetic rounding excess",
        nit=1, optimality=0.))
    result = solve_coordinated_admittance([[1.]], [0.], **state(1, servo_kp_nm_rad=[1e9],
        torque_lower_nm=[-5e7], torque_upper_nm=[5e7]), maximum_contact_residual_m_s=1.,
        bound_tolerance_rad_s=1e-9, torque_tolerance_nm=1e-9)
    assert not result.accepted and result.reason == "REFERENCE_BOUND_VIOLATION"
    assert result.diagnostics["maximum_velocity_bound_violation_rad_s"] < 1e-9
    assert result.diagnostics["maximum_torque_bound_violation_nm"] > 1e-3


def test_real_mujoco_arm_and_hand_position_servo_binding_without_integration(tmp_path):
    import mujoco
    from kuka_sharpa_grasp.scene import build_scene

    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "compiled.xml", "ycb_can",
        physics_profile="physical-v6")))
    data = mujoco.MjData(model)
    joint_names = ("joint1", "joint2", "left_middle_MCP_FE", "left_ring_MCP_FE")
    actuator_names = ("actuator1", "actuator2", *joint_names[2:])
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names])
    assert np.all(joints >= 0) and np.all(actuators >= 0)
    np.testing.assert_array_equal(model.actuator_trnid[actuators, 0], joints)
    np.testing.assert_array_equal(model.actuator_gear[actuators], np.tile([1., 0., 0., 0., 0., 0.], (4, 1)))
    gain, bias = model.actuator_gainprm[actuators], model.actuator_biasprm[actuators]
    assert np.all(model.actuator_gaintype[actuators] == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype[actuators] == mujoco.mjtBias.mjBIAS_AFFINE)
    np.testing.assert_array_equal(bias[:, 1], -gain[:, 0])
    np.testing.assert_array_equal(bias[:, 0], 0.)
    qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    actual = np.mean(model.jnt_range[joints], axis=1)
    data.qpos[qpos], data.qvel[dofs] = actual, [.01, -.01, .02, -.02]
    data.ctrl[actuators] = actual - bias[:, 2] * data.qvel[dofs] / gain[:, 0]
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_middle_DP")
    jacobian, rotation_jacobian = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacobian, rotation_jacobian, data.xpos[body], body)
    matrix = jacobian[[2], :][:, dofs]  # Synthetic normal direction, robot geometry only.
    reference = np.array([.001, -.001, .002, -.002])
    command_lower = model.jnt_range[joints, 0].copy()
    command_upper = model.jnt_range[joints, 1].copy()
    torque_lower, torque_upper = np.full(4, -1.), np.full(4, 1.)
    for index, (joint, actuator) in enumerate(zip(joints, actuators, strict=True)):
        if model.actuator_ctrllimited[actuator]:
            command_lower[index] = max(command_lower[index], model.actuator_ctrlrange[actuator, 0])
            command_upper[index] = min(command_upper[index], model.actuator_ctrlrange[actuator, 1])
        if model.actuator_forcelimited[actuator]:
            torque_lower[index] = max(torque_lower[index], model.actuator_forcerange[actuator, 0])
            torque_upper[index] = min(torque_upper[index], model.actuator_forcerange[actuator, 1])
        if model.jnt_actfrclimited[joint]:
            torque_lower[index] = max(torque_lower[index], model.jnt_actfrcrange[joint, 0])
            torque_upper[index] = min(torque_upper[index], model.jnt_actfrcrange[joint, 1])
    result = solve_coordinated_admittance(matrix, matrix @ reference,
        actual_position_rad=data.qpos[qpos], actual_velocity_rad_s=data.qvel[dofs],
        previous_command_rad=data.ctrl[actuators], timestep_s=model.opt.timestep,
        joint_lower_rad=command_lower, joint_upper_rad=command_upper,
        command_rate_lower_rad_s=np.full(4, -.1), command_rate_upper_rad_s=np.full(4, .1),
        deflection_lower_rad=np.full(4, -.05), deflection_upper_rad=np.full(4, .05),
        servo_kp_nm_rad=gain[:, 0], servo_kd_nm_s_rad=-bias[:, 2],
        torque_lower_nm=torque_lower, torque_upper_nm=torque_upper,
        task_reference_velocity_rad_s=reference, task_weights=np.ones(4),
        maximum_contact_residual_m_s=1e-10)
    assert result.accepted, (result.reason, result.diagnostics, result.bounds)
    before = data.qpos.copy()
    data.ctrl[actuators] = result.command_rad
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_allclose(data.actuator_force[actuators], result.predicted_motor_torque_nm, atol=1e-10)
    assert data.time == 0.  # Forward evaluation is not a rollout or integration.
