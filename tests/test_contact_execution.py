"""Native contact signs, joint coupling and non-mutating robot-only prediction."""

from pathlib import Path
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from kuka_sharpa_grasp.contact_execution import (
    ContactExecutionAdapter, read_native_contact_sample,
)
from kuka_sharpa_grasp.contact_policy import FingerContactPolicy
from kuka_sharpa_grasp.tactile import TactileContactReader


def model_with_contact(external_first=False):
    arm = '''
    <body name="arm_mount">
      <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
      <joint name="arm_hinge" axis="0 0 1" range="-1 1"/>
      <body name="left_thumb_distal" pos=".1 0 0">
        <joint name="left_thumb_IP" axis="0 0 1" range="-1 1"/>
        <geom name="hand_contact" type="sphere" size=".03" pos=".1 0 0" mass=".1"/>
      </body>
    </body>'''
    external = '''
    <body name="unlabelled_external_body" pos=".2 .05 0">
      <freejoint name="external_free"/>
      <geom type="sphere" size=".03" mass=".2"/>
    </body>'''
    world = external + arm if external_first else arm + external
    model = mujoco.MjModel.from_xml_string(f'''
    <mujoco><compiler angle="radian"/>
      <option timestep=".002" gravity="0 0 0"/>
      <worldbody>{world}</worldbody>
      <actuator>
        <position name="arm_hinge" joint="arm_hinge" kp="10" kv="1"
          forcelimited="true" forcerange="-1 1"/>
        <position name="left_thumb_IP" joint="left_thumb_IP" kp="10" kv="1"
          forcelimited="true" forcerange="-1 1"/>
      </actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = FingerContactPolicy(("thumb",), {"thumb": ("left_thumb_IP",)})
    return model, data, policy


def observation(model, data, policy):
    raw = read_native_contact_sample(model, data)
    return policy.validate_contacts(raw.patches, raw.measured_pressures_n,
        controlled_joint_names=policy.controlled_joint_names, timestamp_s=raw.timestamp_s)


@pytest.mark.parametrize("external_first", [False, True])
def test_native_patch_normal_always_points_from_hand_into_external_surface(external_first):
    model, data, policy = model_with_contact(external_first)
    raw = read_native_contact_sample(model, data)
    assert raw.patches and raw.measured_pressures_n["thumb"] > .05
    assert all(patch["hand_body_name"] == "left_thumb_distal" for patch in raw.patches)
    np.testing.assert_allclose(raw.patches[0]["normal_on_object_world"], [0., 1., 0.], atol=1e-12)
    assert "friction_coefficient" not in raw.patches[0]
    assert "target" not in raw.patches[0]
    expected = TactileContactReader(model).read(data)
    assert raw.measured_pressures_n["thumb"] == pytest.approx(expected.fingers["thumb"].normal_force_n)
    for value in (raw.patches[0], raw.measured_pressures_n):
        with pytest.raises(TypeError):
            value["modified"] = True
    observation(model, data, policy)


def test_arm_and_finger_jacobian_matches_independent_attached_point_finite_difference():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    jacobian = adapter.compression_jacobian(data, observed)
    np.testing.assert_allclose(jacobian, [[.2, .1]], atol=1e-12)
    patch = observed.patches[0]
    body = model.body(patch.hand_body_name).id
    local_point = data.xmat[body].reshape(3, 3).T @ (np.asarray(patch.position_world_m) - data.xpos[body])
    scratch = mujoco.MjData(model)
    finite = []
    for servo in adapter.servos:
        points = []
        for offset in (-1e-6, 1e-6):
            scratch.qpos[:] = data.qpos
            scratch.qpos[servo.qpos_index] += offset
            mujoco.mj_kinematics(model, scratch)
            points.append(scratch.xpos[body] + scratch.xmat[body].reshape(3, 3) @ local_point)
        finite.append(np.dot(patch.normal_on_object_world, (points[1] - points[0]) / 2e-6))
    np.testing.assert_allclose(jacobian[0], finite, atol=1e-10)


def test_jacobian_excludes_live_external_state_and_does_not_change_data():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    expected = adapter.compression_jacobian(data, observed)
    address = int(model.jnt_qposadr[model.joint("external_free").id])
    data.qpos[address:address + 3] = [99., -31., 7.]
    data.qvel[:] = .123
    data.ctrl[:] = .04
    before = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl", "xpos", "xmat")}
    np.testing.assert_array_equal(adapter.compression_jacobian(data, observed), expected)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


def test_real_joint_coupling_solver_produces_a_bound_motor_command_without_applying_it():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
    result = adapter.solve(data, observed, [.001],
        command_rate_lower_rad_s=[-.1, 0.], command_rate_upper_rad_s=[.1, 0.],
        deflection_lower_rad=[-.1, -.1], deflection_upper_rad=[.1, .1],
        maximum_contact_residual_m_s=1e-8)
    assert result.accepted
    np.testing.assert_allclose(result.command_velocity_rad_s, [.005, 0.], atol=1e-9)
    np.testing.assert_allclose(result.predicted_normal_velocity_m_s, [.001], atol=1e-10)
    assert np.max(np.abs(result.predicted_motor_torque_nm)) <= 1.
    for actual, expected in zip((data.qpos, data.qvel, data.ctrl, float(data.time)), before):
        np.testing.assert_array_equal(actual, expected)


def test_adapter_rejects_external_joint_masks_and_unsupported_motor_laws():
    model, data, policy = model_with_contact()
    with pytest.raises(ValueError, match="hinge position servo"):
        ContactExecutionAdapter(model, policy, arm_joint_names=("external_free",))
    model.actuator_gear[0, 0] = 2.
    with pytest.raises(ValueError, match="force law"):
        ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))


def test_stale_contact_timestamp_cannot_be_reused_as_current_geometry():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    data.time = .002
    with pytest.raises(ValueError, match="current native sample"):
        adapter.compression_jacobian(data, observed)


def test_unbound_observation_cannot_be_used_as_a_native_sample():
    model, data, policy = model_with_contact()
    observed = replace(observation(model, data, policy), timestamp_s=None)
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    with pytest.raises(ValueError, match="current native sample"):
        adapter.compression_jacobian(data, observed)


def test_revalidated_normal_controls_jacobian_scale():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    nonunit = replace(observed.patches[0], normal_on_object_world=(0., 1.2, 0.))
    observed = replace(observed, patches=(nonunit,))
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    np.testing.assert_allclose(adapter.compression_jacobian(data, observed), [[.2, .1]], atol=1e-12)


def test_noncanonical_contact_row_mask_cannot_silently_shift_velocity_targets():
    model, data, policy = model_with_contact()
    observed = observation(model, data, policy)
    unloaded = replace(observed.patches[0], input_index=999, finger="index",
        hand_body_name="left_index_distal", measured_pressure_n=0.)
    observed = replace(observed, patches=(*observed.patches, unloaded))
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=("arm_hinge",))
    with pytest.raises(ValueError, match="rows must already match"):
        adapter.compression_jacobian(data, observed)


def test_vendored_kuka_sharpa_position_servo_binding_covers_arm_and_all_fingers():
    from public_ycb_evidence import load_evidence_model

    root = Path(__file__).resolve().parents[1]
    scene = root / "results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/scene.xml"
    model = load_evidence_model(scene)
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    groups = {finger: tuple(f"left_{finger}_{suffix}" for suffix in
        (("MCP_FE", "IP") if finger == "thumb" else ("MCP_FE", "PIP", "DIP"))) for finger in fingers}
    policy = FingerContactPolicy(fingers, groups)
    adapter = ContactExecutionAdapter(model, policy,
        arm_joint_names=tuple(model.joint(index).name for index in range(7)))
    assert len(adapter.servos) == 21
    assert len({servo.actuator_id for servo in adapter.servos}) == 21
    assert all(servo.lower_torque_nm < 0. < servo.upper_torque_nm for servo in adapter.servos)
