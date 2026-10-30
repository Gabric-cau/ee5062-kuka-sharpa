"""Pure robot-kinematic references; no dynamic grasp or lift experiment."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from kuka_sharpa_grasp.contact_policy import CONTACT_CHANNELS, ContactObservation, ContactPatch
from kuka_sharpa_grasp.coordinated_motion_reference import CoordinatedMotionReference


XML = '''<mujoco><compiler angle="radian"/><option timestep=".002"/>
<worldbody>
  <body name="unlabelled_external_body" pos="5 0 0"><freejoint name="external_free"/>
    <geom type="sphere" size=".01" mass=".1"/></body>
  <body name="robot_base"><joint name="arm_yaw" axis="0 0 1" range="-2 2"/>
    <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
    <body name="elbow" pos=".3 0 0"><joint name="arm_pitch" axis="0 1 0" range="-2 2"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
      <body name="link7" pos=".25 0 0">
        <body name="left_thumb_distal" pos=".05 0 0">
          <joint name="left_thumb_IP" axis="0 0 1" range="-1 1"/>
          <geom type="sphere" size=".01" pos=".05 0 0" mass=".1"/>
        </body>
      </body>
    </body>
  </body>
</worldbody></mujoco>'''


@pytest.fixture
def fixture():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    joints = ("arm_yaw", "arm_pitch", "left_thumb_IP")
    for name, value in zip(joints, (.2, -.3, .4), strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    data.time = .12
    mujoco.mj_forward(model, data)
    wrist = data.xpos[model.body("link7").id].copy()
    patches = (
        ContactPatch(7, "thumb", "left_thumb_distal", tuple(wrist + [.1, .02, -.01]), (0., 1., 0.), 3., None),
        ContactPatch(2, "thumb", "left_thumb_distal", tuple(wrist + [.03, -.01, .04]), (0., 0., 1.), 2., None))
    observed = ContactObservation(patches, dict.fromkeys(CONTACT_CHANNELS, 0.) | {"thumb": 5.},
        ("thumb",), (), float(data.time))
    return model, data, joints, observed, CoordinatedMotionReference(model, joint_names=joints)


def prepare(builder, data, observed, **overrides):
    options = dict(desired_linear_velocity_world_m_s=(0., 0., .004),
        joint_velocity_limits_rad_s=(.025, .025, .05))
    options.update(overrides)
    return builder.prepare(data, observed, **options)


def test_actual_robot_wrist_jacobian_matches_independent_pose_finite_difference(fixture):
    model, data, joints, observed, builder = fixture
    result = prepare(builder, data, observed)
    wrist = model.body("link7").id
    scratch = mujoco.MjData(model)
    columns = []
    epsilon = 1e-6
    for name in joints:
        positions, rotations = [], []
        for sign in (-1., 1.):
            scratch.qpos[:] = data.qpos
            scratch.qpos[model.jnt_qposadr[model.joint(name).id]] += sign * epsilon
            mujoco.mj_kinematics(model, scratch)
            positions.append(scratch.xpos[wrist].copy())
            rotations.append(scratch.xmat[wrist].reshape(3, 3).copy())
        columns.append(np.r_[(positions[1] - positions[0]) / (2. * epsilon),
            Rotation.from_matrix(rotations[1] @ rotations[0].T).as_rotvec() / (2. * epsilon)])
    np.testing.assert_allclose(result.wrist_jacobian_world, np.array(columns).T, atol=1e-10)
    np.testing.assert_array_equal(np.array(result.wrist_jacobian_world)[:, 2], np.zeros(6))
    assert result.task_reference_velocity_rad_s[2] == 0.


def test_normal_feedforward_matches_rigid_point_velocity_and_preserves_patch_order(fixture):
    model, data, joints, observed, builder = fixture
    result = prepare(builder, data, observed, desired_linear_velocity_world_m_s=(0., .002, .003),
        desired_angular_velocity_world_rad_s=(0., .01, .02))
    wrist = np.array(result.wrist_position_world_m)
    linear = np.array(result.achieved_linear_velocity_world_m_s)
    angular = np.array(result.achieved_angular_velocity_world_rad_s)
    expected = [np.dot(patch.normal_on_object_world,
        linear + np.cross(angular, np.array(patch.position_world_m) - wrist)) for patch in observed.patches]
    np.testing.assert_allclose(result.rigid_transport_normal_velocity_m_s, expected, atol=1e-15)
    assert result.patch_input_indices == (7, 2)
    assert np.max(np.abs(expected)) > 1e-5
    # Independently differentiate a transported point, including rotation.
    delta = 1e-6
    for patch, target in zip(observed.patches, expected, strict=True):
        offset = np.array(patch.position_world_m) - wrist
        plus = wrist + delta * linear + Rotation.from_rotvec(delta * angular).apply(offset)
        minus = wrist - delta * linear + Rotation.from_rotvec(-delta * angular).apply(offset)
        finite = np.dot(patch.normal_on_object_world, (plus - minus) / (2. * delta))
        assert target == pytest.approx(finite, abs=1e-10)


def test_transport_equals_full_hand_point_jacobian_on_pure_rigid_wrist_task(fixture):
    model, data, joints, observed, builder = fixture
    result = prepare(builder, data, observed, desired_angular_velocity_world_rad_s=(0., .01, .02))
    dofs = [int(model.jnt_dofadr[model.joint(name).id]) for name in joints]
    hand_jacobian = np.zeros((3, model.nv))
    normal_velocity = []
    for patch in observed.patches:
        mujoco.mj_jac(model, data, hand_jacobian, None, np.array(patch.position_world_m),
            model.body(patch.hand_body_name).id)
        normal_velocity.append(np.array(patch.normal_on_object_world) @ hand_jacobian[:, dofs]
            @ np.array(result.task_reference_velocity_rad_s))
    # Zero pressure correction then leaves the rigid arm task kinematically
    # consistent with the contact target rather than demanding a fixed surface.
    np.testing.assert_allclose(result.rigid_transport_normal_velocity_m_s, normal_velocity, atol=1e-15)


def test_pose_reference_errors_are_world_frame_and_twist_limits_are_explicit(fixture):
    model, data, joints, observed, builder = fixture
    wrist = model.body("link7").id
    actual_position = data.xpos[wrist].copy()
    actual_rotation = data.xmat[wrist].reshape(3, 3).copy()
    rotation_error = np.array([.02, -.01, .03])
    result = prepare(builder, data, observed, desired_linear_velocity_world_m_s=(0., 0., 0.),
        desired_position_world_m=actual_position + [.01, -.02, .03],
        desired_rotation_world=Rotation.from_rotvec(rotation_error).as_matrix() @ actual_rotation,
        position_gain_s_inv=2., orientation_gain_s_inv=3., max_linear_speed_m_s=.004,
        max_angular_speed_rad_s=.05)
    np.testing.assert_allclose(result.position_error_world_m, [.01, -.02, .03], atol=1e-15)
    np.testing.assert_allclose(result.orientation_error_world_rad, rotation_error, atol=1e-15)
    np.testing.assert_allclose(result.requested_linear_velocity_world_m_s, [.02, -.04, .06], atol=1e-15)
    np.testing.assert_allclose(result.requested_angular_velocity_world_rad_s, 3. * rotation_error, atol=1e-15)
    assert np.linalg.norm(result.limited_linear_velocity_world_m_s) == pytest.approx(.004)
    assert np.linalg.norm(result.limited_angular_velocity_world_rad_s) == pytest.approx(.05)


def test_joint_speed_limit_uniformly_scales_task_and_transport_uses_achieved_motion(fixture):
    model, data, joints, observed, builder = fixture
    unrestricted = prepare(builder, data, observed, joint_velocity_limits_rad_s=(1., 1., 1.))
    limited = prepare(builder, data, observed, joint_velocity_limits_rad_s=(.00001, .00001, .05))
    assert 0. < limited.joint_velocity_scale < 1.
    assert np.max(np.abs(limited.task_reference_velocity_rad_s[:2])) <= .00001 + 1e-15
    np.testing.assert_allclose(limited.task_reference_velocity_rad_s,
        np.array(unrestricted.task_reference_velocity_rad_s) * limited.joint_velocity_scale, atol=1e-15)
    np.testing.assert_allclose(limited.rigid_transport_normal_velocity_m_s,
        np.array(unrestricted.rigid_transport_normal_velocity_m_s) * limited.joint_velocity_scale, atol=1e-15)
    locked = prepare(builder, data, observed, joint_velocity_limits_rad_s=(0., 0., .05))
    np.testing.assert_array_equal(locked.task_reference_velocity_rad_s, np.zeros(3))
    np.testing.assert_array_equal(locked.rigid_transport_normal_velocity_m_s, np.zeros(2))


def test_numpy_reference_inputs_are_not_modified_or_aliased_by_result(fixture):
    model, data, joints, observed, builder = fixture
    linear, angular, limits = np.array([0., 0., .004]), np.array([0., .01, .02]), np.array([.025, .025, .05])
    copies = [array.copy() for array in (linear, angular, limits)]
    result = prepare(builder, data, observed, desired_linear_velocity_world_m_s=linear,
        desired_angular_velocity_world_rad_s=angular, joint_velocity_limits_rad_s=limits)
    for array, original in zip((linear, angular, limits), copies, strict=True):
        np.testing.assert_array_equal(array, original)
    snapshot = result.requested_linear_velocity_world_m_s
    linear[:] = 99.
    assert result.requested_linear_velocity_world_m_s == snapshot


def test_no_live_external_state_material_or_dynamics_cache_is_used_and_live_data_is_unchanged(fixture, monkeypatch):
    model, data, joints, observed, builder = fixture
    expected = prepare(builder, data, observed)
    external = model.joint("external_free").id
    address = int(model.jnt_qposadr[external])
    data.qpos[address:address + 7] = np.nan
    data.qvel[:] = np.nan
    data.xpos[:] = np.nan
    data.xmat[:] = np.nan
    data.qfrc_applied[:] = 3.
    model.geom_friction[:] = np.nan
    # The reference uses tactile geometry, not a material or pressure estimate.
    observed = replace(observed, patches=tuple(replace(patch, friction_coefficient=object(),
        measured_pressure_n=object()) for patch in observed.patches), measured_pressures_n={"unreadable": object()})
    before = {name: getattr(data, name).copy() for name in
        ("qpos", "qvel", "ctrl", "xpos", "xmat", "qfrc_applied")}
    def no_dynamics(*args, **kwargs):
        pytest.fail("Motion reference must not call a dynamics/contact evaluation")
    monkeypatch.setattr(mujoco, "mj_forward", no_dynamics)
    monkeypatch.setattr(mujoco, "mj_step", no_dynamics)
    actual = prepare(builder, data, observed)
    assert actual == expected
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


@pytest.mark.parametrize("change", [
    {"desired_linear_velocity_world_m_s": (0., 0., float("nan"))},
    {"desired_angular_velocity_world_rad_s": (0., float("inf"), 0.)},
    {"desired_linear_velocity_world_m_s": (True, False, True)},
    {"desired_position_world_m": (1., 2.)}, {"desired_rotation_world": np.zeros((3, 3))},
    {"desired_rotation_world": np.diag([1., 1., -1.])},
    {"damping_m_rad": 0.}, {"damping_m_rad": float("nan")}, {"position_gain_s_inv": -1.},
    {"max_linear_speed_m_s": False}, {"rotational_length_scale_m": 0.},
    {"joint_velocity_limits_rad_s": (.1, -.1, .1)},
    {"joint_velocity_limits_rad_s": (.1, float("nan"), .1)},
    {"joint_velocity_limits_rad_s": (.1, .1)}])
def test_invalid_reference_inputs_are_rejected(fixture, change):
    model, data, joints, observed, builder = fixture
    with pytest.raises(ValueError):
        prepare(builder, data, observed, **change)


@pytest.mark.parametrize("change", ["stale", "untimed", "zero_normal", "nan_point", "external_body", "duplicate_index"])
def test_invalid_observation_is_rejected(fixture, change):
    model, data, joints, observed, builder = fixture
    if change == "stale":
        observed = replace(observed, timestamp_s=.1)
    elif change == "untimed":
        observed = replace(observed, timestamp_s=None)
    else:
        first, second = observed.patches
        if change == "zero_normal":
            first = replace(first, normal_on_object_world=(0., 0., 0.))
        elif change == "nan_point":
            first = replace(first, position_world_m=(0., 0., float("nan")))
        elif change == "external_body":
            first = replace(first, hand_body_name="unlabelled_external_body")
        else:
            first = replace(first, input_index=second.input_index)
        observed = replace(observed, patches=(first, second))
    with pytest.raises(ValueError):
        prepare(builder, data, observed)


def test_nonfinite_robot_state_is_rejected_and_return_value_is_deeply_immutable(fixture):
    model, data, joints, observed, builder = fixture
    result = prepare(builder, data, observed)
    with pytest.raises(FrozenInstanceError):
        result.timestamp_s = 1.
    with pytest.raises(TypeError):
        result.task_reference_velocity_rad_s[0] = 1.
    with pytest.raises(TypeError):
        result.wrist_jacobian_world[0][0] = 1.
    data.qpos[model.jnt_qposadr[model.joint(joints[0]).id]] = np.nan
    with pytest.raises(ValueError, match="robot wrist-ancestor positions"):
        prepare(builder, data, observed)


def test_external_joint_or_invalid_robot_mask_cannot_be_bound(fixture):
    model, data, joints, observed, builder = fixture
    for names in (("external_free",), ("arm_yaw", "arm_yaw"), (), "arm_yaw"):
        with pytest.raises(ValueError):
            CoordinatedMotionReference(model, joint_names=names)
    with pytest.raises(ValueError, match="wrist"):
        CoordinatedMotionReference(model, joint_names=joints, wrist_body_name="absent")


def test_actual_vendored_kuka_sharpa_wrist_reference_has_robot_only_finger_zero_columns():
    from public_ycb_evidence import load_evidence_model
    from kuka_sharpa_grasp.contact_execution import ContactExecutionAdapter
    from kuka_sharpa_grasp.contact_policy import FingerContactPolicy

    root = Path(__file__).resolve().parents[1]
    model = load_evidence_model(root / "results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/scene.xml")
    data = mujoco.MjData(model)
    policy = FingerContactPolicy(("thumb", "middle", "ring"), {
        "thumb": ("left_thumb_MCP_FE", "left_thumb_IP"),
        "middle": ("left_middle_MCP_FE", "left_middle_PIP", "left_middle_DIP"),
        "ring": ("left_ring_MCP_FE", "left_ring_PIP", "left_ring_DIP")})
    adapter = ContactExecutionAdapter(model, policy, arm_joint_names=tuple(f"joint{i}" for i in range(1, 8)))
    builder = CoordinatedMotionReference(model, joint_names=adapter.joint_names)
    observed = ContactObservation((), dict.fromkeys(CONTACT_CHANNELS, 0.), (), (), 0.)
    result = prepare(builder, data, observed, joint_velocity_limits_rad_s=(.025,) * 7 + (.05,) * 8)
    assert len(result.task_reference_velocity_rad_s) == 15
    np.testing.assert_array_equal(np.array(result.wrist_jacobian_world)[:, 7:], np.zeros((6, 8)))
    np.testing.assert_array_equal(result.task_reference_velocity_rad_s[7:], np.zeros(8))
    assert result.rigid_transport_normal_velocity_m_s == ()
