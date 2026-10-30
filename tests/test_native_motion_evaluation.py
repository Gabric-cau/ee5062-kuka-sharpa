"""Real compiled-model sampling only; no integration or controller execution."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kuka_sharpa_grasp.native_motion_evaluation import NativeMotionEvaluator
from kuka_sharpa_grasp.strict_motion_monitor import CHANNELS


XML = '''<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
<worldbody>
  <geom name="table" type="plane" size="2 2 .01"/>
  <body name="target" pos="0 0 .048"><freejoint name="target_free"/>
    <geom name="target_geom" type="sphere" size=".05" mass=".2"/></body>
  <body name="robot_base" pos="0 0 0"><joint name="arm" type="slide" axis="0 0 1"/>
    <inertial pos="0 0 .2" mass="1" diaginertia=".1 .1 .1"/>
    <geom name="base_contact" type="sphere" pos=".4 0 .018" size=".02" mass=".1"/>
    <body name="left_thumb_distal" pos=".068 0 .048">
      <geom name="thumb" type="sphere" size=".02" mass=".1"/>
      <geom name="thumb_disabled_visual" type="sphere" size=".3" contype="0" conaffinity="0" mass=".1"/>
    </body>
    <body name="left_index_distal" pos="-.068 0 .048"><geom name="index" type="sphere" size=".02" mass=".1"/></body>
    <body name="left_middle_distal" pos="0 .068 .048"><geom name="middle" type="sphere" size=".02" mass=".1"/></body>
    <body name="left_ring_distal" pos="0 -.068 .048"><geom name="ring" type="sphere" size=".02" mass=".1"/></body>
    <body name="left_pinky_distal" pos="0 0 .116"><geom name="pinky" type="sphere" size=".02" mass=".1"/></body>
    <body name="left_palm" pos=".068 .068 .048"><geom name="palm" type="sphere" size=".047" mass=".1"/></body>
  </body>
  <body name="clutter" pos=".068 .025 .048"><freejoint/>
    <geom name="clutter_geom" type="sphere" size=".01" mass=".1"/></body>
</worldbody>
<actuator><position joint="arm" kp="100"/></actuator></mujoco>'''


@pytest.fixture
def fixture():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.time = .002
    mujoco.mj_forward(model, data)
    return model, data, NativeMotionEvaluator(model)


def sample(evaluator, data, **overrides):
    kwargs = dict(expected_timestamp_s=float(data.time), release_commanded=False, phase="TEST")
    kwargs.update(overrides)
    return evaluator.sample_after_forward(data, **kwargs)


def independent_contact_truth(model, data):
    totals = dict.fromkeys(CHANNELS, 0.)
    hand = {g: model.body(int(model.geom_bodyid[g])).name or "" for g in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[g])).name or "").startswith("left_")}
    target, table = model.geom("target_geom").id, model.geom("table").id
    hand_forces, table_forces = [], []
    target_hand = False
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact[:data.ncon]):
        first, second = int(contact.geom1), int(contact.geom2)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = abs(float(wrench[0]))
        sides = set((first, second)) & hand.keys()
        if len(sides) == 1:
            body = hand[next(iter(sides))]
            finger = next((name for name in CHANNELS[:-1] if body.startswith("left_" + name + "_")), "palm")
            totals[finger] += normal
            if target in (first, second):
                target_hand = True
                hand_forces.append(normal)
        if {first, second} == {target, table}:
            table_forces.append(normal)
    return totals, target_hand, hand_forces, table_forces


def test_native_sample_matches_independent_all_contact_channels_and_per_contact_gates(fixture):
    model, data, evaluator = fixture
    result = sample(evaluator, data)
    totals, hand_contact, hand, table = independent_contact_truth(model, data)
    assert data.ncon > 5
    assert hand and table and max(hand) > .05 and max(table) > .05
    assert result.target_hand_contact is hand_contact
    assert result.target_hand_peak_normal_n == pytest.approx(max(hand))
    assert result.target_table_peak_normal_n == pytest.approx(max(table))
    np.testing.assert_allclose(result.hand_channel_forces_n, [totals[name] for name in CHANNELS])
    assert sum(result.hand_channel_forces_n) > result.target_hand_peak_normal_n


def test_clutter_contact_contributes_to_hand_channels_but_not_target_hand_peak(fixture):
    model, data, evaluator = fixture
    thumb = model.geom("thumb").id
    clutter = model.geom("clutter_geom").id
    assert any({int(c.geom1), int(c.geom2)} == {thumb, clutter} for c in data.contact[:data.ncon])
    result = sample(evaluator, data)
    totals, _, hand, _ = independent_contact_truth(model, data)
    assert result.hand_channel_forces_n[0] == pytest.approx(totals["thumb"])
    assert result.target_hand_peak_normal_n == pytest.approx(max(hand))


def test_clearance_checks_every_collision_hand_geometry_and_excludes_visuals(fixture):
    model, data, evaluator = fixture
    result = sample(evaluator, data)
    ids = [g for g in range(model.ngeom) if (model.body(int(model.geom_bodyid[g])).name or "").startswith("left_")
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    expected = min(mujoco.mj_geomDistance(model, data, model.geom("target_geom").id, g, 1., None) for g in ids)
    assert result.hand_clearance_m == pytest.approx(expected)
    assert model.geom("thumb_disabled_visual").id not in evaluator._clearance_geoms


def test_robot_table_gate_includes_robot_ancestors_without_link_name(fixture):
    model, data, evaluator = fixture
    base, table = model.geom("base_contact").id, model.geom("table").id
    distances = [float(c.dist) for c in data.contact[:data.ncon] if {int(c.geom1), int(c.geom2)} == {base, table}]
    assert distances and min(distances) == pytest.approx(-.002)
    assert sample(evaluator, data).minimum_robot_table_distance_m <= min(distances)


def test_world_position_and_spatial_velocity_use_same_refreshed_state(fixture):
    model, data, evaluator = fixture
    joint = model.joint("target_free").id
    dof = int(model.jnt_dofadr[joint])
    data.qvel[dof:dof+6] = [.02, -.03, .04, .1, -.2, .3]
    data.qpos[int(model.jnt_qposadr[joint])] += .01
    data.time = .004
    mujoco.mj_forward(model, data)
    result = sample(evaluator, data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body("target").id, velocity, 0)
    np.testing.assert_allclose(result.target_position_world_m, data.xpos[model.body("target").id])
    assert result.target_linear_speed_m_s == pytest.approx(np.linalg.norm(velocity[3:]))
    assert result.target_angular_speed_rad_s == pytest.approx(np.linalg.norm(velocity[:3]))
    assert result.timestamp_s == .004


def test_sampling_cannot_mutate_live_state_or_return_mutable_values(fixture):
    _, data, evaluator = fixture
    before = {name: np.array(getattr(data, name), copy=True) for name in
              ("qpos", "qvel", "ctrl", "qacc", "qfrc_applied", "xfrc_applied", "xpos", "actuator_force")}
    now = float(data.time)
    result = sample(evaluator, data, release_commanded=True)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(data, name), expected)
    assert data.time == now
    assert isinstance(result.hand_channel_forces_n, tuple)
    with pytest.raises(FrozenInstanceError):
        result.release_commanded = False


@pytest.mark.parametrize("timestamp", [True, -1., float("nan"), float("inf"), .004])
def test_timestamp_must_be_finite_current_and_not_boolean(fixture, timestamp):
    _, data, evaluator = fixture
    with pytest.raises(ValueError):
        sample(evaluator, data, expected_timestamp_s=timestamp)


@pytest.mark.parametrize("kwargs", [{"release_commanded": 1}, {"phase": ""}])
def test_ambiguous_flags_are_rejected(fixture, kwargs):
    _, data, evaluator = fixture
    with pytest.raises(ValueError):
        sample(evaluator, data, **kwargs)


def test_missing_or_wrong_target_identity_is_rejected(fixture):
    model, _, _ = fixture
    with pytest.raises(ValueError):
        NativeMotionEvaluator(model, target_geom_name="missing")
    with pytest.raises(ValueError):
        NativeMotionEvaluator(model, target_geom_name="thumb")


def test_no_contacts_is_finite_unloaded_evidence(fixture):
    model, data, evaluator = fixture
    data.qpos[int(model.jnt_qposadr[model.joint("target_free").id]) + 2] = 1.
    data.qpos[int(model.jnt_qposadr[model.joint("arm").id])] = .5
    mujoco.mj_forward(model, data)
    result = sample(evaluator, data)
    assert result.target_hand_contact is False
    assert result.target_hand_peak_normal_n == result.target_table_peak_normal_n == 0.
    assert result.minimum_robot_table_distance_m is None
    assert result.hand_clearance_m > .10
