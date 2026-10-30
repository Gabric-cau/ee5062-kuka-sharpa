"""Force-hold tracking and physically signed table load measurements."""

from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kuka_sharpa_grasp.control import FINGERS, ForceReading
from kuka_sharpa_grasp.hand_grasp import HandConfig, HandGraspController


@pytest.fixture
def empty_model():
    return mujoco.MjModel.from_xml_string('<mujoco/>')


def readings(normal=5., tangential=0.):
    return {finger: ForceReading(normal=normal, tangential=tangential, contacts=1) for finger in FINGERS}


def configured_hand(model):
    hand = HandGraspController(model, HandConfig(
        retention_minimum_force_n=5., retention_rate=.06,
        retention_extra=.1, maximum_force_n=12., relief_rate=.12,
    ))
    hand.closure = {finger: .4 for finger in FINGERS}
    hand.engaged = set(FINGERS)
    return hand


def test_normal_force_hold_closes_a_weak_finger_until_extra_travel_is_exhausted(empty_model):
    hand = configured_hand(empty_model)
    values = readings()
    values['index'] = ForceReading(normal=2., contacts=1)
    hand.retain(values, .1)
    assert hand.closure['index'] == pytest.approx(.406)
    assert all(hand.closure[finger] == .4 for finger in FINGERS if finger != 'index')
    for _ in range(1000):
        hand.retain(values, .1)
    assert hand.closure['index'] == pytest.approx(.5)
    assert hand.hold_reference['index'] == .4
    values['index'].normal = 5.1
    previous = hand.closure.copy()
    hand.retain(values, .1)
    assert hand.closure == previous


def test_force_hold_relief_has_priority_over_retention_and_can_close_again(empty_model):
    hand = configured_hand(empty_model)
    values = readings()
    values['index'].normal = 13.
    hand.retain(values, .1)
    assert hand.closure['index'] == pytest.approx(.388)
    values['index'].normal = 2.
    hand.retain(values, .1)
    assert hand.closure['index'] == pytest.approx(.394)
    values['index'].normal = 13.
    for _ in range(100):
        hand.retain(values, .1)
    assert hand.closure['index'] == 0.


def test_force_hold_does_not_close_a_finger_without_established_contact(empty_model):
    hand = configured_hand(empty_model)
    hand.engaged.remove('index')
    hand.retain(readings(normal=2.), .1)
    assert hand.closure['index'] == .4
    assert hand.closure['middle'] > .4


def test_weak_contact_retention_never_reopens_a_finger_above_its_reference(empty_model):
    hand = configured_hand(empty_model)
    hand.hold_reference = {finger: .1 for finger in FINGERS}
    hand.closure['index'] = .65
    hand.retain(readings(normal=.5), .1)
    assert hand.closure['index'] == .65


def test_legacy_retention_behavior_is_unchanged(empty_model):
    hand = HandGraspController(empty_model)
    hand.closure = {finger: .4 for finger in FINGERS}
    hand.engaged = set(FINGERS)
    values = readings(normal=2.)
    hand.retain(values, .1)
    assert hand.config.retention_minimum_force_n is None
    assert hand.closure == {finger: .4 for finger in FINGERS}
    values['index'].normal = .5
    hand.retain(values, .1)
    assert hand.closure['index'] == pytest.approx(.403)


@pytest.mark.parametrize('kwargs', (
    {'retention_minimum_force_n': float('nan')},
    {'retention_minimum_force_n': 0.},
    {'retention_minimum_force_n': -1.},
    {'retention_minimum_force_n': 13., 'maximum_force_n': 12.},
    {'retention_rate': -1.}, {'relief_rate': float('inf')},
))
def test_invalid_hand_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HandConfig(**kwargs)


def test_zero_rates_and_zero_extra_travel_are_valid(empty_model):
    hand = HandGraspController(empty_model, HandConfig(
        close_rate=0., retention_rate=0., relief_rate=0., retention_extra=0.,
        retention_minimum_force_n=5., maximum_force_n=12.,
    ))
    hand.engaged = set(FINGERS)
    hand.closure = {finger: .4 for finger in FINGERS}
    hand.retain(readings(normal=2.), .1)
    assert hand.closure == {finger: .4 for finger in FINGERS}


@pytest.mark.parametrize('closure', (0., .25, 1.))
def test_thumb_synergy_ratios_are_explicit_and_legacy_angles_remain_identical(empty_model, closure):
    legacy = HandGraspController(empty_model)
    tuned = HandGraspController(empty_model, HandConfig(thumb_mcp_flexion_ratio=.4, thumb_ip_flexion_ratio=1.2))
    legacy.closure['thumb'] = tuned.closure['thumb'] = closure
    old, new = legacy.angles(), tuned.angles()
    assert old['left_thumb_MCP_FE'] == old['left_thumb_IP'] == pytest.approx(.8 * closure)
    assert new['left_thumb_MCP_FE'] == pytest.approx(.4 * closure)
    assert new['left_thumb_IP'] == pytest.approx(1.2 * closure)
    assert new['left_thumb_CMC_AA'] == old['left_thumb_CMC_AA'] == 0.
    assert new['left_thumb_CMC_FE'] == old['left_thumb_CMC_FE']
    assert all(old[name] == new[name] for name in old if name not in ('left_thumb_MCP_FE', 'left_thumb_IP'))


@pytest.mark.parametrize('kwargs', ({'thumb_mcp_flexion_ratio': -.1}, {'thumb_ip_flexion_ratio': float('nan')},
                                   {'thumb_mcp_flexion_ratio': float('inf')}))
def test_thumb_ratios_must_be_finite_and_nonnegative(kwargs):
    with pytest.raises(ValueError):
        HandConfig(**kwargs)


@pytest.mark.parametrize('field,joint,limit', (('thumb_mcp_flexion_ratio', 'left_thumb_MCP_FE', 1.3963),
                                             ('thumb_ip_flexion_ratio', 'left_thumb_IP', 1.7453)))
def test_thumb_full_closure_target_obeys_actual_model_range(field, joint, limit):
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody><body><joint name="{joint}" range="0 {limit}"/>'
        f'<geom type="sphere" size=".01"/></body></worldbody>'
        f'<actuator><position name="{joint}" joint="{joint}" kp="1"/></actuator></mujoco>')
    hand = HandGraspController(model, HandConfig(**{field: limit}))
    hand.closure['thumb'] = 1.
    assert hand.angles()[joint] == limit
    with pytest.raises(ValueError, match='joint limit'):
        HandGraspController(model, HandConfig(**{field: limit + .001}))


@pytest.fixture
def table_model():
    return mujoco.MjModel.from_xml_string('''
        <mujoco>
          <option timestep="0.002" integrator="implicitfast"/>
          <worldbody>
            <geom name="table" type="plane" size="1 1 .1"/>
            <body name="target" pos="0 0 .1">
              <freejoint/>
              <geom name="target_geom" type="box" size=".04 .04 .04" mass=".349"/>
            </body>
          </worldbody>
        </mujoco>''')


def synthetic_load(model, monkeypatch, contact_specs):
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    contacts, forces = [], []
    for order, frame, force in contact_specs:
        geom1, geom2 = (table, target) if order == 'target_second' else (target, table)
        contacts.append(SimpleNamespace(geom1=geom1, geom2=geom2, frame=np.asarray(frame)))
        forces.append(force)

    def contact_force(model, data, index, wrench):
        wrench[:] = 0.
        wrench[:3] = forces[index]

    monkeypatch.setattr(mujoco, 'mj_contactForce', contact_force)
    data = SimpleNamespace(ncon=len(contacts), contact=contacts)
    return HandGraspController(model).table_support(data)


@pytest.mark.parametrize('order,normal_z', (('target_second', 1.), ('target_first', -1.)))
def test_vertical_support_sign_is_correct_for_both_geom_orders(table_model, monkeypatch, order, normal_z):
    frame = [[0., 0., normal_z], [1., 0., 0.], [0., normal_z, 0.]]
    assert synthetic_load(table_model, monkeypatch, [(order, frame, [3.42, 0., 0.])]) == pytest.approx(3.42)


def test_horizontal_collision_is_not_table_support_but_vertical_friction_is(table_model, monkeypatch):
    frame = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    assert synthetic_load(table_model, monkeypatch, [('target_second', frame, [100., 0., 0.])]) == 0.
    assert synthetic_load(table_model, monkeypatch, [('target_second', frame, [100., 3., 0.])]) == pytest.approx(3.)


def test_table_load_sums_signed_vertical_components_before_clamping(table_model, monkeypatch):
    upward = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    downward = [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]]
    assert synthetic_load(table_model, monkeypatch, [
        ('target_second', upward, [5., 0., 0.]),
        ('target_second', downward, [2., 0., 0.]),
    ]) == pytest.approx(3.)
    assert synthetic_load(table_model, monkeypatch, [('target_second', downward, [2., 0., 0.])]) == 0.


def test_actual_settled_box_load_matches_its_weight(table_model):
    data = mujoco.MjData(table_model)
    for _ in range(1000):
        mujoco.mj_step(table_model, data)
    load = HandGraspController(table_model).table_support(data)
    assert data.ncon > 0
    assert load == pytest.approx(.349 * 9.81, rel=.002)
