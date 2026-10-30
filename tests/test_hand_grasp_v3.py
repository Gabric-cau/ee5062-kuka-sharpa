"""Anatomical posture and contact-feedback regressions, including real contacts."""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuka_sharpa_grasp.control import FINGERS, ForceReading, ContactForceReader
from kuka_sharpa_grasp.hand_grasp import (
    CALIBRATED_CAN_CLOSURE, HandConfig, HandGraspController, HandContactReader,
)
from kuka_sharpa_grasp.pick_place import Scenario, reset_task, RobotGravityCompensation
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("typed_hand") / "scene.xml"
    return mujoco.MjModel.from_xml_path(str(build_pick_place_scene(path)))


def readings(normal=0., tangential=0.):
    return {finger: ForceReading(normal, tangential, int(normal > 0.)) for finger in FINGERS}


def test_neutral_open_posture_preserves_thumb_opposition(model):
    hand = HandGraspController(model)
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    hand.write(data)
    np.testing.assert_array_equal(data.qpos, before)
    for name, angle in hand.angles().items():
        assert angle == pytest.approx(1. if name == "left_thumb_CMC_FE" else 0.)
        assert data.ctrl[hand.actuators[name]] == pytest.approx(angle)


def test_asymmetric_closure_does_not_sweep_abduction(model):
    hand = HandGraspController(model)
    contact = readings()
    contact["thumb"] = ForceReading(20., 8., 2)
    contact["ring"] = ForceReading(3., 1., 1)
    for _ in range(100):
        hand.close(contact, .01)
    angles = hand.angles()
    assert hand.closure["thumb"] == 0. and hand.closure["ring"] == 0.
    assert hand.closure["index"] > 0. and hand.closure["middle"] > 0.
    assert angles["left_thumb_CMC_FE"] == pytest.approx(1.)
    assert all(angle == 0. for name, angle in angles.items() if name.endswith("_AA"))
    assert all(model.jnt_range[model.actuator_trnid[hand.actuators[name], 0], 0]
               <= angle <= model.jnt_range[model.actuator_trnid[hand.actuators[name], 0], 1]
               for name, angle in angles.items())


@pytest.mark.parametrize("dt", [.002, .01, .05])
def test_closure_rate_bound_and_contact_stop(model, dt):
    hand = HandGraspController(model)
    hand.closure = dict(zip(FINGERS, [.10, .20, .30, .40, .999]))
    contact = readings()
    contact["index"] = ForceReading(2., .3, 1)
    contact["middle"] = ForceReading(6., 2., 2)
    before = hand.closure.copy()
    hand.close(contact, dt)
    for finger in FINGERS:
        delta = hand.closure[finger] - before[finger]
        assert 0. <= delta <= hand.config.close_rate * dt + 1e-12
        assert hand.closure[finger] <= 1.
    assert hand.closure["index"] == before["index"]
    assert hand.closure["middle"] == before["middle"]


def test_retention_is_one_way_bounded_and_does_not_load_idle_fingers(model):
    hand = HandGraspController(model)
    hand.closure = dict(zip(FINGERS, [.43, .41, .42, .48, .90]))
    hand.engaged = {"thumb", "index", "middle"}
    contact = readings()
    contact["thumb"] = ForceReading(.5, .4, 1)
    contact["index"] = ForceReading(4.1, 8., 2)
    contact["middle"] = ForceReading(2., 1.6, 1)
    baseline = hand.closure.copy()
    for step in range(300):
        dt = .002 if step % 2 else .05
        before = hand.closure.copy()
        hand.retain(contact, dt)
        for finger in FINGERS:
            delta = hand.closure[finger] - before[finger]
            assert 0. <= delta <= hand.config.retention_rate * dt + 1e-12
            assert hand.closure[finger] <= baseline[finger] + hand.config.retention_extra + 1e-12
    assert hand.closure["thumb"] > baseline["thumb"]
    assert hand.closure["middle"] > baseline["middle"]
    assert hand.closure["index"] == baseline["index"]
    assert hand.closure["ring"] == baseline["ring"]
    assert hand.closure["pinky"] == baseline["pinky"]
    pressure = readings(8., 16.)
    before = hand.closure.copy()
    hand.retain(pressure, .05)
    assert hand.closure == before


def test_real_proximal_contact_is_not_equivalent_to_fingertip_contact(model):
    data = mujoco.MjData(model)
    robot, kin, _ = reset_task(model, data, Scenario(0))
    # Reset fixture uses a calibrated robot pose. The target stays a free body;
    # after this setup all motion and contacts are produced by physics.
    grasp = kin.solve(np.array([.0021472421471974, .0153495364083377, .05]))
    assert grasp is not None
    data.qpos[:7] = grasp
    robot._write_arm_ctrl(tuple(grasp))
    hand = HandGraspController(model, HandConfig(contact_threshold_n=3.))
    hand.initialize(data)
    mujoco.mj_forward(model, data)
    gravity = RobotGravityCompensation(model)
    segment = HandContactReader(model)
    fingertip = ContactForceReader(model)
    for _ in range(round(.4 / model.opt.timestep)):
        gravity.apply(data)
        mujoco.mj_step(model, data)
    extra_normal = 0.
    extra_contacts = 0
    actual_proximal_contacts = 0
    for _ in range(round(3. / model.opt.timestep)):
        hand.close(segment.read(data), model.opt.timestep)
        hand.write(data)
        gravity.apply(data)
        mujoco.mj_step(model, data)
        all_read = segment.total(segment.read(data))
        tip_read = fingertip.total(fingertip.read(data))
        assert all_read.normal + 1e-10 >= tip_read.normal
        assert all_read.contacts >= tip_read.contacts
        extra_normal = max(extra_normal, all_read.normal - tip_read.normal)
        extra_contacts = max(extra_contacts, all_read.contacts - tip_read.contacts)
        for contact in data.contact[:data.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            if segment.target in pair:
                other = pair[1] if pair[0] == segment.target else pair[0]
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                actual_proximal_contacts += int(name == "left_thumb_PP_fit")
    assert actual_proximal_contacts > 0
    assert extra_contacts > 0
    assert extra_normal > .5


def test_calibrated_fixed_profile_matches_development_spec_and_is_immutable():
    path = Path(__file__).resolve().parents[1] / 'config/fixed_can_profile.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    assert dict(CALIBRATED_CAN_CLOSURE) == spec['closure_by_finger']
    assert spec['provenance']['calibration_seeds'] == list(range(10))
    assert set(spec['provenance']['trace_sha256']) == {str(seed) for seed in range(10)}
    assert all(len(value) == 64 for value in spec['provenance']['trace_sha256'].values())
    with pytest.raises(TypeError):
        CALIBRATED_CAN_CLOSURE['thumb'] = .9


@pytest.mark.parametrize('dt', [.002, .05])
def test_fixed_targets_are_rate_bounded_in_both_directions(model, dt):
    hand = HandGraspController(model)
    hand.closure = dict(zip(FINGERS, [.9, .1, .4008, .95, .3]))
    for _ in range(round(3. / dt)):
        before = hand.closure.copy()
        hand.approach_fixed_targets(CALIBRATED_CAN_CLOSURE, dt)
        for finger in FINGERS:
            goal = CALIBRATED_CAN_CLOSURE[finger]
            current = hand.closure[finger]
            assert abs(current - before[finger]) <= hand.config.close_rate * dt + 1e-12
            assert min(before[finger], goal) <= current <= max(before[finger], goal)
    assert hand.closure == dict(CALIBRATED_CAN_CLOSURE)
    assert hand.opposition == 1. and not hand.engaged and hand.slip_adjustments == 0
    before = hand.closure.copy()
    hand.approach_fixed_targets({finger: 0. for finger in FINGERS}, 0.)
    assert hand.closure == before


def test_invalid_fixed_targets_fail_before_any_finger_changes(model):
    hand = HandGraspController(model)
    hand.closure = dict(zip(FINGERS, [.1, .2, .3, .4, .5]))
    before = hand.closure.copy()
    invalid = [
        ({finger: .5 for finger in FINGERS if finger != 'pinky'}, .01),
        ({**CALIBRATED_CAN_CLOSURE, 'palm': .5}, .01),
        ({**CALIBRATED_CAN_CLOSURE, 'pinky': float('nan')}, .01),
        ({**CALIBRATED_CAN_CLOSURE, 'pinky': float('inf')}, .01),
        ({**CALIBRATED_CAN_CLOSURE, 'pinky': 1.01}, .01),
        ({**CALIBRATED_CAN_CLOSURE, 'thumb': -.01}, .01),
        (CALIBRATED_CAN_CLOSURE, -.01),
        (CALIBRATED_CAN_CLOSURE, float('inf')),
    ]
    for targets, dt in invalid:
        with pytest.raises(ValueError):
            hand.approach_fixed_targets(targets, dt)
        assert hand.closure == before
