"""V6 command limits, release permission, and per-finger force relief."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuka_sharpa_grasp.control import FINGERS, ForceReading
from kuka_sharpa_grasp.hand_grasp import HandConfig, HandGraspController
from kuka_sharpa_grasp.motion import ArmCommandLimiter
from kuka_sharpa_grasp.support import SupportConfig, TableSupportGate


def test_arm_commands_bound_velocity_and_acceleration_during_reversal():
    velocity_limits = np.array([.7, 1.1])
    acceleration_limits = np.array([2., 3.])
    limiter = ArmCommandLimiter([0., 0.], velocity_limits, acceleration_limits)
    dt = .002
    for target, steps in (([1.2, -1.4], 400), ([-.8, 1.1], 800), ([.2, -.2], 800)):
        for _ in range(steps):
            previous_position = limiter.position.copy()
            previous_velocity = limiter.velocity.copy()
            position = limiter.update(target, dt)
            velocity = (position - previous_position) / dt
            np.testing.assert_allclose(velocity, limiter.velocity, atol=1e-12)
            assert np.all(np.abs(velocity) <= velocity_limits + 1e-10)
            assert np.all(np.abs(velocity - previous_velocity) / dt
                          <= acceleration_limits + 1e-7)


def test_arm_limiter_brakes_before_a_stationary_target_from_rest():
    limiter = ArmCommandLimiter([0., 0.], [1.5, 1.5], [.8, .8])
    target = np.array([.5, -.5])
    positions = np.array([limiter.update(target, .002) for _ in range(1000)])
    assert np.all(positions[:, 0] <= target[0] + 1e-10)
    assert np.all(positions[:, 1] >= target[1] - 1e-10)
    np.testing.assert_allclose(limiter.position, target, atol=1e-6)
    np.testing.assert_allclose(limiter.velocity, 0., atol=1e-6)


def test_arm_limiter_settles_at_a_stationary_target():
    limiter = ArmCommandLimiter([-.4, .6], [.7, 1.1], [2., 3.])
    target = np.array([.2, -.15])
    for _ in range(5000):
        limiter.update(target, .002)
    np.testing.assert_allclose(limiter.position, target, atol=1e-6)
    np.testing.assert_allclose(limiter.velocity, 0., atol=1e-6)


@pytest.mark.parametrize("position,velocity,acceleration", [
    ([0., 0.], [1.], [2., 2.]),
    ([0., 0.], [1., 1.], [2.]),
    ([float("nan")], [1.], [2.]),
    ([0.], [0.], [2.]),
    ([0.], [-1.], [2.]),
    ([0.], [float("inf")], [2.]),
    ([0.], [1.], [0.]),
    ([0.], [1.], [float("nan")]),
])
def test_arm_limiter_rejects_invalid_initial_state_and_limits(position, velocity, acceleration):
    with pytest.raises(ValueError):
        ArmCommandLimiter(position, velocity, acceleration)


@pytest.mark.parametrize("target,dt", [
    ([0.], .002),
    ([0., float("nan")], .002),
    ([0., 0.], 0.),
    ([0., 0.], -.002),
    ([0., 0.], float("inf")),
    ([0., 0.], float("nan")),
])
def test_invalid_arm_update_preserves_command_state(target, dt):
    limiter = ArmCommandLimiter([0., 0.], [1., 1.], [2., 2.])
    limiter.update([1., -1.], .002)
    previous_position, previous_velocity = limiter.position.copy(), limiter.velocity.copy()
    with pytest.raises(ValueError):
        limiter.update(target, dt)
    np.testing.assert_array_equal(limiter.position, previous_position)
    np.testing.assert_array_equal(limiter.velocity, previous_velocity)


@pytest.fixture
def support_gate():
    return TableSupportGate(SupportConfig(minimum_load_n=.7 * .15 * 9.81))


def sustain_support(gate, start=0., duration=.3):
    for timestamp in start + np.arange(0., duration + .001, .002):
        sample = gate.update(float(timestamp), .15 * 9.81)
    return sample


def test_brief_support_spike_never_permits_release(support_gate):
    samples = []
    for timestamp in np.arange(0., .1, .002):
        load = 20. if .004 <= timestamp < .014 else 0.
        samples.append(support_gate.update(float(timestamp), load))
    assert not any(sample["confirmed"] for sample in samples)
    assert support_gate.support_since is None


def test_continuous_weight_support_confirms_only_after_required_duration(support_gate):
    confirmations = []
    for timestamp in np.arange(0., .3, .002):
        sample = support_gate.update(float(timestamp), .15 * 9.81)
        if sample["confirmed"]:
            confirmations.append(timestamp)
            assert sample["continuous_support_seconds"] >= .12 - 1e-12
    assert confirmations and confirmations[0] > .12
    assert support_gate.fresh_confirmation(support_gate.last_time)
    dropped = support_gate.update(support_gate.last_time + .002, 0.)
    assert not dropped["confirmed"] and support_gate.support_since is None


def test_sample_gap_resets_previously_confirmed_support(support_gate):
    assert sustain_support(support_gate)["confirmed"]
    timestamp = support_gate.last_time + .05
    sample = support_gate.update(timestamp, .15 * 9.81)
    assert not sample["confirmed"]
    assert sample["continuous_support_seconds"] == 0.
    assert sample["filtered_load_n"] == 0.
    assert not support_gate.fresh_confirmation(timestamp)
    assert sustain_support(support_gate, timestamp + .002)["confirmed"]


def test_stale_and_future_dated_confirmation_cannot_permit_release(support_gate):
    assert sustain_support(support_gate)["confirmed"]
    latest = support_gate.last_time
    assert support_gate.fresh_confirmation(latest + .015)
    assert not support_gate.fresh_confirmation(latest + .021)
    assert not support_gate.fresh_confirmation(latest - .001)


def test_invalid_support_samples_preserve_a_valid_confirmation(support_gate):
    assert sustain_support(support_gate)["confirmed"]
    latest = support_gate.last_time
    for timestamp, load in ((latest, 2.), (latest - .002, 2.),
                            (latest + .002, -1.), (latest + .002, float("nan")),
                            (float("inf"), 2.)):
        with pytest.raises(ValueError):
            support_gate.update(timestamp, load)
        assert support_gate.last_time == latest
        assert support_gate.fresh_confirmation(latest)


@pytest.fixture
def hand_model():
    return mujoco.MjModel.from_xml_string("<mujoco/>")


def contact_readings():
    return {finger: ForceReading(normal=2., contacts=1) for finger in FINGERS}


@pytest.mark.parametrize("method", ["close", "retain"])
def test_excessive_force_decreases_only_the_loaded_finger_closure(hand_model, method):
    hand = HandGraspController(hand_model, HandConfig(maximum_force_n=8., relief_rate=.12))
    hand.closure = {finger: .5 for finger in FINGERS}
    hand.engaged = set(FINGERS)
    readings = contact_readings()
    readings["index"] = ForceReading(normal=8.1, contacts=1)
    getattr(hand, method)(readings, .1)
    assert hand.closure["index"] == pytest.approx(.488)
    assert all(hand.closure[finger] == .5 for finger in FINGERS if finger != "index")
    for _ in range(100):
        getattr(hand, method)(readings, .1)
    assert hand.closure["index"] == 0.


@pytest.mark.parametrize("method", ["close", "retain"])
def test_legacy_hand_has_no_new_force_relief_by_default(hand_model, method):
    hand = HandGraspController(hand_model)
    hand.closure = {finger: .5 for finger in FINGERS}
    hand.engaged = set(FINGERS)
    readings = {finger: ForceReading(normal=20., contacts=1) for finger in FINGERS}
    getattr(hand, method)(readings, .1)
    assert hand.config.maximum_force_n is None
    assert hand.closure == {finger: .5 for finger in FINGERS}
