import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.control import ForceReading, OuterForceController, hand_synergy_targets
from kuka_sharpa_grasp.scenarios import ScenarioSampler


def test_force_controller_closes_when_force_is_low_and_releases_when_high():
    controller = OuterForceController(target_force=1.0, kp=0.01, ki=0.0, max_step=0.1)
    current = hand_synergy_targets(0.3)
    low = {finger: ForceReading(normal=0.0) for finger in current}
    high = {finger: ForceReading(normal=2.0) for finger in current}
    closed = controller.update(low, current, 0.01)
    released = controller.update(high, closed, 0.01)
    assert all(closed[name] > current[name] for name in current)
    assert all(released[name] < closed[name] for name in current)


def test_synergy_targets_are_bounded():
    values = hand_synergy_targets(1.7, thumb=-1.0)
    assert set(values) == {"thumb", "index", "middle", "ring", "pinky"}
    assert np.all([(0.0 <= value <= 1.0) for value in values.values()])


def test_scenario_sampler_is_reproducible_and_preserves_z():
    sampler = ScenarioSampler((0.70, 0.0, 0.43), xy_jitter=0.01, friction_range=(0.5, 0.8))
    first = sampler.sample(7, "fixed")
    second = sampler.sample(7, "force")
    assert first.object_pos == second.object_pos
    assert first.object_pos[2] == 0.43
    assert first.object_friction == second.object_friction
    assert 0.5 <= first.object_friction <= 0.8
    assert first.controller == "fixed"
