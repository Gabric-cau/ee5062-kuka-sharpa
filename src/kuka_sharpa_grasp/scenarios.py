"""Reproducible trial specifications for paired grasp experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrialSpec:
    """All randomized inputs for one controller trial."""

    seed: int
    object_pos: tuple[float, float, float]
    object_mass: float | None
    controller: str
    object_friction: float | None = None


class ScenarioSampler:
    """Sample deterministic XY placement perturbations around a scene pose.

    The sampler deliberately leaves z unchanged so an experiment changes the
    grasp approach, rather than silently changing table penetration or object
    height. A single spec should be reused for every controller in a pair.
    """

    def __init__(
        self,
        base_position: tuple[float, float, float],
        xy_jitter: float = 0.0,
        friction_range: tuple[float, float] | None = None,
        mass_range: tuple[float, float] | None = None,
    ) -> None:
        if not math.isfinite(xy_jitter) or xy_jitter < 0:
            raise ValueError("xy_jitter must be finite and non-negative")
        self.base_position = np.asarray(base_position, dtype=float)
        if self.base_position.shape != (3,) or not np.all(np.isfinite(self.base_position)):
            raise ValueError("base_position must contain three finite coordinates")
        self.xy_jitter = float(xy_jitter)
        if friction_range is not None and (len(friction_range) != 2 or not all(math.isfinite(x) for x in friction_range)
                                           or friction_range[0] <= 0 or friction_range[0] > friction_range[1]):
            raise ValueError("friction_range must be positive and ordered low to high")
        self.friction_range = friction_range
        if mass_range is not None and (len(mass_range) != 2 or not all(math.isfinite(x) for x in mass_range)
                                      or mass_range[0] <= 0 or mass_range[0] > mass_range[1]):
            raise ValueError("mass_range must be positive and ordered low to high")
        self.mass_range = mass_range

    def sample(self, seed: int, controller: str, object_mass: float | None = None) -> TrialSpec:
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-self.xy_jitter, self.xy_jitter, size=2)
        position = self.base_position.copy()
        position[:2] += offset
        friction = None
        if self.friction_range is not None:
            friction = float(rng.uniform(*self.friction_range))
        if self.mass_range is not None:
            object_mass = float(rng.uniform(*self.mass_range))
        return TrialSpec(
            seed=seed,
            object_pos=tuple(float(value) for value in position),
            object_mass=object_mass,
            object_friction=friction,
            controller=controller,
        )
