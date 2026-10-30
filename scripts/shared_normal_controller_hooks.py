"""Isolated allocator hooks for future experiments, without a runner or budget default."""

from __future__ import annotations

from types import FunctionType, SimpleNamespace

import develop_public_ycb_bias_budget_r17 as r17
from shared_normal_tracking_targets import allocate_tracking_targets


r13, r14 = r17.r16.r13, r17.r16.r14


def _isolated_function(original, **substitutions):
    namespace = dict(original.__globals__)
    namespace.update(substitutions)
    isolated = FunctionType(original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    isolated.__annotations__ = dict(original.__annotations__)
    isolated.__doc__ = original.__doc__
    isolated.__qualname__ = original.__qualname__
    return isolated


_preload_r13 = SimpleNamespace(**{**vars(r13), "allocate_tracking_targets": allocate_tracking_targets})


class SharedNormalBudgetHand(r17.BiasBudgetHand):
    """Only the tracking and gate-only static allocator bindings differ."""

    regulate = _isolated_function(r13.JointAdmittanceProbeHand.regulate,
        allocate_tracking_targets=allocate_tracking_targets)
    current_preload_observation = _isolated_function(r14.ConstrainedPreloadHand.current_preload_observation,
        r13=_preload_r13)


_factory = _isolated_function(r17.controller_for, BiasBudgetHand=SharedNormalBudgetHand)


def controller_for(*args, additional_bias_limit_rad, **kwargs):
    """Require an explicit inherited supported budget; do not select a protocol or clocks."""
    return _factory(*args, additional_bias_limit_rad=additional_bias_limit_rad, **kwargs)
