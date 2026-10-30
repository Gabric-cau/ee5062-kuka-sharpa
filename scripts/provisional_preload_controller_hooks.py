"""Isolated provisional admission policy; no runner, clocks or budget default."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap

import shared_normal_controller_hooks as hooks


r17, r16, r14 = hooks.r17, hooks.r17.r16, hooks.r14
DEFERRED_REASON = "CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE"
SOURCE_PINS = {
    "develop_public_ycb_constrained_preload_r14.py": "32f78b5c5a8d7ddb914d6d502d2c4656bafd77423523cdf80e6a791ff17360c6",
    "develop_public_ycb_bias_recovery_r16.py": "c24cbc678cd6c241f7c57d2a9f255dbc23c8bd6464234064789c6f97a081fe62",
    "develop_public_ycb_bias_budget_r17.py": "c4ce070b16c116677bcf432842885c178d3b790d884c3852d1f9bc1dfbaa1d04",
    "shared_normal_controller_hooks.py": "45af05f935a7199d1668332bca3c715e61cdccea771cae81c5e3dba63ec7453a",
}


def _checked_method(original):
    path = Path(inspect.getsourcefile(original)).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_PINS[path.name]:
        raise RuntimeError("Frozen provisional-policy method source changed: " + path.name)
    class_name, method_name = original.__qualname__.split(".")
    parsed = ast.parse(path.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    expected = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    actual = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    if ast.dump(actual) != ast.dump(expected):
        raise RuntimeError("Loaded method differs from its pinned source")
    return actual


for _module in (r17, hooks):
    _path = Path(_module.__file__).resolve(strict=True)
    if hashlib.sha256(_path.read_bytes()).hexdigest() != SOURCE_PINS[_path.name]:
        raise RuntimeError("Frozen provisional-policy dependency changed: " + _path.name)

_R14_ORIGINAL_AST = _checked_method(r14.ConstrainedPreloadHand.update_preload)
_R16_ORIGINAL_AST = _checked_method(r16.BiasRecoveryHand.update_preload)
_DEFERRED_IF = ast.parse("""if not original["sampled_fixed_normal_1g_feasible"]:
    reasons.append("CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE")
""").body[0]
_R16_PARENT_CALL = ast.parse("super().update_preload(data, admission=admission)", mode="eval").body
_PROVISIONAL_PARENT_CALL = ast.parse(
    "_provisional_r14_update_preload(self, data, admission=admission)", mode="eval").body


def _transform_r14(tree):
    if ast.dump(tree) != ast.dump(_R14_ORIGINAL_AST):
        raise RuntimeError("Frozen R14 update_preload AST changed")
    transformed = copy.deepcopy(tree)
    matches = [node for node in transformed.body if ast.dump(node) == ast.dump(_DEFERRED_IF)]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly the original pre-motion fixed-normal check")
    transformed.body.remove(matches[0])
    return ast.fix_missing_locations(transformed)


def _transform_r16(tree):
    if ast.dump(tree) != ast.dump(_R16_ORIGINAL_AST):
        raise RuntimeError("Frozen R16 update_preload AST changed")
    transformed = copy.deepcopy(tree)
    calls = [node for node in ast.walk(transformed) if isinstance(node, ast.Call)
        and ast.dump(node) == ast.dump(_R16_PARENT_CALL)]
    if len(calls) != 1 or not isinstance(transformed.body[0], ast.Assign) or transformed.body[0].value is not calls[0]:
        raise RuntimeError("Expected exactly the original R16 parent preload call")
    transformed.body[0].value = copy.deepcopy(_PROVISIONAL_PARENT_CALL)
    return ast.fix_missing_locations(transformed)


def _compile_method(tree, original, **substitutions):
    namespace = dict(original.__globals__, **substitutions)
    exec(compile(ast.Module(body=[tree], type_ignores=[]), "<provisional_preload_policy>", "exec"), namespace)
    method = namespace[tree.name]
    method.__qualname__ = original.__qualname__
    return method


_R14_PROVISIONAL_AST = _transform_r14(_R14_ORIGINAL_AST)
_provisional_r14_update_preload = _compile_method(_R14_PROVISIONAL_AST, r14.ConstrainedPreloadHand.update_preload)
_R16_PROVISIONAL_AST = _transform_r16(_R16_ORIGINAL_AST)
_provisional_r16_update_preload = _compile_method(_R16_PROVISIONAL_AST, r16.BiasRecoveryHand.update_preload,
    _provisional_r14_update_preload=_provisional_r14_update_preload)


class ProvisionalPreloadBudgetHand(hooks.SharedNormalBudgetHand):
    """Defer one pre-motion gate; preserve measured confirmation and final success."""

    def update_preload(self, data, *, admission=False):
        row = _provisional_r16_update_preload(self, data, admission=admission)
        row["provisional_preload_policy"] = {
            "format": "provisional_preload_admission_policy_v1",
            "deferred_pre_motion_reason": DEFERRED_REASON,
            "sampled_fixed_normal_1g_feasible_observed": row["original_probe_snapshot"]["sampled_fixed_normal_1g_feasible"],
            "original_probe_snapshot_unchanged": True,
            "continuous_preload_seconds": r14.PRELOAD_PARAMETERS["continuous_seconds"],
            "recovery_inhibition_unchanged": True,
            "actual_fixed_normal_1g_required_for_confirmation": True,
            "confirmation_and_held_policy_unchanged": True,
            "strict_task_success_unchanged": True,
            "additional_bias_limit_rad": self.additional_bias_limit_rad,
            "scope": "Only provisional pre-motion admission changes; readiness is neither probe confirmation nor task success.",
        }
        return row


_factory = hooks._isolated_function(r17.controller_for, BiasBudgetHand=ProvisionalPreloadBudgetHand)


def controller_for(*args, additional_bias_limit_rad, **kwargs):
    """Require an explicit supported budget; leave runner and arm clocks to a future protocol."""
    return _factory(*args, additional_bias_limit_rad=additional_bias_limit_rad, **kwargs)
