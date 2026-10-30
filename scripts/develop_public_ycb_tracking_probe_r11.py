#!/usr/bin/env python3
"""Bounded pressure tracking on the unchanged current-contact R9 probe policy."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import develop_public_ycb_force_band_r9 as r9


PROJECT = r9.PROJECT
PROBE_PARAMETERS = dict(r9.PROBE_PARAMETERS)
TRACKING_PARAMETERS = {
    "pressure_proportional_gain_rad_s_per_n": .1,
    "allocated_load_reserve_n": .4,
    "goal_filter_seconds": .15,
    "target_ceiling_n": 6.,
    "close_seconds": 8.,
    "scope": "Tracking targets only; native pressure, current LP inputs, all admission/confirmation gates and strict task assessment remain unchanged",
}


class BoundedTrackingProbeHand(r9.CurrentPatchProbeHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered_tracking_goals = {}
        self.raw_tracking_goals = {}
        self.reserved_tracking_goals = {}

    def phase_begin(self, phase, data):
        if phase == "CLOSE" and self.phase != phase:
            self.filtered_tracking_goals.clear()
            self.raw_tracking_goals.clear()
            self.reserved_tracking_goals.clear()
        super().phase_begin(phase, data)

    def pressure_goal_for(self, finger, goals, dt):
        raw = min(6., max(.35, float(goals.get(finger, 2.))))
        if self.profile.get("contact_closure", {}).get(finger) == 0.:
            self.raw_tracking_goals[finger] = 0.
            self.reserved_tracking_goals[finger] = 0.
            self.filtered_tracking_goals[finger] = 0.
            return 0.
        desired = min(6., raw + TRACKING_PARAMETERS["allocated_load_reserve_n"])
        previous = self.filtered_tracking_goals.get(finger, raw)
        alpha = dt / (TRACKING_PARAMETERS["goal_filter_seconds"] + dt)
        filtered = min(6., max(.35, previous + alpha * (desired - previous)))
        self.raw_tracking_goals[finger] = raw
        self.reserved_tracking_goals[finger] = desired
        self.filtered_tracking_goals[finger] = filtered
        return float(filtered)

    def regulate(self, readings, dt, carrying):
        external = self.reader.read(self.reader.last_data)
        _bounded_regulate(self, external, dt, carrying)
        if self.emergency_active or self.emergency_phase_hold:
            self.ready_seconds = 0.
            self.feedback_events[-1].update(ready_seconds=0.,
                safety_closure_hold=True, latched_pressure_fingers=list(self.latched_fingers),
                actual_joint_anchored_targets_rad=self.actual_joint_references.copy())


def _tracking_regulator():
    original = r9.r8.r4.WrenchAllocatedHand.regulate
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed_goal = changed_gain = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "goal" for target in node.targets)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "min"):
            node.value = ast.parse("self.pressure_goal_for(finger, goals, dt)", mode="eval").body
            changed_goal += 1
        if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == .03:
            node.value = TRACKING_PARAMETERS["pressure_proportional_gain_rad_s_per_n"]
            changed_gain += 1
    if (changed_goal, changed_gain) != (1, 1):
        raise RuntimeError("Frozen R4 regulator no longer matches the two declared tracking edits")
    tree.body[0].body.extend(ast.parse("""
self.feedback_events[-1].update(
    raw_allocation_targets_n=dict(self.allocation.get('target_pressures_n', {})),
    raw_tracking_goals_n=dict(self.raw_tracking_goals),
    reserved_tracking_goals_n=dict(self.reserved_tracking_goals),
    tracking_parameters=dict(_tracking_parameters),
    tracking_regulator_sha256=_tracking_regulator_sha256)
""").body)
    generated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    digest = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    namespace = dict(original.__globals__)
    namespace.update(_tracking_parameters=TRACKING_PARAMETERS, _tracking_regulator_sha256=digest)
    exec(compile(tree, "<r11_bounded_pressure_regulator>", "exec"), namespace)
    return namespace["regulate"], generated, digest


_bounded_regulate, TRACKING_REGULATOR_SOURCE, TRACKING_REGULATOR_SHA256 = _tracking_regulator()


def controller_for(*args, **kwargs):
    namespace = dict(r9.r8.controller_for.__globals__)
    namespace["LoadTransferProbeHand"] = BoundedTrackingProbeHand
    original = r9.r8.controller_for
    factory = FunctionType(original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__)
    return factory(*args, **kwargs)


def snapshot(profile_path):
    sources = r9.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_tracking_probe_r11.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r9.r8.r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r9.r8._runner)))
    replacements = {
        "public_ycb_load_transfer_probe_development_r8": "public_ycb_bounded_tracking_probe_development_r11",
        "load_transfer_probe_r8": "bounded_tracking_probe_r11",
        "public_ycb_load_transfer_probe_r8": "public_ycb_bounded_tracking_probe_r11",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_bounded_tracking_probe_runner_r11>",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
            if node.value.startswith("\nprotocol['latched_pressure_escape']"):
                node.value = node.value.replace(
                    "Frozen R6 hand pressure regulator and latched safety, original8s CLOSE, physical gains/contact/torque caps unchanged. Emergency actual-joint reference resets and original joint-limit relief blocks remain explicit.",
                    "R11 .1P pressure tracking with .4N reserve for every active finger, 6N target ceiling and .15s target filtering; original free-contact behavior, 8s CLOSE, physical gains/contact/torque caps and R6 emergency precedence retained.")
                node.value += "\nprotocol['pressure_tracking'] = dict(_tracking_parameters)\nprotocol['tracking_regulator_sha256'] = _tracking_regulator_sha256\nprotocol['tracking_regulator_source'] = _tracking_regulator_source\n"
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in
        ("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz")], ctx=ast.Load()))
    namespace_update = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "namespace.update")
    for name in ("_tracking_parameters", "_tracking_regulator_sha256", "_tracking_regulator_source"):
        namespace_update.keywords.append(ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load())))
    namespace = dict(r9.r8._runner.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS,
        _tracking_parameters=TRACKING_PARAMETERS, _tracking_regulator_sha256=TRACKING_REGULATOR_SHA256,
        _tracking_regulator_source=TRACKING_REGULATOR_SOURCE)
    exec(compile(ast.fix_missing_locations(tree), "<r11_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    original = r9.r8.main
    namespace = dict(original.__globals__)
    namespace.update(run=run, __doc__=__doc__)
    entry = FunctionType(original.__code__, namespace, "main", original.__defaults__, original.__closure__)
    return entry()


if __name__ == "__main__":
    raise SystemExit(main())
