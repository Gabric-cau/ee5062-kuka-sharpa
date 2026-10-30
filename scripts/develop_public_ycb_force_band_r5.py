#!/usr/bin/env python3
"""Bounded can pressure-tracking experiment; frozen R4 remains unchanged."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np

import develop_public_ycb_force_band_r4 as r4


PROJECT = r4.PROJECT
PHASE_SECONDS = {**r4.PHASE_SECONDS, "CLOSE": 18.}
TRACKING_PARAMETERS = {
    "pressure_proportional_gain_rad_s_per_n": .06,
    "allocated_load_reserve_n": .4,
    "goal_filter_seconds": .15,
    "last_allocated_goal_hold_seconds": .25,
    "base_close_seconds": 8.,
    "settling_seconds_after_first_feasible_geometry": 6.,
    "maximum_close_seconds": 18.,
    "free_contact_exploration_rad": .08,
    "free_contact_exploration_rate_rad_s": .02,
    "safety_scope": "Original6N target ceiling,7N relief, physical gains/torques/rates,1.5gravity measured-wrench gate and strict12N assessment unchanged",
}


class TrackingHand(r4.WrenchAllocatedHand):
    def __init__(self, model, config, profile, reader, estimated):
        super().__init__(model, config, profile, reader, estimated)
        self.first_feasible_s = None
        self.last_feasible_goal_s = -np.inf
        self.last_feasible_goals = {}
        self.filtered_goals = {finger: 0. for finger in r4.FINGERS}

    def pressure_goal_for(self, finger, goals, dt, data):
        nominal = self.profile.get("contact_closure", {})
        if self.allocation.get("feasible"):
            if self.first_feasible_s is None:
                self.first_feasible_s = float(data.time)
            self.last_feasible_goal_s = self.last_allocation_s
            self.last_feasible_goals = dict(goals)
            selected, scope = goals, "CURRENT_GEOMETRY_ALLOCATION"
        elif (self.association.get("accepted") and
                data.time - self.last_feasible_goal_s <= TRACKING_PARAMETERS["last_allocated_goal_hold_seconds"]):
            selected, scope = self.last_feasible_goals, "RECENT_GOAL_FOR_TRACKING_ONLY_NOT_READINESS"
        else:
            selected, scope = {}, "BOUNDED_CONTACT_ACQUISITION"
        desired = float(selected.get(finger, 2.))
        if selected and desired > .35:
            desired += TRACKING_PARAMETERS["allocated_load_reserve_n"]
        desired = min(6., max(.35, desired))
        if nominal.get(finger) == 0.:
            desired = 0.
        alpha = dt / (TRACKING_PARAMETERS["goal_filter_seconds"] + dt)
        self.filtered_goals[finger] += alpha * (desired - self.filtered_goals[finger])
        self.tracking_goal_scope = scope
        return float(np.clip(self.filtered_goals[finger], 0., 6.))

    def free_contact_velocity(self, finger, previous_closure, dt):
        nominal = self.profile.get("contact_closure", {})
        if nominal.get(finger) == 0.:
            return -self.config.relief_rate
        landing_seconds = max(3., max(nominal.values(), default=.7) / self.config.close_rate)
        contact_reference = nominal.get(finger, .7) * min(1., self.elapsed / landing_seconds)
        exploration = min(TRACKING_PARAMETERS["free_contact_exploration_rad"],
            max(0., self.elapsed - landing_seconds) * TRACKING_PARAMETERS["free_contact_exploration_rate_rad_s"])
        reference = min(1., contact_reference + exploration)
        return float(np.clip(2. * (reference - previous_closure), -.2, self.config.close_rate))


def _tracking_regulator():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r4.WrenchAllocatedHand.regulate)))
    function = tree.body[0]
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "goal" for target in node.targets):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "min":
                node.value = ast.parse("self.pressure_goal_for(finger, goals, dt, data)", mode="eval").body
        if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == .03:
            node.value = TRACKING_PARAMETERS["pressure_proportional_gain_rad_s_per_n"]
        if isinstance(node, ast.If) and ast.unparse(node.test) == "pressure < 0.15":
            node.body = ast.parse("velocity = self.free_contact_velocity(finger, previous_closure, dt)").body
    function.body.extend(ast.parse("""
self.feedback_events[-1].update(
    first_feasible_geometry_timestamp_s=self.first_feasible_s,
    tracking_goal_scope=getattr(self, 'tracking_goal_scope', None),
    raw_allocation_targets_n=dict(self.allocation.get('target_pressures_n', {})),
    tracking_parameters=dict(_tracking_parameters))
""").body)
    namespace = dict(r4.WrenchAllocatedHand.regulate.__globals__)
    namespace["_tracking_parameters"] = TRACKING_PARAMETERS
    exec(compile(ast.fix_missing_locations(tree), "<r5_bounded_pressure_tracking>", "exec"), namespace)
    return namespace["regulate"]


TrackingHand.regulate = _tracking_regulator()


def controller_for(kind, directory, observations, guard, profile, mass_kg):
    original, source, _, estimated = r4.controller_for(kind, directory, observations, guard, profile, mass_kg)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    loop = next(node for node in move.body if isinstance(node, ast.For))
    loop.body.append(ast.parse("""
if state == State.CLOSE:
    elapsed_close_s = (step + 1) * model.opt.timestep
    feasible_elapsed_s = (None if hand.first_feasible_s is None else
        float(data.time) - hand.first_feasible_s)
    if elapsed_close_s >= _tracking_parameters['base_close_seconds']:
        if feasible_elapsed_s is None or feasible_elapsed_s >= _tracking_parameters['settling_seconds_after_first_feasible_geometry']:
            trace[-1]['close_termination'] = 'BOUNDED_ACQUISITION_OR_SETTLING_DEADLINE'
            break
""").body[0])
    hands = []

    def make_hand(model, config):
        hand = TrackingHand(model, config, profile, guard, estimated)
        hands.append(hand)
        return hand

    namespace = dict(original.__globals__)
    namespace.update(_phase_seconds=PHASE_SECONDS, _tracking_parameters=TRACKING_PARAMETERS,
        HandGraspController=make_hand)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_bounded_tracking_r5>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r4.snapshot(profile_path)
    path = Path(__file__).resolve()
    sources[path.relative_to(PROJECT).as_posix()] = r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r4.run)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = {"public_ycb_tactile_wrench_force_band_development_r4": "public_ycb_bounded_tracking_development_r5",
                "tactile_wrench_band_r4": "tactile_tracking_band_r5",
                "public_ycb_tactile_wrench_force_band_r4": "public_ycb_bounded_tracking_r5"}.get(node.value, node.value)
    protocol_write = next(i for i, node in enumerate(tree.body[0].body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "previous.atomic_json")
    tree.body[0].body.insert(protocol_write, ast.parse("protocol['tracking_experiment'] = dict(_tracking_parameters)").body[0])
    capture = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "capture_phase")
    pixels = next(i for i, node in enumerate(capture.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pixels" for target in node.targets))
    capture.body[pixels + 1:pixels + 1] = ast.parse("""
from OpenGL import GL
protocol.setdefault('actual_overview_gl', {})
for gl_name, gl_token in (('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)):
    gl_value = GL.glGetString(gl_token)
    protocol['actual_overview_gl'][gl_name] = None if gl_value is None else gl_value.decode('ascii', errors='replace')
""").body
    namespace = dict(r4.run.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PHASE_SECONDS=PHASE_SECONDS,
        _tracking_parameters=TRACKING_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_bounded_tracking_runner_r5>", "exec"), namespace)
    return namespace["run"]


run = _runner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind != "ycb_can" or seed < 0 for kind, seed in cases):
        parser.error("R5 tracking runs are restricted to unique nonnegative ycb_can seeds while the meat penetration fault is unresolved")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
