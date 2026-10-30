#!/usr/bin/env python3
"""Can-only catalogue acquisition and bounded tracking under frozen R6 safety."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np

import develop_public_ycb_force_band_r4 as r4
import develop_public_ycb_force_band_r5 as r5
import develop_public_ycb_force_band_r6 as r6


PROJECT = r6.PROJECT
PHASE_SECONDS = dict(r5.PHASE_SECONDS)
TRACKING_PARAMETERS = {**r5.TRACKING_PARAMETERS,
    "acquisition_prior": "Frozen public-catalogue profile force_targets_n, capped6N; tracking only, never measured-wrench readiness proof"}


class CatalogueTrackingHand(r6.ActualJointReliefHand):
    def __init__(self, model, config, profile, reader, estimated):
        super().__init__(model, config, profile, reader, estimated)
        self.first_feasible_s = None
        self.last_feasible_goal_s = -np.inf
        self.last_feasible_goals = {}
        self.filtered_goals = {finger: 0. for finger in r4.FINGERS}

    def phase_begin(self, phase, data):
        if phase == "CLOSE" and self.phase != phase:
            self.first_feasible_s = None
            self.last_feasible_goal_s = -np.inf
            self.last_feasible_goals = {}
            self.filtered_goals = {finger: 0. for finger in r4.FINGERS}
        super().phase_begin(phase, data)

    def pressure_goal_for(self, finger, goals, dt, data):
        if self.allocation.get("feasible"):
            if self.first_feasible_s is None:
                self.first_feasible_s = float(data.time)
            self.last_feasible_goal_s = self.last_allocation_s
            self.last_feasible_goals = dict(goals)
            selected, scope = goals, "CURRENT_GEOMETRY_ALLOCATION"
        elif (self.association.get("accepted") and data.time - self.last_feasible_goal_s <=
                TRACKING_PARAMETERS["last_allocated_goal_hold_seconds"]):
            selected, scope = self.last_feasible_goals, "RECENT_GOAL_FOR_TRACKING_ONLY_NOT_READINESS"
        else:
            selected, scope = self.profile["force_targets_n"], "PUBLIC_CATALOGUE_ACQUISITION_PRIOR_ONLY"
        desired = float(selected.get(finger, 0.))
        if scope != "PUBLIC_CATALOGUE_ACQUISITION_PRIOR_ONLY" and desired > .35:
            desired += TRACKING_PARAMETERS["allocated_load_reserve_n"]
        desired = float(np.clip(desired, 0., 6.))
        if self.profile.get("contact_closure", {}).get(finger) == 0.:
            desired = 0.
        alpha = dt / (TRACKING_PARAMETERS["goal_filter_seconds"] + dt)
        self.filtered_goals[finger] += alpha * (desired - self.filtered_goals[finger])
        self.tracking_goal_scope = scope
        return float(np.clip(self.filtered_goals[finger], 0., 6.))

    free_contact_velocity = r5.TrackingHand.free_contact_velocity

    def regulate(self, readings, dt, carrying):
        external_readings = self.reader.read(self.reader.last_data)
        r5.TrackingHand.regulate(self, external_readings, dt, carrying)
        if self.emergency_active or self.emergency_phase_hold:
            self.ready_seconds = 0.
            self.feedback_events[-1].update(ready_seconds=0., safety_closure_hold=True,
                latched_pressure_fingers=list(self.latched_fingers),
                actual_joint_anchored_targets_rad=self.actual_joint_references.copy())
        self.feedback_events[-1].update(catalogue_acquisition_targets_n=dict(self.profile["force_targets_n"]),
            tracking_parameters=dict(TRACKING_PARAMETERS))


def controller_for(kind, directory, observations, guard, profile, mass_kg):
    if kind != "ycb_can":
        raise ValueError("R7 is restricted to public ycb_can")
    priors = profile.get("force_targets_n", {})
    if set(r4.FINGERS) - priors.keys() or any(not np.isfinite(value) or value < 0. or value > 6.
            for value in priors.values()):
        raise ValueError("Explicit finite catalogue force targets in [0,6] required for all fingers")
    original, source, _, estimated = r6.controller_for(kind, directory, observations, guard, profile, mass_kg)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    loop = next(node for node in move.body if isinstance(node, ast.For))
    loop.body.append(ast.parse("""
if state == State.CLOSE:
    elapsed_close_s = (step + 1) * model.opt.timestep
    feasible_elapsed_s = None if hand.first_feasible_s is None else float(data.time) - hand.first_feasible_s
    if elapsed_close_s >= _tracking_parameters['base_close_seconds']:
        if feasible_elapsed_s is None or feasible_elapsed_s >= _tracking_parameters['settling_seconds_after_first_feasible_geometry']:
            trace[-1]['close_termination'] = 'BOUNDED_ACQUISITION_OR_SETTLING_DEADLINE'
            break
""").body[0])
    hands = []

    def make_hand(model, config):
        hand = CatalogueTrackingHand(model, config, profile, guard, estimated)
        hands.append(hand)
        guard.robot_hand = hand
        return hand

    namespace = dict(original.__globals__)
    namespace.update(HandGraspController=make_hand, _phase_seconds=PHASE_SECONDS,
        _tracking_parameters=TRACKING_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_catalogue_tracking_r7>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r6.snapshot(profile_path)
    for path in (Path(r5.__file__).resolve(), Path(__file__).resolve()):
        sources[path.relative_to(PROJECT).as_posix()] = r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r4.run)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = {"public_ycb_tactile_wrench_force_band_development_r4": "public_ycb_catalogue_tracking_development_r7",
                "tactile_wrench_band_r4": "catalogue_tracking_r7",
                "public_ycb_tactile_wrench_force_band_r4": "public_ycb_catalogue_tracking_r7"}.get(node.value, node.value)
    protocol_write = next(i for i, node in enumerate(tree.body[0].body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "previous.atomic_json")
    tree.body[0].body[protocol_write:protocol_write] = ast.parse("""
protocol['latched_pressure_escape'] = dict(_relief_parameters)
protocol['tracking_experiment'] = dict(_tracking_parameters)
protocol['force_control'] = 'R6 actual-joint anchored latched safety has precedence over R5 bounded tracking; catalogue force_targets_n are acquisition goals only, capped6N. Current measured-wrench feasibility, native pressure limits, physical gains/torques/contact and strict assessment unchanged. Emergency position-reference resets can exceed ordinary closure increments; original joint limits may block relief, explicitly logged.'
protocol['guard'] = 'Frozen R6 per-finger first-overload tactile-patch latch, continuous.2s all-channel<=1N plus actual endpoint; no downward escape for unconfirmed/table-supported payload; no one-sample clearance.'
""").body
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
    artifacts = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) and ast.unparse(target) == "row['artifacts']" for target in node.targets))
    artifacts.value.keys.append(ast.Constant(value="pressure_relief"))
    artifacts.value.values.append(ast.parse("previous.write_gzip(directory / 'pressure_relief.json.gz', [event for hand in hands for event in hand.relief_events])", mode="eval").body)
    namespace = dict(r4.run.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PHASE_SECONDS=PHASE_SECONDS,
        TactilePressureGuard=r6.LatchedPressureGuard, _relief_parameters=r6.RELIEF_PARAMETERS,
        _tracking_parameters=TRACKING_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_catalogue_tracking_runner_r7>", "exec"), namespace)
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
        parser.error("R7 requires unique nonnegative ycb_can seeds")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
