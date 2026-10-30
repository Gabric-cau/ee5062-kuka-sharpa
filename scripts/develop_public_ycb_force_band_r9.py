#!/usr/bin/env python3
"""Recompute current probe contacts on sample or accepted visual-pose changes."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import develop_public_ycb_force_band_r8 as r8


PROJECT = r8.PROJECT
PROBE_PARAMETERS = {**r8.PROBE_PARAMETERS,
    "freshness_repair": "A changed current tactile sample or accepted visual pose forces an immediate current-patch1g/1.5g solve; every original R8 gate must still pass. Unchanged inputs retain25Hz cadence."}


class CurrentPatchProbeHand(r8.LoadTransferProbeHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_solved_visual_pose = None
        self.probe_refresh_events = []

    def refresh_probe_solve(self, data, *, force=False):
        self.reader.read(data)
        current_sample = self.contact_sample(self.reader.raw_contacts)
        changed_contact = not r8.contact_samples_match(current_sample, self.probe_solved_sample)
        changed_visual = self.estimated.pose_fit is not self.last_solved_visual_pose
        immediate = bool(force or changed_contact or changed_visual)
        due = immediate or data.time - self.last_probe_solve_s >= r8.PROBE_PARAMETERS["solve_interval_seconds"] - 1e-9
        readings = super().refresh_probe_solve(data, force=immediate)
        if due:
            self.last_solved_visual_pose = self.estimated.pose_fit
        reasons = ((["DECISIVE_GATE"] if force else []) + (["REQUIRED_CACHE_REFRESH"] if changed_contact else []) +
            (["ACCEPTED_POSE_UPDATE"] if changed_visual else []))
        if not reasons:
            reasons = ["PERIODIC_SOLVE" if due else "MATCHING_INPUT_CACHE"]
        self.probe_refresh_events.append({"timestamp_s": float(data.time),
            "reactive_solve_reason": reasons, "explicit_decisive_force": bool(force),
            "changed_contact": bool(changed_contact), "changed_accepted_visual_pose": bool(changed_visual),
            "immediate_current_solve": immediate, "solve_executed": bool(due),
            "lp_evaluation_scope": "RECOMPUTED_CURRENT_INPUT" if due else "REUSED_MATCHING_INPUT_CACHE",
            "last_solve_timestamp_s": float(self.last_probe_solve_s),
            "current_tactile_sample": current_sample,
            "current_accepted_visual_pose": self.estimated.pose_fit,
            "current_channel_pressures_n": {finger: float(value.normal) for finger, value in readings.items()},
            "association_accepted": bool(self.probe_association.get("accepted")),
            "geometry_1_5g_feasible": bool(self.probe_geometry_15.get("feasible")),
            "capacity_1g_feasible": bool(self.probe_allocation_1g.get("measured_capacity_feasible")),
            "fixed_normal_1g_feasible": bool(self.probe_allocation_1g.get("measured_feasible")),
            "sampled_public_mesh_association": self.probe_association,
            "geometry_allocation_1_5g": self.probe_geometry_15,
            "allocation_1g": self.probe_allocation_1g,
            "scope": "Current recomputation only; no acceptance criterion changed"})
        return readings


def controller_for(*args, **kwargs):
    # The copied factory binds the new hand without changing the frozen module.
    namespace = dict(r8.controller_for.__globals__)
    namespace["LoadTransferProbeHand"] = CurrentPatchProbeHand
    factory = FunctionType(r8.controller_for.__code__, namespace,
        r8.controller_for.__name__, r8.controller_for.__defaults__, r8.controller_for.__closure__)
    return factory(*args, **kwargs)


def snapshot(profile_path):
    sources = r8.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_probe_freshness_r9.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r8.r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r8._runner)))
    replacements = {
        "public_ycb_load_transfer_probe_development_r8": "public_ycb_current_contact_probe_development_r9",
        "load_transfer_probe_r8": "current_contact_probe_r9",
        "public_ycb_load_transfer_probe_r8": "public_ycb_current_contact_probe_r9",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_current_contact_probe_runner_r9>",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
    # Append the recomputation audit beside the inherited native/probe artifacts.
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in
        ("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz")], ctx=ast.Load()))
    namespace = dict(r8._runner.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<r9_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    namespace = dict(r8.main.__globals__)
    namespace.update(run=run, __doc__=__doc__)
    entry = FunctionType(r8.main.__code__, namespace, "main", r8.main.__defaults__, r8.main.__closure__)
    return entry()


if __name__ == "__main__":
    raise SystemExit(main())
