#!/usr/bin/env python3
"""Run one declared upright-tool sorting fixture episode."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import TaskSpec, V7TaskExecutor


class FixtureExecutor(V7TaskExecutor):
    def __init__(self, session, *, speed=.03, dock_xy=(0., 0.), dock_clearance=.006,
                 dock_relief_force_n=8., dock_relief_rate=.8, dock_slowdown_start_n=6.):
        if not 0. < dock_relief_force_n < 12. or not np.isfinite(dock_relief_rate) or dock_relief_rate <= .4:
            raise ValueError("dock relief needs a positive setpoint below 12 N and rate above commanded closure")
        if not np.isfinite(dock_slowdown_start_n) or not 0. < dock_slowdown_start_n < dock_relief_force_n:
            raise ValueError("dock slowdown must start below the relief setpoint")
        super().__init__(session)
        self.speed, self.dock_xy, self.dock_clearance = speed, np.asarray(dock_xy), dock_clearance
        self.dock_relief_force_n = float(dock_relief_force_n)
        self.dock_relief_rate = float(dock_relief_rate)
        self.dock_slowdown_start_n = float(dock_slowdown_start_n)
        self._dock_baseline_force_n = 0.

    def _command(self, target, rotation_rad=None):
        delta = np.asarray(target) - self.session.position_goal
        maximum = self.speed * .02
        if self.current_phase in {"DOCK", "RELEASE"} and self.session.shared_control:
            tactile = self.session.tactile.read(self.session.data)
            pressure = max(reading.normal_force_n for reading in tactile.fingers.values())
            start = max(self.dock_slowdown_start_n, self._dock_baseline_force_n + .5)
            end = min(11., max(self.dock_relief_force_n, self._dock_baseline_force_n + 1.5))
            scale = (0. if pressure >= 11. else
                     .2 + .8 * float(np.clip((end - pressure) / max(.1, end - start), 0., 1.)))
            # Contacts with an already placed tool also need time to unload;
            # a small crawl prevents static pressure from blocking release.
            maximum *= scale
            if scale < 1. and np.linalg.norm(delta) > 1e-12:
                self.session.events.append({"time_s": float(self.session.data.time),
                    "event": "FIXTURE_CONTACT_MOTION_LIMIT", "phase": self.current_phase,
                    "maximum_channel_force_n": pressure, "motion_scale": scale,
                    "held_grip_baseline_force_n": self._dock_baseline_force_n})
        if np.linalg.norm(delta) > maximum:
            target = self.session.position_goal + delta / np.linalg.norm(delta) * maximum
        return super()._command(target, rotation_rad)

    def _phase(self, name, target, closure, **kwargs):
        target = np.asarray(target, dtype=float)
        if name in {"TRANSFER", "DOCK", "RELEASE", "VERIFY"}:
            target = target + np.array([*self.dock_xy, 0.])
        if name in {"DOCK", "RELEASE", "VERIFY"}:
            target = target + np.array([0., 0., self.dock_clearance])
        original = self.session.hand.config
        if name == "DOCK":
            tactile = self.session.tactile.read(self.session.data)
            self._dock_baseline_force_n = max(reading.normal_force_n for reading in tactile.fingers.values())
        # A tool supported by the rack must unload before the 12 N safety
        # boundary. Relief must outrun the session's .4/s target tracking.
        if name in {"DOCK", "RELEASE"}:
            self.session.hand.config = replace(original, maximum_force_n=self.dock_relief_force_n,
                                               relief_rate=self.dock_relief_rate)
        try:
            return super()._phase(name, target, closure, **kwargs)
        finally:
            self.session.hand.config = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=("tool_screwdriver", "tool_wrench"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = upright_tool_fixture(args.seed)
    scene = build_v7_workbench_scene(args.output / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config)
    session.hand.config = replace(session.hand.config, retention_minimum_force_n=3.0)
    dock_xy = (0.010, -0.012) if args.object == "tool_screwdriver" else (-.030, .010)
    offset = (-.065, .037, .210) if args.object == "tool_screwdriver" else (-.065, .037, .233)
    closure = .65 if args.object == "tool_screwdriver" else .75
    spec = TaskSpec(args.object, slot_id=config.registry.by_id(args.object).dock_slot,
                    grasp_offset_m=offset, approach_axis_world=(0., 0., -1.),
                    grasp_closure=closure, lift_height_m=.10, phase_timeout_s=12.,
                    hold_seconds=.10, tactile_force_min_n=.15, release_verify_seconds=3.,
                    dock_tolerance_m=.025)
    dock_clearance = .014 if args.object == "tool_screwdriver" else .006
    result = FixtureExecutor(session, dock_xy=dock_xy, dock_clearance=dock_clearance).run(spec)
    channel_peaks = {name: max(float(row["tactile"]["fingers"][name]["normal_force_n"])
                               for row in result.trace)
                     for name in result.trace[0]["tactile"]["fingers"]}
    force_safe = max(channel_peaks.values()) <= 12.
    success = result.success and force_safe
    payload = {"layout": config.version, "task_spec": asdict(spec),
               "success": success, "task_success": result.success, "force_safe": force_safe,
               "failure_reason": result.failure_reason if force_safe else "CHANNEL_FORCE_LIMIT_EXCEEDED",
               "channel_peak_force_n": channel_peaks,
               "dock_force_control": {"relief_setpoint_n": 8., "relief_rate_per_s": .8,
                                      "motion_slowdown_start_n": 6., "held_grip_baseline_margin_n": .5,
                                      "minimum_motion_scale": .2, "motion_pause_force_n": 11.,
                                      "channel_safety_limit_n": 12.},
               "result": result.as_dict(), "scene_manifest": config.manifest(),
               "evidence_boundary": "fixture-supported tool sorting; original flat layout remains separate"}
    (args.output / "task_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"object": args.object, "seed": args.seed, "success": success,
                      "task_success": result.success, "force_safe": force_safe,
                      "peak_channel_force_n": max(channel_peaks.values()),
                      "failure_reason": payload["failure_reason"],
                      "dock_error_m": result.metrics["dock_error_m"]}))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
