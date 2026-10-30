#!/usr/bin/env python3
"""Run a reproducible task-level obstacle-avoidance planning gate.

This gate evaluates the transfer segment of the upright screwdriver fixture
with a declared carried-tool proxy.  The straight joint interpolation is
deliberately blocked by a physical MuJoCo box.  OMPL RRT-Connect must return
an exact, densely post-validated path with both endpoint states collision-free.
The gate is a planning/geometry acceptance experiment; it does not claim a
completed physical pick-and-dock episode.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_motion import CollisionChecker, HeldToolProxy, RRTConnectPlanner
from kuka_sharpa_grasp.v7_scene import WorkbenchObstacle, build_v7_workbench_scene


# Calibrated transfer endpoints from the upright screwdriver fixture.  They
# are arm joint states only; no object qpos is used to steer planning.
START_Q = np.array((0.063060406, 0.30133993, -0.032897671, -1.69411781,
                    -0.043588108, 0.360931714, 0.07006609), dtype=float)
GOAL_Q = np.array((-0.589702513, 0.33312274, 0.227706963, -1.581354871,
                   0.415183875, 0.53664603, -0.683562746), dtype=float)

# This box lies on the nominal direct transfer, while leaving both calibrated
# endpoints free.  Its dimensions and source are part of the experiment
# manifest, not inferred from a live object body.
TRANSFER_BARRIER = WorkbenchObstacle(
    "transfer_barrier", (0.613, -0.043, 0.707), (0.008, 0.008, 0.008),
    (0.80, 0.15, 0.10, 1.0),
    "Synthetic task-level barrier for planner acceptance; dimensions are engineering assumptions.",
)


def _source_hashes() -> dict[str, str]:
    paths = [PROJECT / "src" / "kuka_sharpa_grasp" / "v7_motion.py",
             PROJECT / "src" / "kuka_sharpa_grasp" / "v7_scene.py",
             PROJECT / "src" / "kuka_sharpa_grasp" / "v7_fixture.py",
             Path(__file__).resolve()]
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _straight_path(start: np.ndarray, goal: np.ndarray, states: int = 301) -> tuple[np.ndarray, ...]:
    return tuple(start * (1.0 - t) + goal * t for t in np.linspace(0.0, 1.0, states))


def _path_length(path: tuple[np.ndarray, ...]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(second - first)
                     for first, second in zip(path, path[1:])))


def run(output: Path, *, seed: int = 0, timebudget: float = 3.0) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    base = upright_tool_fixture(seed)
    config = replace(base, obstacles=base.obstacles + (TRANSFER_BARRIER,),
                     version="v7_obstacle_gate_r1")
    scene_path = build_v7_workbench_scene(output / "scene.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    # The proxy is a conservative screwdriver body carried at the wrist.  The
    # old map pose is excluded by the collision checker once this is active.
    carried_tool = HeldToolProxy((0.014, 0.014, 0.047), object_id="tool_screwdriver")
    hand_qpos = {
        "left_thumb_CMC_FE": 1.08, "left_thumb_MCP_FE": 0.15,
        "left_thumb_IP": 0.51, "left_index_MCP_FE": 0.57,
        "left_index_PIP": 0.67, "left_index_DIP": 0.43,
        "left_middle_MCP_FE": 0.65, "left_middle_PIP": 0.91,
        "left_middle_DIP": 0.52, "left_ring_MCP_FE": 0.65,
        "left_ring_PIP": 0.90, "left_ring_DIP": 0.52,
        "left_pinky_MCP_FE": 0.65, "left_pinky_PIP": 0.91,
        "left_pinky_DIP": 0.52,
    }
    checker = CollisionChecker(model, config, safety_margin=0.004)
    planner = RRTConnectPlanner(checker)
    direct = _straight_path(START_Q, GOAL_Q)
    direct_reports = tuple(checker.check(q, hand_qpos, "tool_screwdriver",
                                         approach=False, heldtool=carried_tool)
                           for q in direct)
    start_report = checker.check(START_Q, hand_qpos, "tool_screwdriver",
                                approach=False, heldtool=carried_tool)
    goal_report = checker.check(GOAL_Q, hand_qpos, "tool_screwdriver",
                               approach=False, heldtool=carried_tool)
    planned = planner.plan(START_Q, GOAL_Q, hand_qpos, "tool_screwdriver",
                           timebudget=timebudget, approach=False,
                           heldtool=carried_tool,
                           scene_snapshot={"barrier": TRANSFER_BARRIER.obstacle_id,
                                           "seed": int(seed)})
    payload = {
        "format": "kuka-sharpa-v7-obstacle-gate-r1",
        "scene": str(scene_path),
        "scene_version": config.version,
        "physics_backend": "MuJoCo",
        "planner": "OMPL_RRTConnect",
        "environment_scope": planned.request_scope,
        "target": "tool_screwdriver",
        "carried_tool_proxy": carried_tool.__dict__,
        "obstacle": config.manifest()["obstacles"][-1],
        "start_q": START_Q.tolist(),
        "goal_q": GOAL_Q.tolist(),
        "start_clearance_m": float(start_report.clearance),
        "goal_clearance_m": float(goal_report.clearance),
        "straight_path": {
            "states": len(direct),
            "collision_states": int(sum(not report.collision_free for report in direct_reports)),
            "minimum_clearance_m": float(min(report.clearance for report in direct_reports)),
            "joint_length_rad": _path_length(direct),
        },
        "planned_path": {
            "status": planned.status,
            "success": planned.success,
            "states": len(planned.path),
            "minimum_clearance_m": float(planned.clearance),
            "collision_states": int(sum(not checker.check(q, hand_qpos, "tool_screwdriver",
                                                            approach=False,
                                                            heldtool=carried_tool).collision_free
                                         for q in planned.path)),
            "joint_length_rad": _path_length(planned.path),
            "diagnostics": dict(planned.diagnostics),
        },
        "acceptance": {
            "endpoints_collision_free": bool(start_report.collision_free and goal_report.collision_free),
            "direct_route_blocked": bool(any(not report.collision_free for report in direct_reports)),
            "planned_route_collision_free": bool(planned.success and all(
                checker.check(q, hand_qpos, "tool_screwdriver", approach=False,
                              heldtool=carried_tool).collision_free for q in planned.path)),
            "planning_only_boundary": "No claim of completed physical pick-and-dock execution.",
        },
        "source_sha256": _source_hashes(),
    }
    (output / "obstacle_gate_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "planned_path.json").write_text(json.dumps([q.tolist() for q in planned.path], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timebudget", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    payload = run(args.output, seed=args.seed, timebudget=args.timebudget)
    return 0 if payload["acceptance"]["planned_route_collision_free"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
