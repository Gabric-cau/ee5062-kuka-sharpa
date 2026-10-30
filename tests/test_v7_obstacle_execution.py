"""Physical regression: actuator-only motion must expose the blocked baseline."""

from dataclasses import replace
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from run_v7_obstacle_execution import execute
from run_v7_obstacle_gate import START_Q, GOAL_Q, TRANSFER_BARRIER
from kuka_sharpa_grasp.v7_motion import CollisionChecker
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config


def test_actuator_execution_records_physical_collision_and_stops(tmp_path):
    base = default_v7_scene_config()
    cfg = replace(base, obstacles=base.obstacles + (TRANSFER_BARRIER,))
    path = build_v7_workbench_scene(tmp_path / "scene.xml", config=cfg)
    model = mujoco.MjModel.from_xml_path(str(path))
    checker = CollisionChecker(model, cfg)
    direct = tuple(START_Q*(1-t) + GOAL_Q*t for t in np.linspace(0., 1., 301))
    result = execute(model, checker, direct, tmp_path / "direct")
    assert result["stopped"]
    assert not result["success"]
    assert result["environment_contact_steps"] > 0
    assert result["peak_environment_force_n"] > .5
    assert result["live_object_qpos_writes"] == 0
    assert result["elapsed_simulation_s"] < 1.
    assert (tmp_path / "direct" / "contacts.jsonl.gz").stat().st_size > 0
