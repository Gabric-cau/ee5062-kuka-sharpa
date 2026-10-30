"""Acceptance contract for the task-level obstacle planning gate."""

from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_v7_obstacle_gate import run  # noqa: E402


def test_obstacle_gate_blocks_direct_route_and_validates_rrt_path(tmp_path):
    pytest.importorskip("ompl.base", reason="Exact RRT path validation requires the optional OMPL binding")
    # RRT-Connect can return an approximate route near the short budget on a
    # different random seed; three seconds is the declared acceptance budget.
    result = run(tmp_path / "gate", timebudget=3.0)
    assert result["acceptance"]["endpoints_collision_free"]
    assert result["acceptance"]["direct_route_blocked"]
    assert result["acceptance"]["planned_route_collision_free"]
    assert result["planned_path"]["status"] == "exact_solution"
    assert result["planned_path"]["collision_states"] == 0
    assert result["planned_path"]["minimum_clearance_m"] > 0.004
