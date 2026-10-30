"""Physical fixture acceptance includes release stability and channel force."""

from dataclasses import replace
from pathlib import Path
import sys

import mujoco
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import TaskSpec
from scripts.run_v7_fixture_task import FixtureExecutor


@pytest.mark.parametrize("object_id", ["tool_wrench", "tool_screwdriver"])
def test_fixture_docks_and_releases_below_channel_safety_limit(tmp_path, object_id):
    config = upright_tool_fixture(0)
    path = build_v7_workbench_scene(tmp_path / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(path)), config)
    session.hand.config = replace(session.hand.config, retention_minimum_force_n=3.)
    original = session.hand.config
    wrench = object_id == "tool_wrench"
    spec = TaskSpec(object_id, slot_id=config.registry.by_id(object_id).dock_slot,
                    grasp_offset_m=(-.065, .037, .233 if wrench else .210),
                    approach_axis_world=(0., 0., -1.), grasp_closure=.75 if wrench else .65,
                    lift_height_m=.10, phase_timeout_s=12., hold_seconds=.1,
                    tactile_force_min_n=.15, release_verify_seconds=3., dock_tolerance_m=.025)
    result = FixtureExecutor(session, dock_xy=(-.030, .010) if wrench else (.010, -.012),
                             dock_clearance=.006 if wrench else .014).run(spec)
    assert result.success, result.failure_reason
    assert result.phases[-1].name == "VERIFY"
    assert result.phases[-1].end_time_s - result.phases[-1].start_time_s >= 3.
    assert not any("robot_geom" in event for event in result.metrics["events"])
    assert max(reading["normal_force_n"] for row in result.trace
               for reading in row["tactile"]["fingers"].values()) <= 12.
    assert session.hand.config is original
    assert original.maximum_force_n == 12.


@pytest.mark.parametrize("force,rate", [(12., .8), (0., .8), (8., .4), (8., float("nan"))])
def test_fixture_rejects_relief_settings_that_cannot_unload_below_safety_boundary(force, rate):
    with pytest.raises(ValueError):
        FixtureExecutor(None, dock_relief_force_n=force, dock_relief_rate=rate)


def test_continuous_fixture_unloads_contact_with_previously_placed_tool(tmp_path):
    from scripts.run_v7_tool_sorting import run

    summary = run(tmp_path / "continuous_seed82", seed=82)
    assert summary["success"]
    assert summary["resets_between_tools"] == 0
    assert all(item["still_docked"] for item in summary["final_tools"].values())
    wrench = next(item for item in summary["results"] if item["object_id"] == "tool_wrench")
    measured = wrench["physical_metrics"]
    assert measured["strict_physical_success"] is True
    assert measured["peak_hand_channel_force_n"] <= 12.
    assert measured["target_held_lift"] is True
    assert measured["collision_free"] is True
    assert measured["dock_error_m"] <= .025
    assert measured["release_evidence"]["stable_seconds"] >= 3. - 1e-9
