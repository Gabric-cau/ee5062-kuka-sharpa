"""Tests for the physical V7 task contract."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config  # noqa: E402
from kuka_sharpa_grasp.v7_session import WorkbenchSession  # noqa: E402
from kuka_sharpa_grasp.v7_task import TaskSpec, V7TaskExecutor, calibrated_task_spec  # noqa: E402
from kuka_sharpa_grasp.teleop import TeleopActionLog, TeleopCommand  # noqa: E402
from kuka_sharpa_grasp.v7_calibrate import GraspCandidate  # noqa: E402


@pytest.fixture
def workbench(tmp_path):
    config = default_v7_scene_config(3)
    scene = build_v7_workbench_scene(tmp_path / "workbench.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(scene))
    return model, config


def test_task_spec_normalizes_approach_axis():
    spec = TaskSpec("tool_marker", approach_axis_world=(0, 0, -4))
    assert spec.approach_axis_world == (0.0, 0.0, -1.0)
    with pytest.raises(ValueError):
        TaskSpec("tool_marker", grasp_closure=1.1)


def test_calibration_candidate_normalizes_approach_axis():
    candidate = GraspCandidate((0.0, 0.0, 0.2), approach_axis_world=(0.0, 0.0, -2.0))
    assert candidate.approach_axis_world == (0.0, 0.0, -1.0)
    with pytest.raises(ValueError):
        GraspCandidate((0.0, 0.0, 0.2), approach_axis_world=(0.0, 0.0, 0.0))


def test_default_tasks_only_contains_actionable_registry_objects(workbench):
    model, config = workbench
    executor = V7TaskExecutor(WorkbenchSession(model, config, shared_control=False))
    tasks = executor.default_tasks()
    assert {task.object_id for task in tasks} == {item.object_id for item in config.registry.actionable}
    with pytest.raises(ValueError):
        executor.default_tasks(("distractor_box",))


def test_short_episode_records_physical_failure_without_qpos_oracle(workbench):
    model, config = workbench
    session = WorkbenchSession(model, config, shared_control=False)
    before = session.data.qpos.copy()
    result = V7TaskExecutor(session).run(TaskSpec("tool_marker", phase_timeout_s=.25, hold_seconds=0.01))
    after = session.data.qpos.copy()
    assert result.success is False
    assert result.failure_reason is not None
    assert result.trace
    assert result.metrics["qpos_write_policy"] == "executor_never_writes_object_qpos"
    # The executor did not assign the free-joint state.  The short episode has
    # no time to move this marker, so the initial physical state is retained.
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "v7_obj_tool_marker_free")]
    assert np.allclose(before[qadr:qadr + 7], after[qadr:qadr + 7], atol=2e-3)
    assert result.trace[-1]["evaluation"]["tool_marker"]["position"] == pytest.approx(
        after[qadr:qadr + 3], abs=.01)


def test_task_result_trace_contains_tactile_and_evaluation_channels(workbench):
    model, config = workbench
    session = WorkbenchSession(model, config, shared_control=False)
    result = V7TaskExecutor(session).run(TaskSpec("tool_marker", phase_timeout_s=.3, hold_seconds=0.01))
    sample = result.trace[-1]
    assert "tactile" in sample and "evaluation" in sample
    assert "total_normal_force_n" in sample["tactile"]
    assert "tool_marker" in sample["evaluation"]


def test_reset_clears_hand_retention_between_episodes(workbench):
    model, config = workbench
    session = WorkbenchSession(model, config)
    session.hand.hold_reference = {"index": .8}
    session.hand.engaged.add("index")
    session.hand.slip_adjustments = 42
    session.reset()
    assert session.hand.hold_reference is None
    assert session.hand.engaged == set()
    assert session.hand.slip_adjustments == 0


def test_manual_mode_preserves_collision_evidence(workbench, monkeypatch):
    model, config = workbench
    session = WorkbenchSession(model, config, shared_control=False)
    event = {"time_s": .002, "robot_geom": "left_index_fingertip_collision",
             "other_geom": "v7_divider_center", "normal_force_n": 5., "penetration_m": .004}
    monkeypatch.setattr(session, "_collision_events", lambda: [event])
    session.step()
    assert event in session.events
    assert not session.stopped


def test_independent_joint_override_accumulates_and_moves_real_finger(workbench):
    model, config = workbench
    session = WorkbenchSession(model, config)
    name = "left_index_MCP_AA"
    aid = session.hand.actuators[name]
    jid = model.actuator_trnid[aid, 0]
    qadr = model.jnt_qposadr[jid]
    initial = float(session.data.qpos[qadr])
    session.set_joint_target(name, .25)
    previous = float(session.data.ctrl[aid])
    for _ in range(200):
        session.step()
        command = float(session.data.ctrl[aid])
        assert abs(command - previous) <= .5 * model.opt.timestep + 1e-12
        previous = command
    assert abs(float(session.data.qpos[qadr]) - initial) > .02
    assert session.data.ctrl[aid] > .15
    session.reset()
    assert session.finger_override_commands == {}
    assert session.finger_overrides == {}


def test_marker_completes_physical_release_and_exact_command_replay(tmp_path):
    config = default_v7_scene_config(100)
    scene = build_v7_workbench_scene(tmp_path / "marker.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(scene))
    session = WorkbenchSession(model, config)
    result = V7TaskExecutor(session).run(calibrated_task_spec("tool_marker", slot_id="slot_marker"))
    assert result.success
    assert result.phases[-1].name == "VERIFY"
    assert max(result.trace[-1]["finger_closure"].values()) <= .03
    assert not any("robot_geom" in event for event in result.metrics["events"])
    close = next(phase for phase in result.phases if phase.name == "CLOSE")
    assert close.end_time_s - close.start_time_s >= 2.
    lift_samples = [row for row in result.trace if row.get("phase") == "LIFT"]
    assert lift_samples[-1]["evaluation"]["tool_marker"]["position"][2] > config.registry.by_id("tool_marker").pose[2] + .08

    replay = WorkbenchSession(model, config)
    log = TeleopActionLog(TeleopCommand.from_dict(row) for row in result.metrics["action_log"])
    while replay.data.time < result.trace[-1]["time_s"] - model.opt.timestep / 2:
        for command in log.due(float(replay.data.time)):
            replay.command(command)
        replay.step()
    assert replay.data.qpos == pytest.approx(session.data.qpos, abs=1e-10)
