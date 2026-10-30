"""Stopping freezes motor goals, not the integrated MuJoCo state."""

import mujoco
import numpy as np
import pytest

from kuka_sharpa_grasp.teleop import TeleopCommand
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession


@pytest.fixture
def moving_session(tmp_path):
    config = default_v7_scene_config(3)
    scene = build_v7_workbench_scene(tmp_path / "workbench.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config)
    session.command(TeleopCommand(timestamp=0., translation_m=(.025, 0., .020),
                                 grip_delta=1., source="stop_regression"))
    assert session.status == "MOVING"
    session.set_joint_target("left_index_MCP_AA", .25)
    for _ in range(25):
        session.step()
    assert not session.stopped
    assert np.linalg.norm(session.limiter.velocity) > 1e-4
    return session


def hand_controls(session):
    return {name: float(session.data.ctrl[aid]) for name, aid in session.hand.actuators.items()
            if name.startswith("left_")}


@pytest.mark.parametrize("reason", ["OPERATOR_STOP", "TACTILE_OVERLOAD_STOP", "ENVIRONMENT_COLLISION_STOP"])
def test_stop_holds_all_motor_references_and_keeps_physics_running(moving_session, reason):
    session = moving_session
    hand = hand_controls(session)
    closure = session.hand.closure.copy()
    arm = session.limiter.position.copy()
    positions = session.data.qpos.copy()
    velocities = session.data.qvel.copy()
    timestamp = float(session.data.time)
    session.stop(reason)
    assert np.array_equal(session.data.qpos, positions)
    assert np.array_equal(session.data.qvel, velocities)
    assert session.requested_closure == closure
    assert session.finger_overrides["left_index_MCP_AA"] == hand["left_index_MCP_AA"]
    assert np.array_equal(session.limiter.velocity, np.zeros(7))
    for _ in range(100):
        snapshot = session.step()
        assert snapshot["status"] == reason
        assert hand_controls(session) == hand
        assert session.hand.closure == closure
        assert np.array_equal(session.arm_goal, arm)
        assert np.array_equal(session.limiter.position, arm)
    assert session.data.time == pytest.approx(timestamp + .2)
    assert np.linalg.norm(session.data.qpos - positions) > 1e-6


@pytest.mark.parametrize("button", ["escape", "space"])
def test_stop_has_priority_over_resume_and_commands_in_the_same_input(moving_session, button):
    session = moving_session
    hand = hand_controls(session)
    closure = session.hand.closure.copy()
    session.command(TeleopCommand(timestamp=float(session.data.time),
                                 buttons=(button, "r", "hold"), grip_delta=-1.,
                                 translation_m=(.05, 0., 0.), source="stop_regression"))
    assert session.stopped
    assert session.status == "OPERATOR_STOP"
    assert session.requested_closure == closure
    for _ in range(25):
        session.step()
    assert hand_controls(session) == hand


def test_stopped_commands_are_discarded_and_resume_does_not_restore_old_goals(moving_session):
    session = moving_session
    session.command(TeleopCommand(timestamp=float(session.data.time), buttons=("escape",),
                                 source="stop_regression"))
    hand = hand_controls(session)
    closure = session.hand.closure.copy()
    arm = session.arm_goal.copy()
    wrist_goal = session.position_goal.copy()
    held_joint = session.finger_overrides["left_index_MCP_AA"]
    session.set_finger_target("index", 1.)
    session.set_joint_target("left_index_MCP_AA", .25)
    session.command(TeleopCommand(timestamp=float(session.data.time), buttons=("hold",),
                                 grip_delta=1., translation_m=(.05, 0., 0.), source="stop_regression"))
    assert session.requested_closure == closure
    assert session.finger_overrides["left_index_MCP_AA"] == held_joint
    assert np.array_equal(session.position_goal, wrist_goal)
    assert session.hand.hold_reference is None
    for _ in range(25):
        session.step()
    session.command(TeleopCommand(timestamp=float(session.data.time), buttons=("r",),
                                 source="stop_regression"))
    for _ in range(100):
        session.step()
        assert hand_controls(session) == hand
        assert session.hand.closure == closure
        assert np.array_equal(session.limiter.position, arm)
    assert not session.stopped
    assert session.status == "READY"
    session.set_finger_target("index", .3)
    session.set_joint_target("left_index_MCP_AA", .2)
    actuator = session.hand.actuators["left_index_MCP_AA"]
    qadr = session.model.jnt_qposadr[session.model.actuator_trnid[actuator, 0]]
    initial_angle = float(session.data.qpos[qadr])
    session.command(TeleopCommand(timestamp=float(session.data.time), translation_m=(0., 0., .01),
                                 source="stop_regression"))
    session.step()
    assert session.hand.closure["index"] - closure["index"] == pytest.approx(.4 * session.model.opt.timestep)
    assert hand_controls(session)["left_index_MCP_AA"] - held_joint == pytest.approx(.5 * session.model.opt.timestep)
    assert session.status == "MOVING"
    for _ in range(100):
        session.step()
    assert float(session.data.qpos[qadr]) > initial_angle + .01


def test_reset_discards_a_stopped_hold(moving_session):
    session = moving_session
    session.stop()
    session.reset()
    assert not session.stopped
    assert session.finger_overrides == {}
    session.set_finger_target("index", .2)
    session.step()
    assert session.hand.closure["index"] > 0.
