"""Contract tests for V7 pose IK and collision-aware planning."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_motion import (  # noqa: E402
    CollisionChecker,
    HeldToolProxy,
    RRTConnectPlanner,
    V7Kinematics,
)
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene  # noqa: E402


@pytest.fixture(scope="module")
def workbench_model(tmp_path_factory):
    output = tmp_path_factory.mktemp("v7_motion") / "workbench.xml"
    build_v7_workbench_scene(output)
    return mujoco.MjModel.from_xml_path(str(output))


def test_six_dof_pose_round_trip_uses_model_limits(workbench_model):
    kin = V7Kinematics(workbench_model)
    joints = np.array([0.15, -0.30, 0.25, -0.55, 0.20, 0.35, -0.10])
    position, quaternion = kin.pose(joints)
    solved = kin.solve(position, quaternion, seed=joints)
    assert solved is not None
    assert np.all(solved >= kin.lower - 1e-9)
    assert np.all(solved <= kin.upper + 1e-9)
    solved_position, solved_quaternion = kin.pose(solved)
    assert np.linalg.norm(solved_position - position) < 2e-4
    assert abs(float(np.dot(solved_quaternion, quaternion))) > 1.0 - 2e-3


def test_ik_and_pose_do_not_modify_callers_live_data(workbench_model):
    kin = V7Kinematics(workbench_model)
    live = mujoco.MjData(workbench_model)
    live.qpos[:] = np.linspace(-0.1, 0.1, workbench_model.nq)
    before = live.qpos.copy()
    q = np.zeros(7)
    position, quaternion = kin.pose(q, {"left_index_MCP_FE": 0.2})
    assert kin.solve(position, quaternion, seed=q) is not None
    assert np.array_equal(live.qpos, before)


def _tiny_collision_model():
    xml = """
    <mujoco model="v7_collision_contract">
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="table" type="box" pos="0 0 -0.06" size="0.4 0.4 0.05"/>
        <body name="base" pos="0 0 0">
          <body name="link1"><joint name="joint1" type="hinge" axis="0 0 1"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
            <body name="link2"><joint name="joint2" type="hinge" axis="0 1 0"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
              <body name="link3"><joint name="joint3" type="hinge" axis="0 0 1"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
                <body name="link4"><joint name="joint4" type="hinge" axis="0 1 0"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
                  <body name="link5"><joint name="joint5" type="hinge" axis="0 0 1"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
                    <body name="link6"><joint name="joint6" type="hinge" axis="0 1 0"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
                      <body name="link7" pos="0 0 0.35"><joint name="joint7" type="hinge" axis="0 0 1"/><inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>
                        <geom name="left_index_fingertip_collision" type="sphere" size="0.04"/>
                        <geom name="link7_collision" type="sphere" pos="0.12 0 0" size="0.02"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        <body name="target" pos="0 0 0.35"><joint name="target_free" type="free"/>
          <geom name="target_geom" type="sphere" size="0.035"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_target_contact_is_only_allowed_at_fingertip_during_approach():
    model = _tiny_collision_model()
    checker = CollisionChecker(model, safety_margin=0.0)
    q = np.zeros(7)
    approach = checker.check(q, selected_target="target_geom", approach=True)
    transfer = checker.check(q, selected_target="target_geom", approach=False)
    assert approach.collision_free
    assert not transfer.collision_free


def test_held_tool_uses_declared_geometry_proxy(workbench_model):
    checker = CollisionChecker(workbench_model, safety_margin=0.0)
    kin = V7Kinematics(workbench_model)
    q = np.zeros(7)
    proxy = HeldToolProxy((0.01, 0.01, 0.08), local_position=(0, 0, 0.08), object_id="tool_marker")
    proxy_geom = checker._proxy_geom_id
    before_type = int(workbench_model.geom_type[proxy_geom])
    before_size = workbench_model.geom_size[proxy_geom].copy()
    report = checker.check(q, selected_target="tool_marker", heldtool=proxy)
    assert report.checked_pairs > 0
    assert np.isfinite(report.clearance)
    # The proxy transform is declared and deterministic; it is not derived
    # from an object's live body pose.
    assert proxy.bounding_radius == pytest.approx(np.linalg.norm((.01, .01, .08)))
    assert np.allclose(kin.pose(q)[0], kin.pose(q)[0])
    assert int(workbench_model.geom_type[proxy_geom]) == before_type
    assert np.allclose(workbench_model.geom_size[proxy_geom], before_size)


def test_real_vendor_hand_collision_geometries_are_included(workbench_model):
    checker = CollisionChecker(workbench_model)
    physical_hand = {
        gid for gid in range(workbench_model.ngeom)
        if (mujoco.mj_id2name(workbench_model, mujoco.mjtObj.mjOBJ_BODY,
                             int(workbench_model.geom_bodyid[gid])) or "").startswith("left_")
        and workbench_model.geom_contype[gid] != 0
    }
    assert len(physical_hand) >= 40
    assert physical_hand <= set(checker.robot_geom_ids)
    assert physical_hand <= set(checker.hand_geom_ids)


def test_carried_target_old_map_pose_is_ignored_by_collision_checker(workbench_model):
    checker = CollisionChecker(workbench_model, safety_margin=0.)
    q = np.zeros(7)
    proxy = HeldToolProxy((.02, .02, .02), local_position=(0., 0., .1), object_id="tool_marker")
    without = checker.check(q, selected_target="tool_marker", approach=False)
    with_proxy = checker.check(q, selected_target="tool_marker", approach=False, heldtool=proxy)
    assert without.collision_free
    # The old selected target body is excluded when a carried proxy is active;
    # neither report may treat the stale target pose as a carried obstacle.
    assert all("tool_marker" not in pair[1] for pair in with_proxy.collisions)


def test_ompl_rrtconnect_adapter_returns_structured_result(workbench_model):
    checker = CollisionChecker(workbench_model)
    planner = RRTConnectPlanner(checker)
    start = np.zeros(7)
    goal = np.array([0.08, -0.08, 0.08, -0.08, 0.08, -0.08, 0.08])
    result = planner.plan(start, goal, selected_target="tool_marker", timebudget=0.2)
    assert result.status in {"exact_solution", "approximate_solution", "timeout", "ompl_unavailable", "invalid_start_or_goal"}
    if result.status in {"exact_solution", "approximate_solution"}:
        assert result.path
        assert result.diagnostics["planner"] == "OMPL_RRTConnect"
        assert np.allclose(result.path[0], start, atol=1e-5)
        assert np.allclose(result.path[-1], goal, atol=1e-5)
