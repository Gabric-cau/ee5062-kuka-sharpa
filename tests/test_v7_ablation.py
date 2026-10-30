from pathlib import Path

import mujoco
import pytest

from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import V7TaskExecutor, calibrated_task_spec


@pytest.mark.parametrize("shared", [False, True])
def test_actual_marker_contacts_drive_default_retention_in_both_protocol_modes(tmp_path: Path, shared):
    config = default_v7_scene_config(0)
    scene = build_v7_workbench_scene(tmp_path / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config, shared)
    calls = []
    retain = session.hand.retain
    def recorded_retain(forces, dt):
        calls.append(forces)
        return retain(forces, dt)
    session.hand.retain = recorded_retain
    result = V7TaskExecutor(session).run(calibrated_task_spec("tool_marker", slot_id="slot_marker"))
    assert result.metrics["peak_tactile_force_n"] > .2
    assert bool(calls)
    assert result.trace[-1]["tactile_retention"] is True


def test_retention_disabled_is_explicit_ablation(tmp_path: Path):
    config = default_v7_scene_config(0)
    scene = build_v7_workbench_scene(tmp_path / "scene.xml", config=config)
    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config, False,
                               tactile_retention=False)
    assert session.tactile_retention is False
