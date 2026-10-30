import sys
from pathlib import Path

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import OBJECT_KINDS, build_scene


def test_combined_scene_compiles_and_force_trial_lifts_object(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml")))
    assert (model.nq, model.nu) == (36, 29)
    result = GraspRollout(model, mujoco.MjData(model)).run(seed=0, controller="force", duration=2.0)
    assert result.status == "success"
    assert result.lifted
    assert result.contacts > 0


def test_ycb_scene_compiles_and_force_trial_holds_the_can(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml", object_kind="ycb_can")))
    result = GraspRollout(model, mujoco.MjData(model)).run(seed=0, controller="force", duration=2.0)
    assert result.status == "success"
    assert result.final_contacts > 0
    assert result.object_z == 0.38


def test_all_catalogued_ycb_meshes_compile(tmp_path):
    for object_kind in OBJECT_KINDS:
        model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / object_kind / "scene.xml", object_kind=object_kind)))
        assert model.nq == 36
        assert model.nu == 29


def test_trace_callback_receives_live_simulation_frames(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml")))
    frames = []
    GraspRollout(model, mujoco.MjData(model)).run_trace(
        seed=0,
        controller="fixed",
        duration=0.02,
        frame_callback=lambda step, data, readings: frames.append((step, float(data.time), readings)),
    )
    assert frames
    assert frames[-1][0] == 10
