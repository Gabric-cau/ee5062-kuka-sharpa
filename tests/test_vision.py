import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver, CameraObservation
from kuka_sharpa_grasp.perception import above_table, estimate_object, largest_component, point_cloud
from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene


@pytest.fixture
def scene(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml", "ycb_can", overhead_camera=True)))
    data = mujoco.MjData(model)
    GraspRollout(model, data).reset()
    return model, data


def test_cameras_exist_and_follow_wrist(scene):
    model, data = scene
    cam = model.camera("wrist_rgbd").id
    assert model.cam_bodyid[cam] == model.body("link7").id
    assert model.cam_bodyid[model.camera("overhead_rgbd").id] == 0
    before = data.cam_xpos.copy()
    rotation = data.cam_xmat[cam].copy()
    data.qpos[0] += .3
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.cam_xpos[cam] - before[cam]) > .05
    assert not np.allclose(data.cam_xmat[cam], rotation)
    np.testing.assert_array_equal(data.cam_xpos[model.camera("overhead_rgbd").id], before[model.camera("overhead_rgbd").id])


@pytest.mark.parametrize("kind", ["ycb_can", "ycb_potted_meat_can", "ycb_cracker_box"])
def test_render_and_known_object(tmp_path, kind):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml", kind)))
    data = mujoco.MjData(model)
    GraspRollout(model, data).reset()
    with CameraObserver(model) as camera:
        obs = camera.poll(data)
    assert obs.rgb.shape == (240, 320, 3)
    assert obs.rgb.std() > 5
    assert np.isfinite(obs.depth).sum() > 1000
    assert np.nanstd(obs.depth) > .01
    belief = estimate_object(obs, kind)
    assert len(belief.points) > 100
    assert belief.confidence > .1
    assert np.all(belief.points[:, 2] > .388)
    assert .60 < belief.centroid[0] < .85
    assert abs(belief.centroid[1]) < .1
    assert belief.yaw_observable == (kind != "ycb_can")


def test_seeded_noise_dropout_delay_and_reset(scene):
    model, data = scene
    config = CameraConfig(depth_noise_std=.002, dropout=.2, delay=.1, seed=42)
    with CameraObserver(model, config) as first, CameraObserver(model, config) as second:
        assert first.poll(data) is None
        assert second.poll(data) is None
        data.time = .1
        a, b = first.poll(data), second.poll(data)
        assert a.timestamp == 0 and a.available_at == .1
        np.testing.assert_array_equal(a.depth, b.depth)
        assert .15 < np.isnan(a.depth).mean() < .5
        first.reset()
        data.time = 0
        assert first.poll(data) is None
        data.time = .1
        np.testing.assert_array_equal(a.depth, first.poll(data).depth)
    with CameraObserver(model) as clean:
        reference = clean.poll(data)
    valid = np.isfinite(a.depth) & np.isfinite(reference.depth)
    assert .0015 < np.std(a.depth[valid] - reference.depth[valid]) < .0025


def test_table_removal_and_disconnected_depth_edges():
    points = np.array([[[.7, 0, .38], [.7, 0, .4], [.7, 0, .6]],
                       [[.7, 0, .38], [.7, .01, .4], [.7, .01, .6]]])
    mask = above_table(points)
    assert not mask[:, 0].any()
    component = largest_component(mask, points)
    assert component.sum() == 2
    assert not (component[:, 1].any() and component[:, 2].any())


def test_empty_belief_and_no_simulator_access():
    obs = CameraObservation(np.zeros((10, 10, 3), dtype=np.uint8), np.full((10, 10), np.nan),
                            2., 2., np.zeros(3), np.eye(3), 10., "synthetic")
    belief = estimate_object(obs, "ycb_can")
    assert belief.centroid is None and belief.confidence == 0 and belief.timestamp == 2
    # Enforce a frame-only module: simulator state/imports cannot enter perception.
    tree = ast.parse((PROJECT / "src/kuka_sharpa_grasp/perception.py").read_text())
    forbidden = {"qpos", "xpos", "geom_xpos", "body_pos", "mj_name2id", "segmentation"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Import) and any(n.name in {"mujoco", "oracle"} for n in node.names) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.ImportFrom) and "oracle" in (node.module or "") for node in ast.walk(tree))


def test_optional_overhead_and_invalid_config(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_scene(tmp_path / "scene.xml")))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_rgbd") == -1
    with pytest.raises(ValueError):
        CameraObserver(model, CameraConfig(name="missing"))
    for options in ({"dropout": 2}, {"delay": -1}, {"depth_noise_std": float("nan")}, {"width": 0}):
        with pytest.raises(ValueError):
            CameraConfig(**options)


def test_depth_coordinate_convention():
    obs = CameraObservation(np.zeros((3, 3, 3)), np.full((3, 3), .5),
                            0., 0., np.array([.7, 0., .88]), np.eye(3), 10., "synthetic")
    cloud = point_cloud(obs)
    np.testing.assert_allclose(cloud[1, 1], [.7, 0., .38])
    assert cloud[0, 0, 0] < .7 and cloud[0, 0, 1] > 0
    assert not above_table(cloud).any()


@pytest.mark.parametrize("controller", ["fixed", "force"])
def test_camera_only_addition_preserves_physics_and_result(tmp_path, controller):
    path = build_scene(tmp_path / "with_camera.xml", "ycb_can")
    root = ET.parse(path)
    for parent in root.iter():
        for camera in list(parent.findall("camera")):
            parent.remove(camera)
    without = tmp_path / "without_camera.xml"
    root.write(without)
    models = [mujoco.MjModel.from_xml_path(str(p)) for p in (path, without)]
    for attribute in ("body_mass", "body_inertia", "geom_pos", "geom_quat", "geom_friction",
                      "actuator_gainprm", "actuator_biasprm", "qpos0"):
        np.testing.assert_array_equal(getattr(models[0], attribute), getattr(models[1], attribute))
    results = [GraspRollout(m, mujoco.MjData(m)).run(controller=controller) for m in models]
    assert results[0] == results[1]
    assert results[0].status == "success"
