"""Focused contracts for the independent V7 cluttered workbench layer."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.object_registry import VisibleDetection, default_workbench_registry
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config


def test_default_registry_has_three_actionable_tools_and_declared_provenance():
    registry = default_workbench_registry()
    assert len(registry.objects) == 6
    assert tuple(item.object_id for item in registry.actionable) == (
        "tool_screwdriver", "tool_wrench", "tool_marker",
    )
    assert len(registry.distractors) == 3
    assert registry.by_id("distractor_ycb_can").mesh_name == "ycb_005_tomato_soup_can"
    assert all(item.source for item in registry.objects)
    assert all(item.mass_kg > 0 and len(item.friction) == 3 for item in registry.objects)


def test_visible_selection_is_known_id_only_deterministic_and_confidence_aware():
    registry = default_workbench_registry()
    detections = [
        {"object_id": "unknown_object", "centroid_xyz": (0.7, 0., .4), "confidence": .99},
        VisibleDetection("tool_wrench", (.73, .04, .4), .70, mask_area=20),
        VisibleDetection("tool_screwdriver", (.61, -.02, .4), .70, mask_area=31),
        VisibleDetection("tool_screwdriver", (.61, -.02, .4), .50, mask_area=100),
        VisibleDetection("distractor_box", (.62, .08, .4), .99),
    ]
    matches = registry.match_visible(detections)
    assert [match.spec.object_id for match in matches] == ["tool_screwdriver", "tool_wrench"]
    assert matches[0].detection.mask_area == 31
    assert registry.select_target(detections=detections).object_id == "tool_screwdriver"
    with pytest.raises(ValueError, match="not visible"):
        registry.select_target("tool_marker", detections=detections)
    assert registry.select_target("tool_marker").category == "marker"


def test_registry_rejects_invalid_detection_and_threshold():
    registry = default_workbench_registry()
    with pytest.raises(ValueError, match="confidence"):
        registry.match_visible([{"object_id": "tool_marker", "centroid_xyz": (0., 0., 0.), "confidence": 2}])
    with pytest.raises(ValueError, match="min_confidence"):
        registry.match_visible([], min_confidence=-.1)


def test_v7_scene_compiles_with_objects_obstacles_slots_and_explicit_pairs(tmp_path):
    config = default_v7_scene_config(seed=17)
    path = build_v7_workbench_scene(tmp_path / "v7_scene.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nbody >= 6 + 1  # six free objects plus the arm/hand hierarchy
    assert model.ngeom >= 6 + len(config.obstacles) + 5  # objects, fixtures and rack
    assert model.nsite >= len(config.slots) + 6
    assert {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)} >= {
        "v7_touch_thumb", "v7_touch_index", "v7_touch_middle", "v7_touch_ring", "v7_touch_pinky",
    }
    numeric = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "v7_scene_version")
    assert numeric >= 0
    assert model.numeric_data[model.numeric_adr[numeric]] == 7
    for spec in config.registry.objects:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"v7_geom_{spec.object_id}")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"v7_obj_{spec.object_id}")
        assert geom >= 0 and body >= 0
        assert model.body_mass[body] == pytest.approx(spec.mass_kg)
        table_pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, f"v7_{spec.object_id}_table")
        assert table_pair >= 0
        assert model.pair_geom1[table_pair] == geom or model.pair_geom2[table_pair] == geom
    # The wrench proxy is authored along local X and must stay horizontal;
    # only capsule/cylinder tools use their declared local-Z grasp axis.
    np.testing.assert_allclose(model.body_quat[model.body("v7_obj_tool_wrench").id], (1., 0., 0., 0.))
    assert not np.allclose(model.body_quat[model.body("v7_obj_tool_screwdriver").id], (1., 0., 0., 0.))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "v7_obstacle_divider_center") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "v7_slot_slot_screwdriver") >= 0

