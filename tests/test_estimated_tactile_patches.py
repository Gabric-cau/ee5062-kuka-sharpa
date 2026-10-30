from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from estimated_tactile_patches import PublicMeshSurface, associate_tactile_contacts


@pytest.fixture
def box_surface():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="public_object"><geom name="public_surface" type="box" size=".1 .1 .1" mass=".3"/></body>
          <body name="association_probe" mocap="true"><geom name="association_probe_geom" type="sphere" size=".000001"/></body>
        </worldbody></mujoco>
    """)
    return PublicMeshSurface(model, source={"kind": "analytical_box"},
        mass_kg=.3, friction_coefficient=.6)


def pose():
    return {"accepted": True, "origin_m": [.7, 0., .4],
        "rotation_world_from_object": np.eye(3).tolist(),
        "rms_m": .001, "distance_p95_m": .002,
        "near_best_origin_xy_spread_m": .001, "near_best_origin_z_spread_m": .001}


def contact(point=(.802, 0., .4), normal=(1., 0., 0.), pressure=2.):
    return {"finger": "thumb", "position_world_m": list(point),
        "normal_on_hand_world": list(normal), "pressure_n": pressure,
        "friction_coefficient": .5}


def test_flat_face_interior_uses_surface_distance_not_nearest_mesh_vertex(box_surface):
    query = box_surface.query_surface([.102, .013, -.027])
    assert query["accepted"]
    assert query["signed_distance_m"] == pytest.approx(.002, abs=1e-8)
    assert query["outward_normal_local"] == pytest.approx([1., 0., 0.], abs=1e-8)
    inside = box_surface.query_surface([.099, 0., 0.])
    assert inside["signed_distance_m"] == pytest.approx(-.001, abs=1e-8)
    assert inside["outward_normal_local"] == pytest.approx([1., 0., 0.], abs=1e-8)


def test_association_uses_estimated_mesh_pose_and_preserves_conservative_friction(box_surface):
    result = associate_tactile_contacts([contact()], pose(), box_surface, now_s=1., pose_timestamp_s=1.)
    assert result["accepted"]
    assert result["patches"][0]["normal_on_object_world"] == [-1., 0., 0.]
    assert result["patches"][0]["friction_coefficient"] == .5
    assert result["estimated_com_world_m"] == pytest.approx([.7, 0., .4])
    assert not result["runtime_target_state_used"]


def test_unassociated_clutter_pressure_and_wrong_normal_reject_readiness(box_surface):
    bad = contact(point=(.9, 0., .4), pressure=3.)
    result = associate_tactile_contacts([contact(), bad], pose(), box_surface,
        now_s=1., pose_timestamp_s=1.)
    assert not result["accepted"]
    assert result["reason"] == "SIGNIFICANT_UNASSOCIATED_EXTERNAL_PRESSURE"
    wrong = associate_tactile_contacts([contact(normal=(-1., 0., 0.))], pose(), box_surface,
        now_s=1., pose_timestamp_s=1.)
    assert not wrong["accepted"]
    assert wrong["contact_association"][0]["reason"] == "CONTACT_NORMAL_DISAGREES_WITH_ESTIMATED_SURFACE"


def test_ambiguity_and_age_are_explicit_even_when_propagated(box_surface):
    fitted = pose()
    fitted["near_best_origin_xy_spread_m"] = .025
    assert associate_tactile_contacts([contact()], fitted, box_surface,
        now_s=1., pose_timestamp_s=1.)["reason"] == "LOW_CONFIDENCE_OR_AMBIGUOUS_VISUAL_POSE"
    fitted = pose()
    del fitted["near_best_origin_z_spread_m"]
    assert associate_tactile_contacts([contact()], fitted, box_surface,
        now_s=1., pose_timestamp_s=1.)["reason"] == "MISSING_POSE_AMBIGUITY_EVIDENCE"
    fitted = pose()
    fitted["proprio_propagated_scope"] = "Last visual pose propagated by measured wrist motion"
    rejected = associate_tactile_contacts([contact()], fitted, box_surface,
        now_s=10., pose_timestamp_s=1.)
    assert rejected["reason"] == "STALE_OR_FUTURE_VISUAL_POSE"
    allowed = associate_tactile_contacts([contact()], fitted, box_surface,
        now_s=10., pose_timestamp_s=1., max_pose_age_s=15.)
    assert allowed["accepted"] and allowed["pose_age_s"] == 9.
    assert allowed["proprio_propagated_scope"] == fitted["proprio_propagated_scope"]
