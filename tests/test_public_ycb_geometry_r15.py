"""R15 private nominal geometry contracts; no integrated manipulation claims."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_public_ycb_geometry_r15 as r15


@pytest.fixture(scope="module")
def nominal_model():
    with tempfile.TemporaryDirectory() as directory:
        model = mujoco.MjModel.from_xml_path(str(r15.build_pick_place_scene(
            Path(directory) / "scene.xml", "ycb_can", physics_profile="physical-v6")))
        yield model


@pytest.fixture(scope="module")
def baseline_row(nominal_model):
    model = nominal_model
    profile = json.loads((r15.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
    with patch.object(mujoco, "mj_step", side_effect=AssertionError("Static screen must never integrate")):
        row = r15.evaluate_candidate(model, r15.RobotKinematics(model),
            r15.V7Kinematics(model, position_tolerance=.0001, orientation_tolerance=.00075),
            r15.HandContactReader(model), r15.PublicMeshSurface.from_public_kind("ycb_can"),
            profile, r15.Candidate(1.2, 0., 0.), .0005)
    return row


def test_nominal_geometry_reproduces_frozen_r14_catalogue(baseline_row):
    profile = json.loads((r15.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
    assert baseline_row["static_screen_pass"], baseline_row["failure_reasons"]
    assert baseline_row["task_success"] is None
    assert baseline_row["authored_target_qpos_unchanged"]
    assert baseline_row["closure"] == pytest.approx(profile["contact_closure"], abs=1e-12)
    for finger, value in profile["force_targets_n"].items():
        assert baseline_row["allocation_1_5g"]["target_pressures_n"][finger] == pytest.approx(value, abs=1e-10)
    assert baseline_row["closed_hand_table_clearance_m"] == pytest.approx(
        profile["static_sidewall_metrics"]["closed_hand_table_clearance_m"], abs=1e-12)


def test_world_rotation_order_is_explicit_and_proper():
    candidate = r15.Candidate(1.1, 6., -6.)
    nominal = r15.Rotation.from_euler("y", 30., degrees=True).as_matrix()
    expected = r15.Rotation.from_euler("z", -6., degrees=True).as_matrix() @ (
        r15.Rotation.from_euler("x", 6., degrees=True).as_matrix() @ nominal)
    actual = r15.candidate_rotation(nominal, candidate)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-14)
    assert np.linalg.det(actual) == pytest.approx(1.)


@pytest.mark.parametrize("values", [(float("nan"), 0., 0.), (.7, 0., 0.),
    (1.5, 0., 0.), (1.2, 13., 0.), (1.2, 0., -13.)])
def test_search_bounds_reject_unreasonable_or_invalid_angles(values):
    with pytest.raises(ValueError):
        r15.Candidate(*values)


@pytest.mark.parametrize("mutation, reason", [
    (lambda row: row.update(closed_hand_table_clearance_m=-.001), "CLOSED_TABLE_CLEARANCE"),
    (lambda row: row["allocation_1g"].update(feasible=False), "STATIC_1G_LP_INFEASIBLE"),
    (lambda row: row["allocation_1_5g"].update(feasible=False), "STATIC_1_5G_LP_INFEASIBLE"),
    (lambda row: row["hypothetical_fixed_normal_1g"].update(measured_feasible=False),
        "HYPOTHETICAL_FIXED_NORMAL_1G_INFEASIBLE"),
    (lambda row: row.update(maximum_contact_penetration_m=.002), "DEEPER_THAN_REQUESTED_SHALLOW_CONTACT"),
    (lambda row: row["allocated_contacts"][0].update(surface_normal_cosine=.5),
        "PUBLIC_SURFACE_GEOMETRY_DISAGREES"),
    (lambda row: row.update(allocated_fingers=["middle", "ring"]),
        "INSUFFICIENT_ALLOCATED_FINGERS_INCLUDING_THUMB"),
    (lambda row: row["allocated_contacts"][0].update(surface_normal_cosine=float("nan")),
        "NONFINITE_STATIC_METRIC"),
])
def test_static_admission_rejects_individual_failed_constraints(baseline_row, mutation, reason):
    row = copy.deepcopy(baseline_row)
    mutation(row)
    assert reason in r15.geometry_reasons(row)


def test_hypothetical_normals_are_not_mistaken_for_measured_preload(baseline_row):
    row = baseline_row
    assert row["allocation_1_5g"]["gravity_margin"] == 1.5
    assert row["allocation_1g"]["gravity_margin"] == 1.
    assert not row["allocation_1g"]["measured_feasible"]
    assert row["hypothetical_fixed_normal_1g"]["measured_feasible"]
    assert "hypothetical" in row["hypothetical_fixed_normal_1g"]["scope"]
    assert row["allocation_1_5g"]["normal_pressure_limit_n"] == 6.
    assert row["allocation_1_5g"]["friction_safety_factor"] == .85
    assert len(row["open_approach_samples"]) == 21


def test_private_state_guard_rejects_target_changes_or_time(nominal_model):
    data = mujoco.MjData(nominal_model)
    r15.assert_private_state(nominal_model, data)
    data.time = .002
    with pytest.raises(RuntimeError, match="authored target qpos unchanged"):
        r15.assert_private_state(nominal_model, data)
    data.time = 0.
    address = nominal_model.jnt_qposadr[nominal_model.joint("target_free").id]
    data.qpos[address] += .001
    with pytest.raises(RuntimeError):
        r15.assert_private_state(nominal_model, data)


def test_invalid_search_does_not_create_output(tmp_path):
    profile = r15.PROJECT / "config/public_ycb_sidewall_20261021.json"
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Shallow penetration"):
        r15.run(output, profile, [1.2], [0.], [0.], [.001])
    assert not output.exists()


def test_rank_minimizes_peak_load_before_thumb_load(baseline_row):
    original = copy.deepcopy(baseline_row)
    candidate = copy.deepcopy(baseline_row)
    candidate["allocation_1_5g"]["target_pressures_n"]["thumb"] -= .5
    assert r15.rank_key(candidate) < r15.rank_key(original)
    candidate["allocation_1_5g"]["target_pressures_n"]["middle"] = 5.
    assert r15.rank_key(candidate) > r15.rank_key(original)
