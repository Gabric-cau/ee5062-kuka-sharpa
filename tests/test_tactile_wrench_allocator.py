from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tactile_wrench_allocator import allocate_contact_wrench


def opposed(pressures=(5., 5.)):
    return [{"finger": finger, "position_world_m": [x, 0., 0.],
        "normal_on_object_world": [normal, 0., 0.], "measured_pressure_n": pressure}
        for finger, x, normal, pressure in zip(
            ("thumb", "index"), (-.04, .04), (1., -1.), pressures)]


def test_opposed_horizontal_contacts_match_analytic_gravity_bound():
    mass, friction, safety, margin = .349, .8, .85, 1.5
    result = allocate_contact_wrench(opposed(), [0., 0., 0.], mass,
        friction_coefficient=friction, friction_safety_factor=safety, gravity_margin=margin)
    assert result["feasible"] and result["measured_feasible"]
    expected = mass * 9.81 * margin / (friction * safety)
    assert result["minimum_total_normal_load_n"] == pytest.approx(expected, abs=1e-7)
    assert result["target_pressures_n"]["thumb"] == pytest.approx(expected / 2.)
    assert result["target_pressures_n"]["palm"] == 0.
    assert np.linalg.norm(result["target_wrench_residual"]) < 1e-7


def test_measured_capacity_does_not_hide_unbalanced_actual_normal_forces():
    result = allocate_contact_wrench(opposed((5., 4.)), [0., 0., 0.], .349)
    assert result["feasible"] and result["measured_capacity_feasible"]
    assert not result["measured_feasible"]


def test_top_contacts_cannot_pull_object_up_and_far_com_exceeds_torque_capacity():
    top = [{"finger": finger, "position_world_m": [x, 0., .05],
        "normal_on_object_world": [0., 0., -1.], "measured_pressure_n": 5.}
        for finger, x in (("thumb", -.02), ("index", .02))]
    assert not allocate_contact_wrench(top, [0., 0., 0.], .349)["feasible"]
    assert not allocate_contact_wrench(opposed(), [.2, 0., 0.], .349)["feasible"]


def test_pressure_cap_and_nonfinite_inputs_fail_without_a_feasibility_claim():
    assert not allocate_contact_wrench(opposed(), [0., 0., 0.], 2.)["feasible"]
    assert allocate_contact_wrench(opposed(), [np.nan, 0., 0.], .349)["reason"] == "INVALID_INPUT"
    assert allocate_contact_wrench(opposed(), [0., 0., 0.], .349,
        pressure_limit_n=12.)["reason"] == "INVALID_INPUT"
    assert allocate_contact_wrench(opposed(), [0., 0., 0.], .349,
        cone_rays=100000)["reason"] == "INVALID_INPUT"


def test_palm_normal_load_cannot_be_omitted_from_measured_equilibrium():
    patches = opposed() + [{"finger": "palm", "position_world_m": [0., 0., -.05],
        "normal_on_object_world": [0., 0., 1.], "measured_pressure_n": 1.}]
    result = allocate_contact_wrench(patches, [0., 0., 0.], .349)
    assert result["feasible"]
    assert not result["measured_feasible"]
    assert result["target_pressures_n"]["palm"] == 0.
