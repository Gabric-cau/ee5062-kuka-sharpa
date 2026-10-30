"""Static allocation contracts, infeasible geometry and adversarial LP outputs."""

import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shared_normal_tracking_targets as shared


def observations():
    definitions = [("thumb", [-.03, 0., 0.], [1., 0., 0.], 2.),
        ("middle", [.03, 0., -.02], [-1., 0., 0.], 1.),
        ("ring", [.03, 0., .02], [-1., 0., 0.], 1.)]
    patches = [dict(finger=finger, input_index=index, position_world_m=point,
        normal_on_object_world=normal, measured_pressure_n=pressure, hand_body_name=finger + "_tip")
        for index, (finger, point, normal, pressure) in enumerate(definitions)]
    return [patches, [0., 0., 0.], .15, {"thumb": 2., "middle": 1., "ring": 1.}]


def test_original_interface_pure_inputs_and_exact_reference_solution():
    args = observations()
    before = copy.deepcopy(args)
    result = shared.allocate_tracking_targets(*args)
    assert args == before and result["feasible"]
    assert list(inspect.signature(shared.allocate_tracking_targets).parameters) == ["patches", "com", "mass", "pressures", "previous"]
    assert not result["runtime_target_state_used"]
    assert result["targets_n"] == {"thumb": 2., "index": 0., "middle": 1., "ring": 1., "pinky": 0., "palm": 0.}
    assert result["reference_n"] == args[3]
    assert result["objective"]["channel_reference_l1_n"] == pytest.approx(0., abs=1e-10)
    assert result["objective"]["patch_measured_l1_n"] == pytest.approx(0., abs=1e-10)
    assert [patch["raw_input_index"] for patch in result["patches"]] == [0, 1, 2]
    assert all(patch["hand_body_name"].endswith("_tip") for patch in result["patches"])
    assert result["hypothetical_target_fixed_normal_1g_feasible"]
    assert result["hypothetical_target_fixed_normal_1_5g_feasible"]
    assert all(row["measured_feasible"] for row in result["independent_fixed_normal_checks"].values())


def test_previous_reference_blending_and_unchanged_patch_l1_objective():
    args = observations()
    previous = {"thumb": 4., "middle": 2., "ring": 2.}
    saved = copy.deepcopy(previous)
    result = shared.allocate_tracking_targets(*args, previous)
    assert result["feasible"] and previous == saved
    assert result["reference_n"] == {"thumb": 3.5, "middle": 1.75, "ring": 1.75}
    targets = result["targets_n"]
    assert result["objective"]["channel_reference_l1_n"] == pytest.approx(sum(abs(targets[f] - result["reference_n"][f]) for f in shared.ACTIVE))
    assert result["objective"]["patch_measured_l1_n"] == pytest.approx(sum(abs(p["target_normal_n"] - p["measured_pressure_n"]) for p in result["patches"]))


def test_shared_normals_reject_two_individually_feasible_upward_contact_loads():
    args = observations()
    points = [[-.03, -.03, 0.], [.03, -.03, 0.], [0., .03, 0.]]
    for patch, point in zip(args[0], points, strict=True):
        patch.update(position_world_m=point, normal_on_object_world=[0., 0., 1.])
    args[2] = .32
    for margin in (1., 1.5):
        independent = shared.allocator.allocate_contact_wrench(args[0], args[1], args[2], gravity_margin=margin)
        assert independent["feasible"]
        assert all(.75 <= independent["target_pressures_n"][finger] <= 6. for finger in shared.ACTIVE)
    result = shared.allocate_tracking_targets(*args)
    assert not result["feasible"] and result["reason"] == "SHARED_NORMAL_TRACKING_LP_FAILED"
    assert result["solver_status"] == 2


def test_original_geometry_failure_is_preserved_without_calling_tracking_lp(monkeypatch):
    args = observations()
    for patch in args[0]:
        patch["normal_on_object_world"] = [1., 0., 0.]
    monkeypatch.setattr(shared, "linprog", lambda *a, **k: pytest.fail("Geometry precheck bypassed"))
    result = shared.allocate_tracking_targets(*args)
    assert not result["feasible"] and result["reason"].startswith("ORIGINAL_GEOMETRY_LP_")


@pytest.mark.parametrize("change", [
    lambda a: a.__setitem__(1, [0., 0.]),
    lambda a: a.__setitem__(2, float("nan")),
    lambda a: a.__setitem__(2, 0.),
    lambda a: a.__setitem__(2, True),
    lambda a: a[3].__setitem__("thumb", float("inf")),
    lambda a: a[3].pop("ring"),
    lambda a: a[3].__setitem__("unknown", 1.),
    lambda a: a[0][0].__setitem__("normal_on_object_world", [0., 0., 0.]),
    lambda a: a[0][0].__setitem__("position_world_m", [0., float("nan"), 0.]),
    lambda a: a[0][0].__setitem__("friction_coefficient", -.1),
    lambda a: a[0][0].__setitem__("measured_pressure_n", float("nan")),
    lambda a: a[0][0].__setitem__("input_index", .2),
    lambda a: a[0][0].__setitem__("input_index", True),
    lambda a: a[0][0].__setitem__("input_index", 1),
])
def test_invalid_observations_fail_closed_before_tracking_solve(monkeypatch, change):
    args = observations()
    change(args)
    monkeypatch.setattr(shared, "linprog", lambda *a, **k: pytest.fail("Invalid data reached tracking solve"))
    result = shared.allocate_tracking_targets(*args)
    assert not result["feasible"] and result["reason"] == "INVALID_INPUT"


@pytest.mark.parametrize("previous", [{"thumb": 1.}, {"thumb": float("nan"), "middle": 1., "ring": 1.}])
def test_invalid_previous_reference_is_not_replaced_with_measured(previous):
    result = shared.allocate_tracking_targets(*observations(), previous)
    assert not result["feasible"] and result["reason"] == "INVALID_INPUT"


def test_patch_floor_and_raw_indices_preserve_excluded_contacts():
    args = observations()
    args[0].append(dict(args[0][0], input_index=7, measured_pressure_n=.049))
    args[0].append(dict(args[0][0], finger="index", input_index=8))
    result = shared.allocate_tracking_targets(*args)
    assert result["feasible"] and result["excluded_tracking_raw_indices"] == [7, 8]
    args[0][0]["measured_pressure_n"] = .049
    assert shared.allocate_tracking_targets(*args)["reason"] == "THREE_OBSERVED_ACTIVE_FINGERS_REQUIRED"


def test_original_all_channel_pressure_consistency_precheck_retained():
    args = observations()
    args[3]["thumb"] = 1.
    result = shared.allocate_tracking_targets(*args)
    assert not result["feasible"] and result["reason"] == "ORIGINAL_GEOMETRY_LP_INVALID_INPUT"


def test_matrix_contract_objective_fixed_parameters_and_runtime_limit(monkeypatch):
    solve = shared.linprog
    def observed(c, **kwargs):
        n, count = 48, 3
        assert len(c) == 2 * n + 3 + count
        np.testing.assert_equal(c[:n], np.full(n, 1e-4))
        np.testing.assert_equal(c[n:2*n], np.full(n, 1e-8))
        np.testing.assert_equal(c[2*n:], np.ones(3 + count))
        assert kwargs["options"]["time_limit"] == .1
        assert kwargs["bounds"] == (0., None)
        assert kwargs["A_eq"].shape == (12 + count, len(c))
        np.testing.assert_equal(kwargs["b_eq"][-count:], np.zeros(count))
        np.testing.assert_equal(kwargs["b_ub"][:6], [6., 6., 6., 6., 6., 0.])
        np.testing.assert_equal(kwargs["b_ub"][6:9], [-.75, -.75, -.75])
        return solve(c, **kwargs)
    monkeypatch.setattr(shared, "linprog", observed)
    assert shared.allocate_tracking_targets(*observations())["feasible"]


@pytest.mark.parametrize("mutation,reason", [
    ("nan", "SHARED_NORMAL_TRACKING_LP_NONFINITE_OR_SHAPE"),
    ("shape", "SHARED_NORMAL_TRACKING_LP_NONFINITE_OR_SHAPE"),
    ("negative", "SHARED_NORMAL_TRACKING_LP_RESIDUAL"),
    ("wrench", "SHARED_NORMAL_TRACKING_LP_RESIDUAL"),
    ("shared", "SHARED_NORMAL_TRACKING_LP_RESIDUAL"),
    ("timeout", "SHARED_NORMAL_TRACKING_LP_FAILED"),
    ("contradictory_status", "SHARED_NORMAL_TRACKING_LP_FAILED"),
    ("overflow", "SHARED_NORMAL_TRACKING_LP_NONFINITE_RESIDUAL"),
])
def test_solver_success_flag_cannot_bypass_numerical_constraints(monkeypatch, mutation, reason):
    solve = shared.linprog
    def changed(*args, **kwargs):
        solved = solve(*args, **kwargs)
        assert solved.success
        if mutation == "nan":
            solved.x[0] = float("nan")
        elif mutation == "shape":
            solved.x = solved.x[:-1]
        elif mutation == "negative":
            solved.x[0] = -.1
        elif mutation == "wrench":
            solved.x[:48] *= 1.1
        elif mutation == "shared":
            # A balanced added squeeze leaves the 1g wrench unchanged but changes normals.
            solved.x[48:64] += .02 / 16
            solved.x[64:96] += .01 / 16
        elif mutation == "contradictory_status":
            solved.status = 1
        elif mutation == "overflow":
            solved.x[:] = 1e308
        else:
            solved.success, solved.status = False, 1
        return solved
    monkeypatch.setattr(shared, "linprog", changed)
    with np.errstate(over="ignore", invalid="ignore"):
        result = shared.allocate_tracking_targets(*observations())
    assert not result["feasible"] and result["reason"] == reason
    json.dumps(result, allow_nan=False)
    if mutation == "shared":
        assert result["numerical_checks"]["force_residual_max_n"] < 1e-5
        assert result["numerical_checks"]["torque_residual_max_nm"] < 1e-6
        assert result["numerical_checks"]["shared_patch_normal_residual_max_n"] > .01


def test_independent_fixed_normal_postcheck_failure_blocks_candidate(monkeypatch):
    allocate = shared.allocator.allocate_contact_wrench
    calls = []
    def changed(*args, **kwargs):
        result = allocate(*args, **kwargs)
        calls.append(kwargs.get("gravity_margin", 1.5))
        if len(calls) > 1 and kwargs["gravity_margin"] == 1.:
            result["measured_feasible"] = False
        return result
    monkeypatch.setattr(shared.allocator, "allocate_contact_wrench", changed)
    result = shared.allocate_tracking_targets(*observations())
    assert calls == [1.5, 1., 1.5]
    assert not result["feasible"] and result["reason"] == "SHARED_NORMAL_INDEPENDENT_FIXED_NORMAL_POSTCHECK_FAILED"


def test_real_r18_failed_states_with_observed_reference_only():
    fixture = json.loads((Path(__file__).parent / "fixtures/shared_normal_r18_observations_20261022.json").read_text())
    assert fixture["source_summary"]["sha256"] == "d3d5dec564810d8c9b6a9cc4cee8354062eb9f99c54960ab49b41ad87a740ff0"
    assert fixture["source_preload"]["sha256"] == "5ad3f42ab572f8e5f28f1e3fc24eb4bf07a79e003db8279f54431f02c0ba7f61"
    assert len(fixture["samples"]) == 2
    for sample in fixture["samples"]:
        assert sample["association_accepted"] and not sample["recorded_target_feasible"]
        result = shared.allocate_tracking_targets(sample["patches"], sample["com"], sample["mass"], sample["pressures"])
        assert result["feasible"]
        assert result["effective_patch_mu"] == pytest.approx([.7745966692414834 * .85] * 4)
        assert all(.75 - 1e-6 <= result["targets_n"][finger] <= 6. + 1e-6 for finger in shared.ACTIVE)
        assert all(result["independent_fixed_normal_checks"][str(margin)]["measured_feasible"] for margin in (1., 1.5))
        json.dumps(result, allow_nan=False)
