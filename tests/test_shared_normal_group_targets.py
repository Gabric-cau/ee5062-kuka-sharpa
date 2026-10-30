"""Original grouping, isolated equality-only extension and infeasibility checks."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shared_normal_group_targets as grouped
import analyze_public_ycb_group_target_feasibility as analysis


def observations():
    definitions = [("thumb", "thumb_tip", [-.03, 0., 0.], [1., 0., 0.], 2.),
        ("middle", "middle_MP", [.03, 0., -.02], [-1., 0., 0.], .4),
        ("middle", "middle_MP", [.03, 0., -.018], [-1., 0., 0.], .6),
        ("ring", "ring_DP", [.03, 0., .02], [-1., 0., 0.], 1.)]
    patches = [dict(finger=finger, input_index=index, hand_body_name=body, position_world_m=point,
        normal_on_object_world=normal, measured_pressure_n=pressure)
        for index, (finger, body, point, normal, pressure) in enumerate(definitions)]
    return [patches, [0., 0., 0.], .15, {"thumb": 2., "middle": 1., "ring": 1.}]


def test_group_proportions_both_gravities_and_no_input_or_original_global_mutation():
    args = observations()
    before, original_solver = copy.deepcopy(args), grouped.shared.linprog
    original_globals = dict(grouped.shared.allocate_tracking_targets.__globals__)
    result = grouped.allocate_group_proportional_targets(*args)
    assert result["feasible"] and args == before and grouped.shared.linprog is original_solver
    assert all(grouped.shared.allocate_tracking_targets.__globals__[key] is value for key, value in original_globals.items())
    assert result["format"] == grouped.FORMAT and result["inherited_target_format"] == "shared_normal_tracking_target_v1"
    witness = result["prospective_group_constraints"]
    assert witness["main_lp_calls"] == 1 and witness["added_equality_rows"] == 1 and witness["previous"] is None
    assert witness["maximum_added_equality_residual_n"] <= 1e-6
    assert witness["maximum_low_gravity_ratio_residual_n"] <= 1e-6
    group = next(item for item in witness["groups"] if item["finger"] == "middle")
    assert group["raw_input_indices"] == [1, 2] and group["measured_proportions"] == pytest.approx([.4, .6])
    high, low = witness["high_patch_normals_n"], witness["low_patch_normals_n"]
    for values in (high, low):
        assert values[1] == pytest.approx(.4 * (values[1] + values[2]), abs=1e-8)
    assert all(row["measured_feasible"] for row in result["independent_fixed_normal_checks"].values())


@pytest.mark.parametrize("change", [
    lambda p: p[2].__setitem__("hand_body_name", "middle_DP"),
    lambda p: p[2].__setitem__("position_world_m", [.03, 0., .01]),
    lambda p: p[2].__setitem__("normal_on_object_world", [0., -1., 0.]),
])
def test_original_same_link_distance_and_normal_rules_are_not_replaced_by_finger_only(change):
    args = observations()
    change(args[0])
    prepared, *_ = grouped.shared._observations(*args, None)
    rows, groups = grouped._group_rows(prepared)
    assert len(groups) == 4 and rows.shape[0] == 0


def test_original_tracking_floor_excludes_weak_patch_before_grouping():
    args = observations()
    args[0].append(dict(args[0][1], input_index=9, measured_pressure_n=.049))
    result = grouped.allocate_group_proportional_targets(*args)
    assert result["feasible"] and result["excluded_tracking_raw_indices"] == [9]
    assert all(9 not in group["raw_input_indices"] for group in result["prospective_group_constraints"]["groups"])


def test_missing_retained_known_hand_link_is_rejected():
    args = observations()
    args[0][1].pop("hand_body_name")
    with pytest.raises(KeyError):
        grouped.allocate_group_proportional_targets(*args)


def main_lp_call():
    count, rays = 4, 16
    n, slack = count * rays, 3 + count
    width = 2 * n + slack
    objective = np.r_[np.full(n, 1e-4), np.full(n, 1e-8), np.ones(slack)]
    kwargs = {"A_eq": np.zeros((12 + count, width)), "b_eq": np.zeros(12 + count),
        "A_ub": np.zeros((15 + 2 * count, width)), "b_ub": np.zeros(15 + 2 * count),
        "bounds": (0., None), "method": "highs", "options": {"primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8, "time_limit": .1}}
    extra = np.zeros((1, width))
    extra[0, 16:32], extra[0, 32:48] = .6, -.4
    return objective, kwargs, extra, count


def test_extension_changes_only_equality_rows_and_appends_zero_rhs():
    objective, kwargs, extra, count = main_lp_call()
    before = copy.deepcopy(kwargs)
    augmented = grouped._append_only_group_equalities(objective, kwargs, extra, count)
    assert np.array_equal(augmented["A_eq"][:-1], kwargs["A_eq"])
    assert np.array_equal(augmented["A_eq"][-1:], extra) and augmented["b_eq"][-1] == 0.
    for key in ("A_ub", "b_ub", "bounds", "method", "options"):
        assert augmented[key] is kwargs[key]
    for key in ("A_eq", "b_eq", "A_ub", "b_ub"):
        assert np.array_equal(kwargs[key], before[key])


@pytest.mark.parametrize("change", [
    lambda c, k: c.__setitem__(0, .001),
    lambda c, k: k["options"].__setitem__("time_limit", 1.),
    lambda c, k: k["options"].__setitem__("primal_feasibility_tolerance", .1),
    lambda c, k: k.__setitem__("method", "highs-ipm"),
    lambda c, k: k.__setitem__("A_eq", k["A_eq"][:-1]),
    lambda c, k: k.__setitem__("bounds", (None, None)),
    lambda c, k: k.__setitem__("unknown", True),
])
def test_main_call_shape_objective_and_time_limit_drift_fail_closed(change):
    objective, kwargs, extra, count = main_lp_call()
    change(objective, kwargs)
    with pytest.raises(ValueError):
        grouped._append_only_group_equalities(objective, kwargs, extra, count)


def test_infeasible_original_shared_normal_problem_is_preserved_without_parameter_sweep():
    args = observations()
    args[0] = [args[0][0], args[0][1], args[0][3]]
    for patch, point in zip(args[0], [[-.03, -.03, 0.], [.03, -.03, 0.], [0., .03, 0.]]):
        patch.update(normal_on_object_world=[0., 0., 1.], position_world_m=point)
    args[2] = .32
    result = grouped.allocate_group_proportional_targets(*args)
    assert result["feasible"] is False and result["reason"] == "SHARED_NORMAL_TRACKING_LP_FAILED"
    assert result["solver_status"] == 2 and result["prospective_group_constraints"]["main_lp_calls"] == 1


@pytest.mark.parametrize("changed", ["observations", "prior_receipt"])
def test_changed_saved_input_or_prior_receipt_is_rejected_before_analysis(tmp_path, monkeypatch, changed):
    observations_path, prior_path = tmp_path / "observations.json", tmp_path / "receipt.json"
    observations_path.write_text(json.dumps([{"timestamp_s": 31., "position_world_m": [0., 0., 0.]}]))
    prior_path.write_text(json.dumps({"recorded_actual_fixed_normal_1g_feasible": False}))
    monkeypatch.setattr(analysis, "SELECTED_OBSERVATIONS_SHA256",
        hashlib.sha256(observations_path.read_bytes()).hexdigest())
    monkeypatch.setattr(analysis, "PRIOR_RECEIPT_SHA256", hashlib.sha256(prior_path.read_bytes()).hexdigest())
    if changed == "observations":
        observations_path.write_text(json.dumps([{"timestamp_s": 31., "position_world_m": [.001, 0., 0.]}]))
    else:
        prior_path.write_text(json.dumps({"recorded_actual_fixed_normal_1g_feasible": True}))
    monkeypatch.setattr(analysis, "analyze_observation", lambda *args: pytest.fail("Modified evidence reached analysis"))
    output = tmp_path / "prospective.json"
    with pytest.raises(ValueError, match="exact saved R19 observations or original source receipt changed"):
        analysis.run(observations_path, prior_path, output)
    assert not output.exists()
