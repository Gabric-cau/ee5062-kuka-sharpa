"""Analytic static-load checks and current observation contract failures."""

import copy
from dataclasses import replace

import numpy as np
import pytest

import kuka_sharpa_grasp.contact_load_targets as loads
from kuka_sharpa_grasp.contact_policy import FingerContactPolicy


def patch(index, finger, point, normal, force, body=None):
    return {"input_index": index, "finger": finger, "hand_body_name": body or "left_" + finger + "_tip",
        "position_world_m": point, "normal_on_object_world": normal, "measured_pressure_n": force}


def pressures(**values):
    return dict(dict.fromkeys(loads.CHANNELS, 0.), **values)


def opposed():
    return [patch(10, "thumb", [-.04, 0., 0.], [1., 0., 0.], .1),
        patch(30, "index", [.04, 0., 0.], [-1., 0., 0.], .1)]


def solve(patches=None, readings=None, **kwargs):
    return loads.allocate_contact_load_targets(opposed() if patches is None else patches,
        [0., 0., 0.], kwargs.pop("mass", .349),
        pressures(thumb=.1, index=.1) if readings is None else readings,
        declared_friction_coefficient=.7,
        config=kwargs.pop("config", loads.ContactLoadTargetConfig(
            active_fingers=("thumb", "index"), active_finger_floor_n=.05)), **kwargs)


def test_one_g_opposed_gravity_bound_and_immutable_observation_adapter():
    rows = opposed()
    policy = FingerContactPolicy(("thumb", "index"),
        {"thumb": ("left_thumb_IP",), "index": ("left_index_DIP",)})
    observed = policy.validate_contacts(rows, pressures(thumb=.1, index=.1),
        controlled_joint_names=policy.controlled_joint_names, timestamp_s=2.)
    inputs = observed.as_allocator_patches()
    saved = copy.deepcopy(inputs)
    result = solve(inputs, dict(observed.measured_pressures_n), observation_timestamp_s=2., now_s=2.)
    assert result["feasible"] and inputs == saved and result["freshness_checked"]
    assert result["checked_gravity_margins"] == [1.]
    expected = .349 * 9.81 / (.7 * .85)
    assert sum(result["targets_n"].values()) == pytest.approx(expected, abs=1e-7)
    assert result["targets_n"]["thumb"] == pytest.approx(expected / 2.)
    np.testing.assert_allclose(np.sum(result["allocated_patch_forces_world_n"], axis=0),
        [0., 0., .349 * 9.81], atol=1e-7)
    assert result["task_success"] is None and not result["force_invariant_proven"]
    assert not result["measured_wrench_equilibrium_checked"]


def test_current_group_ratio_preserves_raw_indices_and_rebuilds_geometry():
    rows = [patch(21, "thumb", [-.04, 0., 0.], [1., 0., 0.], .15),
        patch(8, "index", [.04, 0., 0.], [-1., 0., 0.], .2),
        patch(12, "thumb", [-.04, 0., 0.], [1., 0., 0.], .05)]
    result = solve(rows, pressures(thumb=.2, index=.2))
    assert result["feasible"]
    targets = {p["raw_input_index"]: p["target_normal_n"] for p in result["patches"]}
    assert list(targets) == [21, 8, 12]
    assert targets[21] / targets[12] == pytest.approx(3.)
    assert len(result["groups"]) == 2
    rows[2]["hand_body_name"] = "left_thumb_other_link"
    changed = solve(rows, pressures(thumb=.2, index=.2))
    assert changed["feasible"] and len(changed["groups"]) == 3


def test_declared_three_finger_policy_and_margin_are_actual_constraints():
    rows = [patch(i, finger, point, [0., 0., 1.], 1.) for i, (finger, point) in enumerate([
        ("thumb", [-.03, -.03, 0.]), ("middle", [.03, -.03, 0.]), ("ring", [0., .06, 0.])])]
    readings = pressures(thumb=1., middle=1., ring=1.)
    result = solve(rows, readings, mass=1.3, config=loads.ContactLoadTargetConfig())
    assert result["feasible"] and result["checked_gravity_margins"] == [1.]
    assert all(.75 - 1e-7 <= result["targets_n"][f] <= 6. + 1e-7 for f in ("thumb", "middle", "ring"))
    high = solve(rows, readings, mass=1.3,
        config=loads.ContactLoadTargetConfig(gravity_margin=1.5))
    assert not high["feasible"] and high["reason"] == "STATIC_GROUP_LP_FAILED"
    assert "targets_n" not in high
    assert solve()["feasible"]
    assert solve(config=loads.ContactLoadTargetConfig())["reason"] in (
        "UNCONTROLLED_CONTACT", "MISSING_ACTIVE_CONTACT")


@pytest.mark.parametrize("change", [
    lambda p, r: p[0].update(input_index=30),
    lambda p, r: p[0].update(hand_body_name="left_middle_tip"),
    lambda p, r: p[0].update(normal_on_object_world=[0., 0., 0.]),
    lambda p, r: p[0].update(position_world_m=[0., np.nan, 0.]),
    lambda p, r: p[0].update(friction_coefficient=-1.),
    lambda p, r: r.update(thumb=.01),
    lambda p, r: r.pop("palm"),
])
def test_invalid_contract_returns_no_fallback_targets(change):
    rows, readings = opposed(), pressures(thumb=.1, index=.1)
    change(rows, readings)
    result = solve(rows, readings)
    assert not result["feasible"] and result["reason"] == "INVALID_INPUT"
    assert "targets_n" not in result


def test_missing_weak_uncontrolled_and_stale_contacts_remain_explicit():
    rows = opposed()
    rows[0]["measured_pressure_n"] = .049
    result = solve(rows)
    assert result["reason"] == "MISSING_ACTIVE_CONTACT" and result["excluded_raw_input_indices"] == [10]
    assert solve(readings=pressures(thumb=.1, index=.1, palm=.1))["reason"] == "UNCONTROLLED_CONTACT"
    assert solve(observation_timestamp_s=1., now_s=2.)["reason"] == "STALE_OR_FUTURE_OBSERVATION"
    assert solve(observation_timestamp_s=2., now_s=1.)["reason"] == "STALE_OR_FUTURE_OBSERVATION"
    assert solve(now_s=2.)["reason"] == "INVALID_INPUT"
    assert solve()["freshness_checked"] is False


def test_no_force_clipping_or_measured_force_safety_claim():
    rows = opposed()
    for row in rows:
        row["measured_pressure_n"] = 20.
    result = solve(rows, pressures(thumb=20., index=20.))
    assert result["feasible"]
    assert all(value <= 6. + 1e-7 for value in result["targets_n"].values())
    assert [p["measured_pressure_n"] for p in result["patches"]] == [20., 20.]
    assert not solve(mass=10.)["feasible"]
    top = opposed()
    for row in top:
        row["normal_on_object_world"] = [0., 0., -1.]
    assert not solve(top)["feasible"]
    with pytest.raises(ValueError, match="6 N"):
        loads.ContactLoadTargetConfig(normal_target_limit_n=12.)


def test_solver_success_cannot_hide_bad_wrench_or_timeout(monkeypatch):
    original = loads.linprog
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x[:32] *= 1.1
        return result
    monkeypatch.setattr(loads, "linprog", corrupt)
    result = solve()
    assert result["reason"] == "STATIC_GROUP_LP_RESIDUAL" and "targets_n" not in result
    def timeout(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success, result.status = False, 1
        return result
    monkeypatch.setattr(loads, "linprog", timeout)
    assert solve()["reason"] == "STATIC_GROUP_LP_FAILED"


def test_previous_reference_is_explicit_and_not_a_patch_identity_cache():
    previous = pressures(thumb=4., index=4.)
    result = solve(previous_targets_n=previous)
    assert result["feasible"]
    assert result["reference_n"] == pytest.approx({"thumb": 3.025, "index": 3.025})
    assert solve(previous_targets_n=pressures(thumb=8., index=4.))["reason"] == "INVALID_INPUT"
    with pytest.raises(ValueError):
        replace(loads.ContactLoadTargetConfig(), gravity_margin=.9)
