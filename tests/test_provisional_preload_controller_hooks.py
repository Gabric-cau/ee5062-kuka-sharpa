"""Provisional policy source/control contracts, without physical stepping."""

import ast
from collections import deque
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import provisional_preload_controller_hooks as provisional
from test_public_ycb_bias_budget_r17 import PROFILE, real_model
from test_shared_normal_controller_hooks import stub_hand

hooks, r14, r16, r17 = provisional.hooks, provisional.r14, provisional.r16, provisional.r17


def preload_hand():
    hand = object.__new__(provisional.ProvisionalPreloadBudgetHand)
    hand._additional_bias_limit_rad = .10
    hand.preload_ready = False
    hand.preload_start_s = hand.preload_anchor_targets = hand.preload_previous_sample = None
    hand.preload_events, hand.probe_events = [], []
    hand.phase, hand.joint_mode = "CLOSE", "JOINT_CONTROL"
    hand.probe_eligible_seconds = hand.probe_confirmed_seconds = .25
    hand.bias_recovery_active = hand.bias_failure_latched = hand.probe_active = False
    hand.bias_abort_reason = None
    hand.pressure_window = deque([(0., {})])
    hand.estimated = SimpleNamespace(payload_confirmed=False)
    original = {"preprobe_reasons": [], "sampled_solve_current": True,
        "cached_measurements_match": True, "solve_age_s": 0.,
        "sampled_fixed_normal_1g_feasible": False}
    target, groups, sample = {"feasible": True}, {"thumb:DP:0": {"target_n": 3., "measured_n": 2.8}}, []
    hand.current_preload_observation = lambda data: (original, target, groups, sample)
    return hand, original, target, groups, sample


def test_exact_ast_change_is_one_r14_if_and_one_r16_parent_call_only():
    r14_tree = copy.deepcopy(provisional._R14_ORIGINAL_AST)
    expected_if = next(node for node in r14_tree.body if ast.dump(node) == ast.dump(provisional._DEFERRED_IF))
    r14_tree.body.remove(expected_if)
    assert ast.dump(r14_tree) == ast.dump(provisional._R14_PROVISIONAL_AST)
    r16_tree = copy.deepcopy(provisional._R16_PROVISIONAL_AST)
    r16_tree.body[0].value = copy.deepcopy(provisional._R16_PARENT_CALL)
    assert ast.dump(r16_tree) == ast.dump(provisional._R16_ORIGINAL_AST)
    assert [ast.dump(node) for node in provisional._R16_PROVISIONAL_AST.body[1:]] == [
        ast.dump(node) for node in provisional._R16_ORIGINAL_AST.body[1:]]


@pytest.mark.parametrize("which", ["r14", "r16"])
def test_any_additional_source_ast_change_fails_closed(which):
    tree = copy.deepcopy(getattr(provisional, "_" + which.upper() + "_ORIGINAL_AST"))
    tree.body.append(ast.parse("self.preload_ready = True").body[0])
    with pytest.raises(RuntimeError, match="AST changed"):
        getattr(provisional, "_transform_" + which)(tree)


def test_false_actual_requires_full_125_intervals_and_preserves_snapshot_and_same_tick_time():
    hand, original, *_ = preload_hand()
    before = copy.deepcopy(original)
    for step in range(126):
        row = hand.update_preload(SimpleNamespace(time=1. + step * .002))
        assert row["ready"] is (step == 125)
        assert not row["original_probe_snapshot"]["sampled_fixed_normal_1g_feasible"]
        assert provisional.DEFERRED_REASON not in row["reasons"]
        assert row["original_probe_snapshot"] is original
    recheck = hand.update_preload(SimpleNamespace(time=1.25), admission=True)
    assert recheck["ready"] and recheck["continuous_seconds"] == pytest.approx(.25)
    assert recheck["admission_recheck"] and original == before
    assert recheck["provisional_preload_policy"]["sampled_fixed_normal_1g_feasible_observed"] is False
    assert recheck["provisional_preload_policy"]["actual_fixed_normal_1g_required_for_confirmation"]
    json.dumps(hand.preload_events, allow_nan=False)


@pytest.mark.parametrize("actual_feasible", [False, True])
def test_original_policy_comparison_changes_only_false_actual_premotion_gate(actual_feasible):
    hand, original, *_ = preload_hand()
    baseline, baseline_original, *_ = preload_hand()
    baseline.__class__ = hooks.SharedNormalBudgetHand
    original["sampled_fixed_normal_1g_feasible"] = actual_feasible
    baseline_original["sampled_fixed_normal_1g_feasible"] = actual_feasible
    for time in (1., 1.248, 1.25):
        current = hand.update_preload(SimpleNamespace(time=time))
        previous = baseline.update_preload(SimpleNamespace(time=time))
        if actual_feasible:
            assert {key: value for key, value in current.items() if key != "provisional_preload_policy"} == previous
        else:
            assert previous["reasons"] == [provisional.DEFERRED_REASON]
            assert not previous["ready"] and current["reasons"] == []
    assert current["ready"] and previous["ready"] is actual_feasible


@pytest.mark.parametrize("reason", ["PRESSURE", "PALM", "EMERGENCY", "VISUAL_STALE", "CONTACT_ASSOCIATION",
    "SOLVE_STALE_OR_CHANGED_CONTACT", "CONTACT_LOSS", "GEOMETRY_1_5G_INFEASIBLE",
    "CAPACITY_1G_INFEASIBLE", "PRESSURE_NOT_STABLE", "SPEED"])
def test_all_original_preprobe_reasons_still_block_and_clear(reason):
    hand, original, *_ = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    original["preprobe_reasons"].append(reason)
    row = hand.update_preload(SimpleNamespace(time=1.25))
    assert not row["ready"] and reason in row["reasons"]
    assert hand.preload_start_s is hand.preload_anchor_targets is None


@pytest.mark.parametrize("changed", ["eligibility", "not_current", "not_matching", "old_solve", "no_solve_age",
    "target", "empty_groups", "mode", "group_error", "contact_sample"])
def test_every_remaining_r14_gate_still_blocks(changed, monkeypatch):
    hand, original, target, groups, sample = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    if changed == "eligibility":
        hand.probe_eligible_seconds = .248
    elif changed == "not_current":
        original["sampled_solve_current"] = False
    elif changed == "not_matching":
        original["cached_measurements_match"] = False
    elif changed == "old_solve":
        original["solve_age_s"] = .002
    elif changed == "no_solve_age":
        original["solve_age_s"] = None
    elif changed == "target":
        target["feasible"] = False
    elif changed == "empty_groups":
        groups.clear()
    elif changed == "mode":
        hand.joint_mode = "HOLD_INVALID"
    elif changed == "group_error":
        next(iter(groups.values()))["measured_n"] = 2.64
    else:
        hand.preload_previous_sample = ["different"]
        monkeypatch.setattr(r16.r13.r11.r9.r8, "contact_samples_match", lambda *a: False)
    row = hand.update_preload(SimpleNamespace(time=1.25))
    assert not row["ready"] and row["reasons"]
    assert hand.preload_start_s is hand.preload_anchor_targets is None


@pytest.mark.parametrize("change", ["drift", "identity"])
def test_anchor_reset_restarts_on_same_tick_without_creating_readiness(change):
    hand, _, _, groups, _ = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    if change == "drift":
        groups["thumb:DP:0"] = {"target_n": 3.36, "measured_n": 3.36}
    else:
        groups["thumb:other:1"] = groups.pop("thumb:DP:0")
    row = hand.update_preload(SimpleNamespace(time=1.25))
    assert not row["ready"] and row["continuous_seconds"] == 0.
    assert row["window_reset_reason"] == ("TARGET_MOVED_FROM_WINDOW_ORIGIN" if change == "drift" else "GROUP_IDENTITY_CHANGED")
    assert hand.preload_start_s == 1.25
    assert not hand.update_preload(SimpleNamespace(time=1.498))["ready"]
    assert hand.update_preload(SimpleNamespace(time=1.5))["ready"]


@pytest.mark.parametrize("flag", ["bias_recovery_active", "bias_failure_latched"])
def test_original_r16_recovery_wrapper_inhibits_clears_and_blocks_decisive_admission(flag):
    hand, original, *_ = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    setattr(hand, flag, True)
    row = hand.update_preload(SimpleNamespace(time=1.25))
    assert row["r16_recovery_inhibited"] and not row["ready"]
    assert row["reasons"] == ["R16_BIAS_RECOVERY_OR_FAILURE"]
    assert row["continuous_seconds"] == 0. and row["window_anchor_targets_n"] is None
    assert hand.preload_start_s is hand.preload_anchor_targets is hand.preload_previous_sample is None
    assert hand.probe_eligible_seconds == hand.probe_confirmed_seconds == 0. and not hand.pressure_window
    assert not hand.admit_probe(SimpleNamespace(time=1.252))
    assert hand.probe_events[-1]["r16_recovery_inhibited"]
    assert original["sampled_fixed_normal_1g_feasible"] is False


def test_confirmation_still_rejects_false_actual_after_provisional_ready():
    hand, original, *_ = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    assert hand.update_preload(SimpleNamespace(time=1.25))["ready"]
    original.update(motion_safety_reasons=[], actual_probe_endpoint_reached=True, table_unloaded=True,
        accepted_visual_after_probe_start=True, associated_active_fingers=["thumb", "middle", "ring"])
    hand.probe_active, hand.probe_confirmed_seconds = True, 1.
    hand.actual_wrist_speed_m_s = 0.
    hand.probe_required_fingers = {"thumb", "middle", "ring"}
    hand.refresh_probe_solve = lambda data, force: {}
    hand._snapshot_probe = lambda data, readings: original
    assert not hand.confirm_probe(SimpleNamespace(time=1.252))
    assert hand.probe_events[-1]["confirmation"] is False and not hand.estimated.payload_confirmed


def test_original_decisive_admission_still_refreshes_and_can_reject_new_reason():
    hand, original, *_ = preload_hand()
    hand.update_preload(SimpleNamespace(time=1.))
    assert hand.update_preload(SimpleNamespace(time=1.25))["ready"]
    calls = []
    hand.refresh_probe_solve = lambda data, force: calls.append(force) or {}
    fresh = {**original, "preprobe_reasons": ["SPEED"]}
    hand._snapshot_probe = lambda data, readings: fresh
    assert not hand.admit_probe(SimpleNamespace(time=1.25))
    assert calls == [True] and hand.preload_events[-1]["admission_recheck"]
    assert hand.probe_events[-1]["decisive_current_solve"] and not hand.probe_events[-1]["admission"]
    assert not hand.probe_active


def test_shared_allocator_bindings_and_all_downstream_methods_are_identical():
    for name in ("write", "regulate", "current_preload_observation", "after_step", "admit_probe", "confirm_probe",
        "held_revalidated", "abort_probe", "update_pressure_relief", "angles", "_joint_rows", "phase_begin"):
        assert getattr(provisional.ProvisionalPreloadBudgetHand, name) is getattr(hooks.SharedNormalBudgetHand, name)
    assert provisional._factory.__code__ is r17.controller_for.__code__
    assert provisional._factory.__closure__ is r17.controller_for.__closure__
    assert provisional._factory.__globals__["BiasBudgetHand"] is provisional.ProvisionalPreloadBudgetHand
    assert hooks.SharedNormalBudgetHand.regulate.__globals__["allocate_tracking_targets"] is hooks.allocate_tracking_targets
    assert hooks._preload_r13.allocate_tracking_targets is hooks.allocate_tracking_targets


def test_both_current_shared_allocator_paths_execute_on_new_subclass(monkeypatch):
    hand, data, readings, _ = stub_hand(monkeypatch)
    hand.__class__ = provisional.ProvisionalPreloadBudgetHand
    hand.regulate(readings, .002, False)
    assert hand.joint_target["format"] == "shared_normal_tracking_target_v1"
    _, target, groups, _ = hand.current_preload_observation(data)
    assert target["format"] == "shared_normal_tracking_target_v1" and groups


@pytest.mark.parametrize("budget", [.02, .04, .10])
def test_real_factory_first_retry_budget_and_original_globals_unchanged(real_model, tmp_path, budget):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    methods = (r14.ConstrainedPreloadHand.update_preload, r16.BiasRecoveryHand.update_preload)
    globals_before = [dict(method.__globals__) for method in methods]
    params = copy.deepcopy((r14.PRELOAD_PARAMETERS, r16.r13.JOINT_PARAMETERS, r14.PROBE_PARAMETERS))
    guard = r16.r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    task, source, hands, estimated = provisional.controller_for("ycb_can", tmp_path, [], guard, profile, .349,
        additional_bias_limit_rad=budget)
    estimated.bind(kin)
    config = r16.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    first = task.__globals__["HandGraspController"](real_model, config)
    retry = task.__globals__["HandGraspController"](real_model, config)
    assert hands == [first, retry] and first is not retry
    assert all(type(item) is provisional.ProvisionalPreloadBudgetHand and item.additional_bias_limit_rad == budget for item in hands)
    original_task, original_source, *_ = hooks.controller_for("ycb_can", tmp_path, [], guard, profile, .349,
        additional_bias_limit_rad=budget)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(original_source))
    assert task.__globals__["_phase_seconds"] == original_task.__globals__["_phase_seconds"]
    assert params == (r14.PRELOAD_PARAMETERS, r16.r13.JOINT_PARAMETERS, r14.PROBE_PARAMETERS)
    for method, before in zip(methods, globals_before):
        assert method.__globals__.keys() == before.keys()
        assert all(method.__globals__[key] is value for key, value in before.items())


def test_factory_requires_explicit_supported_budget(tmp_path):
    args = ("ycb_can", tmp_path, [], SimpleNamespace(), {}, .349)
    with pytest.raises(TypeError, match="additional_bias_limit_rad"):
        provisional.controller_for(*args)
    with pytest.raises(ValueError):
        provisional.controller_for(*args, additional_bias_limit_rad=.03)
