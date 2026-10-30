"""Current contact reconstruction and inherited current-write behavior, without stepping."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fresh_contact_tracking_controller_hooks as fresh
from test_public_ycb_bias_budget_r17 import PROFILE, real_model
from test_public_ycb_bias_recovery_r16 import hand_at_first_direction_failure, lower_writer
from test_shared_normal_controller_hooks import stub_hand

hooks, r13, r16 = fresh.hooks, fresh.r13, fresh.provisional.r16
_CONTACT_SAMPLES_MATCH = r13.r11.r9.r8.contact_samples_match


def observed_hand(monkeypatch, *, accepted=True, changed=True, extra_group=False, pressure_override=None):
    hand, data, readings, calls = stub_hand(monkeypatch, accepted=accepted, pressure_override=pressure_override)
    hand.__class__ = fresh.FreshContactTrackingHand
    hand.previous_tracking_sample = (0, 1, 2)
    hand.initial_command_bias = {"fixed_capture": .1}
    hand._activate = lambda data: pytest.fail("Loaded capture must not reset")
    monkeypatch.setattr(r13.r11.r9.r8, "contact_samples_match", lambda current, previous: not changed)
    if extra_group:
        raw = hand.reader.raw_contacts
        raw[2]["measured_pressure_n"] = .75
        raw.append(dict(raw[2], input_index=3, hand_body_name="ring_MP",
            measured_pressure_n=.25, position_world_m=[.03, .01, .02]))
    jac_calls = []
    body_ids = {"thumb_tip": 1, "middle_tip": 2, "ring_tip": 3, "ring_MP": 4}

    def jacobian(model, current_data, jp, jr, point, body):
        assert current_data is hand.reader.last_data
        jac_calls.append((float(current_data.time), body, point.tolist()))
        jp[0, :] = np.arange(1, model.nv + 1) * .01 * body

    stub_mujoco = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mj_name2id=lambda model, kind, name: body_ids[name], mj_jac=jacobian)
    monkeypatch.setitem(fresh._fresh_regulate.__globals__, "mujoco", stub_mujoco)
    return hand, data, readings, calls, jac_calls


def test_exact_branch_change_recovers_complete_original_method_and_rejects_extra_change():
    restored = fresh.restore_original_ast(fresh._REGULATE_AST)
    assert ast.dump(restored) == ast.dump(fresh._ORIGINAL_AST)
    changed = copy.deepcopy(fresh._REGULATE_AST)
    changed.body.insert(0, ast.parse("self.initial_command_bias = {}").body[0])
    with pytest.raises(RuntimeError, match="more than the declared branch"):
        fresh.restore_original_ast(changed)
    with pytest.raises(RuntimeError, match="AST changed"):
        fresh._transform_regulate(changed)


def test_preserves_original_super_closure_globals_and_every_downstream_method():
    method = fresh._fresh_regulate
    original = r13.JointAdmittanceProbeHand.regulate
    inherited = hooks.SharedNormalBudgetHand.regulate
    assert method.__closure__ is original.__closure__
    assert method.__code__.co_freevars == original.__code__.co_freevars == ("__class__",)
    assert method.__closure__[0].cell_contents is r13.JointAdmittanceProbeHand
    assert method.__globals__ is not inherited.__globals__
    assert method.__globals__.keys() == inherited.__globals__.keys()
    assert all(method.__globals__[name] is value for name, value in inherited.__globals__.items())
    assert method.__globals__["allocate_tracking_targets"] is hooks.allocate_tracking_targets
    for name in ("write", "after_step", "update_preload", "current_preload_observation", "_joint_rows",
        "admit_probe", "confirm_probe", "held_revalidated", "phase_begin", "angles", "_activate"):
        assert getattr(fresh.FreshContactTrackingHand, name) is getattr(fresh.provisional.ProvisionalPreloadBudgetHand, name)


def test_changed_patch_set_rebuilds_current_groups_and_jacobian_dimensions_without_capture(monkeypatch):
    hand, data, readings, calls, jac_calls = observed_hand(monkeypatch, extra_group=True)
    hand.joint_solves = [{"finger": "ring", "compression_jacobian_m_rad": [[999.]]}]
    hand.joint_velocities = {"old_joint": 999.}
    capture, activation = hand.initial_command_bias.copy(), hand.joint_activation_s
    hand.regulate(readings, .002, False)
    assert calls[0] == ("original_super_regulate", .002, False)
    assert hand.joint_target["feasible"] and hand.joint_mode == "JOINT_CONTROL"
    assert hand.target_timestamp_s == data.time == 12.
    assert len(jac_calls) == 4 and all(row[0] == 12. for row in jac_calls)
    assert set(hand.joint_velocities) == set(r13.ACTIVE)
    solves = {row["finger"]: row for row in hand.joint_solves}
    assert np.asarray(solves["ring"]["compression_jacobian_m_rad"]).shape == (2, 1)
    np.testing.assert_allclose(solves["ring"]["compression_jacobian_m_rad"], [[-.12], [-.09]])
    assert [group["raw_input_indices"] for group in solves["ring"]["groups"]] == [[3], [2]]
    assert hand.initial_command_bias == capture and hand.joint_activation_s == activation
    telemetry = hand.joint_target["fresh_contact_tracking"]
    assert telemetry["changed_from_previous_sample"] and telemetry["activated_before_current_solve"]
    assert telemetry["current_contact_sample_count"] == telemetry["rebuilt_group_count"] == 4
    assert telemetry["current_target_raw_input_indices"] == [0, 1, 2, 3]
    assert telemetry["status"] == "CURRENT_CONTACT_JACOBIANS_REBUILT"
    assert telemetry["rebuilt_finger_solves"][-1]["jacobian_shape"] == [2, 1]
    json.dumps(hand.joint_target, allow_nan=False)


def test_unchanged_contacts_match_original_shared_controller_outputs(monkeypatch):
    hand, _, readings, _, _ = observed_hand(monkeypatch, changed=False)
    baseline = copy.copy(hand)
    baseline.__class__ = hooks.SharedNormalBudgetHand
    monkeypatch.setitem(hooks.SharedNormalBudgetHand.regulate.__globals__, "mujoco",
        fresh._fresh_regulate.__globals__["mujoco"])
    hand.regulate(readings, .002, False)
    baseline.regulate(readings, .002, False)
    target = dict(hand.joint_target)
    telemetry = target.pop("fresh_contact_tracking")
    assert not telemetry["changed_from_previous_sample"]
    assert target == baseline.joint_target
    assert hand.joint_velocities == baseline.joint_velocities
    assert hand.joint_solves == baseline.joint_solves
    assert hand.previous_feasible_targets == baseline.previous_feasible_targets


def test_same_patch_count_pressure_change_uses_original_matcher_and_rebuilds(monkeypatch):
    hand, _, readings, _, jac_calls = observed_hand(monkeypatch)
    hand.contact_sample = r13.JointAdmittanceProbeHand.contact_sample
    for patch in hand.reader.raw_contacts:
        patch["pressure_n"] = patch["measured_pressure_n"]
        patch["normal_on_hand_world"] = (-np.asarray(patch["normal_on_object_world"])).tolist()
    hand.previous_tracking_sample = hand.contact_sample(hand.reader.raw_contacts)
    hand.previous_tracking_sample[-1]["pressure_n"] = .1
    current_sample = hand.contact_sample(hand.reader.raw_contacts)
    assert len(current_sample) == len(hand.previous_tracking_sample) == 3
    assert not _CONTACT_SAMPLES_MATCH(current_sample, hand.previous_tracking_sample)
    monkeypatch.setattr(r13.r11.r9.r8, "contact_samples_match", _CONTACT_SAMPLES_MATCH)
    hand.regulate(readings, .002, False)
    assert hand.joint_target["feasible"] and len(jac_calls) == 3
    assert hand.joint_target["fresh_contact_tracking"]["changed_from_previous_sample"]
    assert hand.joint_target["fresh_contact_tracking"]["current_contact_sample_count"] == 3


@pytest.mark.parametrize("failure", ["association", "lp", "thumb_floor", "pressure", "palm", "emergency", "mode"])
def test_failed_current_inputs_still_clear_old_solves_and_reject_without_jacobians(monkeypatch, failure):
    pressures = {"thumb": .34} if failure == "thumb_floor" else (
        {"thumb": 6.001} if failure == "pressure" else ({"palm": .051} if failure == "palm" else None))
    hand, _, readings, _, jac_calls = observed_hand(monkeypatch,
        accepted=failure != "association", pressure_override=pressures)
    previous = copy.deepcopy(hand.previous_feasible_targets)
    hand.joint_solves, hand.joint_velocities = ["OLD"], {"old": 99.}
    if failure == "lp":
        monkeypatch.setitem(fresh._fresh_regulate.__globals__, "allocate_tracking_targets",
            lambda *args: {"feasible": False, "reason": "SHARED_NORMAL_TRACKING_LP_FAILED"})
    else:
        monkeypatch.setitem(fresh._fresh_regulate.__globals__, "allocate_tracking_targets",
            lambda *args: pytest.fail("Rejected input reached the target allocator"))
    hand.emergency_active = failure == "emergency"
    if failure == "mode":
        hand.joint_mode = "RELEASING"
    hand.regulate(readings, .002, False)
    assert not hand.joint_target["feasible"] and not jac_calls
    assert hand.joint_solves == [] and hand.joint_velocities == {}
    assert hand.previous_feasible_targets == previous
    assert "fresh_contact_tracking" not in hand.joint_target


@pytest.mark.parametrize("qp_failure", [False, True])
def test_changed_contacts_reach_inherited_current_box_qp_and_failure_still_aborts(lower_writer, monkeypatch, qp_failure):
    observed, _, readings, _, jac_calls = observed_hand(monkeypatch, extra_group=True)
    hand, data = hand_at_first_direction_failure()
    hand.__class__ = fresh.FreshContactTrackingHand
    hand._additional_bias_limit_rad = .10
    hand.model.nv = 8
    hand.joint_dofs = dict(zip(hand.joint_references, range(8), strict=True))
    hand.reader, hand.estimated = observed.reader, observed.estimated
    hand.reader.last_data = data
    data.time, hand.last_joint_write_s = 12., 11.998
    hand.previous_tracking_sample = observed.previous_tracking_sample
    hand.contact_sample = observed.contact_sample
    hand.previous_feasible_targets = {"thumb": 3., "middle": 1.5, "ring": 1.5}
    hand.probe_active = True
    # The observation association closes over this same raw array and surface.
    observed.reader.last_data = data
    capture, qpos = hand.initial_command_bias.copy(), data.qpos.copy()
    real_qp, qp_calls = r16.r14.constrained_joint_velocity, []

    def current_qp(matrix, desired, low, high):
        qp_calls.append((copy.deepcopy(matrix), list(low), list(high)))
        if qp_failure:
            return np.zeros(len(low)), {"accepted": False, "reason": "TEST_CURRENT_DIRECTION_INFEASIBLE"}
        return real_qp(matrix, desired, low, high)

    monkeypatch.setattr(r16.r14, "constrained_joint_velocity", current_qp)
    hand.regulate(readings, .002, False)
    assert len(jac_calls) == 4 and hand.joint_target["feasible"]
    hand.write(data)
    assert len(qp_calls) == 3
    assert [np.asarray(row[0]).shape for row in qp_calls] == [(1, 2), (1, 3), (2, 3)]
    event = hand.bias_recovery_events[-1]
    assert event["fresh_target"] and event["write_verification_passed"]
    assert all(joint["applied_within_original_box"] for finger in event["per_finger"] for joint in finger["joints"].values())
    assert hand.initial_command_bias == capture
    np.testing.assert_equal(data.qpos, qpos)
    telemetry = hand.fresh_contact_telemetry()
    assert telemetry["accepted_reconstruction_count"] == telemetry["changed_current_sample_count"] == 1
    assert telemetry["events"][0]["timestamp_s"] == 12.
    json.dumps(telemetry, allow_nan=False)
    if qp_failure:
        assert hand.bias_recovery_active and hand.bias_abort_reason == "PROBE_BIAS_RECOVERY"
        assert all(row["mode"] == "BIAS_RECOVERY" for row in event["per_finger"])
        monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "after_step", lambda *args: None)
        data.time += .002
        assert hand.after_step(SimpleNamespace(value="TRIAL_LIFT"), data, .002) == "PROBE_BIAS_RECOVERY"
    else:
        assert not hand.bias_recovery_active and hand.bias_abort_reason is None
        assert all(row["mode"] == "NOMINAL" for row in event["per_finger"])
        assert all(row["nominal_direction_satisfied"] for row in event["per_finger"])


def test_factory_constructs_every_retry_with_new_class_without_mutating_old_globals(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    originals = (r13.JointAdmittanceProbeHand.regulate, hooks.SharedNormalBudgetHand.regulate,
        fresh.provisional.ProvisionalPreloadBudgetHand.regulate, fresh.r17.controller_for)
    before = [dict(method.__globals__) for method in originals]
    guard = r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    task, _, hands, estimated = fresh.controller_for("ycb_can", tmp_path, [], guard, profile, .349,
        additional_bias_limit_rad=.10)
    estimated.bind(kin)
    config = r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    first = task.__globals__["HandGraspController"](real_model, config)
    retry = task.__globals__["HandGraspController"](real_model, config)
    assert hands == [first, retry] and first is not retry
    assert all(type(hand) is fresh.FreshContactTrackingHand and hand.additional_bias_limit_rad == .10 for hand in hands)
    for method, globals_before in zip(originals, before):
        assert method.__globals__.keys() == globals_before.keys()
        assert all(method.__globals__[key] is value for key, value in globals_before.items())
    with pytest.raises(TypeError, match="additional_bias_limit_rad"):
        fresh.controller_for("ycb_can", tmp_path, [], guard, profile, .349)
    with pytest.raises(ValueError):
        fresh.controller_for("ycb_can", tmp_path, [], guard, profile, .349, additional_bias_limit_rad=.03)


def test_existing_r20_r21_runtime_bindings_inspect_real_wrapper_without_source_stubs():
    import develop_public_ycb_shared_normal_r20 as r20
    import develop_public_ycb_provisional_preload_r21 as r21
    hand = object.__new__(fresh.FreshContactTrackingHand)
    hand._additional_bias_limit_rad = .10
    for bound in (r20.runtime_binding(hand), r21.runtime_binding(hand)):
        tracking = bound["tracking"]
        assert tracking["method_source_file"] == "scripts/fresh_contact_tracking_controller_hooks.py"
        assert tracking["allocator_is_shared_normal"]
        assert not tracking["same_code_as_original"]
        assert not tracking["same_closure_as_original"]
        assert not tracking["other_globals_match_except_warning_registry"]
        assert bound["gate_only"]["same_code_as_original"]
        assert bound["writer_is_frozen_r16"] and bound["after_step_is_frozen_r16"]
        assert bound["admit_probe_is_frozen_r16"]
        json.dumps(bound, allow_nan=False)
    bound = r21.runtime_binding(hand)
    assert not bound["unchanged_methods"]["regulate"]
    assert all(value for name, value in bound["unchanged_methods"].items() if name != "regulate")
    proof = fresh.tracking_method_binding()
    assert proof["wrapper_source_file_sha256"] == bound["tracking"]["method_source_file_sha256"]
    assert proof["wrapper_method_source_sha256"] == bound["tracking"]["method_source_sha256"]
    assert proof["wrapper_is_thin_delegate"]
    assert proof["original_method_ast_sha256"] == proof["restored_method_ast_sha256"]
    assert proof["transformed_method_ast_sha256"] != proof["original_method_ast_sha256"]
    assert proof["inner_same_closure_as_original"] and proof["inner_original_class_cell_retained"]
    assert proof["inner_other_globals_match_except_allocator_and_warning_registry"]
    assert proof["inner_allocator_is_shared_normal"] and proof["wrapper_allocator_metadata_matches_inner"]
