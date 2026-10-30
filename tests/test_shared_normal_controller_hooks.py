"""Hook execution uses observation/Jacobian stubs, not dynamic simulation evidence."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shared_normal_controller_hooks as hooks
import shared_normal_tracking_targets as shared
from test_public_ycb_bias_budget_r17 import PROFILE, real_model
from test_shared_normal_tracking_targets import observations

r17, r13, r14 = hooks.r17, hooks.r13, hooks.r14


def test_same_code_and_super_closure_with_only_explicit_global_substitutions():
    pairs = [(hooks.SharedNormalBudgetHand.regulate, r13.JointAdmittanceProbeHand.regulate,
        {"allocate_tracking_targets": shared.allocate_tracking_targets}),
        (hooks.SharedNormalBudgetHand.current_preload_observation,
            r14.ConstrainedPreloadHand.current_preload_observation, {"r13": hooks._preload_r13}),
        (hooks._factory, r17.controller_for, {"BiasBudgetHand": hooks.SharedNormalBudgetHand})]
    for isolated, original, replacements in pairs:
        assert isolated.__code__ is original.__code__
        assert isolated.__closure__ is original.__closure__
        assert isolated.__defaults__ is original.__defaults__
        assert isolated.__globals__ is not original.__globals__
        assert isolated.__globals__.keys() == original.__globals__.keys()
        for key, value in original.__globals__.items():
            assert isolated.__globals__[key] is replacements.get(key, value)
    closure = hooks.SharedNormalBudgetHand.regulate.__closure__
    assert closure and closure[0].cell_contents is r13.JointAdmittanceProbeHand
    assert vars(hooks._preload_r13).keys() == vars(r13).keys()
    for key, value in vars(r13).items():
        assert getattr(hooks._preload_r13, key) is (
            shared.allocate_tracking_targets if key == "allocate_tracking_targets" else value)
    assert r13.allocate_tracking_targets is not shared.allocate_tracking_targets


def stub_hand(monkeypatch, *, accepted=True, pressure_override=None):
    patches, com, mass, pressures = observations()
    if pressure_override:
        pressures.update(pressure_override)
    raw = [dict(patch) for patch in patches]
    readings = {finger: SimpleNamespace(normal=pressures.get(finger, 0.)) for finger in shared.allocator.CHANNELS}
    calls = []
    hand = object.__new__(hooks.SharedNormalBudgetHand)
    data = SimpleNamespace(time=12.)
    hand.model = SimpleNamespace(nv=3)
    hand.reader = SimpleNamespace(last_data=data, raw_contacts=raw, read=lambda data: readings)
    hand.estimated = SimpleNamespace(payload_confirmed=False, mass_kg=mass, surface=object(),
        pose_timestamp_s=11.9, estimated_pose=lambda data, carrying: {"estimated": True})
    hand.emergency_active = hand.emergency_phase_hold = False
    hand.joint_mode = "JOINT_CONTROL"
    hand.joint_activation_s = 6.
    hand.previous_feasible_targets = {"thumb": 2., "middle": 1., "ring": 1.}
    hand.previous_tracking_sample = None
    hand.contact_sample = lambda raw: tuple(patch["input_index"] for patch in raw)
    hand._mode = lambda mode, data, reason: setattr(hand, "joint_mode", mode)
    hand.flexion_joints = lambda: {finger: (finger, 1.) for finger in shared.ACTIVE}
    hand.joint_dofs = {finger: index for index, finger in enumerate(shared.ACTIVE)}
    hand.refresh_probe_solve = lambda data, force: calls.append(("refresh", force)) or readings
    hand._snapshot_probe = lambda data, readings: {"pressures_n": dict(pressures), "original_probe_marker": True}

    def association(raw_contacts, pose, surface, **kwargs):
        assert raw_contacts is raw and surface is hand.estimated.surface
        assert kwargs == {"now_s": 12., "pose_timestamp_s": 11.9,
            "max_pose_age_s": 1., "max_surface_distance_m": .006}
        calls.append(("association", accepted))
        return {"accepted": accepted, "reason": "TEST_OBSERVED_ASSOCIATION",
            "patches": [dict(patch) for patch in raw], "estimated_com_world_m": list(com)}
    monkeypatch.setattr(r13.r11.r9.r8.r4, "associate_tactile_contacts", association)
    monkeypatch.setattr(r13.r11.BoundedTrackingProbeHand, "regulate",
        lambda self, readings, dt, carrying: calls.append(("original_super_regulate", dt, carrying)))
    def jacobian(model, data, jp, jr, point, body):
        jp[0, :] = .02
    stub_mujoco = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mj_name2id=lambda *args: 1, mj_jac=jacobian)
    monkeypatch.setitem(hooks.SharedNormalBudgetHand.regulate.__globals__, "mujoco", stub_mujoco)
    return hand, data, readings, calls


def test_tracking_and_gate_paths_actually_run_shared_allocator_without_advancing_gate_reference(monkeypatch):
    hand, data, readings, calls = stub_hand(monkeypatch)
    monkeypatch.setattr(r13, "allocate_tracking_targets", lambda *a: pytest.fail("Original target allocator used"))
    hand.regulate(readings, .002, False)
    assert calls[0] == ("original_super_regulate", .002, False)
    assert hand.joint_target["feasible"] and hand.joint_target["format"] == "shared_normal_tracking_target_v1"
    assert hand.joint_target["timestamp_s"] == 12. and hand.joint_mode == "JOINT_CONTROL"
    assert len(hand.joint_solves) == 3 and set(hand.joint_velocities) == set(shared.ACTIVE)
    previous = hand.previous_feasible_targets
    saved, tracking_target = copy.deepcopy(previous), copy.deepcopy(hand.joint_target)
    original, target, groups, sample = hand.current_preload_observation(data)
    assert original["original_probe_marker"]
    assert target["feasible"] and target["format"] == "shared_normal_tracking_target_v1"
    assert len(groups) == 3 and sample == (0, 1, 2)
    assert hand.previous_feasible_targets is previous and previous == saved
    assert hand.joint_target == tracking_target
    assert calls.count(("association", True)) == 2 and ("refresh", True) in calls


@pytest.mark.parametrize("path", ["tracking", "preload"])
def test_both_paths_retain_failed_association_and_never_call_new_allocator(monkeypatch, path):
    hand, data, readings, _ = stub_hand(monkeypatch, accepted=False)
    def fail(*args):
        pytest.fail("Rejected association reached allocator")
    monkeypatch.setitem(hooks.SharedNormalBudgetHand.regulate.__globals__, "allocate_tracking_targets", fail)
    monkeypatch.setattr(hooks._preload_r13, "allocate_tracking_targets", fail)
    saved = copy.deepcopy(hand.previous_feasible_targets)
    if path == "tracking":
        hand.regulate(readings, .002, False)
        assert not hand.joint_target["feasible"] and hand.joint_mode == "HOLD_INVALID"
    else:
        _, target, groups, _ = hand.current_preload_observation(data)
        assert not target["feasible"] and not groups
    assert hand.previous_feasible_targets == saved


@pytest.mark.parametrize("pressure", [{"thumb": .34}, {"thumb": 6.001}, {"palm": .051}])
def test_original_tracking_entry_pressure_gates_still_precede_allocator(monkeypatch, pressure):
    hand, data, readings, _ = stub_hand(monkeypatch, pressure_override=pressure)
    monkeypatch.setitem(hooks.SharedNormalBudgetHand.regulate.__globals__, "allocate_tracking_targets",
        lambda *args: pytest.fail("Entry-pressure gate bypassed"))
    hand.regulate(readings, .002, False)
    assert not hand.joint_target["feasible"] and hand.joint_mode == "HOLD_INVALID"


def test_tracking_raw_contact_channel_identity_still_fails_loudly(monkeypatch):
    hand, _, readings, _ = stub_hand(monkeypatch)
    original = r13.r11.r9.r8.r4.associate_tactile_contacts
    def changed(*args, **kwargs):
        association = original(*args, **kwargs)
        association["patches"][0]["finger"] = "middle"
        return association
    monkeypatch.setattr(r13.r11.r9.r8.r4, "associate_tactile_contacts", changed)
    with pytest.raises(RuntimeError, match="raw contact index"):
        hand.regulate(readings, .002, False)


def test_original_emergency_priority_and_super_cell_do_not_recurse(monkeypatch):
    hand, _, readings, calls = stub_hand(monkeypatch)
    hand.emergency_active = True
    hand.regulate(readings, .002, True)
    assert calls == [("original_super_regulate", .002, True)]
    assert hand.joint_target["reason"] == "ORIGINAL_EMERGENCY_PRECEDENCE"
    assert not hand.joint_velocities and not hand.joint_solves


def test_inherited_write_afterstep_admission_and_budget_bounds_are_same_functions():
    for name in ("write", "after_step", "admit_probe", "update_preload", "angles", "_joint_rows", "phase_begin"):
        assert getattr(hooks.SharedNormalBudgetHand, name) is getattr(r17.BiasBudgetHand, name)


@pytest.mark.parametrize("budget", [.02, .04, .10])
def test_actual_factory_first_and_retry_classes_budgets_and_old_globals_are_unchanged(real_model, tmp_path, budget):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    original = (r17.BiasBudgetHand, r13.allocate_tracking_targets,
        r13.JointAdmittanceProbeHand.regulate, r14.ConstrainedPreloadHand.current_preload_observation)
    parameters = copy.deepcopy((r13.JOINT_PARAMETERS, r14.PRELOAD_PARAMETERS))
    guard = r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    task, _, hands, estimated = hooks.controller_for("ycb_can", tmp_path, [], guard, profile, .349,
        additional_bias_limit_rad=budget)
    estimated.bind(kin)
    config = r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    first = task.__globals__["HandGraspController"](real_model, config)
    second = task.__globals__["HandGraspController"](real_model, config)
    assert hands == [first, second] and first is not second
    assert all(type(hand) is hooks.SharedNormalBudgetHand and hand.additional_bias_limit_rad == budget for hand in hands)
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 16.
    assert original == (r17.BiasBudgetHand, r13.allocate_tracking_targets,
        r13.JointAdmittanceProbeHand.regulate, r14.ConstrainedPreloadHand.current_preload_observation)
    assert parameters == (r13.JOINT_PARAMETERS, r14.PRELOAD_PARAMETERS)


def test_factory_requires_explicit_supported_budget_before_construction(tmp_path):
    args = ("ycb_can", tmp_path, [], SimpleNamespace(), {}, .349)
    with pytest.raises(TypeError, match="additional_bias_limit_rad"):
        hooks.controller_for(*args)
    with pytest.raises(ValueError):
        hooks.controller_for(*args, additional_bias_limit_rad=.03)
