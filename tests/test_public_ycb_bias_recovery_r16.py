"""R16 write behavior and gates; no full-task success is inferred."""

import ast
from collections import deque
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_bias_recovery_r16 as r16

PROFILE = r16.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def hand_at_first_direction_failure():
    hand = object.__new__(r16.BiasRecoveryHand)
    names = ["left_thumb_MCP_FE", "left_thumb_IP", "left_middle_MCP_FE", "left_middle_PIP",
        "left_middle_DIP", "left_ring_MCP_FE", "left_ring_PIP", "left_ring_DIP"]
    refs = np.array([.2124186585616599, .4802883606021198, .3, .4, .2, .3, .4, .2])
    bias = np.array([.013632440871981885, .021034091189615967, *([.02] * 6)])
    actual = refs - bias
    actual[:2] = [.17875539727254558, .43931480364190806]
    hand.model = SimpleNamespace(actuator_trnid=np.array([[i, 0] for i in range(8)]),
        jnt_qposadr=np.arange(8), jnt_range=np.array([[-.5236, 1.3963], [0., 1.7453],
            *([[0., 1.7]] * 6)]))
    hand.actuators = dict(zip(names, range(8), strict=True))
    hand.config = SimpleNamespace(close_rate=.12, relief_rate=.6,
        thumb_mcp_flexion_ratio=.4, thumb_ip_flexion_ratio=1.2)
    hand.profile = {"contact_closure": {"thumb": .3, "middle": .3, "ring": .3}}
    hand.joint_references = dict(zip(names, refs.tolist(), strict=True))
    hand.initial_command_bias = dict(zip(names, bias.tolist(), strict=True))
    hand.joint_activation_s = 6.154
    hand.last_joint_write_s = 14.920
    hand.target_timestamp_s = 14.922
    hand.joint_target = {"feasible": True, "reason": "STATIC_TRACKING_TARGET_ALLOCATED"}
    hand.joint_mode = "JOINT_CONTROL"
    hand.joint_velocities = dict.fromkeys(names, 0.)
    hand.joint_solves = [
        {"finger": "thumb", "joint_names": names[:2],
            "compression_jacobian_m_rad": [[.05697115184280521, .020116670376867273]],
            "desired_normal_velocity_m_s": [.0006393229852468578]},
        {"finger": "middle", "joint_names": names[2:5],
            "compression_jacobian_m_rad": [[.05, .02, .01]], "desired_normal_velocity_m_s": [.0001]},
        {"finger": "ring", "joint_names": names[5:],
            "compression_jacobian_m_rad": [[.05, .02, .01]], "desired_normal_velocity_m_s": [.0001]}]
    hand.qp_events = []
    hand.joint_events = []
    hand.bias_recovery_events = []
    hand.bias_recovery_active = hand.bias_failure_latched = False
    hand.bias_abort_reason = None
    hand.preload_events = []
    hand.probe_events = []
    hand.phase = "CLOSE"
    hand.preload_ready = True
    hand.preload_start_s = 14.
    hand.preload_anchor_targets = {"g": 3.}
    hand.preload_previous_sample = None
    hand.probe_eligible_seconds = hand.probe_confirmed_seconds = .5
    hand.pressure_window = deque([(14.9, {})])
    hand.probe_active = False
    hand.estimated = SimpleNamespace(payload_confirmed=False, probe_active=False)
    hand.emergency_active = hand.emergency_phase_hold = False
    hand.reader = SimpleNamespace(read=lambda data: {})
    hand.update_pressure_relief = lambda data, readings: None
    return hand, SimpleNamespace(time=14.922, qpos=actual.copy(), ctrl=refs.copy())


@pytest.fixture
def lower_writer(monkeypatch):
    monkeypatch.setattr(r16.r13.r11.BoundedTrackingProbeHand, "angles", lambda self: self.joint_references.copy())
    def write(self, data):
        for name, value in self.angles().items():
            data.ctrl[self.actuators[name]] = value
    monkeypatch.setattr(r16.r13.r11.BoundedTrackingProbeHand, "write", write)
    return write


def test_saved_first_failure_has_nonempty_box_and_a_real_alternative_command():
    box = r16.command_box(.2124186585616599, .17875539727254558, .013632440871981885,
        [-.5236, 1.3963], .4, .002, .12, .6)
    assert box["feasible"]
    projected = r16.minimum_change_projection(.2124186585616599, box["intersection_rad"])
    assert projected == pytest.approx(.21238783814452747, abs=1e-14)
    assert projected - .2124186585616599 == pytest.approx(-3.082041713242667e-5)
    with pytest.raises(ValueError):
        r16.minimum_change_projection(.2, [.3, .1])


@pytest.mark.parametrize("actual,dt", [(float("nan"), .002), (.2, 0.), (.2, -.002)])
def test_invalid_box_inputs_are_rejected(actual, dt):
    with pytest.raises(ValueError):
        r16.command_box(.2, actual, .01, [0., 1.], 1., dt, .12, .6)


def test_one_failed_finger_recovers_while_other_fingers_really_move(lower_writer):
    hand, data = hand_at_first_direction_failure()
    before, qpos, capture = data.ctrl.copy(), data.qpos.copy(), hand.initial_command_bias.copy()
    hand.write(data)
    event = hand.bias_recovery_events[-1]
    modes = {row["finger"]: row["mode"] for row in event["per_finger"]}
    assert modes == {"thumb": "BIAS_RECOVERY", "middle": "NOMINAL", "ring": "NOMINAL"}
    assert data.ctrl[0] < before[0]
    assert data.ctrl[2] > before[2] and data.ctrl[5] > before[5]
    assert event["per_finger"][0]["nominal_qp"]["accepted"] is False
    assert event["per_finger"][0]["nominal_direction_satisfied"] is None
    assert event["write_verification_passed"]
    assert all(j["applied_within_original_box"] for f in event["per_finger"] for j in f["joints"].values())
    assert hand.initial_command_bias == capture
    np.testing.assert_equal(data.qpos, qpos)
    assert not hand.preload_ready and hand.preload_start_s is None
    assert hand.probe_eligible_seconds == hand.probe_confirmed_seconds == 0.
    assert not hand.pressure_window
    assert hand.joint_events[-1]["mode"] == "BIAS_RECOVERY"
    json.dumps(event, allow_nan=False)


@pytest.mark.parametrize("stale", [False, True])
def test_invalid_or_stale_lp_projects_every_captured_joint_without_using_old_jacobian(lower_writer, monkeypatch, stale):
    hand, data = hand_at_first_direction_failure()
    if stale:
        hand.target_timestamp_s -= .002
    else:
        hand.joint_target = {"feasible": False, "reason": "CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE"}
    monkeypatch.setattr(r16.r14, "constrained_joint_velocity", lambda *args: pytest.fail("Stale target solve used"))
    before = data.ctrl.copy()
    hand.write(data)
    event = hand.bias_recovery_events[-1]
    assert not event["fresh_target"]
    assert all(row["mode"] == "BIAS_RECOVERY" for row in event["per_finger"])
    assert data.ctrl[0] < before[0]
    assert data.ctrl[2] == before[2]
    assert event["per_finger"][1]["reason"] == "INVALID_TARGET_PROJECT_HOLD"
    assert all(row["nominal_qp"] is None for row in event["per_finger"])


def test_empty_box_is_explicit_failure_and_original_release_is_not_blocked(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    data.qpos[0] -= .01
    before = data.ctrl.copy()
    hand.write(data)
    event = hand.bias_recovery_events[-1]
    np.testing.assert_equal(data.ctrl, before)
    assert hand.bias_failure_latched and hand.joint_mode == "BIAS_INFEASIBLE"
    assert not event["write_verification_passed"]
    assert event["per_finger"][0]["mode"] == "INFEASIBLE"
    assert event["per_finger"][1]["mode"] == "FAIL_PENDING"
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "after_step", lambda *args: None)
    data.time += .002
    assert hand.after_step(SimpleNamespace(value="CLOSE"), data, .002) == "BIAS_BOX_INFEASIBLE"
    hand.phase, hand.joint_mode = "RELEASE", "RELEASING"
    hand.write(data)
    assert np.all(data.ctrl < before)
    data.time += .002
    assert hand.after_step(SimpleNamespace(value="RELEASE"), data, .002) is None


@pytest.mark.parametrize("probe,held,expected", [(True, False, "PROBE_BIAS_RECOVERY"),
    (False, True, "HELD_POLICY_BIAS_RECOVERY")])
def test_recovery_aborts_already_active_probe_or_held_motion(lower_writer, monkeypatch, probe, held, expected):
    hand, data = hand_at_first_direction_failure()
    hand.probe_active = hand.estimated.probe_active = probe
    hand.estimated.payload_confirmed = held
    hand.write(data)
    assert hand.bias_abort_reason == expected
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "after_step", lambda *args: None)
    data.time += .002
    assert hand.after_step(SimpleNamespace(value="TRIAL_LIFT"), data, .002) == expected
    assert not hand.probe_active and not hand.estimated.payload_confirmed
    event = hand.bias_recovery_events[-1]
    assert event["original_post_step_failure"] is None
    assert event["post_step_returned_failure"] == expected
    assert event["probe_abort_applied"] and event["resolved_abort_reason"] == expected
    assert hand.after_step(SimpleNamespace(value="RECOVER"), data, .002) is None


@pytest.mark.parametrize("probe,held,original", [(True, False, "PROBE_PRESSURE"),
    (False, True, "HELD_POLICY_FIXED_NORMAL_1G_INFEASIBLE")])
def test_original_post_step_safety_failure_has_precedence_over_recovery(lower_writer, monkeypatch, probe, held, original):
    hand, data = hand_at_first_direction_failure()
    hand.probe_active = hand.estimated.probe_active = probe
    hand.estimated.payload_confirmed = held
    hand.write(data)
    def parent(self, state, data, dt):
        assert self.probe_active == probe
        assert self.estimated.payload_confirmed == held
        return original
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "after_step", parent)
    data.time += .002
    assert hand.after_step(SimpleNamespace(value="TRIAL_LIFT"), data, .002) == original
    assert not hand.probe_active and not hand.estimated.payload_confirmed
    event = hand.bias_recovery_events[-1]
    assert event["original_post_step_failure"] == original
    assert event["post_step_returned_failure"] == original
    assert event["probe_abort_applied"] and event["resolved_abort_reason"] == original


def test_parent_post_step_cannot_restore_ready_after_a_recovery_write(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    hand.write(data)
    hand.joint_mode = "JOINT_CONTROL"
    def parent(self, state, data, dt):
        self.preload_ready = True
        self.preload_start_s = data.time
        self.probe_eligible_seconds = self.probe_confirmed_seconds = .5
        self.pressure_window.append((data.time, {}))
        return None
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "after_step", parent)
    data.time += .002
    assert hand.after_step(SimpleNamespace(value="CLOSE"), data, .002) is None
    event = hand.bias_recovery_events[-1]
    assert event["post_step_timestamp_s"] == pytest.approx(event["timestamp_s"] + .002)
    assert event["admission_after_step"] == {"preload_ready": False, "preload_start_s": None,
        "probe_eligible_seconds": 0., "probe_confirmed_seconds": 0., "probe_active": False,
        "payload_confirmed": False, "pressure_window_samples": 0}
    assert event["original_post_step_failure"] is event["post_step_returned_failure"] is None
    assert not event["probe_abort_applied"] and event["resolved_abort_reason"] is None


def test_preload_and_decisive_admission_both_block_recovery(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    hand.write(data)
    def parent(self, data, admission=False):
        self.preload_ready = True
        self.preload_start_s = data.time
        row = {"ready": True, "continuous_seconds": .25, "reasons": [], "original_probe_snapshot": {}}
        self.preload_events.append(row)
        return row
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "update_preload", parent)
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "admit_probe", lambda *args: pytest.fail("Parent admission must not run"))
    assert not hand.admit_probe(data)
    assert not hand.preload_ready and hand.preload_start_s is None
    assert not hand.preload_events[-1]["ready"]
    assert hand.preload_events[-1]["continuous_seconds"] == 0.
    assert hand.probe_events[-1]["admission"] is False


def test_late_original_safety_override_wins_and_is_recorded(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    def safety(self, data):
        self.emergency_phase_hold = True
        data.ctrl[:] = .1
    monkeypatch.setattr(r16.r13.r11.BoundedTrackingProbeHand, "write", safety)
    hand.write(data)
    np.testing.assert_equal(data.ctrl, [.1] * 8)
    event = hand.bias_recovery_events[-1]
    assert event["final_safety_override"] and event["write_verification_passed"] is None
    assert hand.joint_mode == "EMERGENCY_HOLD"
    assert list(hand.joint_references.values()) == [.1] * 8


def test_late_non_safety_mutation_latches_failure_instead_of_just_logging(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    def mutation(self, data):
        lower_writer(self, data)
        data.ctrl[0] += .01
    monkeypatch.setattr(r16.r13.r11.BoundedTrackingProbeHand, "write", mutation)
    hand.write(data)
    assert hand.bias_failure_latched
    assert hand.bias_abort_reason == "BIAS_FINAL_WRITE_VERIFICATION_FAILED"
    assert hand.joint_mode == "BIAS_INFEASIBLE"
    assert hand.bias_recovery_events[-1]["write_verification_violations"]
    assert not hand.bias_recovery_events[-1]["write_verification_passed"]
    assert hand.joint_references["left_thumb_MCP_FE"] == data.ctrl[0]


def test_before_first_capture_uses_original_scalar_writer(lower_writer, monkeypatch):
    hand, data = hand_at_first_direction_failure()
    hand.joint_references = {}
    hand.joint_activation_s = None
    hand.joint_mode = "SCALAR_ACQUISITION"
    called = []
    monkeypatch.setattr(r16.r13.JointAdmittanceProbeHand, "write", lambda self, data: called.append(data.time))
    hand.write(data)
    assert called == [14.922]
    assert not hand.bias_failure_latched
    assert hand.bias_recovery_events[-1]["per_finger"] == []


def test_runner_declares_new_controller_and_reuses_exact_geometry_and_all_old_gates():
    declaration = r16.validate_inputs(PROFILE)
    assert declaration["controller_code_modified"]
    assert not declaration["physics_force_and_task_gates_modified"]
    assert not declaration["runtime_target_pose_or_qpos_added"]
    assert declaration["profile_sha256"] == r16.r15.EXPECTED_PROFILE_SHA256
    assert declaration["r14_source_sha256"] == r16.r15.EXPECTED_R14_SHA256
    for key in ("preload_parameters", "probe_parameters", "joint_parameters", "tracking_parameters", "strict_task_gates"):
        assert declaration[key] == r16.r15.validate_inputs(PROFILE)[key]
    new = r16.controller_for("ycb_can", Path("scratch/r16_contract"), [], SimpleNamespace(),
        json.loads(PROFILE.read_text())["ycb_can"], .349)
    old = r16.r14.controller_for("ycb_can", Path("scratch/r16_contract"), [], SimpleNamespace(),
        json.loads(PROFILE.read_text())["ycb_can"], .349)
    assert ast.dump(ast.parse(new[1])) == ast.dump(ast.parse(old[1]))
    assert new[0].__globals__["HandGraspController"].__globals__["LoadTransferProbeHand"] is r16.BiasRecoveryHand
    runner = r16.isolated_runner(PROFILE, declaration)
    assert "bias_recovery.json.gz" in runner.__code__.co_consts
    for name in ("preload.json.gz", "constrained_joint.json.gz", "joint_admittance.json.gz", "probe_refresh.json.gz"):
        assert name in runner.__code__.co_consts
    sources = r16.snapshot(PROFILE)
    assert sources["scripts/develop_public_ycb_bias_recovery_r16.py"] == r16.sha256(Path(r16.__file__))
    assert sources["scripts/develop_public_ycb_constrained_preload_r14.py"] == r16.r15.EXPECTED_R14_SHA256
    assert sources[str(r16.R15_SUMMARY.relative_to(r16.PROJECT))] == r16.EXPECTED_R15_SUMMARY_SHA256


def test_protocol_keeps_scoring_probe_physics_and_deadlines_unchanged():
    protocol = {"format": "r14", "nominal_profiles": {}, "schedule": [{"profile": "r14"}],
        "scoring": "unchanged", "phase_seconds": {"CLOSE": 16.}, "load_transfer_probe": copy.deepcopy(r16.r14.PROBE_PARAMETERS)}
    before = copy.deepcopy(protocol)
    r16.tagged_protocol(protocol, {"controller": "BiasRecoveryHand"})
    assert protocol["format"] == "public_ycb_bias_recovery_development_r16"
    assert protocol["schedule"][0]["profile"] == "bias_recovery_r16"
    for key in ("scoring", "phase_seconds", "load_transfer_probe"):
        assert protocol[key] == before[key]


def test_real_mujoco_short_invalid_target_recovery_smoke(tmp_path):
    import mujoco
    from public_ycb_evidence import load_evidence_model
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario

    scene = r16.R15_SUMMARY.parent / "ycb_can_fixed_geometry_r15_seed3414/scene.xml"
    model = load_evidence_model(scene)
    data = mujoco.MjData(model)
    _, kin, _ = reset_task(model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    guard = r16.r13.r11.r9.r8.r6.LatchedPressureGuard(model, .349)
    controller, _, hands, estimated = r16.controller_for("ycb_can", tmp_path, [], guard, profile, .349)
    inherited = r16.r13.r11.r9.r8.r4.previous
    _, config, _ = inherited.selected_profile("ycb_can", profile)
    hand = controller.__globals__["HandGraspController"](model, config)
    estimated.bind(kin)
    hand.initialize(data)
    mujoco.mj_forward(model, data)
    native = inherited.RichNativeAudit(model, data)
    state = SimpleNamespace(value="CLOSE")
    with native:
        native.sample("SMOKE_INITIAL", data)
        mujoco.mj_step(model, data)
        native.sample("SMOKE_BEFORE_CAPTURE", data)
        hand.phase_begin("CLOSE", data)
        # Deliberate branch exercise, not a claim of sensor-valid contact admission.
        hand._activate(data)
        hand.last_joint_write_s = 0.
        capture = hand.initial_command_bias.copy()
        hand.joint_target = {"feasible": False, "reason": "SMOKE_NO_VISUAL_OR_CONTACT_TARGET"}
        for _ in range(50):
            hand.write(data)
            mujoco.mj_step(model, data)
            native.sample("SMOKE_INVALID_TARGET_RECOVERY", data)
            assert hand.after_step(state, data, model.opt.timestep) is None
    assert hands == [hand] and isinstance(hand, r16.BiasRecoveryHand)
    assert native.integration_steps == 51 and len(native.samples) == 52
    assert not native.external_qpos_writes
    assert len(hand.bias_recovery_events) == 51  # Includes the pre-capture reset write.
    guarded = [row for row in hand.bias_recovery_events if row["per_finger"]]
    assert len(guarded) == 50 and all(row["recovery_active"] for row in guarded), {
        "modes": sorted({row["mode"] for row in hand.bias_recovery_events}),
        "emergency_active": hand.emergency_active, "emergency_phase_hold": hand.emergency_phase_hold}
    assert all(row["write_verification_passed"] for row in guarded)
    assert all(not row["admission_after_step"]["preload_ready"] for row in guarded)
    assert not hand.probe_active and not estimated.payload_confirmed
    assert hand.initial_command_bias == capture
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    for event, sample in zip(guarded, native.samples[2:], strict=True):
        assert sample["timestamp"] == pytest.approx(event["post_step_timestamp_s"])
        for finger in event["per_finger"]:
            for name, row in finger["joints"].items():
                assert row["applied_command_rad"] == sample["motor_commands"][hand.actuators[name]]
    print(json.dumps({"scope": "Unloaded synthetic activation branch smoke; no task/contact success claim",
        "physical_steps": native.integration_steps, "recovery_writes": len(guarded),
        "simulation_seconds": data.time, "timestep_s": model.opt.timestep,
        "target_qpos_external_writes": len(native.external_qpos_writes),
        "admitted_probe": hand.probe_active, "payload_confirmed": estimated.payload_confirmed,
        "max_joint_speed_rad_s": max(abs(data.qvel)), "scene_sha256": r16.sha256(scene)}))
