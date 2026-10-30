"""Isolated budget, native torque logging and original safety behavior."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_bias_budget_r17 as r17
from test_public_ycb_bias_recovery_r16 import hand_at_first_direction_failure, lower_writer

r16 = r17.r16
PROFILE = r16.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def budgeted_fixture(budget):
    hand, data = hand_at_first_direction_failure()
    hand.__class__ = r17.BiasBudgetHand
    hand._additional_bias_limit_rad = r17.checked_budget(budget)
    return hand, data


def without_budget_metadata(value):
    if isinstance(value, list):
        return [without_budget_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: without_budget_metadata(item) for key, item in value.items()
        if key not in ("additional_bias_limit_rad", "inherited_write_format")}
    if result.get("format") == "r17_bias_budget_write_v1":
        result["format"] = "r16_bias_recovery_write_v1"
    return result


@pytest.mark.parametrize("budget", [.02, .04, .10])
def test_explicit_supported_budget_and_read_only_public_attribute(budget):
    hand, _ = budgeted_fixture(budget)
    assert hand.additional_bias_limit_rad == budget
    with pytest.raises(AttributeError):
        hand.additional_bias_limit_rad = .10


@pytest.mark.parametrize("budget", [0., -.02, .03, float("inf"), float("nan")])
def test_undeclared_or_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        r17.checked_budget(budget)


def test_point02_is_original_r16_command_and_event_identity(lower_writer):
    original, before = hand_at_first_direction_failure()
    hand, data = budgeted_fixture(.02)
    original.write(before)
    hand.write(data)
    np.testing.assert_equal(data.ctrl, before.ctrl)
    np.testing.assert_equal(data.qpos, before.qpos)
    assert hand.joint_references == original.joint_references
    assert without_budget_metadata(hand.bias_recovery_events) == original.bias_recovery_events
    assert without_budget_metadata(hand.joint_events) == original.joint_events
    assert r17.BiasBudgetHand.write is r16.BiasRecoveryHand.write
    assert r17.BiasBudgetHand.after_step is r16.BiasRecoveryHand.after_step
    assert r17.BiasBudgetHand.admit_probe is r16.BiasRecoveryHand.admit_probe


def test_point04_relaxes_only_bias_box_and_saved_qp_can_compress(lower_writer):
    small, small_data = budgeted_fixture(.02)
    larger, large_data = budgeted_fixture(.04)
    small_rows = small._joint_rows(small_data, small.joint_references, .002)
    large_rows = larger._joint_rows(large_data, larger.joint_references, .002)
    for name in small_rows:
        for key in ("range_rad", "rate_interval_rad", "fixed_capture_bias_rad", "before_actual_rad"):
            assert small_rows[name][key] == large_rows[name][key]
    capture = larger.initial_command_bias.copy()
    small.write(small_data)
    larger.write(large_data)
    first = small.bias_recovery_events[-1]["per_finger"][0]
    second = larger.bias_recovery_events[-1]["per_finger"][0]
    assert first["mode"] == "BIAS_RECOVERY"
    assert second["mode"] == "NOMINAL" and second["nominal_qp"]["accepted"]
    assert second["predicted_applied_normal_velocity_m_s"][0] > 0.
    assert larger.initial_command_bias == capture
    assert all(row["applied_within_original_box"] for row in second["joints"].values())
    assert larger.bias_recovery_events[-1]["additional_bias_limit_rad"] == .04


def test_instances_interleave_without_global_or_capture_mutation(lower_writer):
    globals_before = copy.deepcopy(r16.r13.JOINT_PARAMETERS)
    small, a = budgeted_fixture(.02)
    larger, b = budgeted_fixture(.04)
    first = copy.deepcopy(small._joint_rows(a, small.joint_references, .002))
    larger.write(b)
    assert small._joint_rows(a, small.joint_references, .002) == first
    small.write(a)
    assert r16.r13.JOINT_PARAMETERS == globals_before
    assert small.additional_bias_limit_rad == .02 and larger.additional_bias_limit_rad == .04


def test_recovery_still_blocks_decisive_tactile_admission(lower_writer, monkeypatch):
    hand, data = budgeted_fixture(.04)
    hand.joint_target = {"feasible": False, "reason": "NO_CURRENT_TACTILE_TARGET"}
    hand.write(data)
    assert hand.bias_recovery_active
    def parent(self, data, *, admission=False):
        self.preload_ready = True
        row = {"ready": True, "continuous_seconds": .25, "reasons": [], "original_probe_snapshot": {}}
        self.preload_events.append(row)
        return row
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "update_preload", parent)
    monkeypatch.setattr(r16.r14.ConstrainedPreloadHand, "admit_probe", lambda *args: pytest.fail("Admission bypass"))
    assert not hand.admit_probe(data)
    assert not hand.preload_ready and hand.preload_events[-1]["continuous_seconds"] == 0.


def test_preflight_snapshot_and_guard_close_all_validation_inputs(monkeypatch):
    declaration = r17.validate_inputs(PROFILE, .04)
    baseline = json.loads(r17.R16_SUMMARY.read_text())
    assert declaration["inherited_r16_declaration"] == baseline["protocol"]["r16_bias_recovery_experiment"]
    assert declaration["joint_parameters"]["additional_bias_limit_rad"] == .04
    assert declaration["inherited_r16_declaration"]["joint_parameters"]["additional_bias_limit_rad"] == .02
    sources = r17.snapshot(PROFILE)
    for relative, expected in r17.PREFLIGHT_HASHES.items():
        assert sources[relative] == expected
    for relative, expected in baseline["protocol"]["source_hashes_before"].items():
        assert sources[relative] == expected
    for relative in r17.NEW_SOURCES:
        assert sources[relative] == r16.sha256(r17.PROJECT / relative)
    original = r16.sha256
    monkeypatch.setattr(r16, "sha256", lambda path: "0" * 64 if Path(path) == r17.R16_SUMMARY else original(path))
    with pytest.raises(ValueError, match="preflight input SHA256"):
        r17.validate_inputs(PROFILE, .04)


def protocol_stub():
    return {"format": "r14", "nominal_profiles": {}, "schedule": [{"kind": "ycb_can", "seed": 3414}],
        "joint_admittance": copy.deepcopy(r16.r13.JOINT_PARAMETERS),
        "phase_seconds": {"CLOSE": 16.}, "scoring": "unchanged",
        "load_transfer_probe": copy.deepcopy(r16.r14.PROBE_PARAMETERS)}


def test_nested_tagger_preserves_active_budget_geometry_gates_and_scene_binding(tmp_path):
    declaration = r17.validate_inputs(PROFILE, .04)
    runner = r17.isolated_runner(PROFILE, declaration)
    assert runner.__globals__["previous"].RichNativeAudit is r17.TorqueNativeAudit
    protocol = protocol_stub()
    before = copy.deepcopy(protocol)
    r17.tagged_protocol(protocol, declaration)
    r16.tagged_protocol(protocol, declaration["inherited_r16_declaration"])
    assert protocol["format"] == "public_ycb_bias_budget_development_r17"
    assert protocol["joint_admittance"]["additional_bias_limit_rad"] == .04
    for key in ("phase_seconds", "scoring", "load_transfer_probe", "nominal_profiles"):
        assert protocol[key] == before[key]
    assert protocol["schedule"][0]["additional_bias_limit_rad"] == .04
    destination = tmp_path / "protocol.json"
    runner.__globals__["previous"].atomic_json(destination, before)
    saved = json.loads(destination.read_text())
    assert saved["format"] == protocol["format"]
    assert saved["joint_admittance"]["additional_bias_limit_rad"] == .04
    assert saved["r16_bias_recovery_experiment"] == declaration["inherited_r16_declaration"]
    scene = tmp_path / "scene.xml"
    builder = runner.__globals__["build_pick_place_scene"]
    builder(scene, "ycb_can", physics_profile="physical-v6")
    meta = {"kind": "ycb_can", "seed": 3414}
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json", meta)
    saved_meta = json.loads((tmp_path / "metadata.json").read_text())
    assert saved_meta["scene_unchanged_since_load"]
    assert saved_meta["scene_sha256_before_load"] == r16.sha256(scene)
    assert saved_meta["additional_bias_limit_rad"] == .04


@pytest.fixture(scope="module")
def real_model():
    from public_ycb_evidence import load_evidence_model
    scene = r16.R15_SUMMARY.parent / "ycb_can_fixed_geometry_r15_seed3414/scene.xml"
    return load_evidence_model(scene)


def real_hand(model, directory, budget):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(model)
    _, kin, _ = reset_task(model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    guard = r16.r13.r11.r9.r8.r6.LatchedPressureGuard(model, .349)
    controller, source, hands, estimated = r17.controller_for("ycb_can", directory, [], guard, profile, .349,
        additional_bias_limit_rad=budget)
    config = r16.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    estimated.bind(kin)
    hand = controller.__globals__["HandGraspController"](model, config)
    return data, hand, guard, controller, hands, config, source


def test_factory_retry_creates_distinct_consistent_budgeted_instances(real_model, tmp_path):
    _, hand, _, controller, hands, config, source = real_hand(real_model, tmp_path, .04)
    retry = controller.__globals__["HandGraspController"](real_model, config)
    assert retry is not hand and hands == [hand, retry]
    assert [item.additional_bias_limit_rad for item in hands] == [.04, .04]
    old = r16.controller_for("ycb_can", tmp_path, [], SimpleNamespace(),
        json.loads(PROFILE.read_text())["ycb_can"], .349)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(old[1]))


def test_real_r6_final_writer_overrides_nominal_references(real_model, tmp_path):
    data, hand, guard, *_ = real_hand(real_model, tmp_path, .04)
    hand.initialize(data)
    hand.phase_begin("CLOSE", data)
    hand._activate(data)
    hand.joint_references = {name: value + .01 for name, value in hand.joint_references.items()}
    guard.escape = {}  # Explicit R6 escape branch, without manufactured contact readings.
    hand.hold_for_escape(data)
    qpos = data.qpos.copy()
    expected = hand.actual_joint_references.copy()
    hand.write(data)
    for name, value in expected.items():
        assert data.ctrl[hand.actuators[name]] == pytest.approx(value, abs=1e-14)
    np.testing.assert_equal(data.qpos, qpos)
    event = hand.bias_recovery_events[-1]
    assert event["final_safety_override"] and event["write_verification_passed"] is None
    assert event["additional_bias_limit_rad"] == .04
    assert hand.joint_mode == "EMERGENCY_HOLD"


def test_native_force_arrays_are_copies_with_no_forward_step_or_state_write(real_model, monkeypatch):
    import mujoco
    data = mujoco.MjData(real_model)
    mujoco.mj_step(real_model, data)
    audit = r17.TorqueNativeAudit(real_model, data)
    arrays = ("qpos", "qvel", "ctrl", "act", "qacc", "qfrc_actuator", "actuator_force",
        "actuator_length", "actuator_velocity", "qfrc_applied", "xfrc_applied")
    before = {key: getattr(data, key).copy() for key in arrays}
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args: pytest.fail("Logger forwarded model"))
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Logger stepped model"))
    timestamp = data.time
    audit.sample("EVALUATION_ONLY", data)
    audit.sample("DUPLICATE", data)
    assert len(audit.samples) == 1 and data.time == timestamp
    for key in arrays:
        np.testing.assert_equal(getattr(data, key), before[key])
    sample = audit.samples[-1]["native_actuation"]
    np.testing.assert_equal(sample["actuator_force"], data.actuator_force[audit.actuation_ids])
    np.testing.assert_equal(sample["qfrc_actuator"], data.qfrc_actuator[audit.actuation_dofs])
    first = sample["actuator_force"].copy()
    data.actuator_force[:] = 123.
    assert sample["actuator_force"] == first
    metadata = audit.as_dict()["native_actuation_metadata"]
    assert metadata["evaluation_only"] and not metadata["postintegration_qpos_recomputed"]
    metadata["actuator_parameters"].clear()
    assert audit.as_dict()["native_actuation_metadata"]["actuator_parameters"]


def test_real_short_unloaded_recovery_native_force_and_tactile_gate_smoke(real_model, tmp_path):
    import mujoco
    data, hand, _, _, hands, _, _ = real_hand(real_model, tmp_path, .04)
    hand.initialize(data)
    audit = r17.TorqueNativeAudit(real_model, data)
    with audit:
        mujoco.mj_step(real_model, data)
        audit.sample("SMOKE_BEFORE_CAPTURE", data)
        hand.phase_begin("CLOSE", data)
        hand._activate(data)
        hand.last_joint_write_s = 0.
        capture = hand.initial_command_bias.copy()
        hand.joint_target = {"feasible": False, "reason": "SMOKE_NO_VISUAL_OR_CONTACT_TARGET"}
        for _ in range(50):
            hand.write(data)
            mujoco.mj_step(real_model, data)
            audit.sample("SMOKE_INVALID_TARGET_RECOVERY", data)
            assert hand.after_step(SimpleNamespace(value="CLOSE"), data, .002) is None
        assert not hand.admit_probe(data)
    assert hands == [hand] and hand.additional_bias_limit_rad == .04
    assert audit.integration_steps == len(audit.samples) == 51
    assert not audit.external_qpos_writes and hand.initial_command_bias == capture
    guarded = [row for row in hand.bias_recovery_events if row["per_finger"]]
    assert len(guarded) == 50 and all(row["write_verification_passed"] for row in guarded)
    assert all(row["additional_bias_limit_rad"] == .04 for row in hand.bias_recovery_events)
    assert not hand.probe_active and not hand.estimated.payload_confirmed and not hand.preload_ready
    parameters = audit.actuation_metadata["actuator_parameters"]
    for sample in audit.samples:
        forces = sample["native_actuation"]
        for i, model_parameter in enumerate(parameters):
            gain, bias = model_parameter["gainprm"], model_parameter["biasprm"]
            ctrl = sample["motor_commands"][model_parameter["actuator_id"]]
            ctrl = np.clip(ctrl, *model_parameter["ctrlrange"])
            expected = gain[0] * ctrl + bias[0] + bias[1] * forces["actuator_length"][i] + bias[2] * forces["actuator_velocity"][i]
            assert forces["actuator_force"][i] == pytest.approx(expected, abs=1e-12)
            joint_force = expected * model_parameter["gear"][0]
            if model_parameter["jointactfrclimited"]:
                joint_force = np.clip(joint_force, *model_parameter["jointactfrcrange"])
            assert forces["qfrc_actuator"][i] == pytest.approx(joint_force, abs=1e-12)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    print(json.dumps({"scope": "Unloaded synthetic activation smoke, not contact admission or task success",
        "physical_steps": audit.integration_steps, "simulation_seconds": data.time,
        "recovery_writes": len(guarded), "budget_rad": hand.additional_bias_limit_rad,
        "external_target_qpos_writes": len(audit.external_qpos_writes), "admitted_probe": hand.probe_active}))


def test_one_case_only_and_cli_requires_explicit_single_budget(monkeypatch):
    declaration = r17.validate_inputs(PROFILE, .04)
    monkeypatch.setattr(r17, "validate_inputs", lambda *args: declaration)
    monkeypatch.setattr(r17, "isolated_runner", lambda *args: pytest.fail("Invalid case should not start runner"))
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r17.run(Path("unused"), PROFILE, (("ycb_can", 3415),), additional_bias_limit_rad=.04)
    monkeypatch.setattr(sys, "argv", ["r17", "--output", "unused", "--profile-json", str(PROFILE)])
    with pytest.raises(SystemExit) as error:
        r17.main()
    assert error.value.code == 2
