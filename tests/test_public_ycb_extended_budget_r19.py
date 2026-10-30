"""Single-budget isolation and provenance; no physical case is executed."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_extended_budget_r19 as r19
from test_public_ycb_bias_budget_r17 import PROFILE, budgeted_fixture, real_model
from test_public_ycb_bias_recovery_r16 import lower_writer

r17, r18 = r19.r17, r19.r18


def baseline():
    return json.loads(r19.R18_SUMMARY.read_text())


def test_exact_recovered_failed_baseline_and_single_active_change():
    declaration = r19.validate_inputs(PROFILE)
    old = baseline()["protocol"]
    assert declaration["changed_parameters"] == {
        "joint_admittance.additional_bias_limit_rad": {"before": .04, "after": .10}}
    assert declaration["additional_bias_limit_rad"] == .10
    assert declaration["active_r17_declaration"] == r17.validate_inputs(PROFILE, .10)
    assert declaration["inherited_r18_declaration"] == old["r18_close_deadline_experiment"]
    assert declaration["inherited_r18_r17_declaration"] == old["r17_bias_budget_experiment"]
    assert declaration["inherited_r18_joint_admittance"]["additional_bias_limit_rad"] == .04
    assert declaration["inherited_r18_recovery"] == old["finalization_recovery"]
    assert not declaration["baseline_task_success"] and not declaration["baseline_held_lift"]
    assert not declaration["baseline_prefix_comparison_passed"]
    assert declaration["inherited_r18_recovery"]["runner_exit_code_observed"] is None
    assert not declaration["inherited_r18_recovery"]["simulation_rerun"]
    assert declaration["effective_phase_seconds"]["CLOSE"] == 32.
    assert declaration["arm_interpolation_seconds"] == 16.


def test_source_snapshot_closes_all73_and_all_preflight_inputs():
    values = r19.snapshot(PROFILE)
    inherited = baseline()["protocol"]["source_hashes_before"]
    assert len(inherited) == 73
    assert all(values[key] == value for key, value in inherited.items())
    assert all(values[key] == value for key, value in r19.PREFLIGHT_HASHES.items())
    for name in (*r19.NEW_SOURCES, *r19.OPTIONAL_AUDIT_SOURCES, *r19.OPTIONAL_SUPERVISOR_SOURCES):
        path = r19.PROJECT / name
        if path.exists():
            assert values[name] == r17.r16.sha256(path)


@pytest.mark.parametrize("relative", [
    "scripts/develop_public_ycb_close_deadline_r18.py",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/recovery_receipt.json",
    "results/r17_r18_close_deadline_comparison_20261022/receipt.json",
])
def test_bound_preflight_input_mutation_is_rejected(monkeypatch, relative):
    sha = r17.r16.sha256
    monkeypatch.setattr(r17.r16, "sha256", lambda path:
        "0" * 64 if Path(path) == r19.PROJECT / relative else sha(path))
    with pytest.raises(ValueError, match="R19 preflight input SHA256"):
        r19.validate_inputs(PROFILE)


def test_inherited_source_mutation_is_rejected_even_outside_new_pins(monkeypatch):
    sha = r17.r16.sha256
    relative = next(name for name in baseline()["protocol"]["source_hashes_before"]
        if name not in r19.PREFLIGHT_HASHES and name.startswith("src/"))
    monkeypatch.setattr(r17.r16, "sha256", lambda path:
        "0" * 64 if Path(path) == r19.PROJECT / relative else sha(path))
    with pytest.raises(ValueError, match="source mismatch"):
        r19.validate_inputs(PROFILE)


def test_real_factory_all_retry_hands_use_point10_with_same_source_and_gates(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    guard = r17.r16.r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    parameters = (r17.r16.r14.PRELOAD_PARAMETERS, r17.r16.r13.JOINT_PARAMETERS,
        r17.r16.r14.PROBE_PARAMETERS, r17.r16.r13.r11.r9.r8.PHASE_SECONDS)
    before = copy.deepcopy(parameters)
    args = ("ycb_can", tmp_path, [], guard, profile, .349)
    old, old_source, _, _ = r18.controller_for(*args)
    task, source, hands, estimated = r19.controller_for(*args)
    assert r19.hashlib.sha256(source.encode()).hexdigest() == r19.EXPECTED_R18_DERIVED_SHA256
    assert task.__globals__["_phase_seconds"] == old.__globals__["_phase_seconds"]
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.
    assert task.__globals__["_phase_seconds"] is not old.__globals__["_phase_seconds"]
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(old_source))
    config = r17.r16.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    estimated.bind(kin)
    first = task.__globals__["HandGraspController"](real_model, config)
    second = task.__globals__["HandGraspController"](real_model, config)
    assert hands == [first, second] and first is not second
    assert all(type(hand) is r17.BiasBudgetHand and hand.additional_bias_limit_rad == .10 for hand in hands)
    for hand in hands:
        assert hand.write.__func__ is r17.r16.BiasRecoveryHand.write
        assert hand.after_step.__func__ is r17.r16.BiasRecoveryHand.after_step
        assert hand.admit_probe.__func__ is r17.r16.BiasRecoveryHand.admit_probe
        with pytest.raises(AttributeError):
            hand.additional_bias_limit_rad = .04
    assert parameters == before


def test_controller_semantic_drift_cannot_hide_behind_frozen_source(monkeypatch, tmp_path):
    transform = r18.transform_controller
    def changed(source):
        tree, counts = transform(source)
        tree.body[0].body.insert(0, ast.parse("unexpected_controller_change = True").body[0])
        return tree, counts
    monkeypatch.setattr(r18, "transform_controller", changed)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    with pytest.raises(ValueError, match="transformed controller AST differs"):
        r19.controller_for("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349)


def test_point10_changes_only_bias_intersection_and_actual_event_budget(lower_writer):
    old, old_data = budgeted_fixture(.04)
    active, data = budgeted_fixture(.10)
    old_rows = old._joint_rows(old_data, old.joint_references, .002)
    rows = active._joint_rows(data, active.joint_references, .002)
    assert old_rows.keys() == rows.keys()
    for name, row in rows.items():
        for key in ("range_rad", "rate_interval_rad", "fixed_capture_bias_rad", "before_actual_rad"):
            assert row[key] == old_rows[name][key]
        assert row["bias_interval_rad"][0] == pytest.approx(old_rows[name]["bias_interval_rad"][0] - .06)
        assert row["bias_interval_rad"][1] == pytest.approx(old_rows[name]["bias_interval_rad"][1] + .06)
    capture, qpos = active.initial_command_bias.copy(), data.qpos.copy()
    active.write(data)
    assert active.initial_command_bias == capture
    np.testing.assert_equal(data.qpos, qpos)
    assert active.bias_recovery_events[-1]["additional_bias_limit_rad"] == .10


def test_nested_writers_keep_actual_r17_point10_and_historical_r18_point04(tmp_path):
    declaration = r19.validate_inputs(PROFILE)
    runner = r19.isolated_runner(PROFILE, declaration)
    old = copy.deepcopy(baseline()["protocol"])
    protocol = copy.deepcopy(old)
    runner.__globals__["previous"].atomic_json(tmp_path / "protocol.json", protocol)
    saved = json.loads((tmp_path / "protocol.json").read_text())
    assert saved["format"] == saved["runner_revision"] == r19.PROTOCOL_FORMAT
    assert saved["joint_admittance"]["additional_bias_limit_rad"] == .10
    assert saved["r17_bias_budget_experiment"] == declaration["active_r17_declaration"]
    assert saved["r19_extended_budget_experiment"] == declaration
    assert "r18_close_deadline_experiment" not in saved
    assert saved["inherited_r18_close_deadline_experiment"] == old["r18_close_deadline_experiment"]
    assert saved["inherited_r18_joint_admittance"] == old["joint_admittance"]
    assert saved["inherited_r18_force_control"] == old["force_control"]
    for key in ("scoring", "load_transfer_probe", "probe_confirmation", "nominal_profiles", "phase_seconds", "constrained_preload"):
        assert saved[key] == old[key]
    assert saved["schedule"][0]["additional_bias_limit_rad"] == .10
    assert saved["schedule"][0]["close_seconds"] == 32.
    assert saved["schedule"][0]["arm_interpolation_seconds"] == 16.
    summary = {"protocol": saved, "rows": [{"kind": "ycb_can", "seed": 3414}],
        "controller_revision": "public_ycb_constrained_preload_r14"}
    runner.__globals__["previous"].atomic_json(tmp_path / "summary.json", summary)
    saved_summary = json.loads((tmp_path / "summary.json").read_text())
    assert saved_summary["protocol"] == saved
    assert saved_summary["controller_revision"] == "public_ycb_bias_budget_development_r17"
    assert saved_summary["runner_revision"] == r19.PROTOCOL_FORMAT
    assert saved_summary["rows"][0]["additional_bias_limit_rad"] == .10
    assert runner.__globals__["previous"].RichNativeAudit is r17.TorqueNativeAudit
    assert runner.__globals__["PHASE_SECONDS"]["CLOSE"] == 32.


def test_metadata_reads_actual_live_retry_hands_and_scene_binding(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    declaration = r19.validate_inputs(PROFILE)
    runner = r19.isolated_runner(PROFILE, declaration)
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, hands, estimated = runner.__globals__["controller_for"]("ycb_can", tmp_path, [],
        SimpleNamespace(), profile, .349)
    estimated.bind(kin)
    config = r17.r16.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    task.__globals__["HandGraspController"](real_model, config)
    task.__globals__["HandGraspController"](real_model, config)
    assert len(hands) == 2
    scene = tmp_path / "scene.xml"
    runner.__globals__["build_pick_place_scene"](scene, "ycb_can", physics_profile="physical-v6")
    value = {"kind": "ycb_can", "seed": 3414, "runner_error": None}
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json", value)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["effective_hand_budgets_rad"] == [.10, .10]
    assert saved["effective_hand_controller_types"] == ["BiasBudgetHand", "BiasBudgetHand"]
    assert saved["effective_hand_instances"] == 2
    assert saved["effective_task_phase_seconds"] == task.__globals__["_phase_seconds"]
    assert saved["effective_runner_phase_seconds"] == runner.__globals__["PHASE_SECONDS"]
    assert saved["arm_source_transform"]["derived_controller_sha256"] == r19.hashlib.sha256(source.encode()).hexdigest()
    assert saved["scene_unchanged_since_load"] and saved["scene_sha256_before_load"] == r17.r16.sha256(scene)
    assert "missing_runtime_controller_binding" not in saved


def test_constructor_failure_does_not_invent_controller_binding(tmp_path):
    runner = r19.isolated_runner(PROFILE, r19.validate_inputs(PROFILE))
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json",
        {"kind": "ycb_can", "seed": 3414, "runner_error": "CONSTRUCTION_FAILED"})
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["runner_error"] == "CONSTRUCTION_FAILED" and saved["missing_runtime_controller_binding"]
    for key in ("effective_task_phase_seconds", "effective_runner_phase_seconds", "effective_hand_budgets_rad"):
        assert key not in saved


@pytest.mark.parametrize("cases", [(("ycb_can", 3415),), (("ycb_can", 3414), ("ycb_can", 3414))])
def test_other_cases_do_not_start_runner(monkeypatch, tmp_path, cases):
    monkeypatch.setattr(r19, "isolated_runner", lambda *args: pytest.fail("Invalid case started"))
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r19.run(tmp_path / "new", PROFILE, cases)


def test_existing_output_cannot_be_reused(tmp_path):
    with pytest.raises(FileExistsError):
        r19.run(tmp_path, PROFILE, (("ycb_can", 3414),))
