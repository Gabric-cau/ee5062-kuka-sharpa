"""R21 source/factory/metadata checks; no integration or dynamic success evidence."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_provisional_preload_r21 as r21
from test_public_ycb_bias_budget_r17 import PROFILE, real_model


def baseline():
    return json.loads(r21.R20_SUMMARY.read_text())


def metadata_declaration():
    """Unchecked test fixture for serialization; never substituted in run/preflight."""
    prior = baseline()["protocol"]
    return {"active_r20_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]),
        "active_r19_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]["active_r19_declaration"]),
        "inherited_r20_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]),
        "inherited_r20_force_control": prior["force_control"]}


def test_pending_final_receipt_cannot_pass_preflight_or_snapshot(monkeypatch):
    monkeypatch.setitem(r21.PREFLIGHT_HASHES, "scripts/audit_public_ycb_shared_normal_postrun.py", "PENDING")
    for function in (r21.validate_inputs, r21.snapshot):
        with pytest.raises(ValueError, match="all final R20 postrun audit hashes"):
            function(PROFILE)


def test_final_preflight_preserves_exact_failed_r20_and_only_changes_pre_motion_reason():
    declaration = r21.validate_inputs(PROFILE)
    old = baseline()["protocol"]
    assert declaration["changed_parameters"] == {
        "pre_motion_preload_reasons.CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE": {
            "before": "blocks_ready", "after": "observed_but_deferred_to_unchanged_probe_confirmation_and_held_policy"}}
    assert declaration["active_r20_declaration"] == declaration["inherited_r20_declaration"] == old["r20_shared_normal_experiment"]
    assert declaration["active_r19_declaration"] == old["r20_shared_normal_experiment"]["active_r19_declaration"]
    assert declaration["target_allocator_revision"] == r21.r20.PROTOCOL_FORMAT
    assert declaration["admission_policy_revision"] == r21.PROTOCOL_FORMAT
    assert declaration["additional_bias_limit_rad"] == .10
    assert declaration["close_seconds"] == 32. and declaration["arm_interpolation_seconds"] == 16.
    assert not declaration["baseline_task_success"] and not declaration["baseline_held_lift"]
    assert declaration["baseline_child_exit_observed"] and declaration["baseline_child_returncode"] == 0
    assert not declaration["baseline_original_command_audit_passed"]
    assert declaration["baseline_postrun_extension_format"] == r21.POSTRUN_FORMAT
    assert declaration["actual_fixed_normal_1g_observation_retained"]
    assert declaration["actual_fixed_normal_1g_confirmation_and_held_policy_unchanged"]
    assert not declaration["physics_force_and_strict_task_gates_modified"]
    values = r21.snapshot(PROFILE)
    assert all(values[key] == value for key, value in old["source_hashes_before"].items())
    assert all(values[key] == value for key, value in r21.PREFLIGHT_HASHES.items())
    for relative in (*r21.NEW_SOURCES, *r21.OPTIONAL_AUDIT_SOURCES):
        if (r21.PROJECT / relative).exists():
            assert values[relative] == r21.r20.r19.r17.r16.sha256(r21.PROJECT / relative)


@pytest.mark.parametrize("relative", [
    "scripts/provisional_preload_controller_hooks.py",
    "results/r20_shared_normal_audit_20261022/receipt.json",
    "results/jobs/r20_shared_normal_bias10_close32_seed3414_20261022/status.json",
])
def test_changed_policy_or_failed_receipt_or_exit_binding_is_rejected(monkeypatch, relative):
    # Isolate the first hash guard without depending on later preflight checks.
    monkeypatch.setattr(r21, "_require_final_pins", lambda: None)
    sha256 = r21.r20.r19.r17.r16.sha256
    monkeypatch.setattr(r21.r20.r19.r17.r16, "sha256", lambda path:
        "0" * 64 if Path(path) == r21.PROJECT / relative else sha256(path))
    with pytest.raises(ValueError, match="R21 preflight input SHA256"):
        r21.validate_inputs(PROFILE)


def test_factory_keeps_r19_arm_transform_and_only_constructor_binding_changes():
    clone, old = r21._controller_factory, r21.r20.r19.controller_for
    assert clone.__code__ is old.__code__ and clone.__closure__ is old.__closure__
    for name, value in old.__globals__.items():
        assert clone.__globals__[name] is (r21._r17_proxy if name == "r17" else value)
    for name, value in vars(r21.r20.r19.r17).items():
        assert getattr(r21._r17_proxy, name) is (r21.policy.controller_for if name == "controller_for" else value)


def test_ast_hash_is_portable_for_empty_type_parameters_but_preserves_semantic_fields():
    original = ast.parse("def f(x=None):\n    return []\n").body[0]
    without = copy.deepcopy(original)
    without._fields = tuple(name for name in original._fields if name != "type_params")
    with_empty = copy.deepcopy(without)
    with_empty._fields += ("type_params",)
    with_empty.type_params = []
    with_empty.lineno = 500
    assert r21._ast_sha(without) == r21._ast_sha(with_empty)
    with_empty.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert r21._ast_sha(without) != r21._ast_sha(with_empty)
    changed = copy.deepcopy(without)
    changed.body[0].value = ast.Constant(None)
    assert r21._ast_sha(without) != r21._ast_sha(changed)


def test_live_runtime_binding_covers_both_allocators_policy_asts_and_unchanged_methods():
    hand = object.__new__(r21.policy.ProvisionalPreloadBudgetHand)
    hand._additional_bias_limit_rad = .10
    binding = r21.runtime_binding(hand)
    assert binding["controller_class"] == "ProvisionalPreloadBudgetHand"
    assert binding["additional_bias_limit_rad"] == .10
    assert binding["admission_policy_revision"] == r21.PROTOCOL_FORMAT
    assert all(binding["unchanged_methods"].values())
    for path in ("tracking", "gate_only"):
        assert binding[path]["allocator_is_shared_normal"]
        assert binding[path]["same_code_as_original"] and binding[path]["same_closure_as_original"]
    admission = binding["admission_policy"]
    assert admission["ast_hash_format"] == r21.AST_HASH_FORMAT
    assert admission["wrapper_is_provisional"] and admission["r16_helper_is_provisional"] and admission["r14_helper_is_provisional"]
    assert admission["deferred_pre_motion_reason"] == r21.policy.DEFERRED_REASON
    assert admission["original_probe_snapshot_and_actual_fixed_normal_value_retained"]
    assert admission["update_preload"]["method_source_file"] == "scripts/provisional_preload_controller_hooks.py"
    assert admission["update_preload"]["method_source_file_sha256"] == r21.PREFLIGHT_HASHES["scripts/provisional_preload_controller_hooks.py"]
    for key in ("r14_parent", "r16_recovery_wrapper"):
        item = admission[key]
        assert item["original_ast_sha256"] != item["transformed_ast_sha256"]
        assert item["other_globals_match_except_warning_registry"]
    assert admission["r14_parent"]["deleted_reason_if_count"] == 1
    assert admission["r16_recovery_wrapper"]["replaced_parent_call_count"] == 1
    hand.update_preload = r21.policy.r16.BiasRecoveryHand.update_preload.__get__(hand)
    altered = r21.runtime_binding(hand)["admission_policy"]
    assert not altered["wrapper_is_provisional"] and not altered["r16_helper_is_provisional"]


def test_real_first_retry_hands_runtime_metadata_and_scene_preserve_r20_contract(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    old_globals = dict(r21.r20.isolated_runner.__globals__)
    runner = r21.isolated_runner(PROFILE, metadata_declaration())
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, hands, estimated = runner.__globals__["controller_for"]("ycb_can", tmp_path,
        [], SimpleNamespace(), profile, .349)
    estimated.bind(kin)
    config = r21.r20.hooks.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    task.__globals__["HandGraspController"](real_model, config)
    task.__globals__["HandGraspController"](real_model, config)
    assert len(hands) == 2 and hands[0] is not hands[1]
    assert all(type(hand) is r21.policy.ProvisionalPreloadBudgetHand for hand in hands)
    assert [hand.additional_bias_limit_rad for hand in hands] == [.10, .10]
    for hand in hands:
        with pytest.raises(AttributeError):
            hand.additional_bias_limit_rad = .04
    assert r21.hashlib.sha256(source.encode()).hexdigest() == baseline()["rows"][0]["derived_controller_sha256"]
    assert task.__globals__["_phase_seconds"]["CLOSE"] == runner.__globals__["PHASE_SECONDS"]["CLOSE"] == 32.
    assert runner.__globals__["previous"].RichNativeAudit is r21.r20.r19.r17.TorqueNativeAudit
    scene = tmp_path / "scene.xml"
    runner.__globals__["build_pick_place_scene"](scene, "ycb_can", physics_profile="physical-v6")
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json",
        {"kind": "ycb_can", "seed": 3414, "runner_error": None,
            "controller_revision": "public_ycb_constrained_preload_r14"})
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["effective_hand_controller_types"] == ["ProvisionalPreloadBudgetHand"] * 2
    assert saved["effective_hand_budgets_rad"] == [.10] * 2 and saved["effective_hand_instances"] == 2
    assert saved["shared_normal_runtime_bindings"] == [r21.runtime_binding(hand) for hand in hands]
    assert saved["effective_task_phase_seconds"] == saved["effective_runner_phase_seconds"] == baseline()["protocol"]["phase_seconds"]
    assert saved["controller_class"] == "ProvisionalPreloadBudgetHand"
    assert saved["controller_revision"] == "public_ycb_bias_budget_development_r17"
    assert saved["controller_revision_scope"] == r21.CONTROLLER_REVISION_SCOPE
    assert saved["target_allocator_revision"] == r21.r20.PROTOCOL_FORMAT
    assert saved["admission_policy_revision"] == saved["runner_revision"] == r21.PROTOCOL_FORMAT
    assert saved["scene_unchanged_since_load"]
    assert saved["scene_sha256_before_load"] == r21.r20.r19.r17.r16.sha256(scene)
    actual_xml = ET.parse(scene).getroot()
    old_xml = ET.parse(r21.R20_DIRECTORY / baseline()["rows"][0]["path"] / "scene.xml").getroot()
    for tree in (actual_xml, old_xml):
        tree.find("compiler").set("meshdir", "SAME_FROZEN_ASSET_ROOT")
    assert ET.tostring(actual_xml) == ET.tostring(old_xml)
    assert all(r21.r20.isolated_runner.__globals__[key] is value for key, value in old_globals.items())


def test_nested_protocol_and_result_tags_preserve_allocator_and_task_gates(tmp_path):
    declaration = metadata_declaration()
    runner = r21.isolated_runner(PROFILE, declaration)
    old = baseline()["protocol"]
    runner.__globals__["previous"].atomic_json(tmp_path / "protocol.json", copy.deepcopy(old))
    saved = json.loads((tmp_path / "protocol.json").read_text())
    assert saved["format"] == saved["runner_revision"] == saved["admission_policy_revision"] == r21.PROTOCOL_FORMAT
    assert saved["target_allocator_revision"] == r21.r20.PROTOCOL_FORMAT
    assert saved["controller_class"] == "ProvisionalPreloadBudgetHand"
    assert saved["r21_provisional_preload_experiment"] == declaration
    assert "r20_shared_normal_experiment" not in saved
    assert saved["inherited_r20_shared_normal_experiment"] == old["r20_shared_normal_experiment"]
    assert saved["inherited_r20_force_control"] == old["force_control"]
    for key in ("joint_admittance", "phase_seconds", "constrained_preload", "scoring", "grip_gate",
            "load_transfer_probe", "probe_confirmation", "r17_bias_budget_experiment", "nominal_profiles"):
        assert saved[key] == old[key]
    assert saved["schedule"][0]["admission_policy_revision"] == r21.PROTOCOL_FORMAT
    value = {"protocol": saved, "rows": [{"kind": "ycb_can", "seed": 3414}],
        "controller_revision": "public_ycb_constrained_preload_r14"}
    runner.__globals__["previous"].atomic_json(tmp_path / "summary.json", value)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["protocol"] == saved
    assert summary["rows"][0]["target_allocator_revision"] == r21.r20.PROTOCOL_FORMAT
    assert summary["rows"][0]["admission_policy_revision"] == r21.PROTOCOL_FORMAT
    assert summary["controller_revision"] == "public_ycb_bias_budget_development_r17"
    runner.__globals__["previous"].atomic_json(tmp_path / "result.json",
        {"kind": "ycb_can", "seed": 3414, "controller_revision": "public_ycb_joint_admittance_r13"})
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["controller_class"] == "ProvisionalPreloadBudgetHand"
    assert result["target_allocator_revision"] == r21.r20.PROTOCOL_FORMAT
    assert result["admission_policy_revision"] == r21.PROTOCOL_FORMAT


def test_constructor_error_does_not_fabricate_runtime_hands(tmp_path):
    runner = r21.isolated_runner(PROFILE, metadata_declaration())
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json",
        {"kind": "ycb_can", "seed": 3414, "runner_error": "CONSTRUCTION_FAILED"})
    value = json.loads((tmp_path / "metadata.json").read_text())
    assert value["runner_error"] == "CONSTRUCTION_FAILED"
    assert value["missing_runtime_allocator_binding"] and value["missing_runtime_controller_binding"]
    assert "shared_normal_runtime_bindings" not in value and "effective_hand_instances" not in value


def test_invalid_output_or_case_cannot_start(monkeypatch, tmp_path):
    monkeypatch.setattr(r21, "isolated_runner", lambda *args: pytest.fail("Invalid run started"))
    with pytest.raises(FileExistsError):
        r21.run(tmp_path, PROFILE, (("ycb_can", 3414),))
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r21.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),))
