"""R20 isolation and runtime bindings; no simulation steps or full task are run."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_shared_normal_r20 as r20
from test_public_ycb_bias_budget_r17 import PROFILE, real_model


def baseline():
    return json.loads(r20.R19_SUMMARY.read_text())


def test_preflight_preserves_failed_r19_with_observed_normal_child_exit():
    declaration = r20.validate_inputs(PROFILE)
    old = baseline()["protocol"]
    assert declaration["changed_parameters"] == {"tracking_and_gate_allocator": {
        "before": "r13_single_1_5g_then_fixed_1g_postcheck", "after": "shared_normal_tracking_target_v1"}}
    assert declaration["active_r19_declaration"] == old["r19_extended_budget_experiment"]
    assert declaration["inherited_r19_declaration"] == old["r19_extended_budget_experiment"]
    assert declaration["additional_bias_limit_rad"] == .10
    assert declaration["close_seconds"] == 32. and declaration["arm_interpolation_seconds"] == 16.
    assert not declaration["baseline_task_success"] and not declaration["baseline_held_lift"]
    assert declaration["baseline_child_exit_observed"] and declaration["baseline_child_returncode"] == 0
    assert declaration["allocator_paths"] == ["tracking", "gate_only"]
    assert declaration["lp_time_limit_per_solve_s"] == .1


def test_snapshot_inherits_all90_sources_and_closes_each_new_preflight_input():
    values = r20.snapshot(PROFILE)
    before = baseline()["protocol"]["source_hashes_before"]
    assert len(before) == 90
    assert all(values[key] == value for key, value in before.items())
    assert all(values[key] == value for key, value in r20.PREFLIGHT_HASHES.items())
    for relative in (*r20.NEW_SOURCES, *r20.OPTIONAL_AUDIT_SOURCES):
        if (r20.PROJECT / relative).exists():
            assert values[relative] == r20.r19.r17.r16.sha256(r20.PROJECT / relative)


@pytest.mark.parametrize("relative", [
    "scripts/shared_normal_tracking_targets.py",
    "scripts/shared_normal_controller_hooks.py",
    "results/jobs/r19_bias10_close32_seed3414_20261022/status.json",
    "results/r19_extended_budget_audit_20261022/receipt.json",
])
def test_changed_allocator_or_baseline_receipt_is_rejected(monkeypatch, relative):
    sha = r20.r19.r17.r16.sha256
    monkeypatch.setattr(r20.r19.r17.r16, "sha256", lambda path:
        "0" * 64 if Path(path) == r20.PROJECT / relative else sha(path))
    with pytest.raises(ValueError, match="R20 preflight input SHA256"):
        r20.validate_inputs(PROFILE)


def test_controller_factory_has_identical_r19_code_and_only_r17_controller_binding_changes():
    clone, old = r20._controller_factory, r20.r19.controller_for
    assert clone.__code__ is old.__code__ and clone.__closure__ is old.__closure__
    for name, value in old.__globals__.items():
        assert clone.__globals__[name] is (r20._r17_proxy if name == "r17" else value)
    for name, value in vars(r20.r19.r17).items():
        assert getattr(r20._r17_proxy, name) is (r20.hooks.controller_for if name == "controller_for" else value)


def test_runtime_binding_records_resolved_functions_not_only_declared_tags():
    hand = object.__new__(r20.hooks.SharedNormalBudgetHand)
    hand._additional_bias_limit_rad = .10
    binding = r20.runtime_binding(hand)
    assert binding["controller_class"] == "SharedNormalBudgetHand"
    assert binding["additional_bias_limit_rad"] == .10
    for path in ("tracking", "gate_only"):
        value = binding[path]
        assert value["same_code_as_original"] and value["same_closure_as_original"]
        assert value["allocator_is_shared_normal"] and value["other_globals_match_except_warning_registry"]
        assert len(value["method_source_sha256"]) == len(value["method_bytecode_sha256"]) == 64
        assert value["allocator"] == {"module": "shared_normal_tracking_targets",
            "qualname": "allocate_tracking_targets",
            "source_sha256": r20.PREFLIGHT_HASHES["scripts/shared_normal_tracking_targets.py"]}
    assert binding["gate_only"]["r13_proxy_other_members_match"]
    assert all(binding[key] for key in ("writer_is_frozen_r16", "after_step_is_frozen_r16", "admit_probe_is_frozen_r16"))
    hand.regulate = r20.hooks.r13.JointAdmittanceProbeHand.regulate.__get__(hand)
    changed = r20.runtime_binding(hand)
    assert not changed["tracking"]["allocator_is_shared_normal"]
    assert changed["tracking"]["allocator"]["module"] == "develop_public_ycb_joint_admittance_r13"


def test_real_first_retry_factory_and_metadata_keep_r19_clocks_bytes_scene_and_new_type(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    declaration = r20.validate_inputs(PROFILE)
    runner = r20.isolated_runner(PROFILE, declaration)
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, hands, estimated = runner.__globals__["controller_for"]("ycb_can", tmp_path,
        [], SimpleNamespace(), profile, .349)
    estimated.bind(kin)
    config = r20.hooks.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    task.__globals__["HandGraspController"](real_model, config)
    task.__globals__["HandGraspController"](real_model, config)
    assert len(hands) == 2 and hands[0] is not hands[1]
    assert all(type(hand) is r20.hooks.SharedNormalBudgetHand for hand in hands)
    assert [hand.additional_bias_limit_rad for hand in hands] == [.10, .10]
    assert r20.hashlib.sha256(source.encode()).hexdigest() == r20.r19.EXPECTED_R18_DERIVED_SHA256
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.
    assert runner.__globals__["PHASE_SECONDS"]["CLOSE"] == 32.
    assert runner.__globals__["previous"].RichNativeAudit is r20.r19.r17.TorqueNativeAudit
    scene = tmp_path / "scene.xml"
    runner.__globals__["build_pick_place_scene"](scene, "ycb_can", physics_profile="physical-v6")
    value = {"kind": "ycb_can", "seed": 3414, "runner_error": None,
        "controller_revision": "public_ycb_constrained_preload_r14"}
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json", value)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["effective_hand_controller_types"] == ["SharedNormalBudgetHand"] * 2
    assert saved["effective_hand_budgets_rad"] == [.10] * 2 and saved["effective_hand_instances"] == 2
    assert saved["shared_normal_runtime_bindings"] == [r20.runtime_binding(hand) for hand in hands]
    assert saved["effective_task_phase_seconds"]["CLOSE"] == saved["effective_runner_phase_seconds"]["CLOSE"] == 32.
    assert saved["controller_class"] == "SharedNormalBudgetHand"
    assert saved["controller_revision"] == "public_ycb_bias_budget_development_r17"
    assert saved["controller_revision_scope"] == r20.CONTROLLER_REVISION_SCOPE
    assert saved["target_allocator_revision"] == saved["runner_revision"] == r20.PROTOCOL_FORMAT
    assert saved["scene_unchanged_since_load"]
    assert saved["scene_sha256_before_load"] == r20.r19.r17.r16.sha256(scene)
    # This local test may run on Mac; the formal Linux auditor keeps exact scene SHA.
    actual_xml = ET.parse(scene).getroot()
    old_xml = ET.parse(r20.R19_DIRECTORY / baseline()["rows"][0]["path"] / "scene.xml").getroot()
    actual_compiler, old_compiler = actual_xml.find("compiler"), old_xml.find("compiler")
    assert actual_compiler.get("meshdir") == str(r20.PROJECT / "vendor/mujoco_menagerie")
    assert old_compiler.get("meshdir").endswith("/vendor/mujoco_menagerie")
    actual_compiler.set("meshdir", "SAME_FROZEN_ASSET_ROOT")
    old_compiler.set("meshdir", "SAME_FROZEN_ASSET_ROOT")
    assert ET.tostring(actual_xml) == ET.tostring(old_xml)


def test_nested_protocol_result_summary_tags_keep_only_allocator_change(tmp_path):
    declaration = r20.validate_inputs(PROFILE)
    runner = r20.isolated_runner(PROFILE, declaration)
    old = baseline()["protocol"]
    protocol = copy.deepcopy(old)
    runner.__globals__["previous"].atomic_json(tmp_path / "protocol.json", protocol)
    saved = json.loads((tmp_path / "protocol.json").read_text())
    assert saved["format"] == saved["runner_revision"] == r20.PROTOCOL_FORMAT
    assert saved["r20_shared_normal_experiment"] == declaration
    assert "r19_extended_budget_experiment" not in saved
    assert saved["inherited_r19_extended_budget_experiment"] == old["r19_extended_budget_experiment"]
    for key in ("joint_admittance", "phase_seconds", "constrained_preload", "scoring",
            "load_transfer_probe", "probe_confirmation", "r17_bias_budget_experiment", "nominal_profiles"):
        assert saved[key] == old[key]
    assert saved["schedule"][0]["controller_class"] == "SharedNormalBudgetHand"
    summary = {"protocol": saved, "rows": [{"kind": "ycb_can", "seed": 3414}],
        "controller_revision": "public_ycb_constrained_preload_r14"}
    runner.__globals__["previous"].atomic_json(tmp_path / "summary.json", summary)
    saved_summary = json.loads((tmp_path / "summary.json").read_text())
    assert saved_summary["protocol"] == saved
    assert saved_summary["rows"][0]["target_allocator_revision"] == r20.PROTOCOL_FORMAT
    assert saved_summary["controller_revision"] == "public_ycb_bias_budget_development_r17"
    assert saved_summary["controller_class"] == "SharedNormalBudgetHand"
    result = {"kind": "ycb_can", "seed": 3414, "controller_revision": "public_ycb_joint_admittance_r13"}
    runner.__globals__["previous"].atomic_json(tmp_path / "result.json", result)
    saved_result = json.loads((tmp_path / "result.json").read_text())
    assert saved_result["controller_class"] == "SharedNormalBudgetHand"
    assert saved_result["target_allocator_revision"] == r20.PROTOCOL_FORMAT


def test_constructor_error_keeps_missing_binding_flags_instead_of_fabricated_actual_instances(tmp_path):
    runner = r20.isolated_runner(PROFILE, r20.validate_inputs(PROFILE))
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json",
        {"kind": "ycb_can", "seed": 3414, "runner_error": "CONSTRUCTION_FAILED"})
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["runner_error"] == "CONSTRUCTION_FAILED"
    assert saved["missing_runtime_allocator_binding"] and saved["missing_runtime_controller_binding"]
    assert "shared_normal_runtime_bindings" not in saved and "effective_hand_instances" not in saved


def test_no_old_output_case_or_sweep_can_start(monkeypatch, tmp_path):
    monkeypatch.setattr(r20, "isolated_runner", lambda *args: pytest.fail("Invalid run started"))
    with pytest.raises(FileExistsError):
        r20.run(tmp_path, PROFILE, (("ycb_can", 3414),))
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r20.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),))
