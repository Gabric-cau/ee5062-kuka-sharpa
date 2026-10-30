"""R24 nested hand-factory, serializer and snapshot contracts without integration."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import pytest

import develop_public_ycb_fresh_contact_r24 as r24
from test_public_ycb_bias_budget_r17 import real_model


r23 = r24.r23
PROFILE = r23.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def declaration():
    protocol = json.loads((r24.BASELINE / "protocol.json").read_text())
    return {"inherited_r23_declaration": protocol[r23.DECLARATION],
        "inherited_r23_source_hashes": {}, "preflight_input_sha256": {},
        "inherited_r23_force_control": protocol["force_control"],
        "changed_control": "unit fresh contact branch"}


def archived_r22_adapter(monkeypatch):
    original_clone = r24.clone

    def clone(function, **bindings):
        cloned = original_clone(function, **bindings)
        if function is not r23.r22.controller_for:
            return cloned

        def adapted(*args, **kwargs):
            task, source, hands, estimated = cloned(*args, **kwargs)
            if hashlib.sha256(source.encode()).hexdigest() == r23.probe.INHERITED_SOURCE_SHA256:
                return task, source, hands, estimated
            case = r23.R22_DIRECTORY / "ycb_can_bias_budget_r17_seed3414"
            frozen = (case / "derived_controller.py").read_text()
            assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(frozen))
            namespace = dict(task.__globals__)
            exec(compile(frozen, "<actual_r22_fixture>", "exec"), namespace)
            bound = namespace["run_task"]
            assert r23.r22.alignment._code_semantics(task.__code__) == r23.r22.alignment._code_semantics(bound.__code__)
            bound.__dict__.update(task.__dict__)
            transform = json.loads((case / "metadata.json").read_text())["alignment_source_transform"]
            bound.r22_alignment_transform = copy.deepcopy(transform)
            bound.alignment_source_transform = copy.deepcopy(transform)
            return bound, frozen, hands, estimated

        return adapted

    monkeypatch.setattr(r24, "clone", clone)


def real_factory(monkeypatch, tmp_path, real_model, *, runner=False):
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    archived_r22_adapter(monkeypatch)
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    r13 = r24.fresh.r13
    guard = r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    owner = r24.isolated_runner(PROFILE, declaration()) if runner else None
    factory = owner.__globals__["controller_for"] if runner else r24.controller_for
    task, source, hands, estimated = factory("ycb_can", tmp_path, [], guard, profile, .349)
    estimated.bind(kin)
    config = r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    task.__globals__["HandGraspController"](real_model, config)
    task.__globals__["HandGraspController"](real_model, config)
    return owner, task, source, hands


def test_actual_nested_factory_uses_new_hand_twice_and_retains_whole_r23_arm(monkeypatch, tmp_path, real_model):
    _, task, source, hands = real_factory(monkeypatch, tmp_path, real_model)
    archived = r24.BASELINE / "ycb_can_bias_budget_r17_seed3414/derived_controller.py"
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(archived.read_text()))
    assert len(hands) == 2 and hands[0] is not hands[1]
    assert all(type(hand) is r24.fresh.FreshContactTrackingHand for hand in hands)
    assert all(hand.additional_bias_limit_rad == .10 for hand in hands)
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.
    for hand in hands:
        assert hand.write.__func__ is r24.fresh.provisional.r16.BiasRecoveryHand.write
        assert hand.after_step.__func__ is r24.fresh.provisional.r16.BiasRecoveryHand.after_step
        assert hand.admit_probe.__func__ is r24.fresh.provisional.r16.BiasRecoveryHand.admit_probe


def test_complete_nested_serializer_records_actual_hands_sources_and_all_prior_telemetry(monkeypatch, tmp_path, real_model):
    runner, task, _, hands = real_factory(monkeypatch, tmp_path, real_model, runner=True)
    writer = runner.__globals__["previous"].atomic_json
    for name in ("result.json", "metadata.json"):
        writer(tmp_path / name, {"kind": "ycb_can", "seed": 3414, "controller_revision": "unit_no_integration"})
        value = json.loads((tmp_path / name).read_text())
        assert value["runner_revision"] == r24.PROTOCOL_FORMAT
        assert value["controller_class"] == r24.CONTROLLER_CLASS
        assert value["alignment_settling"]["attempts"] == []
        assert value["actual_wrist_probe"]["attempts"] == []
        assert value["fresh_contact_tracking"]["hands"] == [hand.fresh_contact_telemetry() for hand in hands]
        assert value["actual_wrist_probe_source_transform"] == task.actual_wrist_probe_source_transform
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["effective_hand_controller_types"] == [r24.CONTROLLER_CLASS] * 2
    assert metadata["effective_hand_budgets_rad"] == [.10, .10]
    assert len(metadata["shared_normal_runtime_bindings"]) == 2
    for binding in metadata["shared_normal_runtime_bindings"]:
        assert binding["controller_class"] == r24.CONTROLLER_CLASS
        assert binding["tracking"]["method_source_file"] == "scripts/fresh_contact_tracking_controller_hooks.py"
        assert binding["tracking"]["allocator_is_shared_normal"] is True
        assert binding["tracking"]["same_code_as_original"] is False
        assert binding["writer_is_frozen_r16"] is True


def test_final_nested_snapshot_callback_accepts_actual_adapter_signature():
    runner = r24.isolated_runner(PROFILE, declaration())
    result = runner.__globals__["snapshot"](PROFILE)
    assert result == {name: r24.sha256(r24.PROJECT / name) for name in r24.NEW_SOURCES}


def test_tagging_preserves_original_physics_probe_and_strict_gates():
    before = json.loads((r24.BASELINE / "protocol.json").read_text())
    value = r24.tagged_protocol(copy.deepcopy(before), declaration())
    assert value["format"] == r24.PROTOCOL_FORMAT
    assert value["controller_class"] == r24.CONTROLLER_CLASS
    assert r24.DECLARATION in value and r23.DECLARATION not in value
    for key in ("phase_seconds", "scoring", "load_transfer_probe", "actual_wrist_probe", "alignment_settling"):
        assert value[key] == before[key]
    assert all(row["controller_class"] == r24.CONTROLLER_CLASS for row in value["schedule"])


def test_conflicting_inherited_snapshot_cannot_be_replaced():
    relative = Path(__file__).relative_to(r24.PROJECT).as_posix()
    declared = declaration()
    declared["inherited_r23_source_hashes"] = {relative: r24.sha256(r24.PROJECT / relative)}
    declared["preflight_input_sha256"] = {relative: "0" * 64}
    with pytest.raises(ValueError, match="frozen input changed"):
        r24.snapshot(PROFILE, declared)


def test_wrong_summary_or_existing_output_or_wrong_case_rejected_before_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(r23, "validate_inputs", lambda *a: pytest.fail("Invalid SHA reached inherited preflight"))
    with pytest.raises(ValueError, match="explicit finalized R23"):
        r24.validate_inputs(PROFILE, "0" * 64)
    monkeypatch.setattr(r24, "validate_inputs", lambda *a: pytest.fail("Invalid run reached preflight"))
    with pytest.raises(FileExistsError):
        r24.run(tmp_path, PROFILE, (("ycb_can", 3414),), expected_baseline_summary_sha256=r24.BASELINE_SHA256)
    with pytest.raises(ValueError, match="isolates"):
        r24.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),), expected_baseline_summary_sha256=r24.BASELINE_SHA256)
