"""Exercise the real nested R25 factory, metadata writer and snapshot callback."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import pytest

import develop_public_ycb_rate_limited_probe_r25 as r25
from test_public_ycb_bias_budget_r17 import real_model
from test_public_ycb_fresh_contact_r24 import archived_r22_adapter


r24 = r25.r24
PROFILE = r24.r23.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def declaration():
    protocol = json.loads((r25.BASELINE / "protocol.json").read_text())
    return {"inherited_r24_declaration": protocol[r24.DECLARATION],
        "inherited_r24_source_hashes": {}, "preflight_input_sha256": {},
        "inherited_r24_force_control": protocol["force_control"],
        "changed_control": "unit reference governor"}


def factory(monkeypatch, tmp_path, model, *, runner=False):
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    archived_r22_adapter(monkeypatch)
    original = r24.controller_for

    def source_adapter(*args, **kwargs):
        task, source, hands, estimated = original(*args, **kwargs)
        if hashlib.sha256(source.encode()).hexdigest() == r25.governor.INHERITED_SOURCE_SHA256:
            return task, source, hands, estimated
        case = r25.BASELINE / "ycb_can_bias_budget_r17_seed3414"
        frozen = (case / "derived_controller.py").read_text()
        assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(frozen))
        namespace = dict(task.__globals__)
        exec(compile(frozen, "<actual_r24_fixture>", "exec"), namespace)
        bound = namespace["run_task"]
        assert r25.governor._code_semantics(task.__code__) == r25.governor._code_semantics(bound.__code__)
        bound.__dict__.update(task.__dict__)
        return bound, frozen, hands, estimated

    # Production remains byte-pinned; this fixture checks 3.13 against the real 3.10 source.
    monkeypatch.setattr(r24, "controller_for", source_adapter)
    data = mujoco.MjData(model)
    _, kin, _ = reset_task(model, data, Scenario(seed=3414, object_kind="ycb_can"))
    guard = r24.fresh.r13.r11.r9.r8.r6.LatchedPressureGuard(model, .349)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    owner = r25.isolated_runner(PROFILE, declaration()) if runner else None
    constructor = owner.__globals__["controller_for"] if runner else r25.controller_for
    task, source, hands, estimated = constructor("ycb_can", tmp_path, [], guard, profile, .349)
    estimated.bind(kin)
    config = r24.fresh.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    task.__globals__["HandGraspController"](model, config)
    task.__globals__["HandGraspController"](model, config)
    return owner, task, source, hands


def test_real_factory_retains_hand_writer_and_readable_arm_but_changes_probe_binding(monkeypatch, tmp_path, real_model):
    _, task, source, hands = factory(monkeypatch, tmp_path, real_model)
    frozen = r25.BASELINE / "ycb_can_bias_budget_r17_seed3414/derived_controller.py"
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(frozen.read_text()))
    assert task.__globals__["_r23_probe_telemetry"] is task.actual_wrist_probe_telemetry
    assert isinstance(task.actual_wrist_probe_telemetry, r25.governor.RateLimitedProbeTelemetry)
    assert len(hands) == 2 and hands[0] is not hands[1]
    for hand in hands:
        assert type(hand) is r24.fresh.FreshContactTrackingHand
        assert hand.additional_bias_limit_rad == .10
        assert hand.after_step.__func__ is r24.fresh.provisional.r16.BiasRecoveryHand.after_step
        assert hand.write.__func__ is r24.fresh.provisional.r16.BiasRecoveryHand.write
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.


def test_complete_nested_serializer_keeps_real_governor_and_inherited_hand_records(monkeypatch, tmp_path, real_model):
    owner, task, _, hands = factory(monkeypatch, tmp_path, real_model, runner=True)
    writer = owner.__globals__["previous"].atomic_json
    for filename in ("metadata.json", "result.json"):
        writer(tmp_path / filename, {"kind": "ycb_can", "seed": 3414, "controller_revision": "unit_no_integration"})
        value = json.loads((tmp_path / filename).read_text())
        assert value["runner_revision"] == r25.PROTOCOL_FORMAT
        assert value["rate_limited_probe_source_transform"] == task.r25_rate_limited_probe_transform
        assert value["actual_wrist_probe"] == task.actual_wrist_probe_telemetry.as_dict()
        assert value["actual_wrist_probe"]["format"] == "rate_limited_actual_wrist_probe_policy_v1"
        assert value["fresh_contact_tracking"]["hands"] == [h.fresh_contact_telemetry() for h in hands]
        if filename == "metadata.json":
            assert value["effective_hand_controller_types"] == [r25.CONTROLLER_CLASS] * 2
        assert value["alignment_settling"]["attempts"] == []


def test_actual_final_snapshot_callback_accepts_all_nested_adapter_signatures():
    owner = r25.isolated_runner(PROFILE, declaration())
    actual = owner.__globals__["snapshot"](PROFILE)
    assert actual == {name: r25.sha256(r25.PROJECT / name) for name in r25.NEW_SOURCES}


def test_protocol_declares_new_reference_binding_and_preserves_original_gates():
    before = json.loads((r25.BASELINE / "protocol.json").read_text())
    actual = r25.tagged_protocol(copy.deepcopy(before), declaration())
    assert actual["format"] == r25.PROTOCOL_FORMAT
    assert r25.DECLARATION in actual and r24.DECLARATION not in actual
    assert actual["probe_reference_governor"] == r25.governor.PARAMETERS
    for key in ("phase_seconds", "scoring", "load_transfer_probe", "actual_wrist_probe", "alignment_settling"):
        assert actual[key] == before[key]
    assert all(row["runner_revision"] == r25.PROTOCOL_FORMAT for row in actual["schedule"])


def test_conflicting_snapshot_cannot_replace_a_frozen_input():
    declared = declaration()
    name = Path(__file__).relative_to(r25.PROJECT).as_posix()
    declared["inherited_r24_source_hashes"] = {name: r25.sha256(r25.PROJECT / name)}
    declared["preflight_input_sha256"] = {name: "0" * 64}
    with pytest.raises(ValueError, match="frozen input changed"):
        r25.snapshot(PROFILE, declared)


def test_wrong_summary_existing_output_and_other_cases_are_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="explicit finalized R24"):
        r25.validate_inputs(PROFILE, "0" * 64)
    monkeypatch.setattr(r25, "validate_inputs", lambda *a: pytest.fail("Invalid run reached preflight"))
    with pytest.raises(FileExistsError):
        r25.run(tmp_path, PROFILE, (("ycb_can", 3414),), expected_baseline_summary_sha256=r25.BASELINE_SHA256)
    with pytest.raises(ValueError, match="isolates"):
        r25.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),), expected_baseline_summary_sha256=r25.BASELINE_SHA256)
