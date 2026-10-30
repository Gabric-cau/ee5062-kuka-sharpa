"""Focused R26 orientation-candidate contracts; no native integration run."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import pytest

import develop_public_ycb_orientation_candidate_r26 as r26
from test_public_ycb_bias_budget_r17 import real_model
from test_public_ycb_rate_limited_probe_r25 import factory as r25_factory


PROFILE = r26.CANDIDATE_PROFILE


def test_profile_rotation_is_read_exactly_and_fail_closed_for_invalid_shapes(tmp_path):
    expected = np.asarray(json.loads(PROFILE.read_text(encoding="utf-8-sig"))["desired_world_wrist_rotation"])
    np.testing.assert_array_equal(r26.load_nominal_rotation(PROFILE), expected)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"desired_world_wrist_rotation": [[1., 0.], [0., 1.]]}))
    with pytest.raises(ValueError, match="finite 3x3"):
        r26.load_nominal_rotation(invalid)


def test_ast_replacement_keeps_r3_pose_rotation_composition_and_does_not_mutate_source():
    source = """
def run_task():
    _nominal_wrist_rotation = kin.rotation.copy()
    pose_rotation = np.eye(3)
    kin.rotation = pose_rotation @ _nominal_wrist_rotation
""".strip() + "\n"
    rotation = np.diag([1., 1., 1.])
    transformed = r26.replace_nominal_wrist_rotation(source, rotation)
    assert source == """\ndef run_task():
    _nominal_wrist_rotation = kin.rotation.copy()
    pose_rotation = np.eye(3)
    kin.rotation = pose_rotation @ _nominal_wrist_rotation
""".strip() + "\n"
    tree = ast.parse(transformed)
    initializer = next(node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "_nominal_wrist_rotation" for target in node.targets))
    assert "np.asarray" in ast.unparse(initializer.value)
    assert "pose_rotation @ _nominal_wrist_rotation" in ast.unparse(tree)
    np.testing.assert_array_equal(rotation, np.eye(3))


def test_real_nested_factory_retains_r25_probe_binding_and_changes_only_orientation_source(
        monkeypatch, tmp_path, real_model):
    # This is the same 3.10 archived-arm adapter used by the R25 contract tests.
    r25_factory(monkeypatch, tmp_path / "r25_fixture", real_model)
    profile = json.loads(PROFILE.read_text(encoding="utf-8-sig"))["ycb_can"]
    rotation = r26.load_nominal_rotation(PROFILE)
    task, source, hands, _ = r26.controller_for("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349,
        nominal_wrist_rotation=rotation)
    assert hands == []
    assert task.__globals__["_r23_probe_telemetry"] is task.actual_wrist_probe_telemetry
    assert task.r25_rate_limited_probe_transform["readable_source_unchanged"] is True
    assert task.r26_orientation_candidate_transform["nominal_wrist_rotation_source"] == "profile.desired_world_wrist_rotation"
    assert task.r26_orientation_candidate_transform["r3_pose_rotation_composition_preserved"] is True
    tree = ast.parse(source)
    assert any(isinstance(node, ast.Assign) and "np.asarray" in ast.unparse(node.value)
        and any(isinstance(target, ast.Name) and target.id == "_nominal_wrist_rotation"
                for target in node.targets) for node in ast.walk(tree))
    assert any(isinstance(node, ast.Assign) and ast.unparse(node.value) ==
        "pose_rotation @ _nominal_wrist_rotation" for node in ast.walk(tree))


def test_actual_plan_composes_measured_pose_with_rotated_candidate_not_baseline(monkeypatch, tmp_path, real_model):
    r25_factory(monkeypatch, tmp_path / "r25_fixture", real_model)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    rotation = r26.load_nominal_rotation(PROFILE)
    _, source, _, _ = r26.controller_for("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349,
        nominal_wrist_rotation=rotation)
    body = ast.parse(source).body[0].body
    initializer = next(node for node in body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_nominal_wrist_rotation"
                for target in node.targets))
    plan = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == "plan")
    angle = .2
    measured_pose = np.array([[np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    kin = SimpleNamespace(copy_open_hand=lambda _: None)
    scope = {"np": np, "time": time, "kin": kin, "data": object(),
        "_latest_pose": {"SEARCH": {"accepted": True, "rotation_world_from_object": measured_pose.tolist()}}}
    # Execute the initializer and actual nested plan AST; an empty option list
    # reaches orientation assignment without invoking IK or native integration.
    mini = ast.Module(body=[copy.deepcopy(initializer), copy.deepcopy(plan)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mini), "<r26_actual_plan_contract>", "exec"), scope)
    assert scope["plan"]([]) == (None, None)
    np.testing.assert_allclose(kin.rotation, measured_pose @ rotation, atol=1e-15)
    np.testing.assert_allclose(kin.approach_axis, (measured_pose @ rotation)[:, 2], atol=1e-15)
    baseline_rotation = r26.load_nominal_rotation(r26.BASELINE_PROFILE)
    assert not np.allclose(rotation, baseline_rotation, atol=.01)
    assert not np.allclose(kin.rotation, measured_pose @ baseline_rotation, atol=.01)


def test_protocol_and_snapshot_are_r26_bound_to_r25_and_profile(monkeypatch):
    original_validate = r26.r25.validate_inputs
    calls = []

    def baseline_validator(profile_path, expected):
        calls.append(profile_path)
        return original_validate(profile_path, expected)

    monkeypatch.setattr(r26.r25, "validate_inputs", baseline_validator)
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    assert calls == [r26.BASELINE_PROFILE]
    value = json.loads((r26.BASELINE / "protocol.json").read_text())
    tagged = r26.tagged_protocol(copy.deepcopy(value), declaration)
    assert tagged["format"] == r26.PROTOCOL_FORMAT
    assert tagged[r26.DECLARATION]["nominal_wrist_rotation_source"] == "profile.desired_world_wrist_rotation"
    assert tagged["r25_rate_limited_probe_experiment"]
    assert tagged["r26_orientation_candidate_experiment"]["r3_pose_rotation_composition"] == \
        "kin.rotation = pose_rotation @ _nominal_wrist_rotation"
    snapshot = r26.snapshot(PROFILE, declaration)
    assert set(r26.NEW_SOURCES) <= set(snapshot)
    assert snapshot["scripts/develop_public_ycb_rate_limited_probe_r25.py"] == \
        r26.sha256(r26.PROJECT / "scripts/develop_public_ycb_rate_limited_probe_r25.py")
    assert declaration["candidate_profile"]["sha256"] == r26.CANDIDATE_PROFILE_SHA256
    assert snapshot[PROFILE.relative_to(r26.PROJECT).as_posix()] == r26.CANDIDATE_PROFILE_SHA256
    assert len(declaration["inherited_r25_source_hashes"]) == 207
    assert all(snapshot[path] == digest for path, digest in declaration["inherited_r25_source_hashes"].items())
    assert declaration["candidate_hand_profile_changes"]["contact_closure"]["candidate"]["pinky"] > 0.


def test_frozen_candidate_rejects_fixed_baseline_profile_and_changed_candidate_hash(monkeypatch):
    with pytest.raises(ValueError, match="separately frozen rotated candidate"):
        r26.validate_inputs(r26.BASELINE_PROFILE, r26.BASELINE_SHA256)
    monkeypatch.setattr(r26, "CANDIDATE_PROFILE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="separately frozen rotated candidate"):
        r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)


def test_isolated_runner_serializer_records_candidate_without_changing_r25_metadata(monkeypatch, tmp_path, real_model):
    r25_factory(monkeypatch, tmp_path / "r25_fixture", real_model)
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    runner = r26.isolated_runner(PROFILE, declaration)
    assert runner.__globals__["snapshot"](PROFILE) == r26.snapshot(PROFILE, declaration)
    profile = json.loads(PROFILE.read_text(encoding="utf-8-sig"))["ycb_can"]
    task, _, _, _ = runner.__globals__["controller_for"](
        "ycb_can", tmp_path, [], SimpleNamespace(), profile, .349)
    writer = runner.__globals__["previous"].atomic_json
    writer(tmp_path / "metadata.json", {"kind": "ycb_can", "seed": 3414, "controller_revision": "unit"})
    value = json.loads((tmp_path / "metadata.json").read_text())
    assert value["runner_revision"] == r26.PROTOCOL_FORMAT
    assert value["rate_limited_probe_source_transform"] == task.r25_rate_limited_probe_transform
    assert value["orientation_candidate_source_transform"] == task.r26_orientation_candidate_transform
    np.testing.assert_array_equal(value["nominal_wrist_rotation_world"],
        declaration["nominal_wrist_rotation_world"])
    assert value["inherited_r25_orientation_audit_does_not_audit_r26"] is True


def test_rotated_hand_profile_enters_actual_nested_factory_and_fixed_profile_is_rejected(monkeypatch, tmp_path, real_model):
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario

    r25_factory(monkeypatch, tmp_path / "r25_fixture", real_model)
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    runner = r26.isolated_runner(PROFILE, declaration)
    profile_document = json.loads(PROFILE.read_text())
    spec, config, profile = runner.__globals__["previous"].selected_profile("ycb_can", profile_document["ycb_can"])
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    guard = r26.r25.r24.fresh.r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    task, _, hands, estimated = runner.__globals__["controller_for"](
        "ycb_can", tmp_path, [], guard, profile, .349)
    estimated.bind(kin)
    task.__globals__["HandGraspController"](real_model, config)
    assert len(hands) == 1
    assert hands[0].profile == profile_document["ycb_can"]
    assert hands[0].profile["contact_closure"]["pinky"] > 0.
    assert hands[0].profile["force_targets_n"]["pinky"] > 0.
    assert tuple(spec.aperture_offset) == tuple(profile_document["ycb_can"]["aperture_offset"])
    fixed_profile = json.loads(r26.BASELINE_PROFILE.read_text())["ycb_can"]
    with pytest.raises(ValueError, match="runtime hand profile differs"):
        runner.__globals__["controller_for"]("ycb_can", tmp_path / "wrong", [], guard, fixed_profile, .349)


def test_run_passes_candidate_file_to_actual_runner_argument(monkeypatch, tmp_path):
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    calls = []
    monkeypatch.setattr(r26, "validate_inputs", lambda path, expected: declaration)

    def factory(profile_path, declared):
        assert profile_path == PROFILE
        assert declared == declaration

        def no_integration_runner(output, actual_profile, cases):
            calls.append((output, actual_profile, cases))
            return {"unit": "no native integration"}

        return no_integration_runner

    monkeypatch.setattr(r26, "isolated_runner", factory)
    output = tmp_path / "new"
    cases = (("ycb_can", 3414),)
    assert r26.run(output, PROFILE, cases, expected_baseline_summary_sha256=r26.BASELINE_SHA256) == \
        {"unit": "no native integration"}
    assert calls == [(output, PROFILE, cases)]


def test_wrong_baseline_and_case_fail_before_any_candidate_run(tmp_path):
    with pytest.raises(ValueError, match="explicit finalized R25"):
        r26.validate_inputs(PROFILE, "0" * 64)
    with pytest.raises(FileExistsError):
        r26.run(tmp_path, PROFILE, (("ycb_can", 3414),), expected_baseline_summary_sha256=r26.BASELINE_SHA256)
    with pytest.raises(ValueError, match="isolates"):
        r26.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),), expected_baseline_summary_sha256=r26.BASELINE_SHA256)
