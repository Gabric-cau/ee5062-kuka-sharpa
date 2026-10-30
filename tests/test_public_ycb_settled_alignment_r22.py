"""Next-run preflight and serialization contracts; no task or integration is run."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_settled_alignment_r22 as r22

PROFILE = r22.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def recorded_protocol():
    return json.loads((r22.R21_DIRECTORY / "protocol.json").read_text())


def serialization_declaration():
    prior = recorded_protocol()
    return {"inherited_r21_declaration": copy.deepcopy(prior["r21_provisional_preload_experiment"]),
        "inherited_r21_force_control": prior["force_control"], "preflight_input_sha256": {}}


@pytest.mark.parametrize("value", [None, "PENDING", "A" * 64, "a" * 63, "g" * 64])
def test_baseline_sha_must_be_explicit_final_lowercase_hex(value):
    with pytest.raises(ValueError, match="explicit lowercase"):
        r22.checked_sha256(value)
    assert r22.checked_sha256("a" * 64) == "a" * 64


@pytest.mark.parametrize("changed", [{"state": "RUNNING", "terminal": False, "returncode": None},
    {"returncode": 1}, {"returncode": False}, {"child_exit_observed": False}])
def test_baseline_requires_actual_observed_normal_child_exit(changed):
    status = {"state": "FINISHED", "terminal": True, "returncode": 0, "child_exit_observed": True, **changed}
    request = json.loads((r22.R21_JOB / "request.json").read_text())
    with pytest.raises(ValueError, match="observed FINISHED rc0"):
        r22.check_exit(status, request, recorded_protocol(), PROFILE)


def test_partial_baseline_is_rejected_before_any_runner_or_inherited_preflight(monkeypatch, tmp_path):
    baseline, job = tmp_path / "baseline", tmp_path / "job"
    baseline.mkdir()
    job.mkdir()
    for root, name, value in ((baseline, "summary.json", {"complete": False}),
            (baseline, "protocol.json", {}), (job, "status.json", {"state": "RUNNING"}),
            (job, "request.json", {})):
        (root / name).write_text(json.dumps(value))
    monkeypatch.setattr(r22, "R21_DIRECTORY", baseline)
    monkeypatch.setattr(r22, "R21_JOB", job)
    monkeypatch.setattr(r22.r21, "validate_inputs", lambda *args: pytest.fail("Partial baseline reached inherited preflight"))
    expected = hashlib.sha256((baseline / "summary.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="complete unchanged 130-source"):
        r22.validate_inputs(PROFILE, expected)


def receipt_fixture():
    """Synthetic receipt binding fixture, never a recorded R21 outcome."""
    row = {"path": "unit_case", "success": False, "held_lift": False,
        "effective_hand_instances": 1, "native_audit": {"native_samples": 123}}
    protocol = {"source_hashes_before": {r22.R21_AUDITOR: r22.sha256(r22.PROJECT / r22.R21_AUDITOR)}}
    native = {"summary_sha256": "a" * 64, "format": "public_ycb_development_native_verification_r1",
        "complete": True, "unknowns": 0, "source_hashes_before_equal_after": True,
        "native_samples_contiguous_2ms": True, "audited_episodes": 1, "frozen_source_sha256": "c" * 64,
        "native_samples": 123, "rows": [{"path": "unit_case", "outcome": False, "held_lift": False, "unknown": False}]}
    strict = {"summary_sha256": "a" * 64, "format": "public_ycb_strict_native_audit_20261021",
        "accepted_evidence": True, "unknowns": 0, "denominator": 1, "rows": [{"path": "unit_case", "unknown": False,
            "checks": {"legacy_task_success": False, "held_lift": False}}]}
    command = {"summary_sha256": "a" * 64, "format": r22.R21_AUDIT_FORMAT, "protocol_sha256": "b" * 64,
        "script_sha256": protocol["source_hashes_before"][r22.R21_AUDITOR],
        **dict.fromkeys(r22.AUDIT_CHECKS, True), "additional_bias_limit_rad": .10,
        "close_seconds": 32., "arm_interpolation_seconds": 16., "cases": [{"case": "unit_case",
            "actual_controller_class": r22.CONTROLLER_CLASS, "effective_hand_instances": 1,
            "task_success": False, "held_lift": False,
            "actuation": {"all_cached_force_and_transmitted_torque_consistent": True},
            "provisional_policy": {"all_recorded_policy_rows_consistent": True}}]}
    return {"rows": [row]}, protocol, "a" * 64, "b" * 64, "c" * 64, native, strict, command


def test_valid_failed_baseline_with_variable_native_count_and_one_or_two_attempts_is_accepted():
    args = receipt_fixture()
    r22.check_receipts(*args)
    args[0]["rows"][0]["effective_hand_instances"] = 2
    args[-1]["cases"][0]["effective_hand_instances"] = 2
    args[0]["rows"][0]["native_audit"]["native_samples"] = args[-3]["native_samples"] = 46448
    r22.check_receipts(*args)


@pytest.mark.parametrize("which", ["unknown_native", "unknown_strict", "wrong_summary", "failed_policy"])
def test_unknown_misbound_or_failed_command_policy_receipt_cannot_authorize_next_run(which):
    args = receipt_fixture()
    if which == "unknown_native":
        args[-3]["unknowns"] = 1
    elif which == "unknown_strict":
        args[-2]["rows"][0]["unknown"] = True
    elif which == "wrong_summary":
        args[-1]["summary_sha256"] = "d" * 64
    else:
        args[-1]["all_recorded_provisional_policy_rows_consistent"] = False
    with pytest.raises(ValueError, match="missing, unknown or inconsistent"):
        r22.check_receipts(*args)


def test_real_factory_retains_original_transform_attributes_and_only_declares_new_alignment_source(tmp_path):
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, hands, _ = r22.controller_for("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349)
    assert hands == []
    assert task.r19_source_transform["derived_controller_sha256"] == r22.alignment.INHERITED_SOURCE_SHA256
    assert task.alignment_source_transform == task.r22_alignment_transform
    assert task.alignment_source_transform["transformed_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert task.alignment_source_transform["transformed_source_sha256"] != r22.alignment.INHERITED_SOURCE_SHA256
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.
    assert task.alignment_settling_telemetry.as_dict()["attempts"] == []


def test_runner_result_and_protocol_serializers_keep_r21_admission_explicitly_inherited(tmp_path):
    declaration = serialization_declaration()
    runner = r22.isolated_runner(PROFILE, declaration)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, _, hands, _ = runner.__globals__["controller_for"]("ycb_can", tmp_path,
        [], SimpleNamespace(), profile, .349)
    assert hands == []
    writer = runner.__globals__["previous"].atomic_json
    writer(tmp_path / "result.json", {"kind": "ycb_can", "seed": 3414, "controller_revision": "unit_no_task_run"})
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["runner_revision"] == r22.PROTOCOL_FORMAT
    assert result["admission_policy_revision"] == r22.r21.PROTOCOL_FORMAT
    assert result["controller_class"] == "ProvisionalPreloadBudgetHand"
    assert result["alignment_source_transform"] == task.alignment_source_transform
    assert result["alignment_settling"]["attempts"] == []
    assert result["inherited_r21_command_audit_does_not_audit_r22"] is True
    writer(tmp_path / "metadata.json", {"kind": "ycb_can", "seed": 3414, "runner_error": None,
        "controller_revision": "unit_no_task_run"})
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["alignment_settling"] == result["alignment_settling"]
    assert metadata["alignment_source_transform"] == result["alignment_source_transform"]
    assert metadata["arm_source_transform"] == task.r19_source_transform
    assert metadata["arm_source_transform_is_inherited_r19_prefix_only"] is True
    assert metadata["effective_hand_instances"] == 0
    writer(tmp_path / "protocol.json", copy.deepcopy(recorded_protocol()))
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["format"] == protocol["runner_revision"] == r22.PROTOCOL_FORMAT
    assert protocol[r22.DECLARATION] == declaration
    assert "r21_provisional_preload_experiment" not in protocol
    assert protocol["inherited_r21_provisional_preload_experiment"] == declaration["inherited_r21_declaration"]
    for key in ("joint_admittance", "phase_seconds", "constrained_preload", "load_transfer_probe", "scoring"):
        assert protocol[key] == recorded_protocol()[key]
    writer(tmp_path / "summary.json", {"protocol": protocol, "rows": [metadata], "complete": False})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["protocol"] == protocol
    assert summary["rows"][0] == metadata


def test_snapshot_closes_original_r21_and_each_new_source_without_running_baseline_preflight():
    original = r22.r21.snapshot(PROFILE)
    values = r22.snapshot(PROFILE, {"preflight_input_sha256": {}})
    assert len(original) == 130
    assert all(values[key] == value for key, value in original.items())
    assert all(values[name] == r22.sha256(r22.PROJECT / name) for name in r22.NEW_SOURCES)


def test_existing_output_or_other_case_is_rejected_before_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(r22, "validate_inputs", lambda *args: pytest.fail("Invalid run reached preflight"))
    with pytest.raises(FileExistsError):
        r22.run(tmp_path, PROFILE, (("ycb_can", 3414),), expected_baseline_summary_sha256="a" * 64)
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r22.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),), expected_baseline_summary_sha256="a" * 64)
