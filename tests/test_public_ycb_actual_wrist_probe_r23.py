"""R23 preflight and serializer contracts; no integration or actual run launch."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import develop_public_ycb_actual_wrist_probe_r23 as r23


PROFILE = r23.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def bind_archived_r22_when_host_unparse_differs(monkeypatch):
    """Keep the production SHA pin when the host formats identical AST differently."""
    original_factory = r23.r22.controller_for

    def factory(*args, **kwargs):
        original, source, hands, estimated = original_factory(*args, **kwargs)
        if hashlib.sha256(source.encode()).hexdigest() == r23.probe.INHERITED_SOURCE_SHA256:
            return original, source, hands, estimated
        case = r23.R22_DIRECTORY / "ycb_can_bias_budget_r17_seed3414"
        source = (case / "derived_controller.py").read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == r23.probe.INHERITED_SOURCE_SHA256
        namespace = dict(original.__globals__)
        exec(compile(source, "<actual_archived_r22_source>", "exec"), namespace)
        bound = namespace["run_task"]
        assert r23.r22.alignment._code_semantics(original.__code__) == r23.r22.alignment._code_semantics(bound.__code__)
        bound.__dict__.update(original.__dict__)
        recorded = json.loads((case / "metadata.json").read_text())["alignment_source_transform"]
        assert recorded["transformed_source_sha256"] == r23.probe.INHERITED_SOURCE_SHA256
        bound.r22_alignment_transform = copy.deepcopy(recorded)
        bound.alignment_source_transform = copy.deepcopy(recorded)
        return bound, source, hands, estimated

    monkeypatch.setattr(r23.r22, "controller_for", factory)


def serialization_declaration():
    prior = json.loads((r23.r22.R21_DIRECTORY / "protocol.json").read_text())
    inherited = {"inherited_r21_declaration": copy.deepcopy(prior["r21_provisional_preload_experiment"]),
        "inherited_r21_force_control": prior["force_control"], "preflight_input_sha256": {}}
    return prior, {"inherited_r22_declaration": inherited, "inherited_r22_force_control": "unit R22 scope",
        "inherited_r22_source_hashes": {}, "preflight_input_sha256": {}}


def exit_fixture():
    status = {"state": "FINISHED", "terminal": True, "child_exit_observed": True, "returncode": 0}
    protocol = {"source_hashes_before": {"scripts/run_supervised_experiment.py": "a" * 64},
        r23.r22.DECLARATION: {"r21_comparison_summary_sha256": "b" * 64}}
    request = {"supervisor_sha256": "a" * 64, "command": ["python",
        "scripts/develop_public_ycb_settled_alignment_r22.py", "--output",
        r23.R22_DIRECTORY.relative_to(r23.PROJECT).as_posix(), "--profile-json",
        PROFILE.relative_to(r23.PROJECT).as_posix(), "--expected-baseline-summary-sha256", "b" * 64,
        "--cases", "ycb_can:3414"]}
    return status, request, protocol, PROFILE


@pytest.mark.parametrize("changed", [{"state": "RUNNING", "terminal": False, "returncode": None},
    {"returncode": 1}, {"returncode": False}, {"child_exit_observed": False}])
def test_unobserved_failed_or_noninteger_exit_cannot_authorize_r23(changed):
    values = exit_fixture()
    r23.check_exit(*values)
    values[0].update(changed)
    with pytest.raises(ValueError, match="observed FINISHED rc0 R22"):
        r23.check_exit(*values)


def test_wrong_r21_pin_or_duplicate_output_in_supervisor_request_is_rejected():
    values = exit_fixture()
    values[1]["command"][values[1]["command"].index("--expected-baseline-summary-sha256") + 1] = "c" * 64
    with pytest.raises(ValueError, match="observed FINISHED rc0 R22"):
        r23.check_exit(*values)
    values = exit_fixture()
    values[1]["command"] += ["--output", "other"]
    with pytest.raises(ValueError):
        r23.check_exit(*values)


@pytest.mark.parametrize("value", [None, "PENDING", "A" * 64, "a" * 63, "g" * 64])
def test_baseline_sha_must_be_explicit_final_lowercase_hex(value):
    with pytest.raises(ValueError, match="explicit lowercase"):
        r23.checked_sha256(value)


def test_partial_baseline_rejected_before_source_or_runner_factory(monkeypatch, tmp_path):
    baseline, job = tmp_path / "baseline", tmp_path / "job"
    baseline.mkdir()
    job.mkdir()
    for root, name, value in ((baseline, "summary.json", {"complete": False}),
            (baseline, "protocol.json", {}), (job, "status.json", {"state": "RUNNING"}),
            (job, "request.json", {})):
        (root / name).write_text(json.dumps(value))
    monkeypatch.setattr(r23, "R22_DIRECTORY", baseline)
    monkeypatch.setattr(r23, "R22_JOB", job)
    monkeypatch.setattr(r23.r22, "snapshot", lambda *args: pytest.fail("Partial baseline reached source factory"))
    with pytest.raises(ValueError, match="complete unchanged 146-source R22"):
        r23.validate_inputs(PROFILE, r23.R22_SUMMARY_SHA256)


def receipt_fixture(tmp_path):
    names = ("summary", "protocol", "archive", "scene", "controller", "metadata", "result",
        "native", "raw_native", "trace", "model_loader", "alignment_auditor")
    files = {name: tmp_path / (name + ".unit") for name in names}
    for name, path in files.items():
        path.write_text(name)
    hashes = {path: r23.sha256(path) for path in files.values()}
    row = {"path": "unit_case", "success": False, "held_lift": False, "effective_hand_instances": 1,
        "native_audit": {"native_samples": 321}, "alignment_settling": {"attempts": [{}]}}
    native = {"summary_sha256": hashes[files["summary"]], "format": "public_ycb_development_native_verification_r1",
        "complete": True, "unknowns": 0, "source_hashes_before_equal_after": True,
        "native_samples_contiguous_2ms": True, "audited_episodes": 1, "frozen_source_sha256": hashes[files["archive"]],
        "native_samples": 321, "rows": [{"path": "unit_case", "outcome": False, "held_lift": False,
            "unknown": False, "native_samples": 321}]}
    strict = {"summary_sha256": hashes[files["summary"]], "format": "public_ycb_strict_native_audit_20261021",
        "accepted_evidence": True, "unknowns": 0, "denominator": 1, "rows": [{"path": "unit_case", "unknown": False,
            "checks": {"legacy_task_success": False, "held_lift": False}, "native_samples": 321,
            "result_sha256": hashes[files["result"]], "scene_sha256": hashes[files["scene"]]}]}
    alignment = {"summary_sha256": hashes[files["summary"]], "format": r23.ALIGNMENT_AUDIT_FORMAT,
        "accepted_alignment_evidence": True, "protocol_sha256": hashes[files["protocol"]],
        "script_sha256": hashes[files["alignment_auditor"]], **dict.fromkeys(r23.ALIGNMENT_CHECKS, True),
        "inputs_unchanged_during_audit": True,
        "source": {"source_count": 146, "all_archived_input_hashes_match": True,
            "before_after_manifest_equal": True, "archive_sha256": hashes[files["archive"]]},
        "case": "unit_case", "native_samples": 321, "native_timestep_s": .002, "actual_hand_instances": 1,
        "task_success": False, "held_lift": False, "alignments": [{"settled": True}], "inputs": {}}
    for name in ("summary", "protocol", "archive", "scene", "controller", "metadata", "result", "native", "trace", "model_loader"):
        path = files["raw_native"] if name == "native" else files[name]
        alignment["inputs"][name] = {"sha256": hashes[path], "bytes": path.stat().st_size}
    return {"rows": [row]}, hashes, files, native, strict, alignment


def test_failed_task_with_one_or_two_actual_hands_and_accepted_alignment_is_valid_baseline(tmp_path):
    args = receipt_fixture(tmp_path)
    r23.check_receipts(*args)
    args[0]["rows"][0].update(effective_hand_instances=2, alignment_settling={"attempts": [{}, {}]})
    args[-1].update(actual_hand_instances=2, alignments=[{"settled": True}, {"settled": True}])
    r23.check_receipts(*args)
    assert args[-1]["task_success"] is False


@pytest.mark.parametrize("change", ["native_unknown", "strict_unknown", "wrong_summary", "alignment_flag", "raw_binding"])
def test_misbound_unknown_or_unaccepted_alignment_receipts_failclosed(tmp_path, change):
    args = receipt_fixture(tmp_path)
    if change == "native_unknown":
        args[-3]["unknowns"] = 1
    elif change == "strict_unknown":
        args[-2]["rows"][0]["unknown"] = True
    elif change == "wrong_summary":
        args[-1]["summary_sha256"] = "0" * 64
    elif change == "alignment_flag":
        args[-1][r23.ALIGNMENT_CHECKS[0]] = False
    else:
        args[-1]["inputs"]["native"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="R22 .*inconsistent|input binding differs"):
        r23.check_receipts(*args)


def test_real_factory_preserves_r19_prefix_r22_alignment_and_new_r23_probe_attrs(monkeypatch, tmp_path):
    bind_archived_r22_when_host_unparse_differs(monkeypatch)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, hands, _ = r23.controller_for("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349)
    assert hands == [] and task.r19_source_transform
    assert task.alignment_source_transform == task.r22_alignment_transform
    assert task.alignment_source_transform["transformed_source_sha256"] == r23.probe.INHERITED_SOURCE_SHA256
    assert task.actual_wrist_probe_source_transform == task.r23_actual_wrist_probe_transform
    assert task.actual_wrist_probe_source_transform["transformed_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert task.alignment_settling_telemetry.as_dict()["attempts"] == []
    assert task.actual_wrist_probe_telemetry.as_dict()["attempts"] == []
    assert task.__globals__["_phase_seconds"]["CLOSE"] == 32.


def test_serializers_attach_all_actual_transforms_and_telemetry_without_claiming_prior_audit_coverage(monkeypatch, tmp_path):
    bind_archived_r22_when_host_unparse_differs(monkeypatch)
    prior, declaration = serialization_declaration()
    runner = r23.isolated_runner(PROFILE, declaration)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, _, _, _ = runner.__globals__["controller_for"]("ycb_can", tmp_path, [], SimpleNamespace(), profile, .349)
    writer = runner.__globals__["previous"].atomic_json
    for name in ("result.json", "metadata.json"):
        writer(tmp_path / name, {"kind": "ycb_can", "seed": 3414, "controller_revision": "unit_no_task_run"})
        value = json.loads((tmp_path / name).read_text())
        assert value["runner_revision"] == r23.PROTOCOL_FORMAT
        assert value["controller_class"] == "ProvisionalPreloadBudgetHand"
        assert value["admission_policy_revision"] == r23.r22.r21.PROTOCOL_FORMAT
        assert value["alignment_source_transform"] == task.alignment_source_transform
        assert value["alignment_settling"]["attempts"] == []
        assert value["actual_wrist_probe_source_transform"] == task.actual_wrist_probe_source_transform
        assert value["actual_wrist_probe"]["attempts"] == []
        assert value["inherited_r21_command_audit_does_not_audit_r23"] is True
        assert value["inherited_r22_alignment_audit_does_not_audit_r23_probe"] is True
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["arm_source_transform"] == task.r19_source_transform
    assert metadata["arm_source_transform_is_inherited_r19_prefix_only"] is True
    assert metadata["effective_hand_instances"] == 0
    writer(tmp_path / "protocol.json", copy.deepcopy(prior))
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["format"] == protocol["runner_revision"] == r23.PROTOCOL_FORMAT
    assert protocol[r23.DECLARATION] == declaration
    assert r23.r22.DECLARATION not in protocol
    assert protocol["inherited_r22_settled_alignment_experiment"] == declaration["inherited_r22_declaration"]
    assert protocol["alignment_settling"] == r23.r22.alignment.PARAMETERS
    assert protocol["actual_wrist_probe"] == r23.probe.PARAMETERS
    for key in ("phase_seconds", "constrained_preload", "load_transfer_probe", "scoring"):
        assert protocol[key] == prior[key]
    writer(tmp_path / "summary.json", {"protocol": protocol, "rows": [metadata], "complete": False})
    assert json.loads((tmp_path / "summary.json").read_text())["rows"][0] == metadata


def test_snapshot_inherits_frozen_sources_once_and_detects_changed_preflight_input(tmp_path):
    relative = Path(__file__).relative_to(r23.PROJECT).as_posix()
    declaration = {"inherited_r22_source_hashes": {relative: r23.sha256(r23.PROJECT / relative)},
        "preflight_input_sha256": {relative: r23.sha256(r23.PROJECT / relative)}}
    values = r23.snapshot(PROFILE, declaration)
    assert values[relative] == r23.sha256(r23.PROJECT / relative)
    assert all(values[name] == r23.sha256(r23.PROJECT / name) for name in r23.NEW_SOURCES)
    declaration["preflight_input_sha256"][relative] = "0" * 64
    with pytest.raises(ValueError, match="frozen preflight input changed"):
        r23.snapshot(PROFILE, declaration)


def test_final_cloned_runner_snapshot_callback_executes_through_every_adapter():
    _, declaration = serialization_declaration()
    runner = r23.isolated_runner(PROFILE, declaration)
    values = runner.__globals__["snapshot"](PROFILE)
    assert values == {name: r23.sha256(r23.PROJECT / name) for name in r23.NEW_SOURCES}


def test_existing_output_or_other_case_rejected_before_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(r23, "validate_inputs", lambda *args: pytest.fail("Invalid run reached preflight"))
    with pytest.raises(FileExistsError):
        r23.run(tmp_path, PROFILE, (("ycb_can", 3414),), expected_baseline_summary_sha256=r23.R22_SUMMARY_SHA256)
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r23.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),), expected_baseline_summary_sha256=r23.R22_SUMMARY_SHA256)
