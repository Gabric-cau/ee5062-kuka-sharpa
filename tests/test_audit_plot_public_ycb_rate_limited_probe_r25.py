"""Independent arithmetic and finalized evidence wiring for the R25 post-analysis."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_plot_public_ycb_rate_limited_probe_r25 as audit
import rate_limited_probe_controller_hooks as runtime
from test_public_ycb_probe_speed_plots import saved_run, write_array, write_json


def telemetry_fixture(actuals=None):
    start = np.asarray([.6, .02, .62])
    hand = SimpleNamespace(probe_start_s=.004, probe_start_world_m=start,
        probe_confirmed_seconds=0., probe_active=True)
    data = SimpleNamespace(time=.004, xpos=np.asarray([start]))
    kin = SimpleNamespace(body=0)
    telemetry = runtime.RateLimitedProbeTelemetry()
    tracker = telemetry.begin(data, hand, True, RuntimeError)
    actuals = actuals if actuals is not None else [start, start + [.003, -.004, -.002], start + [-.005, .002, .004]]
    for segment, actual in enumerate(actuals):
        data.time = .004 + segment * .02
        data.xpos[0] = actual
        tracker.target(data, kin, (segment + 1) * .02, RuntimeError)
        for sample in range(1, 11):
            data.time = .004 + segment * .02 + sample * .002
            tracker.sample(data, kin, hand, RuntimeError)
    tracker.finish(data, hand, "FIXTURE_FINAL")
    return telemetry.as_dict()


def test_independent_recomputation_matches_runtime_fixture_and_vector_limiter():
    telemetry = telemetry_fixture()
    original = copy.deepcopy(telemetry)
    result = audit.check_governor(telemetry)
    assert telemetry == original
    assert result["ik_rows_recomputed"] == 3
    assert result["native_samples_bound"] == 30
    assert result["probes"][0]["limiter_active_ik_rows"] == 2
    assert result["probes"][0]["maximum_applied_reference_speed_m_s"] == pytest.approx(.004)
    # This update has multiple components: the bound is on their vector norm.
    delta = np.asarray(telemetry["attempts"][0]["ik_samples"][1]["applied_target_delta_world_m"])
    assert np.count_nonzero(delta) == 3
    assert np.linalg.norm(delta) == pytest.approx(.00008)
    assert max(abs(delta)) < .00008


@pytest.mark.parametrize("change", ["endpoint", "lookahead", "first_actual", "raw", "correction",
    "filtered", "applied", "delta", "alpha", "admission_speed", "actual_delta", "actual_speed",
    "sample_target", "sample_governor", "sample_clock", "sample_age", "same_clock_future_ik", "format"])
def test_corrupted_reference_arithmetic_and_native_sample_binding_are_rejected(change):
    telemetry = telemetry_fixture()
    record = telemetry["attempts"][0]
    first, row = record["ik_samples"][:2]
    sample = record["samples"][10]
    if change == "endpoint":
        row["reference_endpoint_elapsed_s"] += .002
    elif change == "lookahead":
        row["timestamp_s"] += .002
    elif change == "first_actual":
        first["actual_wrist_at_ik_sample_world_m"][0] += .001
    elif change == "raw":
        row["raw_cartesian_ik_target_world_m"][0] += .001
    elif change == "correction":
        row["bounded_reference_correction_world_m"][0] += .001
    elif change == "filtered":
        row["filtered_cartesian_ik_target_world_m"][0] += .001
    elif change == "applied":
        row["applied_cartesian_ik_target_world_m"][0] += .001
    elif change == "delta":
        row["applied_target_delta_world_m"][0] += .00008
    elif change == "alpha":
        row["filter_alpha"] += .01
    elif change == "admission_speed":
        first["actual_wrist_speed_since_previous_ik_m_s"] = 0.
    elif change == "actual_delta":
        row["actual_wrist_delta_since_previous_ik_world_m"][0] += .001
    elif change == "actual_speed":
        row["actual_wrist_speed_since_previous_ik_m_s"] += .01
    elif change == "sample_target":
        sample["cartesian_ik_target_world_m"][0] += .001
    elif change == "sample_governor":
        sample["reference_governor"]["raw_cartesian_ik_target_world_m"][0] += .001
    elif change == "sample_clock":
        sample["latest_ik_timestamp_s"] -= .02
    elif change == "sample_age":
        sample["ik_sample_age_s"] += .002
    elif change == "same_clock_future_ik":
        # A native sample at the next IK clock still precedes that IK update.
        sample = record["samples"][9]
        sample["latest_ik_timestamp_s"] = row["timestamp_s"]
        sample["ik_sample_age_s"] = 0.
        sample["reference_governor"] = {field: row[field] for field in audit.SAMPLE_GOVERNOR_FIELDS}
    else:
        telemetry["format"] = "actual_wrist_probe_policy_v1"
    with pytest.raises(ValueError):
        audit.check_governor(telemetry)


def test_first_native_sample_at_segment_end_keeps_previous_reference():
    telemetry = telemetry_fixture()
    rows = telemetry["attempts"][0]
    assert rows["samples"][9]["timestamp_s"] == pytest.approx(rows["ik_samples"][1]["timestamp_s"])
    assert rows["samples"][9]["latest_ik_timestamp_s"] == rows["ik_samples"][0]["timestamp_s"]
    assert rows["samples"][10]["latest_ik_timestamp_s"] == rows["ik_samples"][1]["timestamp_s"]
    audit.check_governor(telemetry)


def test_format_adapter_changes_only_copies_and_preserves_exact_sample_objects():
    telemetry = telemetry_fixture()
    run = {"row": {"actual_wrist_probe": telemetry, "success": False}, "inputs": {}}
    original = copy.deepcopy(run)
    adapted = audit.analysis_adapter(run)
    assert run == original
    assert adapted is not run and adapted["row"] is not run["row"]
    assert adapted["row"]["actual_wrist_probe"] is not telemetry
    assert adapted["row"]["actual_wrist_probe"]["attempts"] is telemetry["attempts"]
    assert adapted["row"]["actual_wrist_probe"]["format"] == "actual_wrist_probe_policy_v1"
    assert telemetry["format"] == audit.TELEMETRY_FORMAT


def upgrade_saved_run(fixture):
    """Use real frozen hook bytes and recording schema in the tiny legacy run."""
    directory, case, job = (fixture[key] for key in ("directory", "case", "job"))
    row, protocol = fixture["row"], fixture["protocol"]
    start = np.asarray(fixture["trace"][1]["wrist_position"])
    telemetry = telemetry_fixture([start])
    record = telemetry["attempts"][0]
    record["samples"] = record["samples"][:2]
    record["outcome"]["timestamp_s"] = .008
    record["outcome"]["elapsed_s"] = .004
    for sample, trace in zip(record["samples"], fixture["trace"][2:4]):
        sample["current_actual_wrist_world_m"] = list(trace["wrist_position"])
        trace["actual_wrist_probe"] = copy.deepcopy(sample)
    row.update(actual_wrist_probe=telemetry, controller_class=audit.CLASS,
        effective_hand_instances=1, effective_hand_controller_types=[audit.CLASS], effective_hand_budgets_rad=[.1],
        inherited_r23_probe_audit_does_not_audit_r25_governor=True)
    row["rate_limited_probe_source_transform"] = {
        "format": "r25_rate_limited_probe_global_binding_v1",
        "inherited_source_sha256": audit.ARM_SHA256, "readable_source_sha256": audit.ARM_SHA256,
        "readable_source_unchanged": True, "compiled_arm_code_unchanged": True,
        "replacement_global": "_r23_probe_telemetry", "inherited_telemetry_class": "ProbeTelemetry",
        "replacement_telemetry_class": "RateLimitedProbeTelemetry", "replacement_tracker_class": "RateLimitedProbeTracker",
        "inherited_probe_hook_sha256": audit.INHERITED_HOOK_SHA256,
        "raw_reference_source_sha256": audit.RAW_SHA256, "raw_reference_parameters": audit.RAW_PARAMETERS,
        "reference_governor": audit.GOVERNOR, "original_decisive_confirmation_called": True,
        "all_original_native_safety_checks_retained": True, "runtime_target_state_used": False,
        "target_qpos_writes_added": False}
    write_array(case / "trace.json.gz", fixture["trace"])
    row["artifacts"]["trace"].update(audit.speed.evidence.capture(case / "trace.json.gz"), path="trace.json.gz")
    # The schema uses the same source identity as R25, with a disclosed fixture stub.
    (case / "derived_controller.py").write_text("# fixture arm, not the actual R25 derived source\n")
    row["derived_controller_sha256"] = audit.speed.evidence.digest(case / "derived_controller.py")
    sources = {name: (audit.PROJECT / name).read_bytes() for name in audit.FROZEN_PINS}
    sources["scripts/run_supervised_experiment.py"] = b"# fixture supervisor\n"
    for index in range(207 - len(sources)):
        sources[f"fixtures/source_{index}.py"] = b"# fixture\n"
    with zipfile.ZipFile(directory / "frozen_source.zip", "w") as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    protocol.update(format=audit.REVISION, source_hashes_before=manifest, source_hashes_after=manifest.copy(),
        probe_reference_governor=copy.deepcopy(audit.GOVERNOR),
        r25_rate_limited_probe_experiment={"governor_parameters": copy.deepcopy(audit.GOVERNOR)})
    summary = dict(fixture["summary"], runner_revision=audit.REVISION)
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "summary.json", summary)
    write_json(case / "metadata.json", row)
    write_json(case / "result.json", {key: row[key] for key in ("success", "failure", "attempts",
        "actual_wrist_probe", "rate_limited_probe_source_transform", "inherited_r23_probe_audit_does_not_audit_r25_governor")})
    sha = audit.speed.evidence.digest(directory / "summary.json")
    for name in ("verification_receipt.json", "strict_native_verification.json"):
        receipt = json.loads((directory / name).read_text())
        receipt["summary_sha256"] = sha
        if name == "verification_receipt.json":
            receipt["frozen_source_sha256"] = audit.speed.evidence.digest(directory / "frozen_source.zip")
        else:
            receipt["rows"][0]["result_sha256"] = audit.speed.evidence.digest(case / "result.json")
        write_json(directory / name, receipt)
    write_json(job / "request.json", {"command": ["python", audit.ENTRY, "--output", str(directory)],
        "supervisor_sha256": manifest["scripts/run_supervised_experiment.py"]})
    return sha


def test_real_schema_source_contract_and_full_native_plot_path(saved_run, tmp_path, monkeypatch):
    sha = upgrade_saved_run(saved_run)
    run = audit.speed.read_run(saved_run["directory"], sha, saved_run["job"], audit.REVISION, audit.CLASS, audit.ENTRY)
    # A tiny recorded fixture cannot contain the actual f1a source bytes; every other
    # contract remains real. This is explicitly a fixture-only identity substitution.
    monkeypatch.setattr(audit, "ARM_SHA256", run["row"]["derived_controller_sha256"])
    transform = run["row"]["rate_limited_probe_source_transform"]
    transform["inherited_source_sha256"] = audit.ARM_SHA256
    transform["readable_source_sha256"] = audit.ARM_SHA256
    result_path = Path(run["inputs"]["result"]["path"])
    result = json.loads(result_path.read_text())
    result["rate_limited_probe_source_transform"] = transform
    write_json(result_path, result)
    assert audit.source_contract(run)["source_count"] == 207
    run["row"]["rate_limited_probe_source_transform"]["replacement_tracker_class"] = "WrongTracker"
    with pytest.raises(ValueError, match="replacement binding"):
        audit.source_contract(run)
    # Rebuild bound receipts after that deliberate corruption, then use the same
    # fixture identity substitution throughout the complete CLI path.
    sha = upgrade_saved_run(saved_run)
    output = tmp_path / "r25_diagnostics"
    assert audit.main([str(saved_run["directory"]), "--expected-summary-sha256", sha,
        "--job", str(saved_run["job"]), "--output", str(output)]) == 0
    receipt = json.loads((output / "diagnostics.json").read_text())
    assert receipt["accepted_reference_governor_evidence"] is True
    assert receipt["run"]["case"]["success"] is False
    assert receipt["run"]["case"]["held_lift"] is False
    assert receipt["reference_audit"]["native_samples_bound"] == 2
    assert receipt["run"]["native_samples_streamed"] == 5
    assert receipt["part_of_runtime_frozen_manifest"] is False
    assert len(receipt["figures"]) == 4
    with gzip.open(output / "recorded_probes_and_reference_audit.json.gz", "rt") as stream:
        assert len(json.load(stream)["reference_audit"]["probes"][0]["reference_difference_rates"]) == 1
    for path in output.glob("*.png"):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert pixels.shape[0] > 1000
        assert pixels.std() > 10.
    with pytest.raises(FileExistsError):
        audit.main([str(saved_run["directory"]), "--expected-summary-sha256", sha,
            "--job", str(saved_run["job"]), "--output", str(output)])


@pytest.mark.parametrize("change", ["frozen_hook_pin", "source_count", "declared_governor", "hand_budget"])
def test_source_contract_rejects_wrong_frozen_or_declared_binding(saved_run, change):
    sha = upgrade_saved_run(saved_run)
    run = audit.speed.read_run(saved_run["directory"], sha, saved_run["job"], audit.REVISION, audit.CLASS, audit.ENTRY)
    path = Path(run["inputs"]["protocol"]["path"])
    protocol = json.loads(path.read_text())
    if change == "frozen_hook_pin":
        protocol["source_hashes_before"]["scripts/rate_limited_probe_controller_hooks.py"] = "0" * 64
    elif change == "source_count":
        protocol["source_hashes_before"].pop("fixtures/source_0.py")
    elif change == "declared_governor":
        protocol["probe_reference_governor"]["maximum_target_speed_m_s"] = .008
    else:
        run["row"]["effective_hand_budgets_rad"] = [.15, .15]
    write_json(path, protocol)
    with pytest.raises(ValueError, match="Frozen R25"):
        audit.source_contract(run)


@pytest.mark.parametrize("count,attempts,accepted", [(1, 1, True), (2, 2, True),
    (0, 0, False), (3, 3, False), (1, 2, False), (True, 1, False)])
def test_actual_hand_instances_match_one_or_two_attempts(saved_run, monkeypatch, count, attempts, accepted):
    sha = upgrade_saved_run(saved_run)
    run = audit.speed.read_run(saved_run["directory"], sha, saved_run["job"], audit.REVISION, audit.CLASS, audit.ENTRY)
    monkeypatch.setattr(audit, "ARM_SHA256", run["row"]["derived_controller_sha256"])
    row = run["row"]
    row.update(effective_hand_instances=count, effective_hand_controller_types=[audit.CLASS] * count,
        effective_hand_budgets_rad=[.1] * count, attempts=[copy.deepcopy(row["attempts"][0])] * attempts,
        success=bool(accepted and count == 1))
    path = Path(run["inputs"]["result"]["path"])
    result = json.loads(path.read_text())
    for value in (row, result):
        value["rate_limited_probe_source_transform"].update(inherited_source_sha256=audit.ARM_SHA256,
            readable_source_sha256=audit.ARM_SHA256)
    write_json(path, result)
    if accepted:
        assert audit.source_contract(run)["actual_hand_instances"] == count
    else:
        with pytest.raises(ValueError, match="Frozen R25"):
            audit.source_contract(run)
