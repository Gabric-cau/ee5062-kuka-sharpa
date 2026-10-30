"""Exact recorded-cache speed, complete evidence gates, and failed-probe plots."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_public_ycb_probe_speed as plot


REVISION = "explicit_fixture_probe_version"
CLASS = "FixtureHand"
ENTRY = "scripts/fixture_runner.py"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_array(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream)


@pytest.fixture
def saved_run(tmp_path):
    directory, job = tmp_path / "run", tmp_path / "job"
    case = directory / "case"
    case.mkdir(parents=True)
    job.mkdir()
    start = np.asarray([.6, .02, .62])
    positions = [start, start, start + [0., .000010, .000002], start + [0., -.000012, .000005], start]
    samples, trace, native = [], [], []
    phases = ["SEARCH", "CLOSE", "TRIAL_LIFT", "TRIAL_LIFT", "RELEASE"]
    for k, (position, phase) in enumerate(zip(positions, phases), 1):
        timestamp = k * .002
        policy = None
        sample = None
        if phase == "TRIAL_LIFT":
            speed = float(np.linalg.norm(position - positions[k - 2]) / .002)
            policy = {"actual_wrist_position_world_m": position.tolist(), "actual_wrist_speed_m_s": speed,
                "stop_reason": "PROBE_SPEED" if k == 4 else None, "motion_safety_reasons": []}
            sample = {"timestamp_s": timestamp, "elapsed_s": timestamp - .004,
                "current_actual_wrist_world_m": position.tolist(), "currently_probe_active": True,
                "inherited_confirmed_seconds": 0.}
            samples.append(sample)
        record = {"timestamp": timestamp, "state": phase, "wrist_position": position.tolist(),
            "arm_motor_control": [0.] * 7, "hand_command": [0.] * 22,
            "probe_policy": policy, "actual_wrist_probe": sample}
        if k != 4:
            record["contact_guard"] = {"observation_timestamp_s": timestamp - .002,
                "mode": "pressure_crawl", "scale": .4, "retreat_reason": None}
        trace.append(record)
        native.append({"timestamp": timestamp, "state": phase, "motor_commands": [0.] * 29,
            "native_contacts": [{"geom_ids": [0, 2 if k == 4 else 1]}]})
    ik = [{"timestamp_s": .004, "actual_wrist_at_ik_sample_world_m": start.tolist(),
        "desired_actual_path_world_m": (start + [0., 0., .000001]).tolist(),
        "cartesian_ik_target_world_m": (start + [0., -.00001, .000003]).tolist(),
        "bounded_reference_correction_world_m": [0., -.00001, .000002]},
        {"timestamp_s": .006, "actual_wrist_at_ik_sample_world_m": positions[2].tolist(),
        "desired_actual_path_world_m": (start + [0., 0., .000002]).tolist(),
        "cartesian_ik_target_world_m": (start + [0., -.00002, .000006]).tolist(),
        "bounded_reference_correction_world_m": [0., -.00002, .000004]}]
    outcome = {"timestamp_s": .008, "final_reason": "PROBE_SPEED", "arm_exit_reason": "MOTION_EXCEPTION",
        "motion_failure": "PROBE_SPEED", "probe_active_at_motion_end": False,
        "original_confirmation_attempted": False, "original_confirmation_accepted": None}
    telemetry = {"format": "actual_wrist_probe_policy_v1", "attempts": [{"start_s": .004,
        "actual_admission_start_world_m": start.tolist(), "samples": samples, "ik_samples": ik, "outcome": outcome}]}
    guards = [{"observation_timestamp_s": .004, "mode": "pressure_crawl", "scale": .42},
        {"observation_timestamp_s": .006, "mode": "pressure_crawl", "scale": .405}]
    for name, rows in (("native", native), ("trace", trace), ("guard", guards)):
        write_array(case / (name + ".json.gz"), rows)
    (case / "scene.xml").write_text('<mujoco><option timestep="0.002"/></mujoco>', encoding="utf-8")
    (case / "derived_controller.py").write_text("# fixture executed source\n", encoding="utf-8")
    sources = {ENTRY: b"# fixture runner\n", "scripts/run_supervised_experiment.py": b"# fixture supervisor\n"}
    with zipfile.ZipFile(directory / "frozen_source.zip", "w") as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    protocol = {"format": REVISION, "source_hashes_before": manifest, "source_hashes_after": manifest.copy(),
        "source_changed_during_run": False, "public_meshes_unchanged": True,
        "schedule": [{"kind": "fixture_can", "seed": 1}],
        "load_transfer_probe": {"maximum_actual_wrist_speed_m_s": .01}}
    row = {"kind": "fixture_can", "seed": 1, "path": "case", "runner_error": None,
        "controller_class": CLASS, "success": False, "held_lift": False, "failure": "PROBE_SPEED",
        "duration_s": .01, "attempts": [{"attempt": 1, "failure": "PROBE_SPEED"}],
        "actual_wrist_probe": telemetry,
        "native_audit": {"native_samples": 5, "integration_steps": 5, "external_target_qpos_writes": 0},
        "scene_unchanged_since_load": True,
        "scene_sha256_before_load": plot.evidence.digest(case / "scene.xml"),
        "scene_sha256_at_finalize": plot.evidence.digest(case / "scene.xml"),
        "derived_controller_sha256": plot.evidence.digest(case / "derived_controller.py"),
        "artifacts": {name: {**plot.evidence.capture(case / (name + ".json.gz")), "path": name + ".json.gz"}
            for name in ("native", "trace", "guard")}}
    summary = {"complete": True, "runner_errors": 0, "runner_revision": REVISION, "protocol": protocol, "rows": [row]}
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "summary.json", summary)
    write_json(case / "metadata.json", row)
    result = {key: row[key] for key in ("success", "failure", "attempts", "actual_wrist_probe")}
    write_json(case / "result.json", result)
    summary_sha = plot.evidence.digest(directory / "summary.json")
    native_receipt = {"summary_sha256": summary_sha, "format": "public_ycb_development_native_verification_r1",
        "complete": True, "unknowns": 0, "source_hashes_before_equal_after": True,
        "native_samples_contiguous_2ms": True, "audited_episodes": 1, "native_samples": 5,
        "frozen_source_sha256": plot.evidence.digest(directory / "frozen_source.zip"),
        "rows": [{"path": "case", "unknown": False, "outcome": False, "held_lift": False, "native_samples": 5}]}
    strict_receipt = {"summary_sha256": summary_sha, "format": "public_ycb_strict_native_audit_20261021",
        "accepted_evidence": True, "unknowns": 0, "denominator": 1,
        "rows": [{"path": "case", "unknown": False, "native_samples": 5,
            "checks": {"legacy_task_success": False, "held_lift": False}, "strict_success": False,
            "result_sha256": plot.evidence.digest(case / "result.json"),
            "scene_sha256": plot.evidence.digest(case / "scene.xml")}]}
    write_json(directory / "verification_receipt.json", native_receipt)
    write_json(directory / "strict_native_verification.json", strict_receipt)
    write_json(job / "status.json", {"state": "FINISHED", "terminal": True, "child_exit_observed": True, "returncode": 0})
    write_json(job / "request.json", {"command": ["python", ENTRY, "--output", str(directory)],
        "supervisor_sha256": manifest["scripts/run_supervised_experiment.py"]})
    return {"directory": directory, "job": job, "case": case, "summary_sha": summary_sha,
        "summary": summary, "protocol": protocol, "row": row, "native": native, "trace": trace}


def read(fixture):
    return plot.read_run(fixture["directory"], fixture["summary_sha"], fixture["job"], REVISION, CLASS, ENTRY)


def test_generic_explicit_version_and_failed_native_strict_evidence_are_preserved(saved_run):
    run = read(saved_run)
    assert run["source_count"] == 2
    assert run["row"]["success"] is False
    assert run["native_and_strict"]["strict_task_success"] is False
    assert set(run["inputs"]) >= {"artifact_native", "artifact_trace", "artifact_guard", "source_archive"}


@pytest.mark.parametrize("change", ["running", "unobserved", "nonzero", "boolean_rc", "wrong_runner"])
def test_job_must_be_the_bound_observed_successful_process_exit(saved_run, change):
    status_path = saved_run["job"] / "status.json"
    status = json.loads(status_path.read_text())
    if change == "running":
        status["state"] = "RUNNING"
    elif change == "unobserved":
        status["child_exit_observed"] = False
    elif change == "nonzero":
        status["returncode"] = 1
    elif change == "boolean_rc":
        status["returncode"] = False
    else:
        request = json.loads((saved_run["job"] / "request.json").read_text())
        request["command"][1] = "scripts/other.py"
        write_json(saved_run["job"] / "request.json", request)
    write_json(status_path, status)
    with pytest.raises(ValueError, match="FINISHED rc0"):
        read(saved_run)


@pytest.mark.parametrize("change", ["revision", "class", "running", "changed_source", "target_write", "duration"])
def test_explicit_source_class_and_native_final_summary_contract(saved_run, change):
    summary, protocol = copy.deepcopy((saved_run["summary"], saved_run["protocol"]))
    if change == "revision":
        summary["runner_revision"] = "other_version"
    elif change == "class":
        summary["rows"][0]["controller_class"] = "OtherHand"
    elif change == "running":
        summary["complete"] = False
    elif change == "changed_source":
        protocol["source_changed_during_run"] = True
        summary["protocol"] = protocol
    elif change == "target_write":
        summary["rows"][0]["native_audit"]["external_target_qpos_writes"] = 1
    else:
        summary["rows"][0]["duration_s"] += .002
    with pytest.raises(ValueError):
        plot.check_summary(summary, protocol, REVISION, CLASS)


@pytest.mark.parametrize("filename", ["summary.json", "case/metadata.json", "case/result.json",
    "case/scene.xml", "case/derived_controller.py", "case/guard.json.gz", "case/native.json.gz",
    "case/trace.json.gz", "frozen_source.zip", "verification_receipt.json", "strict_native_verification.json"])
def test_modified_final_bound_evidence_is_rejected(saved_run, filename):
    path = saved_run["directory"] / filename
    if filename == "frozen_source.zip":
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("unrecorded.py", b"extra")
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        if filename == "summary.json":
            value["complete"] = False
        elif filename == "case/metadata.json":
            value["success"] = True
        elif filename == "case/result.json":
            value["held_lift"] = True
        else:
            value["summary_sha256"] = "b" * 64
        write_json(path, value)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        read(saved_run)


def test_source_zip_cannot_change_bytes_with_the_same_inventory(saved_run):
    path = saved_run["directory"] / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries[ENTRY] += b"# modified\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="ZIP bytes"):
        read(saved_run)


def test_adjacent_trace_speed_uses_admission_and_keeps_missing_final_guard(saved_run):
    run = plot.extract(read(saved_run))
    probe = run["probes"][0]
    first, final = probe["series"]
    assert first["velocity_world_m_s"] == pytest.approx([0., .005, .001])
    assert final["velocity_world_m_s"] == pytest.approx([0., -.011, .0015])
    assert final["speed_m_s"] == pytest.approx(np.linalg.norm([0., -.011, .0015]))
    assert final["policy_speed_m_s"] == pytest.approx(final["speed_m_s"])
    assert probe["statistics"]["speed_exceeds_limit_samples"] == 1
    assert probe["statistics"]["final_trace_contact_guard_present"] is False
    assert final["trace_contact_guard"] is None
    assert [r["kind"] for r in probe["events"]] == ["native_geometry_pair_change", "recorded_probe_guard", "recorded_final_outcome"]
    assert probe["ik_target_difference_rates"][0]["sample_interval_s"] == pytest.approx(.002)
    assert probe["ik_target_difference_rates"][0]["target_difference_velocity_world_m_s"] == pytest.approx([0., -.005, .0015])


@pytest.mark.parametrize("change", ["native_gap", "motor", "missing_trace", "admission", "sample_wrist",
    "sample_gap", "policy_speed", "policy_wrist", "ik_wrist"])
def test_no_interpolation_or_unmatched_cache_and_clock_is_accepted(saved_run, monkeypatch, change):
    run = read(saved_run)
    native, trace = copy.deepcopy((saved_run["native"], saved_run["trace"]))
    record = run["row"]["actual_wrist_probe"]["attempts"][0]
    if change == "native_gap":
        native.pop(2)
    elif change == "motor":
        native[2]["motor_commands"][0] = .1
    elif change == "missing_trace":
        trace.pop(2)
    elif change == "admission":
        trace[1]["wrist_position"][0] += .001
    elif change == "sample_wrist":
        record["samples"][0]["current_actual_wrist_world_m"][0] += .001
    elif change == "sample_gap":
        record["samples"].pop(0)
    elif change == "policy_speed":
        trace[2]["probe_policy"]["actual_wrist_speed_m_s"] += .001
    elif change == "policy_wrist":
        trace[2]["probe_policy"]["actual_wrist_position_world_m"][0] += .001
    else:
        record["ik_samples"][0]["actual_wrist_at_ik_sample_world_m"][0] += .001
    original = plot.evidence.json_array
    monkeypatch.setattr(plot.evidence, "json_array", lambda path:
        iter(native) if path.name == "native.json.gz" else iter(trace) if path.name == "trace.json.gz" else original(path))
    with pytest.raises(ValueError):
        plot.extract(run)


def test_first_same_clock_trace_is_used_and_later_phase_end_duplicate_is_counted(saved_run, monkeypatch):
    run = read(saved_run)
    trace = copy.deepcopy(saved_run["trace"])
    later = copy.deepcopy(trace[3])
    later["state"] = "RELEASE"
    later["wrist_position"][0] += 10.
    trace.insert(4, later)
    original = plot.evidence.json_array
    monkeypatch.setattr(plot.evidence, "json_array", lambda path: iter(trace) if path.name == "trace.json.gz" else original(path))
    assert plot.extract(run)["duplicate_trace_clocks"] == 1


def test_contact_guard_retreat_keeps_its_observation_clock(saved_run, monkeypatch):
    run = read(saved_run)
    guard = {"observation_timestamp_s": .006, "mode": "retreat", "scale": 0., "retreat_reason": "FIXTURE_PRESSURE"}
    original = plot.evidence.json_array
    monkeypatch.setattr(plot.evidence, "json_array", lambda path: iter([guard]) if path.name == "guard.json.gz" else original(path))
    probe = plot.extract(run)["probes"][0]
    event = next(event for event in probe["events"] if event["kind"] == "recorded_contact_guard_retreat")
    assert event["timestamp_s"] == .006
    assert event["clock"] == "original_observation_timestamp_s"
    assert event["reason"] == "FIXTURE_PRESSURE"
    assert probe["series"][-1]["trace_contact_guard"] is None


def test_six_panel_failed_probe_is_nonblank_and_output_binds_all_inputs(saved_run, tmp_path):
    output = tmp_path / "diagnostics"
    assert plot.main([str(saved_run["directory"]), "--expected-summary-sha256", saved_run["summary_sha"],
        "--job", str(saved_run["job"]), "--expected-runner-revision", REVISION,
        "--expected-controller-class", CLASS, "--runner-entry", ENTRY, "--output", str(output)]) == 0
    receipt = json.loads((output / "diagnostics.json").read_text())
    assert receipt["run"]["case"]["success"] is False
    assert receipt["run"]["case"]["held_lift"] is False
    assert receipt["run"]["native_samples_streamed"] == 5
    assert receipt["part_of_runtime_frozen_manifest"] is False
    assert len(receipt["figures"]) == 2
    assert "LP" in receipt["scope"] and "not independently audit" in receipt["scope"]
    png = np.asarray(Image.open(output / "probe1_speed.png").convert("RGB"))
    assert png.shape == (2550, 1800, 3)
    assert float(png.std()) > 10.
    assert np.count_nonzero(np.any(png < 235, axis=2)) > 10000
    assert all(record["bytes"] > 10000 for record in receipt["figures"])
    with pytest.raises(FileExistsError):
        plot.main([str(saved_run["directory"]), "--expected-summary-sha256", saved_run["summary_sha"],
            "--job", str(saved_run["job"]), "--expected-runner-revision", REVISION,
            "--expected-controller-class", CLASS, "--runner-entry", ENTRY, "--output", str(output)])


def test_absent_probe_remains_absent():
    assert plot.prepare_probes({"format": "actual_wrist_probe_policy_v1", "attempts": []}) == ([], {}, {}, {})
