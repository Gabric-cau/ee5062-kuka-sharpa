from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_public_ycb_close_deadline.py"
SPEC = importlib.util.spec_from_file_location("close_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def write_stream(path, rows):
    with gzip.open(path, "wt") as stream:
        json.dump(rows, stream)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": comparison.digest(path)}


def fixture_run(root, *, baseline=None, mutate=None):
    root.mkdir()
    case = root / "case"
    case.mkdir()
    close = .016 if baseline is None else .032
    protocol = {"format": "public_ycb_bias_budget_development_r17" if baseline is None else "public_ycb_close_deadline_development_r18",
        "phase_seconds": {"CLOSE": close, "RECOVER": .004}, "joint_admittance": {"additional_bias_limit_rad": .04},
        "source_changed_during_run": False, "source_hashes_before": {"source": "a"}, "source_hashes_after": {"source": "a"}}
    if baseline is not None:
        protocol["r18_close_deadline_experiment"] = {"r17_comparison_summary_sha256": comparison.digest(baseline / "summary.json")}
    scene = case / "scene.xml"
    scene.write_text('<mujoco><option timestep="0.002"/></mujoco>')
    records = {key: [] for key in comparison.STREAMS}
    for index in range(1, 23):
        time = index * .002
        phase = "SEARCH" if index < 3 else "CLOSE" if time <= .004 + close + 1e-9 else "RECOVER"
        native = {"timestamp": time, "state": phase, "motor_commands": [.1, .2], "target_qpos": [0., 0., .3, 1., 0., 0., 0.],
            "target_spatial_velocity_world": [0.] * 6,
            "actual_hand_joints": {"left_thumb_IP": {"position_rad": .2, "velocity_rad_s": .001}},
            "native_contacts": [{"geom_ids": [1, 2], "force_torque_contact_frame": [4., 0., 0., 0., 0., 0.]}],
            "native_actuation": {"actuator_force": [.03], "qfrc_actuator": [.03]}}
        records["native"].append(native)
        records["grip_feedback"].append({"timestamp_s": time, "pressures_n": {"thumb": 4.}, "carrying": False})
        probe = {"timestamp_s": time, "phase": phase, "preprobe_reasons": [], "probe_active": False,
            "probe_eligible_seconds": max(0., time - .006), "probe_confirmed_seconds": 0.}
        records["load_transfer_probe"].append(probe)
        records["bias_recovery"].append({"timestamp_s": time, "phase": phase, "recovery_active": False, "dt_s": .002,
            "per_finger": [{"finger": "thumb", "mode": "NOMINAL", "nominal_qp": {"accepted": True},
                "joints": {"left_thumb_IP": {"applied_command_rad": .23, "before_actual_rad": .2,
                    "previous_reference_rad": .229998, "fixed_capture_bias_rad": .02,
                    "bias_interval_rad": [.18, .26], "rate_interval_rad": [.228, .231]}}}]})
        records["joint_admittance"].append({"timestamp_s": time, "phase": phase, "event": "NOMINAL_REFERENCE_WRITE", "command_rad": {"left_thumb_IP": .23}})
        if phase == "CLOSE":
            preload = {"timestamp_s": time, "phase": phase, "admission_recheck": False, "ready": False,
                "continuous_seconds": 0., "window_reset_reason": None,
                "reasons": ["GROUP_PRESSURE_ERROR"], "current_groups": {"thumb:left_thumb_DP:0": {"target_n": 5., "measured_n": 4.}},
                "original_probe_snapshot": {"sampled_fixed_normal_1g_feasible": False, "preprobe_reasons": [],
                    "actual_wrist_speed_m_s": .0002, "probe_eligible_seconds": max(0., time - .006)}}
            records["preload"].append(preload)
            if baseline is None and index == 10:
                recheck = copy.deepcopy(preload)
                recheck["admission_recheck"] = True
                records["preload"].append(recheck)
                records["load_transfer_probe"].append(dict(probe, admission=False, preload_admission=False))
    # The write at t=.020 belongs to the next integration interval; it may differ.
    if baseline is None:
        for key in ("grip_feedback", "bias_recovery", "joint_admittance"):
            records[key][9]["deadline_branch"] = "baseline_release"
    if mutate is not None:
        mutate(records)
    row = {"path": "case", "kind": "ycb_can", "seed": 3414, "runner_error": None,
        "scene_sha256_before_load": comparison.digest(scene), "scene_unchanged_since_load": True,
        "artifacts": {key: write_stream(case / (key + ".json.gz"), values) for key, values in records.items()}}
    write_json(case / "metadata.json", row)
    write_json(root / "protocol.json", protocol)
    write_json(root / "summary.json", {"complete": True, "protocol": protocol, "rows": [row]})
    if baseline is None:
        comparison.EXPECTED_R17_SUMMARY_SHA256 = comparison.digest(root / "summary.json")
    return root


def pair(tmp_path, mutate=None):
    baseline = fixture_run(tmp_path / "baseline")
    candidate = fixture_run(tmp_path / "candidate", baseline=baseline, mutate=mutate)
    return baseline, candidate


def test_derived_inclusive_native_prefix_and_declared_recheck_exclusion(tmp_path):
    baseline, candidate = pair(tmp_path)
    result = comparison.compare_runs(baseline, candidate)
    assert result["passed"]
    assert result["prefix_inclusive_deadline_s"] == pytest.approx(.020)
    assert result["derived_phase_start_s"] == pytest.approx(.004)
    assert result["streams"]["native"]["compared_rows"] == 10
    assert result["streams"]["bias_recovery"]["compared_rows"] == 9
    assert result["streams"]["bias_recovery"]["excluded_decisive_rechecks"][0] == {"write_at_old_deadline_next_transition": 1}
    assert result["streams"]["preload"]["excluded_decisive_rechecks"][0] == {"original_deadline_decisive_recheck": 1}
    tails = result["streams"]["preload"]["candidate_tail_windows"][0]["windows"]
    assert [row["requested_seconds"] for row in tails] == list(comparison.WINDOWS)
    assert tails[0]["groups"]["thumb:left_thumb_DP:0"]["signed_error_n"]["mean"] == 1.


@pytest.mark.parametrize("field", ["motor_commands", "target_qpos", "actual_hand_joints", "native_contacts", "native_actuation"])
def test_physical_prefix_difference_fails_with_first_time_and_maximum(tmp_path, field):
    def mutate(records):
        row = records["native"][9]
        if field == "motor_commands":
            row[field][0] += .001
        elif field == "target_qpos":
            row[field][2] += .001
        elif field == "actual_hand_joints":
            row[field]["left_thumb_IP"]["velocity_rad_s"] += .001
        elif field == "native_contacts":
            row[field][0]["force_torque_contact_frame"][0] += .001
        else:
            row[field]["qfrc_actuator"][0] += .001
    result = comparison.compare_runs(*pair(tmp_path, mutate))
    stream = result["streams"]["native"]
    assert not result["passed"] and stream["first_difference"]["time_s"] == pytest.approx(.020)
    assert field in stream["first_difference"]["field"]
    assert max(row["maximum_absolute_difference"] for row in stream["numeric_fields"].values()) == pytest.approx(.001)


def test_pressure_difference_and_missing_record_fail(tmp_path):
    def mutate(records):
        records["grip_feedback"][6]["pressures_n"]["thumb"] += .01
        records["load_transfer_probe"].pop(5)
    result = comparison.compare_runs(*pair(tmp_path, mutate))
    assert not result["passed"]
    assert result["streams"]["grip_feedback"]["first_difference"]["time_s"] == pytest.approx(.014)
    assert not result["streams"]["load_transfer_probe"]["passed"]


def test_extra_decisive_record_before_deadline_is_not_silently_excluded(tmp_path):
    def mutate(records):
        records["preload"][0]["admission_recheck"] = True
    result = comparison.compare_runs(*pair(tmp_path, mutate))
    assert not result["streams"]["preload"]["passed"]


def test_stale_artifact_hash_and_wrong_baseline_binding_rejected(tmp_path):
    baseline, candidate = pair(tmp_path)
    with (candidate / "case/native.json.gz").open("ab") as stream:
        stream.write(b"different")
    with pytest.raises(ValueError, match="artifact hash/size"):
        comparison.compare_runs(baseline, candidate)


def test_wrong_baseline_summary_binding_rejected(tmp_path):
    baseline, candidate = pair(tmp_path)
    protocol = json.loads((candidate / "protocol.json").read_text())
    protocol["r18_close_deadline_experiment"]["r17_comparison_summary_sha256"] = "wrong"
    summary = json.loads((candidate / "summary.json").read_text())
    summary["protocol"] = protocol
    write_json(candidate / "protocol.json", protocol)
    write_json(candidate / "summary.json", summary)
    with pytest.raises(ValueError, match="different R17"):
        comparison.compare_runs(baseline, candidate)


def test_exact_frozen_baseline_guard_rejects_other_summary(tmp_path, monkeypatch):
    baseline, candidate = pair(tmp_path)
    monkeypatch.setattr(comparison, "EXPECTED_R17_SUMMARY_SHA256", "other")
    with pytest.raises(ValueError, match="exact frozen R17"):
        comparison.compare_runs(baseline, candidate)


def test_empty_required_stream_is_rejected(tmp_path):
    baseline, candidate = pair(tmp_path, lambda rows: rows["grip_feedback"].clear())
    with pytest.raises(ValueError, match="prefix is empty"):
        comparison.compare_runs(baseline, candidate)


def test_inputs_changed_after_capture_are_rejected(tmp_path, monkeypatch):
    baseline, candidate = pair(tmp_path)
    original = comparison.json_array
    changed = False
    def mutate_after_capture(path):
        nonlocal changed
        for row in original(path):
            if not changed:
                changed = True
                with (candidate / "summary.json").open("a") as stream:
                    stream.write(" ")
            yield row
    monkeypatch.setattr(comparison, "json_array", mutate_after_capture)
    with pytest.raises(ValueError, match="input changed"):
        comparison.compare_runs(baseline, candidate)


def test_symlink_stream_is_rejected(tmp_path):
    baseline, candidate = pair(tmp_path)
    path = candidate / "case/preload.json.gz"
    target = candidate / "case/renamed.gz"
    path.rename(target)
    path.symlink_to(target.name)
    with pytest.raises(ValueError, match="Symlink"):
        comparison.compare_runs(baseline, candidate)


def test_native_gap_is_rejected_even_when_both_match(tmp_path):
    baseline, candidate = pair(tmp_path, lambda rows: rows["native"].pop(4))
    with pytest.raises(ValueError, match="remain contiguous"):
        comparison.compare_runs(baseline, candidate)


def test_single_passing_frame_does_not_claim_continuous_admission():
    rows = [{"time_s": .002, "pass": False}, {"time_s": .004, "pass": True}, {"time_s": .006, "pass": False}]
    value = comparison.flag_summary(rows, lambda row: row["pass"], .002)
    assert value["maximum_consecutive_samples"] == 1
    assert value["sample_interval_coverage_s"] == .002
    assert value["elapsed_between_first_and_last_pass_s"] == 0.


def test_numeric_nonfinite_and_inventory_changes_fail():
    difference = comparison.Differences()
    difference.compare({"state": "CLOSE", "value": [1.]}, {"state": "RECOVER", "value": []}, .002)
    assert difference.mismatches == 2
    with pytest.raises(ValueError, match="finite"):
        difference.compare(float("nan"), 0., .004)


def test_identity_smoke_is_explicit(tmp_path):
    baseline = fixture_run(tmp_path / "baseline")
    with pytest.raises(ValueError, match="Candidate protocol"):
        comparison.compare_runs(baseline, baseline)
    assert comparison.compare_runs(baseline, baseline, identity_check=True)["identity_check_only"]
