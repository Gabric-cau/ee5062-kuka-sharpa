"""Final evidence gates and exact-clock R23 probe-only plot reduction."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_public_ycb_actual_wrist_probe as plot


def exit_fixture(tmp_path):
    status = dict(state="FINISHED", terminal=True, child_exit_observed=True, returncode=0)
    request = {"command": ["python", "scripts/develop_public_ycb_actual_wrist_probe_r23.py",
        "--output", str(tmp_path)], "supervisor_sha256": "a" * 64}
    protocol = {"source_hashes_before": {"scripts/run_supervised_experiment.py": "a" * 64}}
    return status, request, tmp_path, protocol


@pytest.mark.parametrize("change", [{"state": "RUNNING"}, {"terminal": False},
    {"child_exit_observed": False}, {"returncode": 1}, {"returncode": False}])
def test_running_unobserved_and_nonzero_or_boolean_exit_rejected(tmp_path, change):
    values = exit_fixture(tmp_path)
    plot.check_exit(*values)
    values[0].update(change)
    with pytest.raises(ValueError, match="FINISHED rc0"):
        plot.check_exit(*values)


def test_supervisor_output_and_runner_identity_are_bound(tmp_path):
    values = exit_fixture(tmp_path)
    values[1]["command"][-1] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="FINISHED rc0"):
        plot.check_exit(*values)
    values = exit_fixture(tmp_path)
    values[1]["command"] += ["--output", str(tmp_path)]
    with pytest.raises(ValueError, match="unique"):
        plot.check_exit(*values)


def receipt_fixture():
    keys = ("summary", "protocol", "archive", "scene", "controller", "metadata", "result", "native",
        "trace", "load_transfer_probe", "bias_recovery", "model_loader", "alignment_auditor", "probe_auditor")
    inputs = {key: {"path": "/bound/" + key, "bytes": 1, "sha256": "a" * 64} for key in keys}
    row = {"native_audit": {"native_samples": 10}, "effective_hand_instances": 2, "success": False, "held_lift": False}
    receipt = {"format": plot.AUDIT_FORMAT, "accepted_probe_evidence": True,
        "checks": dict.fromkeys(plot.CHECKS, True), "summary_sha256": "a" * 64, "protocol_sha256": "a" * 64,
        "script_sha256": "a" * 64, "source": {"source_count": 167, "inherited_r22_source_count": 146},
        "native_samples": 10, "native_timestep_s": .002, "actual_hand_instances": 2,
        "task_success": False, "held_lift": False, "inputs": copy.deepcopy(inputs)}
    return receipt, inputs, row


def test_failed_task_can_have_accepted_complete_diagnostic_evidence():
    plot.check_probe_receipt(*receipt_fixture())


@pytest.mark.parametrize("change", ["check", "source", "outcome", "native", "prefix", "script",
    "bias_sha256", "bias_bytes", "bias_path"])
def test_unknown_prefix_misbound_or_unaccepted_probe_receipt_rejected(change):
    receipt, inputs, row = receipt_fixture()
    if change == "check":
        receipt["checks"][plot.CHECKS[0]] = False
    elif change == "source":
        receipt["source"]["source_count"] = 146
    elif change == "outcome":
        receipt["task_success"] = True
    elif change == "native":
        receipt["inputs"]["native"]["sha256"] = "b" * 64
    elif change == "script":
        receipt["script_sha256"] = "b" * 64
    elif change.startswith("bias_"):
        field = change.removeprefix("bias_")
        receipt["inputs"]["bias_recovery"][field] = 2 if field == "bytes" else "different"
    else:
        receipt["native_samples"] = 9
    with pytest.raises(ValueError, match="receipt"):
        plot.check_probe_receipt(receipt, inputs, row)


class Kinematics:
    def __init__(self, _=None):
        pass

    def height_at(self, command):
        return .8


def trace_row(timestamp, phase="TRIAL_LIFT", probe=None):
    return {"timestamp": timestamp, "state": phase, "wrist_position": [0., 0., .8],
        "arm_command": [0.] * 7, "hand_command": [0.] * 22,
        "actual_wrist_probe": probe, "probe_policy": {"probe_confirmed_seconds": 0.,
            "actual_probe_endpoint_reached": False, "table_unloaded": True,
            "sampled_fixed_normal_1g_feasible": True}}


def test_first_same_clock_trace_keeps_probe_and_excludes_phase_end_duplicate():
    first = trace_row(.002, "CLOSE")
    later = trace_row(.002, probe={"later": True})
    values, duplicates = plot.reduce_trace([first, later], Kinematics())
    assert duplicates == 1 and values[1]["phase"] == "CLOSE"
    assert values[1]["actual_wrist_probe"] is None
    native = {"timestamp": .002, "state": "CLOSE", "motor_commands": [0.] * 29}
    assert plot.common.exact_native_trace(native, 1, values)[1]["phase"] == "CLOSE"
    with pytest.raises(ValueError, match="phase/hand"):
        plot.common.exact_native_trace({**native, "motor_commands": [0.] * 28 + [.1]}, 1, values)


def extraction_fixture(monkeypatch, *, failed_first_attempt=False):
    class Model:
        ngeom = 0

        def geom(self, _):
            return SimpleNamespace(id=0)

    native, trace = [], []
    def append(phase, probe=None):
        timestamp = (len(native) + 1) * .002
        native.append({"timestamp": timestamp, "state": phase, "motor_commands": [0.] * 29})
        trace.append(trace_row(timestamp, phase, probe))
        return timestamp
    if failed_first_attempt:
        append("SEARCH")
        append("RELEASE")
    append("SEARCH")
    start = append("CLOSE")
    samples = []
    for _ in range(2):
        timestamp = (len(native) + 1) * .002
        sample = {"timestamp_s": timestamp, "elapsed_s": timestamp - start,
            "current_actual_wrist_world_m": [0., 0., .8], "inherited_confirmed_seconds": 0.,
            "currently_probe_active": True}
        samples.append(sample)
        append("TRIAL_LIFT", sample)
    # This later native TRIAL_LIFT is a different interval and must not be included.
    append("TRIAL_LIFT")
    outcome = {"arm_exit_reason": "MAXIMUM_PROBE_DURATION_REACHED",
        "final_reason": "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED", "original_confirmation_attempted": True,
        "original_confirmation_accepted": False, "confirmation_timestamp_s": samples[-1]["timestamp_s"],
        "inherited_confirmed_seconds": 0., "timestamp_s": samples[-1]["timestamp_s"],
        "motion_failure": None, "probe_active_at_motion_end": True}
    record = {"start_s": start, "actual_admission_start_world_m": [0., 0., .8], "samples": samples,
        "ik_samples": [{"timestamp_s": start, "desired_actual_path_world_m": [0., 0., .800001],
            "cartesian_ik_target_world_m": [0., 0., .800003], "bounded_reference_correction_world_m": [0., 0., .000002]}],
        "outcome": outcome}
    attempts = [{"attempt": 1, "failure": "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED"}]
    if failed_first_attempt:
        attempts.insert(0, {"attempt": 1, "failure": "PROBE_NOT_ELIGIBLE"})
        attempts[-1]["attempt"] = 2
    row = {"native_audit": {"native_samples": len(native)}, "duration_s": len(native) * .002,
        "actual_wrist_probe": {"format": "actual_wrist_probe_policy_v1", "attempts": [record]},
        "attempts": attempts, "success": False, "held_lift": False}
    monkeypatch.setattr(plot.common, "CommandKinematics", Kinematics)
    bias = []
    streams = {"native": native, "trace": trace, "bias_recovery": bias}
    monkeypatch.setattr(plot.common, "json_array", lambda path: iter(streams[path.name]))
    monkeypatch.setattr(plot.common, "vertical_forces", lambda *args: {key: value for key, value in
        zip(plot.common.FORCES, (0., 5., -1., 4., 0.))})
    audited = {key: outcome[key] for key in ("arm_exit_reason", "final_reason", "original_confirmation_attempted",
        "original_confirmation_accepted")}
    audited.update(original_r16_pre_sample_abort_observed=False, classification="original_confirmation_rejected",
        start_s=start, end_s=outcome["timestamp_s"], native_samples=2, ik_samples=1, final_confirmed_seconds=0.)
    return {"model": Model(), "row": row, "inputs": {key: {"path": key} for key in streams},
        "audit_probe_count": 1, "audited_probes": [audited], "weight_n": 4.}, native, trace, bias


@pytest.mark.parametrize("failed_first", [False, True])
def test_admission_clock_can_be_close_and_only_recorded_probe_native_is_selected(monkeypatch, failed_first):
    run, native, trace, _ = extraction_fixture(monkeypatch, failed_first_attempt=failed_first)
    result = plot.extract(run)
    assert len(result["probes"]) == 1
    probe = result["probes"][0]
    assert probe["global_attempt"] == (2 if failed_first else 1)
    assert len(probe["series"]) == 2
    assert probe["ik_samples"][0]["timestamp_s"] == probe["start_s"]
    assert probe["series"][0]["elapsed_s"] == pytest.approx(.002)
    assert probe["statistics"]["hand_tangent_z_n"]["last"] == -1.
    assert len(native) > len(probe["series"])


@pytest.mark.parametrize("change", ["missing_tick", "wrist", "counter", "missing_probe", "short_native"])
def test_missing_or_mismatched_probe_native_is_rejected_without_interpolation(monkeypatch, change):
    run, native, trace, _ = extraction_fixture(monkeypatch)
    if change == "missing_tick":
        trace[2]["timestamp"] += .001
    elif change == "wrist":
        trace[2]["wrist_position"][2] += .001
    elif change == "counter":
        trace[2]["probe_policy"]["probe_confirmed_seconds"] = .01
    elif change == "missing_probe":
        trace[2]["actual_wrist_probe"] = None
    else:
        native.pop()
    with pytest.raises(ValueError):
        plot.extract(run)


def test_absent_probe_does_not_become_synthetic_zero_measurements(monkeypatch):
    run, _, trace, _ = extraction_fixture(monkeypatch)
    run["audit_probe_count"] = 0
    run["audited_probes"] = []
    run["row"]["actual_wrist_probe"]["attempts"] = []
    assert plot.extract(run)["probes"] == []


def test_five_panel_failed_probe_png_and_pdf_are_nonblank(monkeypatch, tmp_path):
    run, _, _, _ = extraction_fixture(monkeypatch)
    run = plot.extract(run)
    probe = run["probes"][0]
    template = probe["series"][0]
    probe["series"] = [{**template, "elapsed_s": float(t), "wrist_delta_z_m": .004 * min(1., t / 3.),
        "endpoint_error_3d_m": .005 - .004 * min(1., t / 3.)}
        for t in np.arange(.002, 6.001, .002)]
    probe["ik_samples"] = []
    for t in np.arange(0., 6., .02):
        fraction = min(1., (t + .02) / 3.)
        desired = .005 * fraction * fraction * (3. - 2. * fraction)
        actual = .004 * min(1., t / 3.)
        correction = min(.002, max(-.002, 2. * (desired - actual)))
        probe["ik_samples"].append({"timestamp_s": probe["start_s"] + float(t),
            "desired_actual_path_world_m": [0., 0., .8 + desired],
            "cartesian_ik_target_world_m": [0., 0., .8 + desired + correction],
            "bounded_reference_correction_world_m": [0., 0., correction]})
    probe["outcome"]["confirmation_timestamp_s"] = probe["start_s"] + 6.
    artifacts = plot.figures(run, tmp_path)
    assert {Path(row["path"]).suffix for row in artifacts} == {".png", ".pdf"}
    png = next(Path(row["path"]) for row in artifacts if row["path"].endswith(".png"))
    pixels = np.asarray(Image.open(png).convert("RGB"))
    assert pixels.shape == (2100, 1800, 3)
    assert float(pixels.std()) > 10.
    assert np.count_nonzero(np.any(pixels < 235, axis=2)) > 10000
    assert all(row["bytes"] > 10000 for row in artifacts)
    assert run["row"]["success"] is False


def abort_fixture(monkeypatch):
    run, native, trace, bias = extraction_fixture(monkeypatch)
    record = run["row"]["actual_wrist_probe"]["attempts"][0]
    outcome, audited = record["outcome"], run["audited_probes"][0]
    end = outcome["timestamp_s"]
    outcome.update(arm_exit_reason="MOTION_EXCEPTION", final_reason="PROBE_BIAS_RECOVERY",
        motion_failure="PROBE_BIAS_RECOVERY", probe_active_at_motion_end=False,
        original_confirmation_attempted=False, original_confirmation_accepted=None)
    outcome.pop("confirmation_timestamp_s")
    audited.update({key: outcome[key] for key in ("arm_exit_reason", "final_reason",
        "original_confirmation_attempted", "original_confirmation_accepted")})
    audited.update(original_r16_pre_sample_abort_observed=True, classification="motion_failure")
    record["samples"][-1]["currently_probe_active"] = False
    trace[3]["probe_policy"]["probe_confirmed_seconds"] = .002
    bias.append({"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
        "additional_bias_limit_rad": .10, "timestamp_s": end - .002, "post_step_timestamp_s": end,
        "phase": "TRIAL_LIFT", "probe_abort_applied": True, "recovery_active": True, "failure_latched": False,
        "admission_before": {"probe_active": True, "payload_confirmed": False},
        "admission_after_step": {"probe_active": False, "payload_confirmed": False, "probe_confirmed_seconds": 0.,
            "probe_eligible_seconds": 0., "pressure_window_samples": 0, "preload_ready": False},
        "post_step_returned_failure": "PROBE_BIAS_RECOVERY", "resolved_abort_reason": "PROBE_BIAS_RECOVERY",
        "original_post_step_failure": None})
    return run, trace, bias


def test_audited_exact_final_r16_abort_preserves_both_counters_and_no_confirmation(monkeypatch):
    run, _, _ = abort_fixture(monkeypatch)
    result = plot.extract(run)
    probe = result["probes"][0]
    assert [row["audited_r16_abort_before_sample"] for row in probe["series"]] == [False, True]
    assert probe["series"][-1]["parent_policy_confirmed_seconds"] == .002
    assert probe["series"][-1]["inherited_confirmed_seconds"] == 0.
    assert probe["outcome"]["original_confirmation_attempted"] is False
    assert probe["outcome"]["original_confirmation_accepted"] is None


def test_r16_abort_plot_exposes_both_counters_without_decisive_confirmation(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    run, _, _ = abort_fixture(monkeypatch)
    result = plot.extract(run)
    captured = {}
    original_close = plt.close

    def capture(figure):
        if hasattr(figure, "axes"):
            captured["labels"] = figure.axes[-1].get_legend_handles_labels()[1]
            captured["footer"] = figure.texts[-1].get_text()
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture)
    artifacts = plot.figures(result, tmp_path)
    assert "Parent hand policy before final R16 inhibit" in captured["labels"]
    assert "Recorded hand counter after native-step / R16 inhibit" in captured["labels"]
    assert "Audited R16 abort; no decisive confirmation" in captured["labels"]
    assert "Original decisive confirmation" not in captured["labels"]
    assert "parent=0.002000s, post-R16/sample=0.000000s" in captured["footer"]
    assert "decisive confirmation attempted=False" in captured["footer"]
    png = next(Path(row["path"]) for row in artifacts if row["path"].endswith(".png"))
    pixels = np.asarray(Image.open(png).convert("RGB"))
    assert float(pixels.std()) > 10.
    assert all(row["bytes"] > 10000 for row in artifacts)


@pytest.mark.parametrize("change", ["unclassified", "wrong_classification", "early_difference", "wrong_end",
    "missing_raw_abort", "wrong_raw_clock", "outcome_active", "sample_active", "wrong_failure", "parent_too_large"])
def test_counter_difference_needs_classified_exact_last_tick_inactive_abort_chain(monkeypatch, change):
    run, trace, bias = abort_fixture(monkeypatch)
    record, audited = run["row"]["actual_wrist_probe"]["attempts"][0], run["audited_probes"][0]
    if change == "unclassified":
        audited["original_r16_pre_sample_abort_observed"] = False
    elif change == "wrong_classification":
        audited["classification"] = "original_confirmation_accepted_probe_only"
    elif change == "early_difference":
        trace[2]["probe_policy"]["probe_confirmed_seconds"] = .002
    elif change == "wrong_end":
        record["outcome"]["timestamp_s"] -= .002
        audited["end_s"] -= .002
    elif change == "missing_raw_abort":
        bias.clear()
    elif change == "wrong_raw_clock":
        bias[0]["timestamp_s"] -= .002
    elif change == "outcome_active":
        record["outcome"]["probe_active_at_motion_end"] = True
    elif change == "sample_active":
        record["samples"][-1]["currently_probe_active"] = True
    elif change == "wrong_failure":
        bias[0]["post_step_returned_failure"] = "OTHER_FAILURE"
    else:
        trace[3]["probe_policy"]["probe_confirmed_seconds"] = .004
    with pytest.raises(ValueError):
        plot.extract(run)
