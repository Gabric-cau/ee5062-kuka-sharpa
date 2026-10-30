"""Reference and binding checks only; no native integration or task claim."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import actual_wrist_probe_controller_hooks as inherited
import rate_limited_probe_controller_hooks as hook


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/public_ycb_actual_wrist_probe_20261022_r23_bias10_close32_seed3414_retry1/ycb_can_bias_budget_r17_seed3414/derived_controller.py"
START = np.array([.6, .02, .6])
KIN = SimpleNamespace(body=0)


class Failure(RuntimeError):
    pass


def data(clock=0., actual=START):
    return SimpleNamespace(time=clock, xpos=np.array([actual, [11., 22., 33.]]))


def tracker():
    return hook.RateLimitedProbeTracker(0., 0., START)


@pytest.fixture
def bound():
    source = SOURCE.read_text()
    namespace = {"sentinel": object(), "_r23_probe_telemetry": inherited.ProbeTelemetry(),
        "_r23_probe_parameters": dict(inherited.PARAMETERS)}
    exec(compile(source, "<actual_frozen_arm_for_governor_binding>", "exec"), namespace)
    original = namespace["run_task"]
    original.r19_source_transform = object()
    original.r22_alignment_transform = object()
    original.actual_wrist_probe_telemetry = namespace["_r23_probe_telemetry"]
    return original, source


def test_first_endpoint_starts_once_at_admission_and_has_a_vector_speed_bound():
    item = tracker()
    actual = START + [.001, -.001, 0.]
    applied = item.target(data(actual=actual), KIN, .02, Failure)
    row = item.ik_samples[-1]
    raw = inherited.bounded_probe_reference(START, actual, .02)["cartesian_ik_target_world_m"]
    alpha = 1. - np.exp(-.02 / .10)
    expected_filtered = START + alpha * (np.asarray(raw) - START)
    np.testing.assert_allclose(row["raw_cartesian_ik_target_world_m"], raw, atol=1e-15)
    np.testing.assert_allclose(row["filtered_cartesian_ik_target_world_m"], expected_filtered, atol=1e-15)
    assert np.linalg.norm(applied - START) == pytest.approx(.004 * .02)
    assert row["reference_dt_s"] == .02
    assert row["actual_wrist_sample_dt_s"] == 0.
    assert row["actual_wrist_speed_since_previous_ik_m_s"] is None
    assert not np.allclose(applied, actual, rtol=0., atol=1e-5)


def test_full_six_second_reference_keeps_raw_policy_and_all_3d_segments_bounded():
    item, previous = tracker(), START.copy()
    for index in range(300):
        clock, endpoint = index * .02, (index + 1) * .02
        actual = START + np.array([.002 * np.sin(7. * clock), .002 * np.cos(9. * clock), .0004 * clock])
        applied = item.target(data(clock, actual), KIN, endpoint, Failure)
        row = item.ik_samples[-1]
        raw = inherited.bounded_probe_reference(START, actual, endpoint)
        np.testing.assert_array_equal(row["raw_cartesian_ik_target_world_m"], raw["cartesian_ik_target_world_m"])
        np.testing.assert_array_equal(row["desired_actual_path_world_m"], raw["desired_actual_path_world_m"])
        assert np.linalg.norm(applied - previous) <= .004 * .02 + 1e-15
        assert row["applied_target_speed_m_s"] <= .004 + 1e-15
        np.testing.assert_allclose(row["applied_target_delta_world_m"], applied - previous, atol=1e-16)
        if index:
            assert row["actual_wrist_sample_dt_s"] == pytest.approx(.02)
        previous = applied
    assert len(item.ik_samples) == 300
    np.testing.assert_array_equal(item.ik_samples[-1]["desired_actual_path_world_m"], START + [0., 0., .005])
    assert item.ik_samples[-1]["reference_endpoint_elapsed_s"] == 6.
    json.dumps(item.as_dict(), allow_nan=False)


def test_reversing_actual_error_filters_persistently_and_slews_without_reset():
    item = tracker()
    targets = []
    for index in range(30):
        actual = START + [0., .003 if index < 15 else -.003, 0.]
        targets.append(item.target(data(index * .02, actual), KIN, (index + 1) * .02, Failure))
    before, reversal = item.ik_samples[14:16]
    assert before["raw_cartesian_ik_target_world_m"][1] < START[1]
    assert reversal["raw_cartesian_ik_target_world_m"][1] > START[1]
    assert reversal["filtered_cartesian_ik_target_world_m"][1] < START[1]
    assert targets[-1][1] > targets[15][1]
    for first, second in zip(targets, targets[1:]):
        assert np.linalg.norm(second - first) <= .00008 + 1e-15
    assert reversal["actual_wrist_delta_since_previous_ik_world_m"][1] == pytest.approx(-.006)


@pytest.mark.parametrize("clock,endpoint", [(0., 0.), (0., .04), (0., np.nan),
    (.02, .02), (-.02, .02), (0., 6.02)])
def test_invalid_initial_endpoint_or_native_clock_fails_before_state_advances(clock, endpoint):
    item = tracker()
    with pytest.raises(Failure, match="RATE_LIMITED_PROBE_INVALID_REFERENCE_CLOCK"):
        item.target(data(clock), KIN, endpoint, Failure)
    assert item.ik_samples == []
    np.testing.assert_array_equal(item.applied_target, START)
    np.testing.assert_array_equal(item.filtered_target, START)


def test_repeated_or_skipped_endpoint_and_excess_duration_fail_without_reset():
    item = tracker()
    item.target(data(), KIN, .02, Failure)
    before = copy.deepcopy(item.ik_samples)
    for clock, endpoint in [(0., .02), (.04, .06), (.02, .02)]:
        with pytest.raises(Failure, match="RATE_LIMITED_PROBE_INVALID_REFERENCE_CLOCK"):
            item.target(data(clock), KIN, endpoint, Failure)
        assert item.ik_samples == before
    item.previous_endpoint_s = 6.
    with pytest.raises(Failure, match="RATE_LIMITED_PROBE_INVALID_REFERENCE_CLOCK"):
        item.target(data(6.), KIN, 6.02, Failure)


def test_callable_binding_keeps_arm_code_source_globals_and_old_metadata(bound):
    original, source = bound
    original_globals = dict(original.__globals__)
    original_attributes = dict(original.__dict__)
    wrapped, rendered, telemetry = hook.transform(original, source)
    assert rendered == source
    assert wrapped.__code__ is original.__code__
    assert original.__globals__ == original_globals
    assert original.__dict__ == original_attributes
    for name, value in original_globals.items():
        expected = wrapped if name == "run_task" else telemetry if name == "_r23_probe_telemetry" else value
        assert wrapped.__globals__[name] is expected
    assert wrapped.r19_source_transform is original.r19_source_transform
    assert wrapped.r22_alignment_transform is original.r22_alignment_transform
    assert wrapped.actual_wrist_probe_telemetry is telemetry
    assert wrapped.rate_limited_probe_telemetry is telemetry
    assert original.actual_wrist_probe_telemetry is original_globals["_r23_probe_telemetry"]
    declaration = wrapped.r25_rate_limited_probe_transform
    assert declaration["readable_source_unchanged"] and declaration["compiled_arm_code_unchanged"]
    assert declaration["raw_reference_parameters"] == inherited.PARAMETERS
    assert declaration["runtime_target_state_used"] is False
    assert declaration["target_qpos_writes_added"] is False
    assert telemetry.as_dict()["format"] == "rate_limited_actual_wrist_probe_policy_v1"


def test_changed_source_callable_or_used_telemetry_is_rejected(bound):
    original, source = bound
    with pytest.raises(ValueError, match="source SHA256"):
        hook.transform(original, source + "\n")
    changed = {}
    exec("def run_task():\n    return None\n", changed)
    with pytest.raises(ValueError, match="callable differs"):
        hook.transform(changed["run_task"], source)
    original.__globals__["_r23_probe_telemetry"].attempts.append(object())
    with pytest.raises(ValueError, match="unused original"):
        hook.transform(original, source)


def test_sampling_and_original_decisive_confirmation_share_the_governed_tracker():
    telemetry = hook.RateLimitedProbeTelemetry()
    calls = []
    hand = SimpleNamespace(probe_start_s=0., probe_start_world_m=START,
        probe_active=True, probe_confirmed_seconds=.25,
        confirm_probe=lambda unused: calls.append("original") or False)
    state = data()
    item = telemetry.begin(state, hand, True, Failure)
    item.target(state, KIN, .02, Failure)
    state.time = .002
    sample = item.sample(state, KIN, hand, Failure)
    np.testing.assert_array_equal(sample["cartesian_ik_target_world_m"],
        item.ik_samples[-1]["applied_cartesian_ik_target_world_m"])
    assert sample["reference_governor"] == item.samples[-1]["reference_governor"]
    item.finish(state, hand, "ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED")
    assert telemetry.confirm_probe(hand, state) is False
    assert calls == ["original"]
    assert hook.RateLimitedProbeTelemetry.confirm_probe is inherited.ProbeTelemetry.confirm_probe
    assert item.outcome["original_confirmation_attempted"] is True
    assert item.outcome["original_confirmation_accepted"] is False
    assert item.outcome["final_reason"] == "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED"
    json.dumps(telemetry.as_dict(), allow_nan=False)
