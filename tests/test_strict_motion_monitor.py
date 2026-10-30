"""Native-evidence boundaries, retained failure extrema and sustained predicates."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kuka_sharpa_grasp.strict_motion_monitor import CHANNELS, StrictMotionMonitor, StrictMotionSample


def sample(timestamp=.002, **overrides):
    values = dict(timestamp_s=timestamp, target_position_world_m=(0., 0., .071),
        target_linear_speed_m_s=0., target_angular_speed_rad_s=0., target_hand_contact=True,
        target_hand_peak_normal_n=.051, target_table_peak_normal_n=.05, hand_clearance_m=.11,
        hand_channel_forces_n=(1., 0., 0., 0., 0., 0.), minimum_robot_table_distance_m=None,
        release_commanded=False, phase="HELD")
    values.update(overrides)
    return StrictMotionSample(**values)


def monitor():
    return StrictMotionMonitor(initial_target_height_m=0., goal_position_world_m=(0., 0., 0.))


def released_sample(timestamp, **overrides):
    values = dict(target_position_world_m=(0., 0., 0.), target_hand_contact=False,
        target_hand_peak_normal_n=0., target_table_peak_normal_n=.051, release_commanded=True,
        hand_channel_forces_n=(0.,)*6, phase="RELEASE")
    values.update(overrides)
    return sample(timestamp, **values)


def complete_episode(instance):
    for step in range(1, 52):
        instance.observe(sample(step * .002))
    for step in range(52, 1553):
        instance.observe(released_sample(step * .002))


def final(instance, **overrides):
    count = instance.result()["accepted_samples"]
    audit = dict(external_target_qpos_writes=0, integration_steps=count, native_samples=count,
                 original_task_success=True)
    audit.update(overrides)
    return instance.result(**audit)


def test_complete_episode_passes_exact_duration_boundaries_only_with_native_audit():
    instance = monitor()
    complete_episode(instance)
    unaudited = instance.result()
    assert unaudited["held_seconds"] == pytest.approx(.1)
    assert unaudited["released_stable_seconds"] == pytest.approx(3.)
    assert unaudited["strict_motion_gates_passed"]
    assert unaudited["task_success"] is None
    assert unaudited["checks"]["no_external_target_qpos_writes"] is None
    assert final(instance)["task_success"] is True


@pytest.mark.parametrize("overrides", [
    {"target_position_world_m": (0., 0., .07)},
    {"target_hand_peak_normal_n": .05},
    {"target_table_peak_normal_n": .05000001},
])
def test_held_requires_strict_rise_and_loaded_contact_without_loaded_table(overrides):
    instance = monitor()
    assert instance.observe(sample(**overrides))["held"] is False
    assert instance.result()["held_seconds"] == 0.


@pytest.mark.parametrize("overrides", [
    {"release_commanded": False}, {"target_hand_contact": True},
    {"target_table_peak_normal_n": .05},
    {"target_position_world_m": (.025, 0., 0.)},
    {"target_position_world_m": (.018, .018, 0.)},
    {"target_position_world_m": (0., 0., .025)},
    {"target_linear_speed_m_s": .04}, {"target_angular_speed_rad_s": .2},
    {"hand_clearance_m": .10},
])
def test_release_predicates_keep_original_strict_inequalities(overrides):
    instance = monitor()
    assert instance.observe(released_sample(.002, **overrides))["released_stable"] is False


def test_true_contact_with_zero_force_still_prevents_release():
    instance = monitor()
    assert not instance.observe(released_sample(.002, target_hand_contact=True))["released_stable"]


@pytest.mark.parametrize("channel", range(6))
def test_all_six_channel_failure_peaks_survive_later_successful_motion(channel):
    instance = monitor()
    forces = [0.] * 6
    forces[channel] = 12.00001
    instance.observe(sample(.002, hand_channel_forces_n=forces, phase="FAILED_APPROACH"))
    for step in range(2, 53):
        instance.observe(sample(step * .002))
    for step in range(53, 1554):
        instance.observe(released_sample(step * .002))
    report = final(instance)
    assert report["checks"]["held_lift"] and report["checks"]["released_stable_3s"]
    assert report["channel_peaks_n"][CHANNELS[channel]] == 12.00001
    assert report["phase_peaks_n"]["FAILED_APPROACH"] == 12.00001
    assert report["checks"]["all_channels_within_12n"] is False
    assert report["task_success"] is False


def test_force_and_table_penetration_allow_exact_safety_boundaries():
    instance = monitor()
    instance.observe(sample(hand_channel_forces_n=(12.,)*6, minimum_robot_table_distance_m=-.002))
    report = instance.result()
    assert report["checks"]["all_channels_within_12n"]
    assert report["checks"]["no_robot_table_penetration"]
    instance.observe(sample(.004, minimum_robot_table_distance_m=-.00200001))
    instance.observe(sample(.006, minimum_robot_table_distance_m=None))
    assert not instance.result()["checks"]["no_robot_table_penetration"]
    assert instance.result()["minimum_robot_table_distance_m"] == -.00200001


def test_missing_step_resets_held_and_release_runs_and_cannot_recover_continuity():
    instance = monitor()
    for step in range(1, 51):
        instance.observe(sample(step * .002))
    instance.observe(sample(.104))
    assert instance.result()["held_seconds"] == pytest.approx(.098)
    for step in range(53, 104):
        instance.observe(sample(step * .002))
    assert instance.result()["checks"]["held_lift"]
    for step in range(104, 1604):
        instance.observe(released_sample(step * .002))
    instance.observe(released_sample(3.21))
    assert instance.result()["released_stable_seconds"] == pytest.approx(2.998)
    assert not instance.result()["checks"]["released_stable_3s"]
    assert not final(instance)["task_success"]
    assert instance.result()["discontinuities"] == 2


@pytest.mark.parametrize("timestamp", [.002, .001])
def test_duplicate_or_reversed_time_is_rejected_but_safety_extrema_are_retained(timestamp):
    instance = monitor()
    instance.observe(sample())
    with pytest.raises(ValueError, match="duplicate/reversed"):
        instance.observe(sample(timestamp, hand_channel_forces_n=(20.,)*6,
                                minimum_robot_table_distance_m=-.004, phase="REJECTED"))
    report = instance.result()
    assert report["accepted_samples"] == 1 and report["observed_samples"] == 2
    assert report["rejected_timestamps"] == 1
    assert report["whole_episode_peak_n"] == 20.
    assert report["phase_peaks_n"]["REJECTED"] == 20.
    assert not report["checks"]["native_contiguous_2ms"]
    assert not report["checks"]["no_robot_table_penetration"]


@pytest.mark.parametrize("timestamp", [0., 22., .003])
def test_partial_checkpoint_window_does_not_claim_full_native_coverage(timestamp):
    instance = monitor()
    instance.observe(sample(timestamp))
    assert not instance.result()["checks"]["native_contiguous_2ms"]


def test_false_predicate_resets_continuous_hold_and_release():
    instance = monitor()
    for step in range(1, 51):
        instance.observe(sample(step * .002))
    instance.observe(sample(.102, target_hand_peak_normal_n=0.))
    instance.observe(sample(.104))
    assert instance.result()["held_seconds"] == pytest.approx(.098)
    assert not instance.result()["checks"]["held_lift"]
    for step in range(53, 1553):
        instance.observe(released_sample(step * .002))
    instance.observe(released_sample(3.106, target_hand_contact=True))
    instance.observe(released_sample(3.108))
    assert instance.result()["released_stable_seconds"] == pytest.approx(2.998)


@pytest.mark.parametrize("audit", [
    {"external_target_qpos_writes": 1}, {"integration_steps": 1553},
    {"native_samples": 1551}, {"external_target_qpos_writes": None},
])
def test_success_requires_complete_zero_write_matching_native_audit(audit):
    instance = monitor()
    complete_episode(instance)
    assert instance.result()["strict_motion_gates_passed"]
    assert final(instance, **audit)["task_success"] is (None if None in audit.values() else False)


@pytest.mark.parametrize("overrides", [
    {"timestamp_s": True}, {"timestamp_s": -1.}, {"timestamp_s": float("nan")},
    {"target_position_world_m": (0., 0.)}, {"target_position_world_m": (0., False, 0.)},
    {"target_linear_speed_m_s": -1.}, {"target_angular_speed_rad_s": float("inf")},
    {"target_hand_contact": 1}, {"target_hand_contact": False},
    {"hand_channel_forces_n": (0.,)*5}, {"hand_channel_forces_n": (-1.,)*6},
    {"minimum_robot_table_distance_m": float("nan")}, {"release_commanded": 0},
    {"phase": ""},
])
def test_invalid_or_ambiguous_samples_are_rejected(overrides):
    with pytest.raises(ValueError):
        sample(**overrides)


def test_samples_copy_sequences_and_reports_cannot_mutate_monitor_evidence():
    forces, position = [1.] * 6, [0., 0., .071]
    record = sample(hand_channel_forces_n=forces, target_position_world_m=position)
    forces[0], position[2] = 100., 5.
    assert record.hand_channel_forces_n == (1.,)*6
    assert record.target_position_world_m == (0., 0., .071)
    with pytest.raises(FrozenInstanceError):
        record.timestamp_s = .004
    instance = monitor()
    instance.observe(record)
    report = instance.result()
    report["channel_peaks_n"]["thumb"] = 100.
    report["phase_peaks_n"]["HELD"] = 100.
    assert instance.result()["whole_episode_peak_n"] == 1.
    assert instance.result()["phase_peaks_n"]["HELD"] == 1.


def test_motion_probe_and_empty_episode_report_failure_without_raising():
    instance = monitor()
    assert instance.result()["task_success"] is None
    assert final(instance, original_task_success=False)["task_success"] is False
    for step in range(1, 101):
        instance.observe(sample(step * .002, target_position_world_m=(0., 0., .005)))
    report = final(instance)
    assert not report["strict_motion_gates_passed"] and report["accepted_samples"] == 100
    assert report["whole_episode_peak_n"] == 1.
    assert report["target_privileged_evaluation_only"]


@pytest.mark.parametrize("value", [True, -1, .0, "0"])
def test_invalid_native_audit_counts_rejected(value):
    with pytest.raises(ValueError):
        monitor().result(external_target_qpos_writes=value)


def test_monitor_never_creates_or_promotes_an_original_task_label():
    instance = monitor()
    complete_episode(instance)
    assert final(instance, original_task_success=None)["task_success"] is None
    assert final(instance, original_task_success=False)["task_success"] is False
    assert instance.result(original_task_success=True)["task_success"] is None
    with pytest.raises(ValueError):
        instance.result(original_task_success=1)
