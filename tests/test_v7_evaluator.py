"""Evidence contracts for supported regrasp and collision-aware transfer."""

from pathlib import Path
import sys
import math

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_evaluator import (
    CollisionEvent,
    PoseSnapshot,
    RegraspMeasurement,
    RegraspSpec,
    aggregate_collision_events,
    evaluate_regrasp,
    RegraspTraceSample,
)


def test_supported_regrasp_requires_relative_motion_with_bounded_wrist():
    before = PoseSnapshot(0.0, (0., 0., .6), (.10, 0., .6))
    after = PoseSnapshot(1.0, (0., 0., .6), (.125, 0., .6))
    trace = (
        RegraspTraceSample(before, 2., .040, 2, True, .60),
        RegraspTraceSample(PoseSnapshot(.5, (0., 0., .6), (.112, 0., .6)), 2., .034, 1, True, .601),
        RegraspTraceSample(after, 2., .030, 2, True, .6005),
    )
    measurement = RegraspMeasurement(before, after, trace)
    evidence = evaluate_regrasp(measurement)
    assert evidence.accepted
    assert evidence.relative_translation_m == pytest.approx(.025)
    assert evidence.wrist_translation_m == pytest.approx(0.)
    assert "RELATIVE_TRANSLATION_THRESHOLD" in evidence.reasons


def test_wrist_motion_cannot_fake_relative_regrasp():
    before = PoseSnapshot(0.0, (0., 0., .6), (.10, 0., .6))
    after = PoseSnapshot(.5, (.10, 0., .6), (.20, 0., .6))
    trace = (
        RegraspTraceSample(before, 2., .040, 2, True, .60),
        RegraspTraceSample(after, 2., .040, 2, True, .60),
    )
    evidence = evaluate_regrasp(RegraspMeasurement(before, after, trace))
    assert not evidence.accepted
    assert evidence.relative_translation_m == pytest.approx(0.)
    assert "WRIST_MOVED_TOO_FAR" in evidence.reasons
    assert "INSUFFICIENT_RELATIVE_OBJECT_MOTION" in evidence.reasons


def test_relative_rotation_can_satisfy_either_requirement_without_wrist_rotation():
    quarter_turn = (math.cos(math.pi / 8), 0., 0., math.sin(math.pi / 8))
    before = PoseSnapshot(0.0, (0., 0., .6), (.10, 0., .6))
    after = PoseSnapshot(.5, (0., 0., .6), (.10, 0., .6), object_quaternion=quarter_turn)
    trace = (
        RegraspTraceSample(before, 2., .040, 2, True, .60),
        RegraspTraceSample(after, 2., .034, 1, True, .601),
    )
    evidence = evaluate_regrasp(RegraspMeasurement(before, after, trace))
    assert evidence.accepted
    assert evidence.relative_rotation_rad == pytest.approx(math.pi / 4)
    assert evidence.relative_translation_m == pytest.approx(0.)


def test_collision_audit_separates_obstacles_non_targets_and_target_contacts():
    audit = aggregate_collision_events(
        [
            CollisionEvent(.2, "left_index_fingertip_collision", "v7_obstacle_divider_center", 4.),
            CollisionEvent(.3, "left_index_fingertip_collision", "v7_geom_distractor_box", 2.),
            CollisionEvent(.4, "left_index_fingertip_collision", "v7_geom_tool_screwdriver", 1.),
            CollisionEvent(.5, "v7_geom_tool_screwdriver", "table", .5),
        ],
        obstacle_ids=("divider_center",),
        non_target_ids=("distractor_box",),
        target_ids=("tool_screwdriver",),
    )
    assert not audit.workspace_safe
    assert audit.obstacle_events == 1
    assert audit.non_target_events == 1
    assert audit.target_events == 1
    assert audit.unclassified_events == 0
    assert audit.ignored_environment_events == 1
    assert audit.first_obstacle_time_s == pytest.approx(.2)
    assert audit.peak_non_target_force_n == pytest.approx(2.)
    assert audit.as_dict()["workspace_safe"] is False


def test_collision_event_rejects_invalid_force():
    with pytest.raises(ValueError, match="normal_force_n"):
        CollisionEvent(0., "a", "b", -1.)


def test_unknown_robot_contact_is_not_silent_and_collision_id_sets_are_disjoint():
    audit = aggregate_collision_events(
        [CollisionEvent(.1, "left_index_fingertip_collision", "mystery_fixture", 1.)],
        obstacle_ids=("divider_center",), target_ids=("tool_marker",),
    )
    assert not audit.workspace_safe
    assert audit.unknown_robot_events == 1
    with pytest.raises(ValueError, match="disjoint"):
        aggregate_collision_events([], obstacle_ids=("same",), target_ids=("same",))

