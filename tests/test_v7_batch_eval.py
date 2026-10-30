"""Batch metrics keep missing evidence separate from measured failures."""

from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_batch_eval import (  # noqa: E402
    BatchMetricSpec,
    evaluate_batch,
    evaluate_episode,
)
from kuka_sharpa_grasp.v7_scene import default_v7_scene_config  # noqa: E402
from kuka_sharpa_grasp.v7_evaluator import CollisionEvent  # noqa: E402


def _sample(time, position, *, target="tool_marker", normal=1.0, selected=True,
            interventions=0):
    return {
        "time_s": time,
        "selected_target": target if selected else "tool_screwdriver",
        # Close enough to the tool for grasp evidence; contact alone is not
        # sufficient because arbitrary obstacle touch must not count.
        "wrist_position": [0.76, -0.05, 0.46],
        "wrist_quaternion": [1., 0., 0., 0.],
        "interventions": interventions,
        "tactile": {"fingers": {
            "thumb": {"normal_force_n": normal, "tangential_force_n": .1, "slip_score": .1, "contacts": 1},
            "index": {"normal_force_n": normal, "tangential_force_n": .1, "slip_score": .1, "contacts": 1},
        }},
        "evaluation": {target: {"position": list(position), "quaternion": [1., 0., 0., 0.]}},
    }


def test_success_requires_observed_collision_log_and_has_explicit_denominator():
    config = default_v7_scene_config(0)
    slot = next(item for item in config.slots if item.accepted_object_id == "tool_marker")
    initial = config.registry.by_id("tool_marker").pose
    trace = [_sample(0., initial),
             _sample(.1, (initial[0], initial[1], initial[2] + .04)),
             _sample(.2, (initial[0], initial[1], initial[2] + .05)),
             _sample(.3, (initial[0], initial[1], initial[2] + .045)),
             _sample(.4, slot.pose)]
    result = evaluate_episode("good", trace, target_id="tool_marker", events=[], config=config)
    assert result.lift_success is True
    assert result.drop_detected is False  # controlled descent into the slot
    assert result.docking_success is True
    assert result.success is True
    explicit = evaluate_episode("good_event", trace, target_id="tool_marker",
                                events=[{"event": "OPERATOR_TARGET", "target": "tool_marker", "time_s": 0.0}], config=config)
    assert explicit.target_selection_source == "events.OPERATOR_TARGET"
    summary = evaluate_batch([{"episode_id": "good", "trace": trace, "events": [], "target_id": "tool_marker"}], config=config)
    payload = summary.as_dict()
    assert payload["success_count"] == 1
    assert payload["success_denominator"] == 1
    assert payload["success_rate"] == pytest.approx(1.)


def test_missing_collision_log_keeps_whole_task_unknown():
    config = default_v7_scene_config(0)
    trace = [_sample(0., config.registry.by_id("tool_marker").pose)]
    result = evaluate_episode("incomplete", trace, target_id="tool_marker", events=None, config=config)
    assert result.collision_evidence_available is False
    assert result.success is None
    assert result.non_target_collision_count is None
    assert "missing_collision_log" in result.unknown_reasons


def test_drop_and_non_target_collision_are_measured_as_failures():
    config = default_v7_scene_config(0)
    initial = config.registry.by_id("tool_marker").pose
    trace = [_sample(0., initial),
             _sample(.1, (initial[0], initial[1], initial[2] + .05)),
             _sample(.2, (initial[0], initial[1], initial[2] + .05)),
             _sample(.3, (initial[0], initial[1], initial[2] + .05)),
             _sample(.4, (initial[0], initial[1], initial[2] - .03))]
    events = [{"time_s": .2, "robot_geom": "left_index_fingertip_collision",
               "other_geom": "v7_geom_distractor_box", "normal_force_n": 2.}]
    result = evaluate_episode("drop", trace, target_id="tool_marker", events=events, config=config)
    assert result.drop_detected is True
    assert result.non_target_collision_count == 1
    assert result.success is False


def test_regrasp_is_unknown_without_explicit_continuous_trace_fields():
    config = default_v7_scene_config(0)
    trace = [_sample(0., config.registry.by_id("tool_marker").pose),
             _sample(.1, (0.80, -0.05, .48))]
    result = evaluate_episode(
        "regrasp", trace, target_id="tool_marker", events=[], config=config,
        thresholds=BatchMetricSpec(require_regrasp=True))
    assert result.regrasp is None
    assert result.success is None
    assert "missing_regrasp_trace" in result.unknown_reasons


def test_collision_event_objects_are_accepted_as_batch_input():
    config = default_v7_scene_config(0)
    trace = [_sample(0., config.registry.by_id("tool_marker").pose)]
    result = evaluate_episode(
        "event_object", trace, target_id="tool_marker",
        events=[CollisionEvent(.1, "left_index_fingertip_collision", "v7_obstacle_divider_center", 1.)],
        config=config)
    assert result.obstacle_collision_count == 1


def test_force_on_other_contact_is_not_attributed_to_target_grasp():
    config = default_v7_scene_config(0)
    initial = config.registry.by_id("tool_marker").pose
    # Contact is recorded while the wrist is away from the target; the target
    # later moves, but there is no same-sample target/contact overlap.
    trace = [_sample(0., initial, normal=0.),
             {**_sample(.1, (initial[0], initial[1], initial[2] + .04), normal=4.),
              "wrist_position": [0.3, 0.3, 0.8]},
             {**_sample(.2, (initial[0], initial[1], initial[2] + .05), normal=0.),
              "wrist_position": [0.76, -0.05, 0.46]},
             {**_sample(.3, (initial[0], initial[1], initial[2] + .05), normal=0.),
              "wrist_position": [0.76, -0.05, 0.46]}]
    result = evaluate_episode("wrong_contact", trace, target_id="tool_marker", events=[], config=config)
    assert result.grasp_evidence is False
