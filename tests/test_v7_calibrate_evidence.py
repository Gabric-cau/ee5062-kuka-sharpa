"""Adversarial calibration scoring tests: impulses must not win a sweep."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_calibrate import (  # noqa: E402
    CalibrationRecord, EVIDENCE_KEY, GraspCandidate, score_lift_evidence,
)
from kuka_sharpa_grasp.v7_task import PhaseRecord, TaskResult  # noqa: E402


def sample(timestamp, height=.06, *, force=2., support=0., fingers=("thumb", "index"), collision=False):
    return {
        "time_s": timestamp, "wrist_position": [0., 0., .7 + height],
        "wrist_quaternion": [1., 0., 0., 0.],
        "evaluation": {"tool_marker": {"position": [0., 0., .45 + height],
                                        "quaternion": [1., 0., 0., 0.]}},
        EVIDENCE_KEY: {"target_hand_force_n": force, "target_hand_contacts": 2,
                       "target_contact_fingers": list(fingers),
                       "target_support_force_n": support,
                       "environment_collisions": [{"normal_force_n": 107.}] if collision else [],
                       "maximum_hand_channel_force_n": force},
    }


def result(rows, *, phase="LIFT", status="COMPLETE"):
    initial = sample(0., 0.)
    initial.pop(EVIDENCE_KEY)
    phases = (PhaseRecord(phase, 0., rows[-1]["time_s"], status, len(rows), 2., 2),)
    return TaskResult("tool_marker", None, False, "FAILED", "DOCK_NOT_REQUESTED", phases,
                      (initial, *rows), {"initial_object_position_m": [0., 0., .45]})


def record(task_result, max_height=.3, force=2.):
    return CalibrationRecord(GraspCandidate((0., 0., .2)), task_result, max_height, force, 2,
                             score_lift_evidence(task_result, "tool_marker"))


def test_continuous_target_held_lift_can_pass_without_claiming_docking():
    task = result([sample(index * .02) for index in range(1, 12)])
    evidence = score_lift_evidence(task, "tool_marker")
    assert evidence.accepted
    assert evidence.sustained_lift_seconds == pytest.approx(.2)
    assert evidence.sustained_height_change_m == pytest.approx(.06)
    assert record(task).as_dict()["executor_success"] is False


def test_107n_collision_is_penalized_and_cannot_win_over_safe_contact():
    colliding = result([sample(index * .02, force=107., collision=True) for index in range(1, 12)])
    safe_contact = result([sample(index * .02, height=0.) for index in range(1, 12)], phase="CLOSE")
    bad = record(colliding, max_height=.5, force=107.)
    good = record(safe_contact, max_height=0.)
    assert not bad.lift_evidence.accepted
    assert bad.score < 0 <= good.score
    assert "HAND_CHANNEL_OVERLOAD" in bad.lift_evidence.reasons
    assert "ENVIRONMENT_COLLISION" in bad.lift_evidence.reasons


def test_ballistic_one_frame_bounce_does_not_count_as_lift():
    rows = [sample(index * .02, height=.2 if index == 5 else 0.) for index in range(1, 12)]
    evidence = score_lift_evidence(result(rows), "tool_marker")
    assert not evidence.accepted
    assert evidence.sustained_lift_seconds == 0


def test_whole_episode_height_peak_is_diagnostic_only():
    task = result([sample(index * .02, height=0.) for index in range(1, 12)], phase="ALIGN")
    assert record(task, max_height=100.).score == record(task, max_height=0.).score


@pytest.mark.parametrize("kwargs", [{"support": 1.}, {"fingers": ("index",)}, {"force": .1}])
def test_supported_or_wrong_contact_cannot_pass(kwargs):
    task = result([sample(index * .02, **kwargs) for index in range(1, 12)])
    assert not score_lift_evidence(task, "tool_marker").accepted


def test_missing_evaluation_contact_channel_cannot_pass():
    rows = [sample(index * .02) for index in range(1, 12)]
    rows[2].pop(EVIDENCE_KEY)
    evidence = score_lift_evidence(result(rows), "tool_marker")
    assert not evidence.accepted
    assert "CONTACT_EVIDENCE_INCOMPLETE" in evidence.reasons


def test_relative_object_sliding_breaks_continuous_hold():
    rows = [sample(index * .02) for index in range(1, 12)]
    for index, row in enumerate(rows):
        row["evaluation"]["tool_marker"]["position"][0] = index * .02
    evidence = score_lift_evidence(result(rows), "tool_marker")
    assert not evidence.accepted


def test_lift_timeout_does_not_become_accepted_from_trace_alone():
    task = result([sample(index * .02) for index in range(1, 12)], status="TIMEOUT")
    assert not score_lift_evidence(task, "tool_marker").accepted


def test_lift_endpoint_must_still_be_holding_requested_target():
    rows = [sample(index * .02) for index in range(1, 12)]
    rows[-1][EVIDENCE_KEY]["target_hand_force_n"] = 0.
    evidence = score_lift_evidence(result(rows), "tool_marker")
    assert evidence.sustained_lift_seconds >= .1
    assert not evidence.accepted
    assert "LIFT_ENDPOINT_TARGET_NOT_HELD" in evidence.reasons


def test_missing_lift_has_no_positive_contact_force_bonus():
    task = result([sample(index * .02, height=0.) for index in range(1, 12)], phase="CLOSE")
    baseline = record(task)
    assert replace(baseline, max_tactile_force_n=107., max_contacts=100).score == baseline.score


def _calibration_script():
    spec = importlib.util.spec_from_file_location("v7_calibration_script", PROJECT / "scripts/calibrate_v7_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_budget_is_exact_and_mixed_profile_covers_top_down_and_wide_rotations():
    module = _calibration_script()
    generated = module.candidates(0, 128, "mixed")
    assert len(generated) == 128
    assert any(candidate.approach_axis_world == (0., 0., -1.) for candidate in generated)
    assert max(sum(value * value for value in candidate.wrist_rotation_rad) ** .5
               for candidate in generated) > 2.
    assert module.candidates(0, 128, "mixed") == generated


def test_rescore_rejects_tampered_candidate_trace(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "trace.json.gz").write_bytes(b"changed trace")
    (run / "calibration.json").write_text(json.dumps({
        "records": [{"trace_file": {"path": "trace.json.gz", "sha256": "wrong"}}],
    }))
    with pytest.raises(ValueError, match="trace hash mismatch"):
        _calibration_script().rescore_saved_sweep(run, tmp_path / "rescored")
