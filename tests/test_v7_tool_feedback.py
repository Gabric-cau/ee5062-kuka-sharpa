"""Guard against false stable-release and missing-denominator evidence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_v7_tool_feedback import EVIDENCE_KEY, full_rate, release_evidence, task_spec_for
from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture


def release_result(task, config, *, seconds=3., last_force=0., wrong_object=False, missing=False):
    slot = next(slot for slot in config.slots if slot.slot_id == task.slot_id)
    rows = []
    for index in range(round(seconds / .01) + 1):
        row = {"time_s": 1.01 + index * .01,
               "evaluation": {task.object_id: {"position": slot.pose}},
               "finger_closure": {"index": 0.},
               EVIDENCE_KEY: {"target_linear_speed_m_s": 0., "target_angular_speed_rad_s": 0.,
                              "target_hand_force_n": 0.}}
        if wrong_object:
            row["evaluation"]["tool_marker"] = {"position": slot.pose}
            row["evaluation"][task.object_id] = {"position": (2., 2., 2.)}
        rows.append(row)
    rows[-1][EVIDENCE_KEY]["target_hand_force_n"] = last_force
    if missing:
        del rows[-1][EVIDENCE_KEY]
    phases = [SimpleNamespace(name="RELEASE", status="COMPLETE"),
              SimpleNamespace(name="VERIFY", status="COMPLETE", start_time_s=1.,
                              end_time_s=rows[-1]["time_s"])]
    return SimpleNamespace(phases=phases, trace=rows)


def test_stable_release_needs_complete_three_seconds_and_final_unload():
    config = upright_tool_fixture(0)
    task = task_spec_for("tool_wrench")
    assert release_evidence(release_result(task, config), task, config)["stable_released_in_slot"]
    assert not release_evidence(release_result(task, config, seconds=2.99), task, config)["stable_released_in_slot"]
    assert not release_evidence(release_result(task, config, last_force=.10), task, config)["stable_released_in_slot"]


def test_release_uses_requested_tool_and_missing_evidence_is_unknown():
    config = upright_tool_fixture(0)
    task = task_spec_for("tool_screwdriver")
    assert not release_evidence(release_result(task, config, wrong_object=True), task, config)["stable_released_in_slot"]
    assert release_evidence(release_result(task, config, missing=True), task, config)["stable_released_in_slot"] is None


def test_unknown_attempts_remain_in_all_attempts_denominator():
    measured = full_rate([True, False, None])
    assert measured["numerator"] == 1 and measured["denominator"] == 2
    assert measured["total"] == 3 and measured["unknown"] == 1
    assert measured["rate"] == .5 and measured["success_rate_all_attempts"] == 1 / 3


def test_feedback_modes_change_only_declared_tactile_guards_in_task_spec():
    for object_id in ("tool_marker", "tool_screwdriver", "tool_wrench"):
        task = task_spec_for(object_id)
        blind = replace(task, tactile_force_min_n=0., tactile_contacts_min=0)
        assert task.release_verify_seconds == blind.release_verify_seconds == 3.
        assert task.dock_tolerance_m == blind.dock_tolerance_m == .025
        assert task.phase_timeout_s == blind.phase_timeout_s == 12.
        assert task.grasp_offset_m == blind.grasp_offset_m
        assert task.grasp_closure == blind.grasp_closure
