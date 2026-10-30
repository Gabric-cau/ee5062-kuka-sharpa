"""Failure-oriented checks for independent current V7 evidence auditing."""

from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from audit_current_v7_evidence import (Geometry, angle, lift_metrics, relative_pose, release_metrics,
                                      source_audit, time_audit, verify_artifacts)


def row(time, *, height=.10, force=1., support=0., phase="LIFT"):
    return {"time_s": time, "phase": phase, "wrist_position": [0., 0., 0.],
            "wrist_quaternion": [1., 0., 0., 0.], "arm_joint_position": [0.]*7,
            "hand_joint_position": {"left_index_PIP": 0.}, "tactile": {},
            "finger_closure": {"index": 0.},
            "evaluation": {"tool": {"position": [0., 0., height], "quaternion": [1., 0., 0., 0.]}},
            "calibration_evaluation": {"target_hand_force_n": force, "target_hand_contacts": 2,
                "target_contact_fingers": ["thumb", "index"], "target_support_force_n": support,
                "environment_collisions": [], "maximum_hand_channel_force_n": force,
                "target_linear_speed_m_s": 0., "target_angular_speed_rad_s": 0.}}


def test_time_requires_exact_2ms_and_rejects_missing_samples():
    assert time_audit([row(0), row(.002), row(.004)])["accepted"]
    assert not time_audit([row(0), row(.002), row(.006)])["accepted"]
    assert not time_audit([row(0), row(.002), row(.0041)])["accepted"]


def test_only_identical_physics_phase_marker_can_duplicate_time():
    before = row(.002, phase="SUPPORT_SETTLE")
    after = deepcopy(before)
    after["phase"] = "INDEX_PUSH"
    audit = time_audit([row(0), before, after, row(.004)], allow_boundary_duplicates=True)
    assert audit["accepted"] and audit["boundary_duplicate_indices"] == [2]
    after["evaluation"]["tool"]["position"][0] = .01
    assert not time_audit([row(0), before, after, row(.004)], allow_boundary_duplicates=True)["accepted"]


def test_relative_pose_rejects_apparent_motion_from_wrist_and_object_together():
    before = row(0)
    after = deepcopy(before)
    after["wrist_position"] = [1., 0., 0.]
    after["evaluation"]["tool"]["position"] = [1., 0., .1]
    assert np.allclose(relative_pose(before, "tool")[0], relative_pose(after, "tool")[0])
    assert angle([1., 0., 0., 0.], [-1., 0., 0., 0.]) == 0
    assert angle([1., 0., 0., 0.], [math.cos(.2), 0., 0., math.sin(.2)]) == pytest.approx(.4)


def test_held_lift_needs_continuous_contacts_no_support_and_completed_phase():
    rows = [row(i*.002) for i in range(1, 62)]
    phases = [{"name": "LIFT", "start_time_s": 0., "end_time_s": .122, "status": "COMPLETE"}]
    assert lift_metrics(rows, phases, "tool", 0.)["accepted"]
    broken = deepcopy(rows)
    broken[20]["calibration_evaluation"]["target_support_force_n"] = 1.
    assert not lift_metrics(broken, phases, "tool", 0.)["accepted"]
    phases[0]["status"] = "FAILED"
    assert not lift_metrics(rows, phases, "tool", 0.)["accepted"]


def test_release_restarts_window_on_contact_and_missing_velocity_fails_closed():
    rows = [row(i*.002, force=0., phase="VERIFY") for i in range(1, 152)]
    phases = [{"name": "RELEASE", "start_time_s": -1., "end_time_s": 0., "status": "COMPLETE"},
              {"name": "VERIFY", "start_time_s": 0., "end_time_s": .302, "status": "COMPLETE"}]
    task = {"object_id": "tool", "dock_tolerance_m": .025, "release_verify_seconds": .3}
    assert release_metrics(rows, phases, task, [0., 0., .1])["accepted"]
    rows[100]["calibration_evaluation"]["target_hand_force_n"] = 1.
    assert not release_metrics(rows, phases, task, [0., 0., .1])["accepted"]
    del rows[20]["calibration_evaluation"]["target_linear_speed_m_s"]
    with pytest.raises(KeyError):
        release_metrics(rows, phases, task, [0., 0., .1])


def test_hash_checks_fail_closed_for_missing_or_changed_artifact(tmp_path):
    assert not verify_artifacts(tmp_path, {"trace": {"path": "trace.gz", "sha256": "0"*64, "bytes": 10}})["accepted"]
    assert not source_audit({"missing/not_a_source.py": "0"*64})["accepted"]
    assert not source_audit({})["accepted"]


def test_native_hand_self_is_counted_but_never_inflates_target_force():
    geometry = Geometry.__new__(Geometry)
    geometry.names = {name: i for i, name in enumerate(("thumb", "index", "v7_geom_tool", "guard", "arm"))}
    geometry.hand = {"thumb": "thumb", "index": "index"}
    geometry.robot = {"thumb", "index", "arm"}
    def contact(a, b, force, distance=-.001):
        return {"geom1": a, "geom2": b, "distance_m": distance,
                "force_local_n_nm": [force, 0., 0., 0., 0., 0.]}
    result = geometry.contacts({"contacts": [contact("thumb", "index", 100.),
        contact("thumb", "v7_geom_tool", 2.), contact("index", "v7_geom_tool", 3.),
        contact("arm", "guard", 1., -.003)]}, "tool")
    assert result["target_hand_force_n"] == 5.
    assert result["maximum_hand_channel_force_n"] == 3.
    assert result["target_contact_fingers"] == ["index", "thumb"]
    assert result["self_contacts"] == [(100., .001)]
    assert result["non_hand_self"] == 0
    assert len(result["environment_collisions"]) == 1
    assert result["transfer_violations"] == 1


def test_pose_derived_release_does_not_shorten_required_window():
    rows = [row(i*.002, force=0., phase="VERIFY") for i in range(152)]
    phases = [{"name": "RELEASE", "start_time_s": -1., "end_time_s": 0., "status": "COMPLETE"},
              {"name": "VERIFY", "start_time_s": 0., "end_time_s": .302, "status": "COMPLETE"}]
    task = {"object_id": "tool", "dock_tolerance_m": .025, "release_verify_seconds": .3}
    assert release_metrics(rows, phases, task, [0., 0., .1], finite_difference=True)["accepted"]
    assert not release_metrics(rows[1:], phases, task, [0., 0., .1], finite_difference=True)["accepted"]
