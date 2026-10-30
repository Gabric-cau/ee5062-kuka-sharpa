"""Declared R15 geometry differs from frozen R14 only through the input profile."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_fixed_geometry_r15 as r15


PROFILE = r15.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def test_input_manifest_matches_exact_frozen_r14_and_static_candidate():
    declaration = r15.validate_inputs(PROFILE)
    assert declaration["r14_source_sha256"] == r15.EXPECTED_R14_SHA256
    assert declaration["profile_sha256"] == r15.EXPECTED_PROFILE_SHA256
    assert declaration["changed_profile_fields"]["thumb_opposition_rad"] == {"before": 1.2, "after": 1.3}
    assert set(declaration["changed_profile_fields"]) <= r15.ALLOWED_PROFILE_CHANGES
    assert not declaration["controller_code_modified"]
    assert not declaration["physics_force_and_task_gates_modified"]
    assert not declaration["runtime_target_pose_or_qpos_added"]


def test_frozen_controller_code_and_every_behavior_global_are_reused():
    runner = r15.isolated_runner(PROFILE, r15.validate_inputs(PROFILE))
    assert runner.__code__ is r15.r14.run.__code__
    assert runner.__defaults__ is r15.r14.run.__defaults__
    assert runner.__closure__ is r15.r14.run.__closure__
    for name, value in r15.r14.run.__globals__.items():
        if name not in ("previous", "snapshot"):
            assert runner.__globals__[name] is value, name
    previous = r15.r14.run.__globals__["previous"]
    proxy = runner.__globals__["previous"]
    for name, value in vars(previous).items():
        if name != "atomic_json":
            assert getattr(proxy, name) is value, name
    assert previous.atomic_json is not proxy.atomic_json


def test_declared_force_physics_probe_and_hold_parameters_stay_frozen():
    declaration = r15.validate_inputs(PROFILE)
    assert declaration["preload_parameters"] == r15.r14.PRELOAD_PARAMETERS
    assert declaration["probe_parameters"] == r15.r14.PROBE_PARAMETERS
    assert declaration["joint_parameters"] == r15.r14.r13.JOINT_PARAMETERS
    assert declaration["tracking_parameters"] == r15.r14.r13.r11.TRACKING_PARAMETERS
    gates = declaration["strict_task_gates"]
    assert gates == {"channel_force_limit_n": 12., "dock_tolerance_m": .025,
        "stable_release_s": 3., "probe_displacement_m": .005,
        "current_fixed_normal_preload_gravity_margin": 1., "geometry_gravity_margin": 1.5}
    assert declaration["wrist_orientation_change_deg"] == {"roll": 0., "yaw": 0.}


@pytest.mark.parametrize("field, value", [("nominal_shallow_penetration_m", .001),
    ("thumb_mcp_flexion_ratio", .8), ("aperture_offset", [0., .025, .07])])
def test_other_profile_behavior_changes_are_rejected(field, value):
    before = json.loads(r15.BASE_PROFILE.read_text())["ycb_can"]
    after = copy.deepcopy(json.loads(PROFILE.read_text())["ycb_can"])
    after[field] = value
    with pytest.raises(ValueError, match="nominal grasp geometry fields"):
        r15.profile_changes(before, after)


def test_unrecorded_profile_or_parent_source_changes_fail_before_run(tmp_path, monkeypatch):
    changed = tmp_path / "changed.json"
    changed.write_bytes(PROFILE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="profile SHA256 mismatch"):
        r15.validate_inputs(changed)
    monkeypatch.setattr(r15, "EXPECTED_R14_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="R14 source SHA256 mismatch"):
        r15.validate_inputs(PROFILE)


def test_rotated_geometry_is_rejected_even_if_its_file_hash_is_declared(tmp_path, monkeypatch):
    document = json.loads(PROFILE.read_text())
    document["candidate"]["roll_deg"] = -6.
    changed = tmp_path / "rotated.json"
    changed.write_text(json.dumps(document))
    monkeypatch.setattr(r15, "EXPECTED_PROFILE_SHA256", r15.sha256(changed))
    with pytest.raises(ValueError, match="rejects rotated candidates"):
        r15.validate_inputs(changed)


def test_protocol_tagging_changes_labels_and_declaration_only():
    protocol = {"format": "r14", "nominal_profiles": {}, "schedule": [{"profile": "constrained_preload_r14"}],
        "force_control": "unchanged", "scoring": "unchanged", "phase_seconds": {"CLOSE": 16.}}
    original = copy.deepcopy(protocol)
    returned = r15.tagged_protocol(protocol, {"candidate": "test"})
    assert returned is protocol
    assert protocol["format"] == "public_ycb_fixed_geometry_development_r15"
    assert protocol["inherited_controller_protocol_format"] == "r14"
    assert protocol["schedule"][0]["profile"] == "fixed_geometry_r15"
    for key in ("force_control", "scoring", "phase_seconds"):
        assert protocol[key] == original[key]
    once = copy.deepcopy(protocol)
    r15.tagged_protocol(protocol, {"candidate": "test"})
    assert protocol == once


def test_source_snapshot_contains_candidate_parent_and_static_evidence():
    runner = r15.isolated_runner(PROFILE, r15.validate_inputs(PROFILE))
    snapshot = runner.__globals__["snapshot"](PROFILE)
    assert snapshot["scripts/develop_public_ycb_constrained_preload_r14.py"] == r15.EXPECTED_R14_SHA256
    assert snapshot[str(PROFILE.relative_to(r15.PROJECT))] == r15.EXPECTED_PROFILE_SHA256
    assert "scripts/develop_public_ycb_fixed_geometry_r15.py" in snapshot
    assert "tests/test_public_ycb_fixed_geometry_r15.py" in snapshot
    evidence = json.loads(PROFILE.read_text())["ycb_can"]["geometry_evidence"]
    assert snapshot[evidence] == r15.sha256(r15.PROJECT / evidence)
