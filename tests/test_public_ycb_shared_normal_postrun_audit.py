"""Strict initializer classification and isolated postrun target-audit checks."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_public_ycb_shared_normal_postrun as postrun
import shared_normal_tracking_targets as shared


def initializer(timestamp=0., phase="SEARCH", dt=0.):
    return {"timestamp_s": timestamp, "event": "NOMINAL_REFERENCE_WRITE", "phase": phase,
        "mode": "SCALAR_ACQUISITION", "dt_s": dt,
        "current_target": {"feasible": False, "reason": "NOT_EVALUATED"}, "target_age_s": None,
        "joint_solves": [], "joint_reference_steps": {}, "bounded_motion_predictions": [],
        "candidate_command_rad": {}, "applied_command_rad": {}, "command_rad": {},
        "safety_override_at_final_write": False}


def prefix():
    return [initializer(), initializer(.002, "APPROACH", .002), initializer(.004, "ALIGN", .002),
        {"timestamp_s": .006, "event": "NOMINAL_REFERENCE_WRITE", "phase": "CLOSE",
            "mode": "JOINT_CONTROL", "dt_s": .002}]


@pytest.fixture
def target():
    definitions = [("thumb", [-.03, 0., 0.], [1., 0., 0.], 2.),
        ("middle", [.03, 0., -.02], [-1., 0., 0.], 1.),
        ("ring", [.03, 0., .02], [-1., 0., 0.], 1.)]
    patches = [dict(finger=finger, input_index=index, hand_body_name=finger + "_tip",
        position_world_m=point, normal_on_object_world=normal, measured_pressure_n=pressure)
        for index, (finger, point, normal, pressure) in enumerate(definitions)]
    result = shared.allocate_tracking_targets(patches, [0., 0., 0.], .15,
        {"thumb": 2., "middle": 1., "ring": 1.})
    assert result["feasible"]
    result["association"] = {"accepted": True, "estimated_com_world_m": [0., 0., 0.], "patches": patches}
    return result


def streams(target):
    joint, preload = prefix()[:-1], []
    joint.append({"timestamp_s": .006, "event": "MODE_TRANSITION", "from_mode": "SCALAR_ACQUISITION",
        "to_mode": "JOINT_CONTROL", "reason": "synthetic fixture"})
    for index in range(15):
        timestamp = .006 + index * .002
        joint.append({"timestamp_s": timestamp, "event": "NOMINAL_REFERENCE_WRITE", "phase": "CLOSE",
            "mode": "JOINT_CONTROL", "dt_s": .002, "current_target": copy.deepcopy(target)})
        preload.append({"timestamp_s": timestamp + .002, "current_target_lp": copy.deepcopy(target),
            "original_probe_snapshot": {"catalogue_weight_n": .15 * 9.81,
                "sampled_public_mesh_association": copy.deepcopy(target["association"])}})
    return joint, preload


def test_strict_complete_initializer_prefix_has_original_ordinals():
    events = prefix()
    before = copy.deepcopy(events)
    result = postrun.classify_initialization_records(events)
    assert events == before and result["record_indices"] == {0, 1, 2}
    summary = result["summary"]
    assert summary["initialization_records"] == 3 and summary["controller_epochs"] == 1
    assert summary["blocks"][0]["first_close_timestamp_s"] == .006
    assert summary["all_contexts_valid"] and summary["original_stream_ordinals_preserved"]


@pytest.mark.parametrize("change", [
    lambda r: r["current_target"].__setitem__("feasible", True),
    lambda r: r["current_target"].__setitem__("feasible", 0),
    lambda r: r["current_target"].__setitem__("format", "shared_normal_tracking_target_v1"),
    lambda r: r["current_target"].__setitem__("patches", []),
    lambda r: r.__setitem__("event", "ACTIVATION_CAPTURE"),
    lambda r: r.__setitem__("phase", "CLOSE"),
    lambda r: r.__setitem__("mode", "BIAS_RECOVERY"),
    lambda r: r.__setitem__("target_age_s", 0.),
    lambda r: r.__setitem__("joint_solves", [{"accepted": True}]),
    lambda r: r.__setitem__("joint_reference_steps", {"left_thumb_IP": {}}),
    lambda r: r.__setitem__("bounded_motion_predictions", [{}]),
    lambda r: r.__setitem__("candidate_command_rad", {"left_thumb_IP": 0.}),
    lambda r: r.__setitem__("applied_command_rad", {"left_thumb_IP": 0.}),
    lambda r: r.__setitem__("command_rad", {"left_thumb_IP": 0.}),
    lambda r: r.__setitem__("safety_override_at_final_write", True),
])
def test_altered_initializer_is_never_a_blanket_gate_exception(change):
    events = prefix()
    change(events[1])
    with pytest.raises((ValueError, KeyError)):
        postrun.classify_initialization_records(events)


@pytest.mark.parametrize("change", [
    lambda e: e[0].__setitem__("timestamp_s", .002),
    lambda e: e[0].__setitem__("dt_s", .002),
    lambda e: e[1].__setitem__("dt_s", 0.),
    lambda e: e[2].__setitem__("phase", "SEARCH"),
    lambda e: e[2].__setitem__("timestamp_s", .006),
    lambda e: e[-1].__setitem__("timestamp_s", .008),
    lambda e: e.pop(),
    lambda e: e.append(initializer(.008, "ALIGN", .002)),
])
def test_epoch_phase_interval_and_after_close_reappearance_fail_closed(change):
    events = prefix()
    change(events)
    with pytest.raises(ValueError):
        postrun.classify_initialization_records(events)


def test_new_hand_requires_explicit_post_recover_zero_local_dt():
    events = prefix()
    events.extend([dict(events[-1], timestamp_s=.008, phase="RECOVER", mode="RELEASING"),
        initializer(.010), initializer(.012, "APPROACH", .002), initializer(.014, "ALIGN", .002),
        dict(events[-1], timestamp_s=.016)])
    result = postrun.classify_initialization_records(events)
    assert result["summary"]["controller_epochs"] == 2
    assert result["summary"]["initialization_records"] == 6
    events[4]["phase"] = "RELEASE"
    with pytest.raises(ValueError, match="post-RECOVER"):
        postrun.classify_initialization_records(events)


def test_ast_extension_preserves_sample_ordinals_inputs_and_all_original_globals(target):
    joint, preload = streams(target)
    before = copy.deepcopy((joint, preload))
    globals_before = dict(postrun.frozen.__dict__)
    original_function = postrun.frozen.audit_targets
    extended, witness = postrun.extended_target_audit()
    result = extended(joint, preload, .15)
    assert (joint, preload) == before and postrun.frozen.audit_targets is original_function
    assert all(postrun.frozen.__dict__[key] is value for key, value in globals_before.items())
    assert witness["branch_insertions"] == 1 and witness["input_records_filtered_or_reordered"] is False
    tracking = result["full_stream_record_checks"]["tracking"]
    assert tracking["records_with_target"] == 18 and tracking["initialization_records"] == 3
    assert tracking["feasible_targets"] == 15
    assert result["independent_sampled_target_count"] == 12
    samples = result["independent_sampled_checks"]
    assert [sample["stream_record_index"] for sample in samples[:6]] == [4, 6, 9, 12, 15, 18]
    assert all(set(sample["fixed_normal_checks"]) == {"1.0", "1.5"} for sample in samples)


@pytest.mark.parametrize("change", [
    lambda j, p: j[4].__setitem__("current_target", {"feasible": False, "reason": "UNKNOWN_NEW_GATE"}),
    lambda j, p: j[4]["current_target"].pop("format"),
    lambda j, p: j[4]["current_target"]["patches"][0].__setitem__("target_normal_n", 6.1),
    lambda j, p: p[0].__setitem__("current_target_lp", {"feasible": False, "reason": "NOT_EVALUATED"}),
])
def test_original_unknown_version_and_target_math_failures_are_not_masked(target, change):
    joint, preload = streams(target)
    change(joint, preload)
    extended, _ = postrun.extended_target_audit()
    with pytest.raises((ValueError, KeyError)):
        extended(joint, preload, .15)


def test_original_independent_hypothetical_failure_is_not_masked(target, monkeypatch):
    joint, preload = streams(target)
    extended, _ = postrun.extended_target_audit()
    monkeypatch.setattr(postrun.frozen.allocator, "allocate_contact_wrench", lambda *a, **k: {
        "measured_feasible": False, "measured_solver_status": 2})
    with pytest.raises(ValueError, match="Independent sampled"):
        extended(joint, preload, .15)


def test_original_frozen_source_hash_is_required(monkeypatch):
    monkeypatch.setattr(postrun, "FROZEN_AUDITOR_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen R20 auditor changed"):
        postrun.extended_target_audit()


def test_original_failure_receipt_is_exact_and_remains_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "failed.json"
    contents = {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
        "error": "Unversioned target is not an original caller-gate rejection",
        "script_sha256": postrun.FROZEN_AUDITOR_SHA256}
    path.write_text(json.dumps(contents))
    before = path.read_bytes()
    monkeypatch.setattr(postrun, "ORIGINAL_FAILED_RECEIPT_SHA256", postrun.frozen.base.digest(path))
    assert postrun.checked_failure(path)["sha256"] == postrun.ORIGINAL_FAILED_RECEIPT_SHA256
    assert path.read_bytes() == before
    contents["accepted_evidence"] = True
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="failure receipt differs"):
        postrun.checked_failure(path)
