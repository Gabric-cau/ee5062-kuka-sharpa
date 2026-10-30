"""Focused source portability, changed samples and same-write linkage fixtures."""

import copy
import gzip
import json
from pathlib import Path

import pytest

import audit_public_ycb_fresh_contact_r24 as audit


def fixture():
    patches = []
    for index, finger in enumerate(audit.ACTIVE):
        patches.append({"finger": finger, "input_index": index, "raw_input_index": index,
            "hand_body_name": finger + "_tip", "position_world_m": [.01 * index, 0., .4],
            "normal_on_object_world": [1., 0., 0.], "measured_pressure_n": 1., "target_normal_n": 1.1})
    target = {"feasible": True, "runtime_target_state_used": False, "timestamp_s": 1., "patches": patches,
        "association": {"accepted": True, "patches": copy.deepcopy(patches),
            "contact_association": [{"pressure_n": 1.} for _ in patches]}}
    groups = audit.current_groups(patches)
    solves, telemetry_solves, per_finger = [], [], []
    capture = {"event": "ACTIVATION_CAPTURE", "timestamp_s": 1.,
        "captured_command_minus_actual_rad": {name: .01 for names in audit.NAMES.values() for name in names}}
    for finger in audit.ACTIVE:
        names = audit.NAMES[finger]
        matrix, desired = [[.02] * len(names)], [.00005]
        solves.append({"finger": finger, "joint_names": names, "groups": [g for g in groups if g["finger"] == finger],
            "compression_jacobian_m_rad": matrix, "pressure_errors_n": [.1], "desired_normal_velocity_m_s": desired})
        telemetry_solves.append({"finger": finger, "joint_names": names,
            "raw_input_indices_by_group": [[audit.ACTIVE.index(finger)]], "jacobian_shape": [1, len(names)]})
        joints = {name: {"fixed_capture_bias_rad": .01, "capture_timestamp_s": 1., "additional_bias_limit_rad": .10,
            "previous_reference_rad": .1, "intersection_rad": [.0998, .1002]} for name in names}
        qp = {"accepted": True, "compression_jacobian_m_rad": matrix, "desired_normal_velocity_m_s": desired,
            "lower_velocity_rad_s": [-.1] * len(names), "upper_velocity_rad_s": [.1] * len(names)}
        per_finger.append({"finger": finger, "joint_names": names, "joints": joints, "nominal_qp": qp,
            "compression_jacobian_m_rad": matrix, "desired_normal_velocity_m_s": desired,
            "mode": "NOMINAL", "nominal_direction_satisfied": True})
    telemetry = {"format": "r24_current_contact_tracking_v1", "timestamp_s": 1., "changed_from_previous_sample": False,
        "activated_before_current_solve": False, "current_contact_sample_count": 3,
        "current_target_raw_input_indices": [0, 1, 2], "status": "CURRENT_CONTACT_JACOBIANS_REBUILT",
        "rebuilt_group_count": 3, "rebuilt_finger_solves": telemetry_solves}
    target["fresh_contact_tracking"] = telemetry
    event = {"event": "NOMINAL_REFERENCE_WRITE", "timestamp_s": 1., "phase": "CLOSE", "dt_s": .002,
        "target_age_s": 0., "current_target": target, "joint_solves": solves}
    bias = {"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
        "timestamp_s": 1., "phase": "CLOSE", "dt_s": .002, "fresh_target": True,
        "additional_bias_limit_rad": .10, "per_finger": per_finger, "final_safety_override": False,
        "recovery_active": False, "preload_inhibited": False, "abort_reason": None, "admission_before": {"probe_active": True}}
    return event, bias, capture


def test_actual_python310_ast_hashes_reconstruct_portably_and_wrapper_is_exact():
    hook = (audit.PROJECT / audit.HOOK).read_text()
    original = (audit.PROJECT / audit.R13).read_text()
    result = audit.source_contract(hook, original, [3, 10, 12])
    assert result["original_method_ast_sha256"] == result["restored_method_ast_sha256"] == "8132f93f28c03b23354f333b9b713bc7c986d2ed555857eb5867053faf43d862"
    assert result["transformed_method_ast_sha256"] == "511e18238558bc37862e6b3253a92d07a2ebe917bbae78b4b764f37d8e5a7798"
    assert result["wrapper_method_source_sha256"] == "9118c0efe815a41e2bf80f995d8e9df134c3b7c572e97bbc092b2f1cd2b291a3"
    with pytest.raises(ValueError, match="Wrapper does more"):
        audit.source_contract(hook.replace("return _fresh_regulate(self, readings, dt, carrying)", "return None"), original, [3, 10, 12])


def test_current_group_and_qp_linkage_accept_matching_real_shapes():
    event, bias, capture = fixture()
    sample = audit.check_reconstruction(event)
    assert len(sample) == 3
    assert audit.check_qp_linkage(event, bias, capture) == {"current": True, "qp_fingers": 3, "rejected_qp_fingers": 0}


@pytest.mark.parametrize("mutation,match", [("raw", "raw indices"), ("group", "identity/index"),
    ("shape", "Jacobian dimensions"), ("desired", "desired normal"), ("metadata", "metadata differs"), ("age", "write age")])
def test_changed_current_observation_cannot_use_old_indices_dimensions_or_telemetry(mutation, match):
    event, _, _ = fixture()
    if mutation == "raw":
        event["current_target"]["patches"][0]["raw_input_index"] = 9
    elif mutation == "group":
        event["joint_solves"][0]["groups"][0]["raw_input_indices"] = [2]
    elif mutation == "shape":
        event["joint_solves"][0]["compression_jacobian_m_rad"] = [[.02]]
    elif mutation == "desired":
        event["joint_solves"][0]["desired_normal_velocity_m_s"] = [.001]
    elif mutation == "metadata":
        event["current_target"]["fresh_contact_tracking"]["rebuilt_finger_solves"][0]["jacobian_shape"] = [2, 2]
    else:
        event["target_age_s"] = .002
    with pytest.raises(ValueError, match=match):
        audit.check_reconstruction(event)


def test_original_matcher_detects_count_and_same_count_pressure_changes():
    event, _, _ = fixture()
    current = audit.check_reconstruction(event)
    previous = copy.deepcopy(current)
    previous[0]["pressure_n"] = .1
    assert audit.sample_change(current, previous) == (True, ["pressure"])
    assert audit.sample_change(current, previous[:-1]) == (True, ["count"])
    assert audit.sample_change(current, current) == (False, [])


def test_stale_retained_target_never_certifies_a_current_qp():
    event, bias, capture = fixture()
    event["timestamp_s"], event["target_age_s"] = 1.004, .004
    bias["fresh_target"] = False
    with pytest.raises(ValueError, match="Stale retained reconstruction reached"):
        audit.check_qp_linkage(event, bias, capture)
    bias["per_finger"] = []
    event["phase"] = "RELEASE"
    assert audit.check_qp_linkage(event, bias, capture)["current"] is False


def test_actual_rejected_qp_requires_recovery_inhibition_and_active_probe_abort():
    event, bias, capture = fixture()
    finger = bias["per_finger"][0]
    finger["nominal_qp"]["accepted"] = False
    finger["mode"] = "BIAS_RECOVERY"
    with pytest.raises(ValueError, match="did not retain recovery"):
        audit.check_qp_linkage(event, bias, capture)
    bias.update(recovery_active=True, preload_inhibited=True, abort_reason="PROBE_BIAS_RECOVERY")
    assert audit.check_qp_linkage(event, bias, capture)["rejected_qp_fingers"] == 1
    bias["abort_reason"] = None
    with pytest.raises(ValueError, match="did not retain abort"):
        audit.check_qp_linkage(event, bias, capture)


def test_loaded_capture_and_current_box_or_jacobian_cannot_be_replaced():
    event, bias, capture = fixture()
    bias["per_finger"][0]["joints"][audit.NAMES["thumb"][0]]["fixed_capture_bias_rad"] = 0.
    with pytest.raises(ValueError, match="Fixed loaded bias"):
        audit.check_qp_linkage(event, bias, capture)
    event, bias, capture = fixture()
    bias["per_finger"][0]["nominal_qp"]["compression_jacobian_m_rad"] = [[9., 9.]]
    with pytest.raises(ValueError, match="Jacobian/QP linkage"):
        audit.check_qp_linkage(event, bias, capture)


def test_stream_counts_later_retention_separately_and_requires_raw_writer_records(tmp_path):
    event, bias, capture = fixture()
    later, later_bias = copy.deepcopy(event), copy.deepcopy(bias)
    later.update(timestamp_s=1.004, target_age_s=.004, phase="RELEASE")
    later_bias.update(timestamp_s=1.004, phase="RELEASE", fresh_target=False, per_finger=[])
    hands = [{"events": [event["current_target"]["fresh_contact_tracking"], later["current_target"]["fresh_contact_tracking"]],
        "accepted_reconstruction_count": 2, "changed_current_sample_count": 0}]
    paths = [tmp_path / "joint.json.gz", tmp_path / "bias.json.gz"]
    for path, records in zip(paths, ([capture, event, later], [bias, later_bias])):
        with gzip.open(path, "wt") as stream:
            json.dump(records, stream)
    report = audit.audit_streams(*paths, hands)[0]
    assert report["retained_records"] == 2 and report["current_solve_records"] == 1
    assert report["retained_without_current_qp_records"] == 1 and report["rejected_qp_fingers"] == 0
    hands[0]["events"][0] = dict(hands[0]["events"][0], changed_from_previous_sample=True)
    with pytest.raises(ValueError, match="serialized hand records"):
        audit.audit_streams(*paths, hands)


def test_explicit_final_summary_hash_is_required_before_loading(tmp_path):
    with pytest.raises(ValueError, match="Explicit final"):
        audit.audit_directory(tmp_path, "not-a-final-hash")


def test_result_schema_retains_summary_only_r19_arm_metadata_without_requiring_it_in_result():
    common = {"fresh_contact_tracking": {"hands": []}, "attempts": [], "success": False,
        "actual_wrist_probe_source_transform": {"format": "actual_wrist_probe_ast_extension_v1"},
        "alignment_source_transform": {"format": "settled_open_alignment_ast_extension_v1"}}
    row = dict(common, arm_source_transform={"format": "r19_prefix_only"}, held_lift=False)
    result = copy.deepcopy(common)
    audit.check_result_binding(row, result)
    assert "arm_source_transform" not in result and "held_lift" not in result
    del result["actual_wrist_probe_source_transform"]
    with pytest.raises(ValueError, match="lacks field: actual_wrist_probe_source_transform"):
        audit.check_result_binding(row, result)
