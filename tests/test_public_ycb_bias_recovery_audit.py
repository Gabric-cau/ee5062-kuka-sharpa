"""Adjacent-native R16 audit contracts, including independently corrupted inputs."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_public_ycb_bias_recovery as audit


SPECS = [audit.JointSpec(f"left_{finger}_MCP_FE", finger, index, 0., 2., 1.)
         for index, finger in enumerate(audit.ACTIVE_FINGERS)]


def gate():
    return {"preload_ready": False, "preload_start_s": None, "probe_eligible_seconds": 0.,
        "probe_confirmed_seconds": 0., "probe_active": False, "payload_confirmed": False,
        "pressure_window_samples": 0}


def sample(index, command, actual):
    return {"timestamp": index * .002, "state": "SEARCH" if index == 1 else "CLOSE",
        "motor_commands": command,
        "actual_hand_joints": {spec.name: {"position_rad": actual[i], "velocity_rad_s": 0.}
                               for i, spec in enumerate(SPECS)}}


def fixture():
    native = [sample(1, [.2] * 3, [.18] * 3),
        sample(2, [.2001] * 3, [.2003, .18, .18]),
        sample(3, [.2003, .2001, .2001], [.2003, .18, .18]),
        sample(4, [.2004, .2002, .2002], [.2003, .18, .18])]
    capture = {"timestamp_s": .002, "event": "ACTIVATION_CAPTURE",
        "captured_command_rad": {s.name: .2 for s in SPECS},
        "captured_command_minus_actual_rad": {s.name: .2 - .18 for s in SPECS}}
    events, inherited = [], [capture]
    for index in range(4):
        fresh = index in (1, 3)
        mode = "SCALAR_ACQUISITION" if index == 0 else ("BIAS_RECOVERY" if index == 2 else "JOINT_CONTROL")
        event = {"timestamp_s": index * .002, "post_step_timestamp_s": (index + 1) * .002,
            "dt_s": 0. if index == 0 else .002, "phase": "SEARCH" if index == 0 else "CLOSE",
            "mode": mode, "fresh_target": fresh, "final_safety_override": False,
            "recovery_active": index == 2, "failure_latched": False, "preload_inhibited": index == 2,
            "write_verification_passed": True, "write_verification_violations": [], "abort_reason": None,
            "original_post_step_failure": None, "post_step_returned_failure": None,
            "probe_abort_applied": False, "resolved_abort_reason": None,
            "admission_before": gate(), "admission_after_write": gate(), "admission_after_step": gate(),
            "per_finger": []}
        if index:
            before, after = native[index - 1], native[index]
            for spec in SPECS:
                previous = before["motor_commands"][spec.actuator]
                actual = before["actual_hand_joints"][spec.name]["position_rad"]
                rate = [previous - .6 * .002, previous + .12 * .002]
                window = [actual + (.2 - .18) - .02, actual + (.2 - .18) + .02]
                applied = after["motor_commands"][spec.actuator]
                detail = {"before_actual_rad": actual, "before_command_rad": previous,
                    "previous_reference_rad": previous, "fixed_capture_bias_rad": .2 - .18,
                    "capture_timestamp_s": .002, "range_rad": [0., 2.],
                    "rate_interval_rad": rate, "bias_interval_rad": window,
                    "intersection_rad": [max(0., rate[0], window[0]), min(2., rate[1], window[1])],
                    "candidate_command_rad": applied, "applied_command_rad": applied}
                event["per_finger"].append({"finger": spec.finger,
                    "mode": "BIAS_RECOVERY" if index == 2 else "NOMINAL",
                    "joint_names": [spec.name], "joints": {spec.name: detail},
                    "compression_jacobian_m_rad": [[1.]] if fresh else None,
                    "desired_normal_velocity_m_s": [.001] if fresh else None,
                    "predicted_applied_normal_velocity_m_s": [(applied - previous) / .002] if fresh else None,
                    "nominal_direction_satisfied": True if fresh else None})
        inherited.append({"timestamp_s": index * .002, "event": "NOMINAL_REFERENCE_WRITE",
            "mode": mode, "phase": event["phase"],
            "current_target": {"feasible": fresh}, "target_age_s": 0. if index else None,
            "safety_override_at_final_write": False})
        events.append(event)
    return {"native": native, "events": events, "joint_events": inherited, "specs": SPECS,
        "close_rate": .12, "relief_rate": .6,
        "preload_events": [{"timestamp_s": .006, "ready": False, "continuous_seconds": 0.}],
        "probe_events": [{"timestamp_s": .006, "probe_active": False, "admission": False}],
        "native_audit": {"integration_steps": 4, "native_samples": 4,
            "initial_target_qpos": [0., 0., .38, 1., 0., 0., 0.],
            "external_target_qpos_writes": 0, "write_events": []}}


def test_nominal_recovery_and_invalid_target_hold_use_adjacent_native():
    receipt = audit.reconstruct_case(**fixture())
    assert receipt["accepted_evidence_consistency"]
    assert receipt["original_intersection_satisfied_for_all_audited_writes"]
    assert receipt["counts"]["nominal_joint_writes"] == 6
    assert receipt["counts"]["bias_recovery_joint_writes"] == 3
    assert receipt["counts"]["activation_captures"] == 1
    assert not receipt["target_qpos_guard"]["independently_reconstructed"]


@pytest.mark.parametrize("change", [
    lambda x: x["events"][2].update(timestamp_s=.006),
    lambda x: x["events"][2].update(post_step_timestamp_s=.008),
    lambda x: x["native"][1]["motor_commands"].__setitem__(0, .3),
    lambda x: x["events"][2]["per_finger"][0]["joints"][SPECS[0].name].update(fixed_capture_bias_rad=.04),
    lambda x: x["joint_events"][0]["captured_command_minus_actual_rad"].__setitem__(SPECS[0].name, .04),
    lambda x: x["events"][2]["admission_after_step"].update(preload_ready=True),
    lambda x: x["events"][2]["admission_after_write"].update(probe_active=True),
    lambda x: x["events"][2]["admission_after_step"].update(probe_eligible_seconds=.25),
    lambda x: x["preload_events"][0].update(ready=True),
    lambda x: x["probe_events"][0].update(admission=True),
    lambda x: x["events"][2].update(fresh_target=True),
    lambda x: x["events"][2].update(preload_inhibited=False),
    lambda x: x["events"][2]["per_finger"].pop(),
    lambda x: x["events"].pop(2),
    lambda x: x["native"].pop(1),
    lambda x: x["native_audit"].update(external_target_qpos_writes=1),
    lambda x: x["preload_events"].clear(),
])
def test_corrupt_or_missing_required_evidence_is_rejected(change):
    values = fixture()
    change(values)
    with pytest.raises((ValueError, KeyError)):
        audit.reconstruct_case(**values)


def test_recovery_cannot_recapture_bias_to_hide_a_moved_window():
    values = fixture()
    recapture = copy.deepcopy(values["joint_events"][0])
    recapture["timestamp_s"] = .004
    for spec in SPECS:
        current = values["native"][1]
        command = current["motor_commands"][spec.actuator]
        recapture["captured_command_rad"][spec.name] = command
        recapture["captured_command_minus_actual_rad"][spec.name] = command - current["actual_hand_joints"][spec.name]["position_rad"]
    values["joint_events"].append(recapture)
    with pytest.raises(ValueError, match="once in each CLOSE"):
        audit.reconstruct_case(**values)


def test_corrupt_native_and_diagnostics_together_still_violate_original_bias():
    values = fixture()
    values["native"][2]["motor_commands"][0] = .21
    detail = values["events"][2]["per_finger"][0]["joints"][SPECS[0].name]
    detail.update(applied_command_rad=.21, candidate_command_rad=.21)
    with pytest.raises(ValueError):
        audit.reconstruct_case(**values)


def test_feasible_recovery_must_use_minimum_change_projection():
    values = fixture()
    command = .20032
    values["native"][2]["motor_commands"][0] = command
    detail = values["events"][2]["per_finger"][0]["joints"][SPECS[0].name]
    detail.update(applied_command_rad=command, candidate_command_rad=command)
    with pytest.raises(ValueError, match="minimum-change projection"):
        audit.reconstruct_case(**values)


def test_future_encoder_cannot_replace_missing_contemporaneous_state():
    values = fixture()
    del values["native"][1]["actual_hand_joints"][SPECS[0].name]
    with pytest.raises(KeyError):
        audit.reconstruct_case(**values)


def test_exclusion_does_not_claim_biased_bounds_or_hide_a_tracking_hold():
    values = fixture()
    values["events"][2]["mode"] = "SCALAR_ACQUISITION"
    values["joint_events"][3]["mode"] = "SCALAR_ACQUISITION"
    values["events"][2]["per_finger"] = []
    with pytest.raises(ValueError, match="conceal a captured"):
        audit.reconstruct_case(**values)


def test_explicit_release_is_counted_and_not_claimed_as_bounded():
    values = fixture()
    values["native"][3]["state"] = "RELEASE"
    event = values["events"][3]
    event.update(phase="RELEASE", mode="RELEASING", fresh_target=False, per_finger=[])
    legacy = values["joint_events"][4]
    legacy.update(phase="RELEASE", mode="RELEASING", current_target={"feasible": False})
    receipt = audit.reconstruct_case(**values)
    assert receipt["counts"]["excluded_release_writes"] == 1
    assert "do not establish .02rad" in receipt["scope"]


def truncate(values, count):
    values["native"] = values["native"][:count]
    values["events"] = values["events"][:count]
    values["joint_events"] = [event for event in values["joint_events"]
        if event["timestamp_s"] < count * .002]
    values["native_audit"].update(integration_steps=count, native_samples=count)


def test_empty_box_is_a_truthful_failure_and_never_a_bound_certificate():
    values = fixture()
    truncate(values, 3)
    values["native"][1]["actual_hand_joints"][SPECS[0].name]["position_rad"] = .25
    values["native"][2]["motor_commands"] = [.2001] * 3
    event = values["events"][2]
    event.update(mode="BIAS_INFEASIBLE", recovery_active=False, failure_latched=True,
        write_verification_passed=False, abort_reason="BIAS_BOX_INFEASIBLE",
        post_step_returned_failure="BIAS_BOX_INFEASIBLE")
    values["joint_events"][3]["mode"] = "BIAS_INFEASIBLE"
    for i, finger in enumerate(event["per_finger"]):
        finger.update(mode="INFEASIBLE" if i == 0 else "FAIL_PENDING")
        detail = finger["joints"][SPECS[i].name]
        detail.update(candidate_command_rad=.2001, applied_command_rad=.2001)
        if i == 0:
            detail.update(before_actual_rad=.25, bias_interval_rad=[.25, .29],
                intersection_rad=[.25, .20034])
    receipt = audit.reconstruct_case(**values)
    assert receipt["accepted_evidence_consistency"]
    assert not receipt["original_intersection_satisfied_for_all_audited_writes"]
    assert len(receipt["explicit_infeasible_writes"]) == 1


def test_nonidentity_final_writer_is_a_recorded_failure_even_inside_the_box():
    values = fixture()
    values["native"][3]["motor_commands"][0] = .2005
    event = values["events"][3]
    event.update(mode="BIAS_INFEASIBLE", failure_latched=True, preload_inhibited=True,
        write_verification_passed=False, abort_reason="BIAS_FINAL_WRITE_VERIFICATION_FAILED",
        post_step_returned_failure="BIAS_FINAL_WRITE_VERIFICATION_FAILED",
        write_verification_violations=[SPECS[0].name + ":APPLIED_CANDIDATE_MISMATCH"])
    finger = event["per_finger"][0]
    finger["joints"][SPECS[0].name]["applied_command_rad"] = .2005
    finger["predicted_applied_normal_velocity_m_s"] = [(.2005 - .2003) / .002]
    values["joint_events"][4]["mode"] = "BIAS_INFEASIBLE"
    values["preload_events"].append({"timestamp_s": .008, "ready": False, "continuous_seconds": 0.})
    values["probe_events"].append({"timestamp_s": .008, "probe_active": False})
    receipt = audit.reconstruct_case(**values)
    assert receipt["accepted_evidence_consistency"]
    assert not receipt["original_intersection_satisfied_for_all_audited_writes"]
    assert receipt["failed_writes"][0]["final_write_violations"]


def test_original_emergency_branch_may_have_only_safety_precedence_event():
    values = fixture()
    values["events"][3].update(mode="EMERGENCY_HOLD", final_safety_override=True,
        per_finger=[], write_verification_passed=None,
        original_post_step_failure="ORIGINAL_PRESSURE_FAILURE",
        post_step_returned_failure="ORIGINAL_PRESSURE_FAILURE")
    values["joint_events"][4] = {"timestamp_s": .006, "event": "SAFETY_PRECEDENCE",
        "mode": "EMERGENCY_HOLD"}
    receipt = audit.reconstruct_case(**values)
    assert receipt["counts"]["excluded_emergency_writes"] == 1


def test_queued_existing_probe_waits_for_original_safety_then_aborts():
    values = fixture()
    truncate(values, 3)
    event = values["events"][2]
    event["admission_before"]["probe_active"] = True
    event["admission_after_write"]["probe_active"] = True
    event.update(abort_reason="PROBE_BIAS_RECOVERY", probe_abort_applied=True,
        resolved_abort_reason="PROBE_BIAS_RECOVERY", post_step_returned_failure="PROBE_BIAS_RECOVERY")
    values["probe_events"][0].update(probe_active=True, stop_reason=None)
    assert audit.reconstruct_case(**values)["accepted_evidence_consistency"]
    event["post_step_returned_failure"] = None
    with pytest.raises(ValueError, match="not actually aborted"):
        audit.reconstruct_case(**values)


def test_original_failure_has_priority_over_the_queued_bias_reason():
    values = fixture()
    values["events"][2].update(original_post_step_failure="ORIGINAL_FORCE_FAILURE",
        post_step_returned_failure="BIAS_RECOVERY")
    with pytest.raises(ValueError, match="replaced the original"):
        audit.reconstruct_case(**values)


def test_held_recovery_requires_an_actual_abort_and_its_resolved_reason():
    values = fixture()
    truncate(values, 3)
    event = values["events"][2]
    event["admission_before"]["payload_confirmed"] = True
    event["admission_after_write"]["payload_confirmed"] = True
    event.update(abort_reason="HELD_POLICY_BIAS_RECOVERY", probe_abort_applied=True,
        resolved_abort_reason="HELD_POLICY_BIAS_RECOVERY",
        post_step_returned_failure="HELD_POLICY_BIAS_RECOVERY")
    assert audit.reconstruct_case(**values)["accepted_evidence_consistency"]
    event.update(probe_abort_applied=False, resolved_abort_reason=None,
        post_step_returned_failure=None)
    with pytest.raises(ValueError, match="not actually aborted"):
        audit.reconstruct_case(**values)


@pytest.mark.parametrize("change", [
    lambda x: x["events"][2]["admission_before"].pop("payload_confirmed"),
    lambda x: x["events"][2]["admission_after_step"].pop("payload_confirmed"),
    lambda x: x["events"][2].update(resolved_abort_reason="FAKE_ABORT"),
    lambda x: x["preload_events"][0].update(continuous_seconds=-.25),
])
def test_missing_held_evidence_or_false_abort_and_reset_are_rejected(change):
    values = fixture()
    change(values)
    with pytest.raises((ValueError, KeyError)):
        audit.reconstruct_case(**values)


def test_open_hold_cannot_be_used_to_hide_a_loaded_close_epoch():
    values = fixture()
    values["events"][3].update(mode="OPEN_HOLD", per_finger=[])
    values["joint_events"][4]["mode"] = "OPEN_HOLD"
    with pytest.raises(ValueError, match="preceding release"):
        audit.reconstruct_case(**values)


def append_passthrough(values, phase, mode, commands):
    index = len(values["events"])
    state = sample(index + 1, commands, [.15] * 3)
    state["state"] = phase
    values["native"].append(state)
    event = copy.deepcopy(values["events"][0])
    event.update(timestamp_s=index * .002, post_step_timestamp_s=(index + 1) * .002,
        dt_s=.002, phase=phase, mode=mode, fresh_target=False)
    values["events"].append(event)
    values["joint_events"].append({"timestamp_s": index * .002, "event": "NOMINAL_REFERENCE_WRITE",
        "phase": phase, "mode": mode, "current_target": {"feasible": False},
        "target_age_s": None, "safety_override_at_final_write": False})
    values["native_audit"].update(integration_steps=index + 1, native_samples=index + 1)


def test_actual_zero_open_hold_after_release_can_continue_through_search():
    values = fixture()
    append_passthrough(values, "RECOVER", "OPEN_HOLD", [0.] * 3)
    append_passthrough(values, "SEARCH", "OPEN_HOLD", [0.] * 3)
    append_passthrough(values, "CLOSE", "SCALAR_RECONCILING", [.01] * 3)
    receipt = audit.reconstruct_case(**values)
    assert receipt["counts"]["excluded_zero_open_hold_writes"] == 1
    assert receipt["counts"]["excluded_scalar_acquisition_writes"] == 2
    values["native"][5]["motor_commands"][0] = .01
    with pytest.raises(ValueError, match="zero open-hold"):
        audit.reconstruct_case(**values)


def retry_reset_fixture():
    values = fixture()
    append_passthrough(values, "RECOVER", "OPEN_HOLD", [0.] * 3)
    append_passthrough(values, "SEARCH", "SCALAR_ACQUISITION", [0.] * 3)
    values["events"][-1]["dt_s"] = 0.
    values["joint_events"][-1].update(dt_s=0., current_target={"feasible": False,
        "reason": "NOT_EVALUATED"}, command_rad={}, candidate_command_rad={},
        applied_command_rad={}, joint_reference_steps={})
    return values


def test_frozen_retry_new_controller_resets_only_local_timer_and_capture_epoch():
    receipt = audit.reconstruct_case(**retry_reset_fixture())
    assert receipt["counts"]["controller_attempt_resets"] == 1
    assert receipt["native_timestep_s"] == .002


@pytest.mark.parametrize("change", [
    lambda x: x["events"][-1].update(phase="CLOSE"),
    lambda x: x["joint_events"][-1].update(command_rad={SPECS[0].name: 0.}),
    lambda x: x["joint_events"][-1].update(current_target={"feasible": False, "reason": "STALE"}),
    lambda x: x["native"][-1]["motor_commands"].__setitem__(0, .01),
    lambda x: x["native"][-2]["motor_commands"].__setitem__(0, .01),
])
def test_new_attempt_local_timer_reset_cannot_hide_tracking_or_nonzero_commands(change):
    values = retry_reset_fixture()
    change(values)
    with pytest.raises(ValueError):
        audit.reconstruct_case(**values)


@pytest.fixture
def directory_fixture(tmp_path, monkeypatch):
    values = fixture()
    directory, case = tmp_path / "run", tmp_path / "run/case"
    case.mkdir(parents=True)
    profile = {"ycb_can": {"contact_closure": {finger: 1. for finger in audit.ACTIVE_FINGERS}}}
    source = {
        "scripts/develop_public_ycb_constrained_preload_r14.py": b"# immutable test parent\n",
        "scripts/develop_public_ycb_joint_admittance_r13.py": b"JOINT_PARAMETERS = {'additional_bias_limit_rad': .02}\n",
        "scripts/develop_public_ycb_force_band.py": b"configuration = f(close_rate=.12, relief_rate=.6)\n",
        "config/profile.json": json.dumps(profile).encode(),
    }
    digest_bytes = lambda value: hashlib.sha256(value).hexdigest()
    inherited = {key: digest_bytes(value) for key, value in source.items()}
    prior = {"complete": True, "protocol": {"source_changed_during_run": False,
        "source_hashes_before": inherited, "source_hashes_after": inherited}}
    source[audit.R15_SUMMARY] = json.dumps(prior).encode()
    source[audit.R16_SOURCE] = b"RECOVERY_PARAMETERS = {'fixture': True}\n"
    monkeypatch.setattr(audit, "FROZEN_R14_SHA256", inherited["scripts/develop_public_ycb_constrained_preload_r14.py"])
    monkeypatch.setattr(audit, "FROZEN_PROFILE_SHA256", inherited["config/profile.json"])
    monkeypatch.setattr(audit, "FROZEN_R15_SUMMARY_SHA256", digest_bytes(source[audit.R15_SUMMARY]))
    hashes = {key: digest_bytes(value) for key, value in source.items()}
    with zipfile.ZipFile(directory / "frozen_source.zip", "x") as archive:
        for key, value in source.items():
            archive.writestr(key, value)
    protocol = {"format": "public_ycb_bias_recovery_development_r16",
        "source_hashes_before": hashes, "source_hashes_after": hashes,
        "source_changed_during_run": False, "joint_admittance": {"additional_bias_limit_rad": .02},
        "r16_bias_recovery_experiment": {"recovery_parameters": {"fixture": True},
            "r15_comparison_summary_sha256": digest_bytes(source[audit.R15_SUMMARY])},
        "nominal_profiles": profile,
        "schedule": [{"kind": "ycb_can", "seed": 3414, "profile": "bias_recovery_r16"}]}
    scene = case / "scene.xml"
    scene.write_text("<mujoco><compiler/></mujoco>")
    row = {**protocol["schedule"][0], "path": "case", "success": False, "held_lift": False,
        "runner_error": None, "native_audit": values["native_audit"],
        "scene_sha256_before_load": audit.digest(scene), "scene_sha256_at_finalize": audit.digest(scene),
        "scene_unchanged_since_load": True, "artifacts": {}}
    streams = {"native": values["native"], "bias_recovery": values["events"],
        "joint_admittance": values["joint_events"], "preload": values["preload_events"],
        "load_transfer_probe": values["probe_events"]}
    for key, records in streams.items():
        path = case / f"{key}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
        row["artifacts"][key] = {"path": path.name, "sha256": audit.digest(path), "bytes": path.stat().st_size}
    (case / "metadata.json").write_text(json.dumps(row))
    summary = {"complete": True, "protocol": protocol, "rows": [row]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(protocol))
    import public_ycb_evidence
    monkeypatch.setattr(public_ycb_evidence, "load_evidence_model", lambda _: SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
    monkeypatch.setattr(audit, "joint_specs", lambda *_: SPECS)
    return directory


def test_loader_hashes_archived_sources_scene_and_raw_artifacts(directory_fixture):
    receipt = audit.audit_directory(directory_fixture)
    assert receipt["all_evidence_consistent"]
    assert receipt["all_command_bounds_satisfied"]
    assert receipt["frozen_source"]["all_frozen_source_hashes_match"]
    assert not receipt["target_qpos_writes_independently_proven"]


def rewrite_protocol(directory, change):
    summary = json.loads((directory / "summary.json").read_text())
    change(summary["protocol"])
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(summary["protocol"]))


def test_protocol_cannot_change_original_bias_limit(directory_fixture):
    rewrite_protocol(directory_fixture, lambda p: p["joint_admittance"].update(additional_bias_limit_rad=.2))
    with pytest.raises(ValueError, match="original .02rad"):
        audit.audit_directory(directory_fixture)


def test_changing_raw_bytes_without_the_bound_hash_is_rejected(directory_fixture):
    path = directory_fixture / "case/native.json.gz"
    path.write_bytes(path.read_bytes() + b"extra")
    with pytest.raises(ValueError, match="size/hash"):
        audit.audit_directory(directory_fixture)


def test_runtime_scene_binding_rejects_posthoc_xml_changes(directory_fixture):
    path = directory_fixture / "case/scene.xml"
    path.write_text("<mujoco><compiler/><option timestep='.02'/></mujoco>")
    with pytest.raises(ValueError, match="runtime-loaded joint model"):
        audit.audit_directory(directory_fixture)


def test_inherited_rate_source_cannot_be_relabelled_as_the_frozen_parent(directory_fixture):
    path = directory_fixture / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist()}
    changed = "scripts/develop_public_ycb_force_band.py"
    sources[changed] = b"configuration = f(close_rate=12., relief_rate=60.)\n"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in sources.items():
            archive.writestr(key, value)
    changed_hash = hashlib.sha256(sources[changed]).hexdigest()
    rewrite_protocol(directory_fixture, lambda p: [p[key].update({changed: changed_hash})
        for key in ("source_hashes_before", "source_hashes_after")])
    with pytest.raises(ValueError, match="inherited source"):
        audit.audit_directory(directory_fixture)


def test_artifact_path_cannot_escape_case_root(directory_fixture):
    summary = json.loads((directory_fixture / "summary.json").read_text())
    summary["rows"][0]["artifacts"]["native"]["path"] = "../native.json.gz"
    (directory_fixture / "summary.json").write_text(json.dumps(summary))
    (directory_fixture / "case/metadata.json").write_text(json.dumps(summary["rows"][0]))
    with pytest.raises(ValueError, match="Unsafe evidence-relative"):
        audit.audit_directory(directory_fixture)
