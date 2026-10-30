"""Finite reference, source and timing fixtures; no integration or actual R23 audit."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import pytest

import actual_wrist_probe_controller_hooks as hook
import audit_public_ycb_actual_wrist_probe as audit


START = np.array([.6, 0., .7])
PROJECT = Path(__file__).resolve().parents[1]
OPEN = {"left_thumb_CMC_FE": 1.3, **{f"left_test_{i}": 0. for i in range(21)}}


def fixture(*, timeout=False, reset=False, motion_failure=False):
    start, events, rows, iks, caches, confirmed = 1., [], [], [], {}, 0.
    def position(n):
        if motion_failure:
            return START + ([.003, 0., 0.] if n else np.zeros(3))
        return START + [0., 0., 0. if timeout else min(n * .002 / .6, 1.) * .005]
    caches[audit.tick(start)] = START.copy()
    for n in range(1, 3001):
        timestamp = start + n * .002
        if (n - 1) % 10 == 0:
            clock, elapsed = start + (n - 1) * .002, ((n - 1) // 10 + 1) * .02
            cached = position(n - 1)
            caches[audit.tick(clock)] = cached.copy()
            # The fixture uses the frozen runtime function; the auditor never imports/calls it.
            ik = hook.reference.bounded_probe_reference(START, cached, elapsed)
            ik.update(timestamp_s=clock, actual_elapsed_s=clock - start,
                parameters=dict(audit.PARAMETERS))
            iks.append(ik)
        current, previous = position(n), position(n - 1)
        endpoint = bool(np.linalg.norm(current - START - [0., 0., .005]) <= .0005)
        feasible = not reset or n != 300
        confirmed = confirmed + .002 if endpoint and feasible and not motion_failure else 0.
        stop = "PROBE_HORIZONTAL_DRIFT" if motion_failure else None
        policy = {"timestamp_s": timestamp, "probe_active": True, "probe_start_s": start,
            "actual_wrist_position_world_m": current.tolist(),
            "actual_probe_displacement_m": (current - START).tolist(),
            "actual_wrist_speed_m_s": float(np.linalg.norm(current - previous) / .002),
            "actual_probe_endpoint_reached": endpoint, "table_load_n": 0.,
            "catalogue_weight_n": 3.42369, "table_unloaded": True,
            "sampled_fixed_normal_1g_feasible": feasible, "sampled_solve_current": True,
            "accepted_visual_after_probe_start": True, "associated_active_fingers": ["thumb", "middle"],
            "motion_safety_reasons": [], "probe_confirmed_seconds": confirmed, "stop_reason": stop}
        event = {"timestamp_s": timestamp, "elapsed_s": timestamp - start,
            "parameters": dict(audit.PARAMETERS), "actual_admission_start_world_m": START.tolist(),
            "current_actual_wrist_world_m": current.tolist(), "currently_probe_active": True,
            "inherited_confirmed_seconds": confirmed, "latest_ik_timestamp_s": ik["timestamp_s"],
            "ik_sample_age_s": timestamp - ik["timestamp_s"], "runtime_target_state_used": False}
        for key in ("desired_actual_path_world_m", "bounded_reference_correction_world_m", "cartesian_ik_target_world_m"):
            event[key] = list(ik[key])
        events.append(event)
        ready = confirmed >= .25
        reason = "ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED" if ready else "MAXIMUM_PROBE_DURATION_REACHED"
        rows.append({"timestamp_s": timestamp, "position": current.copy(), "trace_sample": copy.deepcopy(event),
            "policy": policy, "termination": reason if ready else None, "table_load_n": 0.})
        if ready or motion_failure:
            break
    error = "PROBE_HORIZONTAL_DRIFT" if motion_failure else None
    outcome = {"timestamp_s": timestamp, "elapsed_s": timestamp - start,
        "arm_exit_reason": "MOTION_EXCEPTION" if error else reason,
        "final_reason": error or ("ORIGINAL_DECISIVE_CONFIRMATION_ACCEPTED" if ready else "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED"),
        "motion_failure": error, "inherited_confirmed_seconds": confirmed,
        "original_confirmation_attempted": not motion_failure,
        "original_confirmation_accepted": None if motion_failure else ready}
    decisive = None if motion_failure else dict(copy.deepcopy(policy), decisive_current_solve=True, confirmation=ready)
    if not motion_failure:
        outcome["confirmation_timestamp_s"] = timestamp
    item = {"start_s": start, "actual_admission_start_world_m": START.tolist(),
        "samples": events, "ik_samples": iks, "outcome": outcome}
    admission = {"timestamp_s": start, "admission": True, "decisive_current_solve": True,
        "actual_wrist_position_world_m": START.tolist(), "associated_active_fingers": ["thumb", "middle"]}
    transitions = [] if ready else [{"timestamp": timestamp, "state": "RECOVER",
        "reason": error or "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED"}]
    return [item, rows, caches, admission, decisive, transitions, 3.42369]


def test_reference_has_known_cubic_path_and_two_millimetre_vector_cap():
    middle = audit.reference_values([0., 0., 0.], [0., 0., 0.], 1.5)
    assert middle["cubic_progress"] == .5
    np.testing.assert_allclose(middle["desired_actual_path_world_m"], [0., 0., .0025])
    np.testing.assert_allclose(middle["cartesian_ik_target_world_m"], [0., 0., .0045])
    result = audit.reference_values([0., 0., 0.], [.003, 0., .009], 6.)
    np.testing.assert_allclose(result["bounded_reference_correction_world_m"], [-.0012, 0., -.0016])
    np.testing.assert_allclose(result["cartesian_ik_target_world_m"], [-.0012, 0., .0034])
    assert audit.reference_values(START, START, 0.)["cubic_progress"] == 0.


def test_timeout_is_valid_failed_probe_with_original_decisive_rejection():
    result = audit.audit_probe(*fixture(timeout=True))
    assert result["duration_s"] == 6. and result["native_samples"] == 3000
    assert result["ik_samples"] == 300
    assert result["classification"] == "original_confirmation_rejected"
    assert result["recorded_lp_association_and_visual_flags_independently_recomputed"] is False


def test_early_exit_is_first_continuous_window_and_resets_after_a_bad_recorded_condition():
    normal, reset = fixture(), fixture(reset=True)
    normal_result, reset_result = audit.audit_probe(*normal), audit.audit_probe(*reset)
    assert normal_result["classification"] == reset_result["classification"] == "original_confirmation_accepted_probe_only"
    assert reset_result["duration_s"] > normal_result["duration_s"]
    reset[0]["samples"][300]["inherited_confirmed_seconds"] = .25
    reset[1][300]["trace_sample"]["inherited_confirmed_seconds"] = .25
    with pytest.raises(ValueError, match="sample hand window"):
        audit.audit_probe(*reset)


def test_native_hand_failure_is_legal_before_six_seconds_without_confirmation():
    evidence = fixture(motion_failure=True)
    result = audit.audit_probe(*evidence)
    assert result["classification"] == "motion_failure" and result["duration_s"] == pytest.approx(.002)
    assert result["original_confirmation_attempted"] is False
    evidence[0]["outcome"]["motion_failure"] = "UNRELATED_FAILURE"
    with pytest.raises(ValueError, match="hand stop was replaced"):
        audit.audit_probe(*evidence)


def r16_abort_fixture():
    evidence = fixture()
    item, rows, caches, _, _, transitions, _ = evidence
    item["samples"] = item["samples"][:300]
    item["ik_samples"] = item["ik_samples"][:30]
    evidence[1] = rows = rows[:300]
    end = rows[-1]["timestamp_s"]
    for event in (item["samples"][-1], rows[-1]["trace_sample"]):
        event.update(currently_probe_active=False, inherited_confirmed_seconds=0.)
    # write() inhibits before parent after_step; parent records one eligible 2ms,
    # then the R16 after_step wrapper inhibits again and aborts before R23 sample.
    rows[-1]["policy"]["probe_confirmed_seconds"] = .002
    item["outcome"].update(timestamp_s=end, elapsed_s=end - item["start_s"],
        arm_exit_reason="MOTION_EXCEPTION", final_reason="PROBE_BIAS_RECOVERY",
        motion_failure="PROBE_BIAS_RECOVERY", inherited_confirmed_seconds=0.,
        probe_active_at_motion_end=False, original_confirmation_attempted=False,
        original_confirmation_accepted=None)
    item["outcome"].pop("confirmation_timestamp_s")
    evidence[4] = None
    transitions.append({"timestamp": end, "state": "RECOVER", "reason": "PROBE_BIAS_RECOVERY"})
    after = {"probe_active": False, "payload_confirmed": False, "probe_confirmed_seconds": 0.,
        "probe_eligible_seconds": 0., "pressure_window_samples": 0, "preload_ready": False}
    abort = {"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
        "additional_bias_limit_rad": .10, "timestamp_s": end - .002,
        "post_step_timestamp_s": end, "phase": "TRIAL_LIFT", "probe_abort_applied": True,
        "recovery_active": True, "failure_latched": False,
        "admission_before": {"probe_active": True, "payload_confirmed": False},
        "admission_after_step": after, "post_step_returned_failure": "PROBE_BIAS_RECOVERY",
        "resolved_abort_reason": "PROBE_BIAS_RECOVERY", "original_post_step_failure": None}
    return evidence, {audit.tick(end): abort}


def test_original_r16_abort_before_sample_is_bound_and_resets_parent_and_runtime_counters():
    evidence, aborts = r16_abort_fixture()
    result = audit.audit_probe(*evidence, aborts)
    assert result["classification"] == "motion_failure"
    assert result["original_r16_pre_sample_abort_observed"] is True
    assert result["final_confirmed_seconds"] == 0.
    with pytest.raises(ValueError, match="Inactive probe sample lacks"):
        audit.audit_probe(*evidence)


@pytest.mark.parametrize("mutation", ["clock", "returned_failure", "after_active", "not_applied", "not_inhibited",
    "format", "inherited_format", "budget"])
def test_inactive_probe_needs_exact_matching_original_abort_evidence(mutation):
    evidence, aborts = r16_abort_fixture()
    abort = next(iter(aborts.values()))
    if mutation == "clock":
        abort["post_step_timestamp_s"] -= .002
    elif mutation == "returned_failure":
        abort["post_step_returned_failure"] = "OTHER_FAILURE"
    elif mutation == "after_active":
        abort["admission_after_step"]["probe_active"] = True
    elif mutation == "not_applied":
        abort["probe_abort_applied"] = False
    elif mutation == "not_inhibited":
        abort["admission_after_step"]["pressure_window_samples"] = 126
    elif mutation == "format":
        abort["format"] = "r16_bias_recovery_write_v1"
    elif mutation == "inherited_format":
        abort["inherited_write_format"] = "other_format"
    else:
        abort["additional_bias_limit_rad"] = .12
    with pytest.raises(ValueError, match="R16 abort|R16 abort/inhibit"):
        audit.audit_probe(*evidence, aborts)


@pytest.mark.parametrize("mutation,match", [
    ("reference", "IK cartesian_ik_target"), ("lookahead", "lookahead"),
    ("cached_wrist", "sample cached wrist"), ("native_table", "table load"),
    ("decisive", "confirmation result"), ("recovery", "recovery transition"),
    ("extra_time", "six-second bound"),
])
def test_inconsistent_reference_native_geometry_and_outcomes_fail_closed(mutation, match):
    evidence = fixture()
    if mutation == "reference":
        evidence[0]["ik_samples"][0]["cartesian_ik_target_world_m"][0] += .001
    elif mutation == "lookahead":
        evidence[0]["ik_samples"][0]["path_elapsed_s"] = 0.
    elif mutation == "cached_wrist":
        evidence[1][0]["position"][0] += .001
    elif mutation == "native_table":
        evidence[1][0]["table_load_n"] = 2.
    elif mutation == "decisive":
        evidence[4]["confirmation"] = False
    elif mutation == "recovery":
        evidence = fixture(motion_failure=True)
        evidence[5].clear()
    else:
        evidence[0]["outcome"]["timestamp_s"] = 7.002
    with pytest.raises(ValueError, match=match):
        audit.audit_probe(*evidence)


def test_exact_extensions_restore_whole_inherited_arm_ast_and_reject_changed_guard():
    inherited = (PROJECT / audit.BASELINE / "ycb_can_bias_budget_r17_seed3414/derived_controller.py").read_text()
    namespace = {}
    exec(compile(inherited, "<frozen_r22_fixture>", "exec"), namespace)
    _, rendered, _ = hook.transform(namespace["run_task"], inherited)
    assert audit.verify_ast_extension(inherited, rendered)["entire_r22_arm_ast_restored"] is True
    assert "raise MotionFailure('ARM_VELOCITY_LIMIT')" in rendered
    tampered = rendered.replace("raise MotionFailure('ARM_VELOCITY_LIMIT')", "pass", 1)
    with pytest.raises(ValueError, match="entire executed R22"):
        audit.verify_ast_extension(inherited, tampered)


def test_table_contact_force_direction_uses_geom2_and_all_world_components():
    frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    contact = {"geom_ids": [2, 1], "contact_frame_world_rows": frame,
        "force_torque_contact_frame": [4., 5., -2., 0., 0., 0.]}
    assert audit.table_load({"native_contacts": [contact]}, 1, 2) == 4.
    contact["geom_ids"] = [1, 2]
    assert audit.table_load({"native_contacts": [contact]}, 1, 2) == 0.


def test_stream_binding_uses_first_duplicate_and_previous_postintegration_encoder(tmp_path):
    nested = "".join(f'<body name="link{i}" pos="{".6 0 .7" if i == 1 else "0 0 0"}">'
        f'<joint name="arm{i}" axis="0 0 1"/><geom type="sphere" size=".01"/>' for i in range(1, 8))
    hands = "".join(f'<body><joint name="hand{i}"/><geom type="sphere" size=".01"/></body>' for i in range(22))
    motors = "".join(f'<motor joint="arm{i}"/>' for i in range(1, 8))
    motors += "".join(f'<motor name="{name}" joint="hand{i}"/>' for i, name in enumerate(OPEN))
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody>' + nested + hands +
        '</body>' * 7 + '<geom name="target_geom" type="sphere" size=".01"/>' +
        '<geom name="table" type="plane" size="1 1 .1"/></worldbody><actuator>' + motors + '</actuator></mujoco>')
    trace, native = [], []
    for n in (1, 2, 3):
        angle, phase = .1 * (n - 1), "TRIAL_LIFT" if n == 3 else "ALIGN"
        rotation = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
        trace.append({"timestamp": n * .002, "state": phase, "hand_command": list(OPEN.values()),
            "arm_command": [0.] * 7, "wrist_position": START.tolist(), "wrist_rotation_world": rotation,
            "arm_joint_position": [.1 * n] + [0.] * 6, "probe_policy": {},
            "actual_wrist_probe": {"tick": n}})
        native.append({"timestamp": n * .002, "state": phase,
            "motor_commands": [0.] * 7 + list(OPEN.values()), "native_contacts": []})
    trace.append(dict(trace[-1], state="RECOVER", hand_command=[.2] * 22))
    paths = [tmp_path / "native.gz", tmp_path / "trace.gz"]
    for path, content in zip(paths, [native, trace]):
        with gzip.open(path, "wt") as stream:
            json.dump(content, stream)
    probe = {"start_s": .004, "ik_samples": [], "outcome": {"timestamp_s": .006}}
    aligns, probes, caches, _, count, residual, _ = audit.collect_streams(*paths, model,
        [{"start_s": 0., "end_s": .004}], [probe], OPEN)
    assert count == 3 and residual < 1e-15
    assert aligns[0][0]["fk_position"] is None
    assert probes[0][0]["trace_sample"] == {"tick": 3}
    np.testing.assert_allclose(caches[2], START)


def test_final_summary_binding_is_explicit_before_any_source_or_stream_read(tmp_path):
    (tmp_path / "summary.json").write_text('{}')
    (tmp_path / "protocol.json").write_text('{}')
    (tmp_path / "frozen_source.zip").write_bytes(b'')
    with pytest.raises(ValueError, match="explicit lowercase"):
        audit.audit_directory(tmp_path, "pending")
    with pytest.raises(ValueError, match="explicitly expected"):
        audit.audit_directory(tmp_path, "0" * 64)


@pytest.mark.parametrize("mutation", [None, "archive_bytes", "inherited_manifest", "before_after"])
def test_frozen_source_proof_binds_all_167_bytes_and_inherited_146(tmp_path, monkeypatch, mutation):
    sources = {f"scripts/inherited_{n}.py": str(n).encode() for n in range(146)}
    old = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
    baseline = {"protocol": {"source_hashes_before": dict(old)}, "rows": [{"frozen": True}]}
    if mutation == "inherited_manifest":
        baseline["protocol"]["source_hashes_before"]["scripts/inherited_0.py"] = "0" * 64
    sources[audit.BASELINE + "/summary.json"] = json.dumps(baseline).encode()
    inherited_name = audit.BASELINE + "/ycb_can_bias_budget_r17_seed3414/derived_controller.py"
    sources[inherited_name] = b'def run_task():\n    pass\n'
    sources.update({f"scripts/new_{n}.py": str(n).encode() for n in range(19)})
    before = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
    protocol = {"source_hashes_before": before, "source_hashes_after": dict(before), "source_changed_during_run": False}
    monkeypatch.setattr(audit, "PINS", {audit.BASELINE + "/summary.json": before[audit.BASELINE + "/summary.json"],
        inherited_name: before[inherited_name]})
    if mutation == "archive_bytes":
        sources["scripts/inherited_0.py"] = b'changed'
    if mutation == "before_after":
        protocol["source_hashes_after"]["scripts/inherited_0.py"] = "0" * 64
    with zipfile.ZipFile(tmp_path / "frozen_source.zip", "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    if mutation:
        with pytest.raises(ValueError, match="differ"):
            audit.source_proof(tmp_path, protocol)
    else:
        proof, inherited, row = audit.source_proof(tmp_path, protocol)
        assert proof["source_count"] == 167 and proof["inherited_r22_source_count"] == 146
        assert inherited == sources[inherited_name].decode() and row == {"frozen": True}
