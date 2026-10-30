"""Narrow evidence tests; no simulator integration or actual R22 audit."""

import copy
import gzip
import hashlib
import json
import zipfile

import mujoco
import numpy as np
import pytest

import audit_public_ycb_settled_alignment as audit


OPEN = {"left_thumb_CMC_FE": 1.3, **{f"left_test_{i}": 0. for i in range(21)}}
POSITION = np.array([.6, 0., .7])


def stable_evidence(*, timeout=False, reset_at=None):
    count = 3750 if timeout else 750 if reset_at is None else reset_at + 126
    start, observations, events = 1.9, [], []
    for n in range(1, count + 1):
        timestamp = start + n * audit.DT
        incomplete = timeout or n == reset_at
        command_error = .02 if incomplete else 0.
        command = np.array([command_error] + [0.] * 6)
        if incomplete or n == 1:
            continuous = 0.
        elif reset_at is not None and n > reset_at:
            continuous = timestamp - (start + (reset_at + 1) * audit.DT)
        else:
            continuous = timestamp - (start + 2 * audit.DT)
        reasons = (["FULL_GRASP_COMMAND_NOT_REACHED"] if incomplete else [])
        if n == 1:
            reasons += ["ACTUAL_WRIST_SPEED"]
        event = {"timestamp_s": timestamp, "elapsed_s": timestamp - start,
            "nominal_finished": timestamp - start >= 1.5 - 1e-9,
            "extra_elapsed_s": max(0., timestamp - start - 1.5),
            "actual_wrist_position_world_m": POSITION.tolist(),
            "target_wrist_position_world_m": POSITION.tolist(), "position_error_m": 0.,
            "actual_wrist_speed_m_s": None if n == 1 else 0., "orientation_error_rad": 0.,
            "orientation_error_is_a_gate": False, "maximum_arm_command_error_rad": command_error,
            "open_hand_output": {"expected_command_rad": dict(OPEN),
                "actual_command_rad": dict(OPEN), "maximum_error_rad": 0., "accepted": True},
            "continuous_seconds": continuous, "reasons": reasons,
            "ready": not timeout and n == count, "runtime_target_state_used": False}
        events.append(event)
        observations.append({"timestamp_s": timestamp, "position": POSITION.copy(),
            "rotation": np.eye(3), "fk_position": POSITION.copy(), "fk_rotation": np.eye(3),
            "command": command, "open_commands": dict(OPEN), "alignment_settling": event})
    end = start + count * audit.DT
    reason = "ALIGN_SETTLE_TIMEOUT" if timeout else "OPEN_FULL_GRASP_COMMAND_AND_ACTUAL_WRIST_SETTLED"
    attempt = {"start_s": start, "nominal_seconds": 1.5, "target_arm_command_rad": [0.] * 7,
        "target_wrist_rotation_world": np.eye(3).tolist(), "expected_open_command_rad": dict(OPEN),
        "events": events, "outcome": {"settled": not timeout, "reason": reason, "timestamp_s": end}}
    interval = {"start_s": start, "end_s": end, "next_state": "RECOVER" if timeout else "CLOSE",
        "next_reason": reason if timeout else "ALIGN_DONE"}
    return attempt, observations, interval


def reconstruct(evidence):
    return audit.reconstruct_attempt(*evidence, OPEN, lambda command: (POSITION.copy(), np.eye(3)))


def test_recomputes_nominal_open_ready_and_close_start():
    result = reconstruct(stable_evidence())
    assert result["classification"] == "settled"
    assert result["cached_wrist_fk_samples"] == result["native_samples"] == 750
    assert result["extra_seconds"] == 0.
    close = {"first_native_s": result["end_s"] + .002, "command": [0.] * 7}
    audit.verify_close_linkage([close], [result])
    assert result["first_close_full_target_command_error_rad"] == 0.
    close["command"][0] = .0001
    with pytest.raises(ValueError, match="incomplete insertion"):
        audit.verify_close_linkage([close], [result])
    with pytest.raises(ValueError, match="exactly one"):
        audit.verify_close_linkage([close], [{"settled": False}])


def test_command_incompletion_resets_continuous_window():
    evidence = stable_evidence(reset_at=730)
    result = reconstruct(evidence)
    assert result["extra_seconds"] == pytest.approx(.212)
    evidence[0]["events"][730]["continuous_seconds"] = 1.458
    with pytest.raises(ValueError, match="continuous_seconds"):
        reconstruct(evidence)


def test_bounded_timeout_is_a_valid_failed_alignment():
    evidence = stable_evidence(timeout=True)
    result = reconstruct(evidence)
    assert result["classification"] == "timeout" and result["settled"] is False
    assert result["extra_seconds"] == pytest.approx(6.)
    evidence[2]["end_s"] += .002
    with pytest.raises(ValueError, match="six-second bound"):
        reconstruct(evidence)


@pytest.mark.parametrize("mutation,match", [
    ("cached_fk", "previous integrated"), ("native_open", "Open-hand telemetry"),
    ("self_consistent_bad_open", "non-open ALIGN output"), ("clock", "contiguous ALIGN"),
])
def test_independent_measurements_reject_consistent_looking_telemetry(mutation, match):
    evidence = stable_evidence()
    event, observation = evidence[0]["events"][10], evidence[1][10]
    if mutation == "cached_fk":
        observation["fk_position"][2] += .001
    elif mutation == "native_open":
        observation["open_commands"]["left_test_0"] = .1
    elif mutation == "self_consistent_bad_open":
        observation["open_commands"]["left_test_0"] = .1
        event["open_hand_output"].update(actual_command_rad=dict(observation["open_commands"]),
            maximum_error_rad=.1, accepted=False)
    else:
        observation["timestamp_s"] += .002
    with pytest.raises(ValueError, match=match):
        reconstruct(evidence)


def test_safety_abort_can_leave_one_native_sample_without_policy_row():
    evidence = stable_evidence()
    attempt, observations, interval = evidence
    attempt["events"].pop()
    observations[-1]["alignment_settling"] = None
    attempt["outcome"] = None
    interval.update(next_state="RECOVER", next_reason="ARM_VELOCITY_LIMIT")
    result = reconstruct(evidence)
    assert result["classification"] == "inherited_safety_abort"
    assert result["trailing_safety_abort_native_samples"] == 1
    observations[-1]["alignment_settling"] = {"unaccounted": True}
    with pytest.raises(ValueError, match="unaccounted"):
        reconstruct(evidence)


def test_preintegration_nonopen_failure_is_declared_and_never_physical_output_proof():
    attempt, _, interval = stable_evidence()
    bad = dict(OPEN, left_test_0=.1)
    attempt.update(events=[], outcome={"settled": False, "reason": "ALIGN_HAND_NOT_OPEN",
        "timestamp_s": attempt["start_s"], "open_hand_output": {"expected_command_rad": dict(OPEN),
            "actual_command_rad": bad, "maximum_error_rad": .1, "accepted": False}})
    interval.update(end_s=attempt["start_s"], next_state="RECOVER", next_reason="ALIGN_HAND_NOT_OPEN")
    result = reconstruct((attempt, [], interval))
    assert result["classification"] == "declared_unintegrated_nonopen_output_abort"
    assert result["preintegration_failure_outputs_independently_observed"] is False
    attempt["outcome"]["open_hand_output"].update(actual_command_rad=dict(OPEN), maximum_error_rad=0.)
    with pytest.raises(ValueError, match="fully open outputs"):
        reconstruct((attempt, [], interval))


def write_gzip(path, value):
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream)


def test_native_bind_uses_first_duplicate_trace_and_previous_encoder_fk(tmp_path):
    nested = "".join(f'<body name="link{i}" pos="{".6 0 .7" if i == 1 else "0 0 0"}">'
        f'<joint name="arm{i}" axis="0 0 1"/><geom type="sphere" size=".01"/>' for i in range(1, 8))
    hands = "".join(f'<body><joint name="hand{i}"/><geom type="sphere" size=".01"/></body>'
        for i in range(22))
    motors = "".join(f'<motor joint="arm{i}"/>' for i in range(1, 8))
    motors += "".join(f'<motor name="{name}" joint="hand{i}"/>' for i, name in enumerate(OPEN))
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody>' +
        nested + hands + '</body>' * 7 + '</worldbody><actuator>' + motors + '</actuator></mujoco>')
    rows, native = [], []
    for n in (1, 2):
        angle = .1 * (n - 1)
        rot = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
        rows.append({"timestamp": n * .002, "state": "ALIGN", "hand_command": list(OPEN.values()),
            "arm_command": [0.] * 7, "wrist_position": POSITION.tolist(), "wrist_rotation_world": rot,
            "arm_joint_position": [.1 * n] + [0.] * 6, "alignment_settling": {"ordinal": n}})
        native.append({"timestamp": n * .002, "state": "ALIGN", "motor_commands": [0.] * 7 + list(OPEN.values())})
    rows.append(dict(rows[-1], state="RECOVER", hand_command=[.2] * 22))
    rows[-1].pop("alignment_settling")
    trace_path, native_path = tmp_path / "trace.gz", tmp_path / "native.gz"
    write_gzip(trace_path, rows)
    write_gzip(native_path, native)
    samples, closes, count, _ = audit.collect_native(native_path, trace_path, model,
        [{"start_s": 0., "end_s": .004}], OPEN)
    assert count == 2 and closes == [] and samples[0][0]["fk_position"] is None
    np.testing.assert_allclose(samples[0][1]["fk_position"], POSITION)
    np.testing.assert_allclose(samples[0][1]["fk_rotation"], rows[1]["wrist_rotation_world"], atol=1e-15)
    assert samples[0][1]["alignment_settling"] == {"ordinal": 2}


@pytest.mark.parametrize("text", ["[1 2]", "[1,]", "[,1]", "[] {}"])
def test_stream_parser_rejects_invalid_array_structure(tmp_path, text):
    path = tmp_path / "invalid.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(text)
    with pytest.raises(ValueError):
        list(audit.json_array(path))


def test_frozen_closure_binds_all_146_archive_entries(tmp_path, monkeypatch):
    payloads = {name: name.encode() for name in [*audit.PINS, audit.BASELINE_SUMMARY]}
    payloads.update({f"fixture/source_{i}.py": str(i).encode() for i in range(140)})
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    monkeypatch.setattr(audit, "PINS", {name: manifest[name] for name in audit.PINS})
    monkeypatch.setattr(audit, "BASELINE_SHA256", manifest[audit.BASELINE_SUMMARY])
    protocol = {"source_hashes_before": manifest, "source_hashes_after": copy.deepcopy(manifest),
        "source_changed_during_run": False}
    with zipfile.ZipFile(tmp_path / "frozen_source.zip", "w") as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
    assert audit.source_proof(tmp_path, protocol)["source_count"] == 146
    with zipfile.ZipFile(tmp_path / "frozen_source.zip", "a") as archive:
        archive.writestr("unbound.py", "extra")
    with pytest.raises(ValueError, match="manifest differs"):
        audit.source_proof(tmp_path, protocol)
    protocol["source_hashes_after"]["new_source"] = "0" * 64
    with pytest.raises(ValueError, match="146-source identity"):
        audit.source_proof(tmp_path, protocol)


def test_incomplete_run_fails_before_any_stream_is_opened(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"complete": False}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="finalized explicitly versioned R22"):
        audit.audit_directory(tmp_path)
