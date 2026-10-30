"""Adversarial checks of synthetic recording fixtures, never human evidence."""

import importlib.util
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("operator_verifier", PROJECT / "scripts/verify_v7_operator_recording.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def mutate(path, change):
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    write_json(path, payload)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic_video") / "two_frames.mp4"
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :640] = (170, 30, 50)
    frame[:, 640:] = (30, 160, 80)
    with imageio.get_writer(path, fps=20, codec="libx264", macro_block_size=None) as writer:
        writer.append_data(frame)
        writer.append_data(np.roll(frame, 160, axis=1))
    return path


@pytest.fixture
def recording(tmp_path, video):
    gui, replay = tmp_path / "gui", tmp_path / "replay"
    gui.mkdir()
    replay.mkdir()
    events = [
        {"time_s": 0., "event": "TARGET_SELECT", "target": "tool_marker"},
        {"time_s": 0., "event": "MODE_CHANGE", "mode": "manual"},
        {"time_s": 0., "event": "FINGER_TARGET", "finger": "index", "value": .2},
        {"time_s": 0., "event": "KEY_DOWN", "key": "w", "held_keys": ["w"]},
        {"time_s": .05, "event": "KEY_UP", "key": "w", "held_keys": []},
        {"time_s": .05, "event": "MODE_CHANGE", "mode": "shared"},
    ]
    commands = [
        {"timestamp": 0., "translation_m": [0., .01, 0.], "rotation_rad": [0., 0., 0.],
         "grip_delta": 0., "buttons": [], "selected_target": None, "source": "keyboard"},
        {"timestamp": .05, "translation_m": [0., 0., 0.], "rotation_rad": [0., 0., 0.],
         "grip_delta": 0., "buttons": [], "selected_target": None, "source": "keyboard"},
    ]
    trace = []
    for index in range(50):
        trace.append({"time_s": (index + 1) * .002, "status": "READY",
            "shared_control": index >= 25, "selected_target": "tool_marker",
            "wrist_position": [.5, 0., .5], "wrist_quaternion": [1., 0., 0., 0.],
            "arm_command": [0.] * 7, "arm_joint_position": [0.] * 7,
            "arm_joint_velocity": [0.] * 7, "hand_joint_position": {"left_index_PIP": index * .0001},
            "finger_closure": {"index": index * .0008},
            "tactile": {"fingers": {"index": {"normal_force_n": .3, "contacts": 1}}},
            "evaluation": {"tool_marker": {"position": [.5, .1, .1]}}, "task_success": None})
    session_events = [{"time_s": 0., "event": "OPERATOR_TARGET", "target": "tool_marker"}]
    common = {"format": "kuka-sharpa-v7-teleop-episode", "format_version": 1,
        "task_success": None, "duration_s": .1, "frames": 50, "commands": 2,
        "events": 1, "seed": 0, "shared_control": True, "fps": 20,
        "scene": "v7_workbench_r1", "physics_backend": "classic_mujoco_cpu", "mujoco": "3.15.0",
        "source_sha256": {"test_fixture_only.py": "a" * 64}}
    for directory, mode in ((gui, "interactive"), (replay, "replay")):
        metadata = {**common, "mode": mode}
        if directory == gui:
            metadata.update(input_provenance="unverified_gui_input", operator_event_count=len(events))
        else:
            metadata["operator_events_replayed"] = len(events)
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "trace.json", trace)
        write_json(directory / "events.json", session_events)
        write_json(directory / "action_log.json", {"format": "kuka-sharpa-v7-teleop", "commands": commands})
        (directory / "scene.xml").write_text('<mujoco model="synthetic-test-fixture"/>\n')
        shutil.copyfile(video, directory / "workbench.mp4")
    write_json(gui / "operator_events.json", {"format": "kuka-sharpa-v7-operator-events", "version": 1,
                                               "events": events})
    return gui, replay, tmp_path / "audit.json"


def test_unverified_input_stays_unverified_and_hashes_all_artifacts(recording):
    gui, replay, output = recording
    before = verifier.artifact_hashes(gui, gui=True)
    result = verifier.verify(gui, replay, output)
    assert result["status"] == "accepted_recording_and_replay_consistency"
    assert result["human_operator"] is False
    assert result["human_identity_or_input_technically_verified"] is False
    assert result["operator_attestation"]["status"] == "not_supplied"
    assert result["all_serialized_trace_fields_compared"]
    assert result["physics_frames"] == 50
    assert result["commands_checked_against_raw_tk_keys"] == 2
    assert result["artifact_sha256"]["gui"] == before == verifier.artifact_hashes(gui, gui=True)
    assert set(result["artifact_sha256"]["replay"]) == set(verifier.ARTIFACTS)
    assert result["videos"]["gui"]["decoded_frames"] == 2
    assert result["videos"]["gui"]["visible_first_last_change"]
    assert result["task_success"] is None


@pytest.mark.parametrize("field,change", [
    ("trace.json", lambda data: data[20]["tactile"]["fingers"]["index"].update(normal_force_n=.31)),
    ("trace.json", lambda data: data[20]["evaluation"]["tool_marker"]["position"].__setitem__(1, .2)),
    ("trace.json", lambda data: data[20].update(status="STOPPED")),
    ("trace.json", lambda data: data[20].pop("arm_joint_position")),
    ("events.json", lambda data: data[0].update(target="tool_wrench")),
    ("metadata.json", lambda data: data.update(operator_events_replayed=0)),
    ("metadata.json", lambda data: data.update(source_sha256={"changed.py": "b" * 64})),
    ("action_log.json", lambda data: data["commands"][0]["translation_m"].__setitem__(1, .02)),
])
def test_replay_tampering_rejected(recording, field, change):
    gui, replay, output = recording
    mutate(replay / field, change)
    with pytest.raises(ValueError):
        verifier.verify(gui, replay, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", [
    lambda data: data["events"][3].update(held_keys=[]),
    lambda data: data["events"][3].update(key="d", held_keys=["d"]),
    lambda data: data["events"][3].update(time_s=-.1),
    lambda data: data["events"][3].update(time_s=.001),
    lambda data: data["events"][4].update(time_s=.2),
    lambda data: data["events"][2].update(value=1.1),
    lambda data: data["events"][0].update(target="tool_wrench"),
    lambda data: data["events"][1].update(mode="shared"),
])
def test_raw_input_tampering_rejected(recording, mutation):
    gui, replay, output = recording
    mutate(gui / "operator_events.json", mutation)
    with pytest.raises(ValueError):
        verifier.verify(gui, replay, output)


def test_identically_altered_commands_still_checked_against_raw_events(recording):
    gui, replay, output = recording
    for directory in (gui, replay):
        mutate(directory / "action_log.json",
               lambda data: data["commands"][0]["translation_m"].__setitem__(1, .02))
    with pytest.raises(ValueError, match="Numeric replay mismatch: command"):
        verifier.verify(gui, replay, output)


def test_identically_missing_native_step_is_rejected(recording):
    gui, replay, output = recording
    for directory in (gui, replay):
        mutate(directory / "trace.json", lambda data: data.pop(20))
        mutate(directory / "metadata.json", lambda data: data.update(frames=49))
    with pytest.raises(ValueError, match="uninterrupted"):
        verifier.verify(gui, replay, output)


def test_nonfinite_state_rejected_even_when_both_runs_match(recording):
    gui, replay, output = recording
    for directory in (gui, replay):
        mutate(directory / "trace.json", lambda data: data[0]["wrist_position"].__setitem__(0, float("nan")))
    with pytest.raises(ValueError, match="Nonfinite"):
        verifier.verify(gui, replay, output)


def test_focus_loss_and_escape_clear_raw_keys(recording):
    gui, replay, output = recording
    def events(payload):
        payload["events"].insert(4, {"time_s": 0., "event": "KEY_DOWN", "key": "escape",
                                    "held_keys": ["escape", "w"]})
        payload["events"].insert(5, {"time_s": 0., "event": "KEY_DOWN", "key": "w", "held_keys": ["w"]})
        payload["events"].insert(6, {"time_s": 0., "event": "FOCUS_LOST", "held_keys": ["w"]})
    mutate(gui / "operator_events.json", events)
    mutate(gui / "metadata.json", lambda data: data.update(operator_event_count=9))
    mutate(replay / "metadata.json", lambda data: data.update(operator_events_replayed=9))
    for directory in (gui, replay):
        mutate(directory / "action_log.json",
               lambda data: data["commands"][0]["translation_m"].__setitem__(1, 0.))
    assert verifier.verify(gui, replay, output)["operator_event_types"]["FOCUS_LOST"] == 1


def test_decoded_video_count_and_blank_frames_rejected(recording):
    gui, replay, output = recording
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with imageio.get_writer(gui / "workbench.mp4", fps=20, codec="libx264", macro_block_size=None) as writer:
        writer.append_data(frame)
    with pytest.raises(ValueError, match="Incomplete or blank"):
        verifier.verify(gui, replay, output)


def test_nonblank_video_with_missing_frame_is_rejected(recording):
    gui, replay, output = recording
    with imageio.get_reader(gui / "workbench.mp4") as reader:
        frame = reader.get_data(0)
    with imageio.get_writer(gui / "workbench.mp4", fps=20, codec="libx264", macro_block_size=None) as writer:
        writer.append_data(frame)
    with pytest.raises(ValueError, match="Incomplete or blank"):
        verifier.verify(gui, replay, output)


def test_video_timebase_must_match_physical_metadata(recording):
    gui, replay, output = recording
    with imageio.get_reader(gui / "workbench.mp4") as reader:
        frames = [reader.get_data(index) for index in range(2)]
    with imageio.get_writer(gui / "workbench.mp4", fps=25, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    with pytest.raises(ValueError, match="Video fps"):
        verifier.verify(gui, replay, output)


def test_automated_gui_accepted_but_cannot_be_attested_as_human(recording, tmp_path):
    gui, replay, output = recording
    mutate(gui / "metadata.json", lambda data: data.update(input_provenance="automated_gui_input"))
    result = verifier.verify(gui, replay, output)
    assert result["input_provenance"] == "automated_gui_input"
    assert result["human_operator"] is False
    with pytest.raises(ValueError, match="automated GUI recording"):
        verifier.verify(gui, replay, tmp_path / "attested.json", attestation=tmp_path / "unused.json")


def test_optional_attestation_is_bound_self_report_never_technical_verification(recording, tmp_path):
    gui, replay, output = recording
    template = tmp_path / "attestation.json"
    verifier.verify(gui, replay, output, template_output=template)
    with pytest.raises(ValueError, match="requires an operator"):
        verifier.verify(gui, replay, tmp_path / "blank.json", attestation=template)
    mutate(template, lambda data: data.update(operator_name="Synthetic Test Operator",
        statement="Fixture self-report for verifier testing only.", signed_at_utc="2026-10-22T00:00:00Z"))
    result = verifier.verify(gui, replay, tmp_path / "attested.json", attestation=template)
    assert result["operator_attestation"]["status"] == "self_attested_human_operation"
    assert result["operator_attestation"]["technically_authenticated"] is False
    assert result["operator_attestation"]["sha256"] == verifier.sha256(template)
    assert result["human_operator"] is False
    mutate(template, lambda data: data["artifact_sha256"].update({"trace.json": "0" * 64}))
    with pytest.raises(ValueError, match="does not bind"):
        verifier.verify(gui, replay, tmp_path / "wrong_hash.json", attestation=template)


def test_outputs_cannot_overwrite_or_modify_original_recordings(recording, tmp_path):
    gui, replay, output = recording
    with pytest.raises(ValueError, match="immutable"):
        verifier.verify(gui, replay, gui / "new_audit.json")
    with pytest.raises(ValueError, match="different files"):
        verifier.verify(gui, replay, output, template_output=output)
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        verifier.verify(gui, replay, output)
    assert output.read_text() == "existing"
