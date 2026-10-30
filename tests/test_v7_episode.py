"""Portable recording contracts for interactive V7 evidence."""

from pathlib import Path
import json
import importlib.util
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.teleop import TeleopCommand
from kuka_sharpa_grasp.v7_episode import FORMAT, TeleopEpisodeRecorder


def _workbench_script_module():
    script = PROJECT / "scripts" / "run_v7_workbench.py"
    spec = importlib.util.spec_from_file_location("v7_workbench_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(time_s):
    return {
        "time_s": time_s,
        "status": "READY",
        "task_success": None,
        "tactile": {"fingers": {"index": {"normal_force_n": 0.0}}},
    }


def test_episode_recorder_persists_commands_trace_events_and_null_success(tmp_path):
    command = TeleopCommand(.05, translation_m=(.01, 0., 0.), source="operator")
    recorder = TeleopEpisodeRecorder(
        tmp_path, seed=13, mode="interactive", shared_control=True,
        initial_commands=(TeleopCommand(0., source="replay"),),
        metadata={"source_sha256": {"example.py": "abc"}},
    )
    recorder.record_command(command)
    recorder.record_snapshot(_snapshot(0.0))
    recorder.record_snapshot(_snapshot(0.05))
    metadata = recorder.finalize(events=({"event": "OPERATOR_TARGET", "time_s": .0},))

    assert metadata["format"] == FORMAT
    assert metadata["task_success"] is None
    assert metadata["commands"] == 2
    assert metadata["frames"] == 2
    action_log = json.loads((tmp_path / "action_log.json").read_text(encoding="utf-8"))
    assert [item["source"] for item in action_log["commands"]] == ["replay", "operator"]
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))[0]["event"] == "OPERATOR_TARGET"
    assert not list(tmp_path.glob(".*.tmp"))
    assert recorder.closed


def test_episode_recorder_rejects_nonmonotonic_snapshots_and_non_json_values(tmp_path):
    recorder = TeleopEpisodeRecorder(tmp_path, seed=0, mode="test", shared_control=False)
    recorder.record_snapshot(_snapshot(.2))
    with pytest.raises(ValueError, match="nondecreasing"):
        recorder.record_snapshot(_snapshot(.1))
    with pytest.raises(TypeError):
        recorder.record_snapshot({"time_s": .3, "bad": object()})


def test_episode_recorder_is_single_finalize_and_rejects_late_commands(tmp_path):
    recorder = TeleopEpisodeRecorder(tmp_path, seed=0, mode="test", shared_control=False)
    first = recorder.finalize(task_success=False, task_success_note="collision")
    second = recorder.finalize(task_success=True)
    assert second == first
    with pytest.raises(RuntimeError, match="finalized"):
        recorder.record_command(TeleopCommand(0.))


def test_operator_event_log_schema_does_not_claim_human_identity(tmp_path):
    workbench = _workbench_script_module()
    path = tmp_path / "operator_events.json"
    workbench.write_operator_events(path, [{"time_s": 0.0, "event": "KEY_DOWN", "key": "w"}])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "kuka-sharpa-v7-operator-events"
    assert payload["version"] == 1
    assert payload["events"][0]["event"] == "KEY_DOWN"


def test_gui_state_changes_replay_before_the_next_physics_step(tmp_path):
    workbench = _workbench_script_module()
    path = tmp_path / "operator_events.json"
    workbench.write_operator_events(path, [
        {"time_s": 0., "event": "MODE_CHANGE", "mode": "manual"},
        {"time_s": .1, "event": "TARGET_SELECT", "target": "tool_marker"},
        {"time_s": .1, "event": "FINGER_TARGET", "finger": "index", "value": .25},
        {"time_s": .2, "event": "KEY_DOWN", "key": "escape"},
    ])
    changes = []
    session = SimpleNamespace(shared_control=True,
        select_target=lambda target: changes.append(("target", target)),
        set_finger_target=lambda finger, value: changes.append((finger, value)),
        stop=lambda reason: changes.append(("stop", reason)))
    replay = workbench.OperatorEventReplay(path)
    replay.apply_due(0., session)
    assert session.shared_control is False
    assert changes == []
    replay.apply_due(.1, session)
    assert changes == [("target", "tool_marker"), ("index", .25)]
    replay.apply_due(.2, session)
    replay.apply_due(.3, session)
    assert changes[-1] == ("stop", "OPERATOR_ESCAPE")
    assert len(changes) == 3


def test_gui_bracket_keys_reach_keyboard_command_map():
    workbench = _workbench_script_module()
    panel = workbench.DesktopPanel.__new__(workbench.DesktopPanel)
    panel.held, panel.operator_events = set(), []
    panel.session = SimpleNamespace(data=SimpleNamespace(time=0.))
    event = SimpleNamespace(keysym="bracketright")
    panel.key_down(event)
    assert panel.held == {"]"}
    panel.key_up(event)
    assert not panel.held


def test_operator_replay_rejects_reversed_event_time(tmp_path):
    workbench = _workbench_script_module()
    path = tmp_path / "operator_events.json"
    workbench.write_operator_events(path, [{"time_s": .2, "event": "FOCUS_LOST"},
                                         {"time_s": .1, "event": "FOCUS_LOST"}])
    with pytest.raises(ValueError, match="monotonic"):
        workbench.OperatorEventReplay(path)
