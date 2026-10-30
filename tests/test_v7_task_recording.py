import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.teleop import TeleopActionLog, TeleopCommand


SPEC = importlib.util.spec_from_file_location("v7_task_recording_entrypoint", PROJECT / "scripts/run_v7_tasks.py")
ENTRYPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRYPOINT)


def test_task_artifacts_keep_replayable_commands_and_physical_result(tmp_path):
    commands = [TeleopCommand(0., selected_target="tool_marker", source="task_executor"),
                TeleopCommand(.1, translation_m=(.01, 0., 0.), source="task_executor")]
    result = SimpleNamespace(
        object_id="tool_marker", slot_id="slot_marker", success=False,
        failure_reason="LIFT_TIMEOUT",
        trace=[{"time_s": .002, "selected_target": "tool_marker"},
               {"time_s": .2, "selected_target": "tool_marker"}],
        metrics={"action_log": [item.as_dict() for item in commands],
                 "events": [{"time_s": 0., "event": "OPERATOR_TARGET", "target": "tool_marker"}],
                 "qpos_write_policy": "executor_never_writes_object_qpos"})
    ENTRYPOINT.write_episode_artifacts(
        tmp_path, result, seed=100, shared_control=True, hashes={"source.py": "abc"}, video=True)
    episode = tmp_path / "tool_marker"
    replay = TeleopActionLog.from_json(episode / "action_log.json")
    assert replay.commands == tuple(commands)
    metadata = json.loads((episode / "metadata.json").read_text())
    assert metadata["task_success"] is False
    assert metadata["failure_reason"] == "LIFT_TIMEOUT"
    assert metadata["video_recording_mode"] == "synchronized_recorded_physics"
    assert metadata["video"] == "../video/tool_marker.mp4"
    assert metadata["task_metrics"]["qpos_write_policy"] == "executor_never_writes_object_qpos"
    assert len(json.loads((episode / "trace.json").read_text())) == 2
    assert json.loads((episode / "events.json").read_text())[0]["event"] == "OPERATOR_TARGET"
