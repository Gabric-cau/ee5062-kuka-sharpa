"""Physical recovery regression: a deliberate missed grasp is not a reset."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import kuka_sharpa_grasp.pick_place as task_module
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


def test_physical_missed_grasp_reobserves_and_retries_without_reset(tmp_path, monkeypatch):
    scene = build_pick_place_scene(tmp_path / "scene.xml", "ycb_can")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    original_reset = task_module.reset_task
    original_step = mujoco.mj_step
    reset_calls = 0
    step_calls = 0
    last_qpos = None
    sampled = []

    def counted_reset(*args, **kwargs):
        nonlocal reset_calls
        reset_calls += 1
        return original_reset(*args, **kwargs)

    def checked_step(live_model, live_data):
        nonlocal last_qpos, step_calls
        if live_data is data:
            if last_qpos is not None:
                np.testing.assert_array_equal(live_data.qpos, last_qpos,
                                              err_msg="Live generalized positions changed outside physics")
            original_step(live_model, live_data)
            last_qpos = live_data.qpos.copy()
            step_calls += 1
        else:
            original_step(live_model, live_data)

    def observe_step(stage, live):
        assert live is data
        sampled.append((str(stage), float(live.time)))

    monkeypatch.setattr(task_module, "reset_task", counted_reset)
    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    result, trace = task_module.run_task(
        model, data, task_module.Scenario(0), max_attempts=2,
        first_grasp_bias=(0., .12, 0.), step_callback=observe_step,
    )

    assert reset_calls == 1
    assert step_calls > 1000
    assert len(result["attempts"]) == 2
    first, second = result["attempts"]
    assert first["failure"] == "MISSING_CONTACT"
    assert second["belief_timestamp"] > first["belief_timestamp"]
    assert second["selected_candidate"] is not None
    assert first["selected_candidate"][1] - second["selected_candidate"][1] > .06
    assert result["safe_abort_attempts"] == 1
    assert result["maximum_attempts"] == 2
    assert result["success"], result
    assert result["failure"] is None
    assert result["successful_recoveries"] == 1
    assert trace[-1]["oracle"]["success"]
    assert trace[-1]["oracle"]["table_support"]
    assert trace[-1]["oracle"]["released"]
    assert trace[-1]["oracle"]["hand_clearance_m"] > .10

    transitions = result["transitions"]
    search_indices = [index for index, transition in enumerate(transitions)
                      if transition["state"] == "SEARCH"]
    assert len(search_indices) >= 3  # Initial entry plus both physical attempts.
    second_search = next(index for index in search_indices
                         if transitions[index]["timestamp"] >= second["belief_timestamp"] - 1.51)
    assert any(item["state"] == "RECOVER" for item in transitions[:second_search])
    assert {"APPROACH", "ALIGN", "CLOSE", "TRIAL_LIFT", "TRANSFER", "PLACE",
            "RELEASE", "VERIFY"} <= {item["state"] for item in transitions[second_search:]}
    times = [row["timestamp"] for row in trace]
    assert times == sorted(times)
    assert sampled[-1][1] == pytest.approx(result["duration"])
    assert sampled[-1][1] == pytest.approx(step_calls * model.opt.timestep)
