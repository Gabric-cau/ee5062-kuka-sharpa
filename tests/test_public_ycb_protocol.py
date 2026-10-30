"""Public random baseline must preserve its schedule, unknowns and physical state."""

from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_public_ycb import KINDS, NativeAudit, schedule_for, summarize
from develop_public_ycb import longer_motion_controller
from kuka_sharpa_grasp import pick_place


def test_schedule_is_frozen_random_identity_and_new_pose_seeds():
    schedule = schedule_for(20261014, 3200, 18)
    assert schedule == schedule_for(20261014, 3200, 18)
    assert [row["pose_seed"] for row in schedule] == list(range(3200, 3218))
    assert set(row["kind"] for row in schedule) == set(KINDS)
    assert schedule != schedule_for(20261015, 3200, 18)


def test_runtime_errors_remain_unknown_in_all_attempts_denominator():
    rows = [{"kind": KINDS[0], "success": success, "held_lift": success,
             "runner_error": "error" if success is None else None}
            for success in (True, False, None)]
    summary = summarize(rows, 3)
    assert summary["task_success"]["numerator"] == 1
    assert summary["task_success"]["unknown"] == 1
    assert summary["task_success"]["total"] == 3
    assert summary["task_success"]["success_rate_all_attempts"] == 1 / 3


def test_native_audit_detects_external_target_position_write_and_restores_api():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="target" pos="0 0 1"><freejoint name="target_free"/>
      <geom name="target_geom" type="sphere" size=".02" mass=".1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    original = mujoco.mj_step
    with NativeAudit(model, data) as audit:
        data.qpos[0] = .5
        mujoco.mj_step(model, data)
        audit.sample("SEARCH", data)
        audit.sample("SEARCH", data)
        data.qpos[0] += .1
        mujoco.mj_step(model, data)
        audit.sample("SEARCH", data)
    assert mujoco.mj_step is original
    assert audit.integration_steps == len(audit.samples) == 2
    assert len(audit.external_qpos_writes) == 1
    assert np.allclose(audit.initial_target_qpos[:3], [.5, 0., 1.])


def test_development_duration_profile_leaves_shared_controller_unchanged():
    original = pick_place.run_task
    derived, source = longer_motion_controller()
    assert derived is not original and pick_place.run_task is original
    assert derived.__globals__ is not original.__globals__
    assert "duration = max(duration, 3.0)" in source
    assert source.count("duration = max(duration, 3.0)") == 1


def test_vertical_development_changes_only_pregrasp_direction_and_duration():
    _, source = longer_motion_controller(vertical=True)
    assert "grasp_offset + [0.0, 0.0, 0.18]" in source
    assert "duration = max(duration, 3.0)" in source
    assert "maximum_force_n=12.0" in source
