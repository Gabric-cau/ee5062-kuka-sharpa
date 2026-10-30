"""Evidence and model/map isolation contracts for held-out robustness runs."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location("v7_robustness_runner", PROJECT / "scripts" / "evaluate_v7_robustness.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

from kuka_sharpa_grasp.v7_calibrate import EVIDENCE_KEY
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import PhaseRecord, TaskResult, calibrated_task_spec


def test_same_seed_pairs_keep_nominal_map_and_all_distractors():
    nominal, actual, error = RUNNER.paired_configs(200)
    nominal2, actual2, error2 = RUNNER.paired_configs(200)
    assert error == error2
    assert nominal.manifest() == nominal2.manifest()
    assert actual.manifest() == actual2.manifest()
    target = nominal.registry.by_id(RUNNER.TARGET)
    actual_target = actual.registry.by_id(RUNNER.TARGET)
    assert actual_target.pose[0] - target.pose[0] == pytest.approx(error["xy_error_m"][0])
    assert actual_target.pose[1] - target.pose[1] == pytest.approx(error["xy_error_m"][1])
    assert target != actual_target
    assert len(actual.registry.objects) == 6
    assert actual.registry.distractors == nominal.registry.distractors
    assert actual.obstacles == nominal.obstacles and actual.slots == nominal.slots
    assert all(-.005 <= value <= .005 for value in error["xy_error_m"])
    assert .8 <= error["mass_factor"] <= 1.2 and .8 <= error["friction_factor"] <= 1.2


def test_builder_compiles_registry_mass_friction_and_initial_pose(tmp_path):
    nominal, actual, _ = RUNNER.paired_configs(201)
    target = actual.registry.by_id(RUNNER.TARGET)
    scene = build_v7_workbench_scene(tmp_path / "scene.xml", config=actual)
    model = mujoco.MjModel.from_xml_path(str(scene))
    body = model.body(f"v7_obj_{RUNNER.TARGET}")
    assert model.body_mass[body.id] == pytest.approx(target.mass_kg)
    joint = int(model.body_jntadr[body.id])
    address = int(model.jnt_qposadr[joint])
    assert model.qpos0[address:address + 3] == pytest.approx(target.pose)
    target_geoms = np.flatnonzero(model.geom_bodyid == body.id)
    assert len(target_geoms) > 0
    for geom_id in target_geoms:
        assert model.geom_friction[geom_id] == pytest.approx(target.friction)
    assert nominal.registry.by_id(RUNNER.TARGET).pose != target.pose


def test_unknown_rate_has_explicit_denominator():
    assert RUNNER.rate([True, False, None]) == {
        "numerator": 1, "denominator": 2, "unknown": 1, "total": 3, "rate": .5}
    assert RUNNER.rate([None])["rate"] is None


def test_current_session_retention_default_and_explicit_override(tmp_path):
    nominal, actual, _ = RUNNER.paired_configs(203)
    scene = build_v7_workbench_scene(tmp_path / "scene.xml", config=actual)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert WorkbenchSession(model, nominal, shared_control=True).tactile_retention is True
    assert WorkbenchSession(model, nominal, shared_control=False).tactile_retention is True
    assert WorkbenchSession(model, nominal, shared_control=False,
                            tactile_retention=False).tactile_retention is False
    assert WorkbenchSession(model, nominal, shared_control=True, tactile_retention=False).tactile_retention is False


def release_fixture(*, speed=.001, angular=.001, force=0.):
    nominal, _, _ = RUNNER.paired_configs(202)
    task = calibrated_task_spec(RUNNER.TARGET, slot_id="slot_marker")
    slot = next(slot for slot in nominal.slots if slot.slot_id == task.slot_id)
    rows = tuple({
        "time_s": float(timestamp), "evaluation": {RUNNER.TARGET: {"position": list(slot.pose)}},
        "finger_closure": {"thumb": 0., "index": 0.},
        EVIDENCE_KEY: {"target_linear_speed_m_s": speed,
                       "target_angular_speed_rad_s": angular, "target_hand_force_n": force},
    } for timestamp in np.arange(1.002, 1.602, .002))
    phases = (PhaseRecord("RELEASE", .5, 1., "COMPLETE", 250, 0., 0),
              PhaseRecord("VERIFY", 1., 1.6, "COMPLETE", 300, 0., 0))
    return TaskResult(RUNNER.TARGET, task.slot_id, True, "SUCCESS", None, phases, rows, {}), task, nominal


def test_release_requires_measured_low_speed_open_fingers_and_no_target_contact():
    result, task, nominal = release_fixture()
    assert RUNNER.release_metrics(result, task, nominal)["stable_released_in_slot"] is True
    for kwargs in ({"speed": .021}, {"angular": .3}, {"force": .06}):
        result, task, nominal = release_fixture(**kwargs)
        assert RUNNER.release_metrics(result, task, nominal)["stable_released_in_slot"] is False


def test_release_missing_velocity_is_unknown_and_short_window_fails():
    result, task, nominal = release_fixture()
    incomplete = dict(result.trace[10])
    incomplete.pop(EVIDENCE_KEY)
    rows = (*result.trace[:10], incomplete, *result.trace[11:])
    assert RUNNER.release_metrics(replace(result, trace=rows), task, nominal)["stable_released_in_slot"] is None
    assert RUNNER.release_metrics(replace(result, trace=result.trace[:100]), task, nominal)["stable_released_in_slot"] is False


def test_pairs_only_compare_known_outcomes():
    records = [{"seed": seed, "mode": mode, "metrics": {"strict_physical_success": value}}
               for seed, mode, value in ((200, "shared", True), (200, "manual", False),
                                         (201, "shared", None), (201, "manual", True))]
    pairs = RUNNER.summary(records)["paired_comparison"]
    assert pairs["complete_pairs"] == 2 and pairs["unknown_pairs"] == 1
    assert pairs["shared_only"] == 1 and pairs["known_pairs"] == 1
