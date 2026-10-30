"""Actual CPU motion checks with the existing recorded-camera fixture.

The camera is scripted to isolate motion; these checks do not establish real
RGB-D accuracy or random-scene success. Object/hand contacts remain MuJoCo.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'v5_existing_upgrade_contract', PROJECT / 'tests' / 'test_upgrade_contract.py')
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)

model = CONTRACTS.model
scripted_camera = CONTRACTS.scripted_camera
task_module = CONTRACTS.task_module


def execute_motion(model, camera_fixture, cartesian):
    scenario = task_module.Scenario(0)
    sensor = camera_fixture(scenario)
    data = mujoco.MjData(model)
    result, trace = task_module.run_task(
        model, data, scenario, max_attempts=1, cartesian_motion=cartesian,
        visual_transfer=True, placement_retention=False, support_load_sensor=False)
    assert sensor['polls'] >= 2, result
    return result, trace, data


def phase_motion(model, trace, state):
    indexes = [index for index, row in enumerate(trace) if row['state'] == state]
    assert len(indexes) >= 100, state
    # The preceding command is the actual segment start, rather than a nominal
    # Cartesian goal that the IK solution might not reach exactly.
    rows = [trace[indexes[0] - 1]] + [trace[index] for index in indexes]
    kin = task_module.RobotKinematics(model)
    positions, rotations = [], []
    for row in rows:
        positions.append(kin.position(np.asarray(row['arm_command'])))
        rotations.append(kin.scratch.xmat[kin.body].reshape(3, 3).copy())
    positions = np.asarray(positions)
    displacement = positions[-1] - positions[0]
    length = np.linalg.norm(displacement)
    assert length > .1, (state, length)
    axis = displacement / length
    lateral = positions - positions[0]
    lateral -= np.outer(lateral @ axis, axis)
    cosine = [(np.trace(kin.rotation.T @ rotation) - 1.) / 2.
              for rotation in rotations]
    rotation_error = np.arccos(np.clip(cosine, -1., 1.))
    timestamps = np.asarray([row['timestamp'] for row in rows])
    dt = np.diff(timestamps)
    # record() adds an endpoint row without stepping physics; it is not a new
    # commanded velocity sample.
    stepped = dt > model.opt.timestep / 2
    speeds = np.linalg.norm(np.diff(positions, axis=0)[stepped], axis=1) / dt[stepped]
    actual = np.asarray([row['wrist_position'] for row in rows])
    actual_cosine = [(np.trace(kin.rotation.T @ np.asarray(row['wrist_rotation_world'])) - 1.) / 2.
                     for row in rows]
    actual_rotation_error = np.arccos(np.clip(actual_cosine, -1., 1.))
    return dict(maximum_lateral_bow_m=float(np.max(np.linalg.norm(lateral, axis=1))),
                maximum_rotation_error_rad=float(np.max(rotation_error)),
                maximum_commanded_speed_m_s=float(np.max(speeds)),
                commanded_travel_m=float(length),
                actual_wrist_travel_m=float(np.linalg.norm(actual[-1] - actual[0])),
                maximum_actual_rotation_error_rad=float(np.max(actual_rotation_error)),
                maximum_actual_tracking_error_m=float(np.max(np.linalg.norm(actual - positions, axis=1))),
                hand_contact_steps=sum(row['hand_segment_contacts'] > 0 for row in rows),
                physical_steps=int(np.count_nonzero(stepped)))


def test_cartesian_command_removes_joint_path_bow_and_preserves_orientation(model, scripted_camera):
    assert inspect.signature(task_module.run_task).parameters['cartesian_motion'].default is True
    runs = {mode: execute_motion(model, scripted_camera, mode) for mode in (False, True)}
    metrics = {}
    for mode, (result, trace, _) in runs.items():
        assert result['cartesian_motion'] is mode
        assert {'TRANSFER', 'PLACE', 'RELEASE', 'VERIFY'} <= {
            row['state'] for row in trace}, result
        metrics[str(mode)] = {state: phase_motion(model, trace, state)
                              for state in ('TRANSFER', 'PLACE')}
    print(json.dumps(metrics, indent=2))
    for state in ('TRANSFER', 'PLACE'):
        cartesian = metrics['True'][state]
        joint = metrics['False'][state]
        assert cartesian['maximum_lateral_bow_m'] < .001
        assert joint['maximum_lateral_bow_m'] > cartesian['maximum_lateral_bow_m'] + .001
        assert cartesian['maximum_rotation_error_rad'] < .002
        assert cartesian['maximum_commanded_speed_m_s'] < .15
        assert cartesian['actual_wrist_travel_m'] > .10
        assert cartesian['maximum_actual_rotation_error_rad'] < .04
        assert cartesian['maximum_actual_tracking_error_m'] < .01
        assert cartesian['hand_contact_steps'] >= 100
        assert cartesian['physical_steps'] >= 1000


def test_full_cartesian_control_is_independent_of_oracle_labels(model, scripted_camera, monkeypatch):
    original = CONTRACTS.PlacementEvaluator.sample
    outputs = []
    for flag in (False, True):
        def labels(self, model, data, hand, released):
            values = original(self, model, data, hand, released)
            values.update(success=flag, lifted=flag, released=flag,
                          table_support=flag, position=[99., 98., 97.], placement_error=99.)
            return values
        monkeypatch.setattr(CONTRACTS.PlacementEvaluator, 'sample', labels)
        result, trace, data = execute_motion(model, scripted_camera, True)
        outputs.append((result['transitions'], CONTRACTS.rows_without_oracle(trace),
                        data.qpos.copy(), data.qvel.copy()))
    assert outputs[0][:2] == outputs[1][:2]
    assert {'TRIAL_LIFT', 'TRANSFER', 'PLACE', 'RELEASE', 'VERIFY'} <= {
        transition['state'] for transition in outputs[0][0]}
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    np.testing.assert_array_equal(outputs[0][3], outputs[1][3])
