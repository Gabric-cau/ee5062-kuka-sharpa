"""CPU-only routing and phase contracts for the V4 integration.

Scripted camera beliefs isolate control decisions from rendering/perception
accuracy. The robot, hand, target contacts and joint movement remain actual
MuJoCo physics. Real RGB-D success belongs to the separate Dell evaluation.
"""
from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import kuka_sharpa_grasp.pick_place as task_module
from kuka_sharpa_grasp.perception import ObjectBelief
from kuka_sharpa_grasp.pick_place_evaluator import PlacementEvaluator
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    scene = build_pick_place_scene(tmp_path_factory.mktemp('v4_contract') / 'scene.xml', 'ycb_can')
    return mujoco.MjModel.from_xml_path(str(scene))


@pytest.fixture
def scripted_camera(monkeypatch):
    """Return a per-run sensor fixture that reads simulation time only."""
    def install(scenario, held_xy_delta=(0., 0.)):
        counter = {'polls': 0, 'held_xy_delta': list(held_xy_delta)}

        class Camera:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def poll(self, data):
                counter['polls'] += 1
                return SimpleNamespace(timestamp=float(data.time), number=counter['polls'])

        def estimate(observation, kind):
            if observation.number == 1:
                # Fixed SEARCH RGB-D centroid from the development seed-0
                # camera recording; no live target position is read here.
                centroid = np.array([.7271472421471974, -.017650463591662276,
                                     .4597746256041916])
            elif observation.number == 2:
                # The held development observation has moved relative to the
                # initial centroid. Replaying it exercises the new correction.
                centroid = np.array([.7108519995654575, -.02265295194028769,
                                     .6066593969012964])
                centroid[:2] += held_xy_delta
            else:
                centroid = np.array(scenario.zone, dtype=float)
                # Invert the known-can correction for the scripted final frame.
                centroid[:2] += [.005, -.008]
            points = np.tile(centroid, (24, 1))
            points[:, 2] = np.linspace(centroid[2] + .02, centroid[2] + .115, len(points))
            return ObjectBelief(kind, centroid, None, False, .8, observation.timestamp,
                                points, np.ones((2, 2), dtype=bool), 'scripted_camera_fixture')

        monkeypatch.setattr(task_module, 'CameraObserver', Camera)
        monkeypatch.setattr(task_module, 'estimate_object', estimate)
        return counter
    return install


def require_v4_interface():
    parameters = inspect.signature(task_module.run_task).parameters
    assert 'placement_retention' in parameters, 'V4 integration parameter has not landed'
    assert parameters['visual_transfer'].default is True


def execute(model, camera_fixture, *, held_xy_delta=(0., 0.), visual_transfer=True):
    scenario = task_module.Scenario(0)
    sensor = camera_fixture(scenario, held_xy_delta)
    data = mujoco.MjData(model)
    result, trace = task_module.run_task(
        model, data, scenario, max_attempts=1, visual_transfer=visual_transfer,
        placement_retention=False, support_load_sensor=False,
    )
    assert sensor['polls'] >= 2, result
    return result, trace, data


def rows_without_oracle(trace):
    return [{key: value for key, value in row.items() if key != 'oracle'} for row in trace]


def test_v4_keeps_sensor_contract_and_freezes_all_22_commands_during_place(model, scripted_camera):
    require_v4_interface()
    result, trace, _ = execute(model, scripted_camera)
    assert set(result['sensor_contract']) == {
        'wrist_rgbd', 'robot_proprioception', 'simulated_finger_segment_contact_wrenches',
    }
    assert result['visual_transfer'] is True
    assert result['placement_retention'] is False
    assert result['support_load_sensor'] is False
    place = np.array([row['hand_command'] for row in trace if row['state'] == 'PLACE'])
    assert place.shape[0] >= 100, result
    assert place.shape[1] == 22
    assert np.max(np.ptp(place, axis=0)) == 0.
    transfer = [row for row in trace if row['state'] == 'TRANSFER']
    np.testing.assert_array_equal(place[0], transfer[-1]['hand_command'])
    release = np.array([row['hand_command'] for row in trace if row['state'] == 'RELEASE'])
    assert release.shape[0] >= 100
    assert np.max(np.ptp(release, axis=0)) > .1


def test_oracle_counterfactual_cannot_change_v4_sensor_control(model, scripted_camera, monkeypatch):
    require_v4_interface()
    original = PlacementEvaluator.sample
    outputs = []
    for flag in (False, True):
        def labels(self, model, data, hand, released):
            values = original(self, model, data, hand, released)
            values.update(success=flag, lifted=flag, released=flag,
                          table_support=flag, position=[99., 98., 97.], placement_error=99.)
            return values
        monkeypatch.setattr(PlacementEvaluator, 'sample', labels)
        result, trace, data = execute(model, scripted_camera)
        outputs.append((result['transitions'], rows_without_oracle(trace),
                        data.qpos.copy(), data.qvel.copy()))
    assert outputs[0][:2] == outputs[1][:2]
    assert {'TRIAL_LIFT', 'TRANSFER', 'PLACE', 'RELEASE', 'VERIFY'} <= {
        transition['state'] for transition in outputs[0][0]}
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    np.testing.assert_array_equal(outputs[0][3], outputs[1][3])


@pytest.mark.parametrize('visual_transfer', [False, True])
def test_held_rgbd_counterfactual_changes_only_enabled_transfer(model, scripted_camera, visual_transfer):
    require_v4_interface()
    outputs = [execute(model, scripted_camera, held_xy_delta=shift,
                       visual_transfer=visual_transfer)
               for shift in ((0., 0.), (.012, 0.))]
    prefixes = [[row for row in trace if row['state'] in {
        'SEARCH', 'APPROACH', 'ALIGN', 'CLOSE', 'TRIAL_LIFT'}]
                for _, trace, _ in outputs]
    assert prefixes[0] == prefixes[1]
    transfer = [np.array([row['arm_command'] for row in trace if row['state'] == 'TRANSFER'])
                for _, trace, _ in outputs]
    assert min(len(commands) for commands in transfer) > 100
    common = min(len(commands) for commands in transfer)
    if visual_transfer:
        assert np.max(np.abs(transfer[0][:common] - transfer[1][:common])) > 1e-4
    else:
        np.testing.assert_array_equal(transfer[0], transfer[1])
        assert rows_without_oracle(outputs[0][1]) == rows_without_oracle(outputs[1][1])
