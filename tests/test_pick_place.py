import ast
from pathlib import Path
import mujoco
import numpy as np
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene
from kuka_sharpa_grasp.pick_place import Scenario, reset_task, run_task, RobotKinematics, RobotGravityCompensation, robot_table_contacts
from kuka_sharpa_grasp.control import ContactForceReader, ForceReading, FINGERS
from kuka_sharpa_grasp.pick_place_controller import TaskController, State, candidates, interpolate, completed_success
from kuka_sharpa_grasp.pick_place_evaluator import PlacementEvaluator
from kuka_sharpa_grasp.perception import ObjectBelief
import kuka_sharpa_grasp.pick_place as task_module


@pytest.fixture
def scene(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(build_pick_place_scene(tmp_path / 'scene.xml', 'ycb_can')))
    return model, mujoco.MjData(model)


def test_separated_seeded_reset(scene):
    model, data = scene
    _, kin, _ = reset_task(model, data, Scenario(0))
    evaluator = PlacementEvaluator(model, Scenario(0).zone)
    assert np.linalg.norm(data.xpos[kin.body] - data.xpos[evaluator.body]) > .16
    assert Scenario(0).position == Scenario(0).position
    assert Scenario(0).position != Scenario(1).position
    assert data.xpos[evaluator.body, 2] == pytest.approx(.38)
    assert evaluator.clearance(model, data) > .05


def test_abort_step_preserves_contact_and_force_evidence(scene, monkeypatch):
    original_contacts = task_module.robot_table_contacts
    original_read = ContactForceReader.read

    def injected_step(timestamp):
        return .501 < timestamp < .503

    def contacts(model, data):
        if injected_step(data.time):
            return [dict(body='link1', distance_m=-.01, normal_force_n=500.)]
        return original_contacts(model, data)

    def readings(self, data):
        values = original_read(self, data)
        if injected_step(data.time):
            values['index'] = ForceReading(normal=500., contacts=1)
        return values

    monkeypatch.setattr(task_module, 'robot_table_contacts', contacts)
    monkeypatch.setattr(ContactForceReader, 'read', readings)
    result, trace = run_task(*scene, Scenario(0), max_attempts=1)
    assert result['failure'] == 'ROBOT_TABLE_PENETRATION'
    trigger = [row for row in trace if injected_step(row['timestamp'])]
    assert len(trigger) == 1
    assert trigger[0]['state'] == 'APPROACH'
    assert trigger[0]['robot_table_contacts'][0]['normal_force_n'] == 500.
    assert trigger[0]['force'] == 500.
    assert result['maximum_fingertip_total_force'] == 500.


def test_boundary():
    root = Path(__file__).resolve().parents[1] / 'src/kuka_sharpa_grasp'
    for name in ('pick_place_controller.py', 'perception.py'):
        tree = ast.parse((root / name).read_text())
        forbidden = {'qpos', 'xpos', 'body_pos', 'segmentation', 'target_body_id', 'oracle'}
        assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(tree))
        assert not any(isinstance(n, ast.ImportFrom) and 'oracle' in (n.module or '') for n in ast.walk(tree))
    tree = ast.parse((root / 'pick_place.py').read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_task')
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'qpos' for n in ast.walk(run))


def test_candidates_reject_stale_workspace_and_unknown():
    def belief(kind='ycb_can', point=(.72, 0, .42), timestamp=0):
        return ObjectBelief(kind, np.array(point), None, False, .8, timestamp,
                            np.zeros((20, 3)), np.zeros((2, 2), bool), '')
    assert len(candidates(belief(), 0)[0]) == 3
    assert candidates(belief(), 1)[1] == ['STALE_BELIEF']
    assert candidates(belief(point=(2, 0, .42)), 0)[1] == ['WORKSPACE'] * 3
    assert candidates(belief(kind='ycb_cracker_box'), 0)[1] == ['UNCALIBRATED_OBJECT']
    assert candidates(None, 0)[1] == ['NO_BELIEF']


def test_transitions_and_interpolation():
    task = TaskController()
    task.enter(State.SEARCH, 0)
    task.fail(1, 'FAILED_TRIAL_LIFT')
    task.enter(State.ABORT, 2)
    assert [t['state'] for t in task.transitions] == ['SEARCH', 'RECOVER', 'ABORT']
    start, end = np.zeros(7), np.ones(7)
    points = [interpolate(start, end, a) for a in np.linspace(0, 1, 1000)]
    assert np.max(np.diff(points, axis=0)) < .002


def test_physical_release_settle_retreat_required(scene):
    model, data = scene
    robot, _, _ = reset_task(model, data, Scenario(0))
    ev = PlacementEvaluator(model, Scenario(0).zone)
    robot.reset(object_pos=Scenario(0).zone)
    for _ in range(400):
        mujoco.mj_step(model, data)
    # Evaluator isolation fixture: use the real supported object/contact state.
    ev.clearance = lambda m, d: .2
    body = ev.body
    hand = np.array(Scenario(0).zone) + [0, 0, .3]
    assert not ev.sample(model, data, hand, False)['success']
    assert not ev.sample(model, data, hand, True)['success']
    data.time += .6
    assert ev.sample(model, data, hand, True)['success']
    ev.clearance = lambda m, d: .005
    assert not ev.sample(model, data, hand, True)['success']
    ev.clearance = lambda m, d: .2
    data.xpos[body, 0] += .1
    assert not ev.sample(model, data, hand, True)['success']
    data.xpos[body] = Scenario(0).zone
    data.cvel[body, 3] = .2
    assert not ev.sample(model, data, hand, True)['success']
    data.cvel[body] = 0
    data.xpos[body, 2] += .1
    assert not ev.sample(model, data, hand, True)['success']
    data.xpos[body] = Scenario(0).zone
    data.cvel[body, 0] = .3
    assert not ev.sample(model, data, hand, True)['success']
    data.cvel[body] = 0
    data.ncon = 0
    assert not ev.sample(model, data, hand, True)['success']


def test_physical_failure_path_and_trace(scene):
    model, data = scene
    result, trace = run_task(model, data, Scenario(0), force_failed_lift=True, max_attempts=1)
    assert not result['success']
    assert result['safe_abort_attempts'] == 1
    assert result['transitions'][-1]['state'] == 'ABORT'
    assert result['failure'] in ('FAILED_TRIAL_LIFT', 'MISSING_CONTACT', 'APPROACH_TRACKING_TIMEOUT', 'SUSTAINED_CONTACT_LOSS', 'ROBOT_TABLE_PENETRATION')
    times = [t['timestamp'] for t in trace]
    assert times == sorted(times)
    assert {'SEARCH', 'RECOVER'} <= {t['state'] for t in trace}
    commands = np.array([t['arm_command'] for t in trace])
    assert np.max(np.abs(np.diff(commands, axis=0))) < .12
    insertion = [row for row in trace if row['state'] in ('APPROACH', 'ALIGN')]
    assert all(not row['robot_table_contacts'] for row in insertion)
    assert max(row['arm_tracking_error'] for row in insertion) < .02
    assert all(row['force'] == 0 for row in insertion)
    positions = np.array([row['oracle']['position'] for row in insertion])
    assert np.max(np.linalg.norm(positions - positions[0], axis=1)) < .003
    closure = np.array([row['hand_command'] for row in trace if row['state'] == 'CLOSE'])
    assert np.max(np.abs(np.diff(closure, axis=0))) < .01


def test_physical_trial_outcome_is_honest(scene):
    result, trace = run_task(*scene, Scenario(1), max_attempts=1)
    if result['success']:
        assert {'APPROACH', 'CLOSE', 'TRIAL_LIFT', 'TRANSFER', 'PLACE', 'RELEASE', 'VERIFY'} <= {t['state'] for t in trace}
        assert trace[-1]['oracle']['success']
        assert trace[-1]['oracle']['released']
    else:
        assert result['failure'] and result['safe_abort_attempts'] == 1


def test_success_requires_every_stage_in_order():
    sequence = [dict(state=s.value) for s in (State.APPROACH, State.ALIGN, State.CLOSE,
                State.TRIAL_LIFT, State.TRANSFER, State.PLACE, State.RELEASE, State.VERIFY)]
    assert completed_success(sequence, True, None)
    assert not completed_success(sequence, False, None)
    assert not completed_success(sequence, True, 'DROP')
    for index in range(len(sequence)):
        assert not completed_success(sequence[:index] + sequence[index+1:], True, None)


def test_failed_lift_gate_recovery_abort():
    task = TaskController()
    task.enter(State.TRIAL_LIFT, 4)
    assert not task.accept_lift(5, contacts=3, lifted=False)
    assert task.state == 'RECOVER'
    task.enter(State.ABORT, 6, task.failure)
    assert task.transitions[-1]['reason'] == 'FAILED_TRIAL_LIFT'


def test_evaluator_labels_cannot_change_online_behavior(scene, monkeypatch):
    model, data = scene
    original = PlacementEvaluator.sample
    # Corrupt labels only; the real physical/sensor path must remain identical.
    outputs = []
    for flag in (False, True):
        def changed(self, model, data, hand, released):
            result = original(self, model, data, hand, released)
            result.update(lifted=flag, success=flag, position=[99, 99, 99], placement_error=99.)
            return result
        monkeypatch.setattr(PlacementEvaluator, 'sample', changed)
        result, trace = run_task(model, data, Scenario(0), max_attempts=1)
        outputs.append((result['transitions'],
                        [{k: v for k, v in row.items() if k != 'oracle'} for row in trace],
                        data.qpos.copy(), data.qvel.copy()))
    assert outputs[0][:2] == outputs[1][:2]
    assert {'TRIAL_LIFT', 'TRANSFER', 'PLACE', 'RELEASE', 'VERIFY'} <= {t['state'] for t in outputs[0][0]}
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    np.testing.assert_array_equal(outputs[0][3], outputs[1][3])


def test_live_positions_only_changed_by_physics_after_reset(scene, monkeypatch):
    model, data = scene
    original = mujoco.mj_step
    last = None
    calls = 0
    def checked_step(m, d):
        nonlocal last, calls
        if d is data:
            if last is not None:
                np.testing.assert_array_equal(d.qpos, last)
            original(m, d)
            last = d.qpos.copy()
            calls += 1
        else:
            original(m, d)
    monkeypatch.setattr(mujoco, 'mj_step', checked_step)
    result, _ = run_task(model, data, Scenario(0))
    assert calls > 1000
    assert result['success'] and result['failure'] is None
    assert result['final_oracle']['success']
    assert result['safe_abort_attempts'] == 0 and result['successful_recoveries'] == 0


def test_corrected_workcell_settled_start_and_gravity(scene):
    model, data = scene
    robot, kin, command = reset_task(model, data, Scenario(0))
    assert data.ncon == 0
    gravity = RobotGravityCompensation(model)
    start = data.xpos[kin.body].copy()
    for _ in range(500):
        gravity.apply(data)
        mujoco.mj_step(model, data)
        assert not robot_table_contacts(model, data)
        assert np.max(np.abs(data.qfrc_applied[:7])) <= 80.
    assert np.linalg.norm(data.xpos[kin.body] - start) < .001
    assert robot.reader.total(robot.reader.read(data)).contacts == 0
    assert PlacementEvaluator(model, Scenario(0).zone).clearance(model, data) > .08
    assert kin.approach_axis == pytest.approx([.7071043, 0., -.70710926])


def test_gravity_ignores_target_state_and_does_not_actuate_target(scene):
    model, data = scene
    _, kin, _ = reset_task(model, data, Scenario(0))
    gravity = RobotGravityCompensation(model)
    before = gravity.apply(data)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    adr = model.jnt_qposadr[joint]
    data.qpos[adr:adr+3] = [9, 8, 7]
    data.qvel[model.jnt_dofadr[joint]:] = 10.
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(before, gravity.apply(data))
    assert np.all(data.qfrc_applied[7:] == 0)


def test_strict_physical_placement_without_clearance_mocks(scene):
    model, data = scene
    scenario = Scenario(0)
    robot, kin, _ = reset_task(model, data, scenario)
    # Physical evaluator fixture: place at reset in zone, then settle under gravity.
    robot.reset(object_pos=scenario.zone)
    data.qpos[:7] = kin.solve(np.array([0., 0., .22]))
    robot._write_arm_ctrl(tuple(data.qpos[:7]))
    mujoco.mj_forward(model, data)
    gravity = RobotGravityCompensation(model)
    evaluator = PlacementEvaluator(model, scenario.zone)
    for _ in range(1000):
        gravity.apply(data)
        mujoco.mj_step(model, data)
        labels = evaluator.sample(model, data, data.xpos[kin.body], True)
    assert labels['success'] and labels['table_support'] and labels['disengaged']
    assert labels['hand_clearance_m'] > .10
    assert not evaluator.sample(model, data, data.xpos[kin.body], False)['success']
    # No table load: an unsupported near-zone object cannot pass even after time.
    data.ncon = 0
    data.time += 1.
    assert not evaluator.sample(model, data, data.xpos[kin.body], True)['success']
