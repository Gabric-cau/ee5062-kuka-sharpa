"""Pinned R22 arm extensions with fake native dependencies, never live integration."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import actual_wrist_probe_controller_hooks as hook
import settled_alignment_controller_hooks as alignment


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/public_ycb_settled_alignment_20261022_r22_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/derived_controller.py"


def _function(source, name):
    return next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.fixture
def transformed():
    source = SOURCE.read_text()
    namespace = {"sentinel": object()}
    exec(compile(source, "<test_executed_r22_arm>", "exec"), namespace)
    original = namespace["run_task"]
    original.inherited_metadata = object()
    wrapped, rendered, telemetry = hook.transform(original, source)
    return original, wrapped, rendered, telemetry


def _restore_r22(source):
    tree = ast.parse(source)
    task = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    move = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "move")
    restored = []
    for node in move.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_r23_probe":
            continue
        if isinstance(node, ast.If) and any(isinstance(item, ast.Name) and item.id == "_r23_probe_telemetry"
                for item in ast.walk(node)):
            continue
        if isinstance(node, ast.Try) and node.handlers[0].name == "_r23_motion_error":
            restored.extend(node.body)
        else:
            restored.append(node)
    move.body = [node for node in restored if not (
        isinstance(node, ast.If) and ast.unparse(node.test) == "_r23_probe is not None and _r23_probe.outcome is None")]
    loop = next(node for node in move.body if isinstance(node, ast.For))
    segment = next(node for node in ast.walk(loop) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cartesian_motion and step % segment_steps == 0")
    branch = next(node for node in segment.body if isinstance(node, ast.If)
        and "_r23_probe" in ast.unparse(node.test))
    segment.body[segment.body.index(branch)] = branch.orelse[0]
    loop.body = [node for node in loop.body if not (
        isinstance(node, ast.If) and any(isinstance(item, ast.Name) and item.id == "_r23_probe"
            for item in ast.walk(node.test)))]
    for node in ast.walk(task):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_r23_probe_telemetry.confirm_probe":
            node.func = ast.parse("hand.confirm_probe", mode="eval").body
            node.args.pop(0)
    return tree


def test_removing_only_declared_extensions_recovers_the_entire_executed_r22_arm(transformed):
    original, wrapped, source, telemetry = transformed
    assert ast.dump(_restore_r22(source)) == ast.dump(ast.parse(SOURCE.read_text()))
    for name, value in original.__globals__.items():
        assert wrapped.__globals__[name] is (wrapped if name == "run_task" else value)
    assert wrapped.inherited_metadata is original.inherited_metadata
    assert telemetry.as_dict()["attempts"] == []
    assert wrapped.r23_actual_wrist_probe_transform["target_qpos_writes_added"] is False
    assert "data.qpos" not in source
    ast.parse(source, feature_version=(3, 10))
    ast.parse(Path(hook.__file__).read_text(), feature_version=(3, 10))


def test_uses_the_original_r12_function_and_bounded_three_second_path():
    assert hook.bounded_probe_reference is hook.reference.bounded_probe_reference
    assert hashlib.sha256(Path(hook.reference.__file__).read_bytes()).hexdigest() == hook.REFERENCE_SOURCE_SHA256
    start = np.array([.6, .02, .6])
    for elapsed, height in [(0., 0.), (1.5, .0025), (3., .005), (6., .005)]:
        actual = start + [0., 0., height]
        row = hook.bounded_probe_reference(start, actual, elapsed)
        np.testing.assert_allclose(row["desired_actual_path_world_m"], actual)
        np.testing.assert_allclose(row["cartesian_ik_target_world_m"], actual)
    actual = start + [.01, -.02, -.03]
    row = hook.bounded_probe_reference(start, actual, 6.)
    assert np.linalg.norm(row["bounded_reference_correction_world_m"]) == pytest.approx(.002)
    near = start + [.0001, -.0002, .0047]
    row = hook.bounded_probe_reference(start, near, 3.)
    np.testing.assert_allclose(row["bounded_reference_correction_world_m"], [-.0002, .0004, .0006])


def test_ik_target_depends_on_wrist_proprioception_and_ignores_other_body_state():
    kin = SimpleNamespace(body=0)
    failure = type("Failure", (RuntimeError,), {})
    def target(wrist, other_body):
        tracker = hook.ProbeTracker(0., 0., [.6, 0., .6])
        data = SimpleNamespace(time=0., xpos=np.array([wrist, other_body]))
        return tracker.target(data, kin, .02, failure)
    first = target([.6, 0., .6], [10., 20., 30.])
    second = target([.6, 0., .6], [-10., -20., -30.])
    np.testing.assert_array_equal(first, second)
    shifted_wrist = target([.6001, 0., .6], [10., 20., 30.])
    assert shifted_wrist[0] - first[0] == pytest.approx(-.0002)
    body = ast.parse(Path(hook.__file__).read_text())
    method = next(node for node in ast.walk(body) if isinstance(node, ast.FunctionDef) and node.name == "target")
    data_attributes = {node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "data"}
    assert data_attributes == {"time", "xpos"}
    assert not any(isinstance(node, ast.Name) and node.id in {"_estimated", "target_body", "oracle"}
        for node in ast.walk(method))


def _executable_move(source, *, state="TRIAL_LIFT", active=True, payload=False,
        confirmation=None, fail_after_step=False, penetrate=False, null_target=False):
    move = _function(source, "move")
    move.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node
        for node in move.body]
    class Failure(RuntimeError):
        pass
    class State:
        TRIAL_LIFT = SimpleNamespace(value="TRIAL_LIFT")
        TRANSFER = SimpleNamespace(value="TRANSFER")
        ALIGN = SimpleNamespace(value="ALIGN")
        CLOSE = SimpleNamespace(value="CLOSE")
        PLACE = SimpleNamespace(value="PLACE")
        RELEASE = SimpleNamespace(value="RELEASE")
        VERIFY = SimpleNamespace(value="VERIFY")
        RECOVER = SimpleNamespace(value="RECOVER")
    state = getattr(State, state)
    data = SimpleNamespace(time=0., ctrl=np.zeros(8), xpos=np.array([[.6, 0., .6]]),
        xmat=np.eye(3).reshape(1, 9), qvel=np.zeros(7))
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002), jnt_range=np.tile([-10., 10.], (7, 1)))
    ik_calls, native_calls = [], []
    def solve(position, **kwargs):
        ik_calls.append((data.time, np.array(position), kwargs))
        return np.r_[float(position[2] - .6), np.zeros(6)]
    kin = SimpleNamespace(body=0, model=model, origin=np.zeros(3), rotation=np.eye(3),
        scratch=SimpleNamespace(xmat=np.eye(3).reshape(1, 9)),
        position=lambda command: np.array([.6, 0., .6 + command[0]]), solve=solve)
    estimated = SimpleNamespace(payload_confirmed=payload, last_close_observation_s=0.)
    class Hand:
        closure = {"thumb": 0.}
        opposition = 1.3
        config = SimpleNamespace(thumb_opposition_rad=1.3)
        actuators = {"left_thumb_CMC_FE": 7}
        preload_ready = False
        last_probe_visual_s = -1e10
        latest_probe = {}
        probe_start_s = 0.
        probe_start_world_m = np.array([.6, 0., .6])
        probe_confirmed_seconds = 0.
        probe_active = active
        aborts = []
        def write(self, unused):
            data.ctrl[7] = 1.3
        def after_step(self, *args):
            self.probe_confirmed_seconds = 0. if confirmation is None else confirmation(data.time)
            return "PROBE_TEST_FAILURE" if fail_after_step else None
        def abort_probe(self, reason):
            self.aborts.append(reason)
            self.probe_active = False
    hand = Hand()
    hand.release_pressure_relief = hand.close = hand.retain = hand.approach_fixed_targets = lambda *args: None
    trace = []
    def record():
        trace.append({})
        return SimpleNamespace(normal=0., contacts=1)
    def native_step(*unused):
        data.time += .002
        native_calls.append(data.time)
    tele, align_tele = hook.ProbeTelemetry(), alignment.AlignmentTelemetry()
    namespace = dict(np=np, data=data, model=model, State=State, hand=hand, kin=kin,
        command=np.zeros(7), overload_seconds=0., _phase_seconds={},
        _estimated=estimated, _probe_parameters={"confirmation_seconds": .25, "visual_refresh_seconds": .35},
        _r23_probe_parameters=dict(hook.PARAMETERS), _r23_probe_telemetry=tele,
        _r22_alignment_telemetry=align_tele, _r22_alignment_parameters=alignment.PARAMETERS,
        _r22_verify_open=alignment._verify_open, _r22_sample_alignment=alignment._sample_alignment,
        controller=SimpleNamespace(state=state.value, enter=lambda *args: None),
        _contact_guard=SimpleNamespace(begin=lambda *args: None,
            guide=lambda unused, desired, *args: desired, relieve=lambda *args: None, latest={}),
        frame_callback=None, cartesian_motion=True, limiter=None, physical_v6=False,
        robot=SimpleNamespace(_write_arm_ctrl=lambda *args: None), force_mode="contact",
        gravity=SimpleNamespace(apply=lambda *args: None), mujoco=SimpleNamespace(mj_step=native_step),
        hand_reader=SimpleNamespace(read=lambda unused: {}),
        record=record, trace=trace, observe=lambda: None,
        robot_table_contacts=lambda *args: [{"distance_m": -.003}] if penetrate else [],
        MotionFailure=Failure, interpolate=lambda start, end, fraction: start + fraction * (end - start))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[move], type_ignores=[])), "<fake_native_probe_move>", "exec"), namespace)
    target = None if null_target else np.r_[0. if state == State.ALIGN else .005, np.zeros(6)]
    return SimpleNamespace(move=namespace["move"], state=state, target=target, data=data,
        hand=hand, kin=kin, telemetry=tele, alignment=align_tele, trace=trace,
        failure=Failure, native_calls=native_calls, ik_calls=ik_calls)


def test_actual_transformed_loop_has_exact_six_second_budget_and_twenty_ms_ik(transformed):
    fake = _executable_move(transformed[2])
    fake.move(fake.state, fake.target, .1, .85)
    assert fake.data.time == pytest.approx(6.)
    assert len(fake.native_calls) == 3000 and len(fake.ik_calls) == 300
    item = fake.telemetry.attempts[0]
    assert len(item.samples) == 3000 and len(item.ik_samples) == 300
    assert item.ik_samples[0]["path_elapsed_s"] == pytest.approx(.02)
    assert item.ik_samples[-1]["path_elapsed_s"] == pytest.approx(6.)
    np.testing.assert_allclose(item.ik_samples[-1]["desired_actual_path_world_m"], [.6, 0., .605])
    np.testing.assert_allclose(item.ik_samples[-1]["cartesian_ik_target_world_m"], [.6, 0., .607])
    np.testing.assert_allclose(item.samples[-1]["current_actual_wrist_world_m"], [.6, 0., .6])
    assert item.outcome["arm_exit_reason"] == "MAXIMUM_PROBE_DURATION_REACHED"
    assert item.outcome["original_confirmation_accepted"] is None
    assert all(set(kwargs) == {"seed", "tolerance"} for _, _, kwargs in fake.ik_calls)
    np.testing.assert_array_equal(fake.kin.rotation, np.eye(3))
    json.dumps(fake.telemetry.as_dict(), allow_nan=False)


def test_original_confirmation_window_can_end_motion_early_but_does_not_confirm_payload(transformed):
    fake = _executable_move(transformed[2], confirmation=lambda now: .25 if now >= .3 - 1e-10 else 0.)
    fake.move(fake.state, fake.target, .1, .85)
    assert fake.data.time == pytest.approx(.3)
    assert fake.hand.probe_active
    assert fake.telemetry.attempts[0].outcome["final_reason"] == "ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED"
    assert fake.trace[-2]["actual_wrist_probe_termination"] == "ORIGINAL_CONTINUOUS_CONFIRMATION_WINDOW_REACHED"


def test_below_original_confirmation_window_cannot_end_probe_early(transformed):
    fake = _executable_move(transformed[2], confirmation=lambda now: .249)
    fake.move(fake.state, fake.target, .1, .85)
    assert fake.data.time == pytest.approx(6.)


@pytest.mark.parametrize("state,active,payload", [
    ("TRIAL_LIFT", False, True), ("TRIAL_LIFT", True, True),
    ("TRANSFER", True, False), ("CLOSE", False, False)])
def test_nonprobe_and_confirmed_motion_keep_original_clocks_and_no_probe_telemetry(transformed, state, active, payload):
    fake = _executable_move(transformed[2], state=state, active=active, payload=payload)
    fake.move(fake.state, fake.target, .02, .85)
    assert fake.data.time == pytest.approx(.02)
    assert len(fake.native_calls) == 10
    assert fake.telemetry.attempts == []
    assert all("actual_wrist_probe" not in row for row in fake.trace)


def test_r22_settled_open_alignment_still_runs_with_its_own_telemetry(transformed):
    fake = _executable_move(transformed[2], state="ALIGN", active=False)
    fake.move(fake.state, fake.target, .004, 0.)
    assert .25 <= fake.data.time <= 6.004
    assert fake.alignment.attempts[0].ready
    assert fake.telemetry.attempts == []


@pytest.mark.parametrize("case,reason", [("after_step", "PROBE_TEST_FAILURE"),
    ("penetration", "ROBOT_TABLE_PENETRATION"), ("null_target", "MOTION_TARGET_IK")])
def test_original_failures_precede_early_exit_and_always_finish_telemetry(transformed, case, reason):
    fake = _executable_move(transformed[2], confirmation=lambda now: .25,
        fail_after_step=case == "after_step", penetrate=case == "penetration", null_target=case == "null_target")
    with pytest.raises(fake.failure, match=reason):
        fake.move(fake.state, fake.target, .1, .85)
    assert fake.telemetry.attempts[0].outcome["final_reason"] == reason
    assert fake.telemetry.attempts[0].outcome["motion_failure"] == reason
    assert not any("actual_wrist_probe_termination" in row for row in fake.trace)
    if case == "null_target":
        assert not fake.native_calls and not fake.ik_calls


@pytest.mark.parametrize("accepted", [False, True])
def test_original_decisive_confirmation_is_called_once_and_its_result_controls_outcome(transformed, accepted):
    fake = _executable_move(transformed[2])
    fake.move(fake.state, fake.target, .1, .85)
    calls = []
    def original_confirm(data):
        calls.append(data.time)
        fake.hand.probe_active = not accepted
        return accepted
    fake.hand.confirm_probe = original_confirm
    assert fake.telemetry.confirm_probe(fake.hand, fake.data) is accepted
    assert len(calls) == 1
    outcome = fake.telemetry.attempts[0].outcome
    assert outcome["original_confirmation_attempted"]
    assert outcome["original_confirmation_accepted"] is accepted
    assert outcome["final_reason"] == ("ORIGINAL_DECISIVE_CONFIRMATION_ACCEPTED" if accepted else "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED")


@pytest.mark.parametrize("bad", ["start_clock", "sample_clock", "position", "confirmation_clock"])
def test_invalid_or_nonadvancing_measurements_cannot_extend_probe(bad):
    failure = type("Failure", (RuntimeError,), {})
    if bad == "start_clock":
        with pytest.raises(ValueError, match="admission clock"):
            hook.ProbeTracker(.1, 0., np.zeros(3))
        return
    tracker = hook.ProbeTracker(0., 0., np.zeros(3))
    data = SimpleNamespace(time=.002, xpos=np.zeros((1, 3)))
    hand = SimpleNamespace(probe_active=True, probe_confirmed_seconds=0.)
    kin = SimpleNamespace(body=0)
    tracker.sample(data, kin, hand, failure)
    if bad == "sample_clock":
        pass
    elif bad == "position":
        data.time = .004
        data.xpos[0, 0] = float("nan")
    else:
        data.time = .004
        hand.probe_confirmed_seconds = float("nan")
    with pytest.raises(failure, match="INVALID_MEASUREMENT"):
        tracker.sample(data, kin, hand, failure)
    tracker.finish(data, hand, "MOTION_EXCEPTION", motion_error="ACTUAL_WRIST_PROBE_INVALID_MEASUREMENT")
    json.dumps(tracker.as_dict(), allow_nan=False)


def test_changed_source_or_callable_is_rejected(transformed):
    with pytest.raises(ValueError, match="source SHA256"):
        hook.transform(transformed[0], SOURCE.read_text() + "\n")
    with pytest.raises(ValueError, match="Loaded arm callable"):
        hook.transform(lambda: None, SOURCE.read_text())


def test_real_r22_factory_retains_all_globals_hand_and_alignment_metadata(tmp_path):
    import develop_public_ycb_settled_alignment_r22 as r22
    profile = json.loads((PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json").read_text())["ycb_can"]
    sentinel = SimpleNamespace()
    original, source, hands, estimated = r22.controller_for("ycb_can", tmp_path, [], sentinel, profile, .349)
    executed_source = SOURCE.read_text()
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(executed_source))
    namespace = dict(original.__globals__)
    exec(compile(executed_source, "<executed_r22_with_fresh_factory_bindings>", "exec"), namespace)
    rebound = namespace["run_task"]
    assert hook._code_semantics(rebound.__code__) == hook._code_semantics(original.__code__)
    rebound.__dict__.update(original.__dict__)
    original = rebound
    wrapped, rendered, telemetry = hook.transform(original, executed_source)
    assert hashlib.sha256(executed_source.encode()).hexdigest() == hook.INHERITED_SOURCE_SHA256
    assert hands == [] and wrapped.__globals__["_estimated"] is estimated
    assert wrapped.__globals__["HandContactReader"](None) is sentinel
    for name, value in original.__globals__.items():
        assert wrapped.__globals__[name] is (wrapped if name == "run_task" else value)
    for name, value in original.__dict__.items():
        assert wrapped.__dict__[name] is value
    assert wrapped.__globals__["HandGraspController"] is original.__globals__["HandGraspController"]
    assert wrapped.alignment_settling_telemetry is original.alignment_settling_telemetry
    assert wrapped.__globals__["_r23_probe_telemetry"] is telemetry
    assert ast.dump(_restore_r22(rendered)) == ast.dump(ast.parse(executed_source))
