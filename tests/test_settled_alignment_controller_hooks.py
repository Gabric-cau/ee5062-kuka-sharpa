"""Pure timing/output and pinned real-factory checks; no physics integration."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import settled_alignment_controller_hooks as hook


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/derived_controller.py"
OPEN = {"left_thumb_CMC_FE": 1.3, "left_thumb_MCP_FE": 0., "left_thumb_IP": 0.}


def tracker(nominal=.2):
    return hook.AlignmentTracker(0., nominal, np.zeros(3), np.eye(3), np.zeros(7), OPEN)


def sample(item, timestamp, *, position=None, command=None, open_commands=None, rotation=None):
    return item.sample(timestamp, np.zeros(3) if position is None else position,
        np.eye(3) if rotation is None else rotation,
        np.zeros(7) if command is None else command,
        OPEN if open_commands is None else open_commands)


def test_continuous_window_waits_for_complete_command_open_output_and_nominal_end():
    item = tracker(nominal=.6)
    assert not sample(item, .1)["ready"]
    assert not sample(item, .2, command=np.full(7, .0001))["ready"]
    assert item.window_start_s is None
    assert not sample(item, .3)["ready"]
    assert not sample(item, .55)["ready"]
    row = sample(item, .6)
    assert row["ready"] and row["continuous_seconds"] == pytest.approx(.3)
    assert row["maximum_arm_command_error_rad"] == 0.


@pytest.mark.parametrize("bad", ["position", "speed", "command", "open"])
def test_any_failed_necessary_condition_resets_continuous_window(bad):
    item = tracker()
    sample(item, .1)
    sample(item, .2)
    kwargs = {"position": np.array([.002, 0., 0.])} if bad == "position" else (
        {"position": np.array([.0004, 0., 0.])} if bad == "speed" else (
        {"command": np.full(7, 2e-10)} if bad == "command" else
        {"open_commands": dict(OPEN, left_thumb_IP=.01)}))
    row = sample(item, .3, **kwargs)
    assert not row["ready"] and row["reasons"] and item.window_start_s is None
    assert not sample(item, .4)["ready"]


def test_orientation_is_reported_without_becoming_a_new_gate():
    item = tracker()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    sample(item, .1, rotation=rotation)
    sample(item, .2, rotation=rotation)
    row = sample(item, .45, rotation=rotation)
    assert row["ready"] and row["orientation_error_rad"] == pytest.approx(np.pi / 2)
    assert not row["orientation_error_is_a_gate"]


@pytest.mark.parametrize("actual", [{}, dict(OPEN, left_thumb_IP=.001),
    dict(OPEN, left_thumb_CMC_FE=0.), dict(OPEN, left_thumb_IP=float("nan")),
    dict(OPEN, unexpected=0.)])
def test_open_verification_checks_all_actual_output_names_and_values(actual):
    proof = hook.open_command_observation(OPEN, actual)
    assert not proof["accepted"]
    json.dumps(proof, allow_nan=False)


@pytest.mark.parametrize("timestamp", [.1, float("nan"), -.1])
def test_invalid_or_nonadvancing_clock_cannot_extend_window(timestamp):
    item = tracker()
    sample(item, .1)
    with pytest.raises(ValueError, match="actual clock"):
        sample(item, timestamp)


def _function(source, name):
    return next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name)


def _unwrapped_move(source):
    """Remove exactly this declared extension, preserving every inherited AST."""
    move = copy.deepcopy(_function(source, "move"))
    move.body = [node for node in move.body if not (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id in {"_r22_alignment", "_r22_extra_steps"} for target in node.targets)
        or isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.ALIGN")]
    loop = next(node for node in move.body if isinstance(node, ast.For))
    loop.iter = ast.parse("range(steps)", mode="eval").body
    restored = []
    for node in loop.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.ALIGN":
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.ALIGN and step >= steps":
            restored.extend(node.orelse)
        else:
            restored.append(node)
    loop.body = restored
    for node in ast.walk(loop):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)
                and ast.unparse(node.value.test) == "state == State.ALIGN and step >= steps"):
            node.value = node.value.orelse
    return move


@pytest.fixture
def transformed():
    source = SOURCE.read_text()
    namespace = {}
    exec(compile(source, "<test_pinned_arm>", "exec"), namespace)
    wrapped, rendered, telemetry = hook.transform(namespace["run_task"], source)
    return wrapped, rendered, telemetry


def test_exact_ast_recovery_proves_all_inherited_logic_and_nonalign_paths_unchanged(transformed):
    wrapped, source, telemetry = transformed
    assert ast.dump(_unwrapped_move(source)) == ast.dump(_function(SOURCE.read_text(), "move"))
    original_task, new_task = _function(SOURCE.read_text(), "run_task"), _function(source, "run_task")
    for task in (original_task, new_task):
        task.body = [node for node in task.body if not isinstance(node, ast.FunctionDef) or node.name != "move"]
    assert ast.dump(original_task) == ast.dump(new_task)
    assert telemetry.as_dict()["attempts"] == []
    assert wrapped.r22_alignment_transform["target_qpos_writes_added"] is False
    assert wrapped.r22_alignment_transform["close_motion_source_unchanged"] is True


def _executable_move(source, *, duration=.004, settle_position=True, open_hand=True):
    """Extract the real transformed loop, replacing its physical dependencies."""
    move = _function(source, "move")
    move.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node
        for node in move.body]
    callbacks = []
    class Failure(RuntimeError):
        pass
    class State:
        ALIGN = SimpleNamespace(value="ALIGN")
        CLOSE = SimpleNamespace(value="CLOSE")
        PLACE = SimpleNamespace(value="PLACE")
        RELEASE = VERIFY = RECOVER = TRIAL_LIFT = TRANSFER = object()
    data = SimpleNamespace(time=0., ctrl=np.zeros(8), xpos=np.array([[.6, 0., .5]]),
        xmat=np.eye(3).reshape(1, 9), qvel=np.zeros(7))
    target_position = np.array([.6, 0., .5 if settle_position else .6])
    scratch = SimpleNamespace(xmat=np.eye(3).reshape(1, 9))
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002), jnt_range=np.tile([-10., 10.], (7, 1)))
    kin = SimpleNamespace(body=0, model=model, origin=np.zeros(3), scratch=scratch,
        position=lambda command: target_position.copy(), solve=lambda *args, **kwargs: np.zeros(7))
    class Hand:
        closure = {"thumb": 0.}
        opposition = 1.3
        config = SimpleNamespace(thumb_opposition_rad=1.3)
        actuators = {"left_thumb_CMC_FE": 7}
        preload_ready = probe_active = False
        last_probe_visual_s = -1e10
        def write(self, unused):
            data.ctrl[7] = 1.3 if open_hand else 0.
        def after_step(self, *args):
            return None
    def step(*unused):
        data.time += .002
        callbacks.append("native_step")
    trace = []
    def record():
        trace.append({})
        return SimpleNamespace(normal=0., contacts=0)
    hand = Hand()
    hand.release_pressure_relief = lambda *args: None
    hand.latest_probe = {}
    hand.close = hand.approach_fixed_targets = lambda *args: None
    tele = hook.AlignmentTelemetry()
    namespace = dict(np=np, data=data, _phase_seconds={}, command=np.zeros(7), overload_seconds=0.,
        controller=SimpleNamespace(state="ALIGN", enter=lambda *args: None),
        _contact_guard=SimpleNamespace(begin=lambda *args: None, guide=lambda state, desired, *args: desired,
            relieve=lambda *args: None, latest={}), frame_callback=None, model=model, hand=hand,
        kin=kin, State=State, _r22_alignment_telemetry=tele, _r22_alignment_parameters=hook.PARAMETERS,
        _r22_verify_open=hook._verify_open, _r22_sample_alignment=hook._sample_alignment,
        cartesian_motion=True, _estimated=SimpleNamespace(payload_confirmed=False),
        _probe_parameters={}, limiter=None, physical_v6=False,
        robot=SimpleNamespace(_write_arm_ctrl=lambda *args: None), force_mode="contact",
        gravity=SimpleNamespace(apply=lambda *args: None), mujoco=SimpleNamespace(mj_step=step),
        record=record, trace=trace, robot_table_contacts=lambda *args: [], MotionFailure=Failure,
        interpolate=lambda start, end, fraction: start + fraction * (end - start))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[move], type_ignores=[])), "<fake_physics_move>", "exec"), namespace)
    return namespace["move"], State.ALIGN, tele, data, Failure, callbacks, duration


def test_actual_transformed_loop_settles_with_safe_extra_interpolation_and_no_extra_blend(transformed):
    move, state, tele, data, _, callbacks, duration = _executable_move(transformed[1])
    move(state, np.zeros(7), duration, 0.)
    assert .25 <= data.time <= duration + 6.
    assert len(callbacks) == round(data.time / .002)
    assert tele.attempts[0].ready and tele.attempts[0].outcome["settled"]
    assert tele.attempts[0].events[-1]["continuous_seconds"] >= .25 - 1e-9


def test_actual_transformed_loop_times_out_at_exact_six_second_extension(transformed):
    move, state, tele, data, failure, callbacks, duration = _executable_move(transformed[1], settle_position=False)
    with pytest.raises(failure, match="ALIGN_SETTLE_TIMEOUT"):
        move(state, np.zeros(7), duration, 0.)
    assert data.time == pytest.approx(duration + 6.)
    assert len(callbacks) == round((duration + 6.) / .002)
    assert tele.attempts[0].outcome["reason"] == "ALIGN_SETTLE_TIMEOUT"


def test_nonopen_output_aborts_before_any_native_integration(transformed):
    move, state, tele, data, failure, callbacks, duration = _executable_move(transformed[1], open_hand=False)
    with pytest.raises(failure, match="ALIGN_HAND_NOT_OPEN"):
        move(state, np.zeros(7), duration, 0.)
    assert data.time == 0. and callbacks == []
    assert tele.attempts[0].outcome["reason"] == "ALIGN_HAND_NOT_OPEN"


def test_changed_source_or_callable_is_rejected(transformed):
    with pytest.raises(ValueError, match="source SHA256"):
        hook.transform(transformed[0], SOURCE.read_text() + "\n")
    with pytest.raises(ValueError, match="Loaded arm callable"):
        hook.transform(lambda: None, SOURCE.read_text())


def test_real_r21_factory_compiles_without_integration_or_mutating_existing_bindings(tmp_path):
    import develop_public_ycb_provisional_preload_r21 as r21
    profile = json.loads((PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json").read_text())["ycb_can"]
    original, source, hands, estimated = r21.controller_for("ycb_can", tmp_path, [],
        SimpleNamespace(), profile, .349)
    wrapped, rendered, telemetry = hook.transform(original, source)
    assert wrapped is not original and hands == []
    for name, value in original.__globals__.items():
        assert wrapped.__globals__[name] is (wrapped if name == "run_task" else value)
    assert wrapped.r19_source_transform == original.r19_source_transform
    assert wrapped.__globals__["_r22_alignment_telemetry"] is telemetry
    assert telemetry.as_dict()["attempts"] == []
    assert hook.INHERITED_SOURCE_SHA256 == r21.hashlib.sha256(source.encode()).hexdigest()
    assert "data.qpos" not in ast.unparse(_function(rendered, "move"))
