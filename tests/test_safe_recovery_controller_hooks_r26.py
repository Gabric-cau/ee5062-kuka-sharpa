"""Recovery sequencing and sensor gates; no physical safety/success claim."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import safe_recovery_controller_hooks_r26 as hook
import actual_wrist_probe_controller_hooks as probe


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / ('results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/'
    'ycb_can_bias_budget_r17_seed3414/derived_controller.py')


class Failure(RuntimeError):
    pass


@pytest.fixture
def scene():
    phase_events = []
    hand = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)), phase='TRIAL_LIFT',
        joint_references={'loaded': .4}, inhibit_admission=lambda: phase_events.append('INHIBIT'))
    def phase_begin(phase, data):
        hand.phase = phase
        phase_events.append(phase)
    hand.phase_begin = phase_begin
    hand.reader = SimpleNamespace(read=lambda data: {'middle': SimpleNamespace(normal=4.)})
    data = SimpleNamespace(time=0., xpos=np.array([[.6, .02, .6]]))
    targets = []
    def solve(endpoint, **kwargs):
        targets.append(np.asarray(endpoint).copy())
        return np.asarray(endpoint).copy()
    kin = SimpleNamespace(body=0, origin=np.zeros(3), solve=solve)
    gate = SimpleNamespace(fresh_confirmation=lambda now: False)
    return data, kin, hand, gate, phase_events, targets


def test_brake_holds_an_anchor_and_does_not_enter_inherited_opening_phase(scene):
    data, kin, hand, gate, phases, targets = scene
    telemetry = hook.RecoveryTelemetry()
    telemetry.begin('BRAKE', data, kin, hand, gate, Failure)
    assert phases == ['R26_RECOVERY_HOLD', 'INHIBIT']
    assert hand.joint_references == {'loaded': .4}
    anchor = data.xpos[0].copy()
    data.xpos[0] += [.001, .002, .003]
    telemetry.target(data, kin, np.zeros(3), Failure)
    np.testing.assert_array_equal(targets[-1], anchor)
    for index in range(75):
        data.time = (index + 1) * .002
        telemetry.sample(data, kin, hand, gate, Failure)
    assert not telemetry.current['ready']  # First displacement reset the window.
    data.time += .002
    telemetry.sample(data, kin, hand, gate, Failure)
    assert telemetry.current['ready']
    telemetry.finish(Failure)
    json.dumps(telemetry.as_dict(), allow_nan=False)


def test_continuous_speed_window_resets_and_timeout_remains_failure(scene):
    data, kin, hand, gate, _, _ = scene
    telemetry = hook.RecoveryTelemetry()
    telemetry.begin('BRAKE', data, kin, hand, gate, Failure)
    for index in range(10):
        data.time += .002
        telemetry.sample(data, kin, hand, gate, Failure)
    assert telemetry.current['settled_seconds'] == pytest.approx(.02)
    data.time += .002
    data.xpos[0, 0] += .0001
    telemetry.sample(data, kin, hand, gate, Failure)
    assert telemetry.current['settled_seconds'] == 0.
    with pytest.raises(Failure, match='R26_WRIST_NOT_SETTLED'):
        telemetry.finish(Failure)


def test_lower_is_vertical_rate_limited_bounded_and_requires_original_support(scene):
    data, kin, hand, gate, _, targets = scene
    telemetry = hook.RecoveryTelemetry()
    anchor = data.xpos[0].copy()
    telemetry.begin('LOWER', data, kin, hand, gate, Failure)
    for clock in (.02, 1., 20.):
        data.time = clock
        telemetry.target(data, kin, np.zeros(3), Failure)
        np.testing.assert_array_equal(targets[-1][:2], anchor[:2])
        assert targets[-1][2] == pytest.approx(anchor[2] - min(.05, .004 * clock))
    with pytest.raises(Failure, match='R26_TABLE_SUPPORT_NOT_CONFIRMED'):
        telemetry.finish(Failure)
    telemetry.current['ready'] = True
    telemetry.finish(Failure)


def test_release_rejects_missing_unconfirmed_and_lost_support(scene):
    data, kin, hand, gate, _, _ = scene
    telemetry = hook.RecoveryTelemetry()
    for invalid_gate in (None, gate):
        with pytest.raises(Failure, match='R26_RELEASE_REQUIRES_ORIGINAL_TABLE_SUPPORT'):
            telemetry.begin('RELEASE', data, kin, hand, invalid_gate, Failure)
    gate.fresh_confirmation = lambda now: now == 0.
    telemetry.begin('RELEASE', data, kin, hand, gate, Failure)
    data.time = .002
    with pytest.raises(Failure, match='R26_TABLE_SUPPORT_LOST_DURING_RELEASE'):
        telemetry.sample(data, kin, hand, gate, Failure)


def test_force_abort_retains_the_violating_sample_and_is_not_a_clamp(scene):
    data, kin, hand, gate, _, _ = scene
    telemetry = hook.RecoveryTelemetry()
    telemetry.begin('BRAKE', data, kin, hand, gate, Failure)
    hand.reader.read = lambda data: {'middle': SimpleNamespace(normal=12.001)}
    data.time = .002
    with pytest.raises(Failure, match='R26_HAND_FORCE_LIMIT'):
        telemetry.sample(data, kin, hand, gate, Failure)
    assert telemetry.current['samples'][-1]['channel_pressures_n']['middle'] == 12.001
    assert hand.joint_references == {'loaded': .4}


def test_invalid_mode_step_and_noncontiguous_sample_are_rejected(scene):
    data, kin, hand, gate, _, _ = scene
    telemetry = hook.RecoveryTelemetry()
    with pytest.raises(Failure, match='INVALID_RECOVERY_MODE'):
        telemetry.begin('OPEN_ANYWAY', data, kin, hand, gate, Failure)
    hand.model.opt.timestep = .004
    with pytest.raises(Failure, match='INVALID_NATIVE_TIMESTEP'):
        telemetry.begin('BRAKE', data, kin, hand, gate, Failure)
    hand.model.opt.timestep = .002
    telemetry.begin('BRAKE', data, kin, hand, gate, Failure)
    data.time = .004
    with pytest.raises(Failure, match='NONCONTIGUOUS_NATIVE_SAMPLE'):
        telemetry.sample(data, kin, hand, gate, Failure)


def test_source_is_exact_compiled_source_and_normal_checks_are_retained():
    source = SOURCE.read_text()
    namespace = {'_r23_probe_telemetry': probe.ProbeTelemetry(), '_r23_probe_parameters': dict(probe.PARAMETERS)}
    exec(compile(source, '<r26_frozen_source>', 'exec'), namespace)
    original = namespace['run_task']
    wrapped, rendered, telemetry = hook.transform(original, source)
    compiled = {}
    exec(compile(rendered, '<r26_returned_source>', 'exec'), compiled)
    assert hook.inherited._code_semantics(wrapped.__code__) == hook.inherited._code_semantics(compiled['run_task'].__code__)
    assert hashlib.sha256(rendered.encode()).hexdigest() == wrapped.r26_safe_recovery_transform['compiled_source_sha256']
    assert wrapped.safe_recovery_telemetry is telemetry
    assert wrapped.actual_wrist_probe_telemetry is not telemetry
    for check in ('ARM_VELOCITY_LIMIT', 'ROBOT_TABLE_PENETRATION', 'WORKSPACE_VIOLATION',
            'SUSTAINED_HAND_OVERLOAD', 'PROBE_LOAD_OR_GRIP_NOT_CONFIRMED'):
        assert check in rendered
    assert 'R26_RELEASE_CONTACT_NOT_CLEARED' in rendered
    assert "hand.phase_begin(_r26_parameters['hold_hand_phase'], data)" in rendered
    assert "recovery_mode='LOWER'" in rendered
    assert "recovery_mode='BRAKE'" in rendered
    assert "recovery_mode='RELEASE'" in rendered
    assert ast.dump(ast.parse(source)) != ast.dump(ast.parse(rendered))
    with pytest.raises(ValueError, match='source SHA256'):
        hook.transform(original, source + '\n')
