"""Independent recovery reconstruction rejects misleading success/force evidence."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

import actual_wrist_probe_controller_hooks as probe
import audit_public_ycb_safe_recovery_r26 as audit
import safe_recovery_controller_hooks_r26 as runtime
from test_public_ycb_bias_budget_r17 import real_model


PROJECT = Path(__file__).resolve().parents[1]
FROZEN = PROJECT / ('results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414/'
    'ycb_can_bias_budget_r17_seed3414/derived_controller.py')


def stage(mode='BRAKE', start=1000, count=75, supported=True, finished=True):
    anchor = np.array([.6, 0., .6])
    samples, observations, writes, settled = [], {}, {}, 0.
    channels = dict.fromkeys(('thumb', 'index', 'middle', 'ring', 'pinky', 'palm'), 1.)
    observations[start] = {'position': anchor.copy(), 'support': supported, 'controls': {'loaded': .4}}
    previous = anchor.copy()
    for ordinal in range(1, count + 1):
        index = start + ordinal
        position = anchor.copy()
        if mode == 'LOWER':
            position[2] -= min(.05, .004 * ordinal * .002)
        speed = float(np.linalg.norm(position - previous) / .002)
        settled = settled + .002 if speed <= .005 else 0.
        ready = bool(settled + 1e-9 >= .15 and (mode == 'BRAKE' or supported))
        phase = 'RELEASE' if mode == 'RELEASE' else 'R26_RECOVERY_HOLD'
        sample = {'timestamp_s': index * .002, 'mode': mode, 'actual_wrist_world_m': position.tolist(),
            'actual_wrist_speed_m_s': speed, 'settled_seconds': settled,
            'channel_pressures_n': dict(channels), 'original_table_support_confirmed': supported,
            'ready': ready, 'hand_phase': phase}
        samples.append(sample)
        observations[index] = {'position': position, 'support': supported, 'pressures': dict(channels),
            'state': 'RELEASE' if mode == 'RELEASE' else 'RECOVER', 'controls': {}, 'inline': copy.deepcopy(sample)}
        writes[index - 1] = {'phase': phase, 'mode': 'RELEASING' if mode == 'RELEASE' else 'JOINT_CONTROL',
            'per_finger': [], 'final_safety_override': False,
            'admission_after_write': {'preload_ready': False, 'probe_active': False, 'payload_confirmed': False,
                'probe_eligible_seconds': 0., 'probe_confirmed_seconds': 0.}}
        previous = position
    target = anchor.copy()
    if mode == 'LOWER' and count:
        target[2] -= min(.05, .004 * (count - 1) * .002)
    item = {'mode': mode, 'start_s': start * .002, 'start_wrist_world_m': anchor.tolist(),
        'target_world_m': target.tolist(), 'initial_hand_references_rad': {'loaded': .4},
        'samples': samples, 'settled_seconds': settled, 'ready': ready if count else False, 'finished': finished}
    return item, observations, writes


def force_abort_fixture():
    item, observations, writes = stage(count=20, finished=False)
    last = item['samples'][-1]
    last['channel_pressures_n']['middle'] = 12.5
    index = audit.tick(last['timestamp_s'])
    observations[index]['pressures']['middle'] = 12.5
    observations[index]['inline'] = None
    failure = 'RECOVERY_R26_HAND_FORCE_LIMIT'
    result = {'success': False, 'failure': failure, 'successful_recoveries': 0,
        'transitions': [{'timestamp': last['timestamp_s'], 'state': 'ABORT', 'reason': failure}]}
    return item, observations, writes, result


def test_low_speed_window_and_fixed_admission_anchor_are_recomputed():
    item, observations, writes = stage()
    result = audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')
    assert result['ready'] and result['finished'] and result['continuous_settled_seconds'] == pytest.approx(.15)
    assert result['no_explicit_opening_in_hold']
    assert not result['physical_closed_hand_or_force_invariant_proven']
    item['target_world_m'][0] += .001
    with pytest.raises(ValueError, match='Final issued recovery target'):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')


@pytest.mark.parametrize('field', ['actual_wrist_speed_m_s', 'settled_seconds'])
def test_self_reported_speed_or_settling_is_not_trusted(field):
    item, observations, writes = stage()
    item['samples'][20][field] += .001
    with pytest.raises(ValueError):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')


@pytest.mark.parametrize('change', ['phase', 'mode', 'admission'])
def test_hold_rejects_explicit_opening_or_renewed_admission(change):
    item, observations, writes = stage()
    writer = writes[1000]
    if change == 'phase':
        writer['phase'] = 'RECOVER'
    elif change == 'mode':
        writer['mode'] = 'RELEASING'
    else:
        writer['admission_after_write']['payload_confirmed'] = True
    with pytest.raises(ValueError):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')


def test_force_abort_is_accepted_failed_evidence_without_becoming_success():
    item, observations, writes, result = force_abort_fixture()
    report = audit.reconstruct_session(item, observations, writes, result['failure'])
    assert report['reconstructed_peak_channel_n'] == 12.5
    assert report['final_abort_samples_without_inline_assignment'] == 1
    sequence = audit.sequence_contract([item], [report], observations, result, 1020)
    assert not sequence['recovery_completed'] and not sequence['all_recorded_stages_finished']
    result['success'] = True
    with pytest.raises(ValueError, match='relabeled as task success'):
        audit.sequence_contract([item], [report], observations, result, 1020)


def test_abort_must_be_final_native_step_and_retain_reason():
    item, observations, writes, result = force_abort_fixture()
    with pytest.raises(ValueError, match='abort reason'):
        audit.reconstruct_session(item, observations, writes, 'RECOVERY_OTHER_FAILURE')
    report = audit.reconstruct_session(item, observations, writes, result['failure'])
    with pytest.raises(ValueError, match='integration continued'):
        audit.sequence_contract([item], [report], observations, result, 1021)
    item['finished'] = True
    with pytest.raises(ValueError, match='continued/finished'):
        audit.reconstruct_session(item, observations, writes, result['failure'])


def test_missing_inline_requires_a_real_terminal_violation():
    item, observations, writes = stage()
    observations[1001]['inline'] = None
    with pytest.raises(ValueError, match='Missing inline'):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')


def test_release_requires_prior_support_and_rejects_loss():
    item, observations, writes = stage('RELEASE', count=2000)
    report = audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')
    assert report['finished'] and report['native_samples'] == 2000
    observations[1000]['support'] = False
    with pytest.raises(ValueError, match='before original support'):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')
    observations[1000]['support'] = True
    item['finished'] = False
    item['ready'] = False
    item['samples'][-1]['original_table_support_confirmed'] = False
    item['samples'][-1]['ready'] = False
    observations[3000]['support'] = False
    observations[3000]['inline'] = None
    report = audit.reconstruct_session(item, observations, writes, 'RECOVERY_R26_TABLE_SUPPORT_LOST_DURING_RELEASE')
    assert report['abort_reason'] == 'R26_TABLE_SUPPORT_LOST_DURING_RELEASE'


def test_lowering_final_target_uses_pre_step_clock_and_vertical_cap():
    item, observations, writes = stage('LOWER', count=75)
    report = audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')
    assert report['ready']
    item['target_world_m'][2] -= .004 * .002
    with pytest.raises(ValueError, match='Final issued recovery target'):
        audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')


def test_invalid_stage_order_and_fabricated_recovery_count_are_rejected():
    item, observations, writes = stage()
    report = audit.reconstruct_session(item, observations, writes, 'original_grasp_failure')
    result = {'successful_recoveries': 1}
    with pytest.raises(ValueError, match='count exceeds'):
        audit.sequence_contract([item], [report], observations, result, 1075)
    item['mode'] = report['mode'] = 'RELEASE'
    with pytest.raises(ValueError, match='sequence is invalid'):
        audit.sequence_contract([item], [report], observations, result, 1075)


def test_support_filter_requires_raw_filtered_and_continuous_support():
    config = {'minimum_load_n': 2., 'confirmation_seconds': .12,
        'filter_seconds': .03, 'maximum_sample_gap_seconds': .02}
    reconstruction = audit.SupportReconstruction(config)
    flags = [reconstruction.update(index * .002, 4.) for index in range(1, 101)]
    assert not flags[59] and flags[-1]
    assert not reconstruction.update(.202, 0.)
    assert not reconstruction.update(.3, 4.)
    assert reconstruction.filtered == 0.


def test_real_compiled_text_and_frozen_hook_bind_to_contract_without_running():
    namespace = {'_r23_probe_telemetry': probe.ProbeTelemetry(), '_r23_probe_parameters': dict(probe.PARAMETERS)}
    original = FROZEN.read_text()
    exec(compile(original, '<test_frozen_r25>', 'exec'), namespace)
    task, source, _ = runtime.transform(namespace['run_task'], original)
    frozen_hook = (PROJECT / audit.HOOK).read_text()
    report, parameters = audit.source_contract(source, frozen_hook, task.r26_safe_recovery_transform)
    assert report['derived_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert parameters == runtime.PARAMETERS
    assert not report['historical_process_identity_independently_proven']
    with pytest.raises(ValueError, match='compiled source SHA'):
        audit.source_contract(source + '\n', frozen_hook, task.r26_safe_recovery_transform)


def test_native_control_mapping_normalizes_numpy_scalars(monkeypatch, tmp_path, real_model):
    pressures = dict.fromkeys(('thumb', 'index', 'middle', 'ring', 'pinky', 'palm'), 0.)
    native = [{'timestamp': .002, 'state': 'RECOVER', 'motor_commands': [0.] * real_model.nu,
        'native_contacts': []}]
    trace = [{'timestamp': .002, 'state': 'RECOVER', 'wrist_position': [.6, 0., .6],
        'arm_joint_position': [0.] * 7, 'hand_command': [0.] * (real_model.nu - 7),
        'hand_segment_forces': pressures,
        'table_load_sensor': {'raw_load_n': 0., 'filtered_load_n': 0., 'confirmed': False}}]
    monkeypatch.setattr(audit.stream, 'json_array', lambda path: iter(native if path.name == 'native' else trace))
    config = {'minimum_load_n': 2., 'confirmation_seconds': .12,
        'filter_seconds': .03, 'maximum_sample_gap_seconds': .02}
    observations, count, _, _ = audit.collect_native(tmp_path / 'native', tmp_path / 'trace', real_model,
        [{'start_s': .002, 'samples': []}], config)
    assert count == 1 and observations[1]['controls']
    for value in observations[1]['controls'].values():
        assert type(value) is float
        audit.equal(value, 0., 'native control')
