"""Runner declaration checks for R26; no native task-success claim."""

import copy
import hashlib
import json

import pytest

import develop_public_ycb_safe_recovery_r26 as r26
from test_public_ycb_bias_budget_r17 import real_model
from test_public_ycb_rate_limited_probe_r25 import factory as inherited_factory


PROFILE = r26.PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"


def test_declaration_inherits_finalized_r25_inputs_and_adds_safe_recovery():
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    assert declaration["format"] == r26.PROTOCOL_FORMAT
    assert declaration["inherited_r25_runner_revision"] == r26.r25.PROTOCOL_FORMAT
    assert declaration["safe_recovery_parameters"] == r26.recovery.PARAMETERS
    assert declaration["original_task_and_strict_task_gates_unchanged"] is True
    assert declaration["inherited_r24_declaration"]


def test_snapshot_contains_new_r26_sources_and_inherited_bindings():
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    snapshot = r26.snapshot(PROFILE, declaration)
    for relative in r26.NEW_SOURCES:
        assert relative in snapshot
        assert len(snapshot[relative]) == 64
    assert len(snapshot) >= len(declaration["inherited_r24_source_hashes"])


def test_other_case_and_existing_output_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="isolates"):
        r26.run(tmp_path / "new", PROFILE, (("ycb_can", 3415),),
            expected_baseline_summary_sha256=r26.BASELINE_SHA256)
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        r26.run(tmp_path / "existing", PROFILE, (("ycb_can", 3414),),
            expected_baseline_summary_sha256=r26.BASELINE_SHA256)


def test_real_nested_factory_serializer_and_snapshot(monkeypatch, tmp_path, real_model):
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    monkeypatch.setattr(r26.r25, 'controller_for', r26.controller_for)
    monkeypatch.setattr(r26.r25, 'isolated_runner',
        lambda path, inherited: r26.isolated_runner(path, declaration))
    owner, task, source, hands = inherited_factory(monkeypatch, tmp_path, real_model, runner=True)
    assert hashlib.sha256(source.encode()).hexdigest() == task.r26_safe_recovery_transform['compiled_source_sha256']
    assert task.__globals__['_r26_recovery'] is task.safe_recovery_telemetry
    assert task.__globals__['_r23_probe_telemetry'] is task.actual_wrist_probe_telemetry
    assert task.actual_wrist_probe_telemetry is not task.safe_recovery_telemetry
    writer = owner.__globals__['previous'].atomic_json
    for filename in ('metadata.json', 'result.json'):
        writer(tmp_path / filename, {'kind': 'ycb_can', 'seed': 3414, 'controller_revision': 'unit_no_integration'})
        value = json.loads((tmp_path / filename).read_text())
        assert value['safe_recovery'] == task.safe_recovery_telemetry.as_dict()
        assert value['actual_wrist_probe']['format'] == 'rate_limited_actual_wrist_probe_policy_v1'
        assert value['runner_revision'] == r26.PROTOCOL_FORMAT
        assert value['fresh_contact_tracking']['hands'] == [hand.fresh_contact_telemetry() for hand in hands]
        assert value['safe_recovery_source_transform'] == task.r26_safe_recovery_transform
    assert owner.__globals__['snapshot'](PROFILE) == r26.snapshot(PROFILE, declaration)


def test_protocol_retains_task_gates_and_has_one_new_declaration():
    declaration = r26.validate_inputs(PROFILE, r26.BASELINE_SHA256)
    before = json.loads((r26.BASELINE / 'protocol.json').read_text())
    actual = r26.tagged_protocol(copy.deepcopy(before), declaration)
    assert actual['format'] == r26.PROTOCOL_FORMAT
    assert actual[r26.DECLARATION] == declaration
    for key in ('phase_seconds', 'scoring', 'load_transfer_probe', 'actual_wrist_probe', 'alignment_settling'):
        assert actual[key] == before[key]
    assert all(row['runner_revision'] == r26.PROTOCOL_FORMAT for row in actual['schedule'])

