"""Independent R21 source, live binding and retained-observation mutation checks."""

import ast
import copy
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_public_ycb_provisional_preload as audit


@pytest.fixture(scope="module")
def proof():
    project = Path(__file__).resolve().parents[1]
    names = (audit.POLICY, "scripts/develop_public_ycb_constrained_preload_r14.py",
        "scripts/develop_public_ycb_bias_recovery_r16.py", "scripts/develop_public_ycb_joint_admittance_r13.py")
    stream, before = io.BytesIO(), {}
    with zipfile.ZipFile(stream, "w") as archive:
        for name in names:
            data = (project / name).read_bytes()
            before[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    with zipfile.ZipFile(stream) as archive:
        return audit.policy_source_proof(archive, before)


def protocol_pair():
    prior = {key: {"synthetic": key} for key in audit.r20.UNCHANGED_CONTRACTS}
    prior.update(phase_seconds={"CLOSE": 32.}, maximum_attempts=2,
        constrained_preload={"continuous_seconds": .25, "close_seconds": 32.},
        load_transfer_probe={"probe_distance_m": .005, "confirmation_seconds": .25,
            "maximum_remaining_table_weight_fraction": .2},
        r20_shared_normal_experiment={"active_r19_declaration": {"original": "R19"}},
        force_control="Original R20 force control")
    prior["schedule"] = [{"kind": "ycb_can", "seed": 3414, "profile": "bias_budget_r17",
        "runner_revision": audit.r20.FORMAT, "controller_class": "SharedNormalBudgetHand",
        "target_allocator_revision": audit.r20.FORMAT, "additional_bias_limit_rad": .10,
        "close_seconds": 32., "arm_interpolation_seconds": 16.}]
    protocol = copy.deepcopy(prior)
    protocol.pop("r20_shared_normal_experiment")
    protocol.update(format=audit.FORMAT, runner_revision=audit.FORMAT, controller_class=audit.CLASS,
        admission_policy_revision=audit.FORMAT, target_allocator_revision=audit.r20.FORMAT,
        controller_revision_scope="R21 admission only",
        inherited_r20_shared_normal_experiment=copy.deepcopy(prior["r20_shared_normal_experiment"]),
        inherited_r20_force_control=prior["force_control"])
    protocol[audit.DECLARATION] = {
        "format": "r21_provisional_preload_development_protocol_v1", "runner_revision": audit.FORMAT,
        "controller": audit.CLASS, "admission_policy_revision": audit.FORMAT,
        "target_allocator_revision": audit.r20.FORMAT, "changed_parameters": copy.deepcopy(audit.CHANGE),
        "active_r20_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]),
        "inherited_r20_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]),
        "active_r19_declaration": copy.deepcopy(prior["r20_shared_normal_experiment"]["active_r19_declaration"]),
        "inherited_r20_force_control": prior["force_control"], "additional_bias_limit_rad": .10,
        "close_seconds": 32., "arm_interpolation_seconds": 16., "native_timestep_s": .002,
        "actual_fixed_normal_1g_observation_retained": True,
        "actual_fixed_normal_1g_confirmation_and_held_policy_unchanged": True,
        "all_other_preload_reasons_and_recovery_inhibition_unchanged": True,
        "target_allocator_and_write_mechanics_unchanged": True,
        "r20_comparison_summary_sha256": audit.PINS[audit.R20_SUMMARY],
        "policy_hook_source_sha256": audit.FROZEN_POLICY_SHA256,
        "deferred_pre_motion_reason": audit.DEFERRED_REASON,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_strict_task_gates_modified": False,
        "baseline_task_success": False, "baseline_held_lift": False,
        "baseline_original_command_audit_passed": False, "baseline_postrun_extension_format": audit.postrun.FORMAT,
        "baseline_child_exit_observed": True, "baseline_child_returncode": 0}
    baseline = {"protocol": prior, "rows": [{"derived_controller_sha256": "a" * 64,
        "arm_source_transform": {"original_arm_seconds": 16.}, "catalogue_mass_prior_kg": .15}]}
    return protocol, baseline


def runtime_case(proof, count=2):
    protocol, baseline = protocol_pair()
    planned = dict(baseline["protocol"]["schedule"][0], runner_revision=audit.FORMAT,
        controller_class=audit.CLASS, controller_revision_scope=protocol["controller_revision_scope"],
        admission_policy_revision=audit.FORMAT)
    binding = {"controller_class": audit.CLASS, "additional_bias_limit_rad": .10,
        "admission_policy_revision": audit.FORMAT, "python_version": [3, 10, 12],
        "writer_is_frozen_r16": True, "after_step_is_frozen_r16": True, "admit_probe_is_frozen_r16": True,
        "unchanged_methods": dict.fromkeys(audit.UNCHANGED_METHODS, True)}
    for path in ("tracking", "gate_only"):
        binding[path] = dict(proof["allocator_paths"][path], method_bytecode_sha256="b" * 64,
            same_code_as_original=True, same_closure_as_original=True,
            other_globals_match_except_warning_registry=True, allocator_is_shared_normal=True,
            allocator={"module": "shared_normal_tracking_targets", "qualname": "allocate_tracking_targets",
                "source_sha256": audit.r20.FROZEN_PURE_SHA256})
    binding["gate_only"]["r13_proxy_other_members_match"] = True
    policy = {"ast_hash_format": audit.AST_FORMAT, "deferred_pre_motion_reason": audit.DEFERRED_REASON,
        "wrapper_is_provisional": True, "r16_helper_is_provisional": True, "r14_helper_is_provisional": True,
        "original_probe_snapshot_and_actual_fixed_normal_value_retained": True,
        "update_preload": dict(proof["update_preload"], method_bytecode_sha256="c" * 64)}
    for name in ("r14_parent", "r16_recovery_wrapper"):
        policy[name] = copy.deepcopy(proof[name])
        policy[name]["original"]["method_bytecode_sha256"] = "d" * 64
        policy[name]["other_globals_match_except_warning_registry"] = True
    binding["admission_policy"] = policy
    row = dict(planned, runner_error=None, attempts=[{"attempt": index + 1} for index in range(count)],
        effective_task_phase_seconds=copy.deepcopy(protocol["phase_seconds"]),
        effective_runner_phase_seconds=copy.deepcopy(protocol["phase_seconds"]),
        effective_hand_instances=count, effective_hand_budgets_rad=[.10] * count,
        effective_hand_controller_types=[audit.CLASS] * count,
        shared_normal_runtime_bindings=[copy.deepcopy(binding) for _ in range(count)], **baseline["rows"][0])
    return planned, row, protocol, baseline


def policy_row():
    return {"ready": True, "continuous_seconds": .25, "reasons": [],
        "current_target_lp": {"feasible": True}, "current_groups": {"thumb": {"target_n": 2., "measured_n": 2.}},
        "original_probe_snapshot": {"sampled_fixed_normal_1g_feasible": False, "preprobe_reasons": [],
            "sampled_solve_current": True, "cached_measurements_match": True, "solve_age_s": 0.},
        "provisional_preload_policy": {"format": "provisional_preload_admission_policy_v1",
            "deferred_pre_motion_reason": audit.DEFERRED_REASON, "sampled_fixed_normal_1g_feasible_observed": False,
            "additional_bias_limit_rad": .10, "continuous_preload_seconds": .25,
            "original_probe_snapshot_unchanged": True, "recovery_inhibition_unchanged": True,
            "actual_fixed_normal_1g_required_for_confirmation": True,
            "confirmation_and_held_policy_unchanged": True, "strict_task_success_unchanged": True}}


def test_canonical_ast_ignores_only_empty_type_parameters():
    original = ast.parse("def f(x):\n    return x + 1\n").body[0]
    modern = copy.deepcopy(original)
    modern._fields = (*modern._fields, "type_params") if "type_params" not in modern._fields else modern._fields
    modern.type_params = []
    assert audit.ast_sha(modern) == audit.ast_sha(original)
    modern.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert audit.ast_sha(modern) != audit.ast_sha(original)
    assert audit.ast_sha(original) != audit.ast_sha(ast.parse("def f(x):\n    return x + 2\n").body[0])


def test_original_sources_have_exactly_the_two_declared_transformations(proof):
    assert proof["r14_parent"]["deleted_reason_if_count"] == 1
    assert proof["r16_recovery_wrapper"]["replaced_parent_call_count"] == 1
    for key in ("r14_parent", "r16_recovery_wrapper"):
        assert proof[key]["original_ast_sha256"] != proof[key]["transformed_ast_sha256"]
    assert proof["update_preload"]["method_source_file_sha256"] == audit.FROZEN_POLICY_SHA256


def test_true_r21_contract_does_not_relabel_or_mutate_r20():
    protocol, baseline = protocol_pair()
    before = copy.deepcopy((protocol, baseline))
    audit.check_protocol(protocol, baseline)
    assert (protocol, baseline) == before


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__("format", audit.r20.FORMAT),
    lambda p: p.__setitem__("controller_class", "SharedNormalBudgetHand"),
    lambda p: p.__setitem__("target_allocator_revision", audit.FORMAT),
    lambda p: p.__setitem__("scoring", {"relaxed": True}),
    lambda p: p[audit.DECLARATION]["changed_parameters"].__setitem__("table_limit", {"before": .2, "after": .9}),
    lambda p: p[audit.DECLARATION].__setitem__("additional_bias_limit_rad", .04),
    lambda p: p[audit.DECLARATION].__setitem__("actual_fixed_normal_1g_observation_retained", False),
    lambda p: p[audit.DECLARATION].__setitem__("baseline_original_command_audit_passed", True),
    lambda p: p["load_transfer_probe"].__setitem__("probe_distance_m", .004),
    lambda p: p["load_transfer_probe"].__setitem__("maximum_remaining_table_weight_fraction", .9),
    lambda p: p["constrained_preload"].__setitem__("continuous_seconds", .1),
])
def test_contract_and_actual_gate_drift_fail_closed(change):
    protocol, baseline = protocol_pair()
    change(protocol)
    with pytest.raises((ValueError, KeyError)):
        audit.check_protocol(protocol, baseline)


@pytest.mark.parametrize("count", [1, 2])
def test_actual_hand_bindings_support_one_hand_or_one_retry_without_mutation(proof, count):
    values = runtime_case(proof, count)
    before = copy.deepcopy(values)
    audit.check_case_metadata(*values, proof)
    assert values == before


def test_actual_r21_method_recorder_matches_independently_reconstructed_source(proof):
    import develop_public_ycb_provisional_preload_r21 as runner

    hand = object.__new__(runner.policy.ProvisionalPreloadBudgetHand)
    hand._additional_bias_limit_rad = .10
    planned, row, protocol, baseline = runtime_case(proof, count=1)
    row["shared_normal_runtime_bindings"] = [runner.runtime_binding(hand)]
    audit.check_case_metadata(planned, row, protocol, baseline, proof)


@pytest.mark.parametrize("change", [
    lambda r: r.__setitem__("missing_runtime_controller_binding", True),
    lambda r: r.__setitem__("effective_hand_budgets_rad", [.10, .04]),
    lambda r: r.__setitem__("effective_hand_instances", True),
    lambda r: r["shared_normal_runtime_bindings"][0]["unchanged_methods"].__setitem__("confirm_probe", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["unchanged_methods"].pop("held_revalidated"),
    lambda r: r["shared_normal_runtime_bindings"][0]["tracking"].__setitem__("allocator_is_shared_normal", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["gate_only"].__setitem__("r13_proxy_other_members_match", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"].__setitem__("r14_helper_is_provisional", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"].__setitem__("ast_hash_format", "ast_dump"),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"]["r14_parent"].__setitem__("deleted_reason_if_count", 2),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"]["r14_parent"].__setitem__("deleted_reason_if_count", True),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"]["r16_recovery_wrapper"].__setitem__("transformed_ast_sha256", "f" * 64),
    lambda r: r["shared_normal_runtime_bindings"][0]["admission_policy"]["r16_recovery_wrapper"].__setitem__("other_globals_match_except_warning_registry", False),
])
def test_live_class_budget_original_gates_and_compiled_helper_drift_fail_closed(proof, change):
    planned, row, protocol, baseline = runtime_case(proof)
    change(row)
    with pytest.raises((ValueError, KeyError)):
        audit.check_case_metadata(planned, row, protocol, baseline, proof)


def test_provisional_readiness_retains_actual_false_and_scope():
    row = policy_row()
    before = copy.deepcopy(row)
    receipt = audit.audit_policy_rows([row], [])
    assert row == before
    assert receipt["counts"]["ready_rows_with_observed_actual_1g_false"] == 1
    assert receipt["counts"]["actual_fixed_normal_1g_false"] == 1
    assert "not independently rebuilt" in receipt["scope"]


@pytest.mark.parametrize("change", [
    lambda r: r["provisional_preload_policy"].__setitem__("sampled_fixed_normal_1g_feasible_observed", True),
    lambda r: r["original_probe_snapshot"].__setitem__("sampled_fixed_normal_1g_feasible", 0),
    lambda r: r["provisional_preload_policy"].__setitem__("additional_bias_limit_rad", .04),
    lambda r: r["provisional_preload_policy"].__setitem__("actual_fixed_normal_1g_required_for_confirmation", False),
    lambda r: r.__setitem__("continuous_seconds", .24),
    lambda r: r.__setitem__("continuous_seconds", float("nan")),
    lambda r: r.__setitem__("ready", 1),
    lambda r: r.__setitem__("r16_recovery_inhibited", True),
    lambda r: r.__setitem__("reasons", ["OTHER_ORIGINAL_REASON"]),
    lambda r: r.__setitem__("reasons", [audit.DEFERRED_REASON]),
    lambda r: r["original_probe_snapshot"].__setitem__("preprobe_reasons", ["CONTACT_LOSS"]),
    lambda r: r["current_groups"]["thumb"].__setitem__("measured_n", 1.),
    lambda r: r["current_groups"]["thumb"].__setitem__("measured_n", float("nan")),
    lambda r: r["current_groups"]["thumb"].__setitem__("target_n", float("nan")),
    lambda r: r["original_probe_snapshot"].__setitem__("solve_age_s", float("nan")),
    lambda r: r["original_probe_snapshot"].__setitem__("solve_age_s", -.001),
    lambda r: r["original_probe_snapshot"].__setitem__("sampled_solve_current", 1),
    lambda r: r["original_probe_snapshot"].__setitem__("sampled_solve_current", False),
])
def test_corrupted_policy_observation_or_unchanged_readiness_gate_fails_closed(change):
    row = policy_row()
    change(row)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_policy_rows([row], [])


@pytest.mark.parametrize("field,value", [("continuous_seconds", float("nan")), ("ready", "false")])
def test_nonready_record_is_still_required_to_have_valid_numeric_boolean_schema(field, value):
    row = policy_row()
    row.update(ready=False, continuous_seconds=0., reasons=["GROUP_PRESSURE_ERROR"])
    row[field] = value
    with pytest.raises(ValueError):
        audit.audit_policy_rows([row], [])


def confirmation():
    return {"confirmation": True, "actual_wrist_speed_m_s": .009,
        "sampled_fixed_normal_1g_feasible": True, "table_unloaded": True,
        "actual_probe_endpoint_reached": True, "accepted_visual_after_probe_start": True,
        "motion_safety_reasons": []}


def test_confirmation_requires_actual_one_g_even_though_readiness_may_defer_it():
    probe = confirmation()
    assert audit.audit_policy_rows([policy_row()], [probe])["counts"]["confirmed_probe_records"] == 1
    probe["sampled_fixed_normal_1g_feasible"] = False
    with pytest.raises(ValueError):
        audit.audit_policy_rows([policy_row()], [probe])


@pytest.mark.parametrize("field,value", [("confirmation", 1), ("actual_wrist_speed_m_s", float("nan")),
    ("actual_wrist_speed_m_s", -.001), ("actual_wrist_speed_m_s", .011), ("table_unloaded", False)])
def test_confirmation_numeric_boolean_and_unloading_corruption_fails_closed(field, value):
    probe = confirmation()
    probe[field] = value
    with pytest.raises(ValueError):
        audit.audit_policy_rows([policy_row()], [probe])


def test_runner_preflight_resolves_exact_literals_and_rejects_missing_or_duplicate_pins():
    pin = "a" * 64
    assert audit.runner_preflight('PIN = "' + pin + '"\nPREFLIGHT_HASHES = {"one": PIN}') == {"one": pin}
    for source in ('PREFLIGHT_HASHES = {"one": "PENDING"}',
            'PREFLIGHT_HASHES = {"one": "' + pin + '", "one": "' + pin + '"}',
            'PREFLIGHT_HASHES = {}\nPREFLIGHT_HASHES = {}'):
        with pytest.raises(ValueError):
            audit.runner_preflight(source)


def test_current_runner_and_policy_source_are_pinned_to_reviewed_bytes():
    project = Path(__file__).resolve().parents[1]
    assert audit.base.digest(project / audit.SOURCE) == audit.FROZEN_SOURCE_SHA256
    assert audit.base.digest(project / audit.POLICY) == audit.FROZEN_POLICY_SHA256
    pins = audit.runner_preflight((project / audit.SOURCE).read_bytes())
    assert len(pins) == 17
    assert all(pins[path] == expected for path, expected in audit.PINS.items())


def test_initializer_reuse_is_isolated_and_does_not_apply_old_r20_fixed_epoch_counts():
    before = dict(audit.r20.audit_targets.__globals__)
    first, witness = audit.postrun.extended_target_audit()
    second, _ = audit.postrun.extended_target_audit()
    assert first is not second
    assert first.__code__ is not audit.r20.audit_targets.__code__
    assert audit.r20.audit_targets.__globals__ == before
    assert witness
