"""Independent target-output and R20 contract mutation checks."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_public_ycb_shared_normal as audit
import shared_normal_tracking_targets as shared


@pytest.fixture
def target():
    definitions = [("thumb", [-.03, 0., 0.], [1., 0., 0.], 2.),
        ("middle", [.03, 0., -.02], [-1., 0., 0.], 1.),
        ("ring", [.03, 0., .02], [-1., 0., 0.], 1.)]
    patches = [dict(finger=finger, input_index=index, position_world_m=point,
        normal_on_object_world=normal, measured_pressure_n=pressure, hand_body_name=finger + "_tip")
        for index, (finger, point, normal, pressure) in enumerate(definitions)]
    result = shared.allocate_tracking_targets(patches, [0., 0., 0.], .15,
        {"thumb": 2., "middle": 1., "ring": 1.})
    assert result["feasible"]
    result["association"] = {"accepted": True, "estimated_com_world_m": [0., 0., 0.], "patches": patches}
    return result


def streams(target, count=15):
    joint, preload = [], []
    for index in range(count):
        joint.append({"timestamp_s": index * .002, "current_target": copy.deepcopy(target)})
        preload.append({"timestamp_s": (index + 1) * .002, "current_target_lp": copy.deepcopy(target),
            "original_probe_snapshot": {"catalogue_weight_n": .15 * 9.81,
                "sampled_public_mesh_association": copy.deepcopy(target["association"])}})
    return joint, preload


def test_output_math_and_independent_bounded_sampling(target):
    assert audit.check_target(target)
    joint, preload = streams(target)
    before = copy.deepcopy((joint, preload))
    receipt = audit.audit_targets(joint, preload, .15)
    assert (joint, preload) == before
    assert receipt["independent_sampled_target_count"] == 12
    assert receipt["independent_ray_solution_or_optimality_proven"] is False
    samples = receipt["independent_sampled_checks"]
    assert [row["stream_record_index"] for row in samples[:6]] == [0, 2, 5, 8, 11, 14]
    assert all(set(row["fixed_normal_checks"]) == {"1.0", "1.5"} for row in samples)
    assert all(row["feasible_targets"] == 15 for row in receipt["full_stream_record_checks"].values())


@pytest.mark.parametrize("change", [
    lambda t: t.__setitem__("format", "STATIC_TRACKING_TARGET_ALLOCATED"),
    lambda t: t.__setitem__("runtime_target_state_used", True),
    lambda t: t.__setitem__("solver_time_limit_per_lp_s", 1.),
    lambda t: t.__setitem__("solver_status", 1),
    lambda t: t["numerical_checks"].__setitem__("force_residual_max_n", 1e-3),
    lambda t: t["numerical_checks"].__setitem__("shared_patch_normal_residual_max_n", 1e-4),
    lambda t: t["numerical_checks"].pop("inequality_excess_max_n"),
    lambda t: t["target_wrench_residual"].__setitem__(0, 1e-6),
    lambda t: t["independent_fixed_normal_checks"].pop("1.0"),
    lambda t: t["independent_fixed_normal_checks"]["1.5"].__setitem__("measured_feasible", False),
    lambda t: t["independent_fixed_normal_checks"]["1.0"]["measured_wrench_residual"].__setitem__(0, 1e-3),
    lambda t: t["patches"][0].__setitem__("target_normal_n", 6.1),
    lambda t: t["patches"][0].__setitem__("target_normal_n", -.01),
    lambda t: t["patches"][0].__setitem__("measured_pressure_n", .049),
    lambda t: t["patches"][0].__setitem__("raw_input_index", 1),
    lambda t: t["patches"][0].__setitem__("normal_on_object_world", [0., 0., 0.]),
    lambda t: t["patches"][0].__setitem__("position_world_m", [float("nan"), 0., 0.]),
    lambda t: t["effective_patch_mu"].__setitem__(0, .8),
    lambda t: t["targets_n"].__setitem__("thumb", .5),
    lambda t: t["objective"].__setitem__("primary_normal_regularizer", 0.),
])
def test_corrupt_accepted_target_fails_closed(target, change):
    change(target)
    with pytest.raises((ValueError, KeyError)):
        audit.check_target(target)


def test_rejected_solver_diagnostic_is_not_forged_as_accepted(target):
    rejected = {"format": audit.TARGET_FORMAT, "feasible": False, "reason": "SHARED_NORMAL_TRACKING_LP_RESIDUAL",
        "runtime_target_state_used": False, "solver_time_limit_per_lp_s": .1,
        "numerical_checks": dict(target["numerical_checks"], force_residual_max_n=.01)}
    assert audit.check_target(rejected) is False
    rejected["numerical_checks"]["force_residual_max_n"] = float("nan")
    with pytest.raises(ValueError):
        audit.check_target(rejected)


def test_original_caller_gate_rejection_is_counted_without_allocator_claim(target):
    joint, preload = streams(target, 1)
    joint.insert(0, {"timestamp_s": 0., "current_target": {"feasible": False,
        "reason": "ORIGINAL_EMERGENCY_PRECEDENCE"}})
    preload.insert(0, {"timestamp_s": 0., "current_target_lp": {"feasible": False,
        "reason": "CURRENT_GATE_ASSOCIATION_INVALID"}})
    result = audit.audit_targets(joint, preload, .15)
    assert result["full_stream_record_checks"]["tracking"]["preallocator_gate_rejections"] == 1
    assert result["full_stream_record_checks"]["gate_only"]["preallocator_gate_rejections"] == 1


def test_unversioned_feasible_target_and_missing_path_are_rejected(target):
    joint, preload = streams(target, 1)
    joint[0]["current_target"].pop("format")
    with pytest.raises(ValueError):
        audit.audit_targets(joint, preload, .15)
    with pytest.raises(ValueError):
        audit.audit_targets([], preload, .15)


@pytest.mark.parametrize("change", [
    lambda j, p: j[0]["current_target"]["association"].__setitem__("accepted", False),
    lambda j, p: p[0]["original_probe_snapshot"]["sampled_public_mesh_association"]["patches"][0].__setitem__("position_world_m", [0., 0., 0.]),
    lambda j, p: p[0]["original_probe_snapshot"].__setitem__("catalogue_weight_n", .16 * 9.81),
])
def test_association_and_catalogue_prior_are_bound(target, change):
    joint, preload = streams(target, 1)
    change(joint, preload)
    with pytest.raises(ValueError):
        audit.audit_targets(joint, preload, .15)


def test_independent_sampler_does_not_trust_recorded_success(target, monkeypatch):
    joint, preload = streams(target, 1)
    monkeypatch.setattr(audit.allocator, "allocate_contact_wrench", lambda *a, **k: {
        "measured_feasible": False, "measured_solver_status": 2})
    with pytest.raises(ValueError, match="Independent sampled"):
        audit.audit_targets(joint, preload, .15)


def protocol_pair():
    prior = {key: {"synthetic": key} for key in audit.UNCHANGED_CONTRACTS}
    prior.update(joint_admittance={"additional_bias_limit_rad": .10}, phase_seconds={"CLOSE": 32.},
        constrained_preload={"close_seconds": 32., "group_error_limit_n": .35},
        r19_extended_budget_experiment={"original": "exact .10 R19 history"}, force_control="Original R19 force control")
    protocol = copy.deepcopy(prior)
    protocol.pop("r19_extended_budget_experiment")
    protocol.update(format=audit.FORMAT, runner_revision=audit.FORMAT,
        inherited_r19_extended_budget_experiment=copy.deepcopy(prior["r19_extended_budget_experiment"]),
        inherited_r19_force_control=prior["force_control"], controller_class="SharedNormalBudgetHand",
        target_allocator_revision=audit.FORMAT)
    protocol[audit.DECLARATION] = {"format": "r20_shared_normal_development_protocol_v1",
        "runner_revision": audit.FORMAT, "changed_parameters": {"tracking_and_gate_allocator": {
            "before": "r13_single_1_5g_then_fixed_1g_postcheck", "after": audit.TARGET_FORMAT}},
        "r19_comparison_summary_sha256": audit.FROZEN_INPUTS[audit.R19_SUMMARY],
        "additional_bias_limit_rad": .10, "close_seconds": 32., "arm_interpolation_seconds": 16.,
        "native_timestep_s": .002, "controller": "SharedNormalBudgetHand",
        "active_r19_declaration": copy.deepcopy(prior["r19_extended_budget_experiment"]),
        "inherited_r19_declaration": copy.deepcopy(prior["r19_extended_budget_experiment"]),
        "inherited_r19_force_control": prior["force_control"],
        "effective_phase_seconds": copy.deepcopy(prior["phase_seconds"]),
        "effective_constrained_preload": copy.deepcopy(prior["constrained_preload"]),
        "allocator_format": audit.TARGET_FORMAT, "allocator_paths": ["tracking", "gate_only"],
        "allocator_source_sha256": audit.FROZEN_PURE_SHA256, "hook_source_sha256": audit.FROZEN_HOOK_SHA256,
        "lp_time_limit_per_solve_s": .1, "baseline_task_success": False, "baseline_held_lift": False,
        "baseline_child_exit_observed": True, "baseline_child_returncode": 0,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False}
    return protocol, {"protocol": prior}


def test_true_r20_protocol_does_not_mutate_or_relabel_old_protocol():
    protocol, baseline = protocol_pair()
    before = copy.deepcopy((protocol, baseline))
    audit.check_protocol(protocol, baseline)
    assert (protocol, baseline) == before


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__("format", audit.historical.FORMAT),
    lambda p: p["joint_admittance"].__setitem__("additional_bias_limit_rad", .04),
    lambda p: p["phase_seconds"].__setitem__("CLOSE", 16.),
    lambda p: p["constrained_preload"].__setitem__("group_error_limit_n", .5),
    lambda p: p.__setitem__("scoring", {"relaxed": True}),
    lambda p: p[audit.DECLARATION].__setitem__("r19_comparison_summary_sha256", "bad"),
    lambda p: p[audit.DECLARATION].__setitem__("arm_interpolation_seconds", 32.),
    lambda p: p[audit.DECLARATION].__setitem__("native_timestep_s", .004),
    lambda p: p[audit.DECLARATION]["changed_parameters"].__setitem__("mass", {"before": 1., "after": .1}),
    lambda p: p.__setitem__("r19_extended_budget_experiment", {"relabeled": True}),
    lambda p: p.pop("inherited_r19_extended_budget_experiment"),
])
def test_active_and_historical_protocol_drift_is_rejected(change):
    protocol, baseline = protocol_pair()
    change(protocol)
    with pytest.raises((ValueError, KeyError)):
        audit.check_protocol(protocol, baseline)


def metadata_fixture():
    protocol, baseline = protocol_pair()
    prior = {"kind": "ycb_can", "seed": 3414, "profile": "bias_budget_r17", "runner_revision": audit.historical.FORMAT,
        "additional_bias_limit_rad": .10, "close_seconds": 32., "arm_interpolation_seconds": 16.}
    baseline["protocol"]["schedule"] = [prior]
    baseline["rows"] = [{"derived_controller_sha256": "d" * 64, "arm_source_transform": {"original_arm": 16.},
        "catalogue_mass_prior_kg": .349}]
    protocol["maximum_attempts"] = 2
    planned = dict(prior, runner_revision=audit.FORMAT, controller_class="SharedNormalBudgetHand",
        target_allocator_revision=audit.FORMAT)
    expected = {path: {"method_source_sha256": "a" * 64, "method_name": path,
        "method_source_file": path + ".py", "method_source_file_sha256": "b" * 64}
        for path in ("tracking", "gate_only")}
    binding = {"controller_class": "SharedNormalBudgetHand", "additional_bias_limit_rad": .10,
        "python_version": [3, 10, 12], "writer_is_frozen_r16": True, "after_step_is_frozen_r16": True,
        "admit_probe_is_frozen_r16": True}
    for path in ("tracking", "gate_only"):
        binding[path] = dict(expected[path], same_code_as_original=True, same_closure_as_original=True,
            other_globals_match_except_warning_registry=True, allocator_is_shared_normal=True,
            r13_proxy_other_members_match=True, method_bytecode_sha256="c" * 64,
            allocator={"module": "shared_normal_tracking_targets", "qualname": "allocate_tracking_targets",
                "source_sha256": audit.FROZEN_PURE_SHA256})
    row = dict(planned, runner_error=None, attempts=[{"attempt": 1}, {"attempt": 2}],
        effective_task_phase_seconds=copy.deepcopy(protocol["phase_seconds"]),
        effective_runner_phase_seconds=copy.deepcopy(protocol["phase_seconds"]),
        effective_hand_instances=2, effective_hand_budgets_rad=[.10, .10],
        effective_hand_controller_types=["SharedNormalBudgetHand"] * 2,
        shared_normal_runtime_bindings=[copy.deepcopy(binding), copy.deepcopy(binding)],
        **baseline["rows"][0])
    return planned, row, protocol, baseline, expected


def test_actual_two_paths_each_retry_bind_without_cross_python_bytecode_claim():
    args = metadata_fixture()
    before = copy.deepcopy(args)
    audit.check_case_metadata(*args)
    assert args == before


@pytest.mark.parametrize("change", [
    lambda r: r.__setitem__("missing_runtime_controller_binding", True),
    lambda r: r.__setitem__("missing_runtime_allocator_binding", True),
    lambda r: r["effective_hand_budgets_rad"].__setitem__(1, .04),
    lambda r: r["effective_hand_controller_types"].__setitem__(1, "BiasBudgetHand"),
    lambda r: r.__setitem__("effective_hand_instances", 1),
    lambda r: r.__setitem__("effective_hand_instances", True),
    lambda r: r["shared_normal_runtime_bindings"].pop(),
    lambda r: r["shared_normal_runtime_bindings"][1]["tracking"]["allocator"].__setitem__("module", "develop_public_ycb_joint_admittance_r13"),
    lambda r: r["shared_normal_runtime_bindings"][1]["gate_only"]["allocator"].__setitem__("source_sha256", "bad"),
    lambda r: r["shared_normal_runtime_bindings"][0]["tracking"].__setitem__("same_closure_as_original", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["gate_only"].__setitem__("r13_proxy_other_members_match", False),
    lambda r: r["shared_normal_runtime_bindings"][0]["tracking"].__setitem__("method_source_sha256", "bad"),
    lambda r: r["shared_normal_runtime_bindings"][0].__setitem__("admit_probe_is_frozen_r16", False),
    lambda r: r["effective_task_phase_seconds"].__setitem__("CLOSE", 16.),
    lambda r: r.__setitem__("derived_controller_sha256", "other"),
    lambda r: r.__setitem__("catalogue_mass_prior_kg", .15),
])
def test_live_class_allocator_and_original_mechanics_drift_rejected(change):
    args = metadata_fixture()
    change(args[1])
    with pytest.raises((ValueError, KeyError)):
        audit.check_case_metadata(*args)


@pytest.fixture
def frozen_directory(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    protocol, baseline = protocol_pair()
    old_source = b"# synthetic inherited source\n"
    inherited = {"scripts/inherited_fixture.py": hashlib.sha256(old_source).hexdigest()}
    prior = baseline["protocol"]
    prior.update(format=audit.historical.FORMAT, source_changed_during_run=False,
        source_hashes_before=inherited, source_hashes_after=copy.deepcopy(inherited), schedule=[{"kind": "ycb_can"}])
    baseline.update(complete=True, rows=[{"success": False, "held_lift": False}])
    source_paths = (audit.SOURCE, audit.AUDITOR, audit.PURE, audit.HOOK, audit.historical.R19_SOURCE,
        "scripts/develop_public_ycb_joint_admittance_r13.py",
        "scripts/develop_public_ycb_constrained_preload_r14.py", "scripts/tactile_wrench_allocator.py")
    project = Path(audit.__file__).resolve().parents[1]
    sources = {path: (project / path).read_bytes() for path in source_paths}
    sources["scripts/inherited_fixture.py"] = old_source
    encode = lambda value: json.dumps(value, allow_nan=False).encode()
    sources[audit.R19_SUMMARY], sources[audit.R19_PROTOCOL] = encode(baseline), encode(prior)
    summary_sha = hashlib.sha256(sources[audit.R19_SUMMARY]).hexdigest()
    protocol_sha = hashlib.sha256(sources[audit.R19_PROTOCOL]).hexdigest()
    sources[audit.R19_DIRECTORY + "/verification_receipt.json"] = encode({"complete": True,
        "unknowns": 0, "summary_sha256": summary_sha})
    sources[audit.R19_DIRECTORY + "/strict_native_verification.json"] = encode({"accepted_evidence": True,
        "unknowns": 0, "summary_sha256": summary_sha})
    sources["results/r19_extended_budget_audit_20261022/receipt.json"] = encode({
        "all_evidence_consistent": True, "all_declared_command_bounds_satisfied": True,
        "all_native_close_spans_within_declared_deadline": True,
        "script_sha256": "66a1208428856b5d212206c4581d927961fe47f0c8543605198295c3b9738e5d",
        "summary_sha256": summary_sha, "protocol_sha256": protocol_sha})
    job = "results/jobs/r19_bias10_close32_seed3414_20261022/"
    sources[job + "status.json"] = encode({"state": "FINISHED", "terminal": True,
        "child_exit_observed": True, "returncode": 0})
    sources[job + "request.json"] = encode({"command": ["python", "-u", audit.historical.R19_SOURCE]})
    pins = {path: hashlib.sha256(sources[path]).hexdigest() for path in audit.FROZEN_INPUTS}
    declared_pins = dict(pins, **{audit.PURE: hashlib.sha256(sources[audit.PURE]).hexdigest(),
        audit.HOOK: hashlib.sha256(sources[audit.HOOK]).hexdigest()})
    constants = {"BUDGET_RAD": .10, "CLOSE_SECONDS": 32., "ARM_INTERPOLATION_SECONDS": 16.,
        "NATIVE_TIMESTEP_S": .002, "PROTOCOL_FORMAT": audit.FORMAT,
        "EXPECTED_R19_SUMMARY_SHA256": summary_sha, "PREFLIGHT_HASHES": declared_pins}
    sources[audit.SOURCE] = "".join(key + " = " + repr(value) + "\n" for key, value in constants.items()).encode()
    monkeypatch.setattr(audit, "FROZEN_INPUTS", pins)
    monkeypatch.setattr(audit, "FROZEN_SOURCE_SHA256", hashlib.sha256(sources[audit.SOURCE]).hexdigest())
    monkeypatch.setattr(audit.r17, "frozen_inputs", lambda *args: (.10, {}, {"native_actuation_timing_scope": "fixture"}))
    protocol[audit.DECLARATION].update(r19_comparison_summary_sha256=summary_sha,
        preflight_input_sha256=copy.deepcopy(declared_pins))
    protocol["source_hashes_before"] = {path: hashlib.sha256(value).hexdigest() for path, value in sources.items()}
    def archive():
        with zipfile.ZipFile(directory / "frozen_source.zip", "w") as output:
            for path, value in sources.items():
                output.writestr(path, value)
    archive()
    return directory, protocol, sources, archive


def test_source_closure_and_method_source_hashes_are_portable(frozen_directory):
    directory, protocol, _, _ = frozen_directory
    budget, _, source, baseline = audit.frozen_inputs(directory, protocol)
    assert budget == .10 and baseline["complete"]
    import shared_normal_controller_hooks as hooks
    for path, method in (("tracking", hooks.SharedNormalBudgetHand.regulate),
            ("gate_only", hooks.SharedNormalBudgetHand.current_preload_observation)):
        assert source["expected_method_bindings"][path]["method_source_sha256"] == hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


@pytest.mark.parametrize("change", ["missing_baseline", "changed_runner", "corrupt_zip", "unmanifested_zip", "missing_preflight", "missing_nonreceipt_preflight"])
def test_frozen_source_and_baseline_closure_reject_mutations(frozen_directory, change):
    directory, protocol, sources, archive = frozen_directory
    if change == "missing_baseline":
        protocol["source_hashes_before"].pop(audit.R19_SUMMARY)
    elif change == "changed_runner":
        sources[audit.SOURCE] += b"# changed source\n"
        protocol["source_hashes_before"][audit.SOURCE] = hashlib.sha256(sources[audit.SOURCE]).hexdigest()
    elif change == "corrupt_zip":
        sources[audit.PURE] += b"# mismatching archive bytes\n"
    elif change == "unmanifested_zip":
        sources["unmanifested.py"] = b"# hidden\n"
    elif change == "missing_preflight":
        protocol[audit.DECLARATION]["preflight_input_sha256"].pop(audit.R19_PROTOCOL)
    else:
        protocol[audit.DECLARATION]["preflight_input_sha256"].pop(audit.HOOK)
    archive()
    with pytest.raises((ValueError, KeyError)):
        audit.frozen_inputs(directory, protocol)
