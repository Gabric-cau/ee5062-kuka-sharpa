#!/usr/bin/env python3
"""Independent R20 provenance, native command/cache and static-target audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import audit_public_ycb_bias_budget as r17
import audit_public_ycb_close_deadline as r18
import audit_public_ycb_extended_budget as historical
import tactile_wrench_allocator as allocator


base = r17.base
FORMAT = "public_ycb_shared_normal_development_r20"
DECLARATION = "r20_shared_normal_experiment"
SOURCE = "scripts/develop_public_ycb_shared_normal_r20.py"
AUDITOR = "scripts/audit_public_ycb_shared_normal.py"
PURE = "scripts/shared_normal_tracking_targets.py"
HOOK = "scripts/shared_normal_controller_hooks.py"
R19_DIRECTORY = "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414"
R19_SUMMARY = R19_DIRECTORY + "/summary.json"
R19_PROTOCOL = R19_DIRECTORY + "/protocol.json"
FROZEN_INPUTS = {
    R19_SUMMARY: "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8",
    R19_PROTOCOL: "92cfa6fc266488ff79139d7232a907269a0519ca9324246afbf2b05017851989",
    R19_DIRECTORY + "/verification_receipt.json": "2b6261e9978c27dffd824617d5df4ffa1edca9392b6da9fee9eed179a4ab9863",
    R19_DIRECTORY + "/strict_native_verification.json": "afac02d33fc31e4d38168c8a07099049ef961e91028f9d83745ad40fe6c4b1a3",
    "results/r19_extended_budget_audit_20261022/receipt.json": "bc440c0e45f2aa9f612263cbe243af3c74e6213d06668cdb14db30b27c8c3822",
    "results/jobs/r19_bias10_close32_seed3414_20261022/status.json": "431985f891fb99265bcba087e797664bc0a7ff56668c1fcacd3afbd0f473e34e",
    "results/jobs/r19_bias10_close32_seed3414_20261022/request.json": "e7ae04041b4ec406be5751cbef959c7bf10c40dd3328e04e3890ce44eae9ee49",
}
FROZEN_SOURCE_SHA256 = "bd1e8399a0ee34767eb71c54f94bc3c3f7924e1af4b61c499feec72b7a7b433d"
FROZEN_PURE_SHA256 = "f33434846456e02c063dc3224bd2f03cff5812b218aeab7b7f8de5ad88633101"
FROZEN_HOOK_SHA256 = "45af05f935a7199d1668332bca3c715e61cdccea771cae81c5e3dba63ec7453a"
BUDGET_RAD, CLOSE_SECONDS, ARM_SECONDS = .10, 32., 16.
TARGET_FORMAT = "shared_normal_tracking_target_v1"
ACTIVE = ("thumb", "middle", "ring")
NUMERICAL_LIMITS = {"force_residual_max_n": 1e-5, "torque_residual_max_nm": 1e-6,
    "shared_patch_normal_residual_max_n": 1e-6, "inequality_excess_max_n": 1e-6,
    "nonnegative_bound_violation_max_n": 1e-8}
UNCHANGED_CONTRACTS = ("phase_seconds", "constrained_preload", "nominal_profiles", "maximum_attempts",
    "physics_profile", "grip_gate", "scoring", "latched_pressure_escape", "load_transfer_probe",
    "probe_confirmation", "wrench_evidence", "probe_hypothesis", "scalar_acquisition_tracking",
    "tracking_regulator_sha256", "tracking_regulator_source", "r16_bias_recovery_experiment",
    "r17_bias_budget_experiment", "joint_admittance")


def runner_preflight(source):
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PREFLIGHT_HASHES" for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError("R20 runner has no unique literal preflight mapping")
    result = {}
    for key, value in zip(assignments[0].value.keys, assignments[0].value.values):
        path = ast.literal_eval(key)
        if isinstance(value, ast.Name) and value.id == "EXPECTED_R19_SUMMARY_SHA256":
            expected = base.literal_assignment(source, value.id)
            if expected != FROZEN_INPUTS[R19_SUMMARY]:
                raise ValueError("Runner preflight summary constant differs")
        else:
            expected = ast.literal_eval(value)
        if not isinstance(path, str) or path in result or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid or duplicate R20 preflight entry")
        result[path] = expected
    return result


def check_protocol(protocol, baseline):
    prior = baseline["protocol"]
    declaration = protocol[DECLARATION]
    if (protocol["format"] != FORMAT or protocol["runner_revision"] != FORMAT
            or declaration["format"] != "r20_shared_normal_development_protocol_v1"
            or declaration["controller"] != "SharedNormalBudgetHand"
            or declaration["runner_revision"] != FORMAT
            or declaration["changed_parameters"] != {"tracking_and_gate_allocator": {
                "before": "r13_single_1_5g_then_fixed_1g_postcheck", "after": TARGET_FORMAT}}
            or declaration["r19_comparison_summary_sha256"] != FROZEN_INPUTS[R19_SUMMARY]):
        raise ValueError("Unknown or misbound R20 allocator experiment")
    for key, expected in {"additional_bias_limit_rad": BUDGET_RAD, "close_seconds": CLOSE_SECONDS,
            "arm_interpolation_seconds": ARM_SECONDS, "native_timestep_s": base.STEP_S}.items():
        base.equal(declaration[key], expected, "R20 declared " + key)
    if (protocol["joint_admittance"]["additional_bias_limit_rad"] != BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_SECONDS
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_SECONDS
            or "r19_extended_budget_experiment" in protocol
            or protocol["inherited_r19_extended_budget_experiment"] != prior["r19_extended_budget_experiment"]):
        raise ValueError("R20 active settings or historical R19 declaration were relabeled")
    if (declaration["active_r19_declaration"] != prior["r19_extended_budget_experiment"]
            or declaration["inherited_r19_declaration"] != prior["r19_extended_budget_experiment"]
            or declaration["inherited_r19_force_control"] != prior["force_control"]
            or protocol["inherited_r19_force_control"] != prior["force_control"]
            or declaration["effective_phase_seconds"] != prior["phase_seconds"]
            or declaration["effective_constrained_preload"] != prior["constrained_preload"]
            or declaration["allocator_format"] != TARGET_FORMAT
            or declaration["allocator_paths"] != ["tracking", "gate_only"]
            or declaration["allocator_source_sha256"] != FROZEN_PURE_SHA256
            or declaration["hook_source_sha256"] != FROZEN_HOOK_SHA256
            or declaration["lp_time_limit_per_solve_s"] != .1
            or declaration["baseline_task_success"] is not False or declaration["baseline_held_lift"] is not False
            or declaration["baseline_child_exit_observed"] is not True or declaration["baseline_child_returncode"] != 0
            or base.flag(declaration, "runtime_target_pose_or_qpos_added") is not False
            or base.flag(declaration, "physics_force_and_task_gates_modified") is not False
            or protocol["controller_class"] != "SharedNormalBudgetHand"
            or protocol["target_allocator_revision"] != FORMAT):
        raise ValueError("R20 allocator, actual-gate or historical outcome declaration differs")
    for key in UNCHANGED_CONTRACTS:
        if protocol[key] != prior[key]:
            raise ValueError("R20 changed an inherited physical/control/task contract: " + key)


def frozen_inputs(directory, protocol):
    if (base.digest(r17.__file__) != historical.FROZEN_R17_CORE_SHA256
            or base.digest(r18.__file__) != historical.FROZEN_R18_CORE_SHA256):
        raise ValueError("Frozen command/force/deadline audit core changed")
    budget, rates, source = r17.frozen_inputs(directory, protocol)
    base.equal(budget, BUDGET_RAD, "R20 active command budget")
    before = protocol["source_hashes_before"]
    pins = dict(FROZEN_INPUTS)
    pins.update({SOURCE: FROZEN_SOURCE_SHA256, PURE: FROZEN_PURE_SHA256,
        HOOK: FROZEN_HOOK_SHA256, historical.R19_SOURCE: historical.FROZEN_R19_SOURCE_SHA256})
    if any(not value or before.get(key) != value for key, value in pins.items()):
        raise ValueError("R20 frozen runner/allocator/R19 comparison identity differs")
    if before[AUDITOR] != base.digest(__file__):
        raise ValueError("R20 auditor is not bound to the runtime source snapshot")
    preflight = protocol[DECLARATION]["preflight_input_sha256"]
    if (not all(preflight.get(key) == value for key, value in FROZEN_INPUTS.items())
            or any(before.get(key) != value for key, value in preflight.items())):
        raise ValueError("R20 preflight dependency closure is incomplete")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(before):
            raise ValueError("R20 source ZIP has missing, duplicate or unmanifested entries")
        for key, expected in before.items():
            if hashlib.sha256(archive.read(key)).hexdigest() != expected:
                raise ValueError("R20 source archive bytes differ: " + key)
        archived = {key: json.loads(archive.read(key)) for key in FROZEN_INPUTS}
        runner_text = archive.read(SOURCE)
        if preflight != runner_preflight(runner_text):
            raise ValueError("R20 declared preflight is not the complete frozen runner mapping")
        method_bindings = {}
        for path, filename, class_name, name in (
                ("tracking", "scripts/develop_public_ycb_joint_admittance_r13.py", "JointAdmittanceProbeHand", "regulate"),
                ("gate_only", "scripts/develop_public_ycb_constrained_preload_r14.py", "ConstrainedPreloadHand", "current_preload_observation")):
            text = archive.read(filename).decode()
            klass = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == class_name)
            method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
            segment = "".join(text.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])
            method_bindings[path] = {"method_source_sha256": hashlib.sha256(segment.encode()).hexdigest(),
                "method_name": name, "method_source_file": filename, "method_source_file_sha256": before[filename]}
    baseline, prior = archived[R19_SUMMARY], archived[R19_PROTOCOL]
    if (baseline["complete"] is not True or baseline["protocol"] != prior
            or prior["format"] != historical.FORMAT or prior["source_changed_during_run"] is not False
            or prior["source_hashes_before"] != prior["source_hashes_after"]
            or any(before.get(key) != value for key, value in prior["source_hashes_before"].items())
            or len(baseline["rows"]) != 1 or len(prior["schedule"]) != 1):
        raise ValueError("R20 comparison is not the complete frozen R19 experiment")
    if base.digest(allocator.__file__) != before["scripts/tactile_wrench_allocator.py"]:
        raise ValueError("Independent hypothetical checker differs from the frozen allocator source")
    native = archived[R19_DIRECTORY + "/verification_receipt.json"]
    strict = archived[R19_DIRECTORY + "/strict_native_verification.json"]
    commands = archived["results/r19_extended_budget_audit_20261022/receipt.json"]
    status = archived["results/jobs/r19_bias10_close32_seed3414_20261022/status.json"]
    request = archived["results/jobs/r19_bias10_close32_seed3414_20261022/request.json"]
    if (native["complete"] is not True or native["unknowns"] != 0
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or any(receipt["summary_sha256"] != FROZEN_INPUTS[R19_SUMMARY] for receipt in (native, strict, commands))
            or any(commands[key] is not True for key in ("all_evidence_consistent", "all_declared_command_bounds_satisfied",
                "all_native_close_spans_within_declared_deadline"))
            or commands["script_sha256"] != "66a1208428856b5d212206c4581d927961fe47f0c8543605198295c3b9738e5d"
            or commands["protocol_sha256"] != FROZEN_INPUTS[R19_PROTOCOL]
            or status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or status["returncode"] != 0
            or request["command"][2] != historical.R19_SOURCE):
        raise ValueError("R19 native/strict/command or observed normal-exit provenance is incomplete")
    for key, expected in {"BUDGET_RAD": BUDGET_RAD, "CLOSE_SECONDS": CLOSE_SECONDS,
            "ARM_INTERPOLATION_SECONDS": ARM_SECONDS, "NATIVE_TIMESTEP_S": base.STEP_S,
            "PROTOCOL_FORMAT": FORMAT}.items():
        if base.literal_assignment(runner_text, key) != expected:
            raise ValueError("R20 executed constant differs: " + key)
    check_protocol(protocol, baseline)
    source.update(r20_source_sha256=before[SOURCE], r20_auditor_sha256=before[AUDITOR],
        shared_normal_source_sha256=before[PURE], shared_normal_hook_sha256=before[HOOK],
        r19_summary_sha256=FROZEN_INPUTS[R19_SUMMARY], inherited_r19_source_files=len(prior["source_hashes_before"]),
        expected_method_bindings=method_bindings)
    return budget, rates, source, baseline


def check_case_metadata(planned, row, protocol, baseline, method_bindings):
    if (any(row[key] != value for key, value in planned.items()) or row["runner_error"]
            or row.get("missing_runtime_controller_binding", False) is not False
            or row.get("missing_runtime_allocator_binding", False) is not False):
        raise ValueError("R20 planned case, actual factory/allocator binding or runner failed")
    expected = dict(baseline["protocol"]["schedule"][0], runner_revision=FORMAT,
        controller_class="SharedNormalBudgetHand", target_allocator_revision=FORMAT)
    if planned != expected:
        raise ValueError("R20 changed the R19 planned identity, budget or clocks")
    if (row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]
            or row["target_allocator_revision"] != FORMAT):
        raise ValueError("R20 actual task/runner phase or allocator revision differs")
    count = row["effective_hand_instances"]
    if (type(count) is not int or not 1 <= count <= protocol["maximum_attempts"] or count != len(row["attempts"])
            or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * count
            or row["effective_hand_controller_types"] != ["SharedNormalBudgetHand"] * count
            or len(row["shared_normal_runtime_bindings"]) != count):
        raise ValueError("R20 live hand/retry class, count or immutable .10rad budget differs")
    prior = baseline["rows"][0]
    if (row["derived_controller_sha256"] != prior["derived_controller_sha256"]
            or row["arm_source_transform"] != prior["arm_source_transform"]
            or row["catalogue_mass_prior_kg"] != prior["catalogue_mass_prior_kg"]):
        raise ValueError("R20 executed arm controller, transform or mass prior differs from R19")
    for binding in row["shared_normal_runtime_bindings"]:
        if (binding["controller_class"] != "SharedNormalBudgetHand" or binding["additional_bias_limit_rad"] != BUDGET_RAD
                or any(binding[key] is not True for key in
                    ("writer_is_frozen_r16", "after_step_is_frozen_r16", "admit_probe_is_frozen_r16"))
                or not isinstance(binding["python_version"], list) or len(binding["python_version"]) != 3
                or any(type(value) is not int or value < 0 for value in binding["python_version"])):
            raise ValueError("R20 runtime writer/admission or instance budget binding differs")
        for path in ("tracking", "gate_only"):
            item = binding[path]
            if (any(item[key] != expected for key, expected in method_bindings[path].items())
                    or any(item[key] is not True for key in ("same_code_as_original", "same_closure_as_original",
                        "other_globals_match_except_warning_registry", "allocator_is_shared_normal"))
                    or item["allocator"] != {"module": "shared_normal_tracking_targets",
                        "qualname": "allocate_tracking_targets", "source_sha256": FROZEN_PURE_SHA256}
                    or (path == "gate_only" and item["r13_proxy_other_members_match"] is not True)
                    or not isinstance(item["method_bytecode_sha256"], str)
                    or len(item["method_bytecode_sha256"]) != 64
                    or any(value not in "0123456789abcdef" for value in item["method_bytecode_sha256"])):
                raise ValueError("R20 live allocator-path code, closure or global binding differs: " + path)


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid static-target array: " + name)
    return result


def finite_tree(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            base.finite(item, "recorded target numeric field")


def check_target(target):
    """Check every recorded target without claiming a missing ray witness exists."""
    finite_tree(target)
    if (target["format"] != TARGET_FORMAT or type(target["feasible"]) is not bool
            or base.flag(target, "runtime_target_state_used") is not False
            or not isinstance(target["reason"], str) or not target["reason"]
            or target["solver_time_limit_per_lp_s"] != .1):
        raise ValueError("Unknown static-target version, scope or time limit")
    checks = target.get("numerical_checks")
    if checks is not None:
        if set(checks) != set(NUMERICAL_LIMITS):
            raise ValueError("Incomplete numerical-check schema")
        for key, limit in NUMERICAL_LIMITS.items():
            value = base.finite(checks[key], "recorded " + key)
            if value < 0. or (target["feasible"] and value > limit):
                raise ValueError("Accepted static-target residual exceeds bound: " + key)
    if not target["feasible"] and not target.get("patches"):
        return False
    if (target["reason"] not in ("STATIC_SHARED_NORMAL_TRACKING_TARGET_ALLOCATED",
            "CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE")
            or (target["feasible"] and target["reason"] != "STATIC_SHARED_NORMAL_TRACKING_TARGET_ALLOCATED")
            or type(target["solver_status"]) is not int or target["solver_status"] != 0 or checks is None
            or base.flag(target, "hypothetical_target_fixed_normal_1g_feasible") is not True
            or base.flag(target, "hypothetical_target_fixed_normal_1_5g_feasible") is not True):
        raise ValueError("Feasible shared-normal target lacks its declared checks")
    residuals = [finite_array(target[key], (6,), key) for key in
        ("target_wrench_residual", "target_wrench_residual_1g")]
    for group, key in ((slice(0, 3), "force_residual_max_n"), (slice(3, 6), "torque_residual_max_nm")):
        base.equal(checks[key], max(float(np.max(abs(row[group]))) for row in residuals),
            "Recorded residual aggregate", 1e-12)
    postchecks = target["independent_fixed_normal_checks"]
    if set(postchecks) != {"1.0", "1.5"}:
        raise ValueError("Missing hypothetical 1g/1.5g postcheck")
    for check in postchecks.values():
        if (check["measured_feasible"] is not True or type(check["measured_solver_status"]) is not int
                or check["measured_solver_status"] != 0):
            raise ValueError("Accepted target has a rejected hypothetical fixed-normal postcheck")
        residual = finite_array(check["measured_wrench_residual"], (6,), "hypothetical residual")
        if np.max(abs(residual[:3])) > 1e-5 or np.max(abs(residual[3:])) > 1e-6:
            raise ValueError("Hypothetical fixed-normal postcheck residual exceeds bound")
    patches = target["patches"]
    if not patches or {patch["finger"] for patch in patches} != set(ACTIVE):
        raise ValueError("Shared-normal target lacks three observed active fingers")
    indices, loads = set(), {finger: 0. for finger in allocator.CHANNELS}
    for patch in patches:
        index = patch["raw_input_index"]
        if (type(index) is not int or type(patch["input_index"]) is not int
                or index < 0 or index != patch["input_index"] or index in indices):
            raise ValueError("Static-target raw patch binding is duplicated or inconsistent")
        indices.add(index)
        finite_array(patch["position_world_m"], (3,), "patch position")
        normal = finite_array(patch["normal_on_object_world"], (3,), "patch normal")
        if not .5 <= np.linalg.norm(normal) <= 1.5:
            raise ValueError("Invalid static-target normal direction")
        if base.finite(patch["measured_pressure_n"], "observed patch pressure") < .05:
            raise ValueError("Static-target patch is below the unchanged tracking entry floor")
        load = base.finite(patch["target_normal_n"], "target patch normal")
        if load < 0.:
            raise ValueError("Negative target patch normal")
        loads[patch["finger"]] += load
    if set(target["targets_n"]) != set(allocator.CHANNELS):
        raise ValueError("Incomplete target channel inventory")
    for finger, load in loads.items():
        base.equal(target["targets_n"][finger], load, "Patch/channel target normal sum", 1e-8)
        if load > 6. + 1e-6 or (finger in ACTIVE and load < .75 - 1e-6) or (finger == "palm" and load > 1e-6):
            raise ValueError("Target channel violates the original cap/floor")
    mu = finite_array(target["effective_patch_mu"], (len(patches),), "effective friction")
    for index, patch in enumerate(patches):
        friction = base.finite(patch.get("friction_coefficient", .8), "patch friction")
        if friction < 0.:
            raise ValueError("Negative patch friction")
        base.equal(mu[index], min(.8, friction) * .85, "Original friction safety factor", 1e-12)
    reference = {finger: base.finite(target["reference_n"][finger], "reference pressure") for finger in ACTIVE}
    if any(value < 0. for value in reference.values()):
        raise ValueError("Negative target reference pressure")
    objective = target["objective"]
    expected = {"channel_reference_l1_n": sum(abs(loads[finger] - reference[finger]) for finger in ACTIVE),
        "patch_measured_l1_n": sum(abs(patch["target_normal_n"] - patch["measured_pressure_n"]) for patch in patches),
        "primary_normal_regularizer": 1e-4 * sum(loads.values()),
        "secondary_normal_regularizer": 1e-8 * sum(loads.values())}
    if set(objective) != set(expected):
        raise ValueError("Unknown target objective schema")
    for key, value in expected.items():
        base.equal(objective[key], value, "Recorded target objective component", 1e-8)
    return target["feasible"]


def check_association(target, association):
    if association["accepted"] is not True:
        raise ValueError("Feasible target has no accepted current association")
    center = finite_array(association["estimated_com_world_m"], (3,), "estimated COM")
    raw = {patch["input_index"]: patch for patch in association["patches"]}
    if len(raw) != len(association["patches"]):
        raise ValueError("Duplicate associated raw patch")
    for patch in target["patches"]:
        source = raw[patch["raw_input_index"]]
        if source["finger"] != patch["finger"]:
            raise ValueError("Target patch channel differs from association")
        for key in ("position_world_m", "normal_on_object_world"):
            if not np.allclose(finite_array(source[key], (3,), key), patch[key], rtol=0., atol=1e-10):
                raise ValueError("Target patch geometry differs from its current association")
        base.equal(source["measured_pressure_n"], patch["measured_pressure_n"], "Associated measured normal", 1e-10)
    return center


def audit_targets(joint_events, preload_events, mass_kg, *, samples_per_path=6):
    mass = base.finite(mass_kg, "catalogue mass prior")
    if mass <= 0. or type(samples_per_path) is not int or not 1 <= samples_per_path <= 6:
        raise ValueError("Invalid target-audit mass or deterministic sample budget")
    counts, pools = {}, {}
    for path, rows, field in (("tracking", joint_events, "current_target"),
            ("gate_only", preload_events, "current_target_lp")):
        counts[path] = {"records_with_target": 0, "feasible_targets": 0, "infeasible_targets": 0,
            "association_bound_feasible_targets": 0, "preallocator_gate_rejections": 0}
        pools[path] = []
        for ordinal, row in enumerate(rows):
            target = row.get(field)
            if not target:
                continue
            finite_tree(target)
            counts[path]["records_with_target"] += 1
            if "format" not in target:
                reason = target["reason"]
                known_gate = reason in ("CURRENT_GATE_ASSOCIATION_INVALID", "ORIGINAL_EMERGENCY_PRECEDENCE",
                    "MODE_DISALLOWS_NOMINAL_ADMITTANCE") or reason.startswith("CURRENT_ASSOCIATION_")
                if target["feasible"] is not False or not known_gate or target.get("patches"):
                    raise ValueError("Unversioned target is not an original caller-gate rejection")
                counts[path]["preallocator_gate_rejections"] += 1
                continue
            if not check_target(target):
                counts[path]["infeasible_targets"] += 1
                continue
            association = target["association"] if path == "tracking" else row["original_probe_snapshot"]["sampled_public_mesh_association"]
            center = check_association(target, association)
            if path == "gate_only":
                base.equal(row["original_probe_snapshot"]["catalogue_weight_n"], mass * 9.81,
                    "Gate-only mass prior", 1e-10)
            counts[path]["feasible_targets"] += 1
            counts[path]["association_bound_feasible_targets"] += 1
            pools[path].append((ordinal, row["timestamp_s"], target, center))
        if not counts[path]["records_with_target"]:
            raise ValueError("Missing required allocator-path target records: " + path)
    sampled = []
    for path, pool in pools.items():
        selected = sorted(set(np.linspace(0, len(pool) - 1, min(samples_per_path, len(pool)), dtype=int))) if pool else []
        for index in selected:
            ordinal, timestamp, target, center = pool[index]
            hypothetical = [dict(patch, measured_pressure_n=patch["target_normal_n"]) for patch in target["patches"]]
            margins = {}
            for margin in (1., 1.5):
                result = allocator.allocate_contact_wrench(hypothetical, center, mass, gravity_margin=margin)
                if result["measured_feasible"] is not True or result["measured_solver_status"] != 0:
                    raise ValueError("Independent sampled hypothetical fixed-normal solve failed")
                margins[str(margin)] = {key: result[key] for key in
                    ("measured_feasible", "measured_solver_status", "measured_wrench_residual")}
            sampled.append({"path": path, "stream_record_index": ordinal, "timestamp_s": timestamp,
                "fixed_normal_checks": margins})
    return {"full_stream_record_checks": counts, "independent_sampled_checks": sampled,
        "independent_sampled_target_count": len(sampled), "independent_maximum_target_samples": 12,
        "all_full_stream_record_checks_consistent": True,
        "all_sampled_hypothetical_fixed_normal_checks_passed": True,
        "independent_ray_solution_or_optimality_proven": False,
        "scope": "Every recorded nonempty target: version, recorded residuals, exact patch/channel normals, original cap/floor/friction, objective components and association inputs. Deterministic evenly spaced feasible records per path (at most six each) receive independent fixed-normal 1g/1.5g feasibility solves using the recorded estimated COM and catalogue mass prior. No saved ray witness exists; full-stream residual equality and shared-normal equality remain recorded diagnostics, not independently reconstructed ray solutions. Hypothetical targets do not establish actual measured 1g, probe admission, equilibrium or task success."}


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = base.checked_path(directory, "summary.json"), base.checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    initial_hashes = {path: base.digest(path) for path in (summary_path, protocol_path, directory / "frozen_source.zip")}
    if (summary["complete"] is not True or summary["protocol"] != protocol or protocol["format"] != FORMAT
            or summary["runner_revision"] != FORMAT or summary["controller_class"] != "SharedNormalBudgetHand"
            or summary["target_allocator_revision"] != FORMAT
            or summary["close_seconds"] != CLOSE_SECONDS or summary["arm_interpolation_seconds"] != ARM_SECONDS
            or summary["additional_bias_limit_rad"] != BUDGET_RAD):
        raise ValueError("Expected a complete consistently bound actual-format R20 experiment")
    budget, rates, source, baseline = frozen_inputs(directory, protocol)
    if len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1:
        raise ValueError("R20 requires the one isolated R19 comparison case")
    row, planned = summary["rows"][0], protocol["schedule"][0]
    check_case_metadata(planned, row, protocol, baseline, source["expected_method_bindings"])
    case = base.checked_path(directory, row["path"])
    paths = {key: base.checked_path(case, filename) for key, filename in (
        ("metadata", "metadata.json"), ("scene", "scene.xml"), ("controller", "derived_controller.py"), ("result", "result.json"))}
    initial_hashes.update({path: base.digest(path) for path in paths.values()})
    if json.loads(paths["metadata"].read_text()) != row or base.digest(paths["controller"]) != row["derived_controller_sha256"]:
        raise ValueError("R20 case metadata or executed controller hash differs")
    result = json.loads(paths["result"].read_text())
    if (result["attempts"] != row["attempts"] or result["runner_revision"] != FORMAT
            or result["controller_revision"] != "public_ycb_bias_budget_development_r17"
            or result["controller_class"] != "SharedNormalBudgetHand" or result["target_allocator_revision"] != FORMAT
            or result["close_seconds"] != CLOSE_SECONDS or result["arm_interpolation_seconds"] != ARM_SECONDS
            or result["additional_bias_limit_rad"] != BUDGET_RAD):
        raise ValueError("R20 actual class/allocator and inherited mechanical revision result binding differs")
    if (row["scene_unchanged_since_load"] is not True or row["scene_sha256_before_load"] != base.digest(paths["scene"])
            or row["scene_sha256_at_finalize"] != base.digest(paths["scene"])
            or row["scene_sha256_before_load"] != baseline["rows"][0]["scene_sha256_before_load"]):
        raise ValueError("R20 runtime-loaded scene differs from R19")
    declaration = protocol[DECLARATION]
    if (declaration["baseline_scene_sha256"] != baseline["rows"][0]["scene_sha256_before_load"]
            or declaration["baseline_derived_controller_sha256"] != baseline["rows"][0]["derived_controller_sha256"]):
        raise ValueError("R20 declared baseline scene/controller differs")
    model = load_evidence_model(paths["scene"])
    base.equal(model.opt.timestep, base.STEP_S, "R20 native timestep", 1e-12)
    records = {key: base.checked_stream(case, row["artifacts"][key]) for key in
        ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
    if row["native_audit"]["native_actuation_samples"] != len(records["native"]):
        raise ValueError("R20 actuation denominator differs from native samples")
    spans = r18.native_close_spans(records["native"], protocol["maximum_attempts"])
    receipt = r17.reconstruct_budget_case(records["native"], records["bias_recovery"], records["joint_admittance"],
        base.joint_specs(model, protocol["nominal_profiles"][row["kind"]]), budget_rad=budget,
        **rates, preload_events=records["preload"], probe_events=records["load_transfer_probe"], native_audit=row["native_audit"])
    receipt.update(case=row["path"], scene_sha256=base.digest(paths["scene"]),
        metadata_sha256=base.digest(paths["metadata"]), result_sha256=base.digest(paths["result"]),
        derived_controller_sha256=base.digest(paths["controller"]), actual_controller_class="SharedNormalBudgetHand",
        effective_hand_instances=row["effective_hand_instances"], effective_hand_budgets_rad=row["effective_hand_budgets_rad"],
        native_close_spans=spans, all_native_close_spans_within_declared_deadline=True,
        runtime_allocator_path_bindings_checked=2 * row["effective_hand_instances"],
        runtime_method_bytecode_independently_reconstructed=False,
        inputs={key: row["artifacts"][key] for key in records},
        actuation=r17.audit_actuation(records["native"], row["native_audit"]["native_actuation_metadata"], model,
            expected_timing_scope=source["native_actuation_timing_scope"]),
        static_targets=audit_targets(records["joint_admittance"], records["preload"], row["catalogue_mass_prior_kg"]),
        task_success=row["success"], held_lift=row["held_lift"],
        task_outcome_scope="Original reported actual measured/task gates, not independently rescored by this command/static-target auditor")
    for key, artifact in row["artifacts"].items():
        if key in records:
            path = base.checked_path(case, artifact["path"])
            if path.stat().st_size != artifact["bytes"] or base.digest(path) != artifact["sha256"]:
                raise ValueError("R20 input artifact changed during audit")
    if any(base.digest(path) != expected for path, expected in initial_hashes.items()):
        raise ValueError("R20 source or small input changed during audit")
    return {"format": "r20_shared_normal_command_native_and_static_target_audit_v1", "simulation_only": True,
        "all_evidence_consistent": receipt["accepted_evidence_consistency"],
        "all_declared_command_bounds_satisfied": receipt["original_intersection_satisfied_for_all_audited_writes"],
        "all_native_close_spans_within_declared_deadline": True,
        "all_recorded_static_target_checks_consistent": True, "all_sampled_hypothetical_target_checks_passed": True,
        "additional_bias_limit_rad": budget, "close_seconds": CLOSE_SECONDS, "arm_interpolation_seconds": ARM_SECONDS,
        "cases": [receipt], "frozen_source": source, "rates_per_s": rates,
        "summary_sha256": initial_hashes[summary_path], "protocol_sha256": initial_hashes[protocol_path],
        "script_sha256": base.digest(__file__), "target_qpos_writes_independently_proven": False,
        "physical_prefix_independently_compared": False,
        "scope": "R20 actual SharedNormalBudgetHand allocator intervention at unchanged R19 .10rad/CLOSE32/arm16. Frozen R17 adjacent-native command/cache-force and R18 deadline algorithms are reused without protocol/global mutation. Actual per-hand two-path source/global/closure declarations are bound to frozen sources; Python-version-specific bytecode hashes are recorded but not independently reconstructed across Python versions. Static-target full-record checks and bounded independent hypothetical fixed-normal samples are distinct. No ray-solution/optimality proof, task rescoring, independent absence-of-qpos-write proof, normal-exit proof for this current R20 process or randomized-dataset rate. Excluded scalar/release/safety modes do not prove .10rad biased bounds."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = audit_directory(args.directory)
        consistent = receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"]
        status = 0 if consistent else 1
    except (ValueError, KeyError, OSError, TypeError, StopIteration, zipfile.BadZipFile) as error:
        receipt, status = {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
            "error": str(error), "script_sha256": base.digest(__file__)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": base.digest(args.output), "exit_code": status}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
