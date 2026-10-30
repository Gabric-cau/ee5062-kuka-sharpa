#!/usr/bin/env python3
"""One provisional-admission intervention at the frozen R20 allocator and clocks."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import develop_public_ycb_shared_normal_r20 as r20
import provisional_preload_controller_hooks as policy

BUDGET_RAD = .10
CLOSE_SECONDS = 32.
ARM_INTERPOLATION_SECONDS = 16.
NATIVE_TIMESTEP_S = .002
PROTOCOL_FORMAT = "public_ycb_provisional_preload_development_r21"
TARGET_ALLOCATOR_REVISION = "public_ycb_shared_normal_development_r20"
CONTROLLER_CLASS = "ProvisionalPreloadBudgetHand"
AST_HASH_FORMAT = "canonical_ast_json_v1_ignore_empty_type_params"
CONTROLLER_REVISION_SCOPE = "Inherited R17 write/native mechanics; R20 shared-normal allocators; R21 provisional pre-motion admission only"
R20_DIRECTORY = PROJECT / "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414"
R20_SUMMARY = R20_DIRECTORY / "summary.json"
EXPECTED_R20_SUMMARY_SHA256 = "a47ab784ed6b0879f149333a08e02928b5ae5430ec93af76af67de02ef63dd7b"
R20_JOB = "results/jobs/r20_shared_normal_bias10_close32_seed3414_20261022"
R20_FAILED_AUDIT = "results/r20_shared_normal_audit_20261022/receipt.json"
R20_POSTRUN_AUDIT = "results/r20_shared_normal_postrun_audit_20261022/receipt.json"
POSTRUN_FORMAT = "r20_shared_normal_postrun_command_native_and_static_target_audit_v1"
FROZEN_R20_AUDITOR_SHA256 = "c517348408603e70a6432e503ba19b186a4c3773c2336e2830b4ba0e9b34116e"
PREFLIGHT_HASHES = {
    "scripts/develop_public_ycb_shared_normal_r20.py": "bd1e8399a0ee34767eb71c54f94bc3c3f7924e1af4b61c499feec72b7a7b433d",
    "scripts/audit_public_ycb_shared_normal.py": FROZEN_R20_AUDITOR_SHA256,
    "scripts/provisional_preload_controller_hooks.py": "f3578add08ac32bdc68789778da82ef32963b51737c433d5cce7f9c34cda5991",
    "tests/test_provisional_preload_controller_hooks.py": "8e93937e155815caed81010e26f14aec85640b37cb61923f77f6500ddc1004cd",
    "docs/PROVISIONAL_PRELOAD_POLICY_20261022.md": "0386863034b62fd32e1c89100bd9f39c1938ea9ca9d1c9a9d7683baa8da1d2c7",
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/summary.json": EXPECTED_R20_SUMMARY_SHA256,
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/protocol.json": "18f7982dc5e2513e7aa9947af9df5e4086634753858518adaff1f3370349707a",
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/verification_receipt.json": "302af5189b5f475fa2903dae49e69cb088da624b807160f083ff4fb3e1537e30",
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/strict_native_verification.json": "0b81b3c048730ccf4a5d46f857afc28c91745b8ddbb47f4602cc7d2239232897",
    "results/r20_shared_normal_audit_20261022/receipt.json": "aa456ddbcd40accab48be87b650d2c4deaa71917d6838991b7238b9c8c07f740",
    "results/jobs/r20_shared_normal_bias10_close32_seed3414_20261022/status.json": "ba01295e2222f262f890a687858869b8510d671eb1d3a41599cec7739e562ba8",
    "results/jobs/r20_shared_normal_bias10_close32_seed3414_20261022/request.json": "01f4d808fc555a56dadc2efd47bcf2474611a5e22f598cdaba1c55efe8e3f0e9",
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/scene.xml": "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e",
    "results/public_ycb_shared_normal_20261022_r20_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/derived_controller.py": "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485",
    "scripts/audit_public_ycb_shared_normal_postrun.py": "a9bde9be7497a7302e4eea71ed1bf33d38dd4945b1b72df3abb1cbcf3231e75b",
    "tests/test_public_ycb_shared_normal_postrun_audit.py": "95f32ee4602cccb25aee3765230f5ce2bb4bd0d4ef7d81ef259db496d6861e6f",
    "results/r20_shared_normal_postrun_audit_20261022/receipt.json": "7c7848dfa647d312b555f09041d56b9d1a2bbf712f26f49a2d5dd9a8c025d0d0",
}
NEW_SOURCES = ("scripts/develop_public_ycb_provisional_preload_r21.py",
    "tests/test_public_ycb_provisional_preload_r21.py", "docs/R21_PROVISIONAL_PRELOAD_20261022.md")
OPTIONAL_AUDIT_SOURCES = ("scripts/audit_public_ycb_provisional_preload.py",
    "tests/test_public_ycb_provisional_preload_audit.py")


_r17_proxy = SimpleNamespace(**{**vars(r20.r19.r17), "controller_for": policy.controller_for})
_controller_factory = FunctionType(r20.r19.controller_for.__code__,
    dict(r20.r19.controller_for.__globals__, r17=_r17_proxy), r20.r19.controller_for.__name__,
    r20.r19.controller_for.__defaults__, r20.r19.controller_for.__closure__)


def controller_for(*args, **kwargs):
    return _controller_factory(*args, **kwargs)


def _require_final_pins():
    if any(not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value) for value in PREFLIGHT_HASHES.values()):
        raise ValueError("R21 cannot run before all final R20 postrun audit hashes are pinned")


def validate_inputs(profile_path):
    _require_final_pins()
    sha256 = r20.r19.r17.r16.sha256
    for relative, expected in PREFLIGHT_HASHES.items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R21 preflight input SHA256 mismatch: " + relative)
    inherited = r20.validate_inputs(profile_path)
    baseline = json.loads(R20_SUMMARY.read_text())
    protocol, rows = baseline["protocol"], baseline["rows"]
    before = protocol["source_hashes_before"]
    if (baseline["complete"] is not True or baseline["runner_errors"]
            or protocol != json.loads((R20_DIRECTORY / "protocol.json").read_text())
            or protocol["format"] != TARGET_ALLOCATOR_REVISION
            or protocol["r20_shared_normal_experiment"] != inherited
            or protocol["source_changed_during_run"] is not False
            or len(before) != 110 or before != protocol["source_hashes_after"]
            or len(rows) != 1 or len(protocol["schedule"]) != 1):
        raise ValueError("R21 requires the complete, unchanged failed R20 baseline")
    with zipfile.ZipFile(R20_DIRECTORY / "frozen_source.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(before):
            raise ValueError("R20 source ZIP manifest differs")
        for relative, expected in before.items():
            if sha256(PROJECT / relative) != expected or hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError("Frozen R20 inherited source mismatch: " + relative)
    row = rows[0]
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["success"] is not False
            or row["held_lift"] is not False or row["runner_error"] is not None
            or row["derived_controller_sha256"] != inherited["baseline_derived_controller_sha256"]
            or row["scene_sha256_before_load"] != inherited["baseline_scene_sha256"]
            or row["scene_sha256_at_finalize"] != row["scene_sha256_before_load"]
            or row["scene_unchanged_since_load"] is not True
            or row["effective_hand_instances"] != len(row["attempts"])
            or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * len(row["attempts"])
            or row["effective_hand_controller_types"] != ["SharedNormalBudgetHand"] * len(row["attempts"])
            or row["effective_task_phase_seconds"] != protocol["phase_seconds"]
            or row["effective_runner_phase_seconds"] != protocol["phase_seconds"]):
        raise ValueError("R20 task failure, source, scene, clock or hand binding differs")
    native = json.loads((R20_DIRECTORY / "verification_receipt.json").read_text())
    strict = json.loads((R20_DIRECTORY / "strict_native_verification.json").read_text())
    failed = json.loads((PROJECT / R20_FAILED_AUDIT).read_text())
    postrun = json.loads((PROJECT / R20_POSTRUN_AUDIT).read_text())
    status = json.loads((PROJECT / R20_JOB / "status.json").read_text())
    request = json.loads((PROJECT / R20_JOB / "request.json").read_text())
    command = request["command"]
    if (native["complete"] is not True or native["unknowns"] != 0
            or native["task_successes"] != 0 or native["held_lifts"] != 0
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0 or strict["strict_successes"] != 0
            or any(value["summary_sha256"] != EXPECTED_R20_SUMMARY_SHA256 for value in (native, strict, postrun))
            or failed != {"format": "r20_shared_normal_audit_failure_v1", "accepted_evidence": False,
                "error": "Unversioned target is not an original caller-gate rejection", "script_sha256": FROZEN_R20_AUDITOR_SHA256}
            or postrun["format"] != POSTRUN_FORMAT
            or postrun["inherited_frozen_auditor_sha256"] != FROZEN_R20_AUDITOR_SHA256
            or postrun["script_sha256"] != PREFLIGHT_HASHES["scripts/audit_public_ycb_shared_normal_postrun.py"]
            or postrun["original_failed_receipt_sha256"] != PREFLIGHT_HASHES[R20_FAILED_AUDIT]
            or postrun["all_evidence_consistent"] is not True
            or postrun["all_declared_command_bounds_satisfied"] is not True
            or postrun["all_native_close_spans_within_declared_deadline"] is not True
            or postrun["all_recorded_static_target_checks_consistent"] is not True
            or postrun["all_sampled_hypothetical_target_checks_passed"] is not True
            or postrun["protocol_sha256"] != PREFLIGHT_HASHES[R20_DIRECTORY.relative_to(PROJECT).as_posix() + "/protocol.json"]
            or postrun["extension_inputs_unchanged"] is not True
            or postrun["postrun_extension"] is not True or postrun["original_frozen_auditor_succeeded"] is not False
            or postrun["original_failed_audit_overwritten"] is not False
            or postrun["original_runtime_source_modified"] is not False
            or native["frozen_source_sha256"] != sha256(R20_DIRECTORY / "frozen_source.zip")
            or status["state"] != "FINISHED" or status["terminal"] is not True
            or status["returncode"] != 0 or status["child_exit_observed"] is not True
            or request["supervisor_sha256"] != before["scripts/run_supervised_experiment.py"]
            or "scripts/develop_public_ycb_shared_normal_r20.py" not in command
            or command[command.index("--output") + 1] != R20_DIRECTORY.relative_to(PROJECT).as_posix()
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]):
        raise ValueError("R20 native/strict/postrun or observed normal-exit provenance differs")
    return {"format": "r21_provisional_preload_development_protocol_v1", "controller": CONTROLLER_CLASS,
        "runner_revision": PROTOCOL_FORMAT, "target_allocator_revision": TARGET_ALLOCATOR_REVISION,
        "admission_policy_revision": PROTOCOL_FORMAT,
        "additional_bias_limit_rad": BUDGET_RAD, "close_seconds": CLOSE_SECONDS,
        "arm_interpolation_seconds": ARM_INTERPOLATION_SECONDS, "native_timestep_s": NATIVE_TIMESTEP_S,
        "changed_parameters": {"pre_motion_preload_reasons.CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE":
            {"before": "blocks_ready", "after": "observed_but_deferred_to_unchanged_probe_confirmation_and_held_policy"}},
        "active_r20_declaration": copy.deepcopy(inherited),
        "active_r19_declaration": copy.deepcopy(inherited["active_r19_declaration"]),
        "inherited_r20_declaration": copy.deepcopy(protocol["r20_shared_normal_experiment"]),
        "inherited_r20_force_control": protocol["force_control"],
        "baseline_task_success": False, "baseline_held_lift": False,
        "baseline_child_exit_observed": True, "baseline_child_returncode": 0,
        "baseline_original_command_audit_passed": False,
        "baseline_postrun_extension_format": POSTRUN_FORMAT,
        "baseline_scene_sha256": row["scene_sha256_before_load"],
        "baseline_derived_controller_sha256": row["derived_controller_sha256"],
        "r20_comparison_summary_sha256": EXPECTED_R20_SUMMARY_SHA256,
        "preflight_input_sha256": copy.deepcopy(PREFLIGHT_HASHES),
        "policy_hook_source_sha256": PREFLIGHT_HASHES["scripts/provisional_preload_controller_hooks.py"],
        "deferred_pre_motion_reason": policy.DEFERRED_REASON,
        "actual_fixed_normal_1g_observation_retained": True,
        "actual_fixed_normal_1g_confirmation_and_held_policy_unchanged": True,
        "all_other_preload_reasons_and_recovery_inhibition_unchanged": True,
        "target_allocator_and_write_mechanics_unchanged": True,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_strict_task_gates_modified": False,
        "scope": "One same-can same-seed pre-motion admission intervention. Historical counterfactual readiness does not predict a fresh admission, probe confirmation, dynamic lift or randomized-dataset success."}


def snapshot(profile_path):
    _require_final_pins()
    values = r20.snapshot(profile_path)
    for relative in (*PREFLIGHT_HASHES, *NEW_SOURCES):
        values[relative] = r20.r19.r17.r16.sha256(PROJECT / relative)
    for relative in OPTIONAL_AUDIT_SOURCES:
        if (PROJECT / relative).is_file():
            values[relative] = r20.r19.r17.r16.sha256(PROJECT / relative)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r20.tagged_protocol(value, declaration["active_r20_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": CONTROLLER_REVISION_SCOPE,
        "target_allocator_revision": TARGET_ALLOCATOR_REVISION, "admission_policy_revision": PROTOCOL_FORMAT}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop("r20_shared_normal_experiment", None)
        protocol.update(active, format=PROTOCOL_FORMAT,
            r21_provisional_preload_experiment=copy.deepcopy(declaration),
            inherited_r20_shared_normal_experiment=copy.deepcopy(declaration["inherited_r20_declaration"]),
            inherited_r20_force_control=declaration["inherited_r20_force_control"],
            force_control="R21 provisional pre-motion readiness defers only measured fixed-normal 1g infeasibility. Its observed value remains recorded; original fresh admission recheck, recovery inhibition, actual-1g probe confirmation/held policy, strict task gates and R20 shared-normal .10rad/CLOSE32/arm16 control remain.")
        for planned in protocol["schedule"]:
            planned.update(active)
    for row in value.get("rows", []):
        row.update(active)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(active)
    return value


def _ast_sha(tree):
    def canonical(value):
        if isinstance(value, ast.AST):
            return {"node_type": type(value).__name__, "fields": [[name, canonical(field)]
                for name, field in ast.iter_fields(value) if name != "type_params" or field]}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    encoded = json.dumps(canonical(tree), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _method_binding(function):
    path = Path(inspect.getsourcefile(function)).resolve()
    return {"module": function.__module__, "qualname": function.__qualname__,
        "method_source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
        "method_source_file": path.relative_to(PROJECT).as_posix(),
        "method_source_file_sha256": r20.r19.r17.r16.sha256(path),
        "method_bytecode_sha256": hashlib.sha256(function.__code__.co_code).hexdigest()}


def runtime_binding(hand):
    result = r20.runtime_binding(hand)
    wrapper = hand.update_preload.__func__
    r16_method = wrapper.__globals__.get("_provisional_r16_update_preload")
    r14_method = r16_method.__globals__.get("_provisional_r14_update_preload") if r16_method else None
    original_r14 = policy.r14.ConstrainedPreloadHand.update_preload
    original_r16 = policy.r16.BiasRecoveryHand.update_preload
    result["admission_policy_revision"] = PROTOCOL_FORMAT
    result["admission_policy"] = {
        "ast_hash_format": AST_HASH_FORMAT,
        "update_preload": _method_binding(wrapper),
        "wrapper_is_provisional": wrapper is policy.ProvisionalPreloadBudgetHand.update_preload,
        "r16_helper_is_provisional": r16_method is policy._provisional_r16_update_preload,
        "r14_helper_is_provisional": r14_method is policy._provisional_r14_update_preload,
        "r14_parent": {"original": _method_binding(original_r14),
            "original_ast_sha256": _ast_sha(policy._R14_ORIGINAL_AST),
            "transformed_ast_sha256": _ast_sha(policy._R14_PROVISIONAL_AST),
            "deleted_reason_if_count": 1,
            "other_globals_match_except_warning_registry": bool(r14_method) and r20._globals_match(
                r14_method.__globals__, original_r14.__globals__, {"update_preload"})},
        "r16_recovery_wrapper": {"original": _method_binding(original_r16),
            "original_ast_sha256": _ast_sha(policy._R16_ORIGINAL_AST),
            "transformed_ast_sha256": _ast_sha(policy._R16_PROVISIONAL_AST),
            "replaced_parent_call_count": 1,
            "other_globals_match_except_warning_registry": bool(r16_method) and r20._globals_match(
                r16_method.__globals__, original_r16.__globals__, {"update_preload", "_provisional_r14_update_preload"})},
        "deferred_pre_motion_reason": policy.DEFERRED_REASON,
        "original_probe_snapshot_and_actual_fixed_normal_value_retained": True}
    result["unchanged_methods"] = {name: getattr(hand, name).__func__ is getattr(r20.hooks.SharedNormalBudgetHand, name)
        for name in ("write", "after_step", "admit_probe", "confirm_probe", "held_revalidated", "abort_probe",
            "update_pressure_relief", "angles", "phase_begin", "regulate", "current_preload_observation")}
    return result


def isolated_runner(profile_path, declaration):
    namespace = dict(r20.isolated_runner.__globals__, controller_for=controller_for,
        snapshot=snapshot, tagged_protocol=tagged_protocol, runtime_binding=runtime_binding)
    factory = FunctionType(r20.isolated_runner.__code__, namespace, r20.isolated_runner.__name__,
        r20.isolated_runner.__defaults__, r20.isolated_runner.__closure__)
    return factory(profile_path, declaration)


def run(output, profile_path, cases):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R21 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R21 isolates the frozen ycb_can:3414 development case")
    return isolated_runner(profile_path, validate_inputs(profile_path))(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
