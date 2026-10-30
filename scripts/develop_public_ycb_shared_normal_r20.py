#!/usr/bin/env python3
"""One shared-normal allocator experiment at the frozen R19 budget and clocks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import develop_public_ycb_extended_budget_r19 as r19
import shared_normal_controller_hooks as hooks
import shared_normal_tracking_targets as shared

BUDGET_RAD = .10
CLOSE_SECONDS = 32.
ARM_INTERPOLATION_SECONDS = 16.
NATIVE_TIMESTEP_S = .002
PROTOCOL_FORMAT = "public_ycb_shared_normal_development_r20"
TARGET_FORMAT = "shared_normal_tracking_target_v1"
CONTROLLER_CLASS = "SharedNormalBudgetHand"
CONTROLLER_REVISION_SCOPE = "Inherited R17 write/native mechanics; tracking and gate-only allocators changed under R20"
R19_DIRECTORY = PROJECT / "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414"
R19_SUMMARY = R19_DIRECTORY / "summary.json"
EXPECTED_R19_SUMMARY_SHA256 = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
R19_JOB = "results/jobs/r19_bias10_close32_seed3414_20261022"
PREFLIGHT_HASHES = {
    "scripts/develop_public_ycb_extended_budget_r19.py": "619728464296d13ff6476acc16390ef419eb94d2dd8c134b596e40a4d0e417b4",
    "scripts/audit_public_ycb_extended_budget.py": "66a1208428856b5d212206c4581d927961fe47f0c8543605198295c3b9738e5d",
    "scripts/shared_normal_tracking_targets.py": "f33434846456e02c063dc3224bd2f03cff5812b218aeab7b7f8de5ad88633101",
    "tests/test_shared_normal_tracking_targets.py": "d2c96fdaf03eee59066081e1614dd2c417a96e95e6b9e4e1877b823d8f86d450",
    "tests/fixtures/shared_normal_r18_observations_20261022.json": "765a7bdd3f98ccc24053bf724c382a2654404d8fe6476befa3e0a8e709d13795",
    "docs/SHARED_NORMAL_TARGET_DESIGN_20261022.md": "99341c3931bbca23c531bc7d0bed39a8d7da8412c357e871e93e3be2067a438f",
    "scripts/shared_normal_controller_hooks.py": "45af05f935a7199d1668332bca3c715e61cdccea771cae81c5e3dba63ec7453a",
    "tests/test_shared_normal_controller_hooks.py": "67abdb745253c43745964c8b6c67e64ce874ee71f766060b583ffb4f490d7f01",
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/summary.json": EXPECTED_R19_SUMMARY_SHA256,
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/protocol.json": "92cfa6fc266488ff79139d7232a907269a0519ca9324246afbf2b05017851989",
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/verification_receipt.json": "2b6261e9978c27dffd824617d5df4ffa1edca9392b6da9fee9eed179a4ab9863",
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/strict_native_verification.json": "afac02d33fc31e4d38168c8a07099049ef961e91028f9d83745ad40fe6c4b1a3",
    "results/r19_extended_budget_audit_20261022/receipt.json": "bc440c0e45f2aa9f612263cbe243af3c74e6213d06668cdb14db30b27c8c3822",
    "results/jobs/r19_bias10_close32_seed3414_20261022/status.json": "431985f891fb99265bcba087e797664bc0a7ff56668c1fcacd3afbd0f473e34e",
    "results/jobs/r19_bias10_close32_seed3414_20261022/request.json": "e7ae04041b4ec406be5751cbef959c7bf10c40dd3328e04e3890ce44eae9ee49",
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/scene.xml": "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e",
    "results/public_ycb_extended_budget_20261022_r19_bias10_close32_seed3414/ycb_can_bias_budget_r17_seed3414/derived_controller.py": "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485",
}
NEW_SOURCES = ("scripts/develop_public_ycb_shared_normal_r20.py",
    "tests/test_public_ycb_shared_normal_r20.py", "docs/R20_SHARED_NORMAL_20261022.md")
OPTIONAL_AUDIT_SOURCES = ("scripts/audit_public_ycb_shared_normal.py",
    "tests/test_public_ycb_shared_normal_audit.py")


_r17_proxy = SimpleNamespace(**{**vars(r19.r17), "controller_for": hooks.controller_for})
_controller_factory = FunctionType(r19.controller_for.__code__,
    dict(r19.controller_for.__globals__, r17=_r17_proxy), r19.controller_for.__name__,
    r19.controller_for.__defaults__, r19.controller_for.__closure__)


def controller_for(*args, **kwargs):
    return _controller_factory(*args, **kwargs)


def validate_inputs(profile_path):
    for relative, expected in PREFLIGHT_HASHES.items():
        if r19.r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R20 preflight input SHA256 mismatch: " + relative)
    inherited = r19.validate_inputs(profile_path)
    baseline = json.loads(R19_SUMMARY.read_text())
    protocol, rows = baseline["protocol"], baseline["rows"]
    if (baseline["complete"] is not True or baseline["runner_errors"]
            or protocol != json.loads((R19_DIRECTORY / "protocol.json").read_text())
            or protocol["format"] != r19.PROTOCOL_FORMAT
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or len(protocol["source_hashes_before"]) != 90
            or protocol["r19_extended_budget_experiment"] != inherited
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_SECONDS
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_SECONDS
            or len(rows) != 1 or len(protocol["schedule"]) != 1):
        raise ValueError("R20 requires the immutable completed R19 budget/clock baseline")
    for relative, expected in protocol["source_hashes_before"].items():
        if r19.r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R19 inherited source mismatch: " + relative)
    row = rows[0]
    if (row["kind"] != "ycb_can" or row["seed"] != 3414 or row["success"] is not False
            or row["held_lift"] is not False or row["runner_error"] is not None
            or row["derived_controller_sha256"] != r19.EXPECTED_R18_DERIVED_SHA256
            or row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != row["scene_sha256_at_finalize"]
            or row["effective_hand_budgets_rad"] != [BUDGET_RAD] * len(row["attempts"])
            or row["effective_hand_controller_types"] != ["BiasBudgetHand"] * len(row["attempts"])):
        raise ValueError("R19 task failure, runtime hand, source or scene binding differs")
    verification = json.loads((R19_DIRECTORY / "verification_receipt.json").read_text())
    strict = json.loads((R19_DIRECTORY / "strict_native_verification.json").read_text())
    audit = json.loads((PROJECT / "results/r19_extended_budget_audit_20261022/receipt.json").read_text())
    status = json.loads((PROJECT / R19_JOB / "status.json").read_text())
    request = json.loads((PROJECT / R19_JOB / "request.json").read_text())
    command = request["command"]
    if (verification["complete"] is not True or verification["unknowns"]
            or verification["task_successes"] != 0 or verification["held_lifts"] != 0
            or strict["accepted_evidence"] is not True or strict["unknowns"] or strict["strict_successes"] != 0
            or audit["all_evidence_consistent"] is not True
            or audit["all_declared_command_bounds_satisfied"] is not True
            or audit["all_native_close_spans_within_declared_deadline"] is not True
            or any(value["summary_sha256"] != EXPECTED_R19_SUMMARY_SHA256
                for value in (verification, strict, audit))
            or status["state"] != "FINISHED" or status["terminal"] is not True
            or status["returncode"] != 0 or status["child_exit_observed"] is not True
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]
            or "scripts/develop_public_ycb_extended_budget_r19.py" not in command
            or command[command.index("--output") + 1] != R19_DIRECTORY.relative_to(PROJECT).as_posix()
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]):
        raise ValueError("R19 independent receipts or observed child exit are not bound and complete")
    return {"format": "r20_shared_normal_development_protocol_v1", "controller": CONTROLLER_CLASS,
        "runner_revision": PROTOCOL_FORMAT, "additional_bias_limit_rad": BUDGET_RAD,
        "close_seconds": CLOSE_SECONDS, "arm_interpolation_seconds": ARM_INTERPOLATION_SECONDS,
        "native_timestep_s": NATIVE_TIMESTEP_S,
        "changed_parameters": {"tracking_and_gate_allocator": {
            "before": "r13_single_1_5g_then_fixed_1g_postcheck", "after": TARGET_FORMAT}},
        "active_r19_declaration": copy.deepcopy(inherited),
        "inherited_r19_declaration": copy.deepcopy(protocol["r19_extended_budget_experiment"]),
        "inherited_r19_force_control": protocol["force_control"],
        "effective_phase_seconds": copy.deepcopy(protocol["phase_seconds"]),
        "effective_constrained_preload": copy.deepcopy(protocol["constrained_preload"]),
        "baseline_task_success": False, "baseline_held_lift": False,
        "baseline_child_exit_observed": True, "baseline_child_returncode": 0,
        "baseline_scene_sha256": row["scene_sha256_before_load"],
        "baseline_derived_controller_sha256": row["derived_controller_sha256"],
        "r19_comparison_summary_sha256": EXPECTED_R19_SUMMARY_SHA256,
        "preflight_input_sha256": copy.deepcopy(PREFLIGHT_HASHES),
        "allocator_format": TARGET_FORMAT, "allocator_paths": ["tracking", "gate_only"],
        "allocator_source_sha256": PREFLIGHT_HASHES["scripts/shared_normal_tracking_targets.py"],
        "hook_source_sha256": PREFLIGHT_HASHES["scripts/shared_normal_controller_hooks.py"],
        "lp_time_limit_per_solve_s": .1,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False,
        "scope": "One same-can same-seed allocator intervention at unchanged R19 .10/32/16. Static targets, safe commands and normal runner exit are not grasp success or a randomized-dataset rate."}


def snapshot(profile_path):
    values = r19.snapshot(profile_path)
    for relative in (*PREFLIGHT_HASHES, *NEW_SOURCES):
        values[relative] = r19.r17.r16.sha256(PROJECT / relative)
    for relative in OPTIONAL_AUDIT_SOURCES:
        if (PROJECT / relative).is_file():
            values[relative] = r19.r17.r16.sha256(PROJECT / relative)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r19.tagged_protocol(value, declaration["active_r19_declaration"])
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop("r19_extended_budget_experiment", None)
        protocol.update(format=PROTOCOL_FORMAT, runner_revision=PROTOCOL_FORMAT,
            r20_shared_normal_experiment=copy.deepcopy(declaration),
            inherited_r19_extended_budget_experiment=copy.deepcopy(declaration["inherited_r19_declaration"]),
            inherited_r19_force_control=declaration["inherited_r19_force_control"],
            force_control="R20 SharedNormalBudgetHand: both tracking and gate-only targets use shared-normal 1.5g/1g LP. Frozen R17 budget/R16 writer/R6 safety and R19 .10rad/CLOSE32/original-arm16 clocks remain. Static feasibility is not measured support or success.",
            controller_class=CONTROLLER_CLASS, controller_revision_scope=CONTROLLER_REVISION_SCOPE,
            target_allocator_revision=PROTOCOL_FORMAT)
        for planned in protocol["schedule"]:
            planned.update(runner_revision=PROTOCOL_FORMAT, controller_class=CONTROLLER_CLASS,
                target_allocator_revision=PROTOCOL_FORMAT)
    for row in value.get("rows", []):
        row.update(runner_revision=PROTOCOL_FORMAT, controller_class=CONTROLLER_CLASS,
            controller_revision_scope=CONTROLLER_REVISION_SCOPE, target_allocator_revision=PROTOCOL_FORMAT)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(runner_revision=PROTOCOL_FORMAT, controller_class=CONTROLLER_CLASS,
            controller_revision_scope=CONTROLLER_REVISION_SCOPE, target_allocator_revision=PROTOCOL_FORMAT)
    return value


def _globals_match(actual, original, excluded):
    keys = (actual.keys() | original.keys()) - set(excluded) - {"__warningregistry__"}
    return all(key in actual and key in original and actual[key] is original[key] for key in keys)


def _path_binding(method, original, *, gate_only=False):
    function = method.__func__
    actual = function.__globals__["r13"].allocate_tracking_targets if gate_only else function.__globals__["allocate_tracking_targets"]
    binding = {"method_source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
        "method_bytecode_sha256": hashlib.sha256(function.__code__.co_code).hexdigest(),
        "method_name": function.__code__.co_name,
        "method_source_file": Path(inspect.getsourcefile(function)).resolve().relative_to(PROJECT).as_posix(),
        "method_source_file_sha256": r19.r17.r16.sha256(inspect.getsourcefile(function)),
        "same_code_as_original": function.__code__ is original.__code__,
        "same_closure_as_original": function.__closure__ is original.__closure__,
        "other_globals_match_except_warning_registry": _globals_match(function.__globals__, original.__globals__,
            {"r13"} if gate_only else {"allocate_tracking_targets"}),
        "allocator": {"module": actual.__module__, "qualname": actual.__qualname__,
            "source_sha256": r19.r17.r16.sha256(actual.__globals__["__file__"])},
        "allocator_is_shared_normal": actual is shared.allocate_tracking_targets}
    if gate_only:
        binding["r13_proxy_other_members_match"] = _globals_match(vars(function.__globals__["r13"]),
            vars(hooks.r13), {"allocate_tracking_targets"})
    return binding


def runtime_binding(hand):
    return {"controller_class": type(hand).__name__, "additional_bias_limit_rad": hand.additional_bias_limit_rad,
        "python_version": list(sys.version_info[:3]), "bytecode_hash_scope": "co_code only; portable method source hashes and frozen files are recorded separately",
        "tracking": _path_binding(hand.regulate, hooks.r13.JointAdmittanceProbeHand.regulate),
        "gate_only": _path_binding(hand.current_preload_observation,
            hooks.r14.ConstrainedPreloadHand.current_preload_observation, gate_only=True),
        "writer_is_frozen_r16": hand.write.__func__ is r19.r17.r16.BiasRecoveryHand.write,
        "after_step_is_frozen_r16": hand.after_step.__func__ is r19.r17.r16.BiasRecoveryHand.after_step,
        "admit_probe_is_frozen_r16": hand.admit_probe.__func__ is r19.r17.r16.BiasRecoveryHand.admit_probe}


def isolated_runner(profile_path, declaration):
    bindings = {}
    def bound_controller(*args, **kwargs):
        task, source, hands, estimated = controller_for(*args, **kwargs)
        bindings[Path(args[1]).resolve()] = hands
        return task, source, hands, estimated
    namespace = dict(r19.isolated_runner.__globals__, controller_for=bound_controller,
        snapshot=snapshot, tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    factory = FunctionType(r19.isolated_runner.__code__, namespace, r19.isolated_runner.__name__,
        r19.isolated_runner.__defaults__, r19.isolated_runner.__closure__)
    original = factory(profile_path, declaration["active_r19_declaration"])
    globals_ = dict(original.__globals__)
    previous = globals_["previous"]
    proxy = SimpleNamespace(**vars(previous))
    def write_json(path, value):
        if Path(path).name == "metadata.json":
            hands = bindings.get(Path(path).parent.resolve())
            if hands is None:
                value["missing_runtime_allocator_binding"] = True
            else:
                value["shared_normal_runtime_bindings"] = [runtime_binding(hand) for hand in hands]
        return previous.atomic_json(path, value)
    proxy.atomic_json = write_json
    globals_["previous"] = proxy
    return FunctionType(original.__code__, globals_, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R20 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R20 isolates the frozen ycb_can:3414 development case")
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
