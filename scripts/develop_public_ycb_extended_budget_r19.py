#!/usr/bin/env python3
"""One .10rad budget experiment with the frozen R18 deadline and arm clock."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import develop_public_ycb_close_deadline_r18 as r18

r17 = r18.r17
BASELINE_BUDGET_RAD = .04
BUDGET_RAD = .10
CLOSE_SECONDS = 32.0
ARM_INTERPOLATION_SECONDS = 16.0
NATIVE_TIMESTEP_S = .002
PROTOCOL_FORMAT = "public_ycb_extended_budget_development_r19"
R18_DIRECTORY = PROJECT / "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered"
R18_SUMMARY = R18_DIRECTORY / "summary.json"
R18_DERIVED_CONTROLLER = R18_DIRECTORY / "ycb_can_bias_budget_r17_seed3414/derived_controller.py"
EXPECTED_R18_SUMMARY_SHA256 = "d3d5dec564810d8c9b6a9cc4cee8354062eb9f99c54960ab49b41ad87a740ff0"
EXPECTED_R18_DERIVED_SHA256 = "0650eca7f629427d6966a1aef0a9a68645765539970ad4832eee03bb0e25a485"
R18_PREFIX_RECEIPT = "results/r17_r18_close_deadline_comparison_20261022/receipt.json"
PREFLIGHT_HASHES = {
    "scripts/develop_public_ycb_close_deadline_r18.py": "89e600621af15e198ef5b803acf16d968b5f2b3e4d874f9d0890d23484de8a3e",
    "scripts/audit_public_ycb_close_deadline.py": "44f7405cffe1aa87eac98af9764dfd6a0df694c968d4a8f9c7a7703d4b26ef48",
    "scripts/recover_public_ycb_close_deadline_finalization.py": "4d88ef2c34a3c208979a55585180cc1d5ead42eb3142e412a8ebbed7b16f2778",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/summary.json": EXPECTED_R18_SUMMARY_SHA256,
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/protocol.json": "a60d2238665deb40712ad7c66d5b69230a559613f522c9a006faa5b4f6aa0387",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/recovery_receipt.json": "a40d4da5f8f0e8e1cbeb471940ce502241da74c69571e9eb442a7db5f17f77fa",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/verification_receipt.json": "7e97c56ac5b8456cfc1cb93516cfe7dc97fb3f17ab4f94f0e4bd86c159254b21",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/strict_native_verification.json": "98938ea2037f4f32fee56cc18a22d4180e5b17b1bd7d9a9a86d1d4952722b61c",
    "results/r18_close_deadline_audit_20261022/receipt.json": "ade3adf70fd68f8d5d6f41e28d125b0a0e6778da58e3c2c61661957a3bb99dc4",
    "results/r17_r18_close_deadline_comparison_20261022/receipt.json": "76f1cf9f65527d899a0f771c12d397a6c8c45d268c0e4b2055ad391086f321f7",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/ycb_can_bias_budget_r17_seed3414/scene.xml": "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e",
    "results/public_ycb_close_deadline_20261022_r18_close32_seed3414_recovered/ycb_can_bias_budget_r17_seed3414/derived_controller.py": EXPECTED_R18_DERIVED_SHA256,
}
NEW_SOURCES = (
    "scripts/develop_public_ycb_extended_budget_r19.py",
    "tests/test_public_ycb_extended_budget_r19.py",
    "docs/R19_EXTENDED_BUDGET_20261022.md",
)
OPTIONAL_AUDIT_SOURCES = (
    "scripts/audit_public_ycb_extended_budget.py",
    "tests/test_public_ycb_extended_budget_audit.py",
)
OPTIONAL_SUPERVISOR_SOURCES = (
    "scripts/run_supervised_experiment.py",
    "tests/test_supervised_experiment.py",
)


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r17.controller_for(*args,
        additional_bias_limit_rad=BUDGET_RAD, **kwargs)
    tree, changes = r18.transform_controller(source)
    namespace = dict(original.__globals__)
    if namespace["_phase_seconds"]["CLOSE"] != r18.CLOSE_SECONDS_BEFORE:
        raise ValueError("The inherited R17 task must retain its original 16s deadline")
    namespace["_phase_seconds"] = dict(namespace["_phase_seconds"], CLOSE=CLOSE_SECONDS)
    rendered = R18_DERIVED_CONTROLLER.read_text()
    actual_hash = hashlib.sha256(rendered.encode()).hexdigest()
    if actual_hash != EXPECTED_R18_DERIVED_SHA256:
        raise ValueError("R19 must reuse the exact R18 arm-clock-derived controller source")
    # ast.unparse spelling differs across Python versions; compare all semantics first.
    if ast.dump(ast.parse(rendered), include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError("R19 transformed controller AST differs from the frozen R18 source")
    exec(compile(rendered, "<r19_r18_clock_extended_budget>", "exec"), namespace)
    task = namespace["run_task"]
    task.r19_source_transform = {**changes,
        "baseline_derived_controller_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "baseline_r18_derived_controller_sha256": EXPECTED_R18_DERIVED_SHA256,
        "derived_controller_sha256": actual_hash}
    return task, rendered, hands, estimated


def validate_inputs(profile_path):
    for relative, expected in PREFLIGHT_HASHES.items():
        if r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R19 preflight input SHA256 mismatch: " + relative)
    inherited = r18.validate_inputs(profile_path)
    active = r17.validate_inputs(profile_path, BUDGET_RAD)
    baseline = json.loads(R18_SUMMARY.read_text())
    protocol = baseline["protocol"]
    recovery = json.loads((R18_DIRECTORY / "recovery_receipt.json").read_text())
    if (baseline["complete"] is not True or baseline["original_runner_normal_exit"] is not False
            or baseline["artifact_finalization_recovered"] is not True
            or protocol != json.loads((R18_DIRECTORY / "protocol.json").read_text())
            or protocol["format"] != r18.PROTOCOL_FORMAT
            or protocol["source_changed_during_run"] is not False
            or len(protocol["source_hashes_before"]) != 73
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["r18_close_deadline_experiment"] != inherited
            or protocol["r17_bias_budget_experiment"] != inherited["inherited_r17_declaration"]
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BASELINE_BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_SECONDS
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_SECONDS
            or protocol["finalization_recovery"] != recovery
            or recovery["simulation_rerun"] is not False
            or recovery["original_runner_normal_exit_claimed"] is not False
            or recovery["original_complete"] is not False
            or recovery["runner_exit_code_observed"] is not None
            or recovery["observed_ssh_client_exit_code"] != 255):
        raise ValueError("R19 comparison must retain the exact recovered, failed R18 provenance")
    for relative, expected in protocol["source_hashes_before"].items():
        if r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R18 inherited source mismatch: " + relative)
    rows = baseline["rows"]
    if (len(rows) != 1 or len(protocol["schedule"]) != 1 or rows[0]["kind"] != "ycb_can"
            or rows[0]["seed"] != 3414 or rows[0]["success"] is not False
            or rows[0]["held_lift"] is not False or rows[0]["runner_error"] is not None
            or rows[0]["derived_controller_sha256"] != EXPECTED_R18_DERIVED_SHA256
            or rows[0]["scene_sha256_before_load"] != rows[0]["scene_sha256_at_finalize"]
            or rows[0]["scene_unchanged_since_load"] is not True):
        raise ValueError("R18 comparison outcome, scene or controller binding differs")
    verification = json.loads((R18_DIRECTORY / "verification_receipt.json").read_text())
    strict = json.loads((R18_DIRECTORY / "strict_native_verification.json").read_text())
    audit = json.loads((PROJECT / "results/r18_close_deadline_audit_20261022/receipt.json").read_text())
    prefix = json.loads((PROJECT / R18_PREFIX_RECEIPT).read_text())
    if (verification["complete"] is not True or verification["unknowns"]
            or verification["task_successes"] != 0 or verification["held_lifts"] != 0
            or strict["accepted_evidence"] is not True or strict["unknowns"] or strict["strict_successes"] != 0
            or audit["all_evidence_consistent"] is not True
            or audit["all_declared_command_bounds_satisfied"] is not True
            or audit["all_native_close_spans_within_declared_deadline"] is not True
            or any(item["summary_sha256"] != EXPECTED_R18_SUMMARY_SHA256 for item in (verification, strict, audit))
            or prefix["candidate_inputs"]["summary"]["sha256"] != EXPECTED_R18_SUMMARY_SHA256
            or prefix["passed"] is not False):
        raise ValueError("R18 evidence receipts differ or hide the retained prefix failure")
    return {"format": "r19_extended_budget_development_protocol_v1", "controller": "BiasBudgetHand",
        "runner_revision": PROTOCOL_FORMAT, "additional_bias_limit_rad": BUDGET_RAD,
        "baseline_additional_bias_limit_rad": BASELINE_BUDGET_RAD,
        "close_seconds": CLOSE_SECONDS, "arm_interpolation_seconds": ARM_INTERPOLATION_SECONDS,
        "native_timestep_s": NATIVE_TIMESTEP_S,
        "changed_parameters": {"joint_admittance.additional_bias_limit_rad":
            {"before": BASELINE_BUDGET_RAD, "after": BUDGET_RAD}},
        "active_r17_declaration": copy.deepcopy(active),
        "inherited_r18_declaration": copy.deepcopy(inherited),
        "inherited_r18_r17_declaration": copy.deepcopy(protocol["r17_bias_budget_experiment"]),
        "inherited_r18_joint_admittance": copy.deepcopy(protocol["joint_admittance"]),
        "inherited_r18_force_control": protocol["force_control"],
        "effective_phase_seconds": copy.deepcopy(protocol["phase_seconds"]),
        "effective_constrained_preload": copy.deepcopy(protocol["constrained_preload"]),
        "inherited_r18_recovery": copy.deepcopy(recovery),
        "baseline_task_success": False, "baseline_held_lift": False,
        "baseline_prefix_comparison_passed": False,
        "baseline_source_after_timing": protocol["source_hashes_after_timing"],
        "baseline_scene_sha256": rows[0]["scene_sha256_before_load"],
        "baseline_derived_controller_sha256": EXPECTED_R18_DERIVED_SHA256,
        "r18_comparison_summary_sha256": EXPECTED_R18_SUMMARY_SHA256,
        "preflight_input_sha256": copy.deepcopy(PREFLIGHT_HASHES),
        "instance_budget_policy": "All actual factory-created hands including retry use the same .10rad instance budget; final metadata reads their properties",
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False,
        "scope": "One same-can same-seed .04-to-.10 budget test at unchanged R18 clocks; no automatic sweep, normal-exit baseline, task-success or accuracy claim"}


def snapshot(profile_path):
    values = r18.snapshot(profile_path)
    for relative in (*PREFLIGHT_HASHES, *NEW_SOURCES):
        values[relative] = r17.r16.sha256(PROJECT / relative)
    for relative in (*OPTIONAL_AUDIT_SOURCES, *OPTIONAL_SUPERVISOR_SOURCES):
        if (PROJECT / relative).is_file():
            values[relative] = r17.r16.sha256(PROJECT / relative)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    active = declaration["active_r17_declaration"]
    r17.tagged_protocol(value, active)
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.pop("r18_close_deadline_experiment", None)
        protocol.update(format=PROTOCOL_FORMAT, runner_revision=PROTOCOL_FORMAT,
            r19_extended_budget_experiment=copy.deepcopy(declaration),
            r17_bias_budget_experiment=copy.deepcopy(active),
            inherited_r18_close_deadline_experiment=copy.deepcopy(declaration["inherited_r18_declaration"]),
            inherited_r18_joint_admittance=copy.deepcopy(declaration["inherited_r18_joint_admittance"]),
            inherited_r18_force_control=declaration["inherited_r18_force_control"],
            phase_seconds=copy.deepcopy(declaration["effective_phase_seconds"]),
            constrained_preload=copy.deepcopy(declaration["effective_constrained_preload"]),
            joint_admittance={**protocol["joint_admittance"], "additional_bias_limit_rad": BUDGET_RAD},
            force_control="R19 .10rad per-instance bias budget; frozen R17 hand/R16 writer and R6 final safety. R18 CLOSE32/original-arm16 clocks, physics, force limits, admission and task gates retained. Recovered R18 .04rad baseline is historical, including its failed full prefix comparison.",
            close_termination="Original continuous preload/probe eligibility or unchanged 32s CLOSE deadline, original16s arm interpolation, all safety and task gates retained.")
        for planned in protocol["schedule"]:
            planned.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS,
                arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS, additional_bias_limit_rad=BUDGET_RAD)
    for row in value.get("rows", []):
        row.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS,
            arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS, additional_bias_limit_rad=BUDGET_RAD)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS,
            arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS, additional_bias_limit_rad=BUDGET_RAD)
    return value


def isolated_runner(profile_path, declaration):
    original = r17.isolated_runner(profile_path, declaration["active_r17_declaration"])
    namespace = dict(original.__globals__)
    if namespace["PHASE_SECONDS"]["CLOSE"] != r18.CLOSE_SECONDS_BEFORE:
        raise ValueError("The inherited R17 runner must retain its original deadline")
    namespace["PHASE_SECONDS"] = dict(namespace["PHASE_SECONDS"], CLOSE=CLOSE_SECONDS)
    previous, bindings = namespace["previous"], {}
    proxy = SimpleNamespace(**vars(previous))

    def declared_controller_for(*args, **kwargs):
        task, source, hands, estimated = controller_for(*args, **kwargs)
        bindings[Path(args[1]).resolve()] = (task, hands)
        return task, source, hands, estimated

    def write_declared_json(path, value):
        if Path(path).name == "metadata.json":
            binding = bindings.get(Path(path).parent.resolve())
            if binding is None:
                value["missing_runtime_controller_binding"] = True
            else:
                task, hands = binding
                value.update(effective_task_phase_seconds=copy.deepcopy(task.__globals__["_phase_seconds"]),
                    effective_runner_phase_seconds=copy.deepcopy(namespace["PHASE_SECONDS"]),
                    arm_source_transform=copy.deepcopy(task.r19_source_transform),
                    effective_hand_budgets_rad=[hand.additional_bias_limit_rad for hand in hands],
                    effective_hand_controller_types=[type(hand).__name__ for hand in hands],
                    effective_hand_instances=len(hands))
        return previous.atomic_json(path, tagged_protocol(value, declaration))

    proxy.atomic_json = write_declared_json
    namespace.update(previous=proxy, snapshot=snapshot, controller_for=declared_controller_for)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R19 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R19 isolates the frozen ycb_can:3414 development case")
    declaration = validate_inputs(profile_path)
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
