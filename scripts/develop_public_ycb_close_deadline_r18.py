#!/usr/bin/env python3
"""One CLOSE-deadline experiment with the frozen R17 hand and original arm clock."""

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
import develop_public_ycb_bias_budget_r17 as r17

CLOSE_SECONDS_BEFORE = 16.0
CLOSE_SECONDS_AFTER = 32.0
ARM_INTERPOLATION_SECONDS = 16.0
NATIVE_TIMESTEP_S = .002
BUDGET_RAD = .04
PROTOCOL_FORMAT = "public_ycb_close_deadline_development_r18"
R17_DIRECTORY = PROJECT / "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414"
R17_SUMMARY = R17_DIRECTORY / "summary.json"
EXPECTED_R17_SUMMARY_SHA256 = "e2883dce787d3c9140edc59c77e415394d904a7029f9f8e4ec5559b3946d61d8"
PREFLIGHT_HASHES = {
    "scripts/develop_public_ycb_bias_budget_r17.py": "c4ce070b16c116677bcf432842885c178d3b790d884c3852d1f9bc1dfbaa1d04",
    "scripts/audit_public_ycb_bias_budget.py": "4f4cbe14227aa526bb4bb0d1e3ef2707c50dfc184ca539fa185fae7130e97eb7",
    "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414/summary.json": EXPECTED_R17_SUMMARY_SHA256,
    "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414/protocol.json": "05ba5d3b4d2f1ee970b148a971f70046067319204f32d34238ee0cbba20760c8",
    "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414/verification_receipt.json": "df8eaa667f52a07c4d58b2c7ce377987f99a096d1e5fdf974b1ac38e82daf0ec",
    "results/public_ycb_bias_budget_20261022_r17_budget004_seed3414/strict_native_verification.json": "913b1a2a697d67cdeb4fd57c6c80a7ca0fa6dc41f3265aaf4269b9355479f83f",
    "results/r17_004_budget_audit_20261022/receipt.json": "8be63e51f6867a7f8c8ae322ff947bcf9b05e13e961dba8fe8920a5b989060bf",
}
NEW_SOURCES = (
    "scripts/develop_public_ycb_close_deadline_r18.py",
    "tests/test_public_ycb_close_deadline_r18.py",
    "docs/R18_CLOSE_DEADLINE_20261022.md",
)
OPTIONAL_AUDIT_SOURCES = (
    "scripts/audit_public_ycb_close_deadline.py",
    "tests/test_public_ycb_close_deadline_audit.py",
)
OPTIONAL_ANALYSIS_SOURCES = (
    "scripts/compare_public_ycb_close_deadline.py",
    "tests/test_public_ycb_close_deadline_comparison.py",
)


def transform_controller(source):
    """Change the deadline clock without stretching either original arm fraction."""
    tree = ast.parse(source)
    moves = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move"]
    if len(moves) != 1 or any(isinstance(node, ast.Name) and node.id.startswith("_r18_") for node in ast.walk(tree)):
        raise RuntimeError("Expected one unmodified R17 move function")
    move = moves[0]
    steps = [node for node in move.body if isinstance(node, ast.Assign)
        and ast.unparse(node) == "steps = int(duration / model.opt.timestep)"]
    segments = [node for node in move.body if isinstance(node, ast.Assign)
        and ast.unparse(node) == "segment_steps = max(1, round(0.02 / model.opt.timestep))"]
    cartesian = [node for node in ast.walk(move) if isinstance(node, ast.Assign)
        and ast.unparse(node) == "next_position = interpolate(start_position, end_position, end_step / steps)"]
    joint = [node for node in ast.walk(move) if isinstance(node, ast.Assign)
        and ast.unparse(node) == "desired_command = interpolate(start, target, (step + 1) / steps)"]
    if not all(len(nodes) == 1 for nodes in (steps, segments, cartesian, joint)):
        raise RuntimeError("Frozen R17 arm interpolation structure changed")
    injection = ast.parse("""
if float(model.opt.timestep) != 0.002:
    raise ValueError('R18 requires the original 0.002s physics timestep')
_r18_arm_steps = int(16.0 / model.opt.timestep) if state == State.CLOSE else steps
if state == State.CLOSE and _r18_arm_steps % max(1, round(0.02 / model.opt.timestep)) != 0:
    raise ValueError('R18 original arm steps must be divisible by the original segment steps')
""").body
    position = move.body.index(steps[0]) + 1
    move.body[position:position] = injection
    cartesian[0].value.args[2] = ast.parse("min(end_step / _r18_arm_steps, 1.0)", mode="eval").body
    joint[0].value.args[2] = ast.parse("min((step + 1) / _r18_arm_steps, 1.0)", mode="eval").body
    return ast.fix_missing_locations(tree), {"arm_step_injections": 1,
        "arm_fraction_replacements": 2, "hand_fraction_replacements": 0}


def controller_for(*args, **kwargs):
    original, source, hands, estimated = r17.controller_for(*args,
        additional_bias_limit_rad=BUDGET_RAD, **kwargs)
    tree, changes = transform_controller(source)
    namespace = dict(original.__globals__)
    if namespace["_phase_seconds"]["CLOSE"] != CLOSE_SECONDS_BEFORE:
        raise ValueError("The inherited task must use the original 16s CLOSE deadline")
    namespace["_phase_seconds"] = dict(namespace["_phase_seconds"], CLOSE=CLOSE_SECONDS_AFTER)
    exec(compile(tree, "<r18_original_arm_clock_close_deadline>", "exec"), namespace)
    task = namespace["run_task"]
    rendered = ast.unparse(tree) + "\n"
    task.r18_source_transform = {**changes,
        "baseline_derived_controller_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "derived_controller_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
    return task, rendered, hands, estimated


def validate_inputs(profile_path):
    for relative, expected in PREFLIGHT_HASHES.items():
        if r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R18 preflight input SHA256 mismatch: " + relative)
    inherited = r17.validate_inputs(profile_path, BUDGET_RAD)
    baseline = json.loads(R17_SUMMARY.read_text())
    protocol = baseline["protocol"]
    if (not baseline["complete"] or protocol != json.loads((R17_DIRECTORY / "protocol.json").read_text())
            or protocol["source_changed_during_run"]
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["r17_bias_budget_experiment"] != inherited
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BUDGET_RAD
            or protocol["phase_seconds"]["CLOSE"] != CLOSE_SECONDS_BEFORE
            or protocol["constrained_preload"]["close_seconds"] != CLOSE_SECONDS_BEFORE):
        raise ValueError("R17 comparison protocol is not the frozen .04rad/16s baseline")
    for relative, expected in protocol["source_hashes_before"].items():
        if r17.r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R17 inherited source mismatch: " + relative)
    verification = json.loads((R17_DIRECTORY / "verification_receipt.json").read_text())
    strict = json.loads((R17_DIRECTORY / "strict_native_verification.json").read_text())
    audit = json.loads((PROJECT / "results/r17_004_budget_audit_20261022/receipt.json").read_text())
    if (not verification["complete"] or verification["unknowns"]
            or not strict["accepted_evidence"] or strict["unknowns"]
            or not audit["all_evidence_consistent"] or not audit["all_declared_command_bounds_satisfied"]
            or any(value["summary_sha256"] != EXPECTED_R17_SUMMARY_SHA256
                for value in (verification, strict, audit))):
        raise ValueError("R17 baseline receipts are not complete and summary-bound")
    phase = dict(protocol["phase_seconds"], CLOSE=CLOSE_SECONDS_AFTER)
    preload = dict(protocol["constrained_preload"], close_seconds=CLOSE_SECONDS_AFTER)
    return {"format": "r18_close_deadline_development_protocol_v1", "controller": "BiasBudgetHand",
        "runner_revision": PROTOCOL_FORMAT, "close_seconds": CLOSE_SECONDS_AFTER,
        "baseline_close_seconds": CLOSE_SECONDS_BEFORE, "arm_interpolation_seconds": ARM_INTERPOLATION_SECONDS,
        "native_timestep_s": NATIVE_TIMESTEP_S, "additional_bias_limit_rad": BUDGET_RAD,
        "changed_parameters": {"phase_seconds.CLOSE": {"before": CLOSE_SECONDS_BEFORE, "after": CLOSE_SECONDS_AFTER}},
        "arm_source_transform": {"arm_step_injections": 1, "arm_fraction_replacements": 2,
            "hand_fraction_replacements": 0, "baseline_close_steps": 8000,
            "extension_holds_arm_fraction_at_one": True},
        "inherited_r17_declaration": copy.deepcopy(inherited),
        "inherited_r17_phase_seconds": copy.deepcopy(protocol["phase_seconds"]),
        "inherited_r17_constrained_preload": copy.deepcopy(protocol["constrained_preload"]),
        "inherited_r17_force_control": protocol["force_control"],
        "effective_phase_seconds": phase, "effective_constrained_preload": preload,
        "r17_comparison_summary_sha256": EXPECTED_R17_SUMMARY_SHA256,
        "preflight_input_sha256": copy.deepcopy(PREFLIGHT_HASHES),
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False,
        "prefix_validation": {"required": True, "status": "NOT_YET_RUN",
            "scope": "Compare first-attempt native/ctrl/raw-pressure prefix through the original CLOSE deadline; arithmetic tests alone do not prove physical identity"},
        "scope": "One same-seed 32s CLOSE deadline experiment with original 16s arm interpolation; no automatic sweep or task/accuracy claim"}


def snapshot(profile_path):
    values = r17.snapshot(profile_path)
    for relative in (*PREFLIGHT_HASHES, *NEW_SOURCES):
        values[relative] = r17.r16.sha256(PROJECT / relative)
    for relative in (*OPTIONAL_AUDIT_SOURCES, *OPTIONAL_ANALYSIS_SOURCES):
        if (PROJECT / relative).is_file():
            values[relative] = r17.r16.sha256(PROJECT / relative)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r17.tagged_protocol(value, declaration["inherited_r17_declaration"])
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.update(format=PROTOCOL_FORMAT, runner_revision=PROTOCOL_FORMAT,
            r18_close_deadline_experiment=copy.deepcopy(declaration),
            inherited_r17_phase_seconds=copy.deepcopy(declaration["inherited_r17_phase_seconds"]),
            inherited_r17_constrained_preload=copy.deepcopy(declaration["inherited_r17_constrained_preload"]),
            inherited_r17_force_control=declaration["inherited_r17_force_control"],
            phase_seconds=copy.deepcopy(declaration["effective_phase_seconds"]),
            constrained_preload=copy.deepcopy(declaration["effective_constrained_preload"]),
            force_control="R18 32s CLOSE deadline with original16s arm interpolation; the frozen R17 .04rad hand, R16 write/recovery, R6 final safety, physics, force and all admission/task gates remain unchanged.",
            close_termination="Original continuous preload and probe eligibility, or declared 32s CLOSE deadline; original16s arm interpolation retained, all safety and task gates unchanged.")
        for planned in protocol["schedule"]:
            planned.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS_AFTER,
                arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS)
    for row in value.get("rows", []):
        row.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS_AFTER,
            arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(runner_revision=PROTOCOL_FORMAT, close_seconds=CLOSE_SECONDS_AFTER,
            arm_interpolation_seconds=ARM_INTERPOLATION_SECONDS)
    return value


def isolated_runner(profile_path, declaration):
    original = r17.isolated_runner(profile_path, declaration["inherited_r17_declaration"])
    namespace = dict(original.__globals__)
    if namespace["PHASE_SECONDS"]["CLOSE"] != CLOSE_SECONDS_BEFORE:
        raise ValueError("The inherited runner must use the original 16s CLOSE deadline")
    namespace["PHASE_SECONDS"] = dict(namespace["PHASE_SECONDS"], CLOSE=CLOSE_SECONDS_AFTER)
    previous = namespace["previous"]
    proxy = SimpleNamespace(**vars(previous))
    bindings = {}

    def declared_controller_for(*args, **kwargs):
        task, source, hands, estimated = controller_for(*args, **kwargs)
        bindings[Path(args[1]).resolve()] = {
            "effective_task_phase_seconds": copy.deepcopy(task.__globals__["_phase_seconds"]),
            "effective_runner_phase_seconds": copy.deepcopy(namespace["PHASE_SECONDS"]),
            "arm_source_transform": copy.deepcopy(task.r18_source_transform)}
        return task, source, hands, estimated

    def write_declared_json(path, value):
        if Path(path).name == "metadata.json":
            binding = bindings.get(Path(path).parent.resolve())
            if binding is None:
                value["missing_runtime_controller_binding"] = True
            else:
                value.update(binding)
        return previous.atomic_json(path, tagged_protocol(value, declaration))

    proxy.atomic_json = write_declared_json
    namespace.update(previous=proxy, snapshot=snapshot, controller_for=declared_controller_for)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases):
    declaration = validate_inputs(profile_path)
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R18 isolates the frozen ycb_can:3414 development case")
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
