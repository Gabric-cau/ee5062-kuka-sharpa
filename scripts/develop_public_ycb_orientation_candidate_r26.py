#!/usr/bin/env python3
"""Isolated R26 orientation-candidate entry over the finalized R25 case.

R26 reads a separately frozen rotated grasp profile and changes the nominal
wrist orientation initialization in the generated R25 arm function.  The R3
pose composition ``pose_rotation @ _nominal_wrist_rotation`` remains in the
readable source.  The candidate's hand parameters are passed to the inherited
factory.  R25/R24 source and recorded outputs remain frozen inputs; this entry
does not claim that the candidate succeeds.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType
import zipfile

import numpy as np

import develop_public_ycb_rate_limited_probe_r25 as r25


PROJECT = r25.PROJECT
PROTOCOL_FORMAT = "public_ycb_orientation_candidate_development_r26"
DECLARATION = "r26_orientation_candidate_experiment"
CONTROLLER_CLASS = r25.CONTROLLER_CLASS
BASELINE = PROJECT / "results/public_ycb_rate_limited_probe_20261023_r25_bias10_close32_seed3414"
BASELINE_JOB = PROJECT / "results/jobs/r25_rate_limited_probe_bias10_close32_seed3414_20261023"
BASELINE_SHA256 = "8b12661cd1949b6485e398a319cd3407688262205498751f97d0736dea086675"
BASELINE_PROFILE = PROJECT / "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json"
CANDIDATE_PROFILE = PROJECT / "results/public_ycb_geometry_20261022_r15_final/lowest_peak_load_profile.json"
CANDIDATE_PROFILE_SHA256 = "52884749ddd646713cc6f0a14a06adec47547e84c0dbf32fb628d44c8cdeb900"
NEW_SOURCES = (
    "scripts/develop_public_ycb_orientation_candidate_r26.py",
    "tests/test_public_ycb_orientation_candidate_r26.py",
)
sha256 = r25.sha256
checked_file = r25.checked_file


def load_nominal_rotation(profile_path):
    """Read and validate the authored world wrist orientation from a profile."""
    profiles = json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
    value = profiles.get("desired_world_wrist_rotation")
    if value is None:
        raise ValueError("R26 profile is missing desired_world_wrist_rotation")
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("R26 desired_world_wrist_rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.):
        raise ValueError("R26 desired_world_wrist_rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1., atol=2e-5, rtol=0.):
        raise ValueError("R26 desired_world_wrist_rotation must have determinant +1")
    return rotation


def _orientation_literal(rotation):
    values = json.dumps(np.asarray(rotation, dtype=float).tolist(), separators=(",", ":"))
    return f"np.asarray({values}, dtype=float)"


def replace_nominal_wrist_rotation(source, rotation):
    """Replace only R3's nominal initializer and retain its pose composition."""
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_nominal_wrist_rotation"
                for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("R26 requires exactly one R3 nominal wrist initializer")
    composition = [node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.MatMult)
        and ast.unparse(node.value) == "pose_rotation @ _nominal_wrist_rotation"]
    if len(composition) != 1:
        raise ValueError("R26 requires the R3 pose_rotation @ _nominal_wrist_rotation composition")
    rotation = np.asarray(rotation, dtype=float)
    assignments[0].value = ast.parse(_orientation_literal(rotation), mode="eval").body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def controller_for(*args, nominal_wrist_rotation, **kwargs):
    """Wrap R25 and change only the nominal wrist orientation initializer."""
    original, source, hands, estimated = r25.controller_for(*args, **kwargs)
    transformed = replace_nominal_wrist_rotation(source, nominal_wrist_rotation)
    namespace = dict(original.__globals__)
    exec(compile(transformed, "<public_ycb_orientation_candidate_r26>", "exec"), namespace)
    task = namespace["run_task"]
    task.__dict__.update(original.__dict__)
    task.r26_orientation_candidate_transform = {
        "format": "r26_orientation_candidate_source_transform_v1",
        "inherited_r25_source_sha256": sha256_bytes(source.encode()),
        "transformed_source_sha256": sha256_bytes(transformed.encode()),
        "nominal_wrist_rotation_source": "profile.desired_world_wrist_rotation",
        "nominal_wrist_rotation_world": np.asarray(nominal_wrist_rotation, dtype=float).tolist(),
        "r3_pose_rotation_composition_preserved": True,
        "runtime_target_pose_or_qpos_writes_added": False,
    }
    return task, transformed, hands, estimated


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _r25_baseline_inputs():
    """Fail closed on the exact completed R25 run used as the comparison."""
    summary_path = checked_file(BASELINE, "summary.json")
    protocol_path = checked_file(BASELINE, "protocol.json")
    status_path = checked_file(BASELINE_JOB, "status.json")
    request_path = checked_file(BASELINE_JOB, "request.json")
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    status = json.loads(status_path.read_text())
    request = json.loads(request_path.read_text())
    before = protocol.get("source_hashes_before", {})
    row = summary.get("rows", [{}])[0] if summary.get("rows") else {}
    command = request.get("command", [])
    profile_relative = BASELINE_PROFILE.relative_to(PROJECT).as_posix()
    if (sha256(summary_path) != BASELINE_SHA256 or summary.get("complete") is not True
            or summary.get("runner_errors") != 0 or summary.get("protocol") != protocol
            or protocol.get("format") != r25.PROTOCOL_FORMAT or len(summary.get("rows", [])) != 1
            or protocol.get("source_changed_during_run") is not False
            or before != protocol.get("source_hashes_after") or len(before) != 207
            or status.get("state") != "FINISHED" or status.get("terminal") is not True
            or status.get("child_exit_observed") is not True
            or type(status.get("returncode")) is not int or status["returncode"] != 0
            or command.count("scripts/develop_public_ycb_rate_limited_probe_r25.py") != 1
            or r25.r24.r23.r22._flag_argument(command, "--output") != BASELINE.relative_to(PROJECT).as_posix()
            or r25.r24.r23.r22._flag_argument(command, "--profile-json") != profile_relative
            or command[command.index("--cases") + 1:] != ["ycb_can:3414"]
            or row.get("kind") != "ycb_can" or row.get("seed") != 3414
            or row.get("runner_error") is not None or row.get("success") is not False
            or row.get("held_lift") is not False
            or row.get("failure") != "RECOVERY_CONTACT_GUARD_NOT_RELIEVED"):
        raise ValueError("R26 requires the explicit completed R25 failure baseline")
    archive_path = checked_file(BASELINE, "frozen_source.zip")
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != 207 or set(archive.namelist()) != set(before):
            raise ValueError("R25 source ZIP manifest differs")
        for relative, expected in before.items():
            current = checked_file(PROJECT, relative)
            if sha256(current) != expected or sha256_bytes(archive.read(relative)) != expected:
                raise ValueError("Frozen R25 source differs: " + relative)
    return {"summary": summary, "protocol": protocol, "row": row, "source_hashes": before,
        "archive_sha256": sha256(archive_path), "request": request}


def _checked_candidate_profile(profile_path):
    relative = Path(profile_path).absolute().relative_to(PROJECT)
    candidate = checked_file(PROJECT, relative)
    if candidate != CANDIDATE_PROFILE or sha256(candidate) != CANDIDATE_PROFILE_SHA256:
        raise ValueError("R26 requires the separately frozen rotated candidate profile SHA256")
    return candidate


def validate_inputs(profile_path, expected_baseline_summary_sha256):
    if r25.r24.r23.checked_sha256(expected_baseline_summary_sha256) != BASELINE_SHA256:
        raise ValueError("R26 requires the explicit finalized R25 summary SHA256")
    profile_path = _checked_candidate_profile(profile_path)
    rotation = load_nominal_rotation(profile_path)
    # The baseline always uses its original profile; the candidate is a new
    # runtime input and must not be presented as the profile executed by R25.
    inherited = r25.validate_inputs(BASELINE_PROFILE, r25.BASELINE_SHA256)
    baseline = _r25_baseline_inputs()
    candidate_document = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    baseline_document = json.loads(BASELINE_PROFILE.read_text(encoding="utf-8-sig"))
    profile_changes = {key: {"baseline": baseline_document["ycb_can"].get(key), "candidate": value}
        for key, value in candidate_document["ycb_can"].items()
        if value != baseline_document["ycb_can"].get(key)}
    declaration = copy.deepcopy(inherited)
    declaration.update({
        "format": "r26_orientation_candidate_protocol_v1",
        "runner_revision": PROTOCOL_FORMAT,
        "r25_comparison_summary_sha256": BASELINE_SHA256,
        "inherited_r25_declaration": copy.deepcopy(inherited),
        "inherited_r25_source_hashes": copy.deepcopy(baseline["source_hashes"]),
        "inherited_r25_force_control": baseline["protocol"].get("force_control"),
        "baseline_profile": {"path": BASELINE_PROFILE.relative_to(PROJECT).as_posix(),
            "sha256": sha256(BASELINE_PROFILE), "bytes": BASELINE_PROFILE.stat().st_size},
        "candidate_profile": {"path": profile_path.relative_to(PROJECT).as_posix(),
            "sha256": CANDIDATE_PROFILE_SHA256, "bytes": profile_path.stat().st_size},
        "candidate_hand_profile_changes": profile_changes,
        "changed_control": "Use the frozen rotated grasp profile for nominal wrist orientation and hand parameters; retain R3 pose_rotation composition and all R25/R24 control gates.",
        "nominal_wrist_rotation_world": rotation.tolist(),
        "nominal_wrist_rotation_source": "profile.desired_world_wrist_rotation",
        "r3_pose_rotation_composition": "kin.rotation = pose_rotation @ _nominal_wrist_rotation",
        "runtime_target_pose_or_qpos_writes_added": False,
        "scope": "One reused can/seed; orientation candidate only; no success or generalization claim.",
    })
    return declaration


def snapshot(profile_path, declaration):
    profile_path = _checked_candidate_profile(profile_path)
    candidate = declaration["candidate_profile"]
    if (candidate["path"] != profile_path.relative_to(PROJECT).as_posix()
            or candidate["sha256"] != CANDIDATE_PROFILE_SHA256
            or candidate["bytes"] != profile_path.stat().st_size):
        raise ValueError("R26 candidate profile binding differs from its declaration")
    result = {}
    for relative, expected in declaration["inherited_r25_source_hashes"].items():
        actual = sha256(checked_file(PROJECT, relative))
        if actual != expected:
            raise ValueError("R26 frozen R25 input changed: " + relative)
        result[relative] = actual
    result[candidate["path"]] = candidate["sha256"]
    for relative in NEW_SOURCES:
        result[relative] = sha256(checked_file(PROJECT, relative))
    return result


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    r25.tagged_protocol(value, declaration["inherited_r25_declaration"])
    active = {"runner_revision": PROTOCOL_FORMAT, "controller_class": CONTROLLER_CLASS,
        "controller_revision_scope": declaration["changed_control"]}
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        protocol.update(active, format=PROTOCOL_FORMAT,
            r26_orientation_candidate_experiment=copy.deepcopy(declaration),
            inherited_r25_rate_limited_probe_experiment=copy.deepcopy(declaration["inherited_r25_declaration"]),
            candidate_profile=copy.deepcopy(declaration["candidate_profile"]),
            candidate_hand_profile_changes=copy.deepcopy(declaration["candidate_hand_profile_changes"]),
            nominal_wrist_rotation_world=copy.deepcopy(declaration["nominal_wrist_rotation_world"]),
            nominal_wrist_rotation_source=declaration["nominal_wrist_rotation_source"],
            r3_pose_rotation_composition=declaration["r3_pose_rotation_composition"],
            force_control="R26 uses a separately frozen rotated grasp profile, including its nominal hand closure and force goals, and replaces the generated R3 nominal wrist initializer with that profile's rotation. R25 rate limiting, R24 current-contact tracking, physical parameters and strict task gates remain inherited.")
        for planned in protocol.get("schedule", []):
            planned.update(active)
    for row in value.get("rows", []):
        row.update(active)
    if "protocol" in value or {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value.update(active)
    return value


def isolated_runner(profile_path, declaration):
    candidate_path = _checked_candidate_profile(profile_path)
    candidate_hand_profile = json.loads(candidate_path.read_text(encoding="utf-8-sig"))["ycb_can"]
    rotation = np.asarray(declaration["nominal_wrist_rotation_world"], dtype=float)
    bindings = {}

    def bound_controller(kind, directory, observations, guard, profile, mass_kg):
        if kind != "ycb_can" or profile != candidate_hand_profile:
            raise ValueError("R26 runtime hand profile differs from the frozen rotated candidate")
        result = controller_for(kind, directory, observations, guard, profile, mass_kg,
            nominal_wrist_rotation=rotation)
        bindings[Path(directory).resolve()] = result[0]
        return result

    def bound_snapshot(profile, inherited):
        if inherited != declaration["inherited_r25_declaration"]:
            raise ValueError("R26 snapshot received a different inherited R25 declaration")
        return snapshot(profile, declaration)

    namespace = dict(r25.isolated_runner.__globals__, controller_for=bound_controller,
        snapshot=bound_snapshot,
        tagged_protocol=lambda value, inherited: tagged_protocol(value, declaration))
    factory = FunctionType(r25.isolated_runner.__code__, namespace, r25.isolated_runner.__name__,
        r25.isolated_runner.__defaults__, r25.isolated_runner.__closure__)
    original = factory(profile_path, declaration["inherited_r25_declaration"])
    previous = original.__globals__["previous"]
    proxy = copy.copy(previous)

    def write_json(path, value):
        if Path(path).name in ("result.json", "metadata.json"):
            task = bindings.get(Path(path).parent.resolve())
            if task is None:
                raise RuntimeError("Missing actual R26 orientation-candidate binding")
            value.update(
                orientation_candidate_source_transform=copy.deepcopy(task.r26_orientation_candidate_transform),
                nominal_wrist_rotation_world=copy.deepcopy(task.r26_orientation_candidate_transform["nominal_wrist_rotation_world"]),
                inherited_r25_orientation_audit_does_not_audit_r26=True,
            )
        return previous.atomic_json(path, value)

    proxy.atomic_json = write_json
    return r25.r24.clone(original, previous=proxy)


def run(output, profile_path, cases, *, expected_baseline_summary_sha256):
    if output.exists() or output.is_symlink():
        raise FileExistsError("R26 requires a new output directory")
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R26 isolates ycb_can:3414")
    declaration = validate_inputs(profile_path, expected_baseline_summary_sha256)
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--expected-baseline-summary-sha256", required=True)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1])) for item in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases,
        expected_baseline_summary_sha256=args.expected_baseline_summary_sha256)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
