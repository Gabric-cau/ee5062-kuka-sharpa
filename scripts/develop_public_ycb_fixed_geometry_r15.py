#!/usr/bin/env python3
"""Declared fixed-orientation geometry experiment using the frozen R14 controller."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
import develop_public_ycb_constrained_preload_r14 as r14


EXPECTED_R14_SHA256 = "32f78b5c5a8d7ddb914d6d502d2c4656bafd77423523cdf80e6a791ff17360c6"
EXPECTED_PROFILE_SHA256 = "afcbc124960136c95b1f1269dbe8dfe85974061513c5a722dc56f714da202b57"
BASE_PROFILE = PROJECT / "config/public_ycb_sidewall_20261021.json"
ALLOWED_PROFILE_CHANGES = {"thumb_opposition_rad", "contact_closure", "force_targets_n",
    "geometry_evidence", "geometry_evidence_sha256", "geometry_scope", "static_sidewall_metrics"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_changes(before, after):
    if set(before) != set(after):
        raise ValueError("R15 profile must retain exactly the original R14 field set")
    changes = {key: {"before": before[key], "after": after[key]} for key in before if before[key] != after[key]}
    if set(changes) - ALLOWED_PROFILE_CHANGES:
        raise ValueError("R15 may change only declared nominal grasp geometry fields")
    if after["thumb_opposition_rad"] != 1.3:
        raise ValueError("This fixed R15 experiment requires opposition 1.3 rad")
    if after["contact_closure"]["thumb"] != .31112518310546877:
        raise ValueError("R15 requires the frozen re-solved shallow thumb closure")
    if any(after["contact_closure"][finger] != value
            for finger, value in before["contact_closure"].items() if finger != "thumb"):
        raise ValueError("R15 fixed-orientation candidate retains the other four nominal closures")
    return changes


def validate_inputs(profile_path):
    frozen = Path(r14.__file__)
    if sha256(frozen) != EXPECTED_R14_SHA256:
        raise ValueError("Frozen R14 source SHA256 mismatch")
    if sha256(profile_path) != EXPECTED_PROFILE_SHA256:
        raise ValueError("Declared R15 fixed-orientation profile SHA256 mismatch")
    document = json.loads(Path(profile_path).read_text())
    if (document["candidate"] != {"opposition_rad": 1.3, "roll_deg": 0., "yaw_deg": 0.}
            or not document["r14_fixed_orientation_schema_compatible"]
            or document["requires_explicit_wrist_orientation_controller"]):
        raise ValueError("This wrapper rejects rotated candidates")
    original = json.loads(BASE_PROFILE.read_text())["ycb_can"]
    changes = profile_changes(original, document["ycb_can"])
    evidence = PROJECT / document["ycb_can"]["geometry_evidence"]
    if sha256(evidence) != document["ycb_can"]["geometry_evidence_sha256"]:
        raise ValueError("Frozen static geometry evidence SHA256 mismatch")
    return {"format": "r15_fixed_geometry_development_protocol_v1",
        "parent_controller": "R14 ConstrainedPreloadHand and generated run_task unchanged",
        "r14_source_sha256": sha256(frozen), "profile_sha256": sha256(profile_path),
        "base_profile_sha256": sha256(BASE_PROFILE), "changed_profile_fields": changes,
        "wrist_orientation_change_deg": {"roll": 0., "yaw": 0.},
        "controller_code_modified": False, "runtime_target_pose_or_qpos_added": False,
        "physics_force_and_task_gates_modified": False,
        "preload_parameters": copy.deepcopy(r14.PRELOAD_PARAMETERS),
        "probe_parameters": copy.deepcopy(r14.PROBE_PARAMETERS),
        "joint_parameters": copy.deepcopy(r14.r13.JOINT_PARAMETERS),
        "tracking_parameters": copy.deepcopy(r14.r13.r11.TRACKING_PARAMETERS),
        "strict_task_gates": {"channel_force_limit_n": 12., "dock_tolerance_m": .025,
            "stable_release_s": 3., "probe_displacement_m": .005,
            "current_fixed_normal_preload_gravity_margin": 1., "geometry_gravity_margin": 1.5},
        "scope": "Reused development seed with a new nominal grasp geometry. Static allocation does not prove dynamic force convergence or task success; no held-out success-rate claim."}


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        if "r15_geometry_experiment" not in protocol:
            protocol["inherited_controller_protocol_format"] = protocol["format"]
            protocol["format"] = "public_ycb_fixed_geometry_development_r15"
            protocol["r15_geometry_experiment"] = copy.deepcopy(declaration)
            for case in protocol["schedule"]:
                case["profile"] = "fixed_geometry_r15"
    return value


def isolated_runner(profile_path, declaration):
    namespace = dict(r14.run.__globals__)
    previous = namespace["previous"]
    proxy = SimpleNamespace(**vars(previous))
    atomic_json = previous.atomic_json

    def write_declared_json(path, value):
        return atomic_json(path, tagged_protocol(value, declaration))

    def snapshot(path):
        values = r14.snapshot(path)
        for filename in ("scripts/develop_public_ycb_fixed_geometry_r15.py", "tests/test_public_ycb_fixed_geometry_r15.py"):
            values[filename] = sha256(PROJECT / filename)
        evidence = json.loads(Path(profile_path).read_text())["ycb_can"]["geometry_evidence"]
        values[evidence] = sha256(PROJECT / evidence)
        return values

    proxy.atomic_json = write_declared_json
    namespace.update(previous=proxy, snapshot=snapshot)
    return FunctionType(r14.run.__code__, namespace, r14.run.__name__, r14.run.__defaults__, r14.run.__closure__)


def run(output, profile_path, cases):
    declaration = validate_inputs(profile_path)
    if not cases or any(kind != "ycb_can" or seed < 0 for kind, seed in cases) or len(set(cases)) != len(cases):
        raise ValueError("R15 requires unique nonnegative public-can development cases")
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
