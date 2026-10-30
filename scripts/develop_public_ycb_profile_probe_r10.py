#!/usr/bin/env python3
"""Isolated, hash-bound can-profile experiment with the unchanged R9 controller."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import re
import textwrap

import develop_public_ycb_force_band_r9 as r9


PROJECT = r9.PROJECT
PROBE_PARAMETERS = {**r9.PROBE_PARAMETERS,
    "profile_experiment": "Separately hash-bound catalogue hand/wrist profile; unchanged R9 control, physics, admission, confirmation and independent task scoring. Static candidate evidence is not dynamic grasp success."}


def validate_profile(path, expected_sha256):
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Expected profile SHA256 must be 64 lowercase hexadecimal characters")
    if r9.r8.r4.previous.sha256(path) != expected_sha256:
        raise ValueError("Profile bytes do not match the declared SHA256")


def snapshot(profile_path):
    sources = r9.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_profile_probe_r10.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r9.r8.r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r9.r8._runner)))
    replacements = {
        "public_ycb_load_transfer_probe_development_r8": "public_ycb_profile_probe_development_r10",
        "load_transfer_probe_r8": "profile_probe_r10",
        "public_ycb_load_transfer_probe_r8": "public_ycb_profile_probe_r10",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_profile_probe_runner_r10>",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in
        ("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz")], ctx=ast.Load()))
    namespace = dict(r9.r8._runner.__globals__)
    namespace.update(controller_for=r9.controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<r10_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    try:
        cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
        if len(set(cases)) != len(cases) or any(kind != "ycb_can" or seed < 0 for kind, seed in cases):
            raise ValueError("R10 requires unique nonnegative ycb_can seeds")
        validate_profile(args.profile_json, args.expected_profile_sha256)
    except (IndexError, OSError, ValueError) as error:
        parser.error(str(error))
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
