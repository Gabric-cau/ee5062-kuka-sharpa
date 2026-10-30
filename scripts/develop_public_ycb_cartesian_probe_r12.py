#!/usr/bin/env python3
"""Track actual wrist probe progress with bounded Cartesian reference correction."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np

import develop_public_ycb_tracking_probe_r11 as r11


PROJECT = r11.PROJECT
PROBE_PARAMETERS = dict(r11.PROBE_PARAMETERS)
CARTESIAN_PROBE_PARAMETERS = {
    "path_seconds": 3.,
    "settle_seconds": 3.,
    "maximum_probe_seconds": 6.,
    "distance_m": .005,
    "actual_error_gain": 2.,
    "reference_correction_limit_m": .002,
    "cartesian_ik_sample_seconds": .02,
    "scope": "Actual wrist proprioception only; fixed 5mm actual endpoint and every R9/R11 tactile, visual, LP, motion and strict task criterion remain unchanged",
}


def bounded_probe_reference(start_world_m, actual_world_m, elapsed_s):
    start, actual = np.asarray(start_world_m, dtype=float), np.asarray(actual_world_m, dtype=float)
    if start.shape != (3,) or actual.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(actual).all():
        raise ValueError("Probe wrist positions must be finite three-vectors")
    if not np.isfinite(elapsed_s) or elapsed_s < 0.:
        raise ValueError("Probe elapsed time must be finite and nonnegative")
    fraction = float(np.clip(elapsed_s / CARTESIAN_PROBE_PARAMETERS["path_seconds"], 0., 1.))
    blend = 3. * fraction ** 2 - 2. * fraction ** 3
    desired = start + np.array([0., 0., CARTESIAN_PROBE_PARAMETERS["distance_m"] * blend])
    error = desired - actual
    correction = CARTESIAN_PROBE_PARAMETERS["actual_error_gain"] * error
    length = float(np.linalg.norm(correction))
    if length > CARTESIAN_PROBE_PARAMETERS["reference_correction_limit_m"]:
        correction *= CARTESIAN_PROBE_PARAMETERS["reference_correction_limit_m"] / length
    return {"path_elapsed_s": float(elapsed_s), "cubic_progress": blend,
        "actual_admission_start_world_m": start.tolist(),
        "desired_actual_path_world_m": desired.tolist(),
        "actual_wrist_at_ik_sample_world_m": actual.tolist(),
        "actual_path_error_world_m": error.tolist(),
        "bounded_reference_correction_world_m": correction.tolist(),
        "cartesian_ik_target_world_m": (desired + correction).tolist(),
        "target_pose_used": False}


class CartesianProbeHand(r11.BoundedTrackingProbeHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_cartesian_events = []
        self.latest_probe_cartesian = None

    def probe_cartesian_target(self, data, elapsed_s):
        row = bounded_probe_reference(self.probe_start_world_m,
            data.xpos[self.estimated.kin.body], elapsed_s)
        row.update(timestamp_s=float(data.time), scope="Actual-wrist reference correction during active unconfirmed probe only")
        self.latest_probe_cartesian = row
        self.probe_cartesian_events.append(row)
        return np.asarray(row["cartesian_ik_target_world_m"])

    def probe_cartesian_snapshot(self, data):
        if self.latest_probe_cartesian is None:
            return None
        return {**self.latest_probe_cartesian,
            "sample_age_s": float(data.time - self.latest_probe_cartesian["timestamp_s"]),
            "observation_timestamp_s": float(data.time),
            "current_actual_wrist_world_m": data.xpos[self.estimated.kin.body].tolist(),
            "currently_probe_active": bool(self.probe_active)}


def controller_for(*args, **kwargs):
    base = r11.r9.r8.controller_for
    factory_namespace = dict(base.__globals__)
    factory_namespace["LoadTransferProbeHand"] = CartesianProbeHand
    factory = FunctionType(base.__code__, factory_namespace, base.__name__, base.__defaults__, base.__closure__)
    original, source, hands, estimated = factory(*args, **kwargs)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    duration_index = next(index for index, node in enumerate(move.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "duration" for target in node.targets))
    move.body[duration_index + 1:duration_index + 1] = ast.parse("""
if state == State.TRIAL_LIFT and hand.probe_active:
    duration = _cartesian_probe_parameters['maximum_probe_seconds']
""").body
    loop = next(node for node in move.body if isinstance(node, ast.For))
    segment = next(node for node in loop.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cartesian_motion and step % segment_steps == 0")
    position_index = next(index for index, node in enumerate(segment.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "next_position" for target in node.targets))
    original_position = segment.body[position_index]
    branch = ast.parse("""
if state == State.TRIAL_LIFT and hand.probe_active:
    next_position = hand.probe_cartesian_target(data, end_step * model.opt.timestep)
else:
    pass
""").body[0]
    branch.orelse = [original_position]
    segment.body[position_index] = branch
    removed = [node for node in loop.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25"]
    if len(removed) != 1:
        raise RuntimeError("Frozen controller must contain exactly one obsolete CLOSE readiness exit")
    loop.body.remove(removed[0])
    recording = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) and ast.unparse(target) == "trace[-1]['probe_policy']" for target in node.targets))
    loop.body[recording + 1:recording + 1] = ast.parse("trace[-1]['probe_cartesian_tracking'] = hand.probe_cartesian_snapshot(data)").body
    loop.body.extend(ast.parse("""
if state == State.TRIAL_LIFT and hand.probe_active and hand.probe_confirmed_seconds >= _probe_parameters['confirmation_seconds']:
    trace[-1]['probe_termination'] = 'ACTUAL_ENDPOINT_AND_CONTINUOUS_ORIGINAL_CONFIRMATION_GATES'
    break
""").body)
    namespace = dict(original.__globals__)
    namespace["_cartesian_probe_parameters"] = CARTESIAN_PROBE_PARAMETERS
    exec(compile(ast.fix_missing_locations(tree), "<r12_bounded_actual_wrist_probe>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r11.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_cartesian_probe_r12.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r11.r9.r8.r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r11.r9.r8._runner)))
    replacements = {
        "public_ycb_load_transfer_probe_development_r8": "public_ycb_actual_wrist_probe_development_r12",
        "load_transfer_probe_r8": "actual_wrist_probe_r12",
        "public_ycb_load_transfer_probe_r8": "public_ycb_actual_wrist_probe_r12",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_actual_wrist_probe_runner_r12>",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
            if node.value.startswith("\nprotocol['latched_pressure_escape']"):
                node.value = node.value.replace(
                    "Frozen R6 hand pressure regulator and latched safety, original8s CLOSE, physical gains/contact/torque caps unchanged. Emergency actual-joint reference resets and original joint-limit relief blocks remain explicit.",
                    "Inherited frozen R11 .1P/.4N active-finger reserve/6N ceiling/.15s target filter with original free-contact behavior and R6 emergency precedence; physical gains/contact/torque caps unchanged.")
                node.value += "\nprotocol['pressure_tracking'] = dict(_tracking_parameters)\nprotocol['tracking_regulator_sha256'] = _tracking_regulator_sha256\nprotocol['tracking_regulator_source'] = _tracking_regulator_source\nprotocol['actual_wrist_probe_motion'] = dict(_cartesian_probe_parameters)\nprotocol['close_termination'] = 'Only original R8 continuous eligible-probe exit or unchanged 8s deadline; obsolete original1.5g-ready exit removed because it can precede pressure stability.'\n"
                node.value = node.value.replace("Actual5mm upward motion with3s cubic command,", "Actual5mm upward motion with3s cubic actual-wrist reference, bounded2mm Cartesian reference correction and up to3s endpoint settling,")
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    for values in (("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz"),
            ("probe_cartesian", "probe_cartesian_events", "probe_cartesian.json.gz")):
        artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in values], ctx=ast.Load()))
    namespace_update = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "namespace.update")
    for name in ("_tracking_parameters", "_tracking_regulator_sha256", "_tracking_regulator_source", "_cartesian_probe_parameters"):
        namespace_update.keywords.append(ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load())))
    namespace = dict(r11.r9.r8._runner.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS,
        _tracking_parameters=r11.TRACKING_PARAMETERS,
        _tracking_regulator_sha256=r11.TRACKING_REGULATOR_SHA256,
        _tracking_regulator_source=r11.TRACKING_REGULATOR_SOURCE,
        _cartesian_probe_parameters=CARTESIAN_PROBE_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<r12_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    original = r11.r9.r8.main
    namespace = dict(original.__globals__)
    namespace.update(run=run, __doc__=__doc__)
    return FunctionType(original.__code__, namespace, "main", original.__defaults__, original.__closure__)()


if __name__ == "__main__":
    raise SystemExit(main())
