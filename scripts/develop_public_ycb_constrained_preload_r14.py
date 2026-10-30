#!/usr/bin/env python3
"""Bounded direction-constrained nominal tracking and stricter preload admission."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np
from scipy.optimize import minimize

import develop_public_ycb_joint_admittance_r13 as r13


PROJECT = r13.PROJECT
PROBE_PARAMETERS = dict(r13.PROBE_PARAMETERS)
PRELOAD_PARAMETERS = {"close_seconds": 16., "continuous_seconds": .25,
    "group_error_limit_n": .35, "window_target_change_limit_n": .35,
    "direction_error_epsilon_m_s": 1e-12,
    "direction_residual_tolerance_m_s": 1e-10, "joint_velocity_residual_tolerance_rad_s": 1e-9,
    "scope": "Stricter current measured-fixed-normal1g and group-pressure preload before unchanged original probe admission; all original confirmation, motion, force and task gates retained"}


def constrained_joint_velocity(jacobian, desired, lower, upper):
    """Solve the original damped objective with joint-box and row-direction constraints."""
    matrix, target = np.asarray(jacobian, dtype=float), np.asarray(desired, dtype=float)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    result = {"accepted": False, "reason": "INVALID_INPUT", "runtime_target_state_used": False}
    if (matrix.ndim != 2 or min(matrix.shape, default=0) < 1 or target.shape != (matrix.shape[0],) or low.shape != (matrix.shape[1],)
            or high.shape != low.shape or not all(np.isfinite(a).all() for a in (matrix, target, low, high))):
        return np.zeros(low.shape), result
    if np.any(low > high):
        return np.zeros_like(low), dict(result, reason="EMPTY_JOINT_BOX", zero_velocity_in_box=False)
    speed_scale = np.maximum(np.maximum(np.abs(low), np.abs(high)), 1e-6)
    normal_scale = r13.JOINT_PARAMETERS["normal_velocity_limit_m_s"]
    damping = r13.JOINT_PARAMETERS["jacobian_damping_m_rad"]
    scaled_matrix = matrix * speed_scale[None, :] / normal_scale
    scaled_target = target / normal_scale
    regularizer = damping * speed_scale / normal_scale
    direction_mask = np.abs(target) > PRELOAD_PARAMETERS["direction_error_epsilon_m_s"]
    directional = np.sign(target[direction_mask])[:, None] * scaled_matrix[direction_mask]

    def objective(value):
        residual = scaled_matrix @ value - scaled_target
        return .5 * float(residual @ residual + (regularizer * value) @ (regularizer * value))

    def gradient(value):
        return scaled_matrix.T @ (scaled_matrix @ value - scaled_target) + regularizer ** 2 * value

    constraints = [] if not len(directional) else [{"type": "ineq",
        "fun": lambda value: directional @ value, "jac": lambda value: directional}]
    solved = minimize(objective, np.clip(np.zeros_like(low), low, high) / speed_scale,
        jac=gradient, method="SLSQP", bounds=list(zip(low / speed_scale, high / speed_scale, strict=True)),
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 100, "disp": False})
    velocity = np.asarray(solved.x) * speed_scale
    finite = bool(np.isfinite(velocity).all())
    raw_violation = float(max(0., np.max(low - velocity), np.max(velocity - high))) if finite else None
    if finite and raw_violation <= PRELOAD_PARAMETERS["joint_velocity_residual_tolerance_rad_s"]:
        velocity = np.clip(velocity, low, high)
    elif not finite:
        velocity = np.zeros_like(low)
    predicted = matrix @ velocity
    directional_residual = np.sign(target[direction_mask]) * predicted[direction_mask]
    violation = float(max(0., np.max(low - velocity), np.max(velocity - high)))
    reverse = float(max(0., -np.min(directional_residual))) if len(directional_residual) else 0.
    accepted = bool(solved.success and finite and raw_violation <= PRELOAD_PARAMETERS["joint_velocity_residual_tolerance_rad_s"]
        and violation <= PRELOAD_PARAMETERS["joint_velocity_residual_tolerance_rad_s"]
        and reverse <= PRELOAD_PARAMETERS["direction_residual_tolerance_m_s"])
    result.update(accepted=accepted, reason="CONSTRAINED_QP_ACCEPTED" if accepted else "CONSTRAINED_QP_REJECTED",
        solver="scipy.optimize.minimize:SLSQP", solver_status=int(getattr(solved, "status", 0 if solved.success else -1)), solver_message=str(solved.message),
        iterations=int(getattr(solved, "nit", 0)), objective=float(objective(velocity / speed_scale)),
        lower_velocity_rad_s=low.tolist(), upper_velocity_rad_s=high.tolist(),
        zero_velocity_in_box=bool(np.all(low <= 0.) and np.all(high >= 0.)),
        candidate_joint_velocity_rad_s=velocity.tolist(), desired_normal_velocity_m_s=target.tolist(),
        pre_rounding_joint_box_violation_rad_s=raw_violation,
        predicted_normal_velocity_m_s=predicted.tolist(), joint_box_violation_rad_s=violation,
        maximum_reverse_normal_velocity_m_s=reverse, damping_m_rad=damping,
        velocity_coordinate_scale_rad_s=speed_scale.tolist(), normal_scale_m_s=normal_scale,
        direction_error_epsilon_m_s=PRELOAD_PARAMETERS["direction_error_epsilon_m_s"])
    return (velocity if accepted else np.zeros_like(low)), result


def current_joint_box(hand, data, names, dt):
    if not np.isfinite(dt) or dt <= 0.:
        raise ValueError("A positive current write interval is required")
    lower, upper = [], []
    for name in names:
        reference = hand.joint_references[name]
        actual, bias = hand._actual(data, name), hand.initial_command_bias[name]
        joint = int(hand.model.actuator_trnid[hand.actuators[name], 0])
        _, ratio = hand.flexion_joints()[name]
        low = max(float(hand.model.jnt_range[joint, 0]), reference - hand.config.relief_rate * abs(ratio) * dt,
            actual + bias - r13.JOINT_PARAMETERS["additional_bias_limit_rad"])
        high = min(float(hand.model.jnt_range[joint, 1]), reference + hand.config.close_rate * abs(ratio) * dt,
            actual + bias + r13.JOINT_PARAMETERS["additional_bias_limit_rad"])
        lower.append((low - reference) / dt)
        upper.append((high - reference) / dt)
    return np.asarray(lower), np.asarray(upper)


def preload_group_values(groups):
    return {f"{g['finger']}:{g['hand_body_name']}:{','.join(map(str, sorted(g['raw_input_indices'])))}":
        {"target_n": float(g["target_normal_n"]), "measured_n": float(g["measured_normal_n"])} for g in groups}


class ConstrainedPreloadHand(r13.JointAdmittanceProbeHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qp_events, self.preload_events = [], []
        self.preload_ready = False
        self.preload_start_s = None
        self.preload_anchor_targets = None
        self.preload_previous_sample = None

    def phase_begin(self, phase, data):
        changed = phase != self.phase
        super().phase_begin(phase, data)
        if changed:
            self.preload_ready = False
            self.preload_start_s = None
            self.preload_anchor_targets = self.preload_previous_sample = None

    def write(self, data):
        self.update_pressure_relief(data, self.reader.read(data))
        dt = 0. if self.last_joint_write_s is None else float(data.time) - self.last_joint_write_s
        relevant = (self.joint_mode == "JOINT_CONTROL" and self.joint_target.get("feasible")
            and self.target_timestamp_s is not None and abs(data.time - self.target_timestamp_s) < 1e-9
            and not self.emergency_active and not self.emergency_phase_hold and dt > 0.)
        diagnostics = []
        if relevant:
            for solve in self.joint_solves:
                names = solve["joint_names"]
                lower, upper = current_joint_box(self, data, names, dt)
                velocity, row = constrained_joint_velocity(solve["compression_jacobian_m_rad"],
                    solve["desired_normal_velocity_m_s"], lower, upper)
                row.update(finger=solve["finger"], joint_names=names,
                    compression_jacobian_m_rad=solve["compression_jacobian_m_rad"],
                    lower_velocity_rad_s=lower.tolist(), upper_velocity_rad_s=upper.tolist(),
                    desired_normal_velocity_m_s=solve["desired_normal_velocity_m_s"])
                diagnostics.append(row)
                if row["accepted"]:
                    self.joint_velocities.update(zip(names, velocity.tolist(), strict=True))
                    # Preserve the inherited R13 DLS record; these fields describe
                    # the separate current-write constrained solution.
                    solve.update(constrained_joint_velocity_rad_s=velocity.tolist(),
                        constrained_predicted_normal_velocity_m_s=row["predicted_normal_velocity_m_s"],
                        constrained_direction_hold=False, nominal_solver="R14_CONSTRAINED_CURRENT_WRITE_QP",
                        constrained_qp=row)
            if not all(row["accepted"] for row in diagnostics) or not diagnostics:
                self.joint_velocities = {}
                self._mode("HOLD_INVALID", data, "CURRENT_WRITE_QP_FAILED_NO_NOMINAL_UPDATE")
        previous = self.joint_references.copy()
        expected = {name: value + dt * self.joint_velocities.get(name, 0.)
            for name, value in self.joint_references.items()} if relevant and self.joint_mode == "JOINT_CONTROL" else None
        super().write(data)
        if relevant:
            applied = {name: float(data.ctrl[self.actuators[name]]) for name in self.joint_references}
            mismatch = None if expected is None else max((abs(applied[name] - value)
                for name, value in expected.items()), default=0.)
            safety = bool(self.emergency_active or self.emergency_phase_hold)
            applied_checks = []
            for solve in diagnostics:
                names = solve["joint_names"]
                actual_velocity = np.array([(applied[name] - previous[name]) / dt for name in names])
                lower, upper = np.asarray(solve["lower_velocity_rad_s"]), np.asarray(solve["upper_velocity_rad_s"])
                predicted = np.asarray(solve["compression_jacobian_m_rad"]) @ actual_velocity
                desired = np.asarray(solve["desired_normal_velocity_m_s"])
                direction_mask = np.abs(desired) > PRELOAD_PARAMETERS["direction_error_epsilon_m_s"]
                reverse = float(max(0., -np.min(np.sign(desired[direction_mask]) * predicted[direction_mask]))) if np.any(direction_mask) else 0.
                box = float(max(0., np.max(lower - actual_velocity), np.max(actual_velocity - upper)))
                applied_checks.append({"finger": solve["finger"], "applied_joint_velocity_rad_s": actual_velocity.tolist(),
                    "applied_predicted_normal_velocity_m_s": predicted.tolist(), "joint_box_violation_rad_s": box,
                    "maximum_reverse_normal_velocity_m_s": reverse,
                    "satisfies_nominal_constraints": bool(box <= PRELOAD_PARAMETERS["joint_velocity_residual_tolerance_rad_s"]
                        and reverse <= PRELOAD_PARAMETERS["direction_residual_tolerance_m_s"])})
            if expected is not None and not safety and not all(check["satisfies_nominal_constraints"] for check in applied_checks):
                self._mode("HOLD_INVALID", data, "APPLIED_NOMINAL_CONSTRAINT_VERIFICATION_FAILED")
            self.qp_events.append({"timestamp_s": float(data.time), "dt_s": dt, "phase": self.phase,
                "mode": self.joint_mode, "solves": diagnostics, "expected_command_rad": expected,
                "applied_command_rad": applied, "parent_projection_difference_rad": mismatch,
                "applied_motion_checks": applied_checks, "original_safety_override": safety,
                "scope": "Current nominal motion solve only; held references may violate a moved bias window after an explicit failed solve"})

    def current_preload_observation(self, data):
        readings = self.refresh_probe_solve(data, force=True)
        original = self._snapshot_probe(data, readings)
        pose = self.estimated.estimated_pose(data, carrying=self.estimated.payload_confirmed)
        association = r13.r11.r9.r8.r4.associate_tactile_contacts(self.reader.raw_contacts, pose,
            self.estimated.surface, now_s=float(data.time), pose_timestamp_s=self.estimated.pose_timestamp_s,
            max_pose_age_s=1., max_surface_distance_m=.006)
        target = {"feasible": False, "reason": "CURRENT_GATE_ASSOCIATION_INVALID"}
        if association.get("accepted"):
            patches = [dict(patch, hand_body_name=self.reader.raw_contacts[int(patch["input_index"])]["hand_body_name"])
                for patch in association["patches"]]
            target = r13.allocate_tracking_targets(patches, association["estimated_com_world_m"],
                self.estimated.mass_kg, original["pressures_n"], self.previous_feasible_targets)
        groups = preload_group_values(r13.group_tracking_patches(target["patches"])) if target.get("feasible") else {}
        return original, target, groups, self.contact_sample(self.reader.raw_contacts)

    def update_preload(self, data, *, admission=False):
        original, target, groups, sample = self.current_preload_observation(data)
        reasons = list(original["preprobe_reasons"])
        if self.probe_eligible_seconds < PROBE_PARAMETERS["preprobe_seconds"]:
            reasons.append("ORIGINAL_CONTINUOUS_ELIGIBILITY")
        if not (original["sampled_solve_current"] and original["cached_measurements_match"]
                and original["solve_age_s"] is not None and original["solve_age_s"] <= 1e-9):
            reasons.append("CURRENT_FIXED_NORMAL_INPUTS_REQUIRED")
        if not original["sampled_fixed_normal_1g_feasible"]:
            reasons.append("CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE")
        if not target.get("feasible") or not groups:
            reasons.append("CURRENT_TRACKING_TARGET_INVALID")
        if self.joint_mode != "JOINT_CONTROL":
            reasons.append("JOINT_TRACKING_NOT_ACTIVE")
        if any(abs(g["target_n"] - g["measured_n"]) > PRELOAD_PARAMETERS["group_error_limit_n"] for g in groups.values()):
            reasons.append("GROUP_PRESSURE_ERROR")
        if self.preload_previous_sample is not None and not r13.r11.r9.r8.contact_samples_match(sample, self.preload_previous_sample):
            reasons.append("CURRENT_CONTACT_SAMPLE_CHANGED")
        self.preload_previous_sample = sample
        reset = None
        if self.preload_anchor_targets is not None:
            if set(groups) != set(self.preload_anchor_targets):
                reset = "GROUP_IDENTITY_CHANGED"
            elif max((abs(g["target_n"] - self.preload_anchor_targets[key]) for key, g in groups.items()), default=0.) > PRELOAD_PARAMETERS["window_target_change_limit_n"]:
                reset = "TARGET_MOVED_FROM_WINDOW_ORIGIN"
        if reasons or reset:
            self.preload_start_s = None
            self.preload_anchor_targets = None
        if not reasons and self.preload_start_s is None:
            self.preload_start_s = float(data.time)
            self.preload_anchor_targets = {key: value["target_n"] for key, value in groups.items()}
        duration = 0. if self.preload_start_s is None else float(data.time - self.preload_start_s)
        self.preload_ready = not reasons and duration >= PRELOAD_PARAMETERS["continuous_seconds"] - 1e-9
        row = {"timestamp_s": float(data.time), "phase": self.phase, "admission_recheck": admission,
            "ready": bool(self.preload_ready), "continuous_seconds": duration, "reasons": reasons,
            "window_reset_reason": reset, "window_anchor_targets_n": self.preload_anchor_targets,
            "current_groups": groups, "current_target_lp": target, "original_probe_snapshot": original,
            "target_timestamp_s": float(data.time), "target_age_s": 0.,
            "target_scope": "Gate-only LP on current post-integration association; current measured group loads; does not advance tracking reference"}
        self.preload_events.append(row)
        return row

    def after_step(self, state, data, dt):
        failure = super().after_step(state, data, dt)
        if state.value == "CLOSE":
            self.update_preload(data)
        return failure

    def admit_probe(self, data):
        row = self.update_preload(data, admission=True)
        if not row["ready"]:
            self.probe_events.append({**row["original_probe_snapshot"], "admission": False,
                "preload_admission": False, "preload_reasons": row["reasons"],
                "preload_continuous_seconds": row["continuous_seconds"]})
            return False
        return super().admit_probe(data)


def controller_for(*args, **kwargs):
    base = r13.r11.r9.r8.controller_for
    namespace = dict(base.__globals__)
    namespace["LoadTransferProbeHand"] = ConstrainedPreloadHand
    factory = FunctionType(base.__code__, namespace, base.__name__, base.__defaults__, base.__closure__)
    original, source, hands, estimated = factory(*args, **kwargs)
    tree = ast.parse(source)
    obsolete = []
    eligible = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25":
            obsolete.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.CLOSE and hand.probe_eligible_seconds >= _probe_parameters['preprobe_seconds']":
            eligible.append(node)
    if len(obsolete) != 1 or len(eligible) != 1:
        raise RuntimeError("Frozen CLOSE termination structure changed")
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and obsolete[0] in node.body)
    loop.body.remove(obsolete[0])
    eligible[0].test = ast.parse("state == State.CLOSE and hand.preload_ready", mode="eval").body
    namespace = dict(original.__globals__)
    namespace["_phase_seconds"] = dict(namespace["_phase_seconds"], CLOSE=PRELOAD_PARAMETERS["close_seconds"])
    exec(compile(ast.fix_missing_locations(tree), "<r14_constrained_preload>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r13.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_constrained_preload_r14.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r13.r11.r9.r8.r4.previous.sha256(path)
    return sources


def _runner():
    base = r13.r11.r9.r8
    tree = ast.parse(textwrap.dedent(inspect.getsource(base._runner)))
    replacements = {"public_ycb_load_transfer_probe_development_r8": "public_ycb_constrained_preload_development_r14",
        "load_transfer_probe_r8": "constrained_preload_r14", "public_ycb_load_transfer_probe_r8": "public_ycb_constrained_preload_r14",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_constrained_preload_runner_r14>"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
            if node.value.startswith("\nprotocol['latched_pressure_escape']"):
                node.value += "\nprotocol['joint_admittance'] = dict(_joint_parameters)\nprotocol['scalar_acquisition_tracking'] = dict(_tracking_parameters)\nprotocol['tracking_regulator_sha256'] = _tracking_regulator_sha256\nprotocol['tracking_regulator_source'] = _tracking_regulator_source\nprotocol['constrained_preload'] = dict(_preload_parameters)\nprotocol['close_termination'] = 'Original R9 eligibility plus current measured fixed-normal1g and continuous group preload, or declared16s deadline; original probe and task conditions retained.'\nprotocol['force_control'] = 'Original scalar acquisition then R13 independent joint tracking, with R14 current-write direction-constrained joint-box QP in place of nominal DLS/post-bound rollback. Original range/rate/.02addedbias box and R6 final-write safety retained. Current measured fixed-normal1g plus .35N group preload for.25s precedes unchanged5mm probe. CLOSE16s; physical gains/friction/torque limits and all task outcomes unchanged.'\n"
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    for values in (("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz"),
            ("joint_admittance", "joint_events", "joint_admittance.json.gz"),
            ("constrained_joint", "qp_events", "constrained_joint.json.gz"),
            ("preload", "preload_events", "preload.json.gz")):
        artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in values], ctx=ast.Load()))
    update = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "namespace.update")
    for name in ("_joint_parameters", "_tracking_parameters", "_tracking_regulator_sha256", "_tracking_regulator_source", "_preload_parameters"):
        update.keywords.append(ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load())))
    namespace = dict(base._runner.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS,
        PHASE_SECONDS=dict(base.PHASE_SECONDS, CLOSE=PRELOAD_PARAMETERS["close_seconds"]),
        _joint_parameters=r13.JOINT_PARAMETERS, _tracking_parameters=r13.r11.TRACKING_PARAMETERS,
        _tracking_regulator_sha256=r13.r11.TRACKING_REGULATOR_SHA256,
        _tracking_regulator_source=r13.r11.TRACKING_REGULATOR_SOURCE, _preload_parameters=PRELOAD_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<r14_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    original = r13.r11.r9.r8.main
    namespace = dict(original.__globals__)
    namespace.update(run=run, __doc__=__doc__)
    return FunctionType(original.__code__, namespace, "main", original.__defaults__, original.__closure__)()


if __name__ == "__main__":
    raise SystemExit(main())
