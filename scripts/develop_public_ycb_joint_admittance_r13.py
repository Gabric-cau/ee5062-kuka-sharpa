#!/usr/bin/env python3
"""Isolated tactile joint admittance; static target feasibility is not task success."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import textwrap
from types import FunctionType

import mujoco
import numpy as np
from scipy.optimize import linprog

import develop_public_ycb_tracking_probe_r11 as r11
import tactile_wrench_allocator as allocator


PROJECT = r11.PROJECT
PROBE_PARAMETERS = dict(r11.PROBE_PARAMETERS)
ACTIVE = ("thumb", "middle", "ring")
JOINT_PARAMETERS = {
    "tracking_patch_floor_n": .05, "active_finger_floor_n": .75,
    "pressure_limit_n": 6., "friction_coefficient": .8,
    "friction_safety_factor": .85, "gravity_margin": 1.5, "cone_rays": 16,
    "previous_reference_weight": .75, "patch_reference_weight": 1.,
    "normal_velocity_gain_m_per_n_s": .0005, "normal_velocity_limit_m_s": .001,
    "jacobian_damping_m_rad": .005, "additional_bias_limit_rad": .02,
    "group_normal_cosine_minimum": .9, "group_point_distance_m": .015,
    "association_max_pose_age_s": 1., "association_max_distance_m": .006,
    "target_cadence": "Every native regulate call on current tactile association",
    "scope": "Independent nominal flexion references only; original R6 emergency precedence, physics, current LP and strict gates retained; R11 3s/5mm probe motion retained",
}


def allocate_tracking_targets(patches, com, mass, pressures, previous=None):
    """Solve an unfiltered feasible target, retaining association raw indices."""
    result = {"feasible": False, "reason": "INVALID_INPUT", "runtime_target_state_used": False,
        "claim_scope": "Hypothetical static target; not measured equilibrium or dynamic success"}
    try:
        retained = [dict(patch) for patch in patches if patch["finger"] in ACTIVE
            and float(patch["measured_pressure_n"]) >= JOINT_PARAMETERS["tracking_patch_floor_n"]]
        if {patch["finger"] for patch in retained} != set(ACTIVE):
            return dict(result, reason="THREE_OBSERVED_ACTIVE_FINGERS_REQUIRED")
        raw_indices = [int(patch["input_index"]) for patch in retained]
        if len(set(raw_indices)) != len(raw_indices):
            return dict(result, reason="DUPLICATE_RAW_PATCH_INDEX")
        checked = allocator.allocate_contact_wrench(retained, com, mass,
            measured_pressures_n=pressures)
        if not checked.get("feasible"):
            return dict(result, reason="ORIGINAL_GEOMETRY_LP_" + checked.get("reason", "FAILED"))
        count, rays = len(retained), JOINT_PARAMETERS["cone_rays"]
        n = count * rays
        fingers = np.zeros((len(allocator.CHANNELS), n))
        normals = np.zeros((count, n))
        directions, arms, effective_mu = [], [], []
        for index, patch in enumerate(retained):
            normal = np.asarray(patch["normal_on_object_world"], dtype=float)
            normal = normal / np.linalg.norm(normal)
            first, second = allocator._tangents(normal)
            mu = min(.8, float(patch.get("friction_coefficient", .8))) * .85
            theta = 2. * math.pi * np.arange(rays) / rays
            directions.extend(normal + mu * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second))
            arms.extend([np.asarray(patch["position_world_m"]) - np.asarray(com)] * rays)
            section = slice(index * rays, (index + 1) * rays)
            fingers[allocator.CHANNELS.index(patch["finger"]), section] = 1.
            normals[index, section] = 1.
            effective_mu.append(mu)
        directions, arms = np.asarray(directions), np.asarray(arms)
        wrench = np.vstack([directions.T, np.cross(arms, directions).T])
        scaled = wrench.copy()
        scaled[3:] /= .05
        target = np.array([0., 0., float(mass) * 9.81 * 1.5, 0., 0., 0.])
        active_rows = fingers[[allocator.CHANNELS.index(finger) for finger in ACTIVE]]
        reference = np.array([float(pressures[finger]) if previous is None else
            .75 * float(previous[finger]) + .25 * float(pressures[finger]) for finger in ACTIVE])
        measured = np.array([patch["measured_pressure_n"] for patch in retained])
        slack = 3 + count
        channel_identity = np.zeros((3, slack))
        channel_identity[:, :3] = np.eye(3)
        patch_identity = np.zeros((count, slack))
        patch_identity[:, 3:] = np.eye(count)
        inequality = np.vstack([
            np.hstack([fingers, np.zeros((len(allocator.CHANNELS), slack))]),
            np.hstack([-active_rows, np.zeros((3, slack))]),
            np.hstack([active_rows, -channel_identity]), np.hstack([-active_rows, -channel_identity]),
            np.hstack([normals, -patch_identity]), np.hstack([-normals, -patch_identity])])
        upper = np.r_[[0. if finger == "palm" else 6. for finger in allocator.CHANNELS],
            np.full(3, -.75), reference, -reference, measured, -measured]
        solved = linprog(np.r_[np.full(n, 1e-4), np.ones(3), np.ones(count)],
            A_ub=inequality, b_ub=upper, A_eq=np.hstack([scaled, np.zeros((6, slack))]),
            b_eq=target, bounds=(0., None), method="highs",
            options={"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8, "time_limit": .1})
        if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
            return dict(result, reason="TRACKING_LP_FAILED", solver_status=int(solved.status))
        ray_loads = solved.x[:n]
        loads, patch_loads = fingers @ ray_loads, normals @ ray_loads
        residual = wrench @ ray_loads - target
        if (np.max(np.abs(residual[:3])) > 1e-5 or np.max(np.abs(residual[3:])) > 1e-6
                or np.max(inequality @ solved.x - upper) > 1e-6):
            return dict(result, reason="TRACKING_LP_RESIDUAL")
        hypothetical = [dict(patch, measured_pressure_n=float(load))
            for patch, load in zip(retained, patch_loads, strict=True)]
        one_g = allocator.allocate_contact_wrench(hypothetical, com, mass, gravity_margin=1.)
        if not one_g.get("measured_feasible"):
            return dict(result, reason="HYPOTHETICAL_TARGET_NOT_FIXED_NORMAL_1G_COMPATIBLE",
                hypothetical_target_fixed_normal_1g_feasible=False)
        return dict(result, feasible=True, reason="STATIC_TRACKING_TARGET_ALLOCATED",
            targets_n=dict(zip(allocator.CHANNELS, loads.tolist(), strict=True)),
            reference_n=dict(zip(ACTIVE, reference.tolist(), strict=True)),
            effective_patch_mu=effective_mu, target_wrench_residual=residual.tolist(),
            hypothetical_target_fixed_normal_1g_feasible=True,
            hypothetical_check_scope="Proposed exact normals only, not measured pressure evidence",
            excluded_tracking_raw_indices=[int(p["input_index"]) for p in patches
                if int(p["input_index"]) not in raw_indices],
            patches=[dict(patch, raw_input_index=raw, target_normal_n=float(load))
                for patch, raw, load in zip(retained, raw_indices, patch_loads, strict=True)])
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        return dict(result, details=str(error))


def group_tracking_patches(patches):
    """Aggregate close same-link patches, never oppositely directed normals."""
    groups = []
    for patch in sorted(patches, key=lambda p: (p["finger"], p["hand_body_name"],
            *np.asarray(p["position_world_m"]).tolist(), p["raw_input_index"])):
        normal = np.asarray(patch["normal_on_object_world"], dtype=float)
        normal /= np.linalg.norm(normal)
        point = np.asarray(patch["position_world_m"], dtype=float)
        compatible = next((group for group in groups if group["finger"] == patch["finger"]
            and group["hand_body_name"] == patch["hand_body_name"]
            and all(normal @ member["normal"] >= JOINT_PARAMETERS["group_normal_cosine_minimum"]
                and np.linalg.norm(point - member["point"]) <= JOINT_PARAMETERS["group_point_distance_m"]
                for member in group["members"])), None)
        if compatible is None:
            compatible = {"finger": patch["finger"], "hand_body_name": patch["hand_body_name"], "members": []}
            groups.append(compatible)
        compatible["members"].append({"patch": patch, "normal": normal, "point": point})
    output = []
    for group in groups:
        members = group["members"]
        weights = np.array([max(.05, float(m["patch"]["measured_pressure_n"])) for m in members])
        normal = np.average([m["normal"] for m in members], axis=0, weights=weights)
        normal /= np.linalg.norm(normal)
        output.append({"finger": group["finger"], "hand_body_name": group["hand_body_name"],
            "raw_input_indices": [m["patch"]["raw_input_index"] for m in members],
            "position_world_m": np.average([m["point"] for m in members], axis=0, weights=weights).tolist(),
            "normal_on_object_world": normal.tolist(),
            "measured_normal_n": float(sum(m["patch"]["measured_pressure_n"] for m in members)),
            "target_normal_n": float(sum(m["patch"]["target_normal_n"] for m in members))})
    return output


def damped_joint_velocity(compression_jacobian, pressure_errors):
    jacobian, error = np.asarray(compression_jacobian, dtype=float), np.asarray(pressure_errors, dtype=float)
    if (jacobian.ndim != 2 or error.shape != (jacobian.shape[0],)
            or not np.isfinite(jacobian).all() or not np.isfinite(error).all()):
        raise ValueError("Finite matching Jacobian and pressure errors required")
    desired = np.clip(error * JOINT_PARAMETERS["normal_velocity_gain_m_per_n_s"],
        -JOINT_PARAMETERS["normal_velocity_limit_m_s"], JOINT_PARAMETERS["normal_velocity_limit_m_s"])
    damping = JOINT_PARAMETERS["jacobian_damping_m_rad"]
    velocity = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(len(error)), desired)
    predicted = jacobian @ velocity
    incompatible = bool(np.any(predicted * desired < -1e-12))
    if incompatible:
        velocity[:] = 0.
    return velocity, {"compression_jacobian_m_rad": jacobian.tolist(),
        "singular_values_m_rad": np.linalg.svd(jacobian, compute_uv=False).tolist(),
        "pressure_errors_n": error.tolist(), "desired_normal_velocity_m_s": desired.tolist(),
        "unlimited_joint_velocity_rad_s": velocity.tolist(), "damping_m_rad": damping,
        "predicted_unbounded_normal_velocity_m_s": predicted.tolist(),
        "incompatible_direction_hold": incompatible}


def bounded_reference_step(reference, actual, initial_bias, velocity, dt, limits, close_rate, relief_rate,
                           *, constrain_added_bias=True):
    values = np.array([reference, actual, initial_bias, velocity, dt, *limits, close_rate, relief_rate], dtype=float)
    if not np.isfinite(values).all() or dt < 0. or min(close_rate, relief_rate) < 0.:
        raise ValueError("Finite references and nonnegative time/rates required")
    lower = max(float(limits[0]), reference - relief_rate * dt)
    upper = min(float(limits[1]), reference + close_rate * dt)
    if constrain_added_bias:
        lower = max(lower, actual + initial_bias - JOINT_PARAMETERS["additional_bias_limit_rad"])
        upper = min(upper, actual + initial_bias + JOINT_PARAMETERS["additional_bias_limit_rad"])
    conflict = lower > upper + 1e-12
    command = reference if conflict else float(np.clip(reference + velocity * dt, lower, upper))
    return command, {"previous_reference_rad": reference, "actual_rad": actual,
        "captured_initial_command_minus_actual_rad": initial_bias,
        "additional_command_bias_rad": command - actual - initial_bias,
        "command_rad": command, "command_delta_rad": command - reference,
        "allowed_interval_rad": [lower, upper], "bounds_conflict_hold": bool(conflict),
        "additional_bias_bound_applied": constrain_added_bias}


class JointAdmittanceProbeHand(r11.BoundedTrackingProbeHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_mode = "SCALAR_ACQUISITION"
        self.joint_references, self.initial_command_bias = {}, {}
        self.joint_target = {"feasible": False, "reason": "NOT_EVALUATED"}
        self.previous_feasible_targets = None
        self.joint_velocities, self.joint_solves = {}, []
        self.joint_events = []
        self.last_joint_write_s = None
        self.target_timestamp_s = None
        self.joint_activation_s = None
        self.previous_tracking_sample = None

    def flexion_joints(self):
        output = {}
        for finger in ACTIVE:
            if self.profile.get("contact_closure", {}).get(finger) == 0.:
                continue
            suffixes = (("MCP_FE", self.config.thumb_mcp_flexion_ratio),
                ("IP", self.config.thumb_ip_flexion_ratio)) if finger == "thumb" else (
                ("MCP_FE", self.profile.get("nonthumb_mcp_ratio", 1.)),
                ("PIP", self.profile.get("nonthumb_pip_ratio", 1.4)),
                ("DIP", self.profile.get("nonthumb_dip_ratio", .8)))
            for suffix, ratio in suffixes:
                output[f"left_{finger}_{suffix}"] = (finger, float(ratio))
        return output

    def _actual(self, data, name):
        joint = int(self.model.actuator_trnid[self.actuators[name], 0])
        return float(data.qpos[self.model.jnt_qposadr[joint]])

    def _mode(self, mode, data, reason):
        if self.joint_mode != mode:
            self.joint_events.append({"timestamp_s": float(data.time), "event": "MODE_TRANSITION",
                "from_mode": self.joint_mode, "to_mode": mode, "reason": reason})
            self.joint_mode = mode

    def phase_begin(self, phase, data):
        changed = phase != self.phase
        super().phase_begin(phase, data)
        if not changed:
            return
        if phase == "CLOSE":
            self.previous_feasible_targets = None
            self.joint_target = {"feasible": False, "reason": "NEW_CLOSE_REQUIRES_FRESH_TARGET"}
            self.joint_velocities = {}
            self.joint_activation_s = None
            self.previous_tracking_sample = None
            self._mode("SCALAR_RECONCILING" if self.joint_references else "SCALAR_ACQUISITION",
                data, "EXPLICIT_NEW_CLOSE_RESET")
        elif phase in ("RELEASE", "VERIFY", "RECOVER"):
            self.joint_velocities = {}
            self.previous_feasible_targets = None
            self.joint_activation_s = None
            if self.joint_references:
                self._mode("RELEASING", data, "EXPLICIT_RELEASE_OR_RECOVERY_OPEN")

    def _activate(self, data):
        self.joint_references = {name: float(data.ctrl[self.actuators[name]]) for name in self.flexion_joints()}
        self.initial_command_bias = {name: value - self._actual(data, name)
            for name, value in self.joint_references.items()}
        self.joint_activation_s = float(data.time)
        self._mode("JOINT_CONTROL", data, "FIRST_FRESH_THREE_FINGER_TRACKING_TARGET")
        self.joint_events.append({"timestamp_s": float(data.time), "event": "ACTIVATION_CAPTURE",
            "captured_command_rad": self.joint_references.copy(),
            "captured_command_minus_actual_rad": self.initial_command_bias.copy(),
            "scope": "Existing loaded bias retained; only added bias is bounded by .02rad"})

    def regulate(self, readings, dt, carrying):
        super().regulate(readings, dt, carrying)
        data = self.reader.last_data
        self.joint_velocities, self.joint_solves = {}, []
        self.target_timestamp_s = float(data.time)
        if self.emergency_active or self.emergency_phase_hold:
            self.joint_target = {"feasible": False, "reason": "ORIGINAL_EMERGENCY_PRECEDENCE"}
            return
        if self.joint_mode in ("RELEASING", "OPEN_HOLD", "SCALAR_RECONCILING", "EMERGENCY_HOLD"):
            self.joint_target = {"feasible": False, "reason": "MODE_DISALLOWS_NOMINAL_ADMITTANCE"}
            return
        external = self.reader.read(data)
        pressures = {finger: float(value.normal) for finger, value in external.items()}
        pose = self.estimated.estimated_pose(data, carrying)
        association = r11.r9.r8.r4.associate_tactile_contacts(self.reader.raw_contacts, pose,
            self.estimated.surface, now_s=float(data.time), pose_timestamp_s=self.estimated.pose_timestamp_s,
            max_pose_age_s=JOINT_PARAMETERS["association_max_pose_age_s"],
            max_surface_distance_m=JOINT_PARAMETERS["association_max_distance_m"])
        target = {"feasible": False, "reason": "CURRENT_ASSOCIATION_" + association.get("reason", "FAILED")}
        if (association.get("accepted") and all(pressures[finger] >= .35 for finger in ACTIVE)
                and max(pressures.values()) <= 6. and pressures.get("palm", 0.) <= .05):
            patches = []
            for patch in association["patches"]:
                raw_index = int(patch["input_index"])
                raw = self.reader.raw_contacts[raw_index]
                if raw["finger"] != patch["finger"]:
                    raise RuntimeError("Association raw contact index no longer matches tactile channel")
                patches.append(dict(patch, hand_body_name=raw["hand_body_name"]))
            target = allocate_tracking_targets(patches, association["estimated_com_world_m"],
                self.estimated.mass_kg, pressures, self.previous_feasible_targets)
        self.joint_target = dict(target, timestamp_s=float(data.time), association=association,
            all_external_pressures_n=pressures)
        if not target.get("feasible"):
            if self.joint_activation_s is not None:
                self._mode("HOLD_INVALID", data, target["reason"])
            return
        sample = self.contact_sample(self.reader.raw_contacts)
        changed = (self.previous_tracking_sample is not None and not
            r11.r9.r8.contact_samples_match(sample, self.previous_tracking_sample))
        self.previous_tracking_sample = sample
        if self.joint_activation_s is not None and changed:
            self.joint_target.update(feasible=False, reason="CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE")
            self._mode("HOLD_INVALID", data, self.joint_target["reason"])
            return
        if self.joint_activation_s is None:
            self._activate(data)
        self.previous_feasible_targets = target["targets_n"].copy()
        self._mode("JOINT_CONTROL", data, "CURRENT_TARGET_SOLVED_WITH_FRESH_ASSOCIATION")
        groups = group_tracking_patches(target["patches"])
        names = self.flexion_joints()
        for finger in ACTIVE:
            finger_names = [name for name, (channel, _) in names.items() if channel == finger]
            finger_groups = [group for group in groups if group["finger"] == finger]
            rows = []
            for group in finger_groups:
                body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, group["hand_body_name"])
                if body < 0:
                    raise RuntimeError("Unknown observed hand body")
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jp, jr, np.asarray(group["position_world_m"]), body)
                # Positive compression moves the hand along the force normal into the object.
                rows.append(np.asarray(group["normal_on_object_world"]) @ jp[:,
                    [self.joint_dofs[name] for name in finger_names]])
            velocity, diagnostics = damped_joint_velocity(rows,
                [group["target_normal_n"] - group["measured_normal_n"] for group in finger_groups])
            self.joint_velocities.update(zip(finger_names, velocity.tolist(), strict=True))
            self.joint_solves.append(dict(diagnostics, finger=finger, joint_names=finger_names, groups=finger_groups))

    def angles(self):
        original = super().angles()
        if self.emergency_active or self.emergency_phase_hold:
            return original
        return {**original, **self.joint_references}

    def write(self, data):
        self.update_pressure_relief(data, self.reader.read(data))
        dt = 0. if self.last_joint_write_s is None else max(0., float(data.time) - self.last_joint_write_s)
        self.last_joint_write_s = float(data.time)
        if self.emergency_active or self.emergency_phase_hold:
            super().write(data)
            if self.joint_references:
                self.joint_references = {name: float(data.ctrl[self.actuators[name]]) for name in self.joint_references}
                self._mode("EMERGENCY_HOLD", data, "ORIGINAL_R6_SAFETY_REFERENCES_WRITTEN")
            self.joint_events.append({"timestamp_s": float(data.time), "event": "SAFETY_PRECEDENCE",
                "mode": self.joint_mode, "command_rad": self.joint_references.copy()})
            return
        if self.joint_mode == "EMERGENCY_HOLD" and self.phase in ("RELEASE", "VERIFY", "RECOVER"):
            self._mode("RELEASING", data, "ORIGINAL_SAFETY_CLEARED_DURING_RELEASE_OR_RECOVERY")
        nominal = super().angles()
        details = {}
        previous_references = self.joint_references.copy()
        fresh = (self.target_timestamp_s is not None and abs(data.time - self.target_timestamp_s) < 1e-9
            and self.joint_target.get("feasible"))
        for name, reference in list(self.joint_references.items()):
            _, ratio = self.flexion_joints()[name]
            closing, relief = self.config.close_rate * abs(ratio), self.config.relief_rate * abs(ratio)
            joint = int(self.model.actuator_trnid[self.actuators[name], 0])
            opening = self.joint_mode in ("RELEASING", "OPEN_HOLD")
            reconciling = self.joint_mode == "SCALAR_RECONCILING"
            active = self.joint_mode == "JOINT_CONTROL" and fresh
            if opening or reconciling:
                goal = 0. if opening else nominal[name]
                velocity = (goal - reference) / max(dt, 1e-12)
            elif active:
                velocity = self.joint_velocities.get(name, 0.)
            else:
                details[name] = {"command_rad": reference, "hold_reason": "NO_CURRENT_VALID_NOMINAL_UPDATE"}
                continue
            command, detail = bounded_reference_step(reference, self._actual(data, name),
                self.initial_command_bias.get(name, 0.), velocity, dt, self.model.jnt_range[joint],
                closing, relief, constrain_added_bias=active)
            self.joint_references[name] = command
            details[name] = detail
        bounded_predictions = []
        if self.joint_mode == "JOINT_CONTROL" and fresh and dt > 0.:
            for solve in self.joint_solves:
                names = solve["joint_names"]
                velocity = np.array([(self.joint_references[name] - previous_references[name]) / dt for name in names])
                predicted = np.asarray(solve["compression_jacobian_m_rad"]) @ velocity
                desired = np.asarray(solve["desired_normal_velocity_m_s"])
                incompatible = bool(np.any(predicted * desired < -1e-12))
                if incompatible:
                    for name in names:
                        self.joint_references[name] = previous_references[name]
                        details[name].update(command_rad=previous_references[name], command_delta_rad=0.,
                            additional_command_bias_rad=previous_references[name] - self._actual(data, name)
                                - self.initial_command_bias.get(name, 0.),
                            incompatible_bounded_direction_hold=True)
                    predicted[:] = 0.
                bounded_predictions.append({"finger": solve["finger"],
                    "predicted_bounded_normal_velocity_m_s": predicted.tolist(),
                    "desired_minus_predicted_velocity_m_s": (desired - predicted).tolist(),
                    "incompatible_bounded_direction_hold": incompatible})
        if self.joint_mode == "RELEASING" and all(abs(value) < 1e-10 for value in self.joint_references.values()):
            self._mode("OPEN_HOLD", data, "FLEXION_REFERENCES_REACHED_OPEN")
        if self.joint_mode == "SCALAR_RECONCILING" and all(abs(value - nominal[name]) < 1e-10
                for name, value in self.joint_references.items()):
            self.joint_references.clear()
            self.initial_command_bias.clear()
            self._mode("SCALAR_ACQUISITION", data, "CONTINUOUS_RECONCILIATION_FINISHED")
        super().write(data)
        applied = {name: float(data.ctrl[self.actuators[name]]) for name in self.joint_references}
        safety_override = bool(self.emergency_active or self.emergency_phase_hold)
        candidate = self.joint_references.copy()
        if safety_override:
            self.joint_references = applied.copy()
            self._mode("EMERGENCY_HOLD", data, "ORIGINAL_SAFETY_OVERRIDE_AT_FINAL_WRITE")
        self.joint_events.append({"timestamp_s": float(data.time), "event": "NOMINAL_REFERENCE_WRITE",
            "phase": self.phase, "mode": self.joint_mode, "dt_s": dt,
            "current_target": self.joint_target, "target_age_s": None if self.target_timestamp_s is None
                else float(data.time - self.target_timestamp_s),
            "joint_solves": self.joint_solves, "joint_reference_steps": details,
            "bounded_motion_predictions": bounded_predictions,
            "candidate_command_rad": candidate, "applied_command_rad": applied,
            "safety_override_at_final_write": safety_override,
            "command_rad": self.joint_references.copy(),
            "scope": "Pressure-to-motion gain is a bounded heuristic, not measured contact stiffness"})


def controller_for(*args, **kwargs):
    base = r11.r9.r8.controller_for
    namespace = dict(base.__globals__)
    namespace["LoadTransferProbeHand"] = JointAdmittanceProbeHand
    factory = FunctionType(base.__code__, namespace, base.__name__, base.__defaults__, base.__closure__)
    original, source, hands, estimated = factory(*args, **kwargs)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    loop = next(node for node in move.body if isinstance(node, ast.For))
    obsolete = [node for node in loop.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25"]
    if len(obsolete) != 1:
        raise RuntimeError("Expected exactly one frozen obsolete CLOSE readiness exit")
    loop.body.remove(obsolete[0])
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<r13_individual_joint_admittance>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r11.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_joint_admittance_r13.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r11.r9.r8.r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r11.r9.r8._runner)))
    replacements = {
        "public_ycb_load_transfer_probe_development_r8": "public_ycb_joint_admittance_development_r13",
        "load_transfer_probe_r8": "joint_admittance_r13",
        "public_ycb_load_transfer_probe_r8": "public_ycb_joint_admittance_r13",
        "<public_ycb_load_transfer_probe_runner_r8>": "<public_ycb_joint_admittance_runner_r13>",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = replacements.get(node.value, node.value)
            if node.value.startswith("\nprotocol['latched_pressure_escape']"):
                node.value += "\nprotocol['joint_admittance'] = dict(_joint_parameters)\nprotocol['scalar_acquisition_tracking'] = dict(_tracking_parameters)\nprotocol['tracking_regulator_sha256'] = _tracking_regulator_sha256\nprotocol['tracking_regulator_source'] = _tracking_regulator_source\nprotocol['close_termination'] = 'Only original R8 eligibility exit or unchanged8s deadline; obsolete ready exit removed.'\nprotocol['force_control'] = 'Original scalar acquisition until first fresh three-finger target LP; then independent bounded joint admittance with carried command and captured loaded bias. Invalid targets hold, explicit retry reconciles to scalar acquisition, release opens, original R6 emergency references override at write time. Static targets and hypothetical1g compatibility are not measured success. R11 original3s/5mm probe motion and all strict gates retained.'\n"
    artifact_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, field, filename)")
    for values in (("probe_refresh", "probe_refresh_events", "probe_refresh.json.gz"),
            ("joint_admittance", "joint_events", "joint_admittance.json.gz")):
        artifact_loop.iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in values], ctx=ast.Load()))
    update = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "namespace.update")
    for name in ("_joint_parameters", "_tracking_parameters", "_tracking_regulator_sha256", "_tracking_regulator_source"):
        update.keywords.append(ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load())))
    namespace = dict(r11.r9.r8._runner.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PROBE_PARAMETERS=PROBE_PARAMETERS,
        _joint_parameters=JOINT_PARAMETERS, _tracking_parameters=r11.TRACKING_PARAMETERS,
        _tracking_regulator_sha256=r11.TRACKING_REGULATOR_SHA256,
        _tracking_regulator_source=r11.TRACKING_REGULATOR_SOURCE)
    exec(compile(ast.fix_missing_locations(tree), "<r13_isolated_runner_factory>", "exec"), namespace)
    return namespace["_runner"]()


run = _runner()


def main():
    original = r11.r9.r8.main
    namespace = dict(original.__globals__)
    namespace.update(run=run, __doc__=__doc__)
    return FunctionType(original.__code__, namespace, "main", original.__defaults__, original.__closure__)()


if __name__ == "__main__":
    raise SystemExit(main())
