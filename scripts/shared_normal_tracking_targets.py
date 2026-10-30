"""Pure static tracking targets with shared patch normals at 1.5g and 1g."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

import tactile_wrench_allocator as allocator


ACTIVE = ("thumb", "middle", "ring")
PATCH_FLOOR_N = .05
ACTIVE_FLOOR_N = .75
PRESSURE_LIMIT_N = 6.
CONE_RAYS = 16
FRICTION = .8
FRICTION_SAFETY = .85
PREVIOUS_WEIGHT = .75
NORMAL_REGULARIZER = 1e-4
SECONDARY_RAY_WEIGHT = 1e-8
LP_TIME_LIMIT_S = .1


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite numeric scalar")
    number = float(value)
    if not np.isfinite(number) or number < 0. or (positive and number == 0.):
        raise ValueError(name + " must be finite and nonnegative (positive where required)")
    return number


def _observations(patches, com, mass, pressures, previous):
    center = allocator._vector(com, "estimated_com_world_m")
    mass = _scalar(mass, "mass", positive=True)
    if not isinstance(pressures, dict) or any(finger not in pressures for finger in ACTIVE):
        raise ValueError("All three active channel pressures are required")
    if any(finger not in allocator.CHANNELS for finger in pressures):
        raise ValueError("Unknown pressure channel")
    pressure = {finger: _scalar(pressures.get(finger, 0.), "pressure") for finger in allocator.CHANNELS}
    if previous is not None:
        if not isinstance(previous, dict) or any(finger not in previous for finger in ACTIVE):
            raise ValueError("Previous targets must contain all active channels")
        if any(finger not in allocator.CHANNELS for finger in previous):
            raise ValueError("Unknown previous-target channel")
        for value in previous.values():
            _scalar(value, "previous target")
    prepared, indices = [], []
    for patch in patches:
        index = patch["input_index"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("Raw patch indices must be nonnegative integers")
        if index in indices:
            raise ValueError("Duplicate raw patch index")
        indices.append(int(index))
        if patch["finger"] not in allocator.CHANNELS:
            raise ValueError("Unknown patch channel")
        point = allocator._vector(patch["position_world_m"], "position_world_m")
        normal = allocator._vector(patch["normal_on_object_world"], "normal_on_object_world")
        length = float(np.linalg.norm(normal))
        if not .5 <= length <= 1.5:
            raise ValueError("Contact normal must be a usable unit direction")
        observed = _scalar(patch["measured_pressure_n"], "patch pressure")
        mu = min(FRICTION, _scalar(patch.get("friction_coefficient", FRICTION), "friction")) * FRICTION_SAFETY
        if patch["finger"] in ACTIVE and observed >= PATCH_FLOOR_N:
            prepared.append((dict(patch), point, normal / length, mu, observed))
    reference = np.array([pressure[finger] if previous is None else
        PREVIOUS_WEIGHT * _scalar(previous[finger], "previous target")
        + (1. - PREVIOUS_WEIGHT) * pressure[finger] for finger in ACTIVE])
    if not np.isfinite(reference).all():
        raise ValueError("Reference arithmetic overflow")
    return prepared, indices, center, mass, pressure, reference


def allocate_tracking_targets(patches, com, mass, pressures, previous=None):
    """Keep the R13 argument/target contract; require original caller association gates.

    Neither simulator state nor a live object pose is accepted. The caller must
    still enforce association freshness, raw contact identity, measured-pressure
    entry gates, patch-change handling and every physical admission/task gate.
    """
    result = {"feasible": False, "reason": "INVALID_INPUT", "runtime_target_state_used": False,
        "format": "shared_normal_tracking_target_v1", "solver_time_limit_per_lp_s": LP_TIME_LIMIT_S,
        "claim_scope": "Hypothetical shared-normal static target, not measured equilibrium or dynamic success"}
    try:
        prepared, indices, center, mass, pressure, reference = _observations(
            patches, com, mass, pressures, previous)
        retained = [item[0] for item in prepared]
        if {patch["finger"] for patch in retained} != set(ACTIVE):
            return dict(result, reason="THREE_OBSERVED_ACTIVE_FINGERS_REQUIRED")
        geometry = allocator.allocate_contact_wrench(retained, center, mass,
            measured_pressures_n=pressure)
        if not geometry.get("feasible"):
            return dict(result, reason="ORIGINAL_GEOMETRY_LP_" + geometry.get("reason", "FAILED"))

        count, n, slack = len(prepared), len(prepared) * CONE_RAYS, 3 + len(prepared)
        fingers = np.zeros((len(allocator.CHANNELS), n))
        normals = np.zeros((count, n))
        directions, arms = [], []
        for index, (patch, point, normal, mu, _) in enumerate(prepared):
            first, second = allocator._tangents(normal)
            theta = 2. * math.pi * np.arange(CONE_RAYS) / CONE_RAYS
            directions.extend(normal + mu * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second))
            arms.extend([point - center] * CONE_RAYS)
            section = slice(index * CONE_RAYS, (index + 1) * CONE_RAYS)
            fingers[allocator.CHANNELS.index(patch["finger"]), section] = 1.
            normals[index, section] = 1.
        directions, arms = np.asarray(directions), np.asarray(arms)
        wrench = np.vstack([directions.T, np.cross(arms, directions).T])
        scaled = wrench.copy()
        scaled[3:] /= .05
        gravity = np.array([0., 0., mass * 9.81, 0., 0., 0.])
        zeros_wrench = np.zeros_like(scaled)
        equality = np.vstack([np.hstack([scaled, zeros_wrench]),
            np.hstack([zeros_wrench, scaled]), np.hstack([normals, -normals])])
        equality = np.hstack([equality, np.zeros((12 + count, slack))])
        right = np.r_[1.5 * gravity, gravity, np.zeros(count)]
        active = fingers[[allocator.CHANNELS.index(finger) for finger in ACTIVE]]
        channel_slack, patch_slack = np.zeros((3, slack)), np.zeros((count, slack))
        channel_slack[:, :3], patch_slack[:, 3:] = np.eye(3), np.eye(count)
        high_rows = np.vstack([fingers, -active, active, -active, normals, -normals])
        slack_rows = np.vstack([np.zeros((len(allocator.CHANNELS) + 3, slack)),
            -channel_slack, -channel_slack, -patch_slack, -patch_slack])
        inequality = np.hstack([high_rows, np.zeros_like(high_rows), slack_rows])
        measured = np.array([item[4] for item in prepared])
        upper = np.r_[[0. if finger == "palm" else PRESSURE_LIMIT_N for finger in allocator.CHANNELS],
            np.full(3, -ACTIVE_FLOOR_N), reference, -reference, measured, -measured]
        objective = np.r_[np.full(n, NORMAL_REGULARIZER), np.full(n, SECONDARY_RAY_WEIGHT), np.ones(slack)]
        if not all(np.isfinite(value).all() for value in (wrench, equality, right, inequality, upper, objective)):
            return dict(result, reason="NONFINITE_LP_COEFFICIENTS")
        solved = linprog(objective, A_ub=inequality, b_ub=upper, A_eq=equality,
            b_eq=right, bounds=(0., None), method="highs",
            options={"primal_feasibility_tolerance": 1e-8,
                "dual_feasibility_tolerance": 1e-8, "time_limit": LP_TIME_LIMIT_S})
        result.update(solver_status=int(solved.status), solver_message=str(solved.message))
        if not solved.success or solved.status != 0 or solved.x is None:
            return dict(result, reason="SHARED_NORMAL_TRACKING_LP_FAILED")
        values = np.asarray(solved.x, dtype=float)
        if values.shape != (2 * n + slack,) or not np.isfinite(values).all():
            return dict(result, reason="SHARED_NORMAL_TRACKING_LP_NONFINITE_OR_SHAPE")
        high, low = values[:n], values[n:2 * n]
        high_residual, low_residual = wrench @ high - 1.5 * gravity, wrench @ low - gravity
        patch_loads = normals @ high
        load_difference = normals @ high - normals @ low
        inequality_residual = inequality @ values - upper
        if not all(np.isfinite(value).all() for value in
                (high_residual, low_residual, patch_loads, load_difference, inequality_residual)):
            return dict(result, reason="SHARED_NORMAL_TRACKING_LP_NONFINITE_RESIDUAL")
        diagnostics = {"force_residual_max_n": float(max(np.max(abs(high_residual[:3])), np.max(abs(low_residual[:3])))),
            "torque_residual_max_nm": float(max(np.max(abs(high_residual[3:])), np.max(abs(low_residual[3:])))),
            "shared_patch_normal_residual_max_n": float(np.max(abs(load_difference))),
            "inequality_excess_max_n": float(max(0., np.max(inequality_residual))),
            "nonnegative_bound_violation_max_n": float(max(0., -np.min(values)))}
        result.update(numerical_checks=diagnostics, target_wrench_residual=high_residual.tolist(),
            target_wrench_residual_1g=low_residual.tolist())
        if (not np.isfinite(list(diagnostics.values())).all()
                or diagnostics["force_residual_max_n"] > 1e-5
                or diagnostics["torque_residual_max_nm"] > 1e-6
                or diagnostics["shared_patch_normal_residual_max_n"] > 1e-6
                or diagnostics["inequality_excess_max_n"] > 1e-6
                or diagnostics["nonnegative_bound_violation_max_n"] > 1e-8
                or np.any(patch_loads < 0.)):
            return dict(result, reason="SHARED_NORMAL_TRACKING_LP_RESIDUAL")

        proposed = [dict(patch, measured_pressure_n=float(load))
            for patch, load in zip(retained, patch_loads, strict=True)]
        checks = {str(margin): allocator.allocate_contact_wrench(proposed, center, mass,
            gravity_margin=margin) for margin in (1., 1.5)}
        result["independent_fixed_normal_checks"] = {margin: {key: check.get(key) for key in
            ("measured_feasible", "measured_solver_status", "measured_wrench_residual")}
            for margin, check in checks.items()}
        if not all(check.get("measured_feasible") is True for check in checks.values()):
            return dict(result, reason="SHARED_NORMAL_INDEPENDENT_FIXED_NORMAL_POSTCHECK_FAILED")
        loads = fingers @ high
        raw_indices = [int(patch["input_index"]) for patch in retained]
        return dict(result, feasible=True, reason="STATIC_SHARED_NORMAL_TRACKING_TARGET_ALLOCATED",
            targets_n=dict(zip(allocator.CHANNELS, loads.tolist(), strict=True)),
            reference_n=dict(zip(ACTIVE, reference.tolist(), strict=True)),
            effective_patch_mu=[item[3] for item in prepared],
            hypothetical_target_fixed_normal_1g_feasible=True,
            hypothetical_target_fixed_normal_1_5g_feasible=True,
            hypothetical_check_scope="Proposed exact shared normals only, not measured pressure evidence",
            excluded_tracking_raw_indices=[index for index in indices if index not in raw_indices],
            patches=[dict(patch, raw_input_index=raw, target_normal_n=float(load))
                for patch, raw, load in zip(retained, raw_indices, patch_loads, strict=True)],
            objective={"channel_reference_l1_n": float(np.sum(abs(active @ high - reference))),
                "patch_measured_l1_n": float(np.sum(abs(patch_loads - measured))),
                "primary_normal_regularizer": float(NORMAL_REGULARIZER * np.sum(high)),
                "secondary_normal_regularizer": float(SECONDARY_RAY_WEIGHT * np.sum(low))})
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        return dict(result, details=str(error))
