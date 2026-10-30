#!/usr/bin/env python3
"""Static grasp-load allocation from estimated geometry and simulated tactile patches.

The caller supplies tactile observations and an estimated center of mass. This
module has no simulator/model argument and cannot read a live target state.
Feasibility is a gravity-support check, not a dynamic-retention certificate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _tangents(normal):
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(normal)))] = 1.
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def allocate_contact_wrench(
    patches,
    estimated_com_world_m,
    mass_kg,
    measured_pressures_n=None,
    *,
    friction_coefficient=.8,
    pressure_limit_n=6.,
    gravity_margin=1.5,
    friction_safety_factor=.85,
    cone_rays=16,
    gravity_m_s2=9.81,
):
    """Return a minimal normal-load allocation and two measured-load checks.

    ``normal_on_object_world`` points *into* the estimated object. Sixteen rays
    generate an inscribed circular Coulomb cone; coefficients are normal loads
    because each ray has unit normal projection. Finger sums are capped at 6 N
    or less; palm contact is excluded. Measured equilibrium fixes every patch's
    measured normal load, whereas measured capacity only upper-bounds it. The
    latter must not be mistaken for equilibrium of the actual squeeze.
    """
    empty = dict.fromkeys(CHANNELS, 0.)
    result = {
        "feasible": False,
        "reason": "INVALID_INPUT",
        "target_pressures_n": empty.copy(),
        "measured_feasible": False,
        "measured_capacity_feasible": False,
        "claim_scope": "Estimated static gravity and torque support only; not dynamic stability or force closure",
        "runtime_target_state_used": False,
    }
    try:
        com = _vector(estimated_com_world_m, "estimated_com_world_m")
        mass, limit = float(mass_kg), float(pressure_limit_n)
        margin, safety = float(gravity_margin), float(friction_safety_factor)
        friction, gravity = float(friction_coefficient), float(gravity_m_s2)
        scalars = np.array([mass, limit, margin, safety, friction, gravity])
        if not np.isfinite(scalars).all() or mass <= 0. or not 0. < limit <= 6.:
            raise ValueError("Positive finite mass and a pressure limit in (0, 6] N are required")
        if margin < 1. or not 0. < safety <= 1. or friction < 0. or gravity <= 0.:
            raise ValueError("Invalid gravity margin, friction coefficient, or friction safety factor")
        if isinstance(cone_rays, bool) or int(cone_rays) != cone_rays or not 8 <= cone_rays <= 64:
            raise ValueError("cone_rays must be an integer from eight through 64")
        ray_count = int(cone_rays)
        prepared = []
        ignored_palm_pressure = 0.
        for index, patch in enumerate(patches):
            finger = patch["finger"]
            if finger not in CHANNELS:
                raise ValueError(f"Unknown tactile channel: {finger}")
            position = _vector(patch["position_world_m"], "position_world_m")
            normal = _vector(patch["normal_on_object_world"], "normal_on_object_world")
            length = float(np.linalg.norm(normal))
            if not .5 <= length <= 1.5:
                raise ValueError("Contact normal must be a usable unit direction")
            pressure = float(patch.get("measured_pressure_n", patch.get("pressure_n", 0.)))
            patch_friction = float(patch.get("friction_coefficient", friction))
            if not np.isfinite([pressure, patch_friction]).all() or pressure < 0. or patch_friction < 0.:
                raise ValueError("Pressure and friction must be finite and nonnegative")
            if finger == "palm":
                ignored_palm_pressure += pressure
                continue
            prepared.append({"index": index, "finger": finger, "position": position,
                "normal": normal / length, "pressure": pressure,
                "mu": min(friction, patch_friction) * safety})
        if len({patch["finger"] for patch in prepared}) < 2:
            result["reason"] = "INSUFFICIENT_DISTINCT_FINGER_PATCHES"
            return result
        measured = {finger: sum(patch["pressure"] for patch in prepared if patch["finger"] == finger)
            for finger in CHANNELS}
        if measured_pressures_n is not None:
            for finger, value in measured_pressures_n.items():
                if finger not in CHANNELS or not np.isfinite(float(value)) or float(value) < 0.:
                    raise ValueError("Invalid all-contact channel pressure")
            for finger in CHANNELS:
                observed = float(measured_pressures_n.get(finger, 0.))
                if observed + 1e-5 < measured[finger]:
                    raise ValueError("Associated patch pressures exceed the all-contact channel reading")
            measured = {finger: float(measured_pressures_n.get(finger, 0.)) for finger in CHANNELS}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["details"] = str(error)
        return result

    count = len(prepared)
    normal_rows = np.zeros((count, count * ray_count))
    finger_rows = np.zeros((len(CHANNELS), count * ray_count))
    directions = []
    arms = []
    for index, patch in enumerate(prepared):
        first, second = _tangents(patch["normal"])
        theta = 2. * math.pi * np.arange(ray_count) / ray_count
        rays = patch["normal"] + patch["mu"] * (
            np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second)
        directions.extend(rays)
        arms.extend([patch["position"] - com] * ray_count)
        section = slice(index * ray_count, (index + 1) * ray_count)
        normal_rows[index, section] = 1.
        finger_rows[CHANNELS.index(patch["finger"]), section] = 1.
    directions, arms = np.asarray(directions), np.asarray(arms)
    physical_wrench = np.vstack([directions.T, np.cross(arms, directions).T])
    # Meter-scale torque rows otherwise receive weaker numeric tolerances.
    scaled_wrench = physical_wrench.copy()
    scaled_wrench[3:] /= .05
    target_wrench = np.array([0., 0., mass * gravity * margin, 0., 0., 0.])
    limits = np.array([limit if finger != "palm" else 0. for finger in CHANNELS])
    objective = np.ones(count * ray_count)
    options = {"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8,
        "time_limit": .1}

    def solve(a_ub, b_ub, *, fixed_patch_pressures=False):
        a_eq, b_eq = scaled_wrench, target_wrench
        if fixed_patch_pressures:
            a_eq = np.vstack([a_eq, normal_rows])
            b_eq = np.r_[b_eq, [patch["pressure"] for patch in prepared]]
        solved = linprog(objective, A_ub=a_ub, b_ub=b_ub,
            A_eq=a_eq, b_eq=b_eq, bounds=(0., None), method="highs", options=options)
        if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
            return solved, False, None
        residual = physical_wrench @ solved.x - target_wrench
        valid = bool(np.max(np.abs(residual[:3])) <= 1e-5
            and np.max(np.abs(residual[3:])) <= 1e-6)
        return solved, valid, residual

    target, feasible, residual = solve(finger_rows, limits)
    patch_measured = np.array([patch["pressure"] for patch in prepared])
    capacity, capacity_feasible, _ = solve(normal_rows, patch_measured)
    actual, measured_feasible, actual_residual = solve(None, None, fixed_patch_pressures=True)
    if max(ignored_palm_pressure, measured["palm"]) > 1e-5:
        measured_feasible = False
    target_loads = finger_rows @ target.x if feasible else np.zeros(len(CHANNELS))
    result.update({
        "feasible": feasible,
        "reason": "ALLOCATED" if feasible else "ESTIMATED_WRENCH_INFEASIBLE",
        "target_pressures_n": dict(zip(CHANNELS, target_loads.tolist())),
        "measured_feasible": measured_feasible,
        "measured_capacity_feasible": capacity_feasible,
        "palm_pressure_excluded_from_allocation_n": max(ignored_palm_pressure, measured["palm"]),
        "measured_channel_pressures_n": measured,
        "measured_within_target_limits": all(measured[finger] <= limits[i] + 1e-6
            for i, finger in enumerate(CHANNELS)),
        "estimated_com_world_m": com.tolist(),
        "mass_prior_kg": mass,
        "gravity_margin": margin,
        "required_vertical_load_n": float(target_wrench[2]),
        "normal_pressure_limit_n": limit,
        "friction_safety_factor": safety,
        "friction_cone_rays": ray_count,
        "solver_time_limit_per_lp_s": .1,
        "friction_cone_scope": "Inscribed circular Coulomb cone without torsional contact moments",
        "minimum_total_normal_load_n": float(target.fun) if feasible else None,
        "target_wrench_residual": residual.tolist() if residual is not None else None,
        "measured_wrench_residual": actual_residual.tolist() if actual_residual is not None else None,
        "solver_status": int(target.status),
        "solver_message": str(target.message),
        "measured_solver_status": int(actual.status),
        "measured_capacity_solver_status": int(capacity.status),
        "patch_allocations": [{
            "input_index": patch["index"], "finger": patch["finger"],
            "position_world_m": patch["position"].tolist(),
            "normal_on_object_world": patch["normal"].tolist(),
            "measured_pressure_n": patch["pressure"],
            "effective_friction_coefficient": patch["mu"],
            "target_normal_pressure_n": float(normal_rows[index] @ target.x) if feasible else None,
            "target_force_on_object_world_n": (
                directions[index * ray_count:(index + 1) * ray_count].T
                @ target.x[index * ray_count:(index + 1) * ray_count]).tolist() if feasible else None,
        } for index, patch in enumerate(prepared)],
    })
    return result
