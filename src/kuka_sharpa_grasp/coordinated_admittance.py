"""Jointly solve arm/finger position-reference velocities at one fixed state.

The Jacobian predicts normal compression velocity, not contact force. Position
servo torque bounds use ``kp * (command - actual_q) - kd * actual_qvel`` at the
supplied state. They are neither a future-state nor a dynamic force guarantee.
No simulator access, actuator writes, clipping, or recovery policy lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ReferenceVelocityBox:
    accepted: bool
    reason: str
    lower_rad_s: np.ndarray | None = None
    upper_rad_s: np.ndarray | None = None
    command_lower_rad: np.ndarray | None = None
    command_upper_rad: np.ndarray | None = None
    component_lower_rad: dict = field(default_factory=dict)
    component_upper_rad: dict = field(default_factory=dict)
    conflicting_joint_indices: tuple = ()
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatedAdmittanceResult:
    accepted: bool
    reason: str
    solver_converged: bool = False
    command_velocity_rad_s: np.ndarray | None = None
    command_rad: np.ndarray | None = None
    predicted_normal_velocity_m_s: np.ndarray | None = None
    contact_residual_m_s: np.ndarray | None = None
    task_residual_rad_s: np.ndarray | None = None
    predicted_motor_torque_nm: np.ndarray | None = None
    bounds: ReferenceVelocityBox | None = None
    active_constraints: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "fiu" or any(isinstance(item, (bool, np.bool_))
            for item in np.asarray(value, dtype=object).flat)):
        raise ValueError(name + " must contain real numeric values, not booleans")
    result = raw.astype(float, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(name + " has the wrong shape")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _scalar(value, name, *, positive=False):
    result = _array(value, name, ())
    number = float(result)
    if number < 0. or (positive and number == 0.):
        raise ValueError(name + " must be nonnegative" + (" and nonzero" if positive else ""))
    return number


def position_servo_velocity_box(*, actual_position_rad, actual_velocity_rad_s,
        previous_command_rad, timestep_s, joint_lower_rad, joint_upper_rad,
        command_rate_lower_rad_s, command_rate_upper_rad_s, deflection_lower_rad,
        deflection_upper_rad, servo_kp_nm_rad, servo_kd_nm_s_rad, torque_lower_nm,
        torque_upper_nm):
    """Intersect reference limits before converting to command-velocity limits.

    Deflection is the declared absolute ``command - actual_q`` interval, not a
    silently recaptured loaded bias. Joint limits must already intersect any
    actuator control limits. All selected joints must be direct position servos
    with positive kp and nonnegative kd; the runtime adapter verifies binding.
    Empty intersections are rejected even when the discrepancy is tiny.
    """
    try:
        actual = _array(actual_position_rad, "actual_position_rad")
        if actual.ndim != 1 or not len(actual):
            raise ValueError("A nonempty joint vector is required")
        shape = actual.shape
        velocity = _array(actual_velocity_rad_s, "actual_velocity_rad_s", shape)
        previous = _array(previous_command_rad, "previous_command_rad", shape)
        dt = _scalar(timestep_s, "timestep_s", positive=True)
        values = {name: _array(value, name, shape) for name, value in (
            ("joint_lower", joint_lower_rad), ("joint_upper", joint_upper_rad),
            ("rate_lower", command_rate_lower_rad_s), ("rate_upper", command_rate_upper_rad_s),
            ("deflection_lower", deflection_lower_rad), ("deflection_upper", deflection_upper_rad),
            ("kp", servo_kp_nm_rad), ("kd", servo_kd_nm_s_rad),
            ("torque_lower", torque_lower_nm), ("torque_upper", torque_upper_nm))}
        if np.any(values["kp"] <= 0.) or np.any(values["kd"] < 0.):
            raise ValueError("Position servos require kp > 0 and kd >= 0")
        for name in ("joint", "rate", "deflection", "torque"):
            if np.any(values[name + "_lower"] > values[name + "_upper"]):
                raise ValueError(name + " lower limit exceeds upper limit")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            lower = {
                "joint_range": values["joint_lower"],
                "command_rate": previous + dt * values["rate_lower"],
                "deflection": actual + values["deflection_lower"],
                "predicted_servo_torque": actual + (values["torque_lower"]
                    + values["kd"] * velocity) / values["kp"],
            }
            upper = {
                "joint_range": values["joint_upper"],
                "command_rate": previous + dt * values["rate_upper"],
                "deflection": actual + values["deflection_upper"],
                "predicted_servo_torque": actual + (values["torque_upper"]
                    + values["kd"] * velocity) / values["kp"],
            }
            command_lower = np.maximum.reduce(list(lower.values()))
            command_upper = np.minimum.reduce(list(upper.values()))
            low, high = (command_lower - previous) / dt, (command_upper - previous) / dt
        conflicts = tuple(int(index) for index in np.flatnonzero(command_lower > command_upper))
        return ReferenceVelocityBox(not conflicts, "REFERENCE_BOX_FEASIBLE" if not conflicts
            else "EMPTY_REFERENCE_BOX", low, high, command_lower, command_upper, lower, upper, conflicts,
            {"torque_scope": "Fixed supplied q/qvel and unsaturated position-servo prediction only",
             "dynamic_force_guarantee": False, "clipping_used": False})
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        return ReferenceVelocityBox(False, "INVALID_INPUT", diagnostics={"error": str(error)})


def solve_coordinated_admittance(compression_jacobian_m_rad, target_normal_velocity_m_s, *,
        actual_position_rad, actual_velocity_rad_s, previous_command_rad, timestep_s,
        joint_lower_rad, joint_upper_rad, command_rate_lower_rad_s,
        command_rate_upper_rad_s, deflection_lower_rad, deflection_upper_rad,
        servo_kp_nm_rad, servo_kd_nm_s_rad, torque_lower_nm, torque_upper_nm,
        task_reference_velocity_rad_s=None, contact_weights=None, task_weights=None,
        maximum_contact_residual_m_s=1e-4, bound_tolerance_rad_s=1e-9,
        torque_tolerance_nm=1e-9, solver_tolerance=1e-12, max_iterations=200):
    """Minimize sum(w_contact*(J*v-target)^2) + sum(w_task*(v-task)^2).

    Cost weights are nonnegative; zero removes a row from the objective. Every
    supplied contact row still participates in the explicit maximum-residual
    acceptance test. Task references are soft costs. Rejected numerical
    candidates remain diagnostic only and must not be written by the caller.
    """
    box = position_servo_velocity_box(actual_position_rad=actual_position_rad,
        actual_velocity_rad_s=actual_velocity_rad_s, previous_command_rad=previous_command_rad,
        timestep_s=timestep_s, joint_lower_rad=joint_lower_rad, joint_upper_rad=joint_upper_rad,
        command_rate_lower_rad_s=command_rate_lower_rad_s, command_rate_upper_rad_s=command_rate_upper_rad_s,
        deflection_lower_rad=deflection_lower_rad, deflection_upper_rad=deflection_upper_rad,
        servo_kp_nm_rad=servo_kp_nm_rad, servo_kd_nm_s_rad=servo_kd_nm_s_rad,
        torque_lower_nm=torque_lower_nm, torque_upper_nm=torque_upper_nm)
    if not box.accepted:
        return CoordinatedAdmittanceResult(False, box.reason, bounds=box)
    try:
        n = len(box.lower_rad_s)
        matrix = _array(compression_jacobian_m_rad, "compression_jacobian_m_rad")
        if matrix.ndim != 2 or matrix.shape[1] != n or not matrix.shape[0]:
            raise ValueError("Jacobian must have nonempty patch rows and one column per controlled joint")
        p = matrix.shape[0]
        target = _array(target_normal_velocity_m_s, "target_normal_velocity_m_s", (p,))
        task = np.zeros(n) if task_reference_velocity_rad_s is None else _array(
            task_reference_velocity_rad_s, "task_reference_velocity_rad_s", (n,))
        contact_w = np.ones(p) if contact_weights is None else _array(contact_weights, "contact_weights", (p,))
        task_w = np.zeros(n) if task_weights is None else _array(task_weights, "task_weights", (n,))
        if np.any(contact_w < 0.) or np.any(task_w < 0.) or not np.any(np.r_[contact_w, task_w] > 0.):
            raise ValueError("Cost weights must be nonnegative with at least one positive weight")
        residual_limit = _scalar(maximum_contact_residual_m_s, "maximum_contact_residual_m_s")
        bound_tol = _scalar(bound_tolerance_rad_s, "bound_tolerance_rad_s")
        torque_tol = _scalar(torque_tolerance_nm, "torque_tolerance_nm")
        solver_tol = _scalar(solver_tolerance, "solver_tolerance", positive=True)
        if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            objective = np.vstack([np.sqrt(contact_w)[:, None] * matrix, np.diag(np.sqrt(task_w))])
            right = np.r_[np.sqrt(contact_w) * target, np.sqrt(task_w) * task]
        # lsq_linear requires strict intervals; exact fixed references are eliminated.
        fixed = box.lower_rad_s == box.upper_rad_s
        free = ~fixed
        candidate = box.lower_rad_s.copy()
        solved = None
        if np.any(free):
            with np.errstate(over="raise", invalid="raise"):
                right_free = right - objective[:, fixed] @ candidate[fixed]
            solved = lsq_linear(objective[:, free], right_free,
                bounds=(box.lower_rad_s[free], box.upper_rad_s[free]), method="trf",
                lsq_solver="exact", tol=solver_tol, max_iter=int(max_iterations))
            candidate[free] = solved.x
        converged = bool(solved is None or solved.success)
        if not np.isfinite(candidate).all():
            return CoordinatedAdmittanceResult(False, "NONFINITE_SOLVER_CANDIDATE", bounds=box,
                diagnostics={"solver_converged": converged})
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            previous = _array(previous_command_rad, "previous_command_rad", (n,))
            command = previous + float(timestep_s) * candidate
            predicted = matrix @ candidate
            contact_residual, task_residual = predicted - target, candidate - task
            torque = np.asarray(servo_kp_nm_rad) * (command - np.asarray(actual_position_rad)) \
                - np.asarray(servo_kd_nm_s_rad) * np.asarray(actual_velocity_rad_s)
            objective_residual = objective @ candidate - right
            weighted_residual = float(objective_residual @ objective_residual)
            violation = float(max(0., np.max(box.lower_rad_s - candidate), np.max(candidate - box.upper_rad_s)))
            contact_error = float(np.max(np.abs(contact_residual)))
        if not all(np.isfinite(value).all() for value in (command, predicted, torque, objective_residual)):
            raise ValueError("Nonfinite reconstructed solver output")
        # Check reconstructed commands as well as velocities; no post-solve clip.
        position_tol = float(timestep_s) * bound_tol
        command_violation = float(max(0., np.max(box.command_lower_rad - command),
            np.max(command - box.command_upper_rad)))
        torque_violation = float(max(0., np.max(np.asarray(torque_lower_nm) - torque),
            np.max(torque - np.asarray(torque_upper_nm))))
        bounded = (violation <= bound_tol and command_violation <= position_tol
            and torque_violation <= torque_tol)
        accepted = converged and bounded and contact_error <= residual_limit
        reason = ("COORDINATED_REFERENCE_ACCEPTED" if accepted else "SOLVER_DID_NOT_CONVERGE" if not converged
            else "REFERENCE_BOUND_VIOLATION" if not bounded else "CONTACT_TRACKING_RESIDUAL")
        active = {name + "_" + side: tuple(int(i) for i in np.flatnonzero(
            np.abs(command - values) <= position_tol)) for side, components in (
            ("lower", box.component_lower_rad), ("upper", box.component_upper_rad))
            for name, values in components.items()}
        active["fixed_velocity"] = tuple(int(i) for i in np.flatnonzero(fixed))
        diagnostics = {"solver": "scipy.optimize.lsq_linear:trf:exact",
            "solver_status": 3 if solved is None else int(solved.status),
            "solver_message": "All references fixed by the box" if solved is None else str(solved.message),
            "iterations": 0 if solved is None else int(solved.nit),
            "optimality": 0. if solved is None else float(solved.optimality),
            "weighted_squared_residual": weighted_residual,
            "maximum_contact_residual_m_s": contact_error,
            "contact_residual_limit_m_s": residual_limit,
            "maximum_velocity_bound_violation_rad_s": violation,
            "maximum_command_bound_violation_rad": command_violation,
            "maximum_torque_bound_violation_nm": torque_violation,
            "torque_tolerance_nm": torque_tol,
            "clipping_used": False, "dynamic_force_guarantee": False,
            "torque_scope": "Predicted unsaturated motor torque at fixed supplied q/qvel, not a future-state bound"}
        return CoordinatedAdmittanceResult(accepted, reason, converged, candidate, command,
            predicted, contact_residual, task_residual, torque, box, active, diagnostics)
    except (TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
        return CoordinatedAdmittanceResult(False, "INVALID_INPUT_OR_NUMERICAL_FAILURE", bounds=box,
            diagnostics={"error": str(error), "dynamic_force_guarantee": False})
