"""Prospective static targets with original R13 groups and measured patch ratios."""

from __future__ import annotations

import hashlib
from types import FunctionType

import numpy as np

import develop_public_ycb_joint_admittance_r13 as r13
import shared_normal_tracking_targets as shared


FORMAT = "group_proportional_shared_normal_static_target_v1"


def _array_hash(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _group_rows(prepared):
    patches = [dict(patch, raw_input_index=int(patch["input_index"]), target_normal_n=float(pressure))
        for patch, _, _, _, pressure in prepared]
    groups = r13.group_tracking_patches(patches)
    offsets = {patch["input_index"]: index for index, patch in enumerate(patches)}
    columns = 2 * len(patches) * shared.CONE_RAYS + 3 + len(patches)
    rows, descriptions = [], []
    for group in groups:
        members = group["raw_input_indices"]
        measured = [float(patches[offsets[index]]["measured_pressure_n"]) for index in members]
        total = sum(measured)
        if total <= 0. or not np.isfinite(total):
            raise ValueError("A retained original group requires positive finite measured load")
        ratios = [value / total for value in measured]
        descriptions.append({"finger": group["finger"], "hand_body_name": group["hand_body_name"],
            "raw_input_indices": members, "measured_normals_n": measured,
            "measured_group_normal_n": total, "measured_proportions": ratios})
        # The final member equation follows from summing the group normals.
        for member, ratio in zip(members[:-1], ratios[:-1]):
            row = np.zeros(columns)
            for other in members:
                offset = offsets[other] * shared.CONE_RAYS
                row[offset:offset + shared.CONE_RAYS] -= ratio
            offset = offsets[member] * shared.CONE_RAYS
            row[offset:offset + shared.CONE_RAYS] += 1.
            rows.append(row)
    return np.asarray(rows).reshape((-1, columns)), descriptions


def _append_only_group_equalities(objective, kwargs, group_rows, count):
    n, slack = count * shared.CONE_RAYS, 3 + count
    width = 2 * n + slack
    if set(kwargs) != {"A_ub", "b_ub", "A_eq", "b_eq", "bounds", "method", "options"}:
        raise ValueError("Unexpected shared-normal main LP call contract")
    expected = np.r_[np.full(n, shared.NORMAL_REGULARIZER),
        np.full(n, shared.SECONDARY_RAY_WEIGHT), np.ones(slack)]
    if not np.array_equal(np.asarray(objective), expected):
        raise ValueError("Shared-normal main objective changed")
    equalities, right = np.asarray(kwargs["A_eq"]), np.asarray(kwargs["b_eq"])
    if (equalities.shape != (12 + count, width) or right.shape != (12 + count,)
            or np.asarray(kwargs["A_ub"]).shape != (15 + 2 * count, width)
            or np.asarray(kwargs["b_ub"]).shape != (15 + 2 * count,)
            or group_rows.ndim != 2 or group_rows.shape[1] != width
            or len(group_rows) > max(0, count - 1)):
        raise ValueError("Unexpected shared-normal main LP dimensions")
    if (kwargs["bounds"] != (0., None) or kwargs["method"] != "highs"
            or kwargs["options"] != {"primal_feasibility_tolerance": 1e-8,
                "dual_feasibility_tolerance": 1e-8, "time_limit": .1}
            or not all(np.isfinite(value).all() for value in (equalities, right, group_rows))):
        raise ValueError("Shared-normal main solver settings or finite coefficients changed")
    return dict(kwargs, A_eq=np.vstack([equalities, group_rows]), b_eq=np.r_[right, np.zeros(len(group_rows))])


def allocate_group_proportional_targets(patches, com, mass, pressures):
    """Clone only the pure main LP call; always use previous=None in this study."""
    prepared, _, _, _, _, _ = shared._observations(patches, com, mass, pressures, None)
    group_rows, groups = _group_rows(prepared)
    count, calls, witness = len(prepared), 0, {}
    original_linprog = shared.linprog
    def constrained_linprog(objective, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1 or args:
            raise ValueError("Only one declared shared-normal main LP call may be wrapped")
        try:
            augmented = _append_only_group_equalities(objective, kwargs, group_rows, count)
        except ValueError as error:
            witness["contract_error"] = str(error)
            raise
        witness.update(original_equality_rows=len(kwargs["b_eq"]), added_equality_rows=len(group_rows),
            original_equality_sha256=_array_hash(kwargs["A_eq"]),
            augmented_equality_sha256=_array_hash(augmented["A_eq"]),
            unchanged_objective_sha256=_array_hash(objective),
            unchanged_inequality_sha256=_array_hash(kwargs["A_ub"]),
            unchanged_inequality_right_sha256=_array_hash(kwargs["b_ub"]))
        solved = original_linprog(objective, **augmented)
        if solved.success and solved.x is not None and np.isfinite(solved.x).all():
            n = count * shared.CONE_RAYS
            high = np.asarray(solved.x[:n]).reshape((count, shared.CONE_RAYS)).sum(axis=1)
            low = np.asarray(solved.x[n:2 * n]).reshape((count, shared.CONE_RAYS)).sum(axis=1)
            witness.update(high_patch_normals_n=high.tolist(), low_patch_normals_n=low.tolist(),
                maximum_added_equality_residual_n=float(np.max(abs(group_rows @ solved.x))) if len(group_rows) else 0.,
                maximum_low_gravity_ratio_residual_n=float(np.max(abs(group_rows[:, :n] @ solved.x[n:2 * n]))) if len(group_rows) else 0.)
        return solved
    isolated = FunctionType(shared.allocate_tracking_targets.__code__,
        dict(shared.allocate_tracking_targets.__globals__, linprog=constrained_linprog),
        "prospective_group_proportional_targets", shared.allocate_tracking_targets.__defaults__,
        shared.allocate_tracking_targets.__closure__)
    result = isolated(patches, com, mass, pressures, None)
    if (shared.linprog is not original_linprog or calls > 1 or witness.get("contract_error")
            or (result["feasible"] and calls != 1)):
        raise ValueError("Isolated main LP call count/global invariant failed")
    result["inherited_target_format"], result["format"] = result["format"], FORMAT
    result["prospective_group_constraints"] = {"main_lp_calls": calls, "previous": None,
        "groups": groups, **witness, "constraint": "p_i = measured_i/sum_group_measured * sum_group_target",
        "only_main_lp_equalities_appended": True, "original_global_mutation": False,
        "low_gravity_ratios_implied_by_original_exact_shared_patch_normal_equalities": True,
        "scope": "Original R13 membership with retained >=0.05N patches only. A fixed measured proportion is a prospective surrogate for group control, not a proven contact-force response model."}
    if result["feasible"] and max(witness["maximum_added_equality_residual_n"], witness["maximum_low_gravity_ratio_residual_n"]) > 1e-6:
        result.update(feasible=False, reason="GROUP_PROPORTION_EQUALITY_RESIDUAL")
    return result
