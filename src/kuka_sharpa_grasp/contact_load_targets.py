"""Static load targets from declared priors and one current tactile sample.

No simulator state is accepted. A feasible result is an estimated static
wrench witness at the declared gravity margin, never grasp or safety evidence.
Historical ``pressure`` fields contain normal loads in N, not pressure in Pa.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linprog


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def _number(value, name, *, minimum=0., positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (positive and result == 0.):
        raise ValueError(name + " is nonfinite or outside its allowed range")
    return result


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite three-vector")
    return result


@dataclass(frozen=True)
class ContactLoadTargetConfig:
    active_fingers: tuple[str, ...] = ("thumb", "middle", "ring")
    normal_target_limit_n: float = 6.
    active_finger_floor_n: float = .75
    patch_floor_n: float = .05
    uncontrolled_contact_tolerance_n: float = 1e-5
    gravity_margin: float = 1.
    gravity_m_s2: float = 9.81
    maximum_friction_coefficient: float = .8
    friction_safety_factor: float = .85
    cone_rays: int = 16
    group_point_distance_m: float = .015
    group_normal_cosine_minimum: float = .9
    previous_reference_weight: float = .75
    normal_regularizer: float = 1e-4
    torque_scale_m: float = .05
    solver_time_limit_s: float = .1
    maximum_observation_age_s: float = 1e-9

    def __post_init__(self):
        active = self.active_fingers
        if (not isinstance(active, tuple) or not active or len(set(active)) != len(active)
                or any(finger not in CHANNELS[:-1] for finger in active)):
            raise ValueError("active_fingers must be distinct named fingers in an ordered tuple")
        for name in ("normal_target_limit_n", "active_finger_floor_n", "patch_floor_n",
                     "gravity_m_s2", "group_point_distance_m", "normal_regularizer",
                     "torque_scale_m", "solver_time_limit_s"):
            _number(getattr(self, name), name, positive=True)
        for name in ("uncontrolled_contact_tolerance_n", "maximum_friction_coefficient",
                     "maximum_observation_age_s"):
            _number(getattr(self, name), name)
        if (not 0. < self.active_finger_floor_n <= self.normal_target_limit_n <= 6.
                or not self.uncontrolled_contact_tolerance_n < self.patch_floor_n
                or self.patch_floor_n > self.normal_target_limit_n):
            raise ValueError("Target limits require floor <= limit <= 6 N and tolerance < patch floor")
        if (_number(self.gravity_margin, "gravity_margin") < 1.
                or not 0. < _number(self.friction_safety_factor, "friction_safety_factor") <= 1.
                or not 0. <= _number(self.previous_reference_weight, "previous_reference_weight") <= 1.
                or not 0. <= _number(self.group_normal_cosine_minimum, "group_normal_cosine_minimum") <= 1.):
            raise ValueError("Invalid declared margin, friction safety, reference or normal agreement")
        if (isinstance(self.cone_rays, bool) or not isinstance(self.cone_rays, Integral)
                or not 8 <= self.cone_rays <= 64):
            raise ValueError("cone_rays must be an integer from eight through 64")


def _channel_values(values, name):
    if not isinstance(values, Mapping) or set(values) != set(CHANNELS):
        raise ValueError(name + " must contain all six named tactile channels")
    return {finger: _number(values[finger], name + "." + finger) for finger in CHANNELS}


def _prepare(patches, pressures, friction, config):
    retained, excluded, indices = [], [], set()
    totals = dict.fromkeys(CHANNELS, 0.)
    for raw in patches:
        index, finger, body = raw["input_index"], raw["finger"], raw["hand_body_name"]
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                or index < 0 or index in indices):
            raise ValueError("Patch indices must be unique nonnegative integers")
        indices.add(int(index))
        if finger not in CHANNELS or not isinstance(body, str) or not body:
            raise ValueError("A patch requires a known finger and hand body")
        if finger != "palm" and not body.startswith("left_" + finger + "_"):
            raise ValueError("Patch body does not belong to its declared finger")
        point = _vector(raw["position_world_m"], "position_world_m")
        normal = _vector(raw["normal_on_object_world"], "normal_on_object_world")
        length = float(np.linalg.norm(normal))
        if not .5 <= length <= 1.5:
            raise ValueError("Contact normal must be a usable unit direction")
        measured = _number(raw["measured_pressure_n"], "measured_pressure_n")
        mu = min(friction, _number(raw.get("friction_coefficient", friction), "patch friction"),
            config.maximum_friction_coefficient) * config.friction_safety_factor
        totals[finger] += measured
        if not math.isfinite(totals[finger]):
            raise ValueError("Nonfinite patch load sum")
        if finger in config.active_fingers and measured >= config.patch_floor_n:
            retained.append({"input_index": int(index), "raw_input_index": int(index),
                "finger": finger, "hand_body_name": body, "position_world_m": point.tolist(),
                "normal_on_object_world": (normal / length).tolist(),
                "measured_pressure_n": measured, "effective_friction_coefficient": mu})
        else:
            excluded.append(int(index))
    if any(totals[f] > pressures[f] + config.uncontrolled_contact_tolerance_n for f in CHANNELS):
        raise ValueError("Associated patch sum exceeds the all-contact channel reading")
    return retained, excluded


def _current_groups(patches, config):
    groups = []
    ordered = sorted(range(len(patches)), key=lambda i: (patches[i]["finger"],
        patches[i]["hand_body_name"], *patches[i]["position_world_m"], patches[i]["input_index"]))
    for index in ordered:
        patch = patches[index]
        point, normal = np.asarray(patch["position_world_m"]), np.asarray(patch["normal_on_object_world"])
        group = next((members for members in groups if all(
            patches[other]["finger"] == patch["finger"]
            and patches[other]["hand_body_name"] == patch["hand_body_name"]
            and np.linalg.norm(point - patches[other]["position_world_m"]) <= config.group_point_distance_m
            and normal @ patches[other]["normal_on_object_world"] >= config.group_normal_cosine_minimum
            for other in members)), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    return groups


def allocate_contact_load_targets(patches, estimated_com_world_m, catalogue_mass_kg,
                                 measured_pressures_n, *, declared_friction_coefficient,
                                 config=None, previous_targets_n=None,
                                 observation_timestamp_s=None, now_s=None):
    """Allocate current per-patch loads with measured within-group proportions.

    ``patches`` accepts ``ContactObservation.as_allocator_patches()`` directly.
    Normals point into the external object. Grouping is rebuilt from this call's
    points, bodies and normals; raw indices are not persistent contact IDs.
    Supplying both timestamps checks age, not provenance or visual association.
    A caller must separately enforce association and whole-episode force/task
    gates. The 6 N target bound does not bound measured force; 12 N is not an LP
    admission criterion. Infeasible/invalid solves return no replacement targets.
    Only ``config.gravity_margin`` is checked, without claiming other margins.
    """
    result = {"format": "fresh_group_contact_load_targets_v1", "feasible": False,
        "reason": "INVALID_INPUT", "runtime_target_state_used": False,
        "freshness_checked": False, "task_success": None, "force_invariant_proven": False,
        "measured_wrench_equilibrium_checked": False,
        "claim_scope": "Estimated static wrench at one declared gravity margin, not measured equilibrium or task success"}
    try:
        config = ContactLoadTargetConfig() if config is None else config
        if not isinstance(config, ContactLoadTargetConfig):
            raise ValueError("config must be ContactLoadTargetConfig")
        center = _vector(estimated_com_world_m, "estimated_com_world_m")
        mass = _number(catalogue_mass_kg, "catalogue_mass_kg", positive=True)
        friction = _number(declared_friction_coefficient, "declared_friction_coefficient")
        pressures = _channel_values(measured_pressures_n, "measured_pressures_n")
        result.update(config=asdict(config), declared_friction_coefficient=friction)
        if (observation_timestamp_s is None) != (now_s is None):
            raise ValueError("Supply both observation_timestamp_s and now_s, or neither")
        if now_s is not None:
            timestamp = _number(observation_timestamp_s, "observation_timestamp_s")
            age = _number(now_s, "now_s") - timestamp
            result.update(observation_timestamp_s=timestamp, observation_age_s=age, freshness_checked=True)
            if age < 0. or age > config.maximum_observation_age_s:
                return dict(result, reason="STALE_OR_FUTURE_OBSERVATION")
        if any(pressures[f] > config.uncontrolled_contact_tolerance_n
                for f in CHANNELS if f not in config.active_fingers):
            return dict(result, reason="UNCONTROLLED_CONTACT")
        retained, excluded = _prepare(patches, pressures, friction, config)
        result["excluded_raw_input_indices"] = excluded
        if {p["finger"] for p in retained} != set(config.active_fingers):
            return dict(result, reason="MISSING_ACTIVE_CONTACT")
        if previous_targets_n is None:
            reference = np.array([pressures[f] for f in config.active_fingers])
        else:
            previous = _channel_values(previous_targets_n, "previous_targets_n")
            if any(previous[f] != 0. for f in CHANNELS if f not in config.active_fingers):
                raise ValueError("Previous targets load an uncontrolled channel")
            if any(previous[f] > config.normal_target_limit_n for f in CHANNELS):
                raise ValueError("Previous targets exceed the declared target limit")
            weight = config.previous_reference_weight
            reference = np.array([weight * previous[f] + (1. - weight) * pressures[f]
                for f in config.active_fingers])

        groups = _current_groups(retained, config)
        count, active = len(retained), len(config.active_fingers)
        ray_count = count * config.cone_rays
        normals = np.zeros((count, ray_count))
        channels = np.zeros((active, ray_count))
        directions, arms = [], []
        for index, patch in enumerate(retained):
            normal = np.asarray(patch["normal_on_object_world"])
            axis = np.eye(3)[int(np.argmin(abs(normal)))]
            first = np.cross(normal, axis)
            first /= np.linalg.norm(first)
            second = np.cross(normal, first)
            theta = 2. * math.pi * np.arange(config.cone_rays) / config.cone_rays
            rays = normal + patch["effective_friction_coefficient"] * (
                np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second)
            directions.extend(rays)
            arms.extend([np.asarray(patch["position_world_m"]) - center] * config.cone_rays)
            section = slice(index * config.cone_rays, (index + 1) * config.cone_rays)
            normals[index, section] = 1.
            channels[config.active_fingers.index(patch["finger"]), section] = 1.
        directions, arms = np.asarray(directions), np.asarray(arms)
        wrench = np.vstack([directions.T, np.cross(arms, directions).T])
        scaled = wrench.copy()
        scaled[3:] /= config.torque_scale_m
        gravity = np.array([0., 0., mass * config.gravity_m_s2 * config.gravity_margin, 0., 0., 0.])
        ratio_rows = []
        for members in groups:
            total = sum(retained[i]["measured_pressure_n"] for i in members)
            for index in members[:-1]:
                ratio = retained[index]["measured_pressure_n"] / total
                ratio_rows.append(normals[index] - ratio * normals[members].sum(axis=0))
        ratios = np.asarray(ratio_rows).reshape((-1, ray_count))
        equality = np.c_[np.vstack([scaled, ratios]), np.zeros((6 + len(ratios), active + count))]
        right = np.r_[gravity, np.zeros(len(ratios))]
        channel_slack = np.c_[np.eye(active), np.zeros((active, count))]
        patch_slack = np.c_[np.zeros((count, active)), np.eye(count)]
        loads = np.vstack([channels, -channels, channels, -channels, normals, -normals])
        slack = np.vstack([np.zeros((2 * active, active + count)),
            -channel_slack, -channel_slack, -patch_slack, -patch_slack])
        inequality = np.c_[loads, slack]
        measured = np.array([p["measured_pressure_n"] for p in retained])
        upper = np.r_[np.full(active, config.normal_target_limit_n),
            np.full(active, -config.active_finger_floor_n), reference, -reference, measured, -measured]
        objective = np.r_[np.full(ray_count, config.normal_regularizer), np.ones(active + count)]
        if not all(np.isfinite(v).all() for v in (wrench, equality, right, inequality, upper, objective)):
            return dict(result, reason="NONFINITE_LP_COEFFICIENTS")
        solved = linprog(objective, A_ub=inequality, b_ub=upper, A_eq=equality, b_eq=right,
            bounds=(0., None), method="highs", options={"primal_feasibility_tolerance": 1e-8,
                "dual_feasibility_tolerance": 1e-8, "time_limit": config.solver_time_limit_s})
        result.update(solver_status=int(solved.status), solver_message=str(solved.message))
        if not solved.success or solved.status != 0 or solved.x is None:
            return dict(result, reason="STATIC_GROUP_LP_FAILED")
        values = np.asarray(solved.x, dtype=float)
        if values.shape != (ray_count + active + count,) or not np.isfinite(values).all():
            return dict(result, reason="STATIC_GROUP_LP_INVALID_SOLUTION")
        rays = values[:ray_count]
        patch_loads, channel_loads = normals @ rays, channels @ rays
        residual, ratio_residual = wrench @ rays - gravity, ratios @ rays
        excess = inequality @ values - upper
        if not all(np.isfinite(v).all() for v in (patch_loads, channel_loads, residual, ratio_residual, excess)):
            return dict(result, reason="STATIC_GROUP_LP_NONFINITE_RESIDUAL")
        checks = {"force_residual_max_n": float(np.max(abs(residual[:3]))),
            "torque_residual_max_nm": float(np.max(abs(residual[3:]))),
            "group_proportion_residual_max_n": float(np.max(abs(ratio_residual))) if len(ratios) else 0.,
            "inequality_excess_max_n": float(max(0., np.max(excess))),
            "nonnegative_bound_violation_max_n": float(max(0., -np.min(values)))}
        result["numerical_checks"] = checks
        if (checks["force_residual_max_n"] > 1e-5 or checks["torque_residual_max_nm"] > 1e-6
                or checks["group_proportion_residual_max_n"] > 1e-6
                or checks["inequality_excess_max_n"] > 1e-6
                or checks["nonnegative_bound_violation_max_n"] > 1e-8 or np.any(patch_loads < 0.)):
            return dict(result, reason="STATIC_GROUP_LP_RESIDUAL")
        targets = dict.fromkeys(CHANNELS, 0.)
        targets.update(zip(config.active_fingers, channel_loads.tolist()))
        output = [dict(patch, target_normal_n=float(load))
            for patch, load in zip(retained, patch_loads, strict=True)]
        group_output = []
        for members in groups:
            total = sum(retained[i]["measured_pressure_n"] for i in members)
            group_output.append({"finger": retained[members[0]]["finger"],
                "hand_body_name": retained[members[0]]["hand_body_name"],
                "raw_input_indices": [retained[i]["input_index"] for i in members],
                "measured_proportions": [retained[i]["measured_pressure_n"] / total for i in members],
                "measured_normal_n": total, "target_normal_n": float(patch_loads[members].sum())})
        return dict(result, feasible=True, reason="STATIC_GROUP_TARGET_ALLOCATED", targets_n=targets,
            patches=output, groups=group_output, checked_gravity_margins=[config.gravity_margin],
            reference_n=dict(zip(config.active_fingers, reference.tolist())),
            target_wrench_residual=residual.tolist(),
            allocated_patch_forces_world_n=(rays.reshape((count, config.cone_rays, 1))
                * directions.reshape((count, config.cone_rays, 3))).sum(axis=1).tolist())
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return dict(result, details=str(error))
