"""Versioned physical assumptions and consistent runtime parameter changes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


YCB_MASS_SOURCE = "https://people.eecs.berkeley.edu/~pabbeel/papers/2015-RAM-YCB-object-set.pdf"
KUKA_LIMIT_SOURCE = "https://www.kuka.com/-/media/kuka-downloads/imported/87f2706ce77c4318877932fb36f6002d/kuka_lbr_iiwa_ja.pdf"
ARM_TORQUE_LIMITS_NM = (320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0)
ARM_VELOCITY_LIMITS_RAD_S = tuple(math.radians(value) for value in (85, 85, 100, 75, 130, 135, 135))


@dataclass(frozen=True)
class ObjectPhysics:
    kind: str
    mass_kg: float
    mass_source: str
    friction: tuple[float, float, float]
    friction_source: str = "Engineering assumption; not measured for this object/material pair."
    inertia_source: str = "Uniform-density rigid mesh/box approximation compiled by MuJoCo; not measured."
    condim: int = 3
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    version: int
    gravity: tuple[float, float, float]
    timestep: float
    objects: tuple[ObjectPhysics, ...]
    table_friction: tuple[float, float, float]
    hand_friction: tuple[float, float, float]
    arm_torque_limits_nm: tuple[float, ...] | None
    arm_velocity_limits_rad_s: tuple[float, ...] | None
    arm_acceleration_limits_rad_s2: tuple[float, ...] | None
    friction_mixing: str
    limit_source: str
    acceleration_source: str = "Engineering motion-command limit; not a manufacturer acceleration specification."
    integrator: str = "implicitfast"
    solver: str = "Newton"
    iterations: int = 100
    tolerance: float = 1e-8
    solver_impratio: float = 10.0
    hand_rigid_solref: tuple[float, float] = (0.01, 1.0)
    hand_pad_solref: tuple[float, float] = (0.06, 0.9)
    contact_parameter_revision: int = 1
    contact_solver_mixing: str = "Native MuJoCo geom priority and solmix, resolved after XML default inheritance."

    def object_parameters(self, kind: str) -> ObjectPhysics:
        for parameters in self.objects:
            if parameters.kind == kind:
                return parameters
        raise ValueError(f"unsupported object kind: {kind}")


_LEGACY_OBJECTS = tuple(
    ObjectPhysics(kind, 0.15, "Legacy V5 scene-authored assumption.", (0.8, 0.01, 0.001))
    for kind in ("box", "ycb_can", "ycb_potted_meat_can", "ycb_cracker_box")
)
_V6_OBJECTS = (
    ObjectPhysics("box", 0.15, "Engineering test-object assumption.", (0.8, 0.005, 0.0001)),
    ObjectPhysics("ycb_can", 0.349, f"YCB Table II, item 5 (349 g): {YCB_MASS_SOURCE}", (0.6, 0.005, 0.0001)),
    ObjectPhysics("ycb_potted_meat_can", 0.370, f"YCB Table II, item 10 (370 g): {YCB_MASS_SOURCE}", (0.6, 0.005, 0.0001)),
    ObjectPhysics("ycb_cracker_box", 0.411, f"YCB Table II, item 3 (411 g): {YCB_MASS_SOURCE}", (0.65, 0.005, 0.0001)),
)
PHYSICS_PROFILES = {
    "legacy-v5": PhysicsProfile(
        "legacy-v5", 5, (0.0, 0.0, -9.81), 0.002, _LEGACY_OBJECTS,
        (0.8, 0.01, 0.001), (1.0, 0.005, 0.0001), None, None, None,
        "MuJoCo default geom mixing (maximum friction at equal priority)",
        "Legacy arm actuators have no joint-level torque limit.",
    ),
    "physical-v6": PhysicsProfile(
        "physical-v6", 6, (0.0, 0.0, -9.81), 0.002, _V6_OBJECTS,
        (0.6, 0.005, 0.0001), (1.0, 0.005, 0.0001),
        ARM_TORQUE_LIMITS_NM, ARM_VELOCITY_LIMITS_RAD_S, (0.8,) * 7,
        "Explicit target contact pairs: geometric mean; engineering assumption, not a material law.",
        KUKA_LIMIT_SOURCE,
        tolerance=1e-10,
        solver_impratio=100.0,
        contact_parameter_revision=2,
    ),
}


def get_physics_profile(name: str = "legacy-v5") -> PhysicsProfile:
    try:
        return PHYSICS_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unsupported physics profile: {name}; choose {tuple(PHYSICS_PROFILES)}") from error


def model_physics_version(model: mujoco.MjModel) -> int:
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "physics_profile_version")
    return 5 if marker < 0 else int(model.numeric_data[model.numeric_adr[marker]])


def _arm_user_limits(model: mujoco.MjModel, column: int) -> np.ndarray | None:
    if model.nuser_jnt <= column:
        return None
    indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    if min(indices) < 0:
        raise ValueError("model is missing a KUKA arm joint")
    values = np.array(model.jnt_user[indices, column], dtype=float)
    return values if np.all(np.isfinite(values)) and np.all(values > 0) else None


def arm_joint_velocity_limits(model: mujoco.MjModel) -> np.ndarray | None:
    """Command velocity caps; MuJoCo does not impose a native qvel clamp."""
    return _arm_user_limits(model, 0)


def arm_joint_acceleration_limits(model: mujoco.MjModel) -> np.ndarray | None:
    return _arm_user_limits(model, 1)


def pair_friction(first: tuple[float, ...] | np.ndarray, second: tuple[float, ...] | np.ndarray) -> tuple[float, ...]:
    """Map three geom coefficients to five explicit contact-pair coefficients."""
    mixed = np.sqrt(np.asarray(first, dtype=float) * np.asarray(second, dtype=float))
    return tuple(float(mixed[index]) for index in (0, 0, 1, 2, 2))


def native_contact_parameters(model: mujoco.MjModel, first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve native solref/solimp mixing in an explicit pair, after defaults resolve."""
    priorities = model.geom_priority[[first, second]]
    if priorities[0] != priorities[1]:
        selected = first if priorities[0] > priorities[1] else second
        return model.geom_solref[selected].copy(), model.geom_solimp[selected].copy()
    weights = model.geom_solmix[[first, second]]
    if np.all(weights < mujoco.mjMINVAL):
        first_weight = 0.5
    elif weights[0] < mujoco.mjMINVAL:
        first_weight = 0.0
    elif weights[1] < mujoco.mjMINVAL:
        first_weight = 1.0
    else:
        first_weight = float(weights[0] / weights.sum())
    solrefs = model.geom_solref[[first, second]]
    if np.all(solrefs > 0):
        solref = first_weight * solrefs[0] + (1 - first_weight) * solrefs[1]
    else:
        solref = np.minimum(solrefs[0], solrefs[1])
    solimp = first_weight * model.geom_solimp[first] + (1 - first_weight) * model.geom_solimp[second]
    return solref, solimp


@dataclass(frozen=True)
class ObjectParameterReference:
    mass: float
    inertia: tuple[float, float, float]
    friction: tuple[float, float, float]


def capture_object_reference(model: mujoco.MjModel, body_id: int, geom_id: int) -> ObjectParameterReference:
    return ObjectParameterReference(
        float(model.body_mass[body_id]), tuple(float(value) for value in model.body_inertia[body_id]),
        tuple(float(value) for value in model.geom_friction[geom_id]),
    )


def set_object_parameters(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, geom_id: int, *,
    reference: ObjectParameterReference, mass: float | None = None, friction: float | None = None,
) -> tuple[float, float]:
    """Change density at fixed shape/COM; restore the captured reference with None."""
    mass_value = reference.mass if mass is None else float(mass)
    friction_value = reference.friction[0] if friction is None else float(friction)
    if not math.isfinite(mass_value) or mass_value <= 0:
        raise ValueError("object mass must be finite and positive")
    if not math.isfinite(friction_value) or friction_value <= 0:
        raise ValueError("object friction must be finite and positive")
    inertia = np.asarray(reference.inertia) * (mass_value / reference.mass)
    if not np.all(np.isfinite(inertia)) or np.any(inertia <= 0):
        raise ValueError("object inertia must remain finite and positive")
    model.body_mass[body_id] = mass_value
    model.body_inertia[body_id] = inertia
    model.geom_friction[geom_id] = (friction_value, *reference.friction[1:])
    if model_physics_version(model) >= 6:
        for pair_id in range(model.npair):
            first, second = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
            if geom_id in (first, second):
                model.pair_friction[pair_id] = pair_friction(model.geom_friction[first], model.geom_friction[second])
    mujoco.mj_setConst(model, data)
    return mass_value, friction_value
