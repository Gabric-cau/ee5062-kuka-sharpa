"""Explicit finger/actuator contract for a tactile execution layer.

Forces are nonnegative normal loads in newtons (the historical ``pressure``
keys also mean N, not Pa); points are world-frame metres and normals point
into the object. Joint commands are position-servo references in radians.
This module validates interfaces, not contact freshness, Jacobian rank, force
safety, equilibrium, or task success. It never reads target state or steps physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CONTACT_CHANNELS = (*FINGER_NAMES, "palm")


class ContactPolicyError(ValueError):
    """A caller must hold/recover rather than silently drop a failed contract."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def _number(value, name, *, minimum=0., maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContactPolicyError("INVALID_NUMBER", f"{name} must be a real numeric scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ContactPolicyError("INVALID_NUMBER", f"{name} cannot be represented as a finite float") from error
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ContactPolicyError("INVALID_NUMBER", f"{name} is nonfinite or outside its allowed range")
    return result


def _names(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContactPolicyError("INVALID_NAMES", f"{name} must be an ordered sequence")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result) or len(set(result)) != len(result):
        raise ContactPolicyError("INVALID_NAMES", f"{name} must contain unique nonempty strings")
    return result


def _channels(values, name, *, require_palm):
    required = set(CONTACT_CHANNELS if require_palm else FINGER_NAMES)
    if not isinstance(values, Mapping) or not required <= set(values) <= set(CONTACT_CHANNELS):
        raise ContactPolicyError("INVALID_CHANNELS", f"{name} requires all named fingers"
            + (" and palm" if require_palm else " with an optional palm"))
    return {finger: _number(values.get(finger, 0.), f"{name}.{finger}") for finger in CONTACT_CHANNELS}


def _vector(values, name):
    if isinstance(values, (str, bytes)):
        raise ContactPolicyError("INVALID_VECTOR", f"{name} must have three finite components")
    try:
        result = tuple(_number(value, name, minimum=-math.inf) for value in values)
    except TypeError as error:
        raise ContactPolicyError("INVALID_VECTOR", f"{name} must have three finite components") from error
    if len(result) != 3:
        raise ContactPolicyError("INVALID_VECTOR", f"{name} must have three finite components")
    return result


@dataclass(frozen=True)
class ContactProfile:
    contact_closure: Mapping[str, float]
    force_targets_n: Mapping[str, float]


@dataclass(frozen=True)
class ContactPatch:
    input_index: int
    finger: str
    hand_body_name: str
    position_world_m: tuple[float, float, float]
    normal_on_object_world: tuple[float, float, float]
    measured_pressure_n: float
    friction_coefficient: float | None

    def as_allocator_patch(self):
        result = {"input_index": self.input_index, "finger": self.finger,
            "hand_body_name": self.hand_body_name, "position_world_m": list(self.position_world_m),
            "normal_on_object_world": list(self.normal_on_object_world),
            "measured_pressure_n": self.measured_pressure_n}
        if self.friction_coefficient is not None:
            result["friction_coefficient"] = self.friction_coefficient
        return result


@dataclass(frozen=True)
class ContactObservation:
    """Validated contact content with an optional caller-supplied sample time.

    Carrying a timestamp does not itself establish freshness or provenance.
    An execution layer must compare it with its current native sample time.
    """

    patches: tuple[ContactPatch, ...]
    measured_pressures_n: Mapping[str, float]
    observed_fingers: tuple[str, ...]
    excluded_unloaded_patch_indices: tuple[int, ...]
    timestamp_s: float | None = None

    def as_allocator_patches(self):
        """Fresh copies using the existing pure tactile allocator's argument keys."""
        return [patch.as_allocator_patch() for patch in self.patches]


@dataclass(frozen=True)
class BoundFingerJoint:
    finger: str
    name: str
    joint_id: int
    actuator_id: int
    dof_index: int
    qpos_index: int
    command_range_rad: tuple[float, float]


@dataclass(frozen=True)
class FingerContactPolicy:
    """One shared declaration for profile, allocator, Jacobian and command masks.

    ``joints_by_finger`` names the selected independent left-hand joints, not
    every posture joint. Each name must identify both its hinge and its position
    actuator. Call ``bind_mujoco_model`` at startup, ``validate_profile`` before
    acquisition, and ``validate_contacts``/``validate_targets_n`` every solve.
    Actual Jacobian controllability and physical emergency handling remain the
    execution layer's responsibility, including on inactive finger/palm contacts.
    """

    active_fingers: tuple[str, ...]
    joints_by_finger: Mapping[str, tuple[str, ...]]
    normal_target_limit_n: float = 6.
    contact_floor_n: float = .05
    uncontrolled_contact_tolerance_n: float = 1e-5

    def __post_init__(self):
        active = _names(self.active_fingers, "active_fingers")
        if not active or any(finger not in FINGER_NAMES for finger in active):
            raise ContactPolicyError("INVALID_ACTIVE_FINGERS", "select one or more of the five named fingers")
        if not isinstance(self.joints_by_finger, Mapping) or set(self.joints_by_finger) != set(active):
            raise ContactPolicyError("JOINT_FINGER_MASK_MISMATCH", "joint groups must exactly match active fingers")
        groups = {}
        for finger in active:
            names = _names(self.joints_by_finger[finger], f"joints_by_finger.{finger}")
            if not names or any(not name.startswith(f"left_{finger}_") for name in names):
                raise ContactPolicyError("INVALID_FINGER_JOINTS", f"{finger} needs its own executable left-hand joints")
            groups[finger] = names
        limit = _number(self.normal_target_limit_n, "normal_target_limit_n")
        floor = _number(self.contact_floor_n, "contact_floor_n")
        tolerance = _number(self.uncontrolled_contact_tolerance_n, "uncontrolled_contact_tolerance_n")
        if not 0. < floor <= limit or tolerance >= floor:
            raise ContactPolicyError("INVALID_FORCE_THRESHOLDS", "require 0 <= tolerance < contact floor <= target limit")
        object.__setattr__(self, "active_fingers", active)
        object.__setattr__(self, "joints_by_finger", MappingProxyType(groups))
        object.__setattr__(self, "normal_target_limit_n", limit)
        object.__setattr__(self, "contact_floor_n", floor)
        object.__setattr__(self, "uncontrolled_contact_tolerance_n", tolerance)

    @property
    def controlled_joint_names(self):
        return tuple(name for finger in self.active_fingers for name in self.joints_by_finger[finger])

    def validate_control_mask(self, joint_names):
        actual = _names(joint_names, "controlled_joint_names")
        if set(actual) != set(self.controlled_joint_names):
            raise ContactPolicyError("CONTROL_MASK_MISMATCH", "executed joint mask differs from the declared policy")

    def validate_targets_n(self, targets_n):
        """Validate profile or allocator loads without silently clipping/filtering."""
        targets = _channels(targets_n, "targets_n", require_palm=False)
        for finger, target in targets.items():
            if finger not in self.active_fingers and target != 0.:
                raise ContactPolicyError("UNCONTROLLED_FORCE_TARGET", f"{finger} has a {target} N target without control")
            if target > self.normal_target_limit_n:
                raise ContactPolicyError("FORCE_TARGET_LIMIT", f"{finger} exceeds the configured target limit in N")
        return MappingProxyType(targets)

    def validate_profile(self, profile):
        """Check only the acquisition closure and normal-load portions of a profile."""
        if not isinstance(profile, Mapping) or not {"contact_closure", "force_targets_n"} <= set(profile):
            raise ContactPolicyError("INVALID_PROFILE", "profile needs contact_closure and force_targets_n")
        values = profile["contact_closure"]
        if not isinstance(values, Mapping) or set(values) != set(FINGER_NAMES):
            raise ContactPolicyError("INVALID_CLOSURE_CHANNELS", "contact_closure needs exactly all five fingers")
        closure = {finger: _number(values[finger], f"contact_closure.{finger}", maximum=1.)
            for finger in FINGER_NAMES}
        targets = self.validate_targets_n(profile["force_targets_n"])
        for finger in FINGER_NAMES:
            if finger not in self.active_fingers and closure[finger] != 0.:
                raise ContactPolicyError("UNCONTROLLED_CLOSURE", f"{finger} closes without a declared control group")
            if targets[finger] > 0. and closure[finger] == 0.:
                raise ContactPolicyError("FORCE_WITHOUT_ACQUISITION", f"{finger} requests force with zero acquisition closure")
        return ContactProfile(MappingProxyType(closure), targets)

    def validate_contacts(self, patches, measured_pressures_n, *, controlled_joint_names,
                          require_all_active=True, timestamp_s=None):
        """Check one associated sample and retain its original raw patch indices.

        All six target-contact readings are mandatory, so palm and inactive
        loads cannot disappear when associated patches are a subset. A patch's
        body must belong to its stated finger. This does not establish that the
        raw index, body, pose or timestamp is current; association must do that.
        An optional finite nonnegative ``timestamp_s`` is carried unchanged in
        seconds; this validation does not independently prove its freshness.
        ``require_all_active=False`` supports contact acquisition, never lift
        admission. High measured active loads remain visible for safety relief;
        the normal target cap is not a measured-force safety certificate.
        """
        self.validate_control_mask(controlled_joint_names)
        if not isinstance(require_all_active, bool):
            raise ContactPolicyError("INVALID_CONTACT_MODE", "require_all_active must be boolean")
        timestamp = None if timestamp_s is None else _number(timestamp_s, "timestamp_s")
        readings = _channels(measured_pressures_n, "measured_pressures_n", require_palm=True)
        for finger, pressure in readings.items():
            if finger not in self.active_fingers and pressure > self.uncontrolled_contact_tolerance_n:
                raise ContactPolicyError("UNCONTROLLED_CONTACT", f"observed {pressure} N on uncontrolled {finger}")
        totals = dict.fromkeys(CONTACT_CHANNELS, 0.)
        kept, excluded, indices = [], [], set()
        try:
            iterator = iter(patches)
        except TypeError as error:
            raise ContactPolicyError("INVALID_PATCHES", "patches must be iterable") from error
        for raw in iterator:
            if not isinstance(raw, Mapping) or not {"input_index", "finger", "hand_body_name",
                    "position_world_m", "normal_on_object_world", "measured_pressure_n"} <= set(raw):
                raise ContactPolicyError("INVALID_PATCH", "associated patch is missing required fields")
            index, finger, body = raw["input_index"], raw["finger"], raw["hand_body_name"]
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0 or index in indices:
                raise ContactPolicyError("INVALID_PATCH_INDEX", "raw indices must be unique nonnegative integers")
            indices.add(int(index))
            if not isinstance(finger, str) or finger not in CONTACT_CHANNELS:
                raise ContactPolicyError("INVALID_PATCH_CHANNEL", "unknown tactile channel")
            if not isinstance(body, str) or not body or (finger != "palm" and not body.startswith(f"left_{finger}_")):
                raise ContactPolicyError("PATCH_BODY_FINGER_MISMATCH", "hand body must belong to the declared finger")
            point = _vector(raw["position_world_m"], "position_world_m")
            normal = _vector(raw["normal_on_object_world"], "normal_on_object_world")
            length = math.hypot(*normal)
            if not .5 <= length <= 1.5:
                raise ContactPolicyError("INVALID_CONTACT_NORMAL", "normal must be a usable unit direction")
            normal = tuple(value / length for value in normal)
            pressure = _number(raw["measured_pressure_n"], "measured_pressure_n")
            friction = (_number(raw["friction_coefficient"], "friction_coefficient")
                if "friction_coefficient" in raw else None)
            totals[finger] += pressure
            if not math.isfinite(totals[finger]):
                raise ContactPolicyError("INVALID_PATCH_LOAD_SUM", "patch load arithmetic overflow")
            if finger not in self.active_fingers:
                if pressure > self.uncontrolled_contact_tolerance_n:
                    raise ContactPolicyError("UNCONTROLLED_CONTACT", f"raw patch {index} loads uncontrolled {finger}")
                excluded.append(int(index))
            else:
                kept.append(ContactPatch(int(index), finger, body, point, normal, pressure, friction))
        if any(totals[finger] > readings[finger] + self.uncontrolled_contact_tolerance_n for finger in CONTACT_CHANNELS):
            raise ContactPolicyError("PATCH_CHANNEL_LOAD_MISMATCH", "associated patch sum exceeds all-contact normal load")
        observed = tuple(finger for finger in self.active_fingers if totals[finger] >= self.contact_floor_n
            and readings[finger] >= self.contact_floor_n)
        if require_all_active and set(observed) != set(self.active_fingers):
            raise ContactPolicyError("MISSING_ACTIVE_CONTACT", "every declared finger needs observed associated load")
        return ContactObservation(tuple(kept), MappingProxyType(readings), observed, tuple(excluded), timestamp)

    def bind_mujoco_model(self, model):
        """Resolve selected hinge/position-servo identities without reading data.

        A same-named actuator must transmit directly to the named hinge with
        unit gear and a finite, nonempty command interval in radians. The bound
        DOF indices can select columns of ``mj_jac``. This verifies the command
        interface; it does not prove a contact Jacobian has useful authority.
        """
        import mujoco
        import numpy as np

        result = []
        for finger in self.active_fingers:
            for name in self.joints_by_finger[finger]:
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if joint < 0 or actuator < 0:
                    raise ContactPolicyError("MISSING_EXECUTABLE_JOINT", f"{name} needs a hinge and same-named actuator")
                if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                        or model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                        or model.actuator_trnid[actuator, 0] != joint
                        or not np.array_equal(model.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.])):
                    raise ContactPolicyError("INVALID_JOINT_TRANSMISSION", f"{name} must directly control its own hinge in radians")
                gain, bias = model.actuator_gainprm[actuator], model.actuator_biasprm[actuator]
                if (model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                        or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                        or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                        or not np.isfinite(gain).all() or not np.isfinite(bias).all()
                        or gain[0] <= 0. or bias[0] != 0. or bias[1] != -gain[0] or bias[2] > 0.):
                    raise ContactPolicyError("INVALID_POSITION_SERVO", f"{name} does not accept a position reference")
                lower, upper = model.jnt_range[joint]
                if not model.jnt_limited[joint] or not np.isfinite([lower, upper]).all() or lower >= upper:
                    raise ContactPolicyError("INVALID_JOINT_RANGE", f"{name} needs a finite hinge range in radians")
                if model.actuator_ctrllimited[actuator]:
                    ctrl_lower, ctrl_upper = model.actuator_ctrlrange[actuator]
                    if not np.isfinite([ctrl_lower, ctrl_upper]).all() or ctrl_lower >= ctrl_upper:
                        raise ContactPolicyError("INVALID_COMMAND_RANGE", f"{name} has an invalid position-control range")
                    lower, upper = max(lower, ctrl_lower), min(upper, ctrl_upper)
                if lower >= upper:
                    raise ContactPolicyError("INVALID_COMMAND_RANGE", f"{name} has no usable position-control interval")
                result.append(BoundFingerJoint(finger, name, joint, actuator, int(model.jnt_dofadr[joint]),
                    int(model.jnt_qposadr[joint]), (float(lower), float(upper))))
        return tuple(result)
