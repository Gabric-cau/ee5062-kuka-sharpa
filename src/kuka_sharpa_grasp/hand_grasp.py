"""Angle-space hand posture and simulated target-contact feedback.

Flexion and abduction are different motions; joint range interpolation cannot
serve as a grasp synergy. This posture keeps abduction neutral and opposes the
thumb before inserting the open aperture around the known upright can.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
from types import MappingProxyType
import mujoco
import numpy as np
from .control import FINGERS, ForceReading, ContactForceReader


CALIBRATED_CAN_CLOSURE = MappingProxyType({
    'thumb': .4216, 'index': .3916, 'middle': .4008, 'ring': .4660, 'pinky': 1.,
})


class HandContactReader:
    """Finger segment contacts, including proximal links, against the target."""
    def __init__(self, model):
        self.model = model
        self.target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        self.fingers = {}
        for gid in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[gid])) or ''
            if body.startswith('left_'):
                finger = next((f for f in FINGERS if body.startswith('left_' + f + '_')), None)
                self.fingers[gid] = finger or 'palm'

    def read(self, data):
        values = {finger: ForceReading() for finger in (*FINGERS, 'palm')}
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if self.target not in (a, b):
                continue
            other = b if a == self.target else a
            finger = self.fingers.get(other)
            if finger is None:
                continue
            mujoco.mj_contactForce(self.model, data, index, wrench)
            values[finger].normal += abs(float(wrench[0]))
            values[finger].tangential += float(np.linalg.norm(wrench[1:3]))
            values[finger].contacts += 1
        return values

    total = staticmethod(ContactForceReader.total)


@dataclass(frozen=True)
class HandConfig:
    close_rate: float = .4
    contact_threshold_n: float = 2.
    thumb_opposition_rad: float = 1.
    retention_rate: float = .03
    retention_extra: float = .03
    maximum_force_n: float | None = None
    relief_rate: float = .12
    retention_minimum_force_n: float | None = None
    thumb_mcp_flexion_ratio: float = .8
    thumb_ip_flexion_ratio: float = .8

    def __post_init__(self):
        nonnegative = (self.close_rate, self.contact_threshold_n, self.retention_rate,
                       self.retention_extra, self.relief_rate,
                       self.thumb_mcp_flexion_ratio, self.thumb_ip_flexion_ratio)
        if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError('hand rates, contact threshold, and retention travel must be finite and nonnegative')
        if not math.isfinite(self.thumb_opposition_rad):
            raise ValueError('thumb opposition must be finite')
        for value in (self.maximum_force_n, self.retention_minimum_force_n):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('optional hand force thresholds must be finite and positive')
        if (self.maximum_force_n is not None and self.retention_minimum_force_n is not None
                and self.retention_minimum_force_n > self.maximum_force_n):
            raise ValueError('retention force target exceeds the relief threshold')


class HandGraspController:
    def __init__(self, model, config=HandConfig()):
        self.model = model
        self.config = config
        self.closure = {finger: 0. for finger in FINGERS}
        self.opposition = config.thumb_opposition_rad
        self.hold_reference = None
        self.engaged = set()
        self.slip_adjustments = 0
        self.actuators = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                          for i in range(model.nu)}
        for name, ratio in (('left_thumb_MCP_FE', config.thumb_mcp_flexion_ratio),
                            ('left_thumb_IP', config.thumb_ip_flexion_ratio)):
            actuator = self.actuators.get(name)
            if actuator is not None:
                joint = model.actuator_trnid[actuator, 0]
                if model.jnt_limited[joint] and ratio > model.jnt_range[joint, 1]:
                    raise ValueError(f'{name} full-closure target exceeds its model joint limit')

    def angles(self):
        result = {}
        for finger in FINGERS:
            c, prefix = self.closure[finger], 'left_' + finger + '_'
            if finger == 'thumb':
                result.update({prefix + 'CMC_FE': self.opposition, prefix + 'CMC_AA': 0.,
                               prefix + 'MCP_FE': self.config.thumb_mcp_flexion_ratio * c,
                               prefix + 'MCP_AA': 0., prefix + 'IP': self.config.thumb_ip_flexion_ratio * c})
            else:
                result.update({prefix + 'MCP_FE': c, prefix + 'MCP_AA': 0.,
                               prefix + 'PIP': 1.4 * c, prefix + 'DIP': .8 * c})
                if finger == 'pinky':
                    result[prefix + 'CMC'] = .12 * c
        return result

    def close(self, readings, dt):
        for finger in FINGERS:
            if readings[finger].normal > .2:
                self.engaged.add(finger)
            if self.relieve(finger, readings[finger], dt):
                continue
            if readings[finger].normal < self.config.contact_threshold_n:
                self.closure[finger] = min(1., self.closure[finger] + self.config.close_rate * dt)

    def relieve(self, finger, reading, dt):
        if self.config.maximum_force_n is not None and reading.normal > self.config.maximum_force_n:
            self.closure[finger] = max(0., self.closure[finger] - self.config.relief_rate * dt)
            return True
        return False

    def approach_fixed_targets(self, targets, dt):
        """Move toward a frozen per-finger posture without contact feedback."""
        if not isinstance(targets, Mapping) or set(targets) != set(FINGERS):
            raise ValueError('fixed targets must contain exactly the five named fingers')
        try:
            valid_targets = all(math.isfinite(value) and 0. <= value <= 1.
                                for value in targets.values())
            valid_rate = math.isfinite(self.config.close_rate) and self.config.close_rate >= 0.
            valid_dt = math.isfinite(dt) and dt >= 0.
        except TypeError as error:
            raise ValueError('fixed targets, rate, and dt must be finite numbers') from error
        if not valid_targets or not valid_rate or not valid_dt:
            raise ValueError('fixed targets must be finite in [0, 1]; rate and dt must be finite and nonnegative')
        maximum_step = self.config.close_rate * dt
        for finger in FINGERS:
            current, goal = self.closure[finger], targets[finger]
            self.closure[finger] = current + max(-maximum_step, min(maximum_step, goal - current))

    def retain(self, readings, dt):
        if self.hold_reference is None:
            self.hold_reference = self.closure.copy()
        for finger in self.engaged:
            force = readings[finger]
            if self.relieve(finger, force, dt):
                continue
            ratio = force.tangential / max(.1, force.normal)
            below_hold_target = (self.config.retention_minimum_force_n is not None
                                 and force.normal < self.config.retention_minimum_force_n)
            if below_hold_target or force.normal < 1. or (force.normal < 4. and ratio > .65):
                maximum = min(1., self.hold_reference[finger] + self.config.retention_extra)
                updated = max(self.closure[finger],
                              min(maximum, self.closure[finger] + self.config.retention_rate * dt))
                self.slip_adjustments += int(updated > self.closure[finger])
                self.closure[finger] = updated

    def table_support(self, data):
        """Vertical target load on the simulated table, including tangential force."""
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        wrench, force = np.zeros(6), 0.
        for index, contact in enumerate(data.contact[:data.ncon]):
            if {int(contact.geom1), int(contact.geom2)} == {target, table}:
                mujoco.mj_contactForce(self.model, data, index, wrench)
                # MuJoCo reports the force on geom2 in the contact frame.
                world_force = np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3]
                force += float(world_force[2]) * (1. if int(contact.geom2) == target else -1.)
        return max(0., force)

    def write(self, data):
        for name, angle in self.angles().items():
            aid = self.actuators[name]
            joint = self.model.actuator_trnid[aid, 0]
            data.ctrl[aid] = np.clip(angle, *self.model.jnt_range[joint])

    def initialize(self, data):
        """Only called at task reset; later motion uses actuators and physics."""
        for name, angle in self.angles().items():
            aid = self.actuators[name]
            joint = self.model.actuator_trnid[aid, 0]
            data.qpos[self.model.jnt_qposadr[joint]] = angle
        self.write(data)
