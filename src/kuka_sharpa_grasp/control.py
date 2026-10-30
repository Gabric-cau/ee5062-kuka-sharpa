"""Low-dimensional force and grasp controllers for the Sharpa hand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class ForceReading:
    normal: float = 0.0
    tangential: float = 0.0
    contacts: int = 0


class ContactForceReader:
    """Aggregate simulated fingertip contact forces against the target object."""

    def __init__(self, model: mujoco.MjModel, target_geom: str = "target_geom") -> None:
        self.model = model
        self.target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom)
        self.fingertip_geom_ids = {
            gid for gid in range(model.ngeom)
            if "fingertip_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
        }
        self.finger_ids = {
            finger: {
                gid for gid in self.fingertip_geom_ids
                if finger in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            }
            for finger in FINGERS
        }

    def read(self, data: mujoco.MjData) -> dict[str, ForceReading]:
        readings = {finger: ForceReading() for finger in FINGERS}
        wrench = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            if self.target_geom_id not in (geom_a, geom_b):
                continue
            fingertip_id = geom_b if geom_a == self.target_geom_id else geom_a
            if fingertip_id not in self.fingertip_geom_ids:
                continue
            mujoco.mj_contactForce(self.model, data, contact_id, wrench)
            force = np.abs(wrench[:3])
            normal = float(force[0])
            tangential = float(np.linalg.norm(force[1:]))
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, fingertip_id) or ""
            finger = next((name for name in FINGERS if name in geom_name), None)
            if finger is not None:
                reading = readings[finger]
                reading.normal += normal
                reading.tangential += tangential
                reading.contacts += 1
        return readings

    @staticmethod
    def total(readings: dict[str, ForceReading]) -> ForceReading:
        return ForceReading(
            normal=sum(item.normal for item in readings.values()),
            tangential=sum(item.tangential for item in readings.values()),
            contacts=sum(item.contacts for item in readings.values()),
        )


class OuterForceController:
    """Position-interface force loop with an independent per-finger target."""

    def __init__(self, target_force: float = 1.5, kp: float = 0.0015, ki: float = 0.00008,
                 max_step: float = 0.002, max_integral: float = 3.0) -> None:
        # Each finger is regulated independently. ``target_force`` remains the
        # public name for compatibility with the trace CSV and old notebooks.
        self.target_force = float(target_force)
        self.kp = kp
        self.ki = ki
        self.max_step = max_step
        self.max_integral = max_integral
        self.integral = {finger: 0.0 for finger in FINGERS}

    def reset(self) -> None:
        for finger in FINGERS:
            self.integral[finger] = 0.0

    def update(self, readings: dict[str, ForceReading], current_targets: dict[str, float],
               dt: float, active: Iterable[str] = FINGERS) -> dict[str, float]:
        updated = dict(current_targets)
        for finger in active:
            error = self.target_force - readings[finger].normal
            self.integral[finger] = float(np.clip(self.integral[finger] + error * dt, -self.max_integral, self.max_integral))
            delta = np.clip(self.kp * error + self.ki * self.integral[finger], -self.max_step, self.max_step)
            updated[finger] = float(np.clip(updated[finger] + delta, 0.0, 1.0))
        return updated


def hand_synergy_targets(close: float, thumb: float | None = None) -> dict[str, float]:
    """Map one scalar to safe normalized finger targets; no 22-DoF learning is required."""
    close = float(np.clip(close, 0.0, 1.0))
    return {finger: float(np.clip(thumb if finger == "thumb" and thumb is not None else close, 0.0, 1.0)) for finger in FINGERS}
