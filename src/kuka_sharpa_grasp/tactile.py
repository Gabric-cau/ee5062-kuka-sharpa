"""Target-independent tactile abstractions for the V7 manipulation tasks.

The V6 grasp reader deliberately answers a narrower question: contact between
the known ``target_geom`` and the hand.  That is useful for the original
single-object benchmark, but it is an object-state oracle in a cluttered
scene.  :class:`TactileContactReader` instead classifies contacts from the
hand geometry side only.  The other geometry may be any object, obstacle, or
fixture and is never inspected for a target name.

The reader uses MuJoCo's native contact wrench.  Optional MuJoCo touch sensors
can be supplied as an additional per-finger signal; they are reported
separately and only used as a fallback when native contacts are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np

from .control import FINGERS


@dataclass(frozen=True)
class FingerTactileReading:
    """One polling result for one finger or the palm.

    ``slip_score`` is a dimensionless contact heuristic in ``[0, 1]``.  It is
    the tangential-force share of the measured wrench, rather than a claim of
    calibrated material friction.  ``centroid_world`` is ``None`` when no
    native contact was observed.
    """

    normal_force_n: float = 0.0
    tangential_force_n: float = 0.0
    contacts: int = 0
    centroid_world: tuple[float, float, float] | None = None
    slip_score: float = 0.0
    touch_sensor_force_n: float | None = None

    @property
    def total_force_n(self) -> float:
        return float(math.hypot(self.normal_force_n, self.tangential_force_n))

    @property
    def in_contact(self) -> bool:
        return self.contacts > 0 or (self.touch_sensor_force_n or 0.0) > 0.0


@dataclass(frozen=True)
class TactileSnapshot:
    """Immutable aggregate of one tactile poll."""

    timestamp: float
    fingers: Mapping[str, FingerTactileReading]

    @property
    def total_normal_force_n(self) -> float:
        return float(sum(reading.normal_force_n for reading in self.fingers.values()))

    @property
    def total_tangential_force_n(self) -> float:
        return float(sum(reading.tangential_force_n for reading in self.fingers.values()))

    @property
    def contact_count(self) -> int:
        return int(sum(reading.contacts for reading in self.fingers.values()))

    def as_dict(self) -> dict:
        """Return a JSON-compatible representation for traces and replay logs."""
        return {
            "timestamp": float(self.timestamp),
            "fingers": {
                name: {
                    "normal_force_n": float(value.normal_force_n),
                    "tangential_force_n": float(value.tangential_force_n),
                    "contacts": int(value.contacts),
                    "centroid_world": list(value.centroid_world) if value.centroid_world is not None else None,
                    "slip_score": float(value.slip_score),
                    "touch_sensor_force_n": (None if value.touch_sensor_force_n is None
                                               else float(value.touch_sensor_force_n)),
                }
                for name, value in self.fingers.items()
            },
            "total_normal_force_n": self.total_normal_force_n,
            "total_tangential_force_n": self.total_tangential_force_n,
            "contact_count": self.contact_count,
        }


def _finger_from_body_name(body_name: str, prefix: str) -> str | None:
    """Return a canonical finger name for a Sharpa hand body."""
    if not body_name.startswith(prefix):
        return None
    remainder = body_name[len(prefix):]
    for finger in FINGERS:
        if remainder.startswith(finger + "_") or remainder == finger:
            return finger
    return None


class TactileContactReader:
    """Read per-finger contacts without a target-object geometry oracle.

    Geometries are assigned to fingers by their body names (``left_index_*``
    etc.), matching the vendor Sharpa model.  A contact is included if exactly
    one side belongs to the hand.  Contacts between hand parts are ignored;
    contacts to every other geometry are retained, including obstacles and
    multiple simultaneous objects.

    ``touch_sensor_names`` maps finger names to optional MuJoCo touch sensor
    names.  Native contacts remain the primary signal.  If a configured touch
    sensor reports a positive signal while no native contact is present, that
    signal is exposed in ``touch_sensor_force_n`` and ``in_contact`` becomes
    true, without pretending that the sensor identifies an object.
    """

    def __init__(self, model: mujoco.MjModel, *, hand_body_prefix: str = "left_",
                 include_palm: bool = True,
                 touch_sensor_names: Mapping[str, str] | None = None,
                 slip_epsilon_n: float = 1e-6) -> None:
        if not hand_body_prefix or not isinstance(hand_body_prefix, str):
            raise ValueError("hand_body_prefix must be a non-empty string")
        if not math.isfinite(slip_epsilon_n) or slip_epsilon_n <= 0:
            raise ValueError("slip_epsilon_n must be finite and positive")
        self.model = model
        self.hand_body_prefix = hand_body_prefix
        self.include_palm = bool(include_palm)
        self.slip_epsilon_n = float(slip_epsilon_n)
        self.fingers = tuple(FINGERS) + (("palm",) if include_palm else ())
        self._geom_fingers: dict[int, str] = {}
        self._hand_geom_ids: set[int] = set()
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            finger = _finger_from_body_name(body_name, hand_body_prefix)
            if body_name.startswith(hand_body_prefix):
                self._hand_geom_ids.add(geom_id)
            if finger is not None:
                self._geom_fingers[geom_id] = finger
            elif body_name.startswith(hand_body_prefix) and include_palm:
                # Hand bodies without a named finger are palm/wrist geometry.
                self._geom_fingers[geom_id] = "palm"

        self._touch_sensor_ids: dict[str, int] = {}
        for finger, sensor_name in dict(touch_sensor_names or {}).items():
            if finger not in self.fingers:
                raise ValueError(f"unknown tactile channel: {finger!r}")
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise ValueError(f"touch sensor missing: {sensor_name}")
            if model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_TOUCH:
                raise ValueError(f"sensor is not a MuJoCo touch sensor: {sensor_name}")
            self._touch_sensor_ids[finger] = int(sensor_id)

    @property
    def geometry_channels(self) -> Mapping[int, str]:
        """Read-only geometry-to-finger mapping for diagnostics."""
        return MappingProxyType(dict(self._geom_fingers))

    def _touch_value(self, data: mujoco.MjData, sensor_id: int) -> float:
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        values = np.asarray(data.sensordata[address:address + dimension], dtype=float)
        return float(np.sum(np.maximum(values, 0.0)))

    def read(self, data: mujoco.MjData) -> TactileSnapshot:
        values: dict[str, dict] = {
            finger: {"normal": 0.0, "tangent": 0.0, "contacts": 0,
                     "positions": [], "weights": [], "sensor": None}
            for finger in self.fingers
        }
        wrench = np.zeros(6, dtype=float)
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            channel_a = self._geom_fingers.get(geom_a)
            channel_b = self._geom_fingers.get(geom_b)
            if ((geom_a in self._hand_geom_ids) == (geom_b in self._hand_geom_ids)
                    or (channel_a is None and channel_b is None)):
                # Ignore hand-hand contacts and contacts that do not involve
                # the hand.  The non-hand side is deliberately not inspected.
                continue
            channel = channel_a if channel_a is not None else channel_b
            if channel == "palm" and not self.include_palm:
                continue
            mujoco.mj_contactForce(self.model, data, contact_id, wrench)
            normal = abs(float(wrench[0]))
            tangent = float(np.linalg.norm(wrench[1:3]))
            weight = max(normal + tangent, 1e-12)
            item = values[channel]
            item["normal"] += normal
            item["tangent"] += tangent
            item["contacts"] += 1
            item["positions"].append(np.asarray(contact.pos, dtype=float).copy())
            item["weights"].append(weight)

        for finger, sensor_id in self._touch_sensor_ids.items():
            values[finger]["sensor"] = self._touch_value(data, sensor_id)

        readings: dict[str, FingerTactileReading] = {}
        for finger in self.fingers:
            item = values[finger]
            normal, tangent = float(item["normal"]), float(item["tangent"])
            if item["positions"]:
                centroid = np.average(np.asarray(item["positions"]), axis=0,
                                      weights=np.asarray(item["weights"]))
                centroid_world = tuple(float(x) for x in centroid)
            else:
                centroid_world = None
            slip_score = float(np.clip(tangent / max(normal + tangent, self.slip_epsilon_n), 0., 1.))
            readings[finger] = FingerTactileReading(
                normal_force_n=normal,
                tangential_force_n=tangent,
                contacts=int(item["contacts"]),
                centroid_world=centroid_world,
                slip_score=slip_score,
                touch_sensor_force_n=item["sensor"],
            )
        return TactileSnapshot(float(data.time), MappingProxyType(readings))


__all__ = ["FingerTactileReading", "TactileSnapshot", "TactileContactReader"]
