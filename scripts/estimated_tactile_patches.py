#!/usr/bin/env python3
"""Associate external tactile contacts with a public mesh at an estimated pose.

Only an isolated nominal mesh/probe model is queried. No live target geom ID,
position, segmentation, or inertial state is accepted by this interface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.physics import get_physics_profile
from kuka_sharpa_grasp.scene import VENDOR_ROOT, YCB_OBJECTS


class PublicMeshSurface:
    """Private convex collision surface and nominal mesh-local COM prior."""

    PROBE_RADIUS_M = 1e-6

    def __init__(self, model, *, source, mass_kg, friction_coefficient):
        self._model = model
        self._data = mujoco.MjData(model)
        self._surface = model.geom("public_surface").id
        self._probe = model.geom("association_probe_geom").id
        self._mocap = int(model.body_mocapid[model.body("association_probe").id])
        self.com_local_m = model.body_ipos[model.body("public_object").id].copy()
        self.mass_kg = float(mass_kg)
        self.friction_coefficient = float(friction_coefficient)
        self.source = dict(source)
        self.source["com_local_m"] = self.com_local_m.tolist()
        self.source["com_scope"] = "Compiled isolated public mesh at nominal zero body pose; not live target xipos"
        self.source["surface_scope"] = "MuJoCo convex collision surface of the declared public mesh"

    @classmethod
    def from_public_kind(cls, kind, mass_kg=None):
        if kind not in YCB_OBJECTS:
            raise ValueError(f"Unknown public mesh kind: {kind}")
        physics = get_physics_profile("physical-v6").object_parameters(kind)
        nominal_mass = physics.mass_kg if mass_kg is None else float(mass_kg)
        mesh_file = (VENDOR_ROOT / YCB_OBJECTS[kind]["mesh_file"]).resolve()
        return cls.from_mesh_file(mesh_file, mass_kg=nominal_mass,
            friction_coefficient=physics.friction[0], kind=kind)

    @classmethod
    def from_mesh_file(cls, mesh_file, *, mass_kg, friction_coefficient, kind=None):
        mesh_file = Path(mesh_file).resolve()
        if not np.isfinite(float(mass_kg)) or float(mass_kg) <= 0.:
            raise ValueError("mass_kg must be positive and finite")
        if not np.isfinite(float(friction_coefficient)) or float(friction_coefficient) < 0.:
            raise ValueError("friction_coefficient must be finite and nonnegative")
        root = ET.Element("mujoco")
        ET.SubElement(root, "option", gravity="0 0 0")
        assets = ET.SubElement(root, "asset")
        ET.SubElement(assets, "mesh", name="known_public_mesh", file=str(mesh_file))
        world = ET.SubElement(root, "worldbody")
        body = ET.SubElement(world, "body", name="public_object")
        ET.SubElement(body, "geom", name="public_surface", type="mesh",
            mesh="known_public_mesh", mass=str(float(mass_kg)))
        probe = ET.SubElement(world, "body", name="association_probe", mocap="true")
        ET.SubElement(probe, "geom", name="association_probe_geom", type="sphere",
            size=str(cls.PROBE_RADIUS_M))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        return cls(model, source={"kind": kind, "mesh_file": str(mesh_file),
            "mesh_sha256": hashlib.sha256(mesh_file.read_bytes()).hexdigest()},
            mass_kg=mass_kg, friction_coefficient=friction_coefficient)

    def query_surface(self, point_local_m):
        """Return signed surface distance and outward normal, including face interiors."""
        point = np.asarray(point_local_m, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Surface query requires a finite local point")
        self._data.mocap_pos[self._mocap] = point
        mujoco.mj_fwdPosition(self._model, self._data)
        segment = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(self._model, self._data,
            self._surface, self._probe, 10., segment))
        direction = segment[3:] - segment[:3]
        length = float(np.linalg.norm(direction))
        if not np.isfinite(distance) or length < 1e-12 or abs(distance) < 1e-12:
            return {"accepted": False, "reason": "UNDEFINED_SURFACE_NORMAL"}
        outward = direction / length * (1. if distance > 0. else -1.)
        return {"accepted": True, "signed_distance_m": distance + self.PROBE_RADIUS_M,
            "surface_point_local_m": segment[:3].copy(), "outward_normal_local": outward}


def associate_tactile_contacts(
    raw_contacts,
    pose_fit,
    surface,
    *,
    now_s,
    pose_timestamp_s,
    max_pose_age_s=.5,
    max_surface_distance_m=.006,
    min_normal_cosine=.5,
    max_unassociated_pressure_n=.5,
    max_unassociated_pressure_fraction=.15,
):
    """Conservatively associate contact geometry without external identity labels.

    ``raw_contacts`` contain only known hand-channel labels, world contact
    positions, force-on-hand unit normals, and pressure. Estimated geometric
    agreement is not proof of identity when an unknown object occupies the same
    surface. Significant unassociated external pressure rejects readiness.
    """
    result = {"accepted": False, "reason": "INVALID_INPUT", "patches": [],
        "estimated_com_world_m": None,
        "runtime_target_state_used": False,
        "claim_scope": "Geometric association against estimated public collision surface; not simulator identity association"}
    try:
        age = float(now_s) - float(pose_timestamp_s)
        limits = np.array([max_pose_age_s, max_surface_distance_m,
            max_unassociated_pressure_n, max_unassociated_pressure_fraction], dtype=float)
        if not np.isfinite(limits).all() or np.any(limits < 0.) or not np.isfinite(age):
            raise ValueError("Age and association limits must be finite and nonnegative")
        if not -1. <= min_normal_cosine <= 1.:
            raise ValueError("min_normal_cosine must lie in [-1, 1]")
        if not pose_fit.get("accepted", False):
            result["reason"] = "VISUAL_POSE_REJECTED"
            return result
        result["pose_age_s"], result["max_pose_age_s"] = age, float(max_pose_age_s)
        result["proprio_propagated_scope"] = pose_fit.get("proprio_propagated_scope")
        if age < -1e-6 or age > max_pose_age_s:
            result["reason"] = "STALE_OR_FUTURE_VISUAL_POSE"
            return result
        quality = {}
        for key, maximum in (("rms_m", .004), ("distance_p95_m", .008),
                ("near_best_origin_xy_spread_m", .005), ("near_best_origin_z_spread_m", .005)):
            if key not in pose_fit:
                result["reason"] = "MISSING_POSE_AMBIGUITY_EVIDENCE" if "spread" in key else "MISSING_POSE_QUALITY_EVIDENCE"
                result["missing_pose_field"] = key
                return result
            value = float(pose_fit[key])
            quality[key] = value
            if not np.isfinite(value) or value < 0. or value > maximum:
                result["reason"] = "LOW_CONFIDENCE_OR_AMBIGUOUS_VISUAL_POSE"
                result["rejected_pose_field"] = key
                return result
        origin = np.asarray(pose_fit["origin_m"], dtype=float)
        rotation = np.asarray(pose_fit["rotation_world_from_object"], dtype=float)
        if origin.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            raise ValueError("Pose requires finite translation and a 3x3 rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
            raise ValueError("Pose rotation is not a proper orthonormal matrix")
        diagnostics, patches = [], []
        total_pressure, unassociated_pressure = 0., 0.
        unassociated_channels = {}
        for index, contact in enumerate(raw_contacts):
            finger = contact["finger"]
            if finger not in ("thumb", "index", "middle", "ring", "pinky", "palm"):
                raise ValueError("Unknown hand channel")
            pressure = float(contact.get("pressure_n", contact.get("measured_pressure_n", 0.)))
            if not np.isfinite(pressure) or pressure < 0.:
                raise ValueError("Invalid tactile pressure")
            total_pressure += pressure
            point = np.asarray(contact["position_world_m"], dtype=float)
            hand_normal = np.asarray(contact["normal_on_hand_world"], dtype=float)
            if point.shape != (3,) or hand_normal.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(hand_normal).all():
                raise ValueError("Invalid tactile point or normal")
            length = float(np.linalg.norm(hand_normal))
            if not .5 <= length <= 1.5:
                raise ValueError("Tactile normal is not a usable unit direction")
            object_normal = -hand_normal / length
            query = surface.query_surface(rotation.T @ (point - origin))
            row = {"input_index": index, "finger": finger, "pressure_n": pressure}
            accepted = bool(query["accepted"])
            reason = query.get("reason", "ASSOCIATED")
            if accepted:
                distance = float(query["signed_distance_m"])
                inward = -(rotation @ np.asarray(query["outward_normal_local"]))
                cosine = float(np.clip(inward @ object_normal, -1., 1.))
                row.update(signed_surface_distance_m=distance, inward_normal_cosine=cosine)
                if abs(distance) > max_surface_distance_m:
                    accepted, reason = False, "OUTSIDE_ESTIMATED_SURFACE_BAND"
                elif cosine < min_normal_cosine:
                    accepted, reason = False, "CONTACT_NORMAL_DISAGREES_WITH_ESTIMATED_SURFACE"
            if accepted:
                friction = float(contact.get("friction_coefficient", surface.friction_coefficient))
                if not np.isfinite(friction) or friction < 0.:
                    raise ValueError("Invalid declared tactile friction prior")
                patches.append({"finger": finger, "position_world_m": point.tolist(),
                    "normal_on_object_world": object_normal.tolist(),
                    "measured_pressure_n": pressure, "friction_coefficient": friction,
                    "input_index": index})
            else:
                unassociated_pressure += pressure
                unassociated_channels[finger] = unassociated_channels.get(finger, 0.) + pressure
            row.update(accepted=accepted, reason=reason)
            diagnostics.append(row)
        fraction = unassociated_pressure / total_pressure if total_pressure > 0. else 0.
        contaminated = (unassociated_pressure > max_unassociated_pressure_n
            and fraction > max_unassociated_pressure_fraction)
        accepted = bool(patches) and not contaminated
        result.update({"accepted": accepted,
            "reason": "ASSOCIATED" if accepted else (
                "SIGNIFICANT_UNASSOCIATED_EXTERNAL_PRESSURE" if contaminated else "NO_ASSOCIATED_TACTILE_PATCHES"),
            "patches": patches,
            "estimated_com_world_m": (origin + rotation @ surface.com_local_m).tolist(),
            "pose_quality": quality,
            "public_mesh_prior": surface.source,
            "contact_association": diagnostics,
            "associated_pressure_n": total_pressure - unassociated_pressure,
            "unassociated_pressure_n": unassociated_pressure,
            "unassociated_pressure_fraction": fraction,
            "unassociated_channel_pressures_n": unassociated_channels,
            "max_surface_distance_m": float(max_surface_distance_m),
            "min_normal_cosine": float(min_normal_cosine),
        })
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["details"] = str(error)
        return result
