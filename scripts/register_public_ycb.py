#!/usr/bin/env python3
"""Known-upright public-mesh registration from actual wrist RGB-D points.

MuJoCo imports STL geometry; SciPy provides nearest-neighbor search and bounded
least squares. Runtime estimation never reads target pose or segmentation IDs.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.catalogue import GraspSpecification
from kuka_sharpa_grasp.perception import ObjectBelief, estimate_object
from kuka_sharpa_grasp.scene import VENDOR_ROOT, YCB_OBJECTS


@dataclass(frozen=True)
class RegisteredBelief(ObjectBelief):
    registered_origin_m: tuple | None = None
    registration_rms_m: float | None = None


@dataclass(frozen=True)
class RegisteredSpecification(GraspSpecification):
    def candidates(self, belief, now):
        if belief is None or belief.confidence < .15 or len(belief.points) < 20:
            return [], ["NO_BELIEF"]
        if now - belief.timestamp > .25 or now < belief.timestamp:
            return [], ["STALE_BELIEF"]
        if belief.object_kind != self.object_kind:
            return [], ["CATALOGUE_MISMATCH"]
        origin = getattr(belief, "registered_origin_m", None)
        if origin is None:
            return super().candidates(belief, now)
        if not (.55 < origin[0] < .85 and abs(origin[1]) < .2):
            return [], ["WORKSPACE"]
        return [np.r_[np.asarray(origin[:2]) + [dx, 0.], .38] for dx in (0., -.008, .008)], []


class NominalMeshRegistration:
    def __init__(self, kind, max_tilt_rad=0.):
        self.kind = kind
        if not np.isfinite(max_tilt_rad) or not 0. <= max_tilt_rad <= np.deg2rad(5.):
            raise ValueError("Nominal upright tilt allowance must be 0-5 degrees")
        self.max_tilt_rad = float(max_tilt_rad)
        mesh_file = (VENDOR_ROOT / YCB_OBJECTS[kind]["mesh_file"]).resolve()
        root = ET.Element("mujoco")
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="public_mesh", file=str(mesh_file))
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="nominal", type="mesh", mesh="public_mesh")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        gid, mid = model.geom("nominal").id, 0
        vertices = np.asarray(model.mesh_vert[
            model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]], dtype=float)
        faces = np.asarray(model.mesh_face[
            model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]], dtype=int)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
        centroids = vertices[faces].mean(axis=1)
        self.points = np.concatenate([vertices, centroids], axis=0)
        self.tree = cKDTree(self.points)
        self.bounds_min, self.bounds_max = vertices.min(axis=0), vertices.max(axis=0)

    def fit(self, points, *, base_height=.38):
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 20:
            return {"accepted": False, "reason": "INSUFFICIENT_VISIBLE_POINTS"}
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) < 20:
            return {"accepted": False, "reason": "INSUFFICIENT_FINITE_POINTS"}
        if len(cloud) > 800:
            cloud = cloud[np.linspace(0, len(cloud) - 1, 800, dtype=int)]
        visible_center = np.quantile(cloud[:, :2], [.01, .99], axis=0).mean(axis=0)
        nominal_center = (self.bounds_min[:2] + self.bounds_max[:2]) / 2
        seed = np.r_[visible_center - nominal_center, base_height]
        lower, upper = np.array([.55, -.2, base_height - .006]), np.array([.85, .2, base_height + .006])
        if self.max_tilt_rad:
            seed = np.r_[seed, 0., 0.]
            lower = np.r_[lower, -self.max_tilt_rad, -self.max_tilt_rad]
            upper = np.r_[upper, self.max_tilt_rad, self.max_tilt_rad]
        seed = np.clip(seed, lower + 1e-6, upper - 1e-6)

        def residual(parameters):
            translation = parameters[:3]
            rotation = (Rotation.from_rotvec([*parameters[3:], 0.]).as_matrix()
                        if self.max_tilt_rad else np.eye(3))
            _, nearest = self.tree.query((cloud - translation) @ rotation, workers=1)
            return (cloud - translation - self.points[nearest] @ rotation.T).ravel()

        solutions = []
        for y_offset in (0., -.012, .012):
            offset = np.zeros_like(seed)
            offset[1] = y_offset
            start = np.clip(seed + offset, lower + 1e-6, upper - 1e-6)
            fit = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1",
                f_scale=.0015, max_nfev=80, xtol=1e-8, ftol=1e-8, gtol=1e-8,
                x_scale=np.array([.01, .01, .004, .02, .02]) if self.max_tilt_rad else 1.)
            distances = np.linalg.norm(residual(fit.x).reshape(-1, 3), axis=1)
            solutions.append((float(np.mean(distances**2)), fit, distances))
        _, fit, distances = min(solutions, key=lambda item: item[0])
        rms, p95 = float(np.sqrt(np.mean(distances**2))), float(np.percentile(distances, 95))
        accepted = bool(fit.success and rms <= .004 and p95 <= .008)
        return {"accepted": accepted, "reason": "REGISTERED" if accepted else "SURFACE_RESIDUAL_REJECTED",
            "origin_m": fit.x[:3].tolist(), "rms_m": rms, "distance_p95_m": p95,
            "estimated_tilt_rotvec_rad": [*fit.x[3:], 0.] if self.max_tilt_rad else [0., 0., 0.],
            "point_count_used": len(cloud), "mesh_samples": len(self.points),
            "optimizer_success": bool(fit.success), "optimizer_evaluations": int(fit.nfev),
            "priors": {"known_public_identity": self.kind, "nominal_zero_yaw_upright": True,
                "rotation_vector_xy_allowance_rad": self.max_tilt_rad,
                "rotation_vector_z_fixed_rad": 0.,
                "body_base_height_m": base_height, "height_allowance_m": .006,
                "known_red_appearance": True},
            "input_contract": "Visible RGB-D surface points and static public mesh only; no live body pose or simulator segmentation"}

    def estimate(self, observation):
        belief = estimate_object(observation, self.kind)
        result = self.fit(belief.points)
        return RegisteredBelief(**vars(belief),
            registered_origin_m=tuple(result["origin_m"]) if result["accepted"] else None,
            registration_rms_m=result.get("rms_m")), result
