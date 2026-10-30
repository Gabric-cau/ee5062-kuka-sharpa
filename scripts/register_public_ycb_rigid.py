#!/usr/bin/env python3
"""Bounded initial pose and multistart held-pose fitting from visible RGB-D."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from register_public_ycb import NominalMeshRegistration


class VisibleMeshPoseRegistration(NominalMeshRegistration):
    def fit(self, points, *, initial=False, translation_seed=None):
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            return {"accepted": False, "reason": "INVALID_POINT_ARRAY"}
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) < 100:
            return {"accepted": False, "reason": "INSUFFICIENT_VISIBLE_POINTS", "point_count": len(cloud)}
        if len(cloud) > 800:
            cloud = cloud[np.linspace(0, len(cloud) - 1, 800, dtype=int)]
        center = np.quantile(cloud, [.01, .99], axis=0).mean(axis=0)
        local_center = (self.bounds_min + self.bounds_max) / 2.
        allowance = np.deg2rad(3.)
        if initial:
            lower = np.array([.55, -.2, .374, -allowance, -allowance])
            upper = np.array([.85, .2, .386, allowance, allowance])
            seeds = [np.r_[center[:2] - local_center[:2], .38, rx, ry]
                for rx, ry in ((0., 0.), (.025, .025), (-.025, -.025))]
            scales = np.array([.01, .01, .004, .02, .02])
        else:
            lower = np.array([.3, -.45, .30, -np.pi, -np.pi, -np.pi])
            upper = np.array([1.05, .45, .95, np.pi, np.pi, np.pi])
            seeds = []
            for yaw in np.deg2rad([0., -30., 30., -60., 60., -90., 90.]):
                rotvec = np.array([0., 0., yaw])
                translation = (center - Rotation.from_rotvec(rotvec).apply(local_center)
                    if translation_seed is None else np.asarray(translation_seed))
                seeds.append(np.r_[translation, rotvec])
            if translation_seed is not None:
                seeds.append(np.r_[center - local_center, 0., 0., 0.])
            scales = np.array([.01, .01, .01, .1, .1, .1])

        def rotation(parameters):
            return Rotation.from_rotvec(np.r_[parameters[3:], 0.] if initial else parameters[3:]).as_matrix()

        def residual(parameters):
            matrix = rotation(parameters)
            _, nearest = self.tree.query((cloud - parameters[:3]) @ matrix, workers=1)
            return (cloud - parameters[:3] - self.points[nearest] @ matrix.T).ravel()

        solutions = []
        for seed in seeds:
            fitted = least_squares(residual, np.clip(seed, lower + 1e-6, upper - 1e-6),
                bounds=(lower, upper), loss="soft_l1", f_scale=.0015, max_nfev=200,
                xtol=1e-8, ftol=1e-8, gtol=1e-8, x_scale=scales)
            distances = np.linalg.norm(residual(fitted.x).reshape(-1, 3), axis=1)
            solutions.append((float(np.sqrt(np.mean(distances ** 2))), fitted, distances))
        rms, fitted, distances = min(solutions, key=lambda row: row[0])
        near = [row for row in solutions if row[0] <= rms + .00025]
        origins = np.asarray([row[1].x[:3] for row in near])
        spread = float(max((np.linalg.norm(a[:2] - b[:2]) for a in origins for b in origins), default=0.))
        z_spread = float(np.ptp(origins[:, 2]))
        p95 = float(np.percentile(distances, 95))
        accepted = bool(fitted.success and rms <= .004 and p95 <= .008 and spread <= .005)
        reason = "REGISTERED" if accepted else ("AMBIGUOUS_ORIGIN" if spread > .005 else "SURFACE_RESIDUAL_OR_OPTIMIZER_REJECTED")
        return {"accepted": accepted, "reason": reason, "origin_m": fitted.x[:3].tolist(),
            "rotation_world_from_object": rotation(fitted.x).tolist(),
            "estimated_rotvec_rad": (np.r_[fitted.x[3:], 0.] if initial else fitted.x[3:]).tolist(),
            "rms_m": rms, "distance_p95_m": p95, "near_best_origin_xy_spread_m": spread,
            "near_best_origin_z_spread_m": z_spread,
            "proprioception_translation_seed_m": None if translation_seed is None else np.asarray(translation_seed).tolist(),
            "held_control_uses_xy_only": not initial,
            "point_count_used": len(cloud), "mesh_samples": len(self.points),
            "optimizer_success": bool(fitted.success), "optimizer_evaluations": int(fitted.nfev),
            "starts": [{"rms_m": row[0], "origin_m": row[1].x[:3].tolist(),
                "rotvec_rad": (np.r_[row[1].x[3:], 0.] if initial else row[1].x[3:]).tolist(),
                "optimizer_success": bool(row[1].success), "evaluations": int(row[1].nfev)} for row in solutions],
            "priors": {"known_public_identity": self.kind, "initial": initial,
                "initial_base_height_m": .38 if initial else None,
                "initial_rotvec_xy_allowance_rad": float(allowance) if initial else None,
                "initial_rotvec_z_fixed": 0. if initial else None,
                "held_rotation_free": not initial, "known_red_appearance": True},
            "input_contract": "Visible RGB-D surface points and public STL only; no target simulator state or segmentation IDs",
            "ambiguity_scope": "Near-best multistart origin spread is a rejection heuristic, not a proof of pose observability"}
