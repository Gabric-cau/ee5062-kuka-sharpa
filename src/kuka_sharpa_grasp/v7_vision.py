"""Known-appearance upright-cylinder localization from wrist RGB-D only.

The radius, upright staging and height are declared workcell priors. This is
an interpretable synthetic-object detector, not general tool pose estimation.
No segmentation buffer, body pose or target geometry ID enters this API.
"""

from dataclasses import dataclass

import numpy as np

from .perception import largest_component, point_cloud


@dataclass(frozen=True)
class CylinderDetection:
    object_id: str
    position_m: tuple[float, float, float] | None
    visible_points: int
    side_points: int
    fit_residual_m: float | None
    reason: str

    def as_dict(self):
        return dict(self.__dict__)


def fit_known_radius(points_xy, radius, prior_xy):
    """Fit a partially visible circular side using its declared radius."""
    points = np.asarray(points_xy, dtype=float)
    centre = np.asarray(prior_xy, dtype=float).copy()
    for _ in range(20):
        delta = centre - points
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        jacobian = delta / distance[:, None]
        if np.linalg.cond(jacobian.T @ jacobian) > 1e5:
            return None, None
        step = np.linalg.lstsq(jacobian, radius - distance, rcond=None)[0]
        norm = float(np.linalg.norm(step))
        if norm > .005:
            step *= .005 / norm
        centre += step
        if norm < 1e-7:
            break
    residual = float(np.sqrt(np.mean((np.linalg.norm(points - centre, axis=1) - radius) ** 2)))
    return centre, residual


def estimate_upright_cylinder(observation, spec, *, search_radius_m=.04):
    if spec.shape != "cylinder" or np.linalg.norm(np.asarray(spec.grasp_axis) - (0., 0., 1.)) > 1e-6:
        raise ValueError("detector requires a declared upright cylinder")
    cloud = point_cloud(observation)
    rgb = observation.rgb.astype(float)
    reference = np.asarray(spec.rgba[:3], dtype=float)
    dominant = int(np.argmax(reference))
    others = [index for index in range(3) if index != dominant]
    appearance = (rgb[..., dominant] > 20.)
    for index in others:
        appearance &= rgb[..., dominant] > 1.5 * rgb[..., index]
    prior = np.asarray(spec.pose)
    near = (np.linalg.norm(cloud[..., :2] - prior[:2], axis=-1) < search_radius_m)
    near &= (cloud[..., 2] > prior[2] - spec.dimensions[1] + .008)
    near &= (cloud[..., 2] < prior[2] + spec.dimensions[1] + .008)
    mask = largest_component(appearance & near & np.isfinite(cloud).all(axis=-1), cloud)
    visible = cloud[mask]
    if len(visible) < 30:
        return CylinderDetection(spec.object_id, None, len(visible), 0, None, "INSUFFICIENT_VISIBLE_POINTS")
    # Exclude the top face: only the cylindrical side lies at the known radius.
    top = prior[2] + spec.dimensions[1]
    sides = visible[(visible[:, 2] < top - .012) & (visible[:, 2] > top - 2 * spec.dimensions[1] + .012)]
    if len(sides) < 20:
        return CylinderDetection(spec.object_id, None, len(visible), len(sides), None, "SIDE_OCCLUDED")
    centre, residual = fit_known_radius(sides[:, :2], spec.dimensions[0], prior[:2])
    if centre is None or residual > .002 or np.linalg.norm(centre - prior[:2]) > search_radius_m:
        return CylinderDetection(spec.object_id, None, len(visible), len(sides), residual, "CYLINDER_FIT_REJECTED")
    return CylinderDetection(spec.object_id, (float(centre[0]), float(centre[1]), float(prior[2])),
                             len(visible), len(sides), residual, "RGBD_KNOWN_RADIUS_XY")
