"""Known-object visible-surface belief from RGB-D, without simulator state.

Uses calibrated horizontal table and workspace bounds, plus the authored red
object appearance to reject the robot. This simulation appearance prior is not
a real-camera detector. Centroid describes visible surface, not mesh origin.
"""
from dataclasses import dataclass
import numpy as np

KNOWN_OBJECTS = ("ycb_can", "ycb_potted_meat_can", "ycb_cracker_box")


@dataclass(frozen=True)
class ObjectBelief:
    object_kind: str
    centroid: np.ndarray | None
    principal_axis: np.ndarray | None
    yaw_observable: bool
    confidence: float
    timestamp: float
    points: np.ndarray
    pixel_mask: np.ndarray
    reason: str


def point_cloud(observation):
    h, w = observation.depth.shape
    v, u = np.indices((h, w))
    z = observation.depth
    local = np.stack(((u - (w - 1) / 2) * z / observation.focal_length,
                      ((h - 1) / 2 - v) * z / observation.focal_length, -z), axis=-1)
    return local @ observation.camera_rotation.T + observation.camera_position


def above_table(points, table_height=.38, tolerance=.008):
    return np.isfinite(points).all(axis=-1) & (points[..., 2] > table_height + tolerance)


def largest_component(mask, points, max_gap=.025):
    """4-connected pixels with a metric depth edge gate; discard occlusion jumps."""
    remaining = mask.copy()
    best = []
    h, w = mask.shape
    for row, col in zip(*np.where(mask)):
        if not remaining[row, col]:
            continue
        stack = [(row, col)]
        remaining[row, col] = False
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and remaining[yy, xx]:
                    if np.linalg.norm(points[yy, xx] - points[y, x]) <= max_gap:
                        remaining[yy, xx] = False
                        stack.append((yy, xx))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(mask)
    if best:
        y, x = np.array(best).T
        result[y, x] = True
    return result


def estimate_object(observation, object_kind, *, table_height=.38):
    if object_kind not in KNOWN_OBJECTS:
        raise ValueError("only catalogued YCB objects are supported")
    points = point_cloud(observation)
    rgb = observation.rgb.astype(float)
    appearance = (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 1.5 * rgb[..., 2])
    workspace = ((points[..., 0] > .45) & (points[..., 0] < 1.05)
                 & (np.abs(points[..., 1]) < .35) & (points[..., 2] < .70))
    candidate = above_table(points, table_height) & workspace & appearance
    mask = largest_component(candidate, points)
    cloud = points[mask]
    observable = object_kind != "ycb_can"
    if len(cloud) < 20:
        return ObjectBelief(object_kind, None, None, observable, 0., observation.timestamp,
                            cloud, mask, "insufficient visible connected points")
    centroid = cloud.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(cloud.T))
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    confidence = float(min(1., len(cloud) / 500) * mask.sum() / max(1, candidate.sum()))
    # PCA on a partial surface is a geometric cue, not a registered 6-D pose.
    return ObjectBelief(object_kind, centroid, axis, observable, confidence,
                        observation.timestamp, cloud, mask, "visible-surface estimate")
