"""Upright catalogue geometry priors; translation comes from wrist RGB-D."""
from dataclasses import dataclass
import numpy as np
from .pick_place_controller import candidates


@dataclass(frozen=True)
class GraspSpecification:
    object_kind: str
    aperture_offset: tuple
    minimum_visible_height: float
    bounds_min: tuple | None = None
    bounds_max: tuple | None = None

    def candidates(self, belief, now):
        if self.object_kind == 'ycb_can':
            return candidates(belief, now)
        if belief is None or belief.centroid is None or belief.confidence < .15:
            return [], ['NO_BELIEF']
        if now - belief.timestamp > .25 or now < belief.timestamp:
            return [], ['STALE_BELIEF']
        if belief.object_kind != self.object_kind or len(belief.points) < 20:
            return [], ['CATALOGUE_MISMATCH']
        bounds = np.quantile(belief.points[:, :2], [.01, .99], axis=0)
        static_center = (np.asarray(self.bounds_min) + self.bounds_max)[:2] / 2
        origin = bounds.mean(axis=0) - static_center
        if not (.55 < origin[0] < .85 and abs(origin[1]) < .2):
            return [], ['WORKSPACE']
        return [np.r_[origin + [dx, 0.], .38] for dx in (0., -.008, .008)], []


# Mesh-body bounds already measured in the catalogue study, not live object pose.
GRASP_SPECIFICATIONS = {
    'ycb_can': GraspSpecification('ycb_can', (0., .025, .045), .465),
    'ycb_potted_meat_can': GraspSpecification(
        'ycb_potted_meat_can', (-.00242397, .02941985, .035), .43,
        (-.04767312, -.03353193, -.00205196), (.04906213, .05490510, .08318695)),
    'ycb_cracker_box': GraspSpecification(
        'ycb_cracker_box', (.01920781, .00499887, .11876794), .54,
        (-.02372567, -.09723518, .00138138), (.06626104, .07054077, .23977416)),
}
