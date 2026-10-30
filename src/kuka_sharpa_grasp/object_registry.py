"""V7 workbench object registry and RGB-D-compatible target selection.

The registry is deliberately independent of MuJoCo and perception pixels.  A
detector can return :class:`VisibleDetection` records, and the controller can
resolve those records to a declared object specification without consulting
simulator state.  The primitive tool geometries are engineering proxies; the
source field keeps that distinction explicit in experiment manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SHAPES = frozenset({"box", "cylinder", "capsule", "mesh"})
_ROLES = frozenset({"target", "distractor"})


def _finite_tuple(values: Iterable[float], length: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class ToolObjectSpec:
    """Physical and semantic contract for one workbench object.

    ``dimensions`` are half-extents for ``box`` and ``(radius, half_length)``
    for ``cylinder``/``capsule``.  A mesh uses the dimensions only as a
    conservative collision bound; the actual mesh is identified separately.
    ``pose`` is the free-joint body origin in world coordinates.
    """

    object_id: str
    label: str
    category: str
    role: str
    shape: str
    dimensions: tuple[float, ...]
    mass_kg: float
    friction: tuple[float, float, float]
    pose: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    source: str
    grasp_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    dock_slot: str | None = None
    collision_class: str = "workbench_object"
    mesh_name: str | None = None
    mesh_file: str | None = None

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.object_id):
            raise ValueError(f"invalid object_id: {self.object_id!r}")
        if not self.label.strip() or not self.category.strip():
            raise ValueError("label and category must be non-empty")
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if self.shape not in _SHAPES:
            raise ValueError(f"shape must be one of {sorted(_SHAPES)}")
        dimensions = tuple(float(value) for value in self.dimensions)
        expected = 3 if self.shape == "box" else 2 if self.shape in {"cylinder", "capsule"} else 3
        if len(dimensions) != expected or not all(math.isfinite(value) and value > 0 for value in dimensions):
            raise ValueError(f"dimensions must contain {expected} positive finite values for {self.shape}")
        if not math.isfinite(float(self.mass_kg)) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")
        friction = _finite_tuple(self.friction, 3, "friction")
        if any(value <= 0 for value in friction):
            raise ValueError("friction values must be positive")
        pose = _finite_tuple(self.pose, 3, "pose")
        rgba = _finite_tuple(self.rgba, 4, "rgba")
        if any(value < 0 or value > 1 for value in rgba):
            raise ValueError("rgba values must be in [0, 1]")
        axis = _finite_tuple(self.grasp_axis, 3, "grasp_axis")
        norm = math.sqrt(sum(value * value for value in axis))
        if norm < 1e-9:
            raise ValueError("grasp_axis must be non-zero")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.shape == "mesh" and not self.mesh_name:
            raise ValueError("mesh objects require mesh_name")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "rgba", rgba)
        object.__setattr__(self, "grasp_axis", axis)

    @property
    def actionable(self) -> bool:
        return self.role == "target"

    def manifest(self) -> dict[str, object]:
        """Return JSON-safe metadata for an experiment manifest."""
        return {
            "object_id": self.object_id,
            "label": self.label,
            "category": self.category,
            "role": self.role,
            "shape": self.shape,
            "dimensions": list(self.dimensions),
            "mass_kg": self.mass_kg,
            "friction": list(self.friction),
            "pose": list(self.pose),
            "rgba": list(self.rgba),
            "grasp_axis": list(self.grasp_axis),
            "dock_slot": self.dock_slot,
            "collision_class": self.collision_class,
            "source": self.source,
            "mesh_name": self.mesh_name,
            "mesh_file": self.mesh_file,
        }


@dataclass(frozen=True)
class VisibleDetection:
    """A detector-side RGB-D record used by :class:`ToolObjectRegistry`."""

    object_id: str
    centroid_xyz: tuple[float, float, float]
    confidence: float
    mask_area: int = 0
    yaw: float | None = None
    source: str = "rgbd"

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.object_id):
            raise ValueError(f"invalid detection object_id: {self.object_id!r}")
        centroid = _finite_tuple(self.centroid_xyz, 3, "centroid_xyz")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if int(self.mask_area) < 0:
            raise ValueError("mask_area must be non-negative")
        if self.yaw is not None and not math.isfinite(float(self.yaw)):
            raise ValueError("yaw must be finite when supplied")
        object.__setattr__(self, "centroid_xyz", centroid)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask_area", int(self.mask_area))


@dataclass(frozen=True)
class VisibleMatch:
    """A resolved pair of declared object metadata and a detector record."""

    spec: ToolObjectSpec
    detection: VisibleDetection


class ToolObjectRegistry:
    """Deterministic object lookup and visibility filtering for V7."""

    def __init__(self, objects: Iterable[ToolObjectSpec]) -> None:
        values = tuple(objects)
        ids = [item.object_id for item in values]
        if len(set(ids)) != len(ids):
            raise ValueError("object_id values must be unique")
        if not values:
            raise ValueError("registry must contain at least one object")
        self._objects = values
        self._by_id = {item.object_id: item for item in values}

    @property
    def objects(self) -> tuple[ToolObjectSpec, ...]:
        return self._objects

    @property
    def actionable(self) -> tuple[ToolObjectSpec, ...]:
        return tuple(item for item in self._objects if item.actionable)

    @property
    def distractors(self) -> tuple[ToolObjectSpec, ...]:
        return tuple(item for item in self._objects if not item.actionable)

    def by_id(self, object_id: str) -> ToolObjectSpec:
        try:
            return self._by_id[object_id]
        except KeyError as error:
            raise KeyError(f"unknown workbench object: {object_id!r}") from error

    @staticmethod
    def _coerce_detection(value: VisibleDetection | Mapping[str, object]) -> VisibleDetection:
        if isinstance(value, VisibleDetection):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("detections must be VisibleDetection or mapping records")
        return VisibleDetection(
            object_id=str(value["object_id"]),
            centroid_xyz=tuple(value["centroid_xyz"]),
            confidence=float(value["confidence"]),
            mask_area=int(value.get("mask_area", 0)),
            yaw=None if value.get("yaw") is None else float(value["yaw"]),
            source=str(value.get("source", "rgbd")),
        )

    def match_visible(
        self,
        detections: Iterable[VisibleDetection | Mapping[str, object]],
        *,
        min_confidence: float = 0.15,
        actionable_only: bool = True,
    ) -> tuple[VisibleMatch, ...]:
        """Resolve known detections, retaining the strongest one per object.

        Unknown detector IDs are ignored so a detector may expose additional
        classes without breaking the task.  Ties are resolved by object ID,
        making selection reproducible across detector implementations.
        """
        threshold = float(min_confidence)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("min_confidence must be finite and in [0, 1]")
        best: dict[str, VisibleDetection] = {}
        for raw in detections:
            detection = self._coerce_detection(raw)
            spec = self._by_id.get(detection.object_id)
            if spec is None or (actionable_only and not spec.actionable):
                continue
            if detection.confidence < threshold:
                continue
            previous = best.get(spec.object_id)
            # Keep duplicate detections reproducible even when confidence and
            # mask area tie across detector backends.
            candidate_key = (detection.confidence, detection.mask_area,
                             tuple(-value for value in detection.centroid_xyz), detection.source)
            previous_key = None if previous is None else (
                previous.confidence, previous.mask_area,
                tuple(-value for value in previous.centroid_xyz), previous.source,
            )
            if previous is None or candidate_key > previous_key:
                best[spec.object_id] = detection
        matches = [VisibleMatch(self._by_id[object_id], detection) for object_id, detection in best.items()]
        matches.sort(key=lambda match: (-match.detection.confidence, match.spec.object_id))
        return tuple(matches)

    def select_target(
        self,
        object_id: str | None = None,
        *,
        detections: Iterable[VisibleDetection | Mapping[str, object]] | None = None,
        min_confidence: float = 0.15,
    ) -> ToolObjectSpec:
        """Select a declared actionable object, optionally requiring visibility."""
        if object_id is not None:
            spec = self.by_id(object_id)
            if not spec.actionable:
                raise ValueError(f"object {object_id!r} is a distractor, not an actionable target")
            if detections is None:
                return spec
            matches = {match.spec.object_id for match in self.match_visible(detections, min_confidence=min_confidence)}
            if object_id not in matches:
                raise ValueError(f"target {object_id!r} is not visible above confidence threshold")
            return spec
        if detections is None:
            raise ValueError("object_id or detections is required")
        matches = self.match_visible(detections, min_confidence=min_confidence)
        if not matches:
            raise ValueError("no actionable object is visible above confidence threshold")
        return matches[0].spec

    def manifest(self) -> tuple[dict[str, object], ...]:
        return tuple(item.manifest() for item in self._objects)


def default_workbench_registry() -> ToolObjectRegistry:
    """Return the fixed V7 maintenance-workbench object set.

    The three actionable items are intentionally primitive tool proxies.  The
    can distractor uses the processed YCB mesh already distributed with this
    project, while the remaining distractors keep collision geometry simple.
    """
    tool_source = "Synthetic primitive proxy based on common hand-tool dimensions; not calibrated hardware geometry."
    ycb_source = "YCB 005 tomato soup can processed mesh in vendor/ycb; mass reference from YCB Table II."
    return ToolObjectRegistry((
        ToolObjectSpec(
            "tool_screwdriver", "Screwdriver", "screwdriver", "target", "capsule",
            (0.014, 0.10), 0.15, (0.70, 0.01, 0.001), (0.61, -0.02, 0.405),
            (0.90, 0.36, 0.08, 1.0), tool_source, grasp_axis=(1.0, 0.0, 0.0), dock_slot="slot_screwdriver",
        ),
        ToolObjectSpec(
            "tool_wrench", "Combination wrench", "wrench", "target", "box",
            (0.060, 0.014, 0.006), 0.18, (0.75, 0.01, 0.001), (0.73, 0.04, 0.394),
            (0.12, 0.55, 0.95, 1.0), tool_source, grasp_axis=(1.0, 0.0, 0.0), dock_slot="slot_wrench",
        ),
        ToolObjectSpec(
            "tool_marker", "Marker", "marker", "target", "cylinder",
            (0.012, 0.075), 0.045, (0.65, 0.01, 0.001), (0.78, -0.05, 0.455),
            (0.15, 0.75, 0.22, 1.0), tool_source, grasp_axis=(0.0, 0.0, 1.0), dock_slot="slot_marker",
        ),
        ToolObjectSpec(
            "distractor_ycb_can", "Soup can", "container", "distractor", "mesh",
            (0.035, 0.035, 0.060), 0.349, (0.60, 0.005, 0.0001), (0.84, -0.08, 0.445),
            (0.90, 0.28, 0.12, 1.0), ycb_source, mesh_name="ycb_005_tomato_soup_can",
            mesh_file="../ycb/005_tomato_soup_can/005_tomato_soup_can/poisson/nontextured.stl",
            collision_class="distractor",
        ),
        ToolObjectSpec(
            "distractor_box", "Parts box", "container", "distractor", "box",
            (0.045, 0.030, 0.030), 0.20, (0.55, 0.01, 0.001), (0.62, 0.14, 0.410),
            (0.42, 0.43, 0.46, 1.0), tool_source, collision_class="distractor",
        ),
        ToolObjectSpec(
            "distractor_bottle", "Bottle", "container", "distractor", "cylinder",
            (0.022, 0.060), 0.11, (0.60, 0.01, 0.001), (0.92, -0.08, 0.440),
            (0.18, 0.48, 0.72, 1.0), tool_source, collision_class="distractor",
        ),
    ))

