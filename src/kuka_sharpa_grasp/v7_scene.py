"""Multi-object cluttered maintenance-workbench scene for the V7 study.

This builder layers on the existing KUKA-Sharpa scene generator and writes a
new XML file.  It never changes the V6/default scene on disk, and the V7
objects are free bodies so later teleoperation and tactile controllers can
select one target while the remaining objects stay physical distractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco

from .object_registry import ToolObjectRegistry, ToolObjectSpec, default_workbench_registry
from .physics import get_physics_profile, pair_friction
from .pick_place_scene import build_pick_place_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V7_SCENE_XML = PROJECT_ROOT / "results" / "v7_workbench_scene.xml"


@dataclass(frozen=True)
class WorkbenchObstacle:
    """A fixed collision obstacle, expressed as a box half-extent."""

    obstacle_id: str
    pose: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    source: str = "Synthetic workbench fixture; dimensions are engineering assumptions."
    collision_class: str = "obstacle"

    def __post_init__(self) -> None:
        if not self.obstacle_id or any(character.isspace() for character in self.obstacle_id):
            raise ValueError("obstacle_id must be a non-empty token")
        if len(self.pose) != 3 or len(self.half_extents) != 3 or len(self.rgba) != 4:
            raise ValueError("obstacle pose, half_extents and rgba have invalid length")
        if not all(math.isfinite(float(value)) for value in (*self.pose, *self.half_extents, *self.rgba)):
            raise ValueError("obstacle values must be finite")
        if not all(float(value) > 0 for value in self.half_extents):
            raise ValueError("obstacle half_extents must be positive")
        if not all(0 <= float(value) <= 1 for value in self.rgba):
            raise ValueError("obstacle rgba values must be in [0, 1]")


@dataclass(frozen=True)
class DockSlot:
    """A semantic rack slot and its visible, non-colliding marker site."""

    slot_id: str
    accepted_object_id: str
    pose: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if not self.slot_id or not self.accepted_object_id:
            raise ValueError("slot_id and accepted_object_id must be non-empty")
        if len(self.pose) != 3 or len(self.half_extents) != 3:
            raise ValueError("slot pose and half_extents have invalid length")
        if not all(math.isfinite(float(value)) for value in (*self.pose, *self.half_extents, self.yaw)):
            raise ValueError("slot values must be finite")
        if not all(float(value) > 0 for value in self.half_extents):
            raise ValueError("slot half_extents must be positive")


@dataclass(frozen=True)
class V7SceneConfig:
    """Complete deterministic maintenance-workbench layout."""

    registry: ToolObjectRegistry
    obstacles: tuple[WorkbenchObstacle, ...]
    slots: tuple[DockSlot, ...]
    seed: int = 0
    version: str = "v7_workbench_r1"

    def __post_init__(self) -> None:
        if int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        obstacle_ids = [item.obstacle_id for item in self.obstacles]
        if len(set(obstacle_ids)) != len(obstacle_ids):
            raise ValueError("obstacle IDs must be unique")
        slot_ids = [item.slot_id for item in self.slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot IDs must be unique")
        known = {item.object_id for item in self.registry.objects}
        for slot in self.slots:
            if slot.accepted_object_id not in known:
                raise ValueError(f"slot {slot.slot_id!r} accepts unknown object {slot.accepted_object_id!r}")
        _validate_layout(self.registry, self.obstacles)

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": int(self.seed),
            "objects": list(self.registry.manifest()),
            "obstacles": [
                {"obstacle_id": item.obstacle_id, "pose": list(item.pose),
                 "half_extents": list(item.half_extents), "rgba": list(item.rgba),
                 "source": item.source, "collision_class": item.collision_class}
                for item in self.obstacles
            ],
            "slots": [
                {"slot_id": item.slot_id, "accepted_object_id": item.accepted_object_id,
                 "pose": list(item.pose), "half_extents": list(item.half_extents), "yaw": item.yaw}
                for item in self.slots
            ],
        }


def default_v7_scene_config(seed: int = 0) -> V7SceneConfig:
    """Return the fixed cluttered maintenance-workbench layout.

    The layout uses a central divider and a rear rail to create a meaningful
    collision-aware transfer corridor.  Rack slots are semantic sites; the
    rack base and dividers are added as fixed MuJoCo geoms by the builder.
    """
    base_registry = default_workbench_registry()
    # Small seeded placement variation makes each scene a real clutter case
    # while preserving deterministic replay and the declared fixture margins.
    rng = random.Random(int(seed))
    registry = ToolObjectRegistry(tuple(
        replace(spec, pose=(spec.pose[0] + rng.uniform(-.0015, .0015),
                            spec.pose[1] + rng.uniform(-.0015, .0015), spec.pose[2]))
        for spec in base_registry.objects
    ))
    fixture = "Synthetic workbench fixture; dimensions are engineering assumptions."
    obstacles = (
        WorkbenchObstacle("divider_center", (0.705, 0.075, 0.445), (0.115, 0.014, 0.065), (0.20, 0.28, 0.34, 1.0), fixture),
        WorkbenchObstacle("divider_rear", (0.95, 0.155, 0.435), (0.015, 0.18, 0.055), (0.20, 0.28, 0.34, 1.0), fixture),
    )
    slots = (
        DockSlot("slot_screwdriver", "tool_screwdriver", (0.58, -0.245, 0.405), (0.075, 0.018, 0.010)),
        DockSlot("slot_wrench", "tool_wrench", (0.70, -0.245, 0.405), (0.075, 0.024, 0.010)),
        # The upright marker's centre sits 75 mm above the rack.  The
        # release target includes 14 mm clearance for the opening fingertip
        # pads; verification observes the free object's settled pose.
        DockSlot("slot_marker", "tool_marker", (0.82, -0.245, 0.485), (0.050, 0.018, 0.010)),
    )
    return V7SceneConfig(registry, obstacles, slots, seed=int(seed))


def _text(values: tuple[float, ...]) -> str:
    return " ".join(format(float(value), ".12g") for value in values)


def _yaw_quat(yaw: float) -> str:
    return _text((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))


def _axis_quat(axis: tuple[float, float, float]) -> str:
    """Return a quaternion that aligns a tool's local +Z axis to ``axis``.

    The primitive capsule/cylinder assets are authored along +Z.  Applying the
    declared grasp axis keeps their initial pose above the table and makes the
    registry's geometry contract visible in the compiled scene.
    """
    vector = [float(value) for value in axis]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("tool grasp axis must be non-zero")
    x, y, z = (value / norm for value in vector)
    if z > 1.0 - 1e-12:
        return "1 0 0 0"
    if z < -1.0 + 1e-12:
        return "0 0 1 0"
    # Quaternion from +Z to the requested unit vector.
    cross_x, cross_y = -y, x
    w = math.sqrt((1.0 + z) / 2.0)
    scale = 1.0 / (2.0 * w)
    return _text((w, cross_x * scale, cross_y * scale, 0.0))


def _object_half_extents(spec: ToolObjectSpec) -> tuple[float, float, float]:
    """Conservative world-axis bounds for seeded fixture validation."""
    if spec.shape == "box":
        return spec.dimensions  # box assets keep their authored axes
    if spec.shape in {"capsule", "cylinder"}:
        radius, half_length = spec.dimensions
        norm = math.sqrt(sum(value * value for value in spec.grasp_axis))
        axis = tuple(value / norm for value in spec.grasp_axis)
        axial_extra = radius if spec.shape == "capsule" else 0.0
        return tuple(float(radius + (half_length + axial_extra - radius) * abs(axis[index])) for index in range(3))
    return spec.dimensions


def _overlap(first_pos: tuple[float, float, float], first_size: tuple[float, float, float],
             second_pos: tuple[float, float, float], second_size: tuple[float, float, float]) -> bool:
    return all(abs(first_pos[index] - second_pos[index]) < first_size[index] + second_size[index] - 1e-6
               for index in range(3))


def _validate_layout(registry: ToolObjectRegistry, obstacles: tuple[WorkbenchObstacle, ...]) -> None:
    """Reject object interpenetration and placements outside the workbench."""
    table_min, table_max, table_top = (0.30, -0.45), (1.10, 0.45), 0.38
    object_bounds: list[tuple[ToolObjectSpec, tuple[float, float, float]]] = []
    for spec in registry.objects:
        half = _object_half_extents(spec)
        x, y, z = spec.pose
        if x - half[0] < table_min[0] or x + half[0] > table_max[0] or y - half[1] < table_min[1] or y + half[1] > table_max[1]:
            raise ValueError(f"object {spec.object_id!r} is outside the table bounds")
        if z - half[2] < table_top - 1e-6:
            raise ValueError(f"object {spec.object_id!r} penetrates the table")
        object_bounds.append((spec, half))
    for index, (first, first_half) in enumerate(object_bounds):
        for second, second_half in object_bounds[index + 1:]:
            if _overlap(first.pose, first_half, second.pose, second_half):
                raise ValueError(f"objects {first.object_id!r} and {second.object_id!r} overlap")
        for obstacle in obstacles:
            if _overlap(first.pose, first_half, obstacle.pose, obstacle.half_extents):
                raise ValueError(f"object {first.object_id!r} overlaps obstacle {obstacle.obstacle_id!r}")
    for index, first in enumerate(obstacles):
        for second in obstacles[index + 1:]:
            if _overlap(first.pose, first.half_extents, second.pose, second.half_extents):
                raise ValueError(f"obstacles {first.obstacle_id!r} and {second.obstacle_id!r} overlap")


def _remove_existing_target(worldbody: ET.Element, contact: ET.Element | None) -> None:
    target = next((body for body in worldbody.findall("body") if body.get("name") == "target"), None)
    if target is not None:
        worldbody.remove(target)
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if "target_geom" in {pair.get("geom1"), pair.get("geom2")}:
                contact.remove(pair)


def _add_object_body(worldbody: ET.Element, spec: ToolObjectSpec, *, condim: int) -> str:
    body_name = f"v7_obj_{spec.object_id}"
    geom_name = f"v7_geom_{spec.object_id}"
    orientation = spec.grasp_axis if spec.shape in {"capsule", "cylinder"} else (0.0, 0.0, 1.0)
    attributes = {"name": body_name, "pos": _text(spec.pose),
                  "quat": _axis_quat(orientation)}
    body = ET.SubElement(worldbody, "body", attributes)
    ET.SubElement(body, "freejoint", {"name": f"{body_name}_free"})
    geom = {
        "name": geom_name,
        "mass": format(spec.mass_kg, ".12g"),
        "friction": _text(spec.friction),
        "condim": str(condim),
        "solref": "0.02 1",
        "solimp": "0.9 0.95 0.001 0.5 2",
        "rgba": _text(spec.rgba),
        "group": "0",
    }
    if spec.shape == "mesh":
        geom.update({"type": "mesh", "mesh": str(spec.mesh_name)})
    elif spec.shape == "box":
        geom.update({"type": "box", "size": _text(spec.dimensions)})
    elif spec.shape == "cylinder":
        geom.update({"type": "cylinder", "size": _text(spec.dimensions)})
    elif spec.shape == "capsule":
        geom.update({"type": "capsule", "size": _text(spec.dimensions)})
    else:  # pragma: no cover - ToolObjectSpec validates shape
        raise ValueError(f"unsupported object shape: {spec.shape}")
    ET.SubElement(body, "geom", geom)
    ET.SubElement(body, "site", {
        "name": f"v7_touch_{spec.object_id}", "type": "sphere", "pos": "0 0 0",
        "size": "0.004", "rgba": "1 0.8 0.1 0.65", "group": "3",
    })
    return geom_name


def build_v7_workbench_scene(
    output: Path = V7_SCENE_XML,
    *,
    config: V7SceneConfig | None = None,
    physics_profile: str = "physical-v6",
    timestep: float | None = None,
) -> Path:
    """Build a deterministic multi-object MuJoCo scene and return its path."""
    config = default_v7_scene_config() if config is None else config
    profile = get_physics_profile(physics_profile)
    base_object_kind = "ycb_can"  # provides the already-vendored YCB mesh asset
    build_pick_place_scene(output, base_object_kind, physics_profile=physics_profile, timestep=timestep)
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("model", config.version)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("base scene is missing worldbody")
    contact = root.find("contact")
    _remove_existing_target(worldbody, contact)
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    existing_meshes = {mesh.get("name") for mesh in assets.findall("mesh")}
    for spec in config.registry.objects:
        if spec.shape == "mesh" and spec.mesh_name not in existing_meshes:
            if not spec.mesh_file:
                raise ValueError(f"mesh object {spec.object_id!r} is missing mesh_file")
            ET.SubElement(assets, "mesh", {"name": str(spec.mesh_name), "file": str(spec.mesh_file)})
            existing_meshes.add(spec.mesh_name)

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", {"name": "v7_scene_version", "data": "7"})
    ET.SubElement(custom, "numeric", {"name": "v7_object_count", "data": str(len(config.registry.objects))})
    ET.SubElement(custom, "numeric", {"name": "v7_seed", "data": str(config.seed)})

    for obstacle in config.obstacles:
        ET.SubElement(worldbody, "geom", {
            "name": f"v7_obstacle_{obstacle.obstacle_id}", "type": "box",
            "pos": _text(obstacle.pose), "size": _text(obstacle.half_extents),
            "rgba": _text(obstacle.rgba), "contype": "1", "conaffinity": "1",
        })

    # Rack base and rails make the slot markers a physical docking task.
    ET.SubElement(worldbody, "geom", {
        "name": "v7_rack_base", "type": "box", "pos": "0.70 -0.245 0.388",
        "size": "0.20 0.075 0.008", "rgba": "0.18 0.42 0.24 1", "contype": "1", "conaffinity": "1",
    })
    for index, x in enumerate((0.505, 0.635, 0.765, 0.895)):
        ET.SubElement(worldbody, "geom", {
            "name": f"v7_rack_rail_{index}", "type": "box", "pos": _text((x, -0.245, 0.425)),
            "size": "0.008 0.075 0.035", "rgba": "0.18 0.42 0.24 1", "contype": "1", "conaffinity": "1",
        })

    object_geom_names = [_add_object_body(worldbody, spec, condim=3) for spec in config.registry.objects]
    for slot in config.slots:
        ET.SubElement(worldbody, "site", {
            "name": f"v7_slot_{slot.slot_id}", "type": "box", "pos": _text(slot.pose),
            "quat": _yaw_quat(slot.yaw), "size": _text(slot.half_extents),
            "rgba": "0.2 0.9 0.3 0.38", "group": "3",
        })

    if contact is None:
        contact = ET.SubElement(root, "contact")
    # The hand uses vendor collision geoms; explicit pairs preserve V6's
    # geometric-mean friction convention for every registered object.
    hand_geoms = []
    hand_body = next((body for body in root.iter("body") if body.get("name") == "left_hand_C_MC"), None)
    if hand_body is not None:
        for geom in hand_body.iter("geom"):
            name = geom.get("name")
            if name and geom.get("class") != "visual" and not name.endswith("_visual"):
                hand_geoms.append(name)
    for spec, geom_name in zip(config.registry.objects, object_geom_names):
        ET.SubElement(contact, "pair", {
            "name": f"v7_{spec.object_id}_table", "geom1": geom_name, "geom2": "table",
            "condim": "3", "friction": _text(pair_friction(spec.friction, profile.table_friction)),
            "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2",
        })
        for hand_index, hand_geom_name in enumerate(hand_geoms):
            ET.SubElement(contact, "pair", {
                "name": f"v7_{spec.object_id}_hand_{hand_index}", "geom1": geom_name, "geom2": hand_geom_name,
                "condim": "3", "friction": _text(pair_friction(spec.friction, profile.hand_friction)),
                "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2",
            })

    # Native per-fingertip touch channels complement the target-independent
    # contact reader.  The vendor fingertip sites are enlarged only in this
    # V7 scene to define a small sensing patch; V6 XML remains untouched.
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        site_name = f"left_{finger}_fingertip"
        site = next((item for item in root.iter("site") if item.get("name") == site_name), None)
        if site is None:
            continue
        site.set("type", "sphere")
        site.set("size", "0.008")
        ET.SubElement(sensor, "touch", {"name": f"v7_touch_{finger}", "site": site_name})

    ET.indent(root, space="  ")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


__all__ = [
    "DockSlot", "V7SceneConfig", "V7_SCENE_XML", "WorkbenchObstacle",
    "build_v7_workbench_scene", "default_v7_scene_config",
]

