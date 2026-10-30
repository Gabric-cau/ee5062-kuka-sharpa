"""Declared upright tool sorting fixture used by the V7 tool evidence.

This is a second, explicitly named workcell profile. It does not overwrite
the original flat-clutter scene or turn its failures into successes.
"""

from dataclasses import replace

from .object_registry import ToolObjectRegistry
from .v7_scene import WorkbenchObstacle, V7SceneConfig, default_v7_scene_config


def upright_tool_fixture(seed: int = 0) -> V7SceneConfig:
    base = default_v7_scene_config(seed)
    objects = []
    for spec in base.registry.objects:
        if spec.object_id == "tool_screwdriver":
            spec = replace(spec, pose=(spec.pose[0], spec.pose[1], .494),
                           grasp_axis=(0., 0., 1.),
                           source=spec.source + " Declared upright staging fixture.")
        elif spec.object_id == "tool_wrench":
            spec = replace(spec, dimensions=(.006, .014, .060),
                           pose=(spec.pose[0], -.080, .440), grasp_axis=(0., 0., 1.),
                           source=spec.source + " Declared upright box fixture with axis permutation.")
        objects.append(spec)
    registry = ToolObjectRegistry(objects)
    obstacles = list(base.obstacles)
    for spec in registry.actionable:
        if spec.object_id == "tool_marker":
            continue
        hx, hy = ((.014, .014) if spec.object_id == "tool_screwdriver" else spec.dimensions[:2])
        x, y, _ = spec.pose
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            gap = (hx, hy)[axis] + .007
            pose = (x + sign * gap, y, .402) if axis == 0 else (x, y + sign * gap, .402)
            half = (.004, hy + .003, .022) if axis == 0 else (hx + .003, .004, .022)
            obstacles.append(WorkbenchObstacle(
                f"fixture_source_{spec.object_id}_{axis}_{sign + 1}", pose, half,
                (.6, .6, .6, 1.), "Declared 3 mm initial side-clearance tool guide."))
    slots = tuple(replace(slot, pose=(*slot.pose[:2],
                                      .510 if slot.accepted_object_id == "tool_screwdriver" else
                                      .456 if slot.accepted_object_id == "tool_wrench" else slot.pose[2]))
                  for slot in base.slots)
    for slot in slots:
        if slot.accepted_object_id == "tool_marker":
            continue
        spec = registry.by_id(slot.accepted_object_id)
        hx, hy = ((.014, .014) if spec.object_id == "tool_screwdriver" else spec.dimensions[:2])
        x, y, _ = slot.pose
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            # Eight millimetres of free side clearance: the guide centre is
            # tool half-width + clearance + its 4 mm rail half-width.
            gap = (hx, hy)[axis] + .012
            guide_top = .490 if spec.object_id == "tool_screwdriver" else .446
            guide_center_z = (.396 + guide_top) / 2.0
            guide_half_z = (guide_top - .396) / 2.0
            pose = (x + sign * gap, y, guide_center_z) if axis == 0 else (x, y + sign * gap, guide_center_z)
            half = ((.004, hy + .008, guide_half_z) if axis == 0
                    else (hx + .008, .004, guide_half_z))
            obstacles.append(WorkbenchObstacle(
                f"fixture_dock_{spec.object_id}_{axis}_{sign + 1}", pose, half,
                (.7, .7, .7, 1.), "Declared tool receiving guide with 8 mm side clearance."))
    return replace(base, registry=registry, obstacles=tuple(obstacles), slots=slots,
                   version="v7_upright_tool_fixture_r1")


__all__ = ["upright_tool_fixture"]
