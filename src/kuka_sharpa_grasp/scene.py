"""Build a self-contained KUKA iiwa + Sharpa Wave MuJoCo scene.

The vendor XML files are kept untouched. This module copies their model
sections into one scene and mounts the hand at the iiwa attachment site.
"""

from __future__ import annotations

import copy
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .physics import get_physics_profile, native_contact_parameters, pair_friction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "mujoco_menagerie"
KUKA_XML = VENDOR_ROOT / "kuka_iiwa_14" / "iiwa14.xml"
SHARPA_XML = VENDOR_ROOT / "sharpa_wave" / "left_hand.xml"
SCENE_XML = PROJECT_ROOT / "scene.xml"
YCB_CAN_MESH = "../ycb/005_tomato_soup_can/005_tomato_soup_can/poisson/nontextured.stl"
YCB_OBJECTS = {
    "ycb_can": {
        "mesh_name": "ycb_005_tomato_soup_can",
        "mesh_file": YCB_CAN_MESH,
        "target_pos": "0.72 0 0.38",
    },
    "ycb_potted_meat_can": {
        "mesh_name": "ycb_010_potted_meat_can",
        "mesh_file": "../ycb/010_potted_meat_can/010_potted_meat_can/poisson/nontextured.stl",
        "target_pos": "0.72 0 0.38",
    },
    "ycb_cracker_box": {
        "mesh_name": "ycb_003_cracker_box",
        "mesh_file": "../ycb/003_cracker_box/003_cracker_box/poisson/nontextured.stl",
        "target_pos": "0.72 0 0.38",
    },
}
OBJECT_KINDS = ("box", *YCB_OBJECTS)


def _copy_children(source: ET.Element | None, target: ET.Element) -> None:
    if source is None:
        return
    for child in source:
        target.append(copy.deepcopy(child))


def _rewrite_asset_paths(asset: ET.Element, prefix: str) -> None:
    for mesh in asset.findall("mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh.set("file", f"{prefix}/{file_name}")


def _namespace_sharpa_classes(root: ET.Element) -> None:
    """Avoid class-name collisions when the two upstream models are merged."""
    mapping: dict[str, str] = {}
    for element in root.iter():
        class_name = element.get("class")
        if class_name:
            mapping.setdefault(class_name, f"sharpa_{class_name}")
        child_class = element.get("childclass")
        if child_class:
            mapping.setdefault(child_class, f"sharpa_{child_class}")
    for element in root.iter():
        for attribute in ("class", "childclass"):
            value = element.get(attribute)
            if value in mapping:
                element.set(attribute, mapping[value])


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"body {name!r} not found")


def build_scene(output: Path = SCENE_XML, object_kind: str = "box", *, overhead_camera: bool = False,
                physics_profile: str = "legacy-v5", timestep: float | None = None) -> Path:
    """Create the combined scene and return its path."""

    if object_kind not in OBJECT_KINDS:
        raise ValueError(f"unsupported object kind: {object_kind}")

    profile = get_physics_profile(physics_profile)
    parameters = profile.object_parameters(object_kind)
    step = profile.timestep if timestep is None else float(timestep)
    if not 0 < step <= 0.01:
        raise ValueError("physics timestep must be finite and in (0, 0.01] seconds")

    def text(values: tuple[float, ...]) -> str:
        return " ".join(format(value, ".12g") for value in values)

    kuka = ET.parse(KUKA_XML).getroot()
    sharpa = ET.parse(SHARPA_XML).getroot()
    _namespace_sharpa_classes(sharpa)
    root = ET.Element("mujoco", {"model": "kuka_sharpa_force_grasp"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": VENDOR_ROOT.resolve().as_posix(), "autolimits": "true"})
    ET.SubElement(root, "option", {
        "integrator": profile.integrator, "impratio": str(profile.solver_impratio), "cone": "elliptic", "timestep": str(step),
        "gravity": text(profile.gravity), "solver": profile.solver,
        "iterations": str(profile.iterations), "tolerance": str(profile.tolerance),
    })
    ET.SubElement(root, "size", {"nuser_jnt": "2"})
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", {"name": "physics_profile_version", "data": str(profile.version)})
    ET.SubElement(custom, "numeric", {"name": "physics_contact_revision", "data": str(profile.contact_parameter_revision)})

    defaults = ET.SubElement(root, "default")
    _copy_children(kuka.find("default"), defaults)
    _copy_children(sharpa.find("default"), defaults)

    assets = ET.SubElement(root, "asset")
    kuka_assets = copy.deepcopy(kuka.find("asset"))
    sharpa_assets = copy.deepcopy(sharpa.find("asset"))
    if kuka_assets is not None:
        _rewrite_asset_paths(kuka_assets, "kuka_iiwa_14/assets")
        _copy_children(kuka_assets, assets)
    if sharpa_assets is not None:
        _rewrite_asset_paths(sharpa_assets, "sharpa_wave/assets/left")
        _copy_children(sharpa_assets, assets)
    object_spec = YCB_OBJECTS.get(object_kind)
    if object_spec is not None:
        ET.SubElement(assets, "mesh", {
            "name": object_spec["mesh_name"],
            "file": object_spec["mesh_file"],
        })

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -1.5 2.5", "dir": "0 0 -1"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.1", "rgba": "0.18 0.2 0.24 1"})
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box", "pos": "0.55 0 0.34", "size": "0.55 0.45 0.04",
        "friction": text(profile.table_friction), "rgba": "0.35 0.38 0.42 1",
    })

    kuka_worldbody = kuka.find("worldbody")
    if kuka_worldbody is None:
        raise ValueError("KUKA worldbody missing")
    kuka_children = [copy.deepcopy(child) for child in kuka_worldbody]
    base = next(child for child in kuka_children if child.tag == "body" and child.get("name") == "base")
    # Mount the arm on the workbench so the home pose places the hand above it.
    base.set("pos", "0 0 0.25")
    link7 = _find_body(base, "link7")
    # Optical +X right, +Y up, viewing along -Z (MuJoCo convention).
    ET.SubElement(link7, "camera", {
        "name": "wrist_rgbd", "pos": "-0.18811327 -0.25 -0.07087866",
        "xyaxes": "-0.70710926 0 0.7071043 -0.43532455 0.78802437 -0.4353276",
        "fovy": "55",
    })
    if overhead_camera:
        ET.SubElement(worldbody, "camera", {
            "name": "overhead_rgbd", "pos": "0.72 0 1.25",
            "xyaxes": "1 0 0 0 1 0", "fovy": "55",
        })
    hand_body = copy.deepcopy(sharpa.find("worldbody/body"))
    if hand_body is None:
        raise ValueError("Sharpa hand body missing")
    hand_body.set("pos", "0 0 0.045")
    link7.append(hand_body)
    if profile.arm_torque_limits_nm is not None:
        for rank, torque in enumerate(profile.arm_torque_limits_nm):
            joint = next(element for element in base.iter("joint") if element.get("name") == f"joint{rank + 1}")
            joint.set("actuatorfrcrange", f"{-torque} {torque}")
            joint.set("actuatorfrclimited", "true")
            joint.set("user", text((profile.arm_velocity_limits_rad_s[rank], profile.arm_acceleration_limits_rad_s2[rank])))
    hand_collision_geoms = []
    if profile.version >= 6:
        for rank, geom in enumerate(hand_body.iter("geom")):
            class_name = geom.get("class", "")
            if "visual" in class_name:
                continue
            if not geom.get("name"):
                geom.set("name", f"sharpa_collision_{rank}")
            geom.set("friction", text(profile.hand_friction))
            hand_collision_geoms.append(geom)
    for child in kuka_children:
        worldbody.append(child)

    # The free joint keeps the test object easy to reset and randomize.
    # The first milestone seeds an object in the hand aperture. The approach
    # planner will replace this with a collision-free pre-grasp trajectory.
    # Mesh objects use their scene-authored base height and a hand-aligned x
    # position; the box keeps the original regression pose.
    target_pos = object_spec["target_pos"] if object_spec is not None else "0.70 0 0.43"
    target = ET.SubElement(worldbody, "body", {"name": "target", "pos": target_pos})
    ET.SubElement(target, "freejoint", {"name": "target_free"})
    if object_spec is not None:
        ET.SubElement(target, "geom", {
            "name": "target_geom", "type": "mesh", "mesh": object_spec["mesh_name"], "mass": str(parameters.mass_kg),
            "friction": text(parameters.friction), "rgba": "0.9 0.28 0.12 1",
        })
    else:
        ET.SubElement(target, "geom", {
            "name": "target_geom", "type": "box", "size": "0.035 0.035 0.05", "mass": str(parameters.mass_kg),
            "friction": text(parameters.friction), "rgba": "0.9 0.28 0.12 1",
        })
    if profile.version >= 6:
        target.find("geom").attrib.update({
            "condim": str(parameters.condim), "solref": text(parameters.solref), "solimp": text(parameters.solimp),
        })
    ET.SubElement(worldbody, "site", {"name": "goal", "pos": "0.65 -0.25 0.44", "size": "0.08 0.08 0.002", "type": "cylinder", "rgba": "0.2 0.8 0.3 0.45"})

    contact = ET.SubElement(root, "contact")
    _copy_children(kuka.find("contact"), contact)
    _copy_children(sharpa.find("contact"), contact)
    ET.SubElement(contact, "exclude", {"body1": "link7", "body2": "left_hand_C_MC"})
    if profile.version >= 6:
        ET.SubElement(contact, "pair", {
            "name": "target_table", "geom1": "target_geom", "geom2": "table",
            "condim": str(parameters.condim), "friction": text(pair_friction(parameters.friction, profile.table_friction)),
            "solref": text(parameters.solref), "solimp": text(parameters.solimp),
        })
        for geom in hand_collision_geoms:
            ET.SubElement(contact, "pair", {
                "name": f"target_{geom.get('name')}", "geom1": "target_geom", "geom2": geom.get("name"),
                "condim": str(parameters.condim), "friction": text(pair_friction(parameters.friction, profile.hand_friction)),
                "solimp": text(parameters.solimp),
            })

    actuator = ET.SubElement(root, "actuator")
    _copy_children(kuka.find("actuator"), actuator)
    _copy_children(sharpa.find("actuator"), actuator)

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": "0 0.785398 0 -1.5708 0 0 0", "ctrl": "0 0.785398 0 -1.5708 0 0 0"})

    if profile.version >= 6:
        # Compilation resolves nested vendor defaults and childclass inheritance.
        compiled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        for pair in contact.findall("pair"):
            if pair.get("geom1") != "target_geom":
                continue
            first = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, pair.get("geom1"))
            second = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, pair.get("geom2"))
            solref, solimp = native_contact_parameters(compiled, first, second)
            pair.set("solref", text(solref))
            pair.set("solimp", text(solimp))

    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, output)
    return output


if __name__ == "__main__":
    print(build_scene())
