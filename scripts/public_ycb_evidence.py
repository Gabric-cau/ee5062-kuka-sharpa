"""Load archived scene geometry without rewriting its recorded XML or hash."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT = Path(__file__).resolve().parents[1]


def load_evidence_model(scene):
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Archived public-YCB scene is missing its compiler/asset root")
    compiler.set("meshdir", str(PROJECT / "vendor" / "mujoco_menagerie"))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
