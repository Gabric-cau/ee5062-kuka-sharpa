"""A source checkout must audit archived evidence from another computer."""

import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_ycb_evidence import load_evidence_model
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


def test_missing_original_asset_root_is_relocated_without_changing_scene(tmp_path):
    path = build_pick_place_scene(tmp_path / "scene.xml", "ycb_cracker_box", physics_profile="physical-v6")
    reference = mujoco.MjModel.from_xml_path(str(path))
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", "/nonexistent/original-computer/vendor/mujoco_menagerie")
    path.write_text(ET.tostring(root, encoding="unicode"))
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_evidence_model(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    assert (loaded.nbody, loaded.ngeom, loaded.nq, loaded.nu) == (reference.nbody, reference.ngeom, reference.nq, reference.nu)
    for field in ("body_mass", "body_inertia", "geom_friction", "mesh_vert", "geom_pos", "geom_quat"):
        assert np.array_equal(getattr(loaded, field), getattr(reference, field))
    assert loaded.opt.timestep == reference.opt.timestep == .002
