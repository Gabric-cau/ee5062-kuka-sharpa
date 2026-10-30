"""V2 workcell v2.1, layered on the immutable accepted scene builder."""
import xml.etree.ElementTree as ET
from .scene import build_scene

WORKCELL_VERSION = 'v2.1_clear_mount'


def build_pick_place_scene(output, object_kind='ycb_can', *, physics_profile='legacy-v5', timestep=None):
    build_scene(output, object_kind, physics_profile=physics_profile, timestep=timestep)
    tree = ET.parse(output)
    table = tree.find(".//geom[@name='table']")
    table.set('pos', '0.70 0 0.34')
    table.set('size', '0.40 0.45 0.04')
    # The fixed arm base is supported by a pedestal whose top meets its mount.
    world = tree.find('worldbody')
    ET.SubElement(world, 'geom', name='arm_pedestal', type='box',
                  pos='0 0 0.125', size='0.12 0.12 0.125', rgba='0.3 0.3 0.32 1')
    goal = tree.find(".//site[@name='goal']")
    goal.set('pos', '0.65 -0.18 0.381')
    goal.set('size', '0.045 0.001')
    tree.getroot().set('model', WORKCELL_VERSION)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return output
