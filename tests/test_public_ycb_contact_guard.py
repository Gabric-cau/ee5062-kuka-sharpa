"""Independent contact guard must preserve the robot's physical command bounds."""

from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from develop_public_ycb_contact_guard import InsertionGuard
from kuka_sharpa_grasp.control import ForceReading
from kuka_sharpa_grasp.motion import ArmCommandLimiter


def test_pressure_reader_accepts_unknown_external_surface_and_excludes_hand_self_contact():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="left_ring_MP"><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".02" mass="1"/></body>
      <body name="left_index_MP" pos="0 .03 0"><geom type="sphere" size=".02"/></body>
      <body name="arbitrary_external_proxy" pos=".03 0 0"><geom type="sphere" size=".02"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    readings = InsertionGuard(model).read(data)
    assert readings["ring"].normal > 0.
    assert readings["ring"].contacts == 1
    assert readings["index"].contacts == 0


def test_existing_grip_is_tared_and_pressure_retreat_preserves_acceleration_limit():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    guard = InsertionGuard(model)
    pressure = {finger: ForceReading() for finger in ("thumb", "index", "middle", "ring", "pinky", "palm")}
    pressure["ring"].normal = 9.
    guard.read = lambda data: pressure
    data = SimpleNamespace(time=0., qpos=np.array([42.]))
    state = SimpleNamespace(value="PLACE")
    guard.begin(state, data)
    kin = SimpleNamespace(origin=np.zeros(3), position=lambda q: q[:3].copy(),
        solve=lambda offset, **kwargs: np.r_[offset, np.zeros(4)])
    command = np.zeros(7)
    nominal = np.array([0., 0., -.01, 0., 0., 0., 0.])
    request = guard.guide(state, nominal, command, kin, data, RuntimeError)
    assert np.array_equal(request, nominal)
    pressure["ring"].normal = 14.
    data.time = .002
    request = guard.guide(state, nominal, command, kin, data, RuntimeError)
    assert request[2] == pytest.approx(.006)
    assert data.qpos.tolist() == [42.]
    limiter = ArmCommandLimiter(command, np.ones(7), np.full(7, .8))
    previous_velocity = limiter.velocity.copy()
    limited = limiter.update(request, .002)
    assert np.max(np.abs(limiter.velocity - previous_velocity)) <= .8 * .002 + 1e-12
    assert np.max(np.abs(limited - command)) <= .8 * .002 ** 2 + 1e-12
