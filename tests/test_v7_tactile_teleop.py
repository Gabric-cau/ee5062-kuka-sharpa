import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuka_sharpa_grasp.tactile import TactileContactReader
from kuka_sharpa_grasp.teleop import KeyboardTeleop, TeleopActionLog, TeleopCommand


def _contact_model(with_touch=False):
    sensor = """
      <site name="index_site" pos="0 0 0" size=".12"/>
      <site name="index_touch_site" pos="0 0 0" size=".12"/>
    """ if with_touch else ""
    sensors = '<sensor><touch name="index_touch" site="index_touch_site"/></sensor>' if with_touch else ""
    xml = f'''<mujoco><option gravity="0 0 0"/>
      <worldbody>
        <body name="left_index_link" pos="0 0 0"><geom name="left_index_pad" type="sphere" size=".08"/>{sensor}</body>
        <body name="left_middle_link" pos="0 0 .3"><geom name="left_middle_pad" type="sphere" size=".04"/></body>
        <body name="crate" pos=".1 0 0"><joint type="free"/><geom name="crate_geom" type="box" size=".08 .08 .08"/></body>
      </worldbody>{sensors}</mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def test_tactile_reader_requires_no_target_geom_and_reports_native_contact():
    model = _contact_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reader = TactileContactReader(model)
    snapshot = reader.read(data)
    index = snapshot.fingers["index"]
    assert index.contacts > 0
    assert index.centroid_world is not None
    assert snapshot.contact_count == index.contacts
    assert "target_geom" not in " ".join(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        for gid in reader.geometry_channels
    )


def test_tactile_touch_sensor_is_optional_fallback_and_json_safe():
    model = _contact_model(with_touch=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reader = TactileContactReader(model, touch_sensor_names={"index": "index_touch"})
    snapshot = reader.read(data)
    assert snapshot.fingers["index"].touch_sensor_force_n is not None
    payload = snapshot.as_dict()
    assert payload["fingers"]["index"]["contacts"] > 0


def test_keyboard_and_replay_are_deterministic(tmp_path):
    keyboard = KeyboardTeleop()
    first = keyboard.command({"w", "d", "]", "target:cracker"}, .2)
    assert first.translation_m == pytest.approx((.01, .01, 0.))
    assert first.grip_delta == pytest.approx(.08)
    assert first.selected_target == "cracker"
    log = TeleopActionLog([TeleopCommand(0., source="test"), first])
    assert log.due(.1) == (log.commands[0],)
    assert log.due(.2) == (first,)
    path = log.to_json(tmp_path / "actions.json")
    loaded = TeleopActionLog.from_json(path)
    assert tuple(loaded.replay()) == log.commands
    loaded.reset()
    assert loaded.due(.2) == log.commands


def test_action_log_rejects_nonmonotonic_commands():
    with pytest.raises(ValueError):
        TeleopActionLog([TeleopCommand(.2), TeleopCommand(.1)])


def test_native_tactile_aggregates_multiple_objects_and_excludes_self_contact(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="left_index_link"><geom type="sphere" size=".01"/></body>
      <body name="left_hand_palm"><geom type="sphere" size=".01"/></body>
      <body name="unlabelled_tool"><geom type="sphere" size=".01"/></body>
      <body name="fixture"><geom type="sphere" size=".01"/></body>
    </worldbody></mujoco>''')
    contacts = [SimpleNamespace(geom1=0, geom2=2, pos=np.array([1., 0., 0.])),
                SimpleNamespace(geom1=3, geom2=0, pos=np.array([0., 1., 0.])),
                SimpleNamespace(geom1=0, geom2=1, pos=np.array([9., 9., 9.]))]
    data = SimpleNamespace(time=.5, ncon=3, contact=contacts)
    wrenches = [np.array([3., 4., 0., 0., 0., 0.]),
                np.array([5., 0., 0., 0., 0., 0.]),
                np.array([100., 0., 0., 0., 0., 0.])]
    def contact_force(_model, _data, index, destination):
        destination[:] = wrenches[index]
    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
    snapshot = TactileContactReader(model, include_palm=False).read(data)
    reading = snapshot.fingers["index"]
    assert reading.normal_force_n == 8.
    assert reading.tangential_force_n == 4.
    assert reading.contacts == 2
    assert reading.centroid_world == pytest.approx((7 / 12, 5 / 12, 0.))
    assert reading.slip_score == pytest.approx(1 / 3)
    assert "palm" not in snapshot.fingers


def test_replay_time_requires_reset_after_simulation_reset():
    log = TeleopActionLog([TeleopCommand(.1), TeleopCommand(.2)])
    assert len(log.due(.2)) == 2
    with pytest.raises(ValueError):
        log.due(.1)
    log.reset()
    assert len(log.due(.1)) == 1


@pytest.mark.parametrize("fields", [
    {"timestamp": float("nan")}, {"timestamp": -.1},
    {"timestamp": 0., "translation_m": [1., 2.]},
    {"timestamp": 0., "rotation_rad": [1., float("inf"), 0.]},
    {"timestamp": 0., "grip_delta": float("nan")},
    {"timestamp": 0., "selected_target": ""},
])
def test_invalid_teleoperation_commands_are_rejected(fields):
    with pytest.raises(ValueError):
        TeleopCommand(**fields)
