"""Known-frame force signs and native/trace timing for the offline load plotter."""

from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_public_ycb_alignment_load_transfer as plot


class ForceModel:
    ngeom = 4
    body_mass = [0., .4]
    opt = SimpleNamespace(gravity=np.array([0., 0., -10.]))

    def body(self, name):
        assert name == "target"
        return SimpleNamespace(id=1)


def contact(ids, frame, local):
    return {"geom_ids": ids, "contact_frame_world_rows": frame,
        "force_torque_contact_frame": [*local, 0., 0., 0.]}


@pytest.mark.parametrize("target_second", [False, True])
def test_rotated_contact_frame_hand_on_target_sign_and_decomposition(target_second):
    # n=(.6,0,.8), t1=(-.8,0,.6): normal Fz=8, tangent Fz=-3.
    basis = [[.6, 0., .8], [-.8, 0., .6], [0., -1., 0.]]
    sample = {"timestamp": .002, "target_spatial_velocity_world": [0.] * 6,
        "native_contacts": [contact([1, 0] if target_second else [0, 1], basis, [10., -5., 0.]),
            contact([0, 2], [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]], [4., 0., 0.]),
            contact([1, 3], np.eye(3).tolist(), [999., 0., 0.])]}
    forces = plot.vertical_forces(sample, ForceModel(), 0, ["target_geom", "finger", "table", "other"],
        ["target", "left_thumb_MP", "world", "world"])
    sign = 1. if target_second else -1.
    assert forces["hand_normal_z_n"] == pytest.approx(sign * 8.)
    assert forces["hand_tangent_z_n"] == pytest.approx(sign * -3.)
    assert forces["hand_net_z_n"] == pytest.approx(sign * 5.)
    assert forces["hand_net_z_n"] == pytest.approx(forces["hand_normal_z_n"] + forces["hand_tangent_z_n"])
    assert forces["table_z_n"] == 4.
    assert forces["other_z_n"] == 0.


def test_invalid_contact_frame_is_not_silently_dropped():
    sample = {"timestamp": .002, "target_spatial_velocity_world": [0.] * 6,
        "native_contacts": [contact([1, 0], [[1., 0., 0.]] * 3, [3., 1., 0.])]}
    with pytest.raises(ValueError, match="contact frame"):
        plot.vertical_forces(sample, ForceModel(), 0, ["target_geom", "finger", "table", "other"],
            ["target", "left_thumb_MP", "world", "world"])


def test_first_same_clock_trace_matches_native_callback_without_interpolation():
    class Kinematics:
        def height_at(self, command):
            return float(command[0])
    def row(time, phase, wrist, command):
        return {"timestamp": time, "state": phase, "wrist_position": [0., 0., wrist],
            "arm_command": [command] * 7, "hand_command": [0.] * 22}
    values, duplicates = plot.reduce_trace([
        row(.002, "ALIGN", .8, .81), row(.002, "RECOVER", .9, .91), row(.006, "RECOVER", .7, .71)], Kinematics())
    assert duplicates == 1
    assert values[1]["phase"] == "ALIGN"
    assert values[1]["wrist_actual_z_m"] == .8
    assert values[1]["wrist_command_fk_z_m"] == .81
    assert 2 not in values
    native = {"timestamp": .002, "state": "ALIGN", "motor_commands": [0.] * 29}
    index, joined = plot.exact_native_trace(native, 1, values)
    assert index == 1 and joined["wrist_actual_z_m"] == .8
    assert "_hand_command" not in joined
    with pytest.raises(ValueError, match="match exactly"):
        plot.exact_native_trace({**native, "timestamp": .004}, 2, values)
    with pytest.raises(ValueError, match="phase/hand controls"):
        plot.exact_native_trace({**native, "motor_commands": [0.] * 28 + [.01]}, 1, values)
    with pytest.raises(ValueError, match="native grid"):
        plot.reduce_trace([row(.003, "ALIGN", .8, .81)], Kinematics())


def test_unexecuted_phase_has_no_zero_movement_or_force_statistics():
    assert plot.phase_summary([], 4.) == {
        "executed": False, "native_samples": 0, "statistics": None, "last_250ms": None}


def test_phase_tail_uses_actual_last_250ms_and_keeps_signed_shear():
    rows = [{"timestamp_s": (index + 1) * plot.STEP, "wrist_actual_z_m": .8,
        "wrist_command_fk_z_m": .801, "object_delta_z_m": index * 1e-6,
        "table_z_n": 6., "hand_normal_z_n": 1., "hand_tangent_z_n": -3.,
        "hand_net_z_n": -2., "other_z_n": 0.} for index in range(200)]
    result = plot.phase_summary(rows, 4.)
    assert result["sampled_phase_duration_s"] == .4
    assert result["last_250ms"]["native_samples"] == 125
    assert result["statistics"]["table_minus_weight_n"]["mean"] == 2.
    assert result["statistics"]["hand_tangent_downward_sample_fraction"] == 1.


def test_command_fk_uses_recorded_logical_joints_and_does_not_integrate():
    root = ET.Element("mujoco")
    body = ET.SubElement(root, "worldbody")
    for index in range(7):
        body = ET.SubElement(body, "body", name="link7" if index == 6 else f"link{index}",
            pos="0 0 0" if index == 0 else "0 0 .1")
        ET.SubElement(body, "joint", type="hinge", axis="0 1 0")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    fk = plot.CommandKinematics(model)
    command = np.zeros(7)
    assert fk.height_at(command) == pytest.approx(.6)
    command[0] = np.pi / 2.
    assert fk.height_at(command) == pytest.approx(0., abs=1e-12)
    assert fk.data.time == 0.
    assert np.array_equal(command, [np.pi / 2., 0., 0., 0., 0., 0., 0.])


@pytest.mark.parametrize("attempt_count", [1, 2])
def test_extract_retains_align_failure_and_absent_later_phases(monkeypatch, attempt_count):
    class Model:
        ngeom = 0

        def geom(self, name):
            assert name == "target_geom"
            return SimpleNamespace(id=0)
    class Kinematics:
        def __init__(self, _):
            pass

        def height_at(self, command):
            return .8
    native, trace = [], []
    for attempt in range(attempt_count):
        for phase in ("SEARCH", "APPROACH", "ALIGN", "RELEASE", "RECOVER"):
            time = (len(native) + 1) * plot.STEP
            native.append({"timestamp": time, "state": phase, "target_qpos": [0., 0., .38 - len(native) * 1e-5, 1., 0., 0., 0.],
                "motor_commands": [0.] * 29})
            trace.append({"timestamp": time, "state": phase, "wrist_position": [0., 0., .8],
                "arm_command": [0.] * 7, "hand_command": [0.] * 22})
    monkeypatch.setattr(plot, "CommandKinematics", Kinematics)
    monkeypatch.setattr(plot, "json_array", lambda path: iter(native if path.name == "native" else trace))
    monkeypatch.setattr(plot, "vertical_forces", lambda *args: {name: 0. for name in plot.FORCES})
    result = plot.extract({"model": Model(), "inputs": {name: {"path": name} for name in ("native", "trace")},
        "weight_n": 4., "attempts_recorded": [{"attempt": index + 1, "failure": "ALIGN_UNSETTLED"} for index in range(attempt_count)],
        "native_samples_expected": len(native), "duration_s": len(native) * plot.STEP})
    assert len(result["attempts"]) == attempt_count
    for attempt in result["attempts"]:
        assert attempt["failure"] == "ALIGN_UNSETTLED"
        assert attempt["phases"]["ALIGN"]["executed"] is True
        assert attempt["phases"]["CLOSE"]["executed"] is False
        assert attempt["phases"]["TRIAL_LIFT"]["executed"] is False
        assert attempt["series"]["ALIGN"][0]["alignment_goal_z_m"] is None
        assert attempt["series"]["ALIGN"][0]["object_delta_z_m"] == pytest.approx(-2e-5)
