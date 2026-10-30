"""Finger-mask regressions at the profile, observation and executable interfaces."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import mujoco
import numpy as np
import pytest

from kuka_sharpa_grasp.contact_policy import (
    CONTACT_CHANNELS, FINGER_NAMES, ContactPolicyError, FingerContactPolicy,
)
from kuka_sharpa_grasp.control import FINGERS
from kuka_sharpa_grasp.hand_grasp import HandGraspController
from kuka_sharpa_grasp.scene import SHARPA_XML
from tactile_wrench_allocator import CHANNELS, allocate_contact_wrench


PROJECT = Path(__file__).resolve().parents[1]


def policy(active=FINGER_NAMES, **kwargs):
    joints = {finger: tuple(f"left_{finger}_{suffix}" for suffix in
        (("MCP_FE", "IP") if finger == "thumb" else ("MCP_FE", "PIP", "DIP")))
        for finger in active}
    return FingerContactPolicy(active, joints, **kwargs)


def profile(active=FINGER_NAMES):
    return {"contact_closure": {finger: .3 if finger in active else 0. for finger in FINGER_NAMES},
        "force_targets_n": {finger: 2. if finger in active else 0. for finger in FINGER_NAMES}}


def pressures(active=FINGER_NAMES):
    return {finger: 2. if finger in active else 0. for finger in CONTACT_CHANNELS}


def patches(active=FINGER_NAMES):
    result = []
    for index, finger in enumerate(active):
        angle = 2. * np.pi * index / len(active)
        normal = [-np.cos(angle), -np.sin(angle), 0.]
        result.append({"input_index": 7 + index, "finger": finger,
            "hand_body_name": f"left_{finger}_distal", "position_world_m":
            [.03 * np.cos(angle), .03 * np.sin(angle), 0.],
            "normal_on_object_world": normal, "measured_pressure_n": 2.,
            "friction_coefficient": .8})
    return result


def observe(contract, raw=None, readings=None, **kwargs):
    return contract.validate_contacts(patches(contract.active_fingers) if raw is None else raw,
        pressures(contract.active_fingers) if readings is None else readings,
        controlled_joint_names=contract.controlled_joint_names, **kwargs)


def expect_reason(reason, function, *args, **kwargs):
    with pytest.raises(ContactPolicyError) as error:
        function(*args, **kwargs)
    assert error.value.reason == reason


def test_public_channel_order_matches_existing_reader_and_allocator():
    assert FINGER_NAMES == FINGERS
    assert CONTACT_CHANNELS == CHANNELS


def test_rotated_profile_cannot_silently_use_three_finger_control():
    candidate = json.loads((PROJECT / "results/public_ycb_geometry_20261022_r15_final/lowest_peak_load_profile.json").read_text())["ycb_can"]
    old_mask = policy(("thumb", "middle", "ring"))
    expect_reason("UNCONTROLLED_FORCE_TARGET", old_mask.validate_profile, candidate)
    all_loaded = policy(("thumb", "middle", "ring", "pinky"))
    checked = all_loaded.validate_profile(candidate)
    assert checked.force_targets_n["pinky"] == pytest.approx(1.5292951083876747)
    assert all_loaded.joints_by_finger["pinky"] == ("left_pinky_MCP_FE", "left_pinky_PIP", "left_pinky_DIP")


@pytest.mark.parametrize("active,groups,reason", [
    ((), {}, "INVALID_ACTIVE_FINGERS"),
    (("palm",), {"palm": ("left_palm_X",)}, "INVALID_ACTIVE_FINGERS"),
    (("thumb", "thumb"), {"thumb": ("left_thumb_IP",)}, "INVALID_NAMES"),
    (("thumb",), {}, "JOINT_FINGER_MASK_MISMATCH"),
    (("thumb",), {"thumb": (),}, "INVALID_FINGER_JOINTS"),
    (("thumb",), {"thumb": ("left_pinky_DIP",)}, "INVALID_FINGER_JOINTS"),
    (("thumb",), {"thumb": ("left_thumb_IP", "left_thumb_IP")}, "INVALID_NAMES"),
])
def test_invalid_declaration_is_rejected(active, groups, reason):
    expect_reason(reason, FingerContactPolicy, active, groups)


def test_policy_and_validated_values_are_detached_from_mutable_input():
    groups = {"thumb": ["left_thumb_IP"]}
    contract = FingerContactPolicy(("thumb",), groups)
    groups["thumb"].append("left_thumb_MCP_FE")
    assert contract.controlled_joint_names == ("left_thumb_IP",)
    values = profile(("thumb",))
    checked = contract.validate_profile(values)
    values["contact_closure"]["thumb"] = .9
    assert checked.contact_closure["thumb"] == .3
    with pytest.raises(TypeError):
        checked.force_targets_n["thumb"] = 4.
    with pytest.raises(TypeError):
        contract.joints_by_finger["pinky"] = ("left_pinky_DIP",)


@pytest.mark.parametrize("field,value", [
    ("normal_target_limit_n", float("inf")), ("contact_floor_n", float("nan")),
    ("contact_floor_n", 0.), ("contact_floor_n", 7.),
    ("uncontrolled_contact_tolerance_n", .05), ("normal_target_limit_n", True),
])
def test_invalid_thresholds_rejected(field, value):
    with pytest.raises(ContactPolicyError):
        policy(**{field: value})


@pytest.mark.parametrize("value", [True, np.bool_(True), "2", float("nan"), float("inf"), -.1])
def test_invalid_force_scalar_rejected(value):
    targets = pressures()
    targets["pinky"] = value
    expect_reason("INVALID_NUMBER", policy().validate_targets_n, targets)


def test_force_targets_cannot_hide_palm_load_or_unknown_missing_channel():
    contract = policy()
    targets = pressures()
    targets["palm"] = 1e-12
    expect_reason("UNCONTROLLED_FORCE_TARGET", contract.validate_targets_n, targets)
    targets = pressures()
    targets["pinky"] = 6.0001
    expect_reason("FORCE_TARGET_LIMIT", contract.validate_targets_n, targets)
    del targets["pinky"]
    expect_reason("INVALID_CHANNELS", contract.validate_targets_n, targets)
    targets = pressures()
    targets["unknown"] = 0.
    expect_reason("INVALID_CHANNELS", contract.validate_targets_n, targets)


def test_profile_rejects_uncontrolled_closure_and_force_without_acquisition():
    contract = policy(("thumb", "middle", "ring"))
    values = profile(contract.active_fingers)
    values["contact_closure"]["pinky"] = .3
    expect_reason("UNCONTROLLED_CLOSURE", contract.validate_profile, values)
    values = profile(contract.active_fingers)
    values["contact_closure"]["thumb"] = 0.
    expect_reason("FORCE_WITHOUT_ACQUISITION", contract.validate_profile, values)


def test_five_finger_observation_reaches_existing_pure_allocator_and_target_gate():
    contract = policy()
    observation = observe(contract)
    assert observation.observed_fingers == FINGER_NAMES
    assert [patch.input_index for patch in observation.patches] == [7, 8, 9, 10, 11]
    result = allocate_contact_wrench(observation.as_allocator_patches(), [0., 0., 0.], .1,
        measured_pressures_n=observation.measured_pressures_n)
    assert result["feasible"] is True
    assert contract.validate_targets_n(result["target_pressures_n"])["palm"] == 0.
    assert all(patch["input_index"] in [7, 8, 9, 10, 11]
        for patch in observation.as_allocator_patches())


def test_observation_without_timestamp_keeps_pure_allocator_call_compatible():
    observation = observe(policy())
    assert observation.timestamp_s is None
    assert all("timestamp_s" not in patch for patch in observation.as_allocator_patches())


@pytest.mark.parametrize("timestamp", [0., 1.25, np.float64(.002)])
def test_valid_sample_timestamp_is_carried_immutably_without_claiming_freshness(timestamp):
    observation = observe(policy(), timestamp_s=timestamp)
    assert observation.timestamp_s == float(timestamp)
    assert type(observation.timestamp_s) is float
    with pytest.raises(FrozenInstanceError):
        observation.timestamp_s = 42.


@pytest.mark.parametrize("timestamp", [True, False, np.bool_(True), -.002,
    float("nan"), float("inf"), -float("inf"), ".002", [.002], 1j])
def test_invalid_sample_timestamp_is_rejected(timestamp):
    expect_reason("INVALID_NUMBER", observe, policy(), timestamp_s=timestamp)


@pytest.mark.parametrize("kind", ["missing_pinky", "extra_arm", "duplicate"])
def test_observation_requires_exact_execution_mask(kind):
    contract = policy()
    mask = list(contract.controlled_joint_names)
    if kind == "missing_pinky":
        mask = [name for name in mask if "pinky" not in name]
    elif kind == "extra_arm":
        mask.append("joint7")
    else:
        mask.append(mask[0])
    expect_reason("INVALID_NAMES" if kind == "duplicate" else "CONTROL_MASK_MISMATCH",
        contract.validate_contacts, patches(), pressures(), controlled_joint_names=mask)


@pytest.mark.parametrize("channel", ["pinky", "palm"])
def test_unassociated_contact_on_uncontrolled_channel_cannot_be_filtered(channel):
    contract = policy(("thumb", "middle", "ring"))
    readings = pressures(contract.active_fingers)
    readings[channel] = .1
    expect_reason("UNCONTROLLED_CONTACT", observe, contract, readings=readings)


def test_raw_uncontrolled_patch_cannot_hide_behind_incorrect_channel_reading():
    contract = policy(("thumb", "middle", "ring"))
    raw = patches(contract.active_fingers) + patches(("pinky",))
    raw[-1]["input_index"] = 20
    expect_reason("UNCONTROLLED_CONTACT", observe, contract, raw=raw)


def test_contacts_missing_from_association_fail_tracking_but_allow_acquisition():
    contract = policy()
    raw = patches()[:-1]
    expect_reason("MISSING_ACTIVE_CONTACT", observe, contract, raw=raw)
    observation = observe(contract, raw=raw, require_all_active=False)
    assert observation.observed_fingers == FINGER_NAMES[:-1]
    assert observation.measured_pressures_n["pinky"] == 2.


def test_patch_sum_and_all_contact_reading_must_be_consistent():
    contract = policy()
    readings = pressures()
    readings["pinky"] = 1.
    expect_reason("PATCH_CHANNEL_LOAD_MISMATCH", observe, contract, readings=readings)
    del readings["palm"]
    expect_reason("INVALID_CHANNELS", observe, contract, readings=readings)


@pytest.mark.parametrize("field,value,reason", [
    ("input_index", True, "INVALID_PATCH_INDEX"),
    ("input_index", -1, "INVALID_PATCH_INDEX"),
    ("input_index", 1.5, "INVALID_PATCH_INDEX"),
    ("hand_body_name", "left_middle_distal", "PATCH_BODY_FINGER_MISMATCH"),
    ("normal_on_object_world", [0., 0., 0.], "INVALID_CONTACT_NORMAL"),
    ("normal_on_object_world", [float("nan"), 0., 0.], "INVALID_NUMBER"),
    ("position_world_m", [0., 0.], "INVALID_VECTOR"),
    ("position_world_m", [0., float("inf"), 0.], "INVALID_NUMBER"),
    ("friction_coefficient", -.1, "INVALID_NUMBER"),
    ("measured_pressure_n", "2", "INVALID_NUMBER"),
])
def test_patch_validation_rejects_corrupt_numeric_geometry_or_identity(field, value, reason):
    raw = patches()
    raw[0][field] = value
    expect_reason(reason, observe, policy(), raw=raw)


def test_duplicate_patch_identity_rejected_even_with_different_fingers():
    raw = patches()
    raw[1]["input_index"] = raw[0]["input_index"]
    expect_reason("INVALID_PATCH_INDEX", observe, policy(), raw=raw)


def test_measurement_above_target_cap_remains_visible_for_emergency_handler():
    raw, readings = patches(), pressures()
    raw[0]["measured_pressure_n"] = readings["thumb"] = 53.25
    observation = observe(policy(), raw=raw, readings=readings)
    assert observation.measured_pressures_n["thumb"] == 53.25
    assert observation.patches[0].measured_pressure_n == 53.25


def test_output_copies_do_not_allow_later_mutation_of_validated_observation():
    raw = patches()
    observation = observe(policy(), raw=raw)
    raw[0]["position_world_m"][0] = 99.
    exported = observation.as_allocator_patches()
    exported[0]["normal_on_object_world"][0] = 99.
    assert observation.patches[0].position_world_m[0] == .03
    assert observation.patches[0].normal_on_object_world[0] == -1.
    with pytest.raises(TypeError):
        observation.measured_pressures_n["pinky"] = 0.


@pytest.fixture(scope="module")
def sharpa_model():
    return mujoco.MjModel.from_xml_path(str(SHARPA_XML))


def test_real_sharpa_five_finger_bindings_are_executable_position_servos(sharpa_model):
    contract = policy()
    bindings = contract.bind_mujoco_model(sharpa_model)
    hand = HandGraspController(sharpa_model)
    posture = hand.angles()
    assert tuple(binding.name for binding in bindings) == contract.controlled_joint_names
    assert {binding.finger for binding in bindings} == set(FINGER_NAMES)
    assert len({binding.actuator_id for binding in bindings}) == len(bindings)
    for binding in bindings:
        assert hand.actuators[binding.name] == binding.actuator_id
        assert binding.name in posture
        assert binding.dof_index == sharpa_model.jnt_dofadr[binding.joint_id]
        assert binding.qpos_index == sharpa_model.jnt_qposadr[binding.joint_id]
        assert binding.command_range_rad[0] < binding.command_range_rad[1]


def miniature_model(actuator='<position name="left_thumb_IP" joint="left_thumb_IP" kp="5" kv=".1"/>',
                    *, joint_type="hinge", joint_range="-1 1"):
    return mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><worldbody><body><joint name="left_thumb_IP" '
        f'type="{joint_type}" range="{joint_range}"/><geom type="sphere" size=".01" mass=".1"/>'
        f'</body></worldbody><actuator>{actuator}</actuator></mujoco>')


def single_joint_policy():
    return FingerContactPolicy(("thumb",), {"thumb": ("left_thumb_IP",)})


@pytest.mark.parametrize("actuator,kind,reason", [
    ('<motor name="left_thumb_IP" joint="left_thumb_IP"/>', "hinge", "INVALID_POSITION_SERVO"),
    ('<position name="left_thumb_IP" joint="left_thumb_IP" gear="2" kp="5"/>', "hinge", "INVALID_JOINT_TRANSMISSION"),
    ('<position name="other" joint="left_thumb_IP" kp="5"/>', "hinge", "MISSING_EXECUTABLE_JOINT"),
    ('<position name="left_thumb_IP" joint="left_thumb_IP" kp="5"/>', "slide", "INVALID_JOINT_TRANSMISSION"),
    ('<position name="left_thumb_IP" joint="left_thumb_IP" kp="5" ctrlrange="2 3"/>', "hinge", "INVALID_COMMAND_RANGE"),
])
def test_model_binding_rejects_unsupported_command_interfaces(actuator, kind, reason):
    expect_reason(reason, single_joint_policy().bind_mujoco_model,
        miniature_model(actuator, joint_type=kind))


def test_joint_command_interval_is_intersection_of_hinge_and_servo_bounds():
    model = miniature_model('<position name="left_thumb_IP" joint="left_thumb_IP" '
        'kp="5" ctrlrange="-.2 .3"/>')
    bound, = single_joint_policy().bind_mujoco_model(model)
    assert bound.command_range_rad == (-.2, .3)
