#!/usr/bin/env python3
"""Independently check a frozen R28 probe, including truthful aborted branches.

No controller is imported, no frozen Python is executed and no physics is
stepped. A valid negative experiment is accepted evidence, not task success.
Journal write counters are checked as assertions, not independently replayed
physics. Prefix contacts retain their declared step-start cache scope; they
must not be used as integrated-state held-lift labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")
ACTIVE = ("thumb", "middle", "ring")
CONDITIONS = ("STATIC_HOLD", "MOVING_WITHOUT_TRANSPORT", "MOVING_WITH_TRANSPORT")
CONTROLLED_NAMES = tuple("joint" + str(i) for i in range(1, 8)) + (
    "left_thumb_MCP_FE", "left_thumb_IP", "left_middle_MCP_FE", "left_middle_PIP",
    "left_middle_DIP", "left_ring_MCP_FE", "left_ring_PIP", "left_ring_DIP")
DT = .002


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array(value, shape=None):
    result = np.asarray(value)
    require(result.dtype.kind in "fiu", "non-numeric or boolean numeric field")
    result = result.astype(float)
    require(np.isfinite(result).all(), "nonfinite numeric field")
    require(shape is None or result.shape == shape, "numeric field shape mismatch")
    return result


def close(actual, expected, label, tolerance=1e-9):
    expected = array(expected)
    actual = array(actual, expected.shape)
    require(np.allclose(actual, expected, atol=tolerance, rtol=0.), label)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(np.isfinite(value), "nonfinite JSON number")


def load_json(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt") as stream:
        result = json.load(stream)
    finite_tree(result)
    return result


def artifact(directory, record):
    path = (directory / record["path"]).resolve()
    require(path.parent == directory.resolve(), "artifact path must remain in experiment directory")
    require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
            "changed artifact: " + path.name)
    return load_json(path, compressed=True)


def verify_source_archive(directory, protocol):
    before, after = protocol["source_hashes_before"], protocol["source_hashes_after"]
    require(before and before == after and protocol["source_changed_during_run"] is False,
            "source changed during experiment")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(before), "source archive member mismatch")
        for name, digest in before.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "unsafe source archive path")
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest,
                    "source archive bytes mismatch: " + name)
    return len(before)


def channels_from_native(model, record):
    """Recompute all six external-hand load sums from raw contact wrenches."""
    totals, patches = dict.fromkeys(CHANNELS, 0.), []
    array(record["motor_commands"], (model.nu,))
    array(record["target_qpos"], (7,))
    array(record["target_position_world_m"], (3,))
    array(record["target_spatial_velocity_world"], (6,))
    hand_joints = {model.joint(j).name for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and (model.joint(j).name or "").startswith("left_")}
    require(set(record["actual_hand_joints"]) == hand_joints, "incomplete actual hand joint coverage")
    for values in record["actual_hand_joints"].values():
        array(values["position_rad"], ())
        array(values["velocity_rad_s"], ())
    for index, contact in enumerate(record["native_contacts"]):
        gids = contact["geom_ids"]
        require(len(gids) == 2 and all(type(g) is int and 0 <= g < model.ngeom for g in gids),
                "invalid native geometry ids")
        wrench = array(contact["force_torque_contact_frame"], (6,))
        frame = array(contact["contact_frame_world_rows"], (3, 3))
        point = array(contact["position_world_m"], (3,))
        array(contact["distance_m"], ())
        names = [model.body(int(model.geom_bodyid[g])).name or "" for g in gids]
        sides = [name.startswith("left_") for name in names]
        if sides[0] == sides[1]:
            continue
        side = 0 if sides[0] else 1
        name = names[side]
        channel = next((finger for finger in CHANNELS[:-1] if name.startswith("left_" + finger + "_")), "palm")
        pressure = abs(float(wrench[0]))
        totals[channel] += pressure
        patches.append({"input_index": index, "finger": channel, "hand_body_name": name,
            "position_world_m": point.tolist(), "normal_on_object_world": (frame[0] * (1. if side == 0 else -1.)).tolist(),
            "measured_pressure_n": pressure})
    return totals, patches


def pressures(value):
    require(set(value) == set(CHANNELS), "six complete force channels required")
    result = array([value[name] for name in CHANNELS], (6,))
    require(np.all(result >= 0.), "negative normal channel load")
    return result


def check_tactile(record):
    totals = np.zeros(6)
    indices = []
    for patch in record["patches"]:
        finger = patch["finger"]
        require(finger in CHANNELS, "invalid patch channel")
        index = patch["input_index"]
        require(type(index) is int and index >= 0 and index not in indices, "duplicate patch index")
        indices.append(index)
        array(patch["position_world_m"], (3,))
        normal = array(patch["normal_on_object_world"], (3,))
        close(np.linalg.norm(normal), 1., "nonunit tactile normal", 1e-8)
        pressure = float(array(patch["measured_pressure_n"], ()))
        require(pressure >= 0., "negative patch load")
        totals[CHANNELS.index(finger)] += pressure
    close(totals, pressures(record["measured_pressures_n"]), "patch/channel totals disagree")
    return totals


def check_patches(actual, expected):
    require(len(actual) == len(expected), "native/tactile patch count mismatch")
    for first, second in zip(actual, expected, strict=True):
        for key in ("input_index", "finger", "hand_body_name"):
            require(first[key] == second[key], "native/tactile patch identity mismatch")
        for key in ("position_world_m", "normal_on_object_world", "measured_pressure_n"):
            close(first[key], second[key], "native/tactile patch geometry or load mismatch")


def check_normal_targets(row, force_targets, condition, contact_config):
    reference, tactile = row["motion_reference"], row["before_tactile"]
    patches = [patch for patch in tactile["patches"] if patch["finger"] in ACTIVE]
    require(reference["patch_input_indices"] == [patch["input_index"] for patch in patches],
            "motion/contact row identity mismatch")
    position = array(reference["wrist_position_world_m"], (3,))
    linear = array(reference["achieved_linear_velocity_world_m_s"], (3,))
    angular = array(reference["achieved_angular_velocity_world_rad_s"], (3,))
    transport = [float(array(p["normal_on_object_world"], (3,)) @
        (linear + np.cross(angular, array(p["position_world_m"], (3,)) - position))) for p in patches]
    gain, cap = contact_config["pressure_velocity_gain_m_n_s"], contact_config["maximum_normal_velocity_m_s"]
    correction = [float(np.clip(gain * (force_targets[p["finger"]] -
        tactile["measured_pressures_n"][p["finger"]]), -cap, cap)) for p in patches]
    close(reference["rigid_transport_normal_velocity_m_s"], transport, "reference transport formula mismatch")
    close(row["rigid_transport_normal_velocity_m_s"], transport, "logged transport formula mismatch")
    close(row["pressure_correction_m_s"], correction, "pressure correction mismatch")
    expected = np.array(correction) + (transport if condition == "MOVING_WITH_TRANSPORT" else np.zeros(len(patches)))
    close(row["target_normal_velocity_m_s"], expected, "transport must be added to pressure correction")
    jacobian = array(reference["wrist_jacobian_world"])
    task = array(reference["task_reference_velocity_rad_s"], (jacobian.shape[1],))
    close(jacobian @ task, np.r_[linear, angular], "achieved twist differs from robot task reference")
    for index, name in enumerate(reference["joint_names"]):
        if name.startswith("left_"):
            close(task[index], 0., "finger closure leaked into rigid transport task")
            close(jacobian[:, index], np.zeros(6), "finger column in wrist Jacobian")


def servo_binding(model, names):
    joints, actuators = [], []
    for name in names:
        joint = model.joint(name).id
        candidates = [a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                      and int(model.actuator_trnid[a, 0]) == joint]
        require(len(candidates) == 1, "ambiguous controlled actuator")
        actuator = candidates[0]
        close(model.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.], "nonunit servo gear")
        require(model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
                and model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
                and model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE,
                "unsupported servo law")
        require(model.actuator_gainprm[actuator, 0] > 0., "nonpositive servo gain")
        close(model.actuator_biasprm[actuator, :2], [0., -model.actuator_gainprm[actuator, 0]],
              "servo bias is not position law")
        joints.append(joint)
        actuators.append(actuator)
    return np.array(joints), np.array(actuators)


def check_command(model, row, initial_bias, contact_config):
    state, solved = row["before_robot_state"], row["solver"]
    names = state["joint_names"]
    joints, actuators = servo_binding(model, names)
    n = len(names)
    q, velocity = array(state["qpos_rad"], (n,)), array(state["qvel_rad_s"], (n,))
    previous = array(state["actual_ctrl"], (model.nu,))
    command = array(solved["command_rad"], (n,))
    command_velocity = array(solved["command_velocity_rad_s"], (n,))
    close(command, previous[actuators] + DT * command_velocity, "command increment mismatch")
    bounds = solved["bounds"]
    lower, upper = array(bounds["command_lower_rad"], (n,)), array(bounds["command_upper_rad"], (n,))
    require(np.all(command >= lower - 2e-12) and np.all(command <= upper + 2e-12), "command outside solver box")
    rates = np.array([contact_config["finger_command_rate_rad_s"] if name.startswith("left_")
        else contact_config["arm_command_rate_rad_s"] for name in names])
    require(np.all(abs(command_velocity) <= rates + 1e-9), "reference rate exceeds declared bounds")
    joint_lower, joint_upper = model.jnt_range[joints].T.copy()
    for i, actuator in enumerate(actuators):
        if model.actuator_ctrllimited[actuator]:
            joint_lower[i] = max(joint_lower[i], model.actuator_ctrlrange[actuator, 0])
            joint_upper[i] = min(joint_upper[i], model.actuator_ctrlrange[actuator, 1])
    margin = contact_config["added_deflection_limit_rad"]
    components_lower, components_upper = [joint_lower, previous[actuators] - DT * rates, q + initial_bias - margin], [
        joint_upper, previous[actuators] + DT * rates, q + initial_bias + margin]
    kp, kd = model.actuator_gainprm[actuators, 0], -model.actuator_biasprm[actuators, 2]
    predicted = kp * (command - q) - kd * velocity
    close(solved["predicted_motor_torque_nm"], predicted, "frozen-state motor torque formula mismatch")
    if "contact_residual_m_s" in solved:
        residual = array(solved["contact_residual_m_s"])
        close(residual, array(solved["predicted_normal_velocity_m_s"]) - array(row["target_normal_velocity_m_s"]),
              "contact residual arithmetic mismatch")
        require(np.max(abs(residual)) <= contact_config["maximum_contact_residual_m_s"], "accepted excessive contact residual")
    if "task_residual_rad_s" in solved:
        close(solved["task_residual_rad_s"], command_velocity - array(row["motion_reference"]["task_reference_velocity_rad_s"]),
              "task residual arithmetic mismatch")
    lows, highs = [], []
    for joint, actuator in zip(joints, actuators, strict=True):
        limits = []
        if model.actuator_forcelimited[actuator]:
            limits.append(model.actuator_forcerange[actuator])
        if model.jnt_actfrclimited[joint]:
            limits.append(model.jnt_actfrcrange[joint])
        require(limits, "missing motor torque bounds")
        lows.append(max(pair[0] for pair in limits))
        highs.append(min(pair[1] for pair in limits))
    torque_low, torque_high = np.array(lows), np.array(highs)
    components_lower.append(q + (torque_low + kd * velocity) / kp)
    components_upper.append(q + (torque_high + kd * velocity) / kp)
    close(lower, np.maximum.reduce(components_lower), "solver lower intersection differs from declared boxes")
    close(upper, np.minimum.reduce(components_upper), "solver upper intersection differs from declared boxes")
    require(np.all(predicted >= torque_low - 1e-9) and np.all(predicted <= torque_high + 1e-9), "motor torque outside bounds")
    require(row["command_applied"] is True and solved["accepted"] is True, "rejected command was applied")
    actual = previous.copy()
    actual[actuators] = command
    close(row["actual_ctrl_for_step"], actual, "actual applied command differs from solver candidate")
    cache = row["native_dynamics_cache"]
    close(cache["actuator_length_rad"], q, "cache servo length is not step-start position")
    close(cache["actuator_velocity_rad_s"], velocity, "cache servo velocity is not step-start velocity")
    close(cache["actuator_force_nm"], predicted, "actual cached motor force differs from prediction")
    close(cache["motor_torque_prediction_residual_nm"], array(cache["actuator_force_nm"]) - predicted,
          "motor prediction residual mismatch")
    return float(np.max(abs(array(cache["actuator_force_nm"]) - predicted)))


def check_robot_state(actual, expected, label):
    require(actual["joint_names"] == expected["joint_names"], label + " joint order")
    for key in ("qpos_rad", "qvel_rad_s", "actual_ctrl", "timestamp_s"):
        close(actual[key], expected[key], label + " " + key)


def privileged_motion_metrics(model, native, *, initial_checkpoint_height_m, initial_episode_height_m,
                              target_geom_id, table_geom_id):
    """Describe refreshed branch truth only; do not feed labels to control."""
    hand_geoms = {g for g in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[g])).name or "").startswith("left_")}
    records, previous, held_since, held_seconds = [], None, None, 0.
    for raw in native:
        timestamp = float(raw["timestamp"])
        hand_loads, table_loads = [], []
        for contact in raw["native_contacts"]:
            pair = set(contact["geom_ids"])
            normal = abs(float(contact["force_torque_contact_frame"][0]))
            if target_geom_id in pair and pair & hand_geoms:
                hand_loads.append(normal)
            if pair == {target_geom_id, table_geom_id}:
                table_loads.append(normal)
        height = float(array(raw["target_qpos"], (7,))[2])
        loaded_hand, loaded_table = max(hand_loads, default=0.) > .05, max(table_loads, default=0.) > .05
        held = height > initial_episode_height_m + .07 and loaded_hand and not loaded_table
        if previous is not None and abs(timestamp - previous - DT) > 1e-9:
            held_since = None
        if held:
            held_since = timestamp if held_since is None else held_since
            held_seconds = max(held_seconds, timestamp - held_since)
        else:
            held_since = None
        previous = timestamp
        records.append(dict(timestamp_s=timestamp, free_joint_height_m=height,
            target_hand_normal_sum_n=sum(hand_loads), target_hand_normal_peak_n=max(hand_loads, default=0.),
            target_table_normal_sum_n=sum(table_loads), target_table_normal_peak_n=max(table_loads, default=0.),
            target_hand_loaded=loaded_hand, target_table_loaded=loaded_table))
    heights = [row["free_joint_height_m"] for row in records]
    return {"target_privileged_evaluation_only": True, "sample_scope": "Refreshed integrated-state branch samples only",
        "samples": len(records), "initial_checkpoint_height_m": initial_checkpoint_height_m,
        "initial_episode_height_m": initial_episode_height_m,
        "maximum_free_joint_rise_from_checkpoint_m": None if not heights else max(heights) - initial_checkpoint_height_m,
        "final_free_joint_rise_from_checkpoint_m": None if not heights else heights[-1] - initial_checkpoint_height_m,
        "maximum_free_joint_rise_from_episode_start_m": None if not heights else max(heights) - initial_episode_height_m,
        "held_over_70mm_loaded_unsupported_seconds": held_seconds,
        "branch_observed_held_lift": bool(held_seconds + 1e-9 >= .1),
        "loaded_hand_samples": sum(row["target_hand_loaded"] for row in records),
        "loaded_table_samples": sum(row["target_table_loaded"] for row in records),
        "loaded_hand_without_loaded_table_samples": sum(row["target_hand_loaded"] and not row["target_table_loaded"] for row in records),
        "maximum_target_table_normal_sum_n": max((row["target_table_normal_sum_n"] for row in records), default=0.),
        "maximum_target_hand_normal_sum_n": max((row["target_hand_normal_sum_n"] for row in records), default=0.),
        "first_refreshed_sample": None if not records else records[0],
        "last_refreshed_sample": None if not records else records[-1],
        "full_task_evaluated": False,
        "scope": "Loaded hand contact is not an equilibrium/support certificate. No prefix cached pose/contact held labels used."}


def audit_branch(model, result, rows, native, checkpoint_state, contact_config):
    condition = result["condition"]
    require(condition in CONDITIONS and result["task_success"] is None, "probe cannot claim task success")
    require(rows, "branch has no before-state evidence")
    check_robot_state(result["initial_robot_state"], checkpoint_state, "branch checkpoint start")
    check_robot_state(rows[0]["before_robot_state"], checkpoint_state, "first row checkpoint start")
    initial_bias = array(result["fixed_capture_deflection_rad"], (len(checkpoint_state["joint_names"]),))
    _, actuator_ids = servo_binding(model, checkpoint_state["joint_names"])
    close(initial_bias, array(checkpoint_state["actual_ctrl"])[actuator_ids] - array(checkpoint_state["qpos_rad"]),
          "initial deflection differs between branches")
    force_targets = result["force_targets_n"]
    pressures(force_targets)
    peaks = {key: np.zeros(6) for key in ("before", "cache", "after")}
    max_torque_error, integrated, previous_after, previous_tactile = 0., 0, None, None
    for index, row in enumerate(rows):
        before = float(array(row["before_time_s"], ()))
        close(before, checkpoint_state["timestamp_s"] + integrated * DT, "noncontiguous branch command timestamps")
        if previous_after is not None:
            check_robot_state(row["before_robot_state"], previous_after, "after-to-before continuity")
        close(row["before_robot_state"]["timestamp_s"], before, "before robot timestamp")
        if "before_tactile" in row:
            close(row["before_tactile"]["timestamp_s"], before, "before tactile timestamp")
            peaks["before"] = np.maximum(peaks["before"], check_tactile(row["before_tactile"]))
            check_patches(sorted(row["before_canonical_patches"], key=lambda p: p["input_index"]),
                          row["before_tactile"]["patches"])
            if previous_tactile is not None:
                check_patches(row["before_tactile"]["patches"], previous_tactile["patches"])
                close(pressures(row["before_tactile"]["measured_pressures_n"]),
                      pressures(previous_tactile["measured_pressures_n"]), "after-to-before tactile continuity")
        if "target_normal_velocity_m_s" in row:
            check_normal_targets(row, force_targets, condition, contact_config)
        if row["command_applied"] is not True:
            require(index == len(rows) - 1 and "abort" in row, "unapplied row must be terminal abort")
            continue
        max_torque_error = max(max_torque_error, check_command(model, row, initial_bias, contact_config))
        cache, after = row["native_dynamics_cache"], row["after_tactile"]
        close(cache["step_start_s"], before, "cache step-start timestamp")
        close(cache["step_end_s"], before + DT, "cache step-end timestamp")
        close(cache["timestamp_s"], before + DT, "cache timestamp")
        cached_totals, cached_patches = channels_from_native(model, cache["native_sample"])
        close(cache["native_sample"]["timestamp"], before + DT, "raw cache timestamp")
        close(pressures(cache["channel_forces_n"]), pressures(cached_totals), "raw cache force mismatch")
        check_patches(sorted(cache["canonical_patches"], key=lambda p: p["input_index"]), cached_patches)
        close(cache["native_sample"]["motor_commands"], row["actual_ctrl_for_step"], "raw cache motor command mismatch")
        peaks["cache"] = np.maximum(peaks["cache"], pressures(cached_totals))
        require(integrated < len(native), "integrated step has no refreshed native record")
        raw = native[integrated]
        close(raw["timestamp"], before + DT, "native branch timestamp")
        close(after["timestamp_s"], before + DT, "after tactile timestamp")
        after_totals, after_patches = channels_from_native(model, raw)
        close(check_tactile(after), pressures(after_totals), "refreshed native force mismatch")
        check_patches(after["patches"], after_patches)
        check_patches(sorted(row["after_canonical_patches"], key=lambda p: p["input_index"]), after_patches)
        close(raw["motor_commands"], row["actual_ctrl_for_step"], "refreshed native motor command mismatch")
        peaks["after"] = np.maximum(peaks["after"], pressures(after_totals))
        previous_after = row["after_robot_state"]
        previous_tactile = after
        close(previous_after["timestamp_s"], before + DT, "after robot timestamp")
        close(previous_after["actual_ctrl"], row["actual_ctrl_for_step"], "after motor command changed")
        close(row["after_wrist_pose"]["timestamp_s"], before + DT, "after wrist timestamp")
        integrated += 1
    require(integrated == len(native) == result["native_steps"], "branch native step/sample count mismatch")
    audit = result["native_audit"]
    require(audit["integration_steps"] == integrated and audit["external_target_qpos_writes"] == 0
            and audit["external_target_qvel_writes"] == 0 and not audit["write_events"]
            and not audit["velocity_write_events"], "branch native write/count audit failed")
    completed = result["terminal"] == "completed" and integrated == result["required_native_steps"]
    require(result["completed"] is completed, "branch completion label mismatch")
    if not completed:
        require("abort" in rows[-1] and rows[-1]["abort"]["reason"] == result["terminal"], "missing real abort reason")
    union = np.maximum.reduce(list(peaks.values()))
    close(result["peak_channel_force_n"], np.max(union), "branch peak omits before/cache/after or failed row")
    return {"condition": condition, "completed": completed, "terminal": result["terminal"],
        "accepted_negative_evidence": not completed, "native_steps": integrated,
        "channel_peaks_n": dict(zip(CHANNELS, union.tolist(), strict=True)),
        "peak_channel_force_n": float(np.max(union)),
        "peaks_by_scope_n": {key: dict(zip(CHANNELS, value.tolist(), strict=True)) for key, value in peaks.items()},
        "maximum_cached_motor_torque_prediction_error_nm": max_torque_error, "task_success": None}


def audit_experiment(directory):
    directory = Path(directory).resolve()
    summary, protocol = load_json(directory / "summary.json"), load_json(directory / "protocol.json")
    require(summary["protocol"] == protocol, "summary/protocol mismatch")
    source_count = verify_source_archive(directory, protocol)
    require(protocol["input_hashes"] == protocol["input_hashes_after"] and not protocol["inputs_changed_during_run"],
            "baseline input changed during experiment")
    require(sha256(directory / "scene.xml") == protocol["input_hashes"]["scene.xml"], "copied scene identity mismatch")
    require(sha256(directory / "compiled_model.mjb") == protocol["compiled_model_sha256"], "compiled model identity mismatch")
    require(sha256(directory / "new_integration_checkpoint.npz") == protocol["checkpoint_sha256"], "checkpoint identity mismatch")
    require(summary["terminal"] == "all_branches_recorded" and summary["task_success"] is None
            and protocol["task_success"] is None, "probe metadata claims success or lacks branches")
    require(tuple(protocol["conditions"]) == CONDITIONS and tuple(row["condition"] for row in summary["conditions"]) == CONDITIONS,
            "matched condition denominator changed")
    contact_config, config = protocol["contact_configuration"], protocol["configuration"]
    require(contact_config["force_abort_n"] == 12., "force abort threshold changed")
    require(0. < config["probe_distance_m"] <= .005 and 0. < config["probe_speed_m_s"] <= .004,
            "experiment exceeds declared short motion probe")
    model = mujoco.MjModel.from_binary_path(str(directory / "compiled_model.mjb"))
    close(model.opt.timestep, DT, "native timestep is not 2ms", 1e-12)
    prefix = artifact(directory, protocol["prefix_native"])
    require(len(prefix) == round(contact_config["replay_until_s"] / DT), "prefix sample count mismatch")
    prefix_peak, window = np.zeros(6), []
    for tick, raw in enumerate(prefix, 1):
        close(raw["timestamp"], tick * DT, "prefix not contiguous from first native step")
        totals, _ = channels_from_native(model, raw)
        prefix_peak = np.maximum(prefix_peak, pressures(totals))
        if raw["timestamp"] > contact_config["replay_until_s"] - contact_config["average_window_s"] + 1e-9:
            window.append(pressures(totals))
    prefix_audit = protocol["prefix_native_audit"]
    require(prefix_audit["integration_steps"] == len(prefix) and prefix_audit["external_target_qpos_writes"] == 0
            and prefix_audit["external_target_qvel_writes"] == 0 and not prefix_audit["write_events"]
            and not prefix_audit["velocity_write_events"], "prefix native audit failed")
    replay = load_json(directory / "replay_verification.json")
    require(replay["accepted"] is True and len(replay["checks"]) == len(prefix)
            and all(row["accepted"] is True for row in replay["checks"]), "replay verification rejected")
    expected_targets = np.mean(window, axis=0)
    expected_targets[0] -= .5
    expected_targets[2] -= .5
    expected_targets[3] += .5
    expected_targets[[1, 4, 5]] = 0.
    require(np.all(expected_targets[[0, 2, 3]] > 0.) and np.max(expected_targets) <= 6., "invalid fixed force targets")
    data = mujoco.MjData(model)
    with np.load(directory / "new_integration_checkpoint.npz", allow_pickle=False) as saved:
        mujoco.mj_setState(model, data, array(saved["state"]), int(saved["state_spec"]))
    first = summary["conditions"][0]["initial_robot_state"]
    names = first["joint_names"]
    require(tuple(names) == CONTROLLED_NAMES, "controlled arm/finger contract changed")
    joints, _ = servo_binding(model, names)
    checkpoint = {"joint_names": names, "qpos_rad": data.qpos[model.jnt_qposadr[joints]].tolist(),
        "qvel_rad_s": data.qvel[model.jnt_dofadr[joints]].tolist(), "actual_ctrl": data.ctrl.tolist(),
        "timestamp_s": float(data.time)}
    close(data.time, contact_config["replay_until_s"], "checkpoint not at prefix endpoint")
    close(data.ctrl, prefix[-1]["motor_commands"], "checkpoint/prefix command mismatch")
    audited = []
    common_first_tactile = None
    for result in summary["conditions"]:
        close(pressures(result["force_targets_n"]), expected_targets, "branch force targets differ from common prefix")
        rows = artifact(directory, result["artifacts"]["rows"])
        native = artifact(directory, result["artifacts"]["native"])
        if common_first_tactile is None:
            common_first_tactile = rows[0]["before_tactile"]
        else:
            check_patches(rows[0]["before_tactile"]["patches"], common_first_tactile["patches"])
            close(pressures(rows[0]["before_tactile"]["measured_pressures_n"]),
                  pressures(common_first_tactile["measured_pressures_n"]), "branch common initial forces differ")
        audited.append(audit_branch(model, result, rows, native, checkpoint, contact_config))
        address = int(model.jnt_qposadr[model.joint("target_free").id])
        audited[-1]["privileged_motion"] = privileged_motion_metrics(model, native,
            initial_checkpoint_height_m=float(data.qpos[address + 2]),
            initial_episode_height_m=float(prefix_audit["initial_target_qpos"][2]),
            target_geom_id=model.geom("target_geom").id, table_geom_id=model.geom("table").id)
    paired = all(row["completed"] for row in audited)
    require(summary["paired_comparison_eligible"] is paired, "paired-comparison label mismatch")
    whole_peak = max(float(np.max(prefix_peak)), *(row["peak_channel_force_n"] for row in audited))
    return {"format": "independent_r28_motion_probe_audit_v1", "accepted_evidence": True,
        "paired_comparison_eligible": paired, "source_archive_verified_files": source_count,
        "summary_sha256": sha256(directory / "summary.json"), "protocol_sha256": sha256(directory / "protocol.json"),
        "frozen_source_sha256": sha256(directory / "frozen_source.zip"),
        "audit_script_sha256": sha256(Path(__file__)), "prefix_native_samples": len(prefix),
        "prefix_channel_peaks_n": dict(zip(CHANNELS, prefix_peak.tolist(), strict=True)),
        "whole_episode_peak_n": whole_peak, "whole_episode_within_12n": whole_peak <= 12.,
        "conditions": audited, "task_success": None, "known_not_full_task": True,
        "scope": ("Finite 5mm development probe only: no 70mm held lift, transfer, placement, generalization, "
            "future force safety or arbitrary-clutter claim. Aborted branches remain valid negative evidence. "
            "Prefix contacts/poses have declared step-start cache scope, not synchronized integrated-state held labels. "
            "Native write counters are checked assertions; no independent dynamics replay or source execution performed.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = audit_experiment(args.directory)
    except (AuditError, KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        receipt = {"format": "independent_r28_motion_probe_audit_v1", "accepted_evidence": False,
            "error": str(error), "task_success": None, "known_not_full_task": True}
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, allow_nan=False))
    return int(not receipt["accepted_evidence"])


if __name__ == "__main__":
    raise SystemExit(main())
