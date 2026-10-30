#!/usr/bin/env python3
"""Latched, proprioception-anchored pressure escape; frozen R4/R5 stay intact."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import textwrap

import mujoco
import numpy as np

import develop_public_ycb_force_band_r4 as r4


PROJECT = r4.PROJECT
PHASE_SECONDS = dict(r4.PHASE_SECONDS)
RELIEF_PARAMETERS = {
    "joint_trigger_n": 6.,
    "wrist_trigger_n": 7.,
    "low_pressure_n": 1.,
    "continuous_low_seconds": .2,
    "initial_retreat_m": .006,
    "retreat_extension_m": .003,
    "retreat_extension_seconds": .15,
    "maximum_retreat_m": .03,
    "retreat_endpoint_tolerance_m": .002,
    "joint_reference_error_limit_rad": .02,
    "joint_admittance_rad_per_nm": .08,
    "latched_patch_pressure_cap_n": 7.,
    "scope": "Pressure escape using initial overloaded tactile patches and actual robot joints; unchanged physical gains, contact parameters, torque caps and strict assessment",
}


def feasible_retreat_direction(normal, approach_axis, *, prohibit_downward):
    normal = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or norm < .05:
        return np.array([0., 0., 1.]), "NO_USABLE_NORMAL_UPWARD_FALLBACK"
    normal = normal / norm
    if not prohibit_downward or normal[2] >= 0.:
        return normal, "TACTILE_OUTWARD_NORMAL"
    horizontal = normal.copy()
    horizontal[2] = 0.
    length = float(np.linalg.norm(horizontal))
    if length >= .05:
        return horizontal / length, "DOWNWARD_NORMAL_PROJECTED_TO_TABLE_TANGENT"
    back = -np.asarray(approach_axis, dtype=float)
    back[2] = 0.
    length = float(np.linalg.norm(back))
    if length < .05:
        back, length = np.array([-1., 0., 0.]), 1.
    back /= length
    if back @ normal < 0.:
        back = -back
    return back, "NEAR_VERTICAL_DOWNWARD_NORMAL_USING_APPROACH_TANGENT"


class LatchedPressureGuard(r4.TactilePressureGuard):
    def __init__(self, model, friction_coefficient):
        super().__init__(model, friction_coefficient)
        self.robot_hand = None
        self.estimated = None
        self.escape = None
        self.current_phase = None

    def begin(self, state, data):
        self.state = self.current_phase = state.value
        self.baseline = {finger: value.normal for finger, value in self.read(data).items()}
        if self.robot_hand is not None:
            self.robot_hand.phase_begin(state.value, data)

    def _start_escape(self, state, kin, data, readings, failure):
        pressures = {finger: value.normal for finger, value in readings.items()}
        finger = max(pressures, key=pressures.get)
        raw_normal = self.world_normals[finger].copy()
        if self.robot_hand is not None and self.robot_hand.first_trigger_normal is not None:
            raw_normal = self.robot_hand.first_trigger_normal.copy()
            finger = self.robot_hand.first_trigger_finger
        confirmed = bool(self.estimated is not None and self.estimated.payload_confirmed)
        load = 0. if self.robot_hand is None else self.robot_hand.table_support(data)
        prohibit_down = not confirmed or load > .1
        direction, reason = feasible_retreat_direction(raw_normal, kin.approach_axis,
            prohibit_downward=prohibit_down)
        self.escape = {"start_s": float(data.time), "start_phase": state.value,
            "origin_world_m": data.xpos[kin.body].copy(),
            "direction_world": direction, "raw_normal_world": raw_normal,
            "direction_reason": reason, "trigger_finger": finger,
            "prohibit_downward": prohibit_down, "table_load_n_at_trigger": load,
            "distance_m": RELIEF_PARAMETERS["initial_retreat_m"],
            "last_extension_s": float(data.time), "low_since_s": None,
            "endpoint_reached": False, "solution": None,
            "hold_until_phase_end": state.value in ("CLOSE", "APPROACH", "ALIGN")}
        if self.robot_hand is not None:
            self.robot_hand.hold_for_escape(data)
        try:
            self._solve_escape(kin, data, failure)
        except failure:
            self.escape = None
            raise

    def _solve_escape(self, kin, data, failure):
        escape = self.escape
        endpoint = escape["origin_world_m"] + escape["direction_world"] * escape["distance_m"]
        kin.copy_open_hand(data)
        solved = kin.solve(endpoint - kin.origin, seed=kin.joint_positions(data), tolerance=.0001)
        if solved is None:
            raise failure("CONTACT_NORMAL_RETREAT_IK")
        if kin.minimum_table_clearance(solved) < .002:
            raise failure("CONTACT_NORMAL_RETREAT_IK")
        escape["solution"], escape["endpoint_world_m"] = solved, endpoint

    def guide(self, state, desired, command, kin, data, failure):
        readings = self.read(data)
        if self.robot_hand is not None:
            self.robot_hand.update_pressure_relief(data, readings)
        pressures = {finger: float(value.normal) for finger, value in readings.items()}
        peak = max(pressures.values())
        phase = state.value
        threshold = 4.5 if phase in ("SEARCH", "APPROACH", "ALIGN", "RECOVER") else RELIEF_PARAMETERS["wrist_trigger_n"]
        hand_latched = self.robot_hand is not None and self.robot_hand.emergency_active
        if self.escape is None and (peak > threshold or hand_latched):
            self._start_escape(state, kin, data, readings, failure)
        mode, reason, direction = "pass", None, None
        scale = 1.
        if self.escape is not None:
            escape = self.escape
            low = peak <= RELIEF_PARAMETERS["low_pressure_n"]
            if low and escape["low_since_s"] is None:
                escape["low_since_s"] = float(data.time)
            elif not low:
                escape["low_since_s"] = None
            error = float(np.linalg.norm(data.xpos[kin.body] - escape["endpoint_world_m"]))
            escape["endpoint_reached"] = error <= RELIEF_PARAMETERS["retreat_endpoint_tolerance_m"]
            low_seconds = 0. if escape["low_since_s"] is None else float(data.time - escape["low_since_s"])
            unloaded = low_seconds >= RELIEF_PARAMETERS["continuous_low_seconds"] and escape["endpoint_reached"]
            phase_hold = escape["hold_until_phase_end"] and phase == escape["start_phase"]
            if unloaded and not phase_hold:
                self.escape = None
                self.baseline = pressures.copy()
                mode = "CONTINUOUS_LOW_PRESSURE_RETREAT_FINISHED"
            else:
                if (not unloaded and escape["endpoint_reached"] and
                        data.time - escape["last_extension_s"] >= RELIEF_PARAMETERS["retreat_extension_seconds"]):
                    escape["distance_m"] = min(RELIEF_PARAMETERS["maximum_retreat_m"],
                        escape["distance_m"] + RELIEF_PARAMETERS["retreat_extension_m"])
                    escape["last_extension_s"] = float(data.time)
                    try:
                        self._solve_escape(kin, data, failure)
                    except failure:
                        self.escape = None
                        raise
                desired = escape["solution"].copy()
                direction, reason = escape["direction_world"], escape["direction_reason"]
                mode = "LATCHED_TACTILE_ESCAPE_HOLD" if unloaded else "LATCHED_TACTILE_ESCAPE"
                if not unloaded and data.time - escape["start_s"] > 1.5:
                    self.latest = {"observation_timestamp_s": float(data.time), "state": phase,
                        "channel_pressures_n": pressures, "mode": mode,
                        "retreat_direction_world": direction.tolist(), "continuous_low_pressure_s": low_seconds,
                        "retreat_reason": reason, "target_pose_used": False}
                    self.events.append(self.latest)
                    raise failure("CONTACT_GUARD_NOT_RELIEVED")
        elif peak > 2.:
            scale = float(np.clip((threshold - peak) / max(1., threshold - 2.), 0., 1.))
            desired = command + scale * (desired - command)
            mode = "pressure_crawl" if scale > 0. else "pressure_brake"
        escape = self.escape
        self.latest = {"observation_timestamp_s": float(data.time), "state": phase,
            "channel_pressures_n": pressures,
            "tared_increase_n": max(pressures[channel] - self.baseline[channel] for channel in pressures),
            "mode": mode, "scale": scale, "retreat_reason": reason,
            "retreat_direction_world": None if direction is None else direction.tolist(),
            "escape_latched": escape is not None,
            "escape_started_s": None if escape is None else escape["start_s"],
            "escape_origin_world_m": None if escape is None else escape["origin_world_m"].tolist(),
            "escape_endpoint_world_m": None if escape is None else escape["endpoint_world_m"].tolist(),
            "escape_distance_m": None if escape is None else escape["distance_m"],
            "escape_prohibit_downward": None if escape is None else escape["prohibit_downward"],
            "initial_tactile_normal_world": None if escape is None else escape["raw_normal_world"].tolist(),
            "continuous_low_pressure_s": None if escape is None or escape["low_since_s"] is None else float(data.time - escape["low_since_s"]),
            "target_pose_used": False,
            "contact_normal_scope": "Initial overloaded hand tactile normal latched across later contact-feature flips"}
        self.events.append(self.latest)
        return desired


class ActualJointReliefHand(r4.WrenchAllocatedHand):
    def __init__(self, model, config, profile, reader, estimated):
        super().__init__(model, config, profile, reader, estimated)
        self.latched_fingers = {}
        self.emergency_active = False
        self.emergency_phase_hold = False
        self.first_trigger_normal = None
        self.first_trigger_finger = None
        self.last_relief_update_s = None
        self.actual_joint_references = {}
        self.posture_offsets = {}
        self.relief_events = []
        self.phase = None
        self.relief_torques = {}
        self.reference_errors = {}
        self.joint_limit_blocks_relief = {}

    def phase_begin(self, phase, data):
        self.phase = phase
        if not self.emergency_active:
            self.emergency_phase_hold = False
            self.first_trigger_normal, self.first_trigger_finger = None, None
            if self.actual_joint_references:
                self._synchronize_posture(data)

    def _synchronize_posture(self, data):
        actual = {}
        for name, aid in self.actuators.items():
            if name.startswith("left_"):
                joint = int(self.model.actuator_trnid[aid, 0])
                actual[name] = float(data.qpos[self.model.jnt_qposadr[joint]])
        for finger in r4.FINGERS:
            prefix = "left_" + finger + "_"
            if finger == "thumb":
                joints = (("MCP_FE", self.config.thumb_mcp_flexion_ratio),
                    ("IP", self.config.thumb_ip_flexion_ratio))
            else:
                joints = (("MCP_FE", self.profile.get("nonthumb_mcp_ratio", 1.)),
                    ("PIP", self.profile.get("nonthumb_pip_ratio", 1.4)),
                    ("DIP", self.profile.get("nonthumb_dip_ratio", .8)))
            denominator = sum(ratio * ratio for _, ratio in joints)
            self.closure[finger] = float(np.clip(sum(actual[prefix + name] * ratio for name, ratio in joints)
                / max(denominator, 1e-9), 0., 1.))
        self.opposition = actual["left_thumb_CMC_FE"]
        self.relief_offsets = {}
        nominal = r4.previous.ForceBandHand.angles(self)
        self.posture_offsets = {name: actual[name] - nominal[name] for name in actual}

    def _capture_patches(self, finger, data):
        patches = []
        for contact in self.reader.raw_contacts:
            if contact["finger"] != finger or contact["pressure_n"] <= .02:
                continue
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, contact["hand_body_name"])
            rotation = data.xmat[body].reshape(3, 3)
            patches.append({"hand_body": body, "hand_body_name": contact["hand_body_name"],
                "point_in_hand_link_m": rotation.T @ (np.asarray(contact["position_world_m"]) - data.xpos[body]),
                "normal_in_hand_link": rotation.T @ np.asarray(contact["normal_on_hand_world"]),
                "pressure_weight_n": min(RELIEF_PARAMETERS["latched_patch_pressure_cap_n"], contact["pressure_n"])})
        return patches

    def update_pressure_relief(self, data, readings):
        if self.last_relief_update_s is not None and data.time <= self.last_relief_update_s + 1e-12:
            return
        self.last_relief_update_s = float(data.time)
        for finger in r4.FINGERS:
            pressure = float(readings[finger].normal)
            if pressure > RELIEF_PARAMETERS["joint_trigger_n"] and finger not in self.latched_fingers:
                patches = self._capture_patches(finger, data)
                self.latched_fingers[finger] = {"trigger_s": float(data.time),
                    "trigger_pressure_n": pressure, "patches": patches, "low_since_s": None}
                if self.first_trigger_normal is None:
                    self.first_trigger_normal = self.reader.world_normals[finger].copy()
                    self.first_trigger_finger = finger
                self.emergency_phase_hold = True
            if finger in self.latched_fingers:
                latch = self.latched_fingers[finger]
                if pressure <= RELIEF_PARAMETERS["low_pressure_n"]:
                    if latch["low_since_s"] is None:
                        latch["low_since_s"] = float(data.time)
                else:
                    latch["low_since_s"] = None
        guard_active = self.reader.escape is not None
        removable = [finger for finger, latch in self.latched_fingers.items()
            if latch["low_since_s"] is not None and
            data.time - latch["low_since_s"] >= RELIEF_PARAMETERS["continuous_low_seconds"] and not guard_active]
        for finger in removable:
            del self.latched_fingers[finger]
        previously_active = self.emergency_active
        self.emergency_active = bool(self.latched_fingers) or guard_active
        if self.emergency_active:
            self._synchronize_posture(data)
            self._anchored_references(data)
        elif previously_active:
            self._synchronize_posture(data)
            if self.phase in ("RELEASE", "VERIFY", "RECOVER"):
                self.emergency_phase_hold = False
        if self.emergency_active or previously_active:
            self.relief_events.append({"timestamp_s": float(data.time), "phase": self.phase,
                "active": self.emergency_active, "phase_closure_hold": self.emergency_phase_hold,
                "pressures_n": {channel: float(value.normal) for channel, value in readings.items()},
                "latched_fingers": {finger: {key: value for key, value in latch.items() if key != "patches"}
                    for finger, latch in self.latched_fingers.items()},
                "initial_tactile_patch_frames": {finger: [{
                    "hand_body_name": patch["hand_body_name"],
                    "point_in_hand_link_m": patch["point_in_hand_link_m"].tolist(),
                    "normal_in_hand_link": patch["normal_in_hand_link"].tolist(),
                    "pressure_weight_n": patch["pressure_weight_n"]} for patch in latch["patches"]]
                    for finger, latch in self.latched_fingers.items()},
                "actual_joint_anchored_targets_rad": self.actual_joint_references.copy(),
                "latched_generalized_pressure_torque_nm": self.relief_torques.copy(),
                "reference_minus_actual_rad": self.reference_errors.copy(),
                "joint_limit_blocks_relief": self.joint_limit_blocks_relief.copy(),
                "scope": "Actuator references only; no qpos writes; each finger latches only its own overloaded tactile patches"})

    def hold_for_escape(self, data):
        self.emergency_phase_hold = self.emergency_active = True
        self._synchronize_posture(data)
        self._anchored_references(data)

    def _anchored_references(self, data):
        torques = np.zeros(self.model.nv)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for latch in self.latched_fingers.values():
            for patch in latch["patches"]:
                body = patch["hand_body"]
                rotation = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] + rotation @ patch["point_in_hand_link_m"]
                force = rotation @ patch["normal_in_hand_link"] * patch["pressure_weight_n"]
                mujoco.mj_jac(self.model, data, jp, jr, point, body)
                torques += jp.T @ force
        references = {}
        errors, generalized, blocked = {}, {}, {}
        for name, dof in self.joint_dofs.items():
            joint = int(self.model.actuator_trnid[self.actuators[name], 0])
            actual = float(data.qpos[self.model.jnt_qposadr[joint]])
            delta = float(np.clip(RELIEF_PARAMETERS["joint_admittance_rad_per_nm"] * torques[dof],
                -RELIEF_PARAMETERS["joint_reference_error_limit_rad"],
                RELIEF_PARAMETERS["joint_reference_error_limit_rad"]))
            references[name] = float(np.clip(actual + delta, *self.model.jnt_range[joint]))
            errors[name] = references[name] - actual
            generalized[name] = float(torques[dof])
            if torques[dof] * errors[name] < -1e-8:
                blocked[name] = {"actual_position_rad": actual,
                    "target_position_rad": references[name],
                    "generalized_pressure_torque_nm": float(torques[dof]),
                    "reason": "ORIGINAL_JOINT_RANGE_PREVENTS_REFERENCE_IN_RELIEF_DIRECTION"}
        self.actual_joint_references = references
        self.reference_errors, self.relief_torques = errors, generalized
        self.joint_limit_blocks_relief = blocked

    def joint_pressure_relief(self, readings, dt):
        if self.reader.last_data is not None:
            self.update_pressure_relief(self.reader.last_data, readings)

    def angles(self):
        if self.emergency_active or self.emergency_phase_hold:
            return self.actual_joint_references.copy()
        return {name: value + self.posture_offsets.get(name, 0.) for name, value in super().angles().items()}

    def regulate(self, readings, dt, carrying):
        external_readings = self.reader.read(self.reader.last_data)
        super().regulate(external_readings, dt, carrying)
        if self.emergency_active or self.emergency_phase_hold:
            self.ready_seconds = 0.
            self.feedback_events[-1].update(ready_seconds=0.,
                safety_closure_hold=True, latched_pressure_fingers=list(self.latched_fingers),
                actual_joint_anchored_targets_rad=self.actual_joint_references.copy())

    def release_pressure_relief(self, readings, dt):
        external_readings = self.reader.read(self.reader.last_data)
        self.joint_pressure_relief(external_readings, dt)

    def write(self, data):
        self.update_pressure_relief(data, self.reader.read(data))
        super().write(data)


def controller_for(kind, directory, observations, guard, profile, mass_kg):
    original, source, _, estimated = r4.controller_for(kind, directory, observations, guard, profile, mass_kg)
    hands = []
    guard.estimated = estimated

    def make_hand(model, config):
        hand = ActualJointReliefHand(model, config, profile, guard, estimated)
        hands.append(hand)
        guard.robot_hand = hand
        return hand

    namespace = dict(original.__globals__)
    namespace.update(HandGraspController=make_hand, _phase_seconds=PHASE_SECONDS,
        _feasible_retreat_direction=feasible_retreat_direction)
    tree = ast.parse(source)
    recovery_direction = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "direction" for target in node.targets)
        and isinstance(node.value, ast.IfExp) and "norm <" in ast.unparse(node.value.test))
    recovery_direction.value = ast.parse("_feasible_retreat_direction(direction, kin.approach_axis, prohibit_downward=True)[0]",
        mode="eval").body
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_latched_relief_r6>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r4.snapshot(profile_path)
    path = Path(__file__).resolve()
    sources[path.relative_to(PROJECT).as_posix()] = r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r4.run)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = {"public_ycb_tactile_wrench_force_band_development_r4": "public_ycb_latched_joint_relief_development_r6",
                "tactile_wrench_band_r4": "latched_joint_relief_r6",
                "public_ycb_tactile_wrench_force_band_r4": "public_ycb_latched_joint_relief_r6"}.get(node.value, node.value)
    protocol_write = next(i for i, node in enumerate(tree.body[0].body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "previous.atomic_json")
    tree.body[0].body[protocol_write:protocol_write] = ast.parse("""
protocol['latched_pressure_escape'] = dict(_relief_parameters)
protocol['force_control'] = 'All-external hand tactile pressures; first finger overload>6N freezes nominal closure and resets position references to actual joint positions plus latched tactile-Jacobian admittance, delta clipped to.02rad before original joint-range clamp. Soft-limit violations may make final reference error larger or block the relief direction; these are logged and never certified as unloading. Emergency reference resets can exceed ordinary closure command increments. Physical gains/contact/torque caps unchanged; no target-pose/COM input. Readiness and independent assessment unchanged.'
protocol['guard'] = 'Latched initial tactile-normal escape,6mm extended only to30mm, requiring all channels<=1N continuously.2s plus actual wrist endpoint before clearance. Unconfirmed/table-supported payload cannot retreat downward. No single low-force sample or contact-feature reversal clears the escape.'
""").body
    capture = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "capture_phase")
    pixels = next(i for i, node in enumerate(capture.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pixels" for target in node.targets))
    capture.body[pixels + 1:pixels + 1] = ast.parse("""
from OpenGL import GL
protocol.setdefault('actual_overview_gl', {})
for gl_name, gl_token in (('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)):
    gl_value = GL.glGetString(gl_token)
    protocol['actual_overview_gl'][gl_name] = None if gl_value is None else gl_value.decode('ascii', errors='replace')
""").body
    artifacts = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) and ast.unparse(target) == "row['artifacts']" for target in node.targets))
    artifacts.value.keys.append(ast.Constant(value="pressure_relief"))
    artifacts.value.values.append(ast.parse("previous.write_gzip(directory / 'pressure_relief.json.gz', [event for hand in hands for event in hand.relief_events])", mode="eval").body)
    namespace = dict(r4.run.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PHASE_SECONDS=PHASE_SECONDS,
        TactilePressureGuard=LatchedPressureGuard, _relief_parameters=RELIEF_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_latched_relief_runner_r6>", "exec"), namespace)
    return namespace["run"]


run = _runner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", help="KIND:SEED, all three reused development identities by default")
    args = parser.parse_args()
    cases = r4.previous.CASES if args.cases is None else tuple((value.rsplit(":", 1)[0],
        int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind not in r4.previous.KINDS or seed < 0 for kind, seed in cases):
        parser.error("Unique public kinds and nonnegative seeds required")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
