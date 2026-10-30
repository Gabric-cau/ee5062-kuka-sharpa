#!/usr/bin/env python3
"""Declared, constrained load-transfer probe on the frozen R6 sidewall can."""

from __future__ import annotations

import argparse
import ast
from collections import deque
import inspect
from pathlib import Path
import textwrap

import numpy as np

import develop_public_ycb_force_band_r4 as r4
import develop_public_ycb_force_band_r6 as r6


PROJECT = r6.PROJECT
PHASE_SECONDS = dict(r6.PHASE_SECONDS)
EXPECTED_PROFILE_SHA256 = "7ffffa83c4ec1b5510537b3f5b82ad113b287b4aed18d57fad6a3b382a7f203e"
PROBE_PARAMETERS = {
    "preprobe_seconds": .25,
    "stable_window_seconds": .25,
    "maximum_pressure_span_n": .3,
    "maximum_channel_pressure_n": 6.,
    "maximum_palm_pressure_n": .05,
    "minimum_associated_active_finger_pressure_n": .35,
    "maximum_visual_age_s": 1.,
    "visual_refresh_seconds": .25,
    "maximum_solve_age_s": .1,
    "solve_interval_seconds": .04,
    "maximum_actual_wrist_speed_m_s": .01,
    "probe_distance_m": .005,
    "probe_endpoint_tolerance_m": .0005,
    "maximum_downward_probe_drift_m": .0005,
    "maximum_horizontal_probe_drift_m": .002,
    "confirmation_seconds": .25,
    "maximum_remaining_table_weight_fraction": .2,
    "scope": "Capacity1g admits a bounded experiment, not a held-object certificate; fixed-normal1g plus actual table unloading confirms only the5mm probe, never task success",
}


def contact_samples_match(current, solved):
    if len(current) != len(solved):
        return False
    for now, old in zip(current, solved, strict=True):
        if now["finger"] != old["finger"] or now["hand_body_name"] != old["hand_body_name"]:
            return False
        if np.linalg.norm(np.asarray(now["position_world_m"]) - old["position_world_m"]) > .001:
            return False
        if np.asarray(now["normal_on_hand_world"]) @ old["normal_on_hand_world"] < .99:
            return False
        if abs(now["pressure_n"] - old["pressure_n"]) > .1 + .2 * old["pressure_n"]:
            return False
    return True


class ProbeEstimatedState(r4.EstimatedContactState):
    def __init__(self, kind, mass_kg):
        super().__init__(kind, mass_kg)
        self.probe_active = False
        self.last_visual_attempt_s = None
        self.last_visual_attempt_accepted = None
        self.last_visual_attempt_reason = None

    def accept_visual(self, fitted, timestamp_s, wrist_position, wrist_rotation, phase):
        super().accept_visual(fitted, timestamp_s, wrist_position, wrist_rotation, phase)
        self.last_visual_attempt_s = float(timestamp_s)
        self.last_visual_attempt_accepted = bool(fitted.get("accepted"))
        self.last_visual_attempt_reason = fitted.get("reason")

    def estimated_pose(self, data, carrying):
        pose = super().estimated_pose(data, carrying and not self.probe_active)
        if self.probe_active:
            pose["probe_pose_scope"] = "Latest accepted actual RGB-D world pose; unconfirmed probe never propagates object pose with wrist motion"
        return pose


class LoadTransferProbeHand(r6.ActualJointReliefHand):
    def __init__(self, model, config, profile, reader, estimated):
        super().__init__(model, config, profile, reader, estimated)
        self.probe_active = False
        self.probe_stage = "IDLE"
        self.probe_eligible_seconds = 0.
        self.probe_confirmed_seconds = 0.
        self.pressure_window = deque()
        self.probe_association = {"accepted": False, "reason": "NOT_EVALUATED"}
        self.probe_geometry_15 = {"feasible": False, "measured_feasible": False}
        self.probe_allocation_1g = {"measured_capacity_feasible": False, "measured_feasible": False}
        self.probe_solved_sample = []
        self.last_probe_solve_s = -np.inf
        self.last_probe_visual_s = -np.inf
        self.probe_start_world_m = None
        self.probe_start_s = None
        self.probe_required_fingers = set()
        self.last_wrist_position = None
        self.last_wrist_timestamp_s = None
        self.actual_wrist_speed_m_s = 0.
        self.latest_probe = {}
        self.probe_events = []

    def phase_begin(self, phase, data):
        if phase == "CLOSE" and self.phase != phase:
            self.probe_eligible_seconds = self.probe_confirmed_seconds = 0.
            self.pressure_window.clear()
            self.probe_stage = "ACQUIRING"
        super().phase_begin(phase, data)

    def refresh_probe_solve(self, data, *, force=False):
        readings = self.reader.read(data)
        if not force and data.time - self.last_probe_solve_s < PROBE_PARAMETERS["solve_interval_seconds"] - 1e-9:
            return readings
        pose = self.estimated.estimated_pose(data, carrying=self.estimated.payload_confirmed)
        self.probe_association = r4.associate_tactile_contacts(self.reader.raw_contacts, pose,
            self.estimated.surface, now_s=float(data.time), pose_timestamp_s=self.estimated.pose_timestamp_s,
            max_pose_age_s=PROBE_PARAMETERS["maximum_visual_age_s"], max_surface_distance_m=.006)
        if self.probe_association["accepted"]:
            args = (self.probe_association["patches"], self.probe_association["estimated_com_world_m"],
                self.estimated.mass_kg)
            pressures = {finger: float(value.normal) for finger, value in readings.items()}
            self.probe_geometry_15 = r4.allocate_contact_wrench(*args,
                measured_pressures_n=pressures, pressure_limit_n=6., gravity_margin=1.5)
            self.probe_allocation_1g = r4.allocate_contact_wrench(*args,
                measured_pressures_n=pressures, pressure_limit_n=6., gravity_margin=1.)
        else:
            self.probe_geometry_15 = {"feasible": False, "measured_feasible": False}
            self.probe_allocation_1g = {"measured_capacity_feasible": False, "measured_feasible": False}
        self.last_probe_solve_s = float(data.time)
        self.probe_solved_sample = self.contact_sample(self.reader.raw_contacts)
        return readings

    def _snapshot_probe(self, data, readings):
        pressures = {finger: float(value.normal) for finger, value in readings.items()}
        now = float(data.time)
        visual_age = None if self.estimated.pose_timestamp_s is None else now - self.estimated.pose_timestamp_s
        solve_age = None if not np.isfinite(self.last_probe_solve_s) else now - self.last_probe_solve_s
        matches = contact_samples_match(self.contact_sample(self.reader.raw_contacts), self.probe_solved_sample)
        fresh = solve_age is not None and solve_age <= PROBE_PARAMETERS["maximum_solve_age_s"] and matches
        associated = {finger: 0. for finger in r4.FINGERS}
        for patch in self.probe_association.get("patches", []):
            if patch["finger"] in associated:
                associated[patch["finger"]] += patch["measured_pressure_n"]
        threshold = PROBE_PARAMETERS["minimum_associated_active_finger_pressure_n"]
        active = {finger for finger in r4.FINGERS if self.profile.get("contact_closure", {}).get(finger, 0.) > 0.}
        contacts = {finger for finger in active if associated[finger] >= threshold and pressures[finger] >= threshold}
        contact_ok = "thumb" in contacts and len(contacts - {"thumb"}) >= 2
        stable_seconds = 0. if len(self.pressure_window) < 2 else now - self.pressure_window[0][0]
        spans = {finger: 0. for finger in pressures}
        if self.pressure_window:
            spans = {finger: max(row[1][finger] for row in self.pressure_window) -
                min(row[1][finger] for row in self.pressure_window) for finger in pressures}
        stable = stable_seconds >= PROBE_PARAMETERS["stable_window_seconds"] - 1e-9 and max(spans.values()) <= PROBE_PARAMETERS["maximum_pressure_span_n"]
        visual_ok = visual_age is not None and visual_age <= PROBE_PARAMETERS["maximum_visual_age_s"]
        no_emergency = not self.emergency_active and not self.emergency_phase_hold and self.reader.escape is None
        reasons = []
        if max(pressures.values()) > PROBE_PARAMETERS["maximum_channel_pressure_n"]:
            reasons.append("PRESSURE")
        if pressures.get("palm", 0.) > PROBE_PARAMETERS["maximum_palm_pressure_n"]:
            reasons.append("PALM")
        if not no_emergency:
            reasons.append("EMERGENCY")
        if not visual_ok:
            reasons.append("VISUAL_STALE")
        if not self.probe_association.get("accepted"):
            reasons.append("CONTACT_ASSOCIATION")
        if not fresh:
            reasons.append("SOLVE_STALE_OR_CHANGED_CONTACT")
        if not contact_ok:
            reasons.append("CONTACT_LOSS")
        if not self.probe_geometry_15.get("feasible"):
            reasons.append("GEOMETRY_1_5G_INFEASIBLE")
        if not self.probe_allocation_1g.get("measured_capacity_feasible"):
            reasons.append("CAPACITY_1G_INFEASIBLE")
        table_load = float(self.table_support(data))
        displacement = None if self.probe_start_world_m is None else data.xpos[self.estimated.kin.body] - self.probe_start_world_m
        endpoint = bool(displacement is not None and np.linalg.norm(displacement - np.array([0., 0., PROBE_PARAMETERS["probe_distance_m"]])) <= PROBE_PARAMETERS["probe_endpoint_tolerance_m"])
        visual_after_start = (self.probe_start_s is not None and self.estimated.pose_timestamp_s is not None
            and self.estimated.pose_timestamp_s > self.probe_start_s + 1e-9)
        return {"timestamp_s": now, "phase": self.phase, "stage": self.probe_stage,
            "probe_active": self.probe_active, "pressures_n": pressures,
            "probe_start_s": self.probe_start_s,
            "accepted_visual_after_probe_start": visual_after_start,
            "pressure_window_s": stable_seconds, "pressure_spans_n": spans, "pressure_stable": stable,
            "visual_age_s": visual_age, "solve_age_s": solve_age, "cached_measurements_match": matches,
            "last_accepted_visual_timestamp_s": self.estimated.pose_timestamp_s,
            "last_visual_attempt_s": getattr(self.estimated, "last_visual_attempt_s", None),
            "last_visual_attempt_accepted": getattr(self.estimated, "last_visual_attempt_accepted", None),
            "last_visual_attempt_reason": getattr(self.estimated, "last_visual_attempt_reason", None),
            "sampled_solve_current": fresh, "associated_active_fingers": sorted(contacts),
            "associated_finger_pressures_n": associated, "contact_gate": contact_ok,
            "actual_wrist_position_world_m": data.xpos[self.estimated.kin.body].tolist(),
            "actual_wrist_speed_m_s": self.actual_wrist_speed_m_s,
            "actual_probe_displacement_m": None if displacement is None else displacement.tolist(),
            "actual_probe_endpoint_reached": endpoint,
            "table_load_n": table_load, "catalogue_weight_n": self.estimated.mass_kg * 9.81,
            "table_unloaded": table_load <= PROBE_PARAMETERS["maximum_remaining_table_weight_fraction"] * self.estimated.mass_kg * 9.81,
            "geometry_1_5g_feasible": bool(self.probe_geometry_15.get("feasible")),
            "original_exact_1_5g_feasible": bool(self.allocation.get("measured_feasible")),
            "original_exact_1_5g_ready_seconds": self.ready_seconds,
            "probe_exact_1_5g_feasible": bool(self.probe_geometry_15.get("measured_feasible")),
            "sampled_capacity_1g_feasible": bool(self.probe_allocation_1g.get("measured_capacity_feasible")),
            "sampled_fixed_normal_1g_feasible": bool(self.probe_allocation_1g.get("measured_feasible")),
            "probe_geometry_allocation_1_5g": self.probe_geometry_15,
            "probe_allocation_1g": self.probe_allocation_1g,
            "sampled_public_mesh_association": self.probe_association,
            "preprobe_reasons": reasons + ([] if stable else ["PRESSURE_NOT_STABLE"]) +
                ([] if self.actual_wrist_speed_m_s <= PROBE_PARAMETERS["maximum_actual_wrist_speed_m_s"] else ["SPEED"]),
            "motion_safety_reasons": reasons,
            "probe_eligible_seconds": self.probe_eligible_seconds,
            "probe_confirmed_seconds": self.probe_confirmed_seconds,
            "target_pose_used": False,
            "wrench_scope": "Sampled LP compatibility with tactile normal loads, not measured tangential equilibrium; capacity allows lower normal loads, fixed-normal equality does not"}

    def after_step(self, state, data, dt):
        self.phase = state.value
        wrist = data.xpos[self.estimated.kin.body].copy()
        if self.last_wrist_timestamp_s is not None and data.time > self.last_wrist_timestamp_s:
            self.actual_wrist_speed_m_s = float(np.linalg.norm(wrist - self.last_wrist_position) /
                (data.time - self.last_wrist_timestamp_s))
        self.last_wrist_position, self.last_wrist_timestamp_s = wrist, float(data.time)
        readings = self.reader.read(data)
        pressures = {finger: float(value.normal) for finger, value in readings.items()}
        self.pressure_window.append((float(data.time), pressures))
        while len(self.pressure_window) > 1 and data.time - self.pressure_window[1][0] >= PROBE_PARAMETERS["stable_window_seconds"] - 1e-9:
            self.pressure_window.popleft()
        relevant = state.value == "CLOSE" or self.probe_active or self.estimated.payload_confirmed
        if relevant:
            readings = self.refresh_probe_solve(data)
        row = self._snapshot_probe(data, readings)
        if state.value == "CLOSE":
            self.probe_eligible_seconds = self.probe_eligible_seconds + dt if not row["preprobe_reasons"] else 0.
        failure = None
        if self.probe_active:
            reasons = list(row["motion_safety_reasons"])
            if self.actual_wrist_speed_m_s > PROBE_PARAMETERS["maximum_actual_wrist_speed_m_s"]:
                reasons.insert(0, "SPEED")
            displacement = np.asarray(row["actual_probe_displacement_m"])
            if displacement[2] > PROBE_PARAMETERS["probe_distance_m"] + PROBE_PARAMETERS["probe_endpoint_tolerance_m"]:
                reasons.insert(0, "DISTANCE_EXCEEDED")
            if displacement[2] < -PROBE_PARAMETERS["maximum_downward_probe_drift_m"]:
                reasons.insert(0, "DOWNWARD_DRIFT")
            if np.linalg.norm(displacement[:2]) > PROBE_PARAMETERS["maximum_horizontal_probe_drift_m"]:
                reasons.insert(0, "HORIZONTAL_DRIFT")
            if not self.probe_required_fingers <= set(row["associated_active_fingers"]):
                reasons.insert(0, "CONTACT_LOSS")
            confirmed = (not reasons and row["actual_probe_endpoint_reached"] and row["table_unloaded"]
                and row["sampled_fixed_normal_1g_feasible"] and row["sampled_solve_current"]
                and row["accepted_visual_after_probe_start"])
            self.probe_confirmed_seconds = self.probe_confirmed_seconds + dt if confirmed else 0.
            if reasons:
                failure = "PROBE_" + reasons[0]
        elif self.estimated.payload_confirmed and state.value in ("TRIAL_LIFT", "TRANSFER"):
            reasons = list(row["motion_safety_reasons"])
            if not row["sampled_fixed_normal_1g_feasible"]:
                reasons.append("FIXED_NORMAL_1G_INFEASIBLE")
            if reasons:
                failure = "HELD_POLICY_" + reasons[0]
        row.update(probe_eligible_seconds=self.probe_eligible_seconds,
            probe_confirmed_seconds=self.probe_confirmed_seconds, stop_reason=failure)
        self.latest_probe = row
        self.probe_events.append(row)
        return failure

    def admit_probe(self, data):
        readings = self.refresh_probe_solve(data, force=True)
        row = self._snapshot_probe(data, readings)
        admitted = self.probe_eligible_seconds >= PROBE_PARAMETERS["preprobe_seconds"] and not row["preprobe_reasons"]
        row.update(decisive_current_solve=True, admission=admitted)
        self.probe_events.append(row)
        if not admitted:
            return False
        self.probe_active = self.estimated.probe_active = True
        self.probe_stage = "LOAD_TRANSFER_PROBE"
        self.probe_start_world_m = data.xpos[self.estimated.kin.body].copy()
        self.probe_start_s = float(data.time)
        self.probe_required_fingers = set(row["associated_active_fingers"])
        self.probe_confirmed_seconds = 0.
        return True

    def confirm_probe(self, data):
        readings = self.refresh_probe_solve(data, force=True)
        row = self._snapshot_probe(data, readings)
        confirmed = (self.probe_active and self.probe_confirmed_seconds >= PROBE_PARAMETERS["confirmation_seconds"]
            and not row["motion_safety_reasons"] and row["actual_probe_endpoint_reached"]
            and row["table_unloaded"] and row["sampled_fixed_normal_1g_feasible"]
            and row["accepted_visual_after_probe_start"]
            and self.actual_wrist_speed_m_s <= PROBE_PARAMETERS["maximum_actual_wrist_speed_m_s"]
            and self.probe_required_fingers <= set(row["associated_active_fingers"]))
        row.update(decisive_current_solve=True, confirmation=bool(confirmed))
        self.probe_events.append(row)
        if confirmed:
            self.probe_active = self.estimated.probe_active = False
            self.estimated.payload_confirmed = True
            self.probe_stage = "PROBE_CONFIRMED_ONLY_NOT_TASK_SUCCESS"
        return bool(confirmed)

    def abort_probe(self, reason):
        self.probe_active = self.estimated.probe_active = False
        self.estimated.payload_confirmed = False
        self.probe_stage = "ABORTED_" + reason

    def held_revalidated(self, data):
        if not self.estimated.payload_confirmed:
            return False
        readings = self.refresh_probe_solve(data, force=True)
        row = self._snapshot_probe(data, readings)
        return bool(not row["motion_safety_reasons"] and row["sampled_fixed_normal_1g_feasible"]
            and row["table_unloaded"] and self.probe_required_fingers <= set(row["associated_active_fingers"]))


def controller_for(kind, directory, observations, guard, profile, mass_kg):
    if kind != "ycb_can":
        raise ValueError("R8 is restricted to the separately declared sidewall-can probe")
    original, source, _, _ = r6.controller_for(kind, directory, observations, guard, profile, mass_kg)
    estimated, hands = ProbeEstimatedState(kind, mass_kg), []
    guard.estimated = estimated

    def make_hand(model, config):
        hand = LoadTransferProbeHand(model, config, profile, guard, estimated)
        hands.append(hand)
        guard.robot_hand = hand
        return hand

    tree = ast.parse(source)
    task = tree.body[0]
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    loop = next(node for node in move.body if isinstance(node, ast.For))
    loop.body[0:0] = ast.parse("""
if (hand.probe_active or _estimated.payload_confirmed) and data.time - hand.last_probe_visual_s >= _probe_parameters['visual_refresh_seconds']:
    observe()
    hand.last_probe_visual_s = float(data.time)
""").body
    recording = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "total" for target in node.targets)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "record")
    loop.body[recording + 1:recording + 1] = ast.parse("""
probe_failure = hand.after_step(state, data, model.opt.timestep)
trace[-1]['probe_policy'] = hand.latest_probe.copy()
if probe_failure is not None:
    hand.abort_probe(probe_failure)
    raise MotionFailure(probe_failure)
""").body
    loop.body.extend(ast.parse("""
if state == State.CLOSE and hand.probe_eligible_seconds >= _probe_parameters['preprobe_seconds']:
    trace[-1]['close_termination'] = 'DECLARED_CAPACITY_1G_PROBE_ELIGIBLE_NOT_EQUILIBRIUM'
    break
""").body)
    execute = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    attempt = next(node for node in execute.body if isinstance(node, ast.Try))
    close = next(i for i, node in enumerate(attempt.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "move"
        and ast.unparse(node.value.args[0]) == "State.CLOSE")
    attempt.body.insert(close + 1, ast.parse("observe()").body[0])
    precondition = next(node for node in attempt.body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and "UNBALANCED_GRIP_BEFORE_LIFT" in ast.unparse(child) for child in node.body))
    precondition.test = ast.parse("not hand.admit_probe(data)", mode="eval").body
    precondition.body = ast.parse("raise MotionFailure('PROBE_NOT_ELIGIBLE')").body
    probe_gate = next(node for node in ast.walk(attempt) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED" in ast.unparse(child) for child in node.body))
    probe_gate.test = ast.parse("not hand.confirm_probe(data)", mode="eval").body
    attempt.handlers[0].body[0:0] = ast.parse("""
if hand.probe_active or str(error).startswith(('PROBE_', 'HELD_POLICY_')):
    hand.abort_probe(str(error))
""").body
    revalidation = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "held_revalidated" for target in node.targets))
    revalidation.value = ast.parse("hand.held_revalidated(data)", mode="eval").body
    namespace = dict(original.__globals__)
    namespace.update(HandGraspController=make_hand, _estimated=estimated,
        _phase_seconds=PHASE_SECONDS, _probe_parameters=PROBE_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_load_transfer_probe_r8>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r6.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "tests/test_public_ycb_probe_policy_r8.py"):
        sources[path.relative_to(PROJECT).as_posix()] = r4.previous.sha256(path)
    return sources


def _runner():
    tree = ast.parse(textwrap.dedent(inspect.getsource(r4.run)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = {"public_ycb_tactile_wrench_force_band_development_r4": "public_ycb_load_transfer_probe_development_r8",
                "tactile_wrench_band_r4": "load_transfer_probe_r8",
                "public_ycb_tactile_wrench_force_band_r4": "public_ycb_load_transfer_probe_r8"}.get(node.value, node.value)
    protocol_write = next(i for i, node in enumerate(tree.body[0].body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "previous.atomic_json")
    tree.body[0].body[protocol_write:protocol_write] = ast.parse("""
protocol['latched_pressure_escape'] = dict(_relief_parameters)
protocol['load_transfer_probe'] = dict(_probe_parameters)
protocol['force_control'] = 'Frozen R6 hand pressure regulator and latched safety, original8s CLOSE, physical gains/contact/torque caps unchanged. Emergency actual-joint reference resets and original joint-limit relief blocks remain explicit.'
protocol['grip_gate'] = 'Declared experimental probe admission: continuous.25s fresh accepted RGB-D/public-mesh association, geometry1.5g feasible, measured upper-capacity1g feasible, stable all-channel<=6N/palm<=.05N and thumb plus two other active associated fingers>=.35N, no emergency. Current-patch LP repeated at admission. Capacity is not actual equilibrium or held evidence.'
protocol['probe_confirmation'] = 'Actual5mm upward motion with3s cubic command, actual wrist speed monitored<=10mm/s and downward drift<=.5mm; every native step can abort. Actual endpoint plus target-specific virtual table load<=20% catalogueweight, maintained associated contacts and fresh fixed-normal1g feasibility for.25s, at least one accepted actual RGB-D after probe start, repeated current-patch LP at decisive confirmation. This confirms only the small probe; original strict25mm/3s/full-task12N outcomes unchanged.'
protocol['visual_updates'] = 'Actual RGB-D during probe and held motion every.25simulated-second; accepted pose must be<=1s old. Unconfirmed probe uses latest visual world pose without held-rigid wrist propagation. Held propagation remains a labelled hypothesis between actual observations.'
protocol['wrench_evidence'] = 'Original exact fixed-normal1.5g diagnostics retained. Separate upper-capacity1g and fixed-normal1g feasibility; unknown tangential forces are LP variables, so neither is measured six-axis actual wrench equilibrium.25Hz compatibility is labelled sampled; decisive admission/confirmation solve current patches again.'
protocol['probe_hypothesis'] = 'Admission deliberately requires only1g measured capacity, even if measured capacity1.5g remains false, while retaining1.5g geometry-allocation feasibility. This is a separate lower-margin bounded load-transfer experiment, not relabelled baseline success.'
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
    for key, field, filename in (("pressure_relief", "relief_events", "pressure_relief.json.gz"),
            ("load_transfer_probe", "probe_events", "load_transfer_probe.json.gz")):
        artifacts.value.keys.append(ast.Constant(value=key))
        artifacts.value.values.append(ast.parse(f"previous.write_gzip(directory / '{filename}', [event for hand in hands for event in hand.{field}])", mode="eval").body)
    namespace = dict(r4.run.__globals__)
    namespace.update(controller_for=controller_for, snapshot=snapshot, PHASE_SECONDS=PHASE_SECONDS,
        TactilePressureGuard=r6.LatchedPressureGuard, _relief_parameters=r6.RELIEF_PARAMETERS,
        _probe_parameters=PROBE_PARAMETERS)
    exec(compile(ast.fix_missing_locations(tree), "<public_ycb_load_transfer_probe_runner_r8>", "exec"), namespace)
    return namespace["run"]


run = _runner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind != "ycb_can" or seed < 0 for kind, seed in cases):
        parser.error("R8 requires unique nonnegative ycb_can seeds")
    if r4.previous.sha256(args.profile_json) != EXPECTED_PROFILE_SHA256:
        parser.error("R8 requires the frozen first sidewall profile used by the actual R6b safety experiment")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
