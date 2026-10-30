#!/usr/bin/env python3
"""Isolated tactile-wrench grasp development with guarded physical recovery."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import zipfile

import mujoco
import numpy as np
import scipy
from PIL import Image

import develop_public_ycb_force_band as previous
import develop_public_ycb_force_band_r3 as r3
from estimated_tactile_patches import PublicMeshSurface, associate_tactile_contacts
from tactile_wrench_allocator import allocate_contact_wrench
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.control import FINGERS
from kuka_sharpa_grasp.physics import get_physics_profile
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


PROJECT = previous.PROJECT
PHASE_SECONDS = {**r3.PHASE_SECONDS, "CLOSE": 8., "TRIAL_LIFT": 3.}


class TactilePressureGuard(r3.NormalPressureGuard):
    """Expose hand tactile patches without external geometry identities."""

    def __init__(self, model, friction_coefficient):
        super().__init__(model)
        self.friction_coefficient = friction_coefficient
        self.raw_contacts = []
        self.last_data = None

    def read(self, data):
        readings = super().read(data)
        patches, wrench = [], np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.hand) == (b in self.hand):
                continue
            side = a if a in self.hand else b
            sign = 1. if side == b else -1.
            mujoco.mj_contactForce(self.model, data, index, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3)
            patches.append({"finger": self.hand[side],
                "position_world_m": contact.pos.tolist(),
                "normal_on_hand_world": (sign * frame[0]).tolist(),
                "force_on_hand_world_n": (sign * frame.T @ wrench[:3]).tolist(),
                "pressure_n": abs(float(wrench[0])),
                "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
                "friction_coefficient": self.friction_coefficient,
                "hand_body_name": self.model.body(int(self.model.geom_bodyid[side])).name})
        self.raw_contacts, self.last_data = patches, data
        return readings

    def guide(self, state, desired, command, kin, data, failure):
        readings = self.read(data)
        pressures = {finger: reading.normal for finger, reading in readings.items()}
        peak = max(pressures.values())
        differences = {finger: pressure - self.baseline[finger] for finger, pressure in pressures.items()}
        increase = max(differences.values())
        phase = state.value
        if phase in ("SEARCH", "APPROACH", "ALIGN", "RECOVER"):
            scale, retreat = float(np.clip((3.5 - peak) / 2., 0., 1.)), peak > 4.5
        elif phase in ("PLACE", "RELEASE", "VERIFY"):
            risk = max(increase, peak - 6.)
            scale, retreat = float(np.clip((2. - risk) / 1.5, 0., 1.)), peak > 8. or increase > 3.5
        else:
            scale, retreat = float(np.clip((8. - peak) / 2., 0., 1.)), peak > 8.
        mode, direction, reason = "pass", None, None
        if retreat:
            finger = max(pressures, key=pressures.get)
            direction = self.world_normals[finger].copy()
            magnitude = float(np.linalg.norm(direction))
            if magnitude < .05:
                direction, reason = np.array([0., 0., 1.]), "missing_normal_upward_fallback"
            else:
                direction, reason = direction / magnitude, "external_hand_normal"
            if self.retreat_offset is None:
                self.retreat_offset = kin.position(command) - kin.origin + direction * .006
                self.retreat_direction = direction.copy()
                self.retreat_reason = reason
                self.blocked_at = float(data.time)
            desired = kin.solve(self.retreat_offset, seed=command, tolerance=.0001)
            if desired is None:
                raise failure("CONTACT_NORMAL_RETREAT_IK")
            mode = "external_normal_retreat_6mm"
            direction, reason = self.retreat_direction, self.retreat_reason
        elif self.retreat_offset is not None and peak > 2.:
            desired = kin.solve(self.retreat_offset, seed=command, tolerance=.0001)
            if desired is None:
                raise failure("CONTACT_NORMAL_RETREAT_IK")
            direction, reason, mode = self.retreat_direction, self.retreat_reason, "finish_contact_retreat"
        elif scale < 1.:
            desired = command + scale * (desired - command)
            mode = "pressure_crawl" if scale else "pressure_brake"
            if scale == 0. and self.blocked_at is None:
                self.blocked_at = float(data.time)
            if scale > .2:
                self.blocked_at = None
        else:
            self.blocked_at, self.retreat_offset = None, None
        self.latest = {"observation_timestamp_s": float(data.time), "state": phase,
            "channel_pressures_n": pressures, "tared_increase_n": increase, "scale": scale,
            "mode": mode, "retreat_reason": reason,
            "retreat_direction_world": None if direction is None else direction.tolist(),
            "world_external_hand_forces_n": {finger: value.tolist() for finger, value in self.world_forces.items()},
            "world_external_hand_pressure_normals_n": {finger: value.tolist() for finger, value in self.world_normals.items()},
            "target_pose_used": False, "contact_normal_scope": "Native external hand tactile contacts without external IDs"}
        self.events.append(self.latest)
        if self.blocked_at is not None and data.time - self.blocked_at > 1.5:
            raise failure("CONTACT_GUARD_NOT_RELIEVED")
        return desired


class EstimatedContactState:
    def __init__(self, kind, mass_kg):
        self.surface = PublicMeshSurface.from_public_kind(kind, mass_kg)
        self.mass_kg = mass_kg
        self.pose_fit = None
        self.pose_timestamp_s = None
        self.anchor_wrist_position = None
        self.anchor_wrist_rotation = None
        self.kin = None
        self.payload_confirmed = False
        self.last_close_observation_s = -np.inf
        self.events = []

    def bind(self, kin):
        self.kin = kin

    def accept_visual(self, fitted, timestamp_s, wrist_position, wrist_rotation, phase):
        if fitted.get("accepted"):
            self.pose_fit = dict(fitted)
            self.pose_timestamp_s = float(timestamp_s)
            self.anchor_wrist_position = np.asarray(wrist_position).copy()
            self.anchor_wrist_rotation = np.asarray(wrist_rotation).copy()
        if phase == "CLOSE":
            self.last_close_observation_s = float(timestamp_s)
        self.events.append({"timestamp_s": float(timestamp_s), "phase": phase,
            "accepted": bool(fitted.get("accepted")), "reason": fitted.get("reason"),
            "last_accepted_visual_timestamp_s": self.pose_timestamp_s})

    def estimated_pose(self, data, carrying):
        if self.pose_fit is None:
            return {"accepted": False, "reason": "NO_ACCEPTED_VISUAL_POSE"}
        fitted = dict(self.pose_fit)
        if carrying:
            wrist_rotation = data.xmat[self.kin.body].reshape(3, 3)
            delta_rotation = wrist_rotation @ self.anchor_wrist_rotation.T
            fitted["origin_m"] = (data.xpos[self.kin.body] + delta_rotation @
                (np.asarray(self.pose_fit["origin_m"]) - self.anchor_wrist_position)).tolist()
            fitted["rotation_world_from_object"] = (delta_rotation @
                np.asarray(self.pose_fit["rotation_world_from_object"])).tolist()
            fitted["proprio_propagated_scope"] = "Last accepted visual pose propagated under a rigid held-object hypothesis; not a new visual measurement"
        return fitted


class WrenchAllocatedHand(previous.ForceBandHand):
    def __init__(self, model, config, profile, reader, estimated):
        super().__init__(model, config, profile, reader)
        self.estimated = estimated
        self.allocation = {"feasible": False, "measured_feasible": False, "reason": "NOT_EVALUATED"}
        self.association = {"accepted": False, "reason": "NOT_EVALUATED"}
        self.last_allocation_s = -np.inf
        self.solved_contact_sample = []
        self.relief_offsets = {}
        self.joint_dofs = {}
        for name, aid in self.actuators.items():
            if name.startswith("left_"):
                joint = int(model.actuator_trnid[aid, 0])
                self.joint_dofs[name] = int(model.jnt_dofadr[joint])

    def angles(self):
        return {name: value + self.relief_offsets.get(name, 0.) for name, value in super().angles().items()}

    def joint_pressure_relief(self, readings, dt):
        data = self.reader.last_data
        if data is None:
            return
        torques = np.zeros(self.model.nv)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for patch in self.reader.raw_contacts:
            if readings[patch["finger"]].normal <= 6.:
                continue
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, patch["hand_body_name"])
            mujoco.mj_jac(self.model, data, jacp, jacr, np.asarray(patch["position_world_m"]), body)
            torques += jacp.T @ np.asarray(patch["force_on_hand_world_n"])
        for name, dof in self.joint_dofs.items():
            if abs(torques[dof]) > .005:
                previous_offset = self.relief_offsets.get(name, 0.)
                velocity = float(np.clip(2. * torques[dof], -.2, .2))
                self.relief_offsets[name] = float(np.clip(previous_offset + velocity * dt, -.15, .15))

    @staticmethod
    def contact_sample(contacts):
        return sorted(({key: contact[key] for key in ("finger", "hand_body_name",
            "position_world_m", "normal_on_hand_world", "pressure_n")}
            for contact in contacts if contact["pressure_n"] >= .05),
            key=lambda row: (row["finger"], row["hand_body_name"], *row["position_world_m"]))

    def measurements_match_solve(self):
        current = self.contact_sample(self.reader.raw_contacts)
        if len(current) != len(self.solved_contact_sample):
            return False
        for now, solved in zip(current, self.solved_contact_sample, strict=True):
            if now["finger"] != solved["finger"] or now["hand_body_name"] != solved["hand_body_name"]:
                return False
            if np.linalg.norm(np.asarray(now["position_world_m"]) - solved["position_world_m"]) > .001:
                return False
            if np.asarray(now["normal_on_hand_world"]) @ solved["normal_on_hand_world"] < .99:
                return False
            if abs(now["pressure_n"] - solved["pressure_n"]) > .1 + .2 * solved["pressure_n"]:
                return False
        return True

    def regulate(self, readings, dt, carrying):
        self.elapsed += dt
        data = self.reader.last_data
        pressures = {finger: float(readings[finger].normal) for finger in readings}
        if data.time - self.last_allocation_s >= .04 - 1e-9:
            pose = self.estimated.estimated_pose(data, carrying)
            self.association = associate_tactile_contacts(self.reader.raw_contacts, pose,
                self.estimated.surface, now_s=float(data.time),
                pose_timestamp_s=self.estimated.pose_timestamp_s,
                max_pose_age_s=15., max_surface_distance_m=.006)
            if self.association["accepted"]:
                self.allocation = allocate_contact_wrench(self.association["patches"],
                    self.association["estimated_com_world_m"], self.estimated.mass_kg,
                    measured_pressures_n=pressures, pressure_limit_n=6., gravity_margin=1.5)
            else:
                self.allocation = {"feasible": False, "measured_feasible": False,
                    "reason": "CONTACT_ASSOCIATION_" + self.association["reason"]}
            self.last_allocation_s = float(data.time)
            self.solved_contact_sample = self.contact_sample(self.reader.raw_contacts)
        goals = self.allocation.get("target_pressures_n", {}) if self.allocation.get("feasible") else {}
        commanded_goals = {}
        nominal = self.profile.get("contact_closure", {})
        for finger in FINGERS:
            pressure, previous_closure = pressures[finger], self.closure[finger]
            if pressure > .2:
                self.engaged.add(finger)
            goal = min(6., max(.35, float(goals.get(finger, 2.))))
            if nominal.get(finger) == 0.:
                goal = 0.
            commanded_goals[finger] = goal
            derivative = (pressure - self.previous_pressure[finger]) / dt
            self.pressure_rate[finger] += dt / (.04 + dt) * (derivative - self.pressure_rate[finger])
            self.previous_pressure[finger] = pressure
            if pressure < .15:
                reference = min(1., nominal.get(finger, .7) + .08)
                velocity = float(np.clip(1.5 * (reference - previous_closure), 0., self.config.close_rate))
                if nominal.get(finger) == 0.:
                    velocity = -self.config.relief_rate
            else:
                velocity = float(np.clip(.03 * (goal - pressure)
                    - .001 * max(0., self.pressure_rate[finger]), -.35, self.config.close_rate))
            if pressure > 7.:
                velocity = -self.config.relief_rate
            self.closure[finger] = float(np.clip(previous_closure + velocity * dt, 0., 1.))
            self.slip_adjustments += int(carrying and self.closure[finger] > previous_closure)
        self.joint_pressure_relief(readings, dt)
        ready = bool(self.association.get("accepted") and self.allocation.get("feasible")
            and self.allocation.get("measured_feasible") and self.measurements_match_solve()
            and max(pressures.values()) <= 7. and pressures.get("palm", 0.) <= .05)
        self.ready_seconds = self.ready_seconds + dt if ready else 0.
        self.feedback_events.append({"timestamp_s": float(data.time), "elapsed_control_s": self.elapsed,
            "carrying": carrying, "pressures_n": pressures,
            "force_targets_n": commanded_goals,
            "closure": self.closure.copy(), "joint_pressure_relief_offsets_rad": self.relief_offsets.copy(),
            "ready_seconds": self.ready_seconds, "wrench_feasible": bool(self.allocation.get("feasible")),
            "measured_wrench_feasible": bool(self.allocation.get("measured_feasible")),
            "allocation": self.allocation, "association": self.association,
            "allocation_age_s": float(data.time - self.last_allocation_s),
            "current_measurements_match_solve": self.measurements_match_solve(),
            "target_pose_used": False, "estimated_pose_source": "Actual wrist RGB-D/public mesh, with declared held-rigid proprioception propagation"})
        self.feedback_applied = True

    def release_pressure_relief(self, readings, dt):
        self.joint_pressure_relief(readings, dt)


RECOVERY = '''
try:
    touching = hand_reader.total(hand_reader.read(data)).contacts > 0
    held_revalidated = (_estimated.payload_confirmed and touching and hand.ready_seconds >= .05
        and hand.allocation.get('measured_feasible', False) and hand.measurements_match_solve()
        and data.time - hand.last_allocation_s <= .1
        and hand.table_support(data) < .2 * _catalogue_mass_kg * 9.81)
    if held_revalidated and targets is not None:
        move(State.PLACE, targets[1], 3., .85)
        handoff(command)
    _estimated.payload_confirmed = False
    move(State.RELEASE, command.copy(), 3., 0.)
    readings = hand_reader.read(data)
    if max(value.normal for value in readings.values()) > 1.:
        finger = max(readings, key=lambda channel: readings[channel].normal)
        direction = _contact_guard.world_normals[finger].copy()
        norm = float(np.linalg.norm(direction))
        direction = np.array([0., 0., 1.]) if norm < .05 else direction / norm
        normal_clearance = kin.solve(kin.position(command) - kin.origin + direction * .012, seed=command)
        move(State.RECOVER, normal_clearance, 2., 0.)
    retreat_position = kin.position(command) - kin.approach_axis * .13 + [0., 0., .10]
    safe = kin.solve(retreat_position - kin.origin, seed=command)
    if safe is None:
        raise MotionFailure('RECOVERY_IK')
    move(State.RECOVER, safe, 2., 0.)
    safe_height = .34 if physical_v6 and scenario.object_kind == 'ycb_cracker_box' else .18
    move(State.RECOVER, kin.solve(np.array([0., 0., safe_height])), 2., 0.)
except MotionFailure as error:
    controller.fail(data.time, 'RECOVERY_' + str(error))
    controller.enter(State.ABORT, data.time, controller.failure)
    break
'''


def controller_for(kind, directory, observations, guard, profile, mass_kg):
    original, source, _ = r3.controller_for(kind, directory, observations, guard, profile)
    tree = ast.parse(source)
    task = tree.body[0]
    namespace = dict(original.__globals__)
    estimated, hands = EstimatedContactState(kind, mass_kg), []
    nominal_reset = next(i for i, node in enumerate(task.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_nominal_wrist_rotation" for target in node.targets))
    task.body.insert(nominal_reset + 1, ast.parse("_estimated.bind(kin)").body[0])
    observe = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "observe")
    returned = next(i for i, node in enumerate(observe.body) if isinstance(node, ast.Return))
    observe.body.insert(returned, ast.parse("""
_estimated.accept_visual(_latest_pose.get(controller.state, {}), float(data.time),
    data.xpos[kin.body].copy(), data.xmat[kin.body].reshape(3, 3).copy(), controller.state)
""").body[0])
    move = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "move")
    loop = next(node for node in move.body if isinstance(node, ast.For))
    writing = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "write" and ast.unparse(node.value.func.value) == "hand")
    loop.body.insert(writing, ast.parse("""
if state in (State.RELEASE, State.VERIFY, State.RECOVER):
    hand.release_pressure_relief(hand_reader.read(data), model.opt.timestep)
""").body[0])
    hand_motion = next(node for node in loop.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare) and ast.unparse(node.test) == "state == State.CLOSE")
    hand_motion.body.insert(0, ast.parse("""
if data.time - _estimated.last_close_observation_s >= .5:
    observe()
""").body[0])
    loop.body.append(ast.parse("""
if state == State.CLOSE and hand.ready_seconds >= .25:
    trace[-1]['close_termination'] = 'CONTINUOUS_MEASURED_WRENCH_FEASIBILITY'
    break
""").body[0])
    # Retraction remains subject to the same geometry and workspace monitors.
    for node in ast.walk(move):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
            if ast.unparse(node.test.values[0]) == "state != State.RECOVER":
                node.test = node.test.values[1] if len(node.test.values) == 2 else ast.BoolOp(op=ast.And(), values=node.test.values[1:])
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state in (State.RELEASE, State.VERIFY, State.RECOVER)":
            if any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Attribute)
                and target.attr == "opposition" for target in child.targets) for child in node.body):
                node.body = [ast.Pass()]
    execute = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    execute.body.insert(1, ast.parse("_estimated.payload_confirmed = False").body[0])
    for node in ast.walk(execute):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "probe" for target in node.targets):
            node.value = ast.parse("kin.solve(data.xpos[kin.body].copy() - kin.origin + [0., 0., .005], seed=command, tolerance=.0001)", mode="eval").body
        if isinstance(node, ast.If) and "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED" in ast.unparse(node):
            if isinstance(node.test, ast.BoolOp) and any(isinstance(child, ast.Raise) for child in node.body):
                node.test = ast.parse("hand.ready_seconds < .25 or hand.table_support(data) > .2 * _catalogue_mass_kg * 9.81", mode="eval").body
    attempt = next(node for node in execute.body if isinstance(node, ast.Try))
    accepted = next(node for node in attempt.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "total.contacts == 0").orelse
    probe_gate = next(i for i, node in enumerate(accepted) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and "PROBE_LOAD_OR_GRIP_NOT_CONFIRMED" in ast.unparse(child) for child in node.body))
    accepted.insert(probe_gate + 1, ast.parse("_estimated.payload_confirmed = True").body[0])
    attempt.handlers[0].body.insert(0, ast.parse("""
if str(error) in ('SUSTAINED_CONTACT_LOSS', 'EARLY_PLACE_CONTACT_LOSS', 'PROBE_LOAD_OR_GRIP_NOT_CONFIRMED'):
    _estimated.payload_confirmed = False
""").body[0])
    attempts = next(node for node in task.body if isinstance(node, ast.For) and ast.unparse(node.target) == "attempt")
    recovery = next(node for node in attempts.body if isinstance(node, ast.Try)
        and any(isinstance(child, ast.If) and "targets is not None" in ast.unparse(child.test) for child in node.body))
    attempts.body[attempts.body.index(recovery)] = ast.parse(RECOVERY).body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1:
            if ast.unparse(node.args[0]) == "model.body_mass[target_body]":
                node.args[0] = ast.Name(id="_catalogue_mass_kg", ctx=ast.Load())

    def make_hand(model, config):
        hand = WrenchAllocatedHand(model, config, profile, guard, estimated)
        hands.append(hand)
        return hand

    namespace.update(_phase_seconds=PHASE_SECONDS, _estimated=estimated,
        _catalogue_mass_kg=mass_kg, HandGraspController=make_hand)
    tree = ast.fix_missing_locations(tree)
    exec(compile(tree, "<public_ycb_wrench_force_band_r4>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands, estimated


def snapshot(profile_path):
    sources = r3.snapshot(profile_path)
    for path in (Path(__file__).resolve(), PROJECT / "scripts" / "tactile_wrench_allocator.py",
                 PROJECT / "scripts" / "estimated_tactile_patches.py"):
        sources[path.relative_to(PROJECT).as_posix()] = previous.sha256(path)
    return sources


def run(output, profile_path, cases):
    output.mkdir(parents=True, exist_ok=False)
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    before, meshes = snapshot(profile_path), previous.public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    protocol = {"format": "public_ycb_tactile_wrench_force_band_development_r4",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes_before": before,
        "public_meshes": meshes, "physics_profile": "physical-v6", "maximum_attempts": 2,
        "phase_seconds": PHASE_SECONDS, "nominal_profiles": profiles,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
            "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "nominal_profile_file": {"path": str(profile_path), "bytes": profile_path.stat().st_size,
            "sha256": previous.sha256(profile_path)},
        "schedule": [{"kind": kind, "seed": seed, "profile": "tactile_wrench_band_r4"} for kind, seed in cases],
        "grip_gate": "Actual hand tactile patches associated with accepted RGB-D/public-mesh estimate;25Hz static wrench allocation with1.5gravity margin and6N/channel targets, native500Hz pressure safety; sampled fixed measured patch normal forces feasible for.25s, resetting on contact topology change,>1mm point motion, normalcosine<.99, or pressure change>.1N+20%; then actual5mm probe and declared target-specific virtual instrumented-table unloading",
        "force_control": "Original physics/gains/torque caps; targets<=6N;7N finger relief; above6N native external contact Jacobian gives bounded.15rad per-joint pressure-relief offsets, including proximal thumb joints",
        "palm_policy": "Palm remains in every native overload channel and patch-association contamination check; no palm force is allocated, so readiness additionally requires palm<=.05N and exact measured-normal equilibrium may reject smaller nonzero palm loads",
        "visual_updates": "Actual CLOSE RGB-D at.5simulated-second intervals; reject ambiguous fits; last accepted visual estimate may be up to15s old and held-rigid proprioception propagation is explicit",
        "recovery": "Unconfirmed payload stays at current wrist while flexion opens; CMC opposition remains fixed; tactile normal and upward physical retreats retain all safety monitors. Held recovery requires prior5mm-probe unloading plus fresh<=.1s measured wrench feasibility, stable current contact sample, and current table unloading; contact loss clears payload belief",
        "observation_contract": "No external geom identities or target simulator pose/COM in tactile patch association and wrench allocation; nominal public catalogue mass/friction/mesh COM; separate explicitly declared target-specific virtual table load cell identifies target/table contacts, never supplies target pose/COM",
        "scoring": "Original full task result retained; strict25mm/3s/full-task12N independent verifier required",
        "claim_scope": "Reused development poses; no held-out accuracy claim"}
    previous.atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in protocol["schedule"]:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        observations, trace, audit, guard, hands = [], [], None, None, []
        phase_frames, overview = [], None
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None,
            "held_lift": None, "runner_error": None}
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
            catalogue = get_physics_profile("physical-v6").object_parameters(case["kind"])
            friction = float(np.sqrt(catalogue.friction[0]))
            data, guard = mujoco.MjData(model), TactilePressureGuard(model, friction)
            spec, config, profile = previous.selected_profile(case["kind"], profiles[case["kind"]])
            config = replace(config, maximum_force_n=7.)
            controller, readable, hands, estimated = controller_for(case["kind"], directory,
                observations, guard, profile, catalogue.mass_kg)
            (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
            row.update(grasp_specification=asdict(spec), hand_config=asdict(config), grip_profile=profile,
                catalogue_mass_prior_kg=catalogue.mass_kg,
                catalogue_pair_friction_prior=friction,
                derived_controller_sha256=previous.sha256(directory / "derived_controller.py"))
            overview = mujoco.Renderer(model, height=720, width=960)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.55, -.02, .52]
            camera.distance, camera.azimuth, camera.elevation = 1.45, 135., -25.
            frame_directory = directory / "phase_frames"
            frame_directory.mkdir()

            def capture_phase(label, frame_data):
                if len(phase_frames) >= 64:
                    return
                overview.update_scene(frame_data, camera=camera)
                pixels = overview.render().copy()
                destination = frame_directory / f"{len(phase_frames):02d}_{label}.png"
                Image.fromarray(pixels).save(destination)
                phase_frames.append({"phase": label, "timestamp_s": float(frame_data.time),
                    "path": destination.relative_to(directory).as_posix(),
                    "sha256": previous.sha256(destination), "shape": list(pixels.shape),
                    "pixel_std": float(pixels.std()), "runtime_render": True})

            with previous.RichNativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    frame_callback=capture_phase, step_callback=audit.sample,
                    max_attempts=2, support_load_sensor=True,
                    support_required=True, grasp_spec=spec, hand_config_override=config,
                    camera_config=CameraConfig(seed=case["seed"]), placement_retention=True, visual_transfer=True)
            assert not audit.external_qpos_writes and len(audit.samples) == audit.integration_steps
            result.update(registration_observations=observations, estimated_pose_events=estimated.events,
                runtime_phase_frames=phase_frames,
                controller_revision="public_ycb_tactile_wrench_force_band_r4")
            previous.atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], duration_s=result["duration"],
                native_audit=audit.as_dict(), maximum_hand_total_force_n=result["maximum_hand_total_force"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        finally:
            if overview is not None:
                overview.close()
        row["runtime_phase_frames"] = phase_frames
        row["artifacts"] = {"trace": previous.write_gzip(directory / "trace.json.gz", trace),
            "native": previous.write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples),
            "guard": previous.write_gzip(directory / "guard.json.gz", [] if guard is None else guard.events),
            "grip_feedback": previous.write_gzip(directory / "grip_feedback.json.gz",
                [event for hand in hands for event in hand.feedback_events])}
        previous.atomic_json(directory / "metadata.json", row)
        rows.append(row)
        previous.atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"],
            "failure", row.get("failure"), "error", row["runner_error"], flush=True)
    after = snapshot(profile_path)
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
        public_meshes_unchanged=meshes == previous.public_mesh_inventory(),
        ended_at_utc=datetime.now(timezone.utc).isoformat())
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    previous.atomic_json(output / "protocol.json", protocol)
    previous.atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", help="KIND:SEED, all three development identities by default")
    args = parser.parse_args()
    cases = previous.CASES if args.cases is None else tuple((value.rsplit(":", 1)[0],
        int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind not in previous.KINDS or seed < 0 for kind, seed in cases):
        parser.error("Unique public kinds and nonnegative seeds required")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
