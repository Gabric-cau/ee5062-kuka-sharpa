#!/usr/bin/env python3
"""Isolated public-mesh grip development with measured pressure and physical motion."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from develop_public_ycb_contact_guard import InsertionGuard, envelope, guarded_controller, snapshot as guard_snapshot
from develop_public_ycb_receiving import hand_config
from evaluate_public_ycb import KINDS, NativeAudit, public_mesh_inventory
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from register_public_ycb import RegisteredSpecification
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.control import ContactForceReader, FINGERS
from kuka_sharpa_grasp.hand_grasp import HandGraspController
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


CASES = (("ycb_can", 3414), ("ycb_potted_meat_can", 3402), ("ycb_cracker_box", 3406))
PHASE_SECONDS = {"CLOSE": 8., "TRIAL_LIFT": 4., "TRANSFER": 6., "PLACE": 5.,
                 "RELEASE": 4., "VERIFY": 4., "RECOVER": 4.}


class RichNativeAudit(NativeAudit):
    """Add actual hand state and contact bases without altering integration."""

    def __init__(self, model, data):
        super().__init__(model, data)
        self.hand_joints = [j for j in range(model.njnt)
            if (model.joint(j).name or "").startswith("left_") and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

    def sample(self, state, data):
        previous = len(self.samples)
        super().sample(state, data)
        if len(self.samples) == previous:
            return
        sample = self.samples[-1]
        sample["actual_hand_joints"] = {self.model.joint(j).name: {
            "position_rad": float(data.qpos[self.model.jnt_qposadr[j]]),
            "velocity_rad_s": float(data.qvel[self.model.jnt_dofadr[j]])} for j in self.hand_joints}
        for row, contact in zip(sample["native_contacts"], data.contact[:data.ncon], strict=True):
            row["contact_frame_world_rows"] = contact.frame.reshape(3, 3).tolist()


class DirectionalPressureGuard(InsertionGuard):
    total = staticmethod(ContactForceReader.total)

    def __init__(self, model):
        super().__init__(model)
        self.world_forces = {}
        self.world_normals = {}
        self.baseline_vectors = {}

    def read(self, data):
        readings = super().read(data)
        vectors = {finger: np.zeros(3) for finger in readings}
        normals = {finger: np.zeros(3) for finger in readings}
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.hand) == (b in self.hand):
                continue
            side = a if a in self.hand else b
            finger = self.hand[side]
            mujoco.mj_contactForce(self.model, data, index, wrench)
            basis = np.asarray(contact.frame).reshape(3, 3)
            sign = 1. if b == side else -1.
            vectors[finger] += sign * basis.T @ wrench[:3]
            normals[finger] += sign * basis[0] * abs(float(wrench[0]))
        self.world_forces, self.world_normals = vectors, normals
        return readings

    def begin(self, state, data):
        super().begin(state, data)
        self.baseline_vectors = {finger: value.copy() for finger, value in self.world_forces.items()}

    def guide(self, state, desired, command, kin, data, failure):
        readings = self.read(data)
        pressures = {finger: reading.normal for finger, reading in readings.items()}
        peak = max(pressures.values())
        differences = {finger: pressures[finger] - self.baseline[finger] for finger in pressures}
        increase = max(differences.values())
        mode, scale = "pass", 1.
        if state.value in ("SEARCH", "APPROACH", "ALIGN", "RECOVER"):
            scale = float(np.clip((4. - peak) / 3., 0., 1.))
            retreat = peak > 6.
        elif state.value in ("PLACE", "RELEASE", "VERIFY"):
            risk = max(increase, peak - 7.5)
            scale = float(np.clip((3. - risk) / 2., 0., 1.))
            retreat = peak > 10. or increase > 5.
        else:
            scale = float(np.clip((10. - peak) / 3., 0., 1.))
            retreat = False
        if retreat:
            finger = max(pressures, key=lambda item: differences[item] if state.value == "PLACE" else pressures[item])
            direction = self.world_forces[finger].copy()
            if state.value == "PLACE":
                direction -= self.baseline_vectors[finger]
            magnitude = float(np.linalg.norm(direction))
            if magnitude < .05:
                direction = np.array([0., 0., 1.])
            else:
                direction /= magnitude
            if self.retreat_offset is None:
                self.retreat_offset = kin.position(command) - kin.origin + direction * .006
                self.blocked_at = float(data.time)
            safe = kin.solve(self.retreat_offset, seed=command, tolerance=.0001)
            if safe is None:
                raise failure("CONTACT_NORMAL_RETREAT_IK")
            desired, mode = safe, "contact_force_direction_retreat_6mm"
        elif scale < 1.:
            desired = command + scale * (desired - command)
            mode = "pressure_crawl" if scale else "pressure_brake"
            if scale == 0. and self.blocked_at is None:
                self.blocked_at = float(data.time)
            if scale > .2:
                self.blocked_at = None
        else:
            self.blocked_at = None
            self.retreat_offset = None
        self.latest = {"observation_timestamp_s": float(data.time), "state": state.value,
            "channel_pressures_n": pressures, "tared_increase_n": increase, "scale": scale,
            "mode": mode, "world_external_hand_forces_n": {finger: value.tolist()
                for finger, value in self.world_forces.items()}, "target_pose_used": False,
            "contact_normal_scope": "MuJoCo native contact basis; simulated tactile geometry, not hardware normals"}
        self.events.append(self.latest)
        if self.blocked_at is not None and data.time - self.blocked_at > 1.5:
            raise failure("CONTACT_GUARD_NOT_RELIEVED")
        return desired

    def relieve(self, hand, data, dt):
        if not getattr(hand, "feedback_applied", False):
            super().relieve(hand, data, dt)


class ForceBandHand(HandGraspController):
    """Bidirectional position increments from native pressure, with fixed servo limits."""

    def __init__(self, model, config, profile, reader):
        super().__init__(model, config)
        self.profile, self.reader = profile, reader
        self.ready_seconds = 0.
        self.feedback_applied = False
        self.feedback_events = []
        self.elapsed = 0.
        self.previous_pressure = dict.fromkeys(FINGERS, 0.)
        self.pressure_rate = dict.fromkeys(FINGERS, 0.)

    def angles(self):
        angles = super().angles()
        for finger in FINGERS:
            if finger == "thumb":
                continue
            prefix, closure = "left_" + finger + "_", self.closure[finger]
            angles[prefix + "MCP_FE"] = self.profile.get("nonthumb_mcp_ratio", 1.) * closure
            angles[prefix + "PIP"] = self.profile.get("nonthumb_pip_ratio", 1.4) * closure
            angles[prefix + "DIP"] = self.profile.get("nonthumb_dip_ratio", .8) * closure
        return angles

    def regulate(self, readings, dt, carrying):
        self.elapsed += dt
        for finger in FINGERS:
            pressure = readings[finger].normal
            if pressure > .2:
                self.engaged.add(finger)
            goal = self.profile.get("force_targets_n", {}).get(finger, self.profile.get("force_target_n", 5.))
            derivative = (pressure - self.previous_pressure[finger]) / dt
            alpha = dt / (.04 + dt)
            self.pressure_rate[finger] += alpha * (derivative - self.pressure_rate[finger])
            self.previous_pressure[finger] = pressure
            if pressure < .15:
                contact_closure = self.profile.get("contact_closure", {}).get(finger)
                velocity = self.config.close_rate if contact_closure is None else min(
                    self.config.close_rate, max(.015, contact_closure / 3.))
            else:
                velocity = float(np.clip(.025 * (goal - pressure)
                    - .0008 * max(0., self.pressure_rate[finger]), -.35, self.config.close_rate))
            if pressure > self.config.maximum_force_n:
                velocity = -self.config.relief_rate
            previous = self.closure[finger]
            self.closure[finger] = float(np.clip(previous + velocity * dt, 0., 1.))
            self.slip_adjustments += int(carrying and self.closure[finger] > previous)
        thumb = self.reader.world_normals.get("thumb", np.zeros(3))
        opposing = sum((self.reader.world_normals.get(finger, np.zeros(3))
            for finger in FINGERS if finger != "thumb"), np.zeros(3))
        lengths = float(np.linalg.norm(thumb) * np.linalg.norm(opposing))
        cosine = None if lengths < 1e-6 else float(thumb @ opposing / lengths)
        contacts = [finger for finger in FINGERS if readings[finger].normal >= .8]
        ready = ("thumb" in contacts and len(contacts) >= 3 and cosine is not None and cosine < -.35)
        self.ready_seconds = self.ready_seconds + dt if ready else 0.
        self.feedback_events.append({"elapsed_control_s": self.elapsed, "carrying": carrying,
            "pressures_n": {finger: readings[finger].normal for finger in readings},
            "force_targets_n": {finger: self.profile.get("force_targets_n", {}).get(finger,
                self.profile.get("force_target_n", 5.)) for finger in FINGERS},
            "closure": self.closure.copy(), "thumb_opposing_force_normal_cosine": cosine,
            "ready_seconds": self.ready_seconds, "target_pose_used": False})
        self.feedback_applied = True

    def close(self, readings, dt):
        self.regulate(readings, dt, False)

    def retain(self, readings, dt):
        self.regulate(readings, dt, True)

    def write(self, data):
        super().write(data)
        self.feedback_applied = False


def selected_profile(kind, supplied):
    if kind == "ycb_can":
        spec = RegisteredSpecification(**asdict(GRASP_SPECIFICATIONS[kind]))
        config = hand_config(kind)
    else:
        spec, config = envelope(kind)
    profile = dict(supplied or {})
    if "aperture_offset" in profile:
        spec = replace(spec, aperture_offset=tuple(profile["aperture_offset"]))
    config = replace(config, close_rate=.12, maximum_force_n=9., relief_rate=.6,
        contact_threshold_n=profile.get("force_target_n", 5.),
        retention_minimum_force_n=profile.get("force_target_n", 5.),
        thumb_opposition_rad=profile.get("thumb_opposition_rad", config.thumb_opposition_rad),
        thumb_mcp_flexion_ratio=profile.get("thumb_mcp_flexion_ratio", config.thumb_mcp_flexion_ratio),
        thumb_ip_flexion_ratio=profile.get("thumb_ip_flexion_ratio", config.thumb_ip_flexion_ratio))
    return spec, config, profile


def controller_for(kind, directory, observations, guard, profile, *, band=True):
    original, source = guarded_controller(kind, directory, observations, guard)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    move.body.insert(1, ast.parse("duration = max(duration, _phase_seconds.get(state.value, duration))").body[0])
    if band:
        execute = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute")
        attempts = next(node for node in execute.body if isinstance(node, ast.Try))
        close = next(index for index, node in enumerate(attempts.body)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "move"
            and isinstance(node.value.args[0], ast.Attribute) and node.value.args[0].attr == "CLOSE")
        attempts.body.insert(close + 1, ast.parse(
            "if hand.ready_seconds < .25:\n    raise MotionFailure('UNBALANCED_GRIP_BEFORE_LIFT')").body[0])
    tree = ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    hands = []

    def make_hand(model, config):
        hand = ForceBandHand(model, config, profile, guard) if band else HandGraspController(model, config)
        hands.append(hand)
        return hand

    namespace.update(_phase_seconds=PHASE_SECONDS, HandGraspController=make_hand,
                     HandContactReader=lambda model: guard)
    exec(compile(tree, "<isolated_public_ycb_force_band>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands


def snapshot():
    result = guard_snapshot()
    path = Path(__file__).resolve()
    result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    profile = PROJECT / "config" / "public_ycb_static_wrench_r1.json"
    if profile.is_file():
        result[profile.relative_to(PROJECT).as_posix()] = sha256(profile)
    return result


def run(output, profiles, cases, nominal_profiles=None, profile_path=None):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    profile_record = None if profile_path is None else {
        "path": str(profile_path), "bytes": profile_path.stat().st_size, "sha256": sha256(profile_path)}
    protocol = {"format": "public_ycb_force_band_development_r2", "source_hashes_before": before,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "public_meshes": meshes,
        "physics_profile": "physical-v6", "maximum_attempts": 2, "phase_seconds": PHASE_SECONDS,
        "schedule": [{"kind": kind, "seed": seed, "profile": profile}
            for kind, seed in cases for profile in profiles], "nominal_profiles": nominal_profiles,
        "nominal_profile_file": profile_record,
        "force_band": "Default5N or frozen per-finger nominal LP load allocation; bidirectional proportional pressure increments with40ms rising-pressure damping, closure<=.12/s, relief .6/s above9N, original physical servo parameters unchanged",
        "precontact": "If supplied, nominal individual shallow-contact closures synchronize the free-space closure rates over3s; actual contact pressure replaces the nominal schedule at.15N",
        "grip_gate": "thumb plus at least two nonthumb channels >=.8N and opposing native pressure-normal cosine<-.35 continuously .25s before physical lift",
        "guard": "Target-independent whole-hand pressure; tared pressure increases brake Cartesian insertion and request6mm along external contact-force direction via original arm limiter",
        "observation": "Initial actual wrist RGB-D/public-STL tilt registration; no live target pose is supplied to localization or control",
        "logging": "Every2ms native contact frame, actual hand qpos/qvel, world object velocity and no-external-object-qpos audit",
        "scoring": "Original V6 task labels retained; <=12N full-native maximum channel evaluated additionally, not inferred from the9N control threshold",
        "claim_scope": "All three identities on previously failed development poses; no held-out rate claim"}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in protocol["schedule"]:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        observations, trace, audit, guard, hands = [], [], None, None, []
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None,
            "held_lift": None, "runner_error": None}
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model, data = mujoco.MjModel.from_xml_path(str(scene)), None
            data, guard = mujoco.MjData(model), DirectionalPressureGuard(model)
            supplied = None if case["profile"].startswith("envelope") else nominal_profiles[case["kind"]]
            spec, config, profile = selected_profile(case["kind"], supplied)
            controller, readable, hands = controller_for(case["kind"], directory, observations, guard, profile,
                band=case["profile"] != "envelope_slow_reference")
            (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
            row.update(grasp_specification=asdict(spec), hand_config=asdict(config), grip_profile=profile,
                derived_controller_sha256=sha256(directory / "derived_controller.py"))
            with RichNativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    step_callback=audit.sample, max_attempts=2, support_load_sensor=True, support_required=True,
                    grasp_spec=spec, hand_config_override=config, camera_config=CameraConfig(seed=case["seed"]),
                    placement_retention=True, visual_transfer=False)
            assert not audit.external_qpos_writes and len(audit.samples) == audit.integration_steps
            result["registration_observations"] = observations
            atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], duration_s=result["duration"],
                native_audit=audit.as_dict(), maximum_hand_total_force_n=result["maximum_hand_total_force"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        row["artifacts"] = {"trace": write_gzip(directory / "trace.json.gz", trace),
            "native": write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples),
            "guard": write_gzip(directory / "guard.json.gz", [] if guard is None else guard.events),
            "grip_feedback": write_gzip(directory / "grip_feedback.json.gz", [event
                for hand in hands for event in getattr(hand, "feedback_events", [])])}
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"], "failure", row.get("failure"),
            "error", row["runner_error"], flush=True)
    after = snapshot()
    profile_unchanged = profile_record is None or sha256(profile_path) == profile_record["sha256"]
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after or not profile_unchanged,
        nominal_profile_file_unchanged=profile_unchanged,
        public_meshes_unchanged=meshes == public_mesh_inventory(), ended_at_utc=datetime.now(timezone.utc).isoformat())
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", choices=("envelope_slow_reference", "envelope_band", "nominal_lp_band"), default=("envelope_band",))
    parser.add_argument("--profile-json", type=Path, help="frozen nominal geometry-only profiles keyed by object kind")
    parser.add_argument("--cases", nargs="+", help="KIND:SEED; all three identities by default")
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    cases = CASES if args.cases is None else tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind not in KINDS or seed < 0 for kind, seed in cases):
        parser.error("unique known kinds and nonnegative seeds required")
    nominal = None if args.profile_json is None else json.loads(args.profile_json.read_text(encoding="utf-8-sig"))
    if "nominal_lp_band" in args.profiles and (nominal is None or any(kind not in nominal for kind, _ in cases)):
        parser.error("nominal_lp_band requires geometry profiles for every selected kind")
    summary = run(args.output.resolve(), tuple(args.profiles), cases, nominal,
        None if args.profile_json is None else args.profile_json.resolve())
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
