#!/usr/bin/env python3
"""Isolated pose-aware public-object contact development; historical r2 is frozen."""

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

import develop_public_ycb_force_band as previous
from kuka_sharpa_grasp.camera import CameraConfig, save_observation
from kuka_sharpa_grasp.perception import estimate_object
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene
from register_public_ycb import RegisteredBelief
from register_public_ycb_rigid import VisibleMeshPoseRegistration

PROJECT = previous.PROJECT
PHASE_SECONDS = {**previous.PHASE_SECONDS, "CLOSE": 10.}


class SynchronizedForceBandHand(previous.ForceBandHand):
    def regulate(self, readings, dt, carrying):
        before = self.closure.copy()
        adjustments = self.slip_adjustments
        super().regulate(readings, dt, carrying)
        nominal = self.profile.get("contact_closure", {})
        landing_seconds = max(3., max(nominal.values(), default=0.) / self.config.close_rate)
        for finger, contact_closure in nominal.items():
            if contact_closure == 0.:
                self.closure[finger] = 0.
            elif readings[finger].normal < .15:
                reference = contact_closure * min(1., self.elapsed / landing_seconds)
                velocity = np.clip(2. * (reference - before[finger]), -.2, self.config.close_rate)
                self.closure[finger] = float(np.clip(before[finger] + velocity * dt, 0., 1.))
        goals = self.feedback_events[-1]["force_targets_n"]
        self.slip_adjustments = adjustments + sum(int(carrying and self.closure[finger] > before[finger])
            for finger in previous.FINGERS)
        active = [finger for finger, goal in goals.items()
            if goal >= .8 and nominal.get(finger, 1.) > 0.]
        converged = all(readings[finger].normal >= .7 * goals[finger] for finger in active)
        opposed = self.feedback_events[-1]["thumb_opposing_force_normal_cosine"]
        ready = converged and opposed is not None and opposed < -.35
        if not ready:
            self.ready_seconds = 0.
        self.feedback_events[-1].update(closure=self.closure.copy(), ready_seconds=self.ready_seconds,
            force_goals_converged=converged, active_load_channels=active,
            free_contact_reference={finger: value * min(1., self.elapsed / landing_seconds)
                for finger, value in nominal.items()}, synchronized_landing_seconds=landing_seconds)


class NormalPressureGuard(previous.DirectionalPressureGuard):
    def guide(self, state, desired, command, kin, data, failure):
        readings = self.read(data)
        pressures = {finger: reading.normal for finger, reading in readings.items()}
        peak = max(pressures.values())
        differences = {finger: pressures[finger] - self.baseline[finger] for finger in pressures}
        increase = max(differences.values())
        mode, scale, reason, direction = "pass", 1., None, None
        if state.value in ("SEARCH", "APPROACH", "ALIGN", "RECOVER"):
            scale, retreat = float(np.clip((4. - peak) / 3., 0., 1.)), peak > 6.
        elif state.value in ("PLACE", "RELEASE", "VERIFY"):
            scale = float(np.clip((3. - max(increase, peak - 7.5)) / 2., 0., 1.))
            retreat = peak > 10. or increase > 5.
        else:
            scale, retreat = float(np.clip((10. - peak) / 3., 0., 1.)), False
        if retreat:
            absolute = peak > (6. if state.value in ("SEARCH", "APPROACH", "ALIGN", "RECOVER") else 10.)
            finger = max(pressures, key=lambda item: pressures[item] if absolute else differences[item])
            direction = self.world_normals[finger].copy()
            reason = "absolute_pressure_untared_normal" if absolute else "insertion_increase_normal"
            magnitude = float(np.linalg.norm(direction))
            if magnitude < .05:
                direction, reason = np.array([0., 0., 1.]), "normal_unavailable_upward_fallback"
            else:
                direction /= magnitude
            if self.retreat_offset is None:
                self.retreat_offset = kin.position(command) - kin.origin + direction * .006
                self.blocked_at = float(data.time)
                self.retreat_direction = direction.copy()
                self.retreat_reason = reason
            safe = kin.solve(self.retreat_offset, seed=command, tolerance=.0001)
            if safe is None:
                raise failure("CONTACT_NORMAL_RETREAT_IK")
            desired, mode = safe, "external_normal_retreat_6mm"
            direction, reason = self.retreat_direction, self.retreat_reason
        elif scale < 1.:
            desired = command + scale * (desired - command)
            mode = "pressure_crawl" if scale else "pressure_brake"
            if scale == 0. and self.blocked_at is None:
                self.blocked_at = float(data.time)
            if scale > .2:
                self.blocked_at = None
        else:
            self.blocked_at, self.retreat_offset = None, None
        self.latest = {"observation_timestamp_s": float(data.time), "state": state.value,
            "channel_pressures_n": pressures, "tared_increase_n": increase, "scale": scale,
            "mode": mode, "retreat_reason": reason,
            "retreat_direction_world": None if direction is None else direction.tolist(),
            "world_external_hand_forces_n": {finger: value.tolist() for finger, value in self.world_forces.items()},
            "world_external_hand_pressure_normals_n": {finger: value.tolist() for finger, value in self.world_normals.items()},
            "target_pose_used": False, "contact_normal_scope": "Native simulated external hand contacts"}
        self.events.append(self.latest)
        if self.blocked_at is not None and data.time - self.blocked_at > 1.5:
            raise failure("CONTACT_GUARD_NOT_RELIEVED")
        return desired


PLAN = '''
def plan(options):
    kin.copy_open_hand(data)
    fitted = _latest_pose.get('SEARCH', {})
    pose_rotation = np.asarray(fitted['rotation_world_from_object']) if fitted.get('accepted') else np.eye(3)
    kin.rotation = pose_rotation @ _nominal_wrist_rotation
    kin.approach_axis = kin.rotation[:, 2].copy()
    deadline = time.monotonic() + 3.
    for point in options:
        if time.monotonic() > deadline:
            reasons.append('PLANNER_TIMEOUT')
            break
        aperture = np.asarray(grasp_spec.aperture_offset)
        relative_wrist = pose_rotation @ (kin.origin - np.array([.72, 0., .38]) + aperture)
        body_origin = np.asarray(point).copy()
        if fitted.get('accepted'):
            body_origin[2] = fitted['origin_m'][2]
        grasp_offset = body_origin + relative_wrist - kin.origin
        receiving = np.asarray(scenario.zone) + relative_wrist - kin.origin
        waypoints = [grasp_offset + [0., 0., .18], grasp_offset, grasp_offset + [0., 0., .15],
            receiving + [0., 0., .15], receiving + [0., 0., .015],
            receiving - kin.approach_axis * .13 + [0., 0., .12]]
        solved = [kin.solve(value) for value in waypoints]
        if all(value is not None for value in solved):
            minimum = min(kin.minimum_table_clearance(interpolate(solved[0], solved[1], alpha))
                for alpha in np.linspace(0., 1., 21))
            if minimum > .002:
                return point, solved
            reasons.append('INSERTION_TABLE_CLEARANCE')
        else:
            reasons.append('IK_TIMEOUT')
    return None, None
'''


def controller_for(kind, directory, observations, guard, profile):
    original, source, _ = previous.controller_for(kind, directory, observations, guard, profile)
    tree = ast.parse(source)
    run_task = tree.body[0]
    reset = next(i for i, node in enumerate(run_task.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "reset_task")
    run_task.body.insert(reset + 1, ast.parse("_nominal_wrist_rotation = kin.rotation.copy()").body[0])
    plan = next(node for node in run_task.body if isinstance(node, ast.FunctionDef) and node.name == "plan")
    run_task.body[run_task.body.index(plan)] = ast.parse(PLAN).body[0]
    execute = next(node for node in run_task.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    attempt = next(node for node in execute.body if isinstance(node, ast.Try))
    align = next(i for i, node in enumerate(attempt.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "move" and isinstance(node.value.args[0], ast.Attribute)
        and node.value.args[0].attr == "ALIGN")
    attempt.body.insert(align + 1, ast.parse("_reference['grasp_wrist'] = data.xpos[kin.body].copy()").body[0])
    trial = next(i for i, node in enumerate(attempt.body) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "total.contacts == 0")
    accepted = attempt.body[trial].orelse
    accepted[0:0] = ast.parse("""
probe = kin.solve(kin.position(grasp) - kin.origin + [0., 0., .005], seed=command, tolerance=.0001)
move(State.TRIAL_LIFT, probe, 2., .85)
if hand.ready_seconds < .25 or hand.table_support(data) > .2 * float(model.body_mass[target_body]) * abs(float(model.opt.gravity[2])):
    raise MotionFailure('PROBE_LOAD_OR_GRIP_NOT_CONFIRMED')
""").body
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "estimate_object":
            node.keywords.append(ast.keyword(arg="wrist_position", value=ast.parse("data.xpos[kin.body].copy()", mode="eval").body))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "placed_center" for target in node.targets):
            node.value = ast.parse("_visible_origin(placed_belief)", mode="eval").body
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and isinstance(node.right, ast.List):
            if ast.unparse(node.right) == "[0.0, 0.0, 0.06]":
                node.right = ast.parse("[0., 0., .12]", mode="eval").body
    namespace = dict(original.__globals__)
    registration, latest_pose, hands, reference = VisibleMeshPoseRegistration(kind), {}, [], {}

    def estimator(observation, object_kind, *, phase, wrist_position):
        belief = estimate_object(observation, object_kind)
        prediction = None
        initial_fit = latest_pose.get("SEARCH", {})
        if phase == "TRIAL_LIFT" and initial_fit.get("accepted") and "grasp_wrist" in reference:
            prediction = np.asarray(initial_fit["origin_m"]) + np.asarray(wrist_position) - reference["grasp_wrist"]
        fitted = registration.fit(belief.points, initial=phase == "SEARCH", translation_seed=prediction)
        latest_pose[phase] = fitted
        observed_directory = directory / "observations" / f"{len(observations):02d}_{phase}"
        camera = save_observation(observation, observed_directory, prefix="wrist")
        np.save(observed_directory / "visible_surface_points_m.npy", belief.points, allow_pickle=False)
        observations.append({"phase": phase, "timestamp_s": observation.timestamp,
            "path": observed_directory.relative_to(directory).as_posix(), "camera": camera,
            "registration": fitted, "runtime_target_pose_input": False})
        return RegisteredBelief(**vars(belief),
            registered_origin_m=tuple(fitted["origin_m"]) if fitted["accepted"] else None,
            registration_rms_m=fitted.get("rms_m"))

    def make_hand(model, config):
        hand = SynchronizedForceBandHand(model, config, profile, guard)
        hands.append(hand)
        return hand

    def visible_origin(belief):
        origin = getattr(belief, "registered_origin_m", None)
        if origin is not None:
            return np.asarray(origin[:2])
        return None if belief.centroid is None else np.asarray(belief.centroid[:2])

    namespace.update(_phase_seconds=PHASE_SECONDS, _latest_pose=latest_pose, _reference=reference,
        _visible_origin=visible_origin, estimate_object=estimator, HandGraspController=make_hand)
    tree = ast.fix_missing_locations(tree)
    exec(compile(tree, "<public_ycb_pose_aware_force_band_r3>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n", hands


def snapshot(profile_path):
    sources = previous.snapshot()
    for path in (Path(__file__).resolve(), PROJECT / "scripts" / "register_public_ycb_rigid.py", profile_path):
        sources[path.relative_to(PROJECT).as_posix()] = previous.sha256(path)
    return sources


def run(output, profile_path, cases):
    output.mkdir(parents=True, exist_ok=False)
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    before, meshes = snapshot(profile_path), previous.public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    protocol = {"format": "public_ycb_pose_aware_force_band_development_r3",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes_before": before,
        "public_meshes": meshes, "physics_profile": "physical-v6", "maximum_attempts": 2,
        "phase_seconds": PHASE_SECONDS, "nominal_profiles": profiles,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
            "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "nominal_profile_file": {"path": str(profile_path), "bytes": profile_path.stat().st_size, "sha256": previous.sha256(profile_path)},
        "schedule": [{"kind": kind, "seed": seed, "profile": "pose_aware_lp_band"} for kind, seed in cases],
        "precontact": "Common nominal-contact landing time; free closure stops at its nominal shallow reference; zero-contact fingers stay open",
        "force_control": "Original physical gains, torque caps and friction; per-finger measured pressure goals and rising-pressure damping;7N relief trigger",
        "grip_gate": "At least70% declared goals on active load channels plus opposition for.25s, followed by5mm slow physical probe and instrumented-table unloading",
        "pose_input": "Only actual visible RGB-D/public mesh; bounded initial tilt/baseheight; held6D multistart pose with residual/spread rejection",
        "plan": "Initial fitted pose transforms the complete nominal wrist-object relative position and wrist rotation; held estimated origin corrects transfer XY",
        "guard": "All external hand channels; absolute overload uses untared native pressure normal and logs fallback; original arm command limiter",
        "release": "Instrumented table support before release; wrist retreats along approach axis and120mm upward; visual receiving observation excludes pickup workspace filter",
        "scoring": "Original V6 labels retained; independent strict25mm/3s/full-task12N verifier required before any success claim",
        "claim_scope": "Three reused development poses; no held-out performance claim"}
    previous.atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in protocol["schedule"]:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        observations, trace, audit, guard, hands = [], [], None, None, []
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None, "held_lift": None, "runner_error": None}
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data, guard = mujoco.MjData(model), NormalPressureGuard(model)
            spec, config, profile = previous.selected_profile(case["kind"], profiles[case["kind"]])
            config = replace(config, maximum_force_n=7.)
            controller, readable, hands = controller_for(case["kind"], directory, observations, guard, profile)
            (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
            row.update(grasp_specification=asdict(spec), hand_config=asdict(config), grip_profile=profile,
                derived_controller_sha256=previous.sha256(directory / "derived_controller.py"))
            with previous.RichNativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    step_callback=audit.sample, max_attempts=2, support_load_sensor=True, support_required=True,
                    grasp_spec=spec, hand_config_override=config, camera_config=CameraConfig(seed=case["seed"]),
                    placement_retention=True, visual_transfer=True)
            assert not audit.external_qpos_writes and len(audit.samples) == audit.integration_steps
            result["registration_observations"] = observations
            previous.atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], duration_s=result["duration"],
                native_audit=audit.as_dict(), maximum_hand_total_force_n=result["maximum_hand_total_force"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        row["artifacts"] = {"trace": previous.write_gzip(directory / "trace.json.gz", trace),
            "native": previous.write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples),
            "guard": previous.write_gzip(directory / "guard.json.gz", [] if guard is None else guard.events),
            "grip_feedback": previous.write_gzip(directory / "grip_feedback.json.gz", [event for hand in hands for event in hand.feedback_events])}
        previous.atomic_json(directory / "metadata.json", row)
        rows.append(row)
        previous.atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"], "failure", row.get("failure"), "error", row["runner_error"], flush=True)
    after = snapshot(profile_path)
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
        public_meshes_unchanged=meshes == previous.public_mesh_inventory(), ended_at_utc=datetime.now(timezone.utc).isoformat())
    summary = {"protocol": protocol, "rows": rows, "complete": True, "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    previous.atomic_json(output / "protocol.json", protocol)
    previous.atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", help="KIND:SEED, all three identities by default")
    args = parser.parse_args()
    cases = previous.CASES if args.cases is None else tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    if len(set(cases)) != len(cases) or any(kind not in previous.KINDS or seed < 0 for kind, seed in cases):
        parser.error("Unique public kinds and nonnegative seeds required")
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
