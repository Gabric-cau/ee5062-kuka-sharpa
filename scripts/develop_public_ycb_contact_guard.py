#!/usr/bin/env python3
"""Known-mesh geometry and whole-hand insertion guard development, isolated from V7."""

import argparse
import ast
from dataclasses import asdict, replace
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from develop_public_ycb_registration import controller_for, snapshot as registration_snapshot
from develop_public_ycb_receiving import hand_config
from evaluate_public_ycb import NativeAudit, public_mesh_inventory
from evaluate_public_ycb_improved import BOX_SPEC
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from register_public_ycb import RegisteredSpecification
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.control import ForceReading
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


PROFILES = ("envelope", "envelope_guard")
CASES = (("ycb_potted_meat_can", 3402), ("ycb_cracker_box", 3406))


class InsertionGuard:
    """Pressure from any external hand contact; no target pose or geometry labels."""
    def __init__(self, model):
        self.model = model
        self.hand = {}
        for gid in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[gid])).name
            if body.startswith("left_") and model.geom_contype[gid]:
                self.hand[gid] = next((finger for finger in ("thumb", "index", "middle", "ring", "pinky")
                    if body.startswith("left_" + finger + "_")), "palm")
        self.events, self.latest = [], {}
        self.state = None
        self.baseline = {}
        self.blocked_at = None
        self.retreat_offset = None

    def read(self, data):
        readings = {finger: ForceReading() for finger in ("thumb", "index", "middle", "ring", "pinky", "palm")}
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.hand) == (b in self.hand):
                continue
            finger = self.hand[a if a in self.hand else b]
            mujoco.mj_contactForce(self.model, data, index, wrench)
            readings[finger].normal += abs(float(wrench[0]))
            readings[finger].tangential += float(np.linalg.norm(wrench[1:3]))
            readings[finger].contacts += 1
        return readings

    def begin(self, state, data):
        self.state = state.value
        self.baseline = {finger: reading.normal for finger, reading in self.read(data).items()}
        self.blocked_at, self.retreat_offset = None, None

    def guide(self, state, desired, command, kin, data, failure):
        readings = self.read(data)
        pressures = {finger: reading.normal for finger, reading in readings.items()}
        peak = max(pressures.values())
        increase = max(pressures[finger] - self.baseline[finger] for finger in pressures)
        mode, scale = "pass", 1.
        if state.value in ("APPROACH", "ALIGN"):
            risk = peak
            scale = float(np.clip((5. - risk) / 3., 0., 1.))
            retreat = risk > 7.
        elif state.value == "PLACE":
            # A stable pre-existing grip is tared. New pressure during descent
            # and an absolute overshoot still stop the Cartesian feed.
            risk = max(increase, peak - 8.)
            scale = float(np.clip((3. - risk) / 2., 0., 1.))
            retreat = peak > 12.
        elif state.value in ("TRIAL_LIFT", "TRANSFER", "CLOSE"):
            scale = float(np.clip((12. - peak) / 4., 0., 1.))
            retreat = False
        else:
            retreat = False
        if retreat:
            if self.retreat_offset is None:
                self.retreat_offset = kin.position(command) - kin.origin + [0., 0., .006]
                self.blocked_at = float(data.time)
            safe = kin.solve(self.retreat_offset, seed=command, tolerance=.0001)
            if safe is None:
                raise failure("CONTACT_RETREAT_IK")
            desired, mode = safe, "vertical_retreat_6mm"
        elif scale < 1.:
            desired = command + scale * (desired - command)
            mode = "pressure_crawl" if scale else "pressure_brake"
            if scale == 0. and self.blocked_at is None:
                self.blocked_at = float(data.time)
            if scale > .2:
                self.blocked_at = None
        else:
            self.blocked_at = None
        self.latest = {"observation_timestamp_s": float(data.time), "state": state.value,
            "channel_pressures_n": pressures, "tared_increase_n": increase,
            "scale": scale, "mode": mode, "target_pose_used": False}
        self.events.append(self.latest)
        if self.blocked_at is not None and data.time - self.blocked_at > 1.:
            raise failure("CONTACT_GUARD_NOT_RELIEVED")
        return desired

    def relieve(self, hand, data, dt):
        for finger, reading in self.read(data).items():
            if finger in hand.closure:
                hand.relieve(finger, reading, dt)


def snapshot():
    result = registration_snapshot()
    path = Path(__file__).resolve()
    result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return result


def guarded_controller(kind, directory, observations, guard):
    original, source = controller_for(kind, directory, observations, allow_tilt=True)
    tree = ast.parse(source)
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    enter = next(index for index, node in enumerate(move.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "enter")
    move.body.insert(enter + 1, ast.parse("_contact_guard.begin(state, data)").body[0])
    loop = next(node for node in move.body if isinstance(node, ast.For))
    command = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
    loop.body.insert(command, ast.parse(
        "desired_command = _contact_guard.guide(state, desired_command, command, kin, data, MotionFailure)").body[0])
    write = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "hand"
        and node.value.func.attr == "write")
    loop.body.insert(write, ast.parse("_contact_guard.relieve(hand, data, model.opt.timestep)").body[0])
    recorded = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "total" for target in node.targets)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "record")
    loop.body.insert(recorded + 1, ast.parse("trace[-1]['contact_guard'] = _contact_guard.latest.copy()").body[0])
    tree = ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_contact_guard"] = guard
    exec(compile(tree, "<public_ycb_whole_hand_contact_guard>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n"


def envelope(kind):
    base = BOX_SPEC if kind == "ycb_cracker_box" else GRASP_SPECIFICATIONS[kind]
    x, y, _ = base.aperture_offset
    spec = replace(base, aperture_offset=(x, y, .22 if kind == "ycb_cracker_box" else .075))
    config = replace(hand_config(kind), thumb_opposition_rad=1.4 if kind == "ycb_cracker_box" else 1.2)
    return RegisteredSpecification(**asdict(spec)), config


def run(output, profiles, cases=CASES):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    protocol = {"format": "public_ycb_contact_envelope_development_r1", "source_hashes_before": before,
        "public_meshes": meshes, "physics_profile": "physical-v6", "maximum_attempts": 2,
        "schedule": [{"kind": kind, "seed": seed, "profile": profile} for kind, seed in cases for profile in profiles],
        "envelope": "Nominal mesh collision scan selects meat aperture z75mm/opposition1.2 and box z220mm/opposition1.4; no live object pose input",
        "guard": "External whole-hand channel pressure scales/brakes Cartesian requests and asks for bounded 6mm upward retreat during insertion/descent. Original arm velocity/acceleration limiter remains in the command path.",
        "guard_hand": "Closure0.15/s, finger relief0.6/s at8N per channel; physics gains/torque limits/friction/mass unchanged",
        "placement": "Existing initial-camera transfer plan and instrumented support retention; original scoring unchanged",
        "claim_scope": "Previously failed poses reused for development, not held-out accuracy or V7 force-qualified success"}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in protocol["schedule"]:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        observations, trace, audit, guard = [], [], None, None
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None,
            "held_lift": None, "runner_error": None}
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data, guard = mujoco.MjData(model), InsertionGuard(model)
            if case["profile"] == "envelope_guard":
                controller, readable = guarded_controller(case["kind"], directory, observations, guard)
            else:
                controller, readable = controller_for(case["kind"], directory, observations, allow_tilt=True)
            (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
            spec, config = envelope(case["kind"])
            if case["profile"] == "envelope_guard":
                config = replace(config, close_rate=.15, maximum_force_n=8., relief_rate=.6)
            row.update(grasp_specification=asdict(spec), hand_config=asdict(config),
                derived_controller_sha256=sha256(directory / "derived_controller.py"))
            with NativeAudit(model, data) as audit:
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
            "guard": write_gzip(directory / "guard.json.gz", [] if guard is None else guard.events)}
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"], "failure", row.get("failure"),
            "error", row["runner_error"], flush=True)
    after = snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
        public_meshes_unchanged=meshes == public_mesh_inventory())
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    summary = run(args.output.resolve(), tuple(args.profiles))
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
