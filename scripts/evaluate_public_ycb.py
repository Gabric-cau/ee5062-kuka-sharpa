#!/usr/bin/env python3
"""Frozen random-identity public-YCB baseline using the current bounded controller."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_v7_robustness import atomic_json, rate, sha256, write_gzip
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.physics import get_physics_profile
from kuka_sharpa_grasp.pick_place import Scenario, run_task
from kuka_sharpa_grasp.pick_place_evaluator import held_lift_evidence
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene
from kuka_sharpa_grasp.scene import VENDOR_ROOT, YCB_OBJECTS


KINDS = ("ycb_can", "ycb_potted_meat_can", "ycb_cracker_box")


def schedule_for(selection_seed, pose_seed_start, count):
    rng = np.random.default_rng(selection_seed)
    return [{"index": index, "kind": str(rng.choice(KINDS)),
             "pose_seed": pose_seed_start + index}
            for index in range(count)]


def source_snapshot():
    paths = [*sorted((PROJECT / "src").rglob("*.py")), Path(__file__).resolve(),
             PROJECT / "scripts" / "evaluate_v7_robustness.py"]
    return {path.relative_to(PROJECT).as_posix(): sha256(path)
            for path in paths if not path.name.startswith("._")}


def public_mesh_inventory():
    records = {}
    for kind in KINDS:
        path = (VENDOR_ROOT / YCB_OBJECTS[kind]["mesh_file"]).resolve()
        records[kind] = {"path": path.relative_to(PROJECT).as_posix(),
                         "bytes": path.stat().st_size, "sha256": sha256(path),
                         "source": "public YCB object dataset", "mesh": "poisson/nontextured.stl"}
    return records


class NativeAudit:
    """Read each integrated state and detect target-qpos changes between steps."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.body = model.body("target").id
        self.geom = model.geom("target_geom").id
        self.address = int(model.jnt_qposadr[model.joint("target_free").id])
        self.target_slice = slice(self.address, self.address + 7)
        self.previous_target_qpos = None
        self.external_qpos_writes = []
        self.integration_steps = 0
        self.samples = []
        self.initial_target_qpos = None
        self.original_step = None

    def __enter__(self):
        self.original_step = mujoco.mj_step

        def audited_step(model, data, *args, **kwargs):
            if data is self.data:
                current = data.qpos[self.target_slice].copy()
                if self.previous_target_qpos is None:
                    self.initial_target_qpos = current.tolist()
                elif not np.array_equal(current, self.previous_target_qpos):
                    self.external_qpos_writes.append({"time_s": float(data.time),
                        "before": self.previous_target_qpos.tolist(), "after": current.tolist()})
                self.original_step(model, data, *args, **kwargs)
                self.integration_steps += 1
                self.previous_target_qpos = data.qpos[self.target_slice].copy()
            else:
                self.original_step(model, data, *args, **kwargs)

        mujoco.mj_step = audited_step
        return self

    def __exit__(self, *_):
        mujoco.mj_step = self.original_step

    def sample(self, state, data):
        if self.samples and data.time <= self.samples[-1]["timestamp"]:
            return
        contacts, wrench = [], np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            mujoco.mj_contactForce(self.model, data, index, wrench)
            contacts.append({"geom_ids": [int(contact.geom1), int(contact.geom2)],
                "distance_m": float(contact.dist), "position_world_m": contact.pos.tolist(),
                "force_torque_contact_frame": wrench.tolist()})
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.body, velocity, 0)
        self.samples.append({"timestamp": float(data.time), "state": state,
            "target_position_world_m": data.xpos[self.body].tolist(),
            "target_quaternion_world_wxyz": data.xquat[self.body].tolist(),
            "target_spatial_velocity_world": velocity.tolist(),
            "target_qpos": data.qpos[self.target_slice].tolist(),
            "motor_commands": data.ctrl.tolist(), "native_contacts": contacts})

    def as_dict(self):
        return {"integration_steps": self.integration_steps, "native_samples": len(self.samples),
                "initial_target_qpos": self.initial_target_qpos,
                "external_target_qpos_writes": len(self.external_qpos_writes),
                "write_events": self.external_qpos_writes,
                "scope": "Reset before first step is allowed; every later pre-step target qpos equals the previous integrated state."}


def summarize(rows, scheduled):
    groups = {}
    for kind in KINDS:
        selected = [row for row in rows if row["kind"] == kind]
        groups[kind] = {"attempted": len(selected),
            "task_success": rate(row["success"] for row in selected),
            "held_lift": rate(row["held_lift"] for row in selected),
            "runner_errors": sum(row["runner_error"] is not None for row in selected),
            "failure_reasons": dict(Counter(row.get("failure") or
                ("SUCCESS" if row["success"] is True else "UNKNOWN") for row in selected))}
    success = rate(row["success"] for row in rows)
    success["success_rate_all_attempts"] = success["numerator"] / len(rows) if rows else None
    return {"scheduled": scheduled, "attempted": len(rows),
        "task_success": success, "held_lift": rate(row["held_lift"] for row in rows),
        "runner_errors": sum(row["runner_error"] is not None for row in rows), "groups": groups}


def run(output, selection_seed=20261014, pose_seed_start=3200, count=18, *,
        controller=run_task, snapshot=source_snapshot, controller_metadata=None,
        derived_sources=None):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    schedule = schedule_for(selection_seed, pose_seed_start, count)
    protocol = {"version": "current_physical_v6_random_ycb_r1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "uniform random identity from three public YCB objects",
        "selection_seed": selection_seed, "schedule": schedule,
        "initial_position": "x=0.72+U(-0.015,0.015), y=U(-0.025,0.025), z=0.38 m",
        "initial_yaw_radians": 0., "maximum_attempts_per_episode": 2,
        "physics_profile": asdict(get_physics_profile("physical-v6")),
        "frozen_grasp_specifications": {kind: asdict(GRASP_SPECIFICATIONS[kind]) for kind in KINDS},
        "controller": "current run_task; contact force mode; bounded actuators; physical recovery; table support required",
        "placement_criteria": {"xy_error_m": .045, "base_height_error_m": .025,
            "linear_speed_m_s": .04, "angular_speed_rad_s": .2, "hand_clearance_m": .10,
            "released_stable_seconds": .5, "table_support_required": True},
        "lift_criteria": {"actual_object_height_change_m": .07, "continuous_seconds": .1,
            "hand_contact_required": True, "table_support_absent": True},
        "sensing_priors": ["known catalogue identity and bounds", "upright zero-yaw staging",
            "authored red appearance", "ideal synchronous wrist RGB-D", "simulated instrumented table load"],
        "scope": "Baseline pipeline evaluation of known public objects; not V7 fixture accuracy, learned policy, general 6-DoF perception, or hardware validation",
        "aggregation": "All scheduled episodes retained; runtime-error outcomes unknown and remain in all-attempt denominator",
        "source_hashes_before": before, "public_mesh_inventory": meshes,
        "runtime": {"python": sys.version, "mujoco": mujoco.__version__, "platform": platform.platform()}}
    if controller_metadata is not None:
        protocol["controller_development"] = controller_metadata
        protocol["version"] = "isolated_improved_physical_v6_random_ycb_r1"
        protocol["controller"] = "Frozen independent object-specific derived controller; see controller_development"
    if derived_sources:
        protocol["derived_controller_sources"] = {}
        for name, source in derived_sources.items():
            if Path(name).name != name or not name.endswith(".py"):
                raise ValueError("Derived source names must be plain Python filenames")
            path = output / name
            path.write_text(source, encoding="utf-8")
            protocol["derived_controller_sources"][name] = sha256(path)
    atomic_json(output / "protocol.json", protocol)
    shutil.copy2(Path(__file__).resolve(), output / "frozen_evaluate_public_ycb.py")
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    rows = []
    for case in schedule:
        directory = output / f"trial_{case['index']:03d}_{case['kind']}_seed_{case['pose_seed']}"
        directory.mkdir()
        scenario = Scenario(case["pose_seed"], case["kind"])
        row = {**case, "initial_position_m": list(scenario.position),
               "success": None, "held_lift": None, "runner_error": None,
               "path": directory.relative_to(output).as_posix()}
        started = time.perf_counter()
        trace, audit, frames = [], None, []
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"],
                                           physics_profile="physical-v6", timestep=.002)
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            body, geom = model.body("target").id, model.geom("target_geom").id
            atomic_json(directory / "physical_model.json", {"scene_sha256": sha256(scene),
                "public_mesh": meshes[case["kind"]], "mass_kg": float(model.body_mass[body]),
                "inertia_kg_m2": model.body_inertia[body].tolist(),
                "friction": model.geom_friction[geom].tolist(), "geom_aabb": model.geom_aabb[geom].tolist(),
                "gravity_m_s2": model.opt.gravity.tolist(), "timestep_s": float(model.opt.timestep),
                "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
                               for index in range(model.ngeom)]})
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
            model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [.58, -.04, .52]
                camera.distance, camera.azimuth, camera.elevation = 1.6, 135, -25

                def frame(stage, live):
                    renderer.update_scene(live, camera=camera)
                    pixels = renderer.render().copy()
                    name = f"{len(frames):02d}_{stage}.png"
                    Image.fromarray(pixels).save(directory / name)
                    frames.append({"stage": stage, "timestamp": float(live.time), "file": name,
                                   "pixel_std": float(pixels.std())})

                with NativeAudit(model, data) as audit:
                    result, trace = controller(model, data, scenario, frame_callback=frame,
                        step_callback=audit.sample, max_attempts=2, support_load_sensor=True,
                        support_required=True, grasp_spec=GRASP_SPECIFICATIONS[case["kind"]],
                        camera_config=CameraConfig(seed=case["pose_seed"]))
            if audit.external_qpos_writes:
                raise RuntimeError("EXTERNAL_TARGET_QPOS_WRITE")
            if audit.integration_steps != len(audit.samples):
                raise RuntimeError("NATIVE_EVIDENCE_SAMPLES_INCOMPLETE")
            held = held_lift_evidence(trace)
            assert held == result["actual_held_lift_evidence"]
            result.update(frames=frames, native_audit=audit.as_dict())
            atomic_json(directory / "result.json", result)
            row.update(success=bool(result["success"]), held_lift=bool(held["observed"]),
                failure=result["failure"], placement_error_m=result["placement_error"],
                maximum_hand_total_force_n=result["maximum_hand_total_force"],
                duration_s=result["duration"], physical_attempts=len(result["attempts"]),
                native_audit=audit.as_dict(), rendering=result["rendering"])
        except Exception as error:
            row.update(runner_error=f"{type(error).__name__}: {error}", failure="RUNTIME_ERROR")
        row["artifacts"] = {"trace": write_gzip(directory / "trace.json.gz", trace),
            "native": write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples)}
        row["wall_seconds"] = time.perf_counter() - started
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows,
            **summarize(rows, len(schedule)), "complete": False})
        print(f"trial={case['index']} kind={case['kind']} seed={case['pose_seed']} "
              f"success={row['success']} lift={row['held_lift']} failure={row.get('failure')}", flush=True)
    after, meshes_after = snapshot(), public_mesh_inventory()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
        public_meshes_unchanged=meshes == meshes_after, ended_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "rows": rows, **summarize(rows, len(schedule)), "complete": True}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20261014)
    parser.add_argument("--pose-seed-start", type=int, default=3200)
    parser.add_argument("--count", type=int, default=18)
    args = parser.parse_args()
    if min(args.selection_seed, args.pose_seed_start) < 0 or args.count < 1:
        parser.error("seeds must be nonnegative and count positive")
    summary = run(args.output.resolve(), args.selection_seed, args.pose_seed_start, args.count)
    return 2 if summary["protocol"]["source_changed_during_run"] or not summary["protocol"]["public_meshes_unchanged"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
