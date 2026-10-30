#!/usr/bin/env python3
"""Capture initial RGB-D and evaluate nominal-mesh registration without a grasp."""

import argparse
import ast
from dataclasses import asdict
import inspect
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_public_ycb import KINDS, NativeAudit, public_mesh_inventory
from evaluate_public_ycb_improved import snapshot as improved_snapshot
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from register_public_ycb import NominalMeshRegistration
from kuka_sharpa_grasp import pick_place
from kuka_sharpa_grasp.camera import CameraConfig, save_observation
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.perception import estimate_object
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


class CaptureComplete(Exception):
    pass


def snapshot():
    result = improved_snapshot()
    for name in ("capture_public_ycb_registration.py", "register_public_ycb.py"):
        path = PROJECT / "scripts" / name
        result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return result


def run(output, seeds):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    protocol = {"format": "initial_public_ycb_mesh_registration_development_r1",
        "source_hashes_before": before, "public_meshes": meshes, "seeds": list(seeds),
        "physics_profile": "physical-v6", "capture_time_s": .4,
        "scope": "Initial-view perception development, not completed robotic tasks or held-out validation",
        "evaluation_body_pose": "Read only after the estimator returned; never an estimation input"}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for kind in KINDS:
        registration = NominalMeshRegistration(kind)
        for seed in seeds:
            directory = output / f"{kind}_seed{seed}"
            directory.mkdir()
            scene = build_pick_place_scene(directory / "scene.xml", kind, physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            measured = {}

            def capture(observation, object_kind):
                metadata = save_observation(observation, directory, prefix="initial")
                belief = estimate_object(observation, object_kind)
                np.save(directory / "visible_surface_points_m.npy", belief.points, allow_pickle=False)
                candidates, reasons = GRASP_SPECIFICATIONS[kind].candidates(belief, observation.timestamp)
                fitted = registration.fit(belief.points)
                measured.update(timestamp_s=observation.timestamp, camera=metadata,
                    bbox_candidate_m=None if not candidates else candidates[0].tolist(),
                    bbox_rejections=reasons, registration=fitted)
                raise CaptureComplete()

            tree = ast.parse(inspect.getsource(pick_place.run_task))
            namespace = dict(vars(pick_place))
            namespace["estimate_object"] = capture
            exec(compile(tree, "<capture_initial_public_rgbd>", "exec"), namespace)
            with NativeAudit(model, data) as audit:
                try:
                    namespace["run_task"](model, data, Scenario(seed, kind), step_callback=audit.sample,
                        max_attempts=2, support_load_sensor=True, support_required=True,
                        grasp_spec=GRASP_SPECIFICATIONS[kind], camera_config=CameraConfig(seed=seed))
                except CaptureComplete:
                    pass
            assert measured and not audit.external_qpos_writes and len(audit.samples) == audit.integration_steps == 200
            true_position = data.xpos[model.body("target").id].copy()
            fitted = measured["registration"]
            candidate = measured["bbox_candidate_m"]
            row = {"kind": kind, "seed": seed, "path": directory.relative_to(output).as_posix(), **measured,
                "evaluation_only_body_position_m": true_position.tolist(),
                "bbox_origin_xy_error_mm": None if candidate is None else 1000 * float(np.linalg.norm(np.asarray(candidate[:2]) - true_position[:2])),
                "registered_origin_xy_error_mm": None if not fitted["accepted"] else 1000 * float(np.linalg.norm(np.asarray(fitted["origin_m"][:2]) - true_position[:2])),
                "native_audit": audit.as_dict(), "native": write_gzip(directory / "native.json.gz", audit.samples)}
            atomic_json(directory / "result.json", row)
            rows.append(row)
            print(kind, seed, "bbox mm", row["bbox_origin_xy_error_mm"], "registered mm",
                  row["registered_origin_xy_error_mm"], "rms mm", fitted.get("rms_m", 0.) * 1000, flush=True)
    after = snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
                    public_meshes_unchanged=meshes == public_mesh_inventory())
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": True})
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 3402, 3406))
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("nonnegative unique seeds required")
    protocol = run(args.output.resolve(), tuple(args.seeds))
    return 2 if protocol["source_changed_during_run"] or not protocol["public_meshes_unchanged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
