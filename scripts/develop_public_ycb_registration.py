#!/usr/bin/env python3
"""Closed-loop development of public-mesh XY candidates on prior failed poses."""

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
from develop_public_ycb import longer_motion_controller
from develop_public_ycb_receiving import DEFAULT_CASES, hand_config
from evaluate_public_ycb import NativeAudit, public_mesh_inventory
from evaluate_public_ycb_improved import BOX_SPEC, snapshot as improved_snapshot
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from register_public_ycb import NominalMeshRegistration, RegisteredSpecification
from kuka_sharpa_grasp import pick_place
from kuka_sharpa_grasp.camera import CameraConfig, save_observation
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.perception import estimate_object
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


PROFILES = ("mesh_current", "mesh_nominal_transfer", "mesh_tilt_current", "mesh_tilt_nominal_transfer")


def snapshot():
    result = improved_snapshot()
    for name in ("register_public_ycb.py", "develop_public_ycb_receiving.py", "develop_public_ycb_registration.py"):
        path = PROJECT / "scripts" / name
        result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return result


def controller_for(kind, directory, observations, *, allow_tilt=False):
    if kind == "ycb_can":
        source = inspect.getsource(pick_place.run_task)
    else:
        _, source = longer_motion_controller(vertical=kind == "ycb_cracker_box",
                                              lower_retreat=kind == "ycb_cracker_box")
    tree = ast.parse(source)
    estimates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "estimate_object"]
    if len(estimates) != 1:
        raise ValueError("Expected one observed RGB-D estimator call")
    estimates[0].keywords.append(ast.keyword(arg="phase", value=ast.Attribute(
        value=ast.Name(id="controller", ctx=ast.Load()), attr="state", ctx=ast.Load())))
    tree = ast.fix_missing_locations(tree)
    registration = NominalMeshRegistration(kind, max_tilt_rad=np.deg2rad(3.) if allow_tilt else 0.)

    def estimator(observation, object_kind, *, phase):
        index = len(observations)
        observation_directory = directory / "observations" / f"{index:02d}_{phase}"
        metadata = save_observation(observation, observation_directory, prefix="wrist")
        if phase in ("SEARCH", "VERIFY"):
            belief, fitted = registration.estimate(observation)
        else:
            belief = estimate_object(observation, object_kind)
            fitted = {"accepted": False, "reason": "UPRIGHT_REGISTRATION_NOT_USED_DURING_HELD_VIEW"}
        np.save(observation_directory / "visible_surface_points_m.npy", belief.points, allow_pickle=False)
        observations.append({"phase": phase, "timestamp_s": observation.timestamp,
            "path": observation_directory.relative_to(directory).as_posix(), "camera": metadata,
            "registration": fitted, "runtime_target_pose_input": False})
        return belief

    namespace = dict(vars(pick_place))
    namespace["estimate_object"] = estimator
    exec(compile(tree, "<public_ycb_registered_candidates>", "exec"), namespace)
    return namespace["run_task"], ast.unparse(tree) + "\n"


def run(output, profiles, cases=DEFAULT_CASES):
    output.mkdir(parents=True, exist_ok=False)
    before, meshes = snapshot(), public_mesh_inventory()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    protocol = {"format": "public_ycb_registered_candidates_development_r1",
        "source_hashes_before": before, "public_meshes": meshes,
        "schedule": [{"kind": kind, "seed": seed, "profile": profile}
            for kind, seed in cases for profile in profiles],
        "physics_profile": "physical-v6", "maximum_attempts": 2, "strict_scoring_changed": False,
        "registered_candidates": "Visible wrist RGB-D points and public STL only, with nominal upright/base-height priors; tilt profiles fit rotation-vector X/Y within +/-3 degrees and fix its Z component to zero",
        "held_view": "Original visible-surface candidates; the upright estimator is not applied to a held rotated object",
        "fallback": "Rejected registered observations retain explicitly logged rejection and use original catalogue candidates",
        "mesh_nominal_transfer": "Existing initial-RGB-D transfer plan and PLACE contact retention until instrumented support; all held observations still exported",
        "claim_scope": "Previously failed validation poses reused for development; not held-out performance"}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in protocol["schedule"]:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        observations = []
        controller, readable = controller_for(case["kind"], directory, observations,
                                              allow_tilt="tilt" in case["profile"])
        (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None,
            "held_lift": None, "runner_error": None, "derived_controller_sha256": sha256(directory / "derived_controller.py")}
        audit, trace = None, []
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            config = hand_config(case["kind"])
            base_spec = BOX_SPEC if case["kind"] == "ycb_cracker_box" else GRASP_SPECIFICATIONS[case["kind"]]
            spec = RegisteredSpecification(**asdict(base_spec))
            nominal = case["profile"].endswith("nominal_transfer")
            with NativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    step_callback=audit.sample, max_attempts=2, support_load_sensor=True, support_required=True,
                    grasp_spec=spec, hand_config_override=config, camera_config=CameraConfig(seed=case["seed"]),
                    placement_retention=nominal, visual_transfer=not nominal)
            assert not audit.external_qpos_writes and audit.integration_steps == len(audit.samples)
            for observation in observations:
                actual = min(audit.samples, key=lambda sample: abs(sample["timestamp"] - observation["timestamp_s"]))
                assert abs(actual["timestamp"] - observation["timestamp_s"]) < 1e-9
                fitted = observation["registration"]
                observation["evaluation_only_body_position_m"] = actual["target_position_world_m"]
                observation["registered_origin_xy_error_mm"] = None if not fitted["accepted"] else 1000 * float(np.linalg.norm(
                    np.asarray(fitted["origin_m"][:2]) - actual["target_position_world_m"][:2]))
            result["registration_observations"] = observations
            atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], duration_s=result["duration"],
                native_audit=audit.as_dict(), maximum_hand_total_force_n=result["maximum_hand_total_force"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        row["artifacts"] = {"trace": write_gzip(directory / "trace.json.gz", trace),
            "native": write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples)}
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"], "failure", row.get("failure"),
              "error", row["runner_error"], flush=True)
    after = snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
                    public_meshes_unchanged=meshes == public_mesh_inventory())
    atomic_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=("mesh_current", "mesh_nominal_transfer"))
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    summary = run(args.output.resolve(), tuple(args.profiles))
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
