#!/usr/bin/env python3
"""Isolated development of public-object motion timing and an existing thumb profile.

The derived controller is saved as readable Python beside its evidence. Its
shared source remains untouched while V7 safety/release work is in progress.
"""

import argparse
import ast
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys
import zipfile

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_public_ycb import NativeAudit, public_mesh_inventory
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.hand_grasp import HandConfig
from kuka_sharpa_grasp import pick_place as tasks
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


PROFILES = ("current", "longer_motion", "longer_motion_can_thumb",
            "longer_motion_vertical", "longer_motion_vertical_clearance",
            "longer_motion_vertical_box_rim", "longer_motion_vertical_box_rim_retreat")
LEGACY_DEPENDENCIES = ("__init__", "camera", "catalogue", "control", "evaluator", "hand_grasp",
    "motion", "oracle", "perception", "physics", "pick_place", "pick_place_controller",
    "pick_place_evaluator", "pick_place_scene", "rollout", "scene", "support")


def source_snapshot():
    paths = [PROJECT / "src" / "kuka_sharpa_grasp" / f"{name}.py" for name in LEGACY_DEPENDENCIES]
    paths += [Path(__file__).resolve(), PROJECT / "scripts" / "evaluate_public_ycb.py",
              PROJECT / "scripts" / "evaluate_v7_robustness.py"]
    return {path.relative_to(PROJECT).as_posix(): sha256(path) for path in paths}


def longer_motion_controller(vertical=False, lower_retreat=False):
    """Derive one explicit phase-duration change without modifying the imported module."""
    tree = ast.parse(inspect.getsource(tasks.run_task))
    moves = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move"]
    if len(moves) != 1:
        raise ValueError("Expected one nested physical move function")
    replacement = ast.parse("if state in (State.APPROACH, State.ALIGN, State.RECOVER):\n"
                            "    duration = max(duration, 3.0)\n").body[0]
    moves[0].body.insert(1, replacement)
    if vertical:
        waypoint_assignments = [node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "waypoints" for target in node.targets)]
        if len(waypoint_assignments) != 1 or not isinstance(waypoint_assignments[0].value, ast.List):
            raise ValueError("Expected one physical waypoint list")
        waypoint_assignments[0].value.elts[0] = ast.parse(
            "grasp_offset + [0., 0., .18]", mode="eval").body
    if lower_retreat:
        retreats = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "retreat" for target in node.targets)]
        if len(retreats) != 1:
            raise ValueError("Expected one visual-transfer retreat")
        retreats[0].value = ast.parse(
            "kin.solve(transfer_offset - kin.approach_axis * .13 + [0., 0., .06])", mode="eval").body
    tree = ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    namespace = dict(vars(tasks))
    exec(compile(tree, "<public_ycb_longer_motion>", "exec"), namespace)
    return namespace["run_task"], source


def can_thumb_profile():
    # The existing physical-v6 soup-can profile; this comparison does not invent
    # new joint ratios from evaluation outcomes.
    return HandConfig(contact_threshold_n=3.5, maximum_force_n=12.,
        thumb_opposition_rad=1.1, thumb_mcp_flexion_ratio=.4, thumb_ip_flexion_ratio=1.2,
        retention_rate=.10, retention_extra=.25, retention_minimum_force_n=2.8)


def run(output, seed=0, kinds=("ycb_potted_meat_can", "ycb_cracker_box"), profiles=PROFILES):
    output.mkdir(parents=True, exist_ok=False)
    before = source_snapshot()
    derived, derived_source = longer_motion_controller()
    vertical, vertical_source = longer_motion_controller(vertical=True)
    rim_retreat, rim_retreat_source = longer_motion_controller(vertical=True, lower_retreat=True)
    (output / "derived_run_task.py").write_text(derived_source, encoding="utf-8")
    (output / "derived_vertical_run_task.py").write_text(vertical_source, encoding="utf-8")
    (output / "derived_box_rim_retreat_run_task.py").write_text(rim_retreat_source, encoding="utf-8")
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    meshes = public_mesh_inventory()
    schedule = [{"kind": kind, "seed": seed, "profile": profile} for kind in kinds for profile in profiles]
    protocol = {"format": "public_ycb_isolated_controller_development_r1", "schedule": schedule,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes_before": before,
        "source_scope": "V6 controller dependency files; V7-only unrelated modules are outside this experiment",
        "physics_profile": "physical-v6", "maximum_attempts": 2,
        "longer_motion_change": "APPROACH, ALIGN, RECOVER duration at least 3 s; torque/velocity/acceleration and tracking gates unchanged",
        "can_thumb_profile": asdict(can_thumb_profile()), "public_meshes": meshes,
        "derived_controller_sha256": sha256(output / "derived_run_task.py"),
        "derived_vertical_controller_sha256": sha256(output / "derived_vertical_run_task.py"),
        "derived_box_rim_retreat_controller_sha256": sha256(output / "derived_box_rim_retreat_run_task.py"),
        "vertical_insertion_change": "Pregrasp is 180 mm vertically above aperture; same wrist orientation",
        "box_open_clearance_change": "Known-box nominal aperture x offset minus 12 mm; offline mesh/hand geometry prior, not runtime target-pose input",
        "box_rim_change": "Known-box aperture z=200 mm and thumb opposition=1.4 rad; open-hand nominal mesh clearance=7 mm, fingers close on upper side faces",
        "box_rim_retreat_change": "Visual-transfer retreat extra vertical height 60 mm rather than 100 mm; 1.1 m workspace guard unchanged",
        "claim_scope": "Development seed comparison; not a held-out success-rate estimate or repair of the original frozen baseline",
        "runtime": {"python": sys.version, "mujoco": mujoco.__version__}}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in schedule:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        row = {**case, "success": None, "held_lift": None, "runner_error": None,
               "path": directory.relative_to(output).as_posix()}
        audit, trace = None, []
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model, data = mujoco.MjModel.from_xml_path(str(scene)), None
            data = mujoco.MjData(model)
            controller = (tasks.run_task if case["profile"] == "current" else
                          rim_retreat if case["profile"].endswith("box_rim_retreat") else
                          vertical if "vertical" in case["profile"] else derived)
            kwargs = {}
            if case["profile"] == "longer_motion_can_thumb":
                kwargs["hand_config_override"] = can_thumb_profile()
            spec = GRASP_SPECIFICATIONS[case["kind"]]
            if case["profile"] == "longer_motion_vertical_clearance" and case["kind"] == "ycb_cracker_box":
                x, y, z = spec.aperture_offset
                spec = replace(spec, aperture_offset=(x - .012, y, z))
            if "box_rim" in case["profile"] and case["kind"] == "ycb_cracker_box":
                x, y, _ = spec.aperture_offset
                spec = replace(spec, aperture_offset=(x, y, .20))
                grip_target = min(8., 2. * float(model.body_mass[model.body("target").id]) / .15)
                kwargs["hand_config_override"] = HandConfig(contact_threshold_n=grip_target,
                    maximum_force_n=12., thumb_opposition_rad=1.4, retention_rate=.06,
                    retention_extra=.10, retention_minimum_force_n=.8 * grip_target)
            with NativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    step_callback=audit.sample, max_attempts=2, support_load_sensor=True, support_required=True,
                    grasp_spec=spec, camera_config=CameraConfig(seed=case["seed"]), **kwargs)
            assert audit.integration_steps == len(audit.samples) and not audit.external_qpos_writes
            atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], native_audit=audit.as_dict(),
                maximum_hand_total_force_n=result["maximum_hand_total_force"], placement_error_m=result["placement_error"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        row["artifacts"] = {"trace": write_gzip(directory / "trace.json.gz", trace),
            "native": write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples)}
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"],
              "failure", row.get("failure"), "runner_error", row["runner_error"], flush=True)
    after = source_snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
                    ended_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "scheduled": len(schedule), "runner_errors": sum(row["runner_error"] is not None for row in rows),
        "first_attempt_failures": dict(Counter((row.get("attempts") or [{}])[0].get("failure") for row in rows))}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kinds", nargs="+", choices=tuple(GRASP_SPECIFICATIONS), default=("ycb_potted_meat_can", "ycb_cracker_box"))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    args = parser.parse_args()
    if args.seed < 0 or len(set(args.kinds)) != len(args.kinds) or len(set(args.profiles)) != len(args.profiles):
        parser.error("seed nonnegative and unique kinds/profiles required")
    summary = run(args.output.resolve(), args.seed, tuple(args.kinds), tuple(args.profiles))
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
