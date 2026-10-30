#!/usr/bin/env python3
"""Independent timing/receiving development on explicitly failed public-YCB poses."""

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from develop_public_ycb import longer_motion_controller
from evaluate_public_ycb import NativeAudit, public_mesh_inventory
from evaluate_public_ycb_improved import BOX_HAND, BOX_SPEC, snapshot as improved_snapshot
from evaluate_v7_robustness import atomic_json, sha256, write_gzip
from kuka_sharpa_grasp import pick_place
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.hand_grasp import HandConfig
from kuka_sharpa_grasp.pick_place import Scenario
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


PHASE_SECONDS = {"APPROACH": 4., "ALIGN": 4., "TRIAL_LIFT": 3.,
    "TRANSFER": 6., "PLACE": 4., "RELEASE": 4., "VERIFY": 4., "RECOVER": 4.}
PROFILES = ("timing", "timing_retention", "gentle_retention", "nominal_transfer_retention")
DEFAULT_CASES = (("ycb_potted_meat_can", 3402), ("ycb_can", 3414), ("ycb_cracker_box", 3406))


def derived_controller(kind, gentle=False):
    _, source = longer_motion_controller(vertical=kind == "ycb_cracker_box",
                                          lower_retreat=kind == "ycb_cracker_box")
    tree = ast.parse(source)
    moves = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move"]
    if len(moves) != 1:
        raise ValueError("Expected one physical move function")
    durations = dict(PHASE_SECONDS)
    if gentle:
        durations["CLOSE"] = 6.
    moves[0].body.insert(1, ast.parse(
        "duration = max(duration, RECEIVING_PHASE_SECONDS.get(state.value, duration))").body[0])
    tree = ast.fix_missing_locations(tree)
    namespace = dict(vars(pick_place))
    namespace["RECEIVING_PHASE_SECONDS"] = durations
    exec(compile(tree, "<public_ycb_receiving_development>", "exec"), namespace)
    readable = "RECEIVING_PHASE_SECONDS = " + repr(durations) + "\n\n" + ast.unparse(tree) + "\n"
    return namespace["run_task"], readable


def hand_config(kind):
    if kind == "ycb_cracker_box":
        return BOX_HAND
    if kind == "ycb_can":
        return HandConfig(contact_threshold_n=3.5, maximum_force_n=12.,
            thumb_opposition_rad=1.1, thumb_mcp_flexion_ratio=.4, thumb_ip_flexion_ratio=1.2,
            retention_rate=.10, retention_extra=.25, retention_minimum_force_n=2.8)
    return HandConfig(contact_threshold_n=4.933333333333334, maximum_force_n=12.,
        retention_rate=.06, retention_extra=.10, retention_minimum_force_n=3.946666666666667)


def snapshot():
    result = improved_snapshot()
    result[Path(__file__).relative_to(PROJECT).as_posix()] = sha256(Path(__file__))
    return result


def run(output, profiles, cases=DEFAULT_CASES):
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    with zipfile.ZipFile(output / "frozen_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in before:
            archive.write(PROJECT / relative, relative)
    schedule = [{"kind": kind, "seed": seed, "profile": profile}
        for kind, seed in cases for profile in profiles]
    protocol = {"format": "public_ycb_timing_receiving_development_r1", "schedule": schedule,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes_before": before,
        "physics_profile": "physical-v6", "public_meshes": public_mesh_inventory(),
        "phase_seconds": PHASE_SECONDS, "gentle_change": "Closure rate .15/s with 6s CLOSE instead of .4/s with 3s CLOSE",
        "receiving_change": "Existing contact retention remains active during PLACE until simulated table load >=0.7 N",
        "nominal_transfer_change": "Initial RGB-D plan supplies receiving translation; held RGB-D observation/validity check still runs",
        "maximum_attempts": 2, "strict_scoring_changed": False,
        "claim_scope": "Failed validation poses deliberately reused for development; not held-out success-rate evidence"}
    atomic_json(output / "protocol.json", protocol)
    rows = []
    for case in schedule:
        directory = output / f"{case['kind']}_{case['profile']}_seed{case['seed']}"
        directory.mkdir()
        gentle = case["profile"] in ("gentle_retention", "nominal_transfer_retention")
        controller, readable = derived_controller(case["kind"], gentle)
        (directory / "derived_controller.py").write_text(readable, encoding="utf-8")
        row = {**case, "path": directory.relative_to(output).as_posix(), "success": None,
            "held_lift": None, "runner_error": None, "derived_controller_sha256": sha256(directory / "derived_controller.py")}
        audit, trace = None, []
        try:
            scene = build_pick_place_scene(directory / "scene.xml", case["kind"], physics_profile="physical-v6")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            config = hand_config(case["kind"])
            if gentle:
                config = replace(config, close_rate=.15)
            spec = BOX_SPEC if case["kind"] == "ycb_cracker_box" else GRASP_SPECIFICATIONS[case["kind"]]
            with NativeAudit(model, data) as audit:
                result, trace = controller(model, data, Scenario(case["seed"], case["kind"]),
                    step_callback=audit.sample, max_attempts=2, support_load_sensor=True, support_required=True,
                    grasp_spec=spec, hand_config_override=config, camera_config=CameraConfig(seed=case["seed"]),
                    placement_retention=case["profile"] != "timing",
                    visual_transfer=case["profile"] != "nominal_transfer_retention")
            assert not audit.external_qpos_writes and audit.integration_steps == len(audit.samples)
            atomic_json(directory / "result.json", result)
            row.update(success=result["success"], held_lift=result["actual_held_lift_evidence"]["observed"],
                failure=result["failure"], attempts=result["attempts"], duration_s=result["duration"],
                hand_parameters=asdict(config), native_audit=audit.as_dict(),
                maximum_hand_total_force_n=result["maximum_hand_total_force"])
        except Exception as error:
            row["runner_error"] = f"{type(error).__name__}: {error}"
        row["artifacts"] = {"trace": write_gzip(directory / "trace.json.gz", trace),
            "native": write_gzip(directory / "native.json.gz", [] if audit is None else audit.samples)}
        atomic_json(directory / "metadata.json", row)
        rows.append(row)
        atomic_json(output / "summary.json", {"protocol": protocol, "rows": rows, "complete": False})
        print(case, "success", row["success"], "held", row["held_lift"], "failure", row.get("failure"), flush=True)
    after = snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
        ended_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "rows": rows, "complete": True,
        "scheduled": len(schedule), "runner_errors": sum(row["runner_error"] is not None for row in rows)}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=("timing", "gentle_retention"))
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    summary = run(args.output.resolve(), tuple(args.profiles))
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
