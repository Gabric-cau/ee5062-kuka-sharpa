"""Frozen three-object catalogue experiment for the Segment C review package."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.pick_place import Scenario, run_task
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene

KINDS = ("ycb_can", "ycb_potted_meat_can", "ycb_cracker_box")


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=18)
    parser.add_argument("--selection-seed", type=int, default=20261010)
    parser.add_argument("--pose-seed-start", type=int, default=1000)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.selection_seed)
    schedule = [{"index": i, "kind": str(rng.choice(KINDS)),
                 "pose_seed": args.pose_seed_start + i}
                for i in range(args.trials)]
    sources = [Path(__file__), *sorted((PROJECT / "src").rglob("*.py")),
               PROJECT / "scratch/geometry_objects/objects_experiment.py",
               PROJECT / "scratch/geometry_objects/place_controller.py"]
    frozen = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources if not p.name.startswith("._")}
    write(args.output / "protocol.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "uniform random identity from three public YCB objects",
        "selection_seed": args.selection_seed, "schedule": schedule,
        "position": "x=0.72+U(-0.015,0.015), y=U(-0.025,0.025), z=0.38 m",
        "initial_yaw_radians": 0., "physics_timestep_s": .002,
        "source_sha256": frozen, "python": sys.version,
        "mujoco": mujoco.__version__, "platform": platform.platform(),
        "profiles": {
            "ycb_can": "V5 Cartesian pipeline, at most two physical attempts",
            "ycb_potted_meat_can": "previous calibrated typed-posture profile, one attempt",
            "ycb_cracker_box": "previous catalogue-bounds typed-posture profile, one attempt"},
        "role": "multi-object challenge with frozen per-identity profiles",
        "aggregation": "all scheduled cases retained; profile differences reported"})
    rows, rendered = [], set()
    for case in schedule:
        kind, seed = case["kind"], case["pose_seed"]
        folder = args.output / f"trial_{case['index']:03d}_{kind}"
        folder.mkdir()
        row = {**case, "x_m": Scenario(seed).position[0],
               "y_m": Scenario(seed).position[1], "z_m": .38,
               "success": False, "grasp_success": False, "runner_error": None}
        try:
            if kind == "ycb_can":
                scene = build_pick_place_scene(folder / "scene.xml", kind)
                model = mujoco.MjModel.from_xml_path(str(scene))
                data = mujoco.MjData(model)
                result, trace = run_task(model, data, Scenario(seed), max_attempts=2)
                write(folder / "result.json", result)
                write(folder / "trace_20ms.json", trace[::10] + [trace[-1]])
                row.update(success=bool(result["success"]),
                           grasp_success=any(r["state"] == "TRIAL_LIFT" and
                                             r["oracle"]["lifted"] and
                                             r["oracle"]["hand_contact"] for r in trace),
                           error_m=result["placement_error"],
                           peak_hand_force_n=result["maximum_hand_total_force"],
                           failure=result["failure"], attempt_budget=2)
            else:
                command = [sys.executable, str(PROJECT / "scratch/geometry_objects/objects_experiment.py"),
                           "--kind", kind, "--seed", str(seed), "--output", str(folder)]
                if kind == "ycb_potted_meat_can":
                    command += ["--offset", "-.00242397", ".02941985", ".035",
                                "--force", "4", "--candidate-bias", "0", ".0065",
                                "--place-extra", ".05"]
                if kind not in rendered:
                    command += ["--render"]
                    rendered.add(kind)
                write(folder / "command.json", command)
                process = subprocess.run(command, cwd=PROJECT, capture_output=True,
                                         text=True, check=False)
                (folder / "stdout.log").write_text(process.stdout, encoding="utf-8")
                (folder / "stderr.log").write_text(process.stderr, encoding="utf-8")
                if process.returncode:
                    raise RuntimeError(f"geometry runner exit {process.returncode}: {process.stderr[-1000:]}")
                result = json.loads((folder / f"seed_{seed:03d}/result.json").read_text())
                row.update(success=bool(result["task_success"]),
                           grasp_success=bool(result["grasp_success"]),
                           error_m=result.get("placement_error_m"),
                           peak_hand_force_n=result.get("peak_hand_force_n"),
                           failure=result.get("reason", None if result["task_success"] else "STRICT_PLACEMENT_OR_LIFT"),
                           attempt_budget=1)
            if any(hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() != h
                   for p, h in frozen.items()):
                raise RuntimeError("Source changed during the frozen batch")
        except Exception as exc:
            row["runner_error"] = f"{type(exc).__name__}: {exc}"
            row["success"] = False
        rows.append(row)
        write(args.output / "rows.json", rows)
        print(json.dumps(row), flush=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write(args.output / "summary.json", {
        "complete": len(rows) == len(schedule), "scheduled": len(schedule),
        "attempted": len(rows), "successes": sum(r["success"] for r in rows),
        "groups": {kind: {"attempted": sum(r["kind"] == kind for r in rows),
                           "successes": sum(r["kind"] == kind and r["success"] for r in rows),
                           "lifted": sum(r["kind"] == kind and r["grasp_success"] for r in rows),
                           "runner_errors": sum(r["kind"] == kind and bool(r["runner_error"]) for r in rows)}
                   for kind in KINDS}})


if __name__ == "__main__":
    main()
