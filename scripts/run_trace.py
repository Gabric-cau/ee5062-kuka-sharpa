#!/usr/bin/env python3
"""Write an isolated CPU trial trace without overwriting earlier evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.visualization import (
    add_trial_arguments, create_run_directory, load_trial_record,
    presentation_manifest, resolve_trial, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_trial_arguments(parser, default_object="box")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = load_trial_record(args)
    if args.output is None:
        run_dir = create_run_directory("trace_runs")
        output = run_dir / "trace.csv"
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        run_dir = output.with_name(f"{output.stem}_evidence")
        if output.exists() or run_dir.exists():
            parser.error(f"refusing to overwrite trace or evidence: {output}")
        run_dir.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(build_scene(run_dir / "scene.xml", object_kind=args.object)))
    rollout = GraspRollout(model, mujoco.MjData(model))
    trial = resolve_trial(args, rollout, record)
    result, rows = rollout.run_trace(**trial.rollout_arguments())
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(run_dir / "metadata.json", {**presentation_manifest(trial, "none/headless", run_dir.name),
                                            "result": asdict(result), "trace": str(output)})
    print(result)
    print(f"wrote {output} ({len(rows)} samples); evidence {run_dir}")


if __name__ == "__main__":
    main()
