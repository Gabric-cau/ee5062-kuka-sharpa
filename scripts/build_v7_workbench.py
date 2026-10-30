#!/usr/bin/env python3
"""Build and inspect the V7 cluttered maintenance-workbench scene.

This is the deterministic scene/contract entry point. It does not run a
controller or report task success; use it to create a fresh XML and manifest
before adding a teleoperation replay or a controlled experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.object_registry import default_workbench_registry
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new XML path; an existing file is never overwritten")
    parser.add_argument("--manifest", type=Path,
                        help="optional new JSON manifest path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physics-profile", choices=("legacy-v5", "physical-v6"), default="physical-v6")
    parser.add_argument("--timestep", type=float, choices=(.002, .001), default=.002)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing scene: {args.output}")
    if args.manifest is not None and args.manifest.exists():
        parser.error(f"refusing to overwrite existing manifest: {args.manifest}")
    config = default_v7_scene_config(args.seed)
    output = build_v7_workbench_scene(args.output, config=config,
                                      physics_profile=args.physics_profile,
                                      timestep=args.timestep)
    model = mujoco.MjModel.from_xml_path(str(output))
    report = {
        "scene": str(output),
        "scene_version": config.version,
        "seed": config.seed,
        "physics_profile": args.physics_profile,
        "timestep_s": args.timestep,
        "objects": len(config.registry.objects),
        "actionable_objects": [item.object_id for item in config.registry.actionable],
        "distractors": [item.object_id for item in config.registry.distractors],
        "obstacles": [item.obstacle_id for item in config.obstacles],
        "slots": [item.slot_id for item in config.slots],
        "compiled_model": {"bodies": model.nbody, "geoms": model.ngeom,
                            "sites": model.nsite, "free_joints": 6,
                            "contacts": model.npair},
        "provenance": "V7_DESIGN.md; primitive tool proxies are declared engineering assumptions.",
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"scene": report, "layout": config.manifest()},
                                             indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
