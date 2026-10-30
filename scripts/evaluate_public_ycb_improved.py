#!/usr/bin/env python3
"""Frozen new-seed public-YCB evaluation after independent seed-0 development.

The original baseline and shared controller remain intact. This entry point
uses a longer approach for the meat can and a box upper-side grasp with a
vertical insertion and bounded retreat, without changing physics or scoring.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from develop_public_ycb import longer_motion_controller, source_snapshot
from evaluate_public_ycb import run
from evaluate_v7_robustness import sha256
from kuka_sharpa_grasp import pick_place
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.hand_grasp import HandConfig


BOX_SPEC = replace(GRASP_SPECIFICATIONS["ycb_cracker_box"],
    aperture_offset=(.01920781, .00499887, .20))
BOX_HAND = HandConfig(contact_threshold_n=5.48, maximum_force_n=12.,
    thumb_opposition_rad=1.4, retention_rate=.06, retention_extra=.10,
    retention_minimum_force_n=4.384)


def frozen_controller():
    longer, longer_source = longer_motion_controller()
    box, box_source = longer_motion_controller(vertical=True, lower_retreat=True)

    def controller(model, data, scenario, **kwargs):
        if scenario.object_kind == "ycb_can":
            selected = pick_place.run_task
        elif scenario.object_kind == "ycb_potted_meat_can":
            selected = longer
        elif scenario.object_kind == "ycb_cracker_box":
            selected = box
            kwargs["grasp_spec"] = BOX_SPEC
            kwargs["hand_config_override"] = BOX_HAND
        else:
            raise ValueError("Unsupported public-object identity")
        return selected(model, data, scenario, **kwargs)

    return controller, {"derived_longer_motion.py": longer_source,
                        "derived_box_upper_side_grasp.py": box_source}


def snapshot():
    result = source_snapshot()
    result[Path(__file__).relative_to(PROJECT).as_posix()] = sha256(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20261014)
    parser.add_argument("--pose-seed-start", type=int, default=3400)
    parser.add_argument("--count", type=int, default=18)
    args = parser.parse_args()
    if min(args.selection_seed, args.pose_seed_start) < 0 or args.count < 1:
        parser.error("seeds must be nonnegative and count positive")
    controller, sources = frozen_controller()
    metadata = {
        "entry_point": "scripts/evaluate_public_ycb_improved.py",
        "development_pose_seeds": [0],
        "development_evidence": [
            "results/public_ycb_isolated_development_seed0_20261014",
            "results/public_ycb_box_vertical_development_seed0_20261014",
            "results/public_ycb_box_rim_development_seed0_20261014",
            "results/public_ycb_box_rim_retreat_development_seed0_20261014"],
        "frozen_before_new_pose_evaluation": True,
        "source_scope": "V6 physical-controller dependency files and standalone evaluators; unrelated V7 modules excluded",
        "can": "Existing current controller/profile unchanged",
        "meat": "APPROACH/ALIGN/RECOVER at least 3 s; posture and force parameters unchanged",
        "box": "Same phase durations; pregrasp 180 mm vertically above aperture; aperture z=200 mm; thumb opposition 1.4 rad; retreat extra height 60 mm",
        "box_grasp_specification": asdict(BOX_SPEC), "box_hand_parameters": asdict(BOX_HAND),
        "physics_or_scoring_changes": False,
        "force_boundary": "V6 feedback relief begins above 12 N per finger, but 12 N is not its hard task acceptance gate; native peaks must be reported separately",
        "claim_scope": "New randomized XY poses for three known zero-yaw public meshes; no unseen identity, hardware, or V7 fixture success claim"}
    summary = run(args.output.resolve(), args.selection_seed, args.pose_seed_start,
        args.count, controller=controller, snapshot=snapshot,
        controller_metadata=metadata, derived_sources=sources)
    return 2 if summary["protocol"]["source_changed_during_run"] or not summary["protocol"]["public_meshes_unchanged"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
