#!/usr/bin/env python3
"""Compare upright/small-tilt mesh fitting on frozen real rendered RGB-D clouds."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_v7_robustness import atomic_json, sha256
from register_public_ycb import NominalMeshRegistration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.directory.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((directory / "summary.json").read_text())
    assert original["complete"] and not original["protocol"]["source_changed_during_run"]
    rows = []
    for row in original["rows"]:
        cloud_file = directory / row["path"] / "visible_surface_points_m.npy"
        points = np.load(cloud_file, allow_pickle=False)
        fitted = NominalMeshRegistration(row["kind"], max_tilt_rad=np.deg2rad(3.)).fit(points)
        # Ground truth enters only the post-hoc score, after estimation returned.
        actual = np.asarray(row["evaluation_only_body_position_m"])
        error = None if not fitted["accepted"] else 1000 * float(np.linalg.norm(np.asarray(fitted["origin_m"][:2]) - actual[:2]))
        rows.append({"kind": row["kind"], "seed": row["seed"],
            "input_cloud": str(cloud_file), "input_sha256": sha256(cloud_file),
            "upright_xy_error_mm": row["registered_origin_xy_error_mm"],
            "small_tilt_xy_error_mm": error, "registration": fitted,
            "evaluation_only_body_position_m": actual.tolist()})
        print(row["kind"], row["seed"], "upright mm", row["registered_origin_xy_error_mm"],
              "tilt mm", error, "tilt degrees", np.rad2deg(fitted.get("estimated_tilt_rotvec_rad", [])).tolist(), flush=True)
    summary = {"format": "public_ycb_registration_small_tilt_offline_replay_r2",
        "input_summary_sha256": sha256(directory / "summary.json"),
        "script_sha256": sha256(Path(__file__)),
        "estimator_sha256": sha256(PROJECT / "scripts" / "register_public_ycb.py"),
        "runtime_scipy": scipy.__version__, "complete": True, "rows": rows,
        "scope": "Same nine captured development clouds, known identity/near-upright and height priors; no new robot trial or hardware/6-DoF generalization",
        "ground_truth_estimator_input": False}
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
