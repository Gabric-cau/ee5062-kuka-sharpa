#!/usr/bin/env python3
"""Run a deterministic physical grasp sweep with target-held-lift evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_calibrate import CalibrationRecord, GraspCandidate, calibrate, score_lift_evidence  # noqa: E402
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config  # noqa: E402
from kuka_sharpa_grasp.v7_session import WorkbenchSession  # noqa: E402
from kuka_sharpa_grasp.v7_task import PhaseRecord, TaskResult, V7TaskExecutor  # noqa: E402


def rescore_saved_sweep(run_directory: Path, output: Path):
    """Recompute evidence from hash-verified raw traces without physics edits."""
    run_directory = run_directory.resolve()
    original = json.loads((run_directory / "calibration.json").read_text(encoding="utf-8"))
    rescored = []
    for summary in original["records"]:
        trace_reference = summary["trace_file"]
        path = (run_directory / trace_reference["path"]).resolve()
        if not path.is_relative_to(run_directory):
            raise ValueError("trace path is outside the saved run")
        if hashlib.sha256(path.read_bytes()).hexdigest() != trace_reference["sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        stored = raw["result"]
        result = TaskResult(stored["object_id"], stored["slot_id"], stored["success"], stored["status"],
                            stored["failure_reason"], tuple(PhaseRecord(**phase) for phase in stored["phases"]),
                            tuple(stored["trace"]), stored["metrics"])
        record = CalibrationRecord(GraspCandidate(**raw["candidate"]), result,
                                   summary["max_height_change_m"], summary["max_tactile_force_n"],
                                   summary["max_contacts"], score_lift_evidence(result, stored["object_id"]))
        rescored.append({**record.as_dict(), "trace_file": trace_reference,
                         "candidate_index": raw["candidate_index"]})
    rescored.sort(key=lambda record: record["score"], reverse=True)
    payload = {**original, "format": "kuka-sharpa-v7-grasp-calibration-rescore-r2",
               "original_run": str(run_directory), "physics_rerun": False,
               "accepted_lifts": sum(record["success"] for record in rescored),
               "records": rescored, "best": rescored[0] if rescored else None,
               "scoring_source_sha256": hashlib.sha256((PROJECT / "src/kuka_sharpa_grasp/v7_calibrate.py").read_bytes()).hexdigest()}
    output.mkdir(parents=True)
    (output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "count": len(rescored),
                      "accepted_lifts": payload["accepted_lifts"], "physics_rerun": False}, indent=2))
    return 0


def candidates(seed: int, count: int, profile: str = "mixed"):
    if count < 1:
        raise ValueError("count must be positive")
    if profile not in {"mixed", "neighborhood"}:
        raise ValueError("profile must be mixed or neighborhood")
    rng = np.random.default_rng(seed)
    anchors = ((-.075, .037, .233), (-.065, .037, .233),
               (-.065, .047, .213), (-.065, .047, .233))
    values = [GraspCandidate(anchor, (0., 0., 0.), (0., 0., -1.), .65)
              for anchor in anchors[:count]]
    # Half the budget refines physical lift candidates. The other half covers
    # wide wrist rotations and top-down/oblique approaches to avoid fixing the
    # search to the original V6 posture.
    for index in range(len(values), count):
        if profile == "neighborhood" or index < max(4, count // 2):
            anchor = np.asarray(anchors[index % len(anchors)])
            offset = tuple(anchor + rng.uniform((-.018, -.018, -.018), (.018, .018, .018)))
            rotation = tuple(rng.uniform(-.12, .12, size=3)) if index % 3 else (0., 0., 0.)
            approach = (0., 0., -1.)
            closure = float(rng.choice((.45, .55, .65, .75, .85)))
        else:
            offset = (float(rng.uniform(-.20, .12)), float(rng.uniform(-.16, .16)),
                      float(rng.uniform(.10, .33)))
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rotation = tuple(axis * rng.uniform(-math.pi, math.pi))
            approach = (0., 0., -1.) if index % 3 == 0 else tuple(
                np.asarray((rng.uniform(-1., 1.), rng.uniform(-1., 1.), -rng.uniform(.3, 1.))))
            closure = float(rng.choice((.2, .35, .5, .65, .8, .95)))
        values.append(GraspCandidate(offset, rotation, approach, closure))
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="tool_marker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--phase-timeout", type=float, default=6.0)
    parser.add_argument("--hold-seconds", type=float, default=.08)
    parser.add_argument("--profile", choices=("mixed", "neighborhood"), default="neighborhood")
    parser.add_argument("--rescore-run", type=Path, help="recompute scores from an existing hash-verified sweep")
    parser.add_argument("--manual", action="store_true", help="explore without shared-control stops; safety violations still reject scores")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.count < 1 or not math.isfinite(args.phase_timeout) or args.phase_timeout <= 0
            or not math.isfinite(args.hold_seconds) or args.hold_seconds < 0):
        parser.error("count/phase timeout must be positive and hold seconds finite and non-negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.rescore_run is not None:
        return rescore_saved_sweep(args.rescore_run, args.output)
    args.output.mkdir(parents=True)
    config = default_v7_scene_config(args.seed)
    scene = build_v7_workbench_scene(args.output / "scene.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(scene))
    config.registry.select_target(args.object)
    session = WorkbenchSession(model, config, shared_control=not args.manual)
    trace_directory = args.output / "candidate_traces"
    trace_directory.mkdir()
    trace_files = {}
    source_paths = [Path(__file__), PROJECT / "src/kuka_sharpa_grasp/v7_calibrate.py",
                    PROJECT / "src/kuka_sharpa_grasp/v7_task.py", PROJECT / "src/kuka_sharpa_grasp/v7_session.py",
                    PROJECT / "src/kuka_sharpa_grasp/v7_scene.py"]
    source_before = {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}

    def record_candidate(index, record):
        path = trace_directory / f"candidate_{index:04d}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"candidate_index": index, "candidate": record.candidate.as_dict(),
                       "lift_evidence": record.lift_evidence.as_dict(),
                       "result": record.result.as_dict()}, handle, separators=(",", ":"), allow_nan=False)
        relative = path.relative_to(args.output).as_posix()
        trace_files[id(record.candidate)] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with (args.output / "candidate_records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"candidate_index": index, **record.as_dict(),
                                     "trace_file": trace_files[id(record.candidate)]}, allow_nan=False) + "\n")
        print(json.dumps({"candidate": index + 1, "count": args.count, "score": record.score,
                          "accepted_lift": record.lift_evidence.accepted,
                          "failure": record.result.failure_reason,
                          "height_m": record.lift_evidence.sustained_height_change_m,
                          "held_seconds": record.lift_evidence.sustained_lift_seconds}), flush=True)

    records = calibrate(V7TaskExecutor(session), args.object, candidates(args.seed, args.count, args.profile),
                        phase_timeout_s=args.phase_timeout, hold_seconds=args.hold_seconds,
                        record_callback=record_candidate, retain_traces=False)
    summaries = [{**record.as_dict(), "trace_file": trace_files[id(record.candidate)]} for record in records]
    source_after = {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_paths}
    payload = {"format": "kuka-sharpa-v7-grasp-calibration-r2", "object_id": args.object,
               "seed": args.seed, "count": len(records),
               "phase_timeout_s": args.phase_timeout, "hold_seconds": args.hold_seconds,
               "search_profile": args.profile, "success_scope": "target_held_lift_only_no_docking",
               "shared_control": session.shared_control,
               "accepted_lifts": sum(record.lift_evidence.accepted for record in records),
               "failure_counts": dict(Counter(record.result.failure_reason for record in records)),
               "scene_manifest": config.manifest(),
               "source_sha256": source_before,
               "source_after_sha256": source_after,
               "source_changed_during_sweep": source_before != source_after,
               "controller_memory_reset_policy": "retention reference, engaged fingers and slip history reset per candidate",
               "records": summaries, "best": summaries[0] if summaries else None,
               "qpos_write_policy": "calibrator never writes object qpos; resets are episode initialization"}
    (args.output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best": payload["best"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
