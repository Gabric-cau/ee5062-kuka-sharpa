#!/usr/bin/env python3
"""Evaluate one or more recorded V7 workbench runs.

Example::

    PYTHONPATH=src .venv/bin/python scripts/evaluate_v7_batch.py \
      --input results/v7_demo_seed0 results/v7_demo_seed1 \
      --output results/v7_batch/summary.json \
      --csv results/v7_batch/episodes.csv

Each input directory must contain ``trace.json``.  ``events.json`` is
optional, but without it collision and whole-task success remain unknown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.v7_batch_eval import BatchMetricSpec, evaluate_batch  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path, index: int, target_id: str | None) -> dict[str, Any]:
    directory = path if path.is_dir() else path.parent
    trace_path = path if path.name == "trace.json" else directory / "trace.json"
    if not trace_path.is_file():
        raise ValueError(f"missing trace.json in {directory}")
    trace = _read_json(trace_path)
    if not isinstance(trace, list):
        raise ValueError(f"trace must be a JSON list: {trace_path}")
    events_path = directory / "events.json"
    events = _read_json(events_path) if events_path.is_file() else None
    if events is not None and not isinstance(events, list):
        raise ValueError(f"events.json must be a JSON list: {events_path}")
    metadata_path = directory / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        metadata = {}
    resolved_target = metadata.get("target_id", target_id)
    return {
        "episode_id": str(metadata.get("episode_id", directory.name or f"episode_{index:04d}")),
        "seed": metadata.get("seed"),
        "target_id": resolved_target,
        "trace": trace,
        "events": events,
    }


def _records_for_input(path: Path, index: int, target_id: str | None) -> list[dict[str, Any]]:
    """Load a teleop run or flatten a ``run_v7_tasks.py`` batch artifact."""
    resolved = path.resolve()
    if resolved.is_file() and resolved.name == "task_results.json":
        payload = _read_json(resolved)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"task_results.json must contain a results list: {resolved}")
        seed = payload.get("seed")
        sibling_events = None
        sibling_events_path = resolved.parent / "events.json"
        if sibling_events_path.is_file():
            loaded_events = _read_json(sibling_events_path)
            if isinstance(loaded_events, list):
                sibling_events = loaded_events

        def events_for_target(object_id: str | None) -> list[dict[str, Any]] | None:
            if sibling_events is None:
                return None
            # run_v7_tasks keeps one event stream for the batch.  Split it at
            # OPERATOR_TARGET markers so a collision is scored only for the
            # episode that was active when it occurred.
            segments: dict[str, list[dict[str, Any]]] = {}
            active: str | None = None
            for event in sibling_events:
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "OPERATOR_TARGET":
                    active = str(event.get("target"))
                    segments.setdefault(active, []).append(event)
                elif active is not None:
                    segments.setdefault(active, []).append(event)
            return segments.get(str(object_id), []) if segments else sibling_events

        records = []
        for result_index, result in enumerate(payload["results"]):
            if not isinstance(result, dict):
                continue
            records.append({
                "episode_id": f"{resolved.parent.name}:{result.get('object_id', result_index)}",
                "seed": seed,
                "target_id": result.get("object_id", target_id),
                "trace": result.get("trace", []),
                "events": events_for_target(result.get("object_id", target_id)),
            })
        return records
    return [_record(resolved, index, target_id)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path,
                        help="run directories or trace.json files")
    parser.add_argument("--output", required=True, type=Path,
                        help="new summary JSON path")
    parser.add_argument("--csv", type=Path,
                        help="optional new per-episode CSV path")
    parser.add_argument("--target", dest="target_id",
                        help="fallback target ID when metadata does not declare one")
    parser.add_argument("--lift-height-m", type=float, default=.020)
    parser.add_argument("--lift-hold-samples", type=int, default=3)
    parser.add_argument("--drop-tolerance-m", type=float, default=.015)
    parser.add_argument("--grasp-proximity-m", type=float, default=.30)
    parser.add_argument("--dock-position-tolerance-m", type=float, default=.035)
    parser.add_argument("--dock-orientation-tolerance-deg", type=float, default=20.)
    parser.add_argument("--require-regrasp", action="store_true",
                        help="make continuous supported-regrasp evidence part of success")
    args = parser.parse_args()
    if args.output.exists() or (args.csv is not None and args.csv.exists()):
        parser.error("refusing to overwrite an existing output")
    return args


def main() -> None:
    args = parse_args()
    records = []
    for index, path in enumerate(args.input):
        records.extend(_records_for_input(path, index, args.target_id))
    thresholds = BatchMetricSpec(
        lift_height_m=args.lift_height_m,
        lift_hold_samples=args.lift_hold_samples,
        drop_tolerance_m=args.drop_tolerance_m,
        grasp_proximity_m=args.grasp_proximity_m,
        dock_position_tolerance_m=args.dock_position_tolerance_m,
        dock_orientation_tolerance_rad=args.dock_orientation_tolerance_deg * 3.141592653589793 / 180.,
        require_regrasp=args.require_regrasp,
    )
    summary = evaluate_batch(records, thresholds=thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.write_json(args.output)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary.write_csv(args.csv)
    report = summary.as_dict()
    print(json.dumps({
        "status": "evaluated",
        "episodes": report["episode_count"],
        "success_rate": report["success_rate"],
        "success_denominator": report["success_denominator"],
        "success_unknown": report["success_unknown"],
        "output": str(args.output),
        "csv": (str(args.csv) if args.csv is not None else None),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
