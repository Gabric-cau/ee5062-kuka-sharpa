#!/usr/bin/env python3
"""Frozen paired wrist-RGBD, map-only and no-tactile marker evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_v7_robustness import (
    TARGET, EvaluationInstrumentation, atomic_json, episode_metrics, paired_configs,
    rate, sha256, source_snapshot, write_gzip,
)
from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver, save_observation
from kuka_sharpa_grasp.object_registry import ToolObjectRegistry
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import V7TaskExecutor, calibrated_task_spec
from kuka_sharpa_grasp.v7_vision import estimate_upright_cylinder


MODES = ("map_shared", "rgbd_shared", "rgbd_no_tactile")


def snapshot_sources():
    sources = source_snapshot()
    sources["scripts/evaluate_v7_rgbd.py"] = sha256(Path(__file__))
    return sources


def summarize(records):
    groups = {}
    for mode in MODES:
        selected = [row for row in records if row["mode"] == mode]
        groups[mode] = {
            "episodes": len(selected),
            "rates": {key: rate(row["metrics"].get(key) for row in selected) for key in
                      ("strict_physical_success", "target_held_lift", "collision_free",
                       "force_safe", "stable_released_in_slot")},
            "failure_reasons": dict(Counter(row["metrics"].get("failure_reason") or "SUCCESS"
                                            for row in selected)),
        }
    return {"modes": groups}


def run(output, start_seed, count, modes=MODES):
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot_sources()
    development = default_v7_scene_config(0)
    frozen = replace(calibrated_task_spec(TARGET, slot_id=development.registry.by_id(TARGET).dock_slot),
                     dock_tolerance_m=.025, release_verify_seconds=3., phase_timeout_s=12.)
    protocol = {
        "version": "v7_paired_rgbd_precise_r2", "seeds": list(range(start_seed, start_seed + count)),
        "modes": list(modes), "task_spec": asdict(frozen), "calibration_seed": 0,
        "perturbations": {"xy_each_axis_m": [-.005, .005], "mass_factor": [.8, 1.2],
                          "friction_factor": [.8, 1.2]},
        "policy": "RGB-D supplies XY grasp localization only; known colour, radius, height and upright priors",
        "no_tactile_mode": "retention, shared relief/inhibition/stops and tactile phase guards disabled; fixed closure",
        "evaluation": "same object-state lift/release and native contact checks in all modes; body poses never localize the grasp",
        "force_limit_per_channel_n": 12.,
        "selection_policy": "all requested seeds and failed/unknown episodes retained; no evaluation-based parameter changes",
        "source_hashes_before": before, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "mujoco": mujoco.__version__},
    }
    atomic_json(output / "protocol.json", protocol)
    records = []
    for seed in protocol["seeds"]:
        nominal, actual, perturbation = paired_configs(seed)
        seed_path = output / f"seed_{seed}"
        seed_path.mkdir()
        scene = build_v7_workbench_scene(seed_path / "scene.xml", config=actual)
        atomic_json(seed_path / "nominal_manifest.json", nominal.manifest())
        atomic_json(seed_path / "actual_manifest.json", actual.manifest())
        for mode in modes:
            directory = seed_path / mode
            directory.mkdir()
            started = time.perf_counter()
            metadata = {"seed": seed, "mode": mode, "perturbation": perturbation,
                        "scene_sha256": sha256(scene), "source_hashes_before": before}
            artifacts = {}
            try:
                model = mujoco.MjModel.from_xml_path(str(scene))
                blind = mode == "rgbd_no_tactile"
                session = WorkbenchSession(model, nominal, shared_control=not blind,
                                           tactile_retention=not blind)
                task = replace(frozen, tactile_force_min_n=0., tactile_contacts_min=0) if blind else frozen
                instrument = EvaluationInstrumentation(session)
                executor = V7TaskExecutor(session, reset_each_episode=False, step_callback=instrument)
                initial_snapshot = session.snapshot()
                target = nominal.registry.by_id(TARGET)
                grasp = np.asarray(target.pose) + np.asarray(task.grasp_offset_m)
                pregrasp = grasp - np.asarray(task.approach_axis_world) * task.approach_distance_m
                preparation = []
                phase, failure = executor._phase("OBSERVE", pregrasp, task.pregrasp_closure,
                                                trace=preparation, timeout_s=task.phase_timeout_s)
                preparation_actions = list(executor.actions)
                if failure is not None:
                    raise RuntimeError(f"observation approach failed: {failure}")
                with CameraObserver(model, CameraConfig(width=320, height=240)) as camera:
                    observation = camera.poll(session.data)
                    if observation is None:
                        raise RuntimeError("wrist frame unavailable")
                    detection = estimate_upright_cylinder(observation, target)
                    metadata["camera_export"] = save_observation(observation, directory)
                metadata["detection"] = detection.as_dict()
                if mode != "map_shared":
                    if detection.position_m is None:
                        raise RuntimeError(f"RGB-D detection rejected: {detection.reason}")
                    detected = ToolObjectRegistry(replace(item, pose=detection.position_m)
                        if item.object_id == TARGET else item for item in nominal.registry.objects)
                    session.config = replace(nominal, registry=detected)
                result = executor.run(task)
                # Keep exactly one uninstrumented initial state, then every
                # integrated step. run() repeats the end-of-observation state.
                full_trace = (initial_snapshot,) + tuple(preparation) + result.trace[1:]
                actions = preparation_actions + result.metrics["action_log"]
                metrics = dict(result.metrics)
                metrics["peak_tactile_force_n"] = max(row["tactile"]["total_normal_force_n"] for row in full_trace)
                result = replace(result, trace=full_trace, metrics=metrics)
                measured = episode_metrics(result, frozen, nominal)
                if measured["peak_hand_channel_force_n"] is not None:
                    measured["force_safe"] = measured["peak_hand_channel_force_n"] <= 12.
                    strict_parts = (
                        measured["executor_success"], measured["target_held_lift"],
                        measured["collision_free"], measured["force_safe"],
                        measured["stable_released_in_slot"])
                    measured["strict_physical_success"] = (False if False in strict_parts else
                                                            None if None in strict_parts else True)
                artifacts = {"trace": write_gzip(directory / "trace.json.gz", full_trace),
                             "actions": write_gzip(directory / "action_log.json.gz", actions),
                             "events": write_gzip(directory / "events.json.gz", result.metrics["events"])}
                compact = result.as_dict()
                compact.pop("trace")
                compact["metrics"] = {k: v for k, v in compact["metrics"].items() if k not in {"events", "action_log"}}
                atomic_json(directory / "task_result.json", compact)
                metadata["task_spec"] = asdict(task)
                metadata["observation_phase"] = phase.as_dict()
                metadata["shared_control"] = session.shared_control
                metadata["tactile_retention"] = session.tactile_retention
            except Exception as error:
                measured = {key: None for key in ("strict_physical_success", "target_held_lift",
                           "collision_free", "force_safe", "stable_released_in_slot")}
                measured.update(failure_reason=f"RUNTIME_ERROR:{type(error).__name__}",
                                unknown_reasons=[repr(error)])
                metadata["runtime_error"] = repr(error)
            metadata["wall_seconds"] = time.perf_counter() - started
            metadata["artifacts"] = artifacts
            atomic_json(directory / "metadata.json", metadata)
            row = {"seed": seed, "mode": mode, "metrics": measured, "artifacts": artifacts,
                   "path": str(directory.relative_to(output))}
            records.append(row)
            atomic_json(output / "summary.json", {"protocol": protocol, "results": records,
                                                  **summarize(records), "complete": False})
            print(f"seed={seed} mode={mode} strict={measured['strict_physical_success']} "
                  f"failure={measured['failure_reason']}", flush=True)
    after = snapshot_sources()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
                    ended_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "summary.json", {"protocol": protocol, "results": records,
                                          **summarize(records), "complete": True})
    return summarize(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=500)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if args.count < 1 or args.start_seed < 1:
        parser.error("use positive held-out seeds and count")
    run(args.output.resolve(), args.start_seed, args.count, tuple(args.modes))


if __name__ == "__main__":
    main()
