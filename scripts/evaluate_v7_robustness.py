#!/usr/bin/env python3
"""Paired marker evaluation with a frozen map and construction-time errors.

The task preset is frozen from development seed 0. Model perturbations are
introduced by the scene builder; evaluation state never enters the policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.object_registry import ToolObjectRegistry
from kuka_sharpa_grasp.v7_calibrate import CalibrationContactReader, EVIDENCE_KEY, score_lift_evidence
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import V7TaskExecutor, calibrated_task_spec


TARGET = "tool_marker"
RELEASE_LINEAR_SPEED = .02
RELEASE_ANGULAR_SPEED = .25
RELEASE_TARGET_FORCE = .05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict[str, str]:
    paths = sorted((PROJECT / "src").rglob("*.py"))
    paths += [Path(__file__).resolve(), PROJECT / "scripts" / "run_v7_tasks.py"]
    return {str(path.relative_to(PROJECT)): sha256(path) for path in paths}


def atomic_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_gzip(path: Path, data) -> dict[str, object]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
    return {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def perturbation_for_seed(seed: int) -> dict[str, object]:
    rng = random.Random(0x50620000 + int(seed))
    return {
        "xy_error_m": [rng.uniform(-.005, .005), rng.uniform(-.005, .005)],
        "mass_factor": rng.uniform(.8, 1.2),
        "friction_factor": rng.uniform(.8, 1.2),
    }


def paired_configs(seed: int):
    nominal = default_v7_scene_config(seed)
    perturbation = perturbation_for_seed(seed)
    dx, dy = perturbation["xy_error_m"]
    target = nominal.registry.by_id(TARGET)
    actual_target = replace(
        target, pose=(target.pose[0] + dx, target.pose[1] + dy, target.pose[2]),
        mass_kg=target.mass_kg * perturbation["mass_factor"],
        friction=tuple(value * perturbation["friction_factor"] for value in target.friction),
    )
    registry = ToolObjectRegistry(
        actual_target if spec.object_id == TARGET else spec for spec in nominal.registry.objects)
    actual = replace(nominal, registry=registry)
    if len(nominal.registry.distractors) != 3 or len(actual.registry.distractors) != 3:
        raise ValueError("robustness protocol requires all three distractors")
    return nominal, actual, perturbation


class EvaluationInstrumentation:
    """Read physical contact and velocity evidence after each integrated step."""

    def __init__(self, session: WorkbenchSession):
        self.session = session
        self.contacts = CalibrationContactReader(session, TARGET)
        self.body_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_BODY, f"v7_obj_{TARGET}")

    def __call__(self, data, snapshot) -> None:
        evidence = self.contacts.read()
        spatial_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.session.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.body_id, spatial_velocity, 0)
        evidence["target_angular_velocity_world_rad_s"] = spatial_velocity[:3].tolist()
        evidence["target_linear_velocity_world_m_s"] = spatial_velocity[3:].tolist()
        evidence["target_linear_speed_m_s"] = float(np.linalg.norm(spatial_velocity[3:]))
        evidence["target_angular_speed_rad_s"] = float(np.linalg.norm(spatial_velocity[:3]))
        snapshot[EVIDENCE_KEY] = evidence


def rate(values) -> dict[str, int | float | None]:
    values = list(values)
    known = [value for value in values if value is not None]
    numerator = sum(value is True for value in known)
    return {"numerator": numerator, "denominator": len(known),
            "unknown": len(values) - len(known), "total": len(values),
            "rate": numerator / len(known) if known else None}


def all_evidence(rows, fields=()) -> bool:
    return bool(rows) and all(EVIDENCE_KEY in row and all(
        key in row[EVIDENCE_KEY] for key in fields) for row in rows)


def release_metrics(result, task, nominal) -> dict[str, object]:
    phases = {phase.name: phase for phase in result.phases}
    release = phases.get("RELEASE")
    verify = phases.get("VERIFY")
    if release is None or verify is None:
        return {"stable_released_in_slot": False, "reason": "RELEASE_OR_VERIFY_NOT_REACHED",
                "stable_seconds": 0., "final_linear_speed_m_s": None,
                "final_angular_speed_rad_s": None, "final_target_hand_force_n": None}
    rows = [row for row in result.trace if verify.start_time_s < row["time_s"] <= verify.end_time_s + 1e-9]
    required = ("target_linear_speed_m_s", "target_angular_speed_rad_s", "target_hand_force_n")
    if not all_evidence(rows, required):
        return {"stable_released_in_slot": None, "reason": "RELEASE_EVIDENCE_INCOMPLETE",
                "stable_seconds": None, "final_linear_speed_m_s": None,
                "final_angular_speed_rad_s": None, "final_target_hand_force_n": None}
    slot = next(slot for slot in nominal.slots if slot.slot_id == task.slot_id)
    stable_start = None
    previous_time = None
    best_seconds = 0.
    endpoint_valid = False
    for row in rows:
        evidence = row[EVIDENCE_KEY]
        position = np.asarray(row["evaluation"][TARGET]["position"])
        timestamp = float(row["time_s"])
        endpoint_valid = (
            np.linalg.norm(position - np.asarray(slot.pose)) <= task.dock_tolerance_m
            and evidence["target_linear_speed_m_s"] < RELEASE_LINEAR_SPEED
            and evidence["target_angular_speed_rad_s"] < RELEASE_ANGULAR_SPEED
            and evidence["target_hand_force_n"] <= RELEASE_TARGET_FORCE
            and max(row["finger_closure"].values()) <= .03
            and (previous_time is None or timestamp - previous_time <= .02 + 1e-9))
        if endpoint_valid:
            if stable_start is None:
                stable_start = timestamp
            best_seconds = max(best_seconds, timestamp - stable_start)
        else:
            stable_start = None
        previous_time = timestamp
    ending_seconds = 0. if stable_start is None else rows[-1]["time_s"] - stable_start
    final = rows[-1][EVIDENCE_KEY]
    return {
        "stable_released_in_slot": bool(release.status == "COMPLETE" and verify.status == "COMPLETE"
                                        and endpoint_valid and ending_seconds >= task.release_verify_seconds - 1e-9),
        "reason": None if ending_seconds >= task.release_verify_seconds - 1e-9 else "NO_STABLE_RELEASE_WINDOW",
        "stable_seconds": ending_seconds, "maximum_stable_seconds": best_seconds,
        "final_linear_speed_m_s": final["target_linear_speed_m_s"],
        "final_angular_speed_rad_s": final["target_angular_speed_rad_s"],
        "final_target_hand_force_n": final["target_hand_force_n"],
    }


def episode_metrics(result, task, nominal) -> dict[str, object]:
    rows = list(result.trace[1:])
    contact_complete = all_evidence(rows, ("environment_collisions", "maximum_hand_channel_force_n"))
    lift = score_lift_evidence(result, TARGET)
    lift_known = contact_complete
    release = release_metrics(result, task, nominal)
    collision_count = sum(bool(row[EVIDENCE_KEY]["environment_collisions"]) for row in rows
                          if EVIDENCE_KEY in row)
    collision_free = collision_count == 0 if contact_complete else None
    peaks = [row[EVIDENCE_KEY]["maximum_hand_channel_force_n"] for row in rows if EVIDENCE_KEY in row]
    force_safe = max(peaks, default=0.) <= 30. if contact_complete else None
    held = lift.accepted if lift_known else None
    strict_parts = [bool(result.success), held, collision_free, force_safe, release["stable_released_in_slot"]]
    strict = False if False in strict_parts else None if None in strict_parts else True
    final_q = result.trace[-1]["evaluation"][TARGET]["quaternion"] if result.trace else None
    upright_error = None
    if final_q is not None:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(final_q, dtype=float))
        upright_error = math.degrees(math.acos(float(np.clip(rotation.reshape(3, 3)[2, 2], -1., 1.))))
    return {
        "executor_success": bool(result.success), "strict_physical_success": strict,
        "failure_reason": result.failure_reason, "target_held_lift": held,
        "collision_free": collision_free, "collision_sample_count": collision_count if contact_complete else None,
        "force_safe": force_safe, "peak_hand_channel_force_n": max(peaks) if peaks else None,
        "peak_total_tactile_force_n": result.metrics.get("peak_tactile_force_n"),
        "stable_released_in_slot": release["stable_released_in_slot"],
        "release_evidence": release, "lift_evidence": lift.as_dict(),
        "dock_error_m": result.metrics.get("dock_error_m"), "final_upright_error_deg": upright_error,
        "evidence_samples": len(peaks), "expected_evidence_samples": len(rows),
        "unknown_reasons": [] if contact_complete else ["CONTACT_EVIDENCE_INCOMPLETE"],
    }


def summary(records) -> dict[str, object]:
    metrics = ("executor_success", "strict_physical_success", "target_held_lift", "collision_free",
               "force_safe", "stable_released_in_slot")
    groups = {}
    for mode in ("shared", "manual"):
        selected = [record for record in records if record["mode"] == mode]
        groups[mode] = {
            "episodes": len(selected),
            "rates": {name: rate(record["metrics"].get(name) for record in selected) for name in metrics},
            "failure_reasons": dict(Counter(record["metrics"].get("failure_reason") or "SUCCESS" for record in selected)),
            "peak_hand_channel_force_n": max((record["metrics"]["peak_hand_channel_force_n"]
                                              for record in selected if record["metrics"].get("peak_hand_channel_force_n") is not None), default=None),
        }
    by_seed = {}
    for record in records:
        by_seed.setdefault(record["seed"], {})[record["mode"]] = record["metrics"]["strict_physical_success"]
    pairs = [pair for pair in by_seed.values() if set(pair) == {"shared", "manual"}]
    known_pairs = [pair for pair in pairs if None not in pair.values()]
    return {"modes": groups, "paired_comparison": {
        "complete_pairs": len(pairs), "known_pairs": len(known_pairs), "unknown_pairs": len(pairs) - len(known_pairs),
        "both_success": sum(pair["shared"] and pair["manual"] for pair in known_pairs),
        "shared_only": sum(pair["shared"] and not pair["manual"] for pair in known_pairs),
        "manual_only": sum(pair["manual"] and not pair["shared"] for pair in known_pairs),
        "both_failed": sum(not pair["shared"] and not pair["manual"] for pair in known_pairs),
    }}


def run(output: Path, start_seed: int, count: int, *, retain_disabled: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    sources_before = source_snapshot()
    development = default_v7_scene_config(0)
    task = calibrated_task_spec(TARGET, slot_id=development.registry.by_id(TARGET).dock_slot)
    frozen_task = asdict(task)
    protocol = {
        "version": "v7_marker_robustness_r2", "target_id": TARGET,
        "calibration_seed": 0, "evaluation_seeds": list(range(start_seed, start_seed + count)),
        "modes": ["shared", "manual"], "frozen_task_spec": frozen_task,
        "mode_interpretation": {
            "shared": ("fixed hand command with retention disabled" if retain_disabled else "tactile retention")
                      + " plus shared overload/collision/inhibition safety",
            "manual": "fixed hand command; retention and shared safety stops/inhibition disabled",
            "comparison_scope": "task contact guards remain active in both modes; no human teleoperation claim",
        },
        "retention_protocol": {"retain_disabled": retain_disabled,
                               "shared_retention": not retain_disabled,
                               "manual_retention": not retain_disabled},
        "perturbations": {"xy_error_each_axis_m": [-.005, .005], "mass_factor": [.8, 1.2],
                          "friction_factor": [.8, 1.2], "friction_rule": "one factor multiplies all three registry coefficients"},
        "map_protocol": "controller receives same-seed nominal map; model receives perturbed marker registry",
        "construction_policy": "perturb before XML build; never write live object qpos",
        "preserved_clutter": "all three targets and all three distractors; all obstacles and slots",
        "release_criteria": {"linear_speed_m_s": RELEASE_LINEAR_SPEED, "angular_speed_rad_s": RELEASE_ANGULAR_SPEED,
                             "maximum_target_hand_force_n": RELEASE_TARGET_FORCE,
                             "minimum_stable_seconds": task.release_verify_seconds,
                             "dock_tolerance_m": task.dock_tolerance_m, "maximum_finger_closure": .03},
        "success_scope": "known upright primitive marker under specified small errors; no arbitrary-tool or learned-vision claim",
        "source_hashes_before": sources_before, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "mujoco": mujoco.__version__, "physics": "native MuJoCo CPU serial"},
    }
    atomic_json(output / "protocol.json", protocol)
    records = []
    for seed in range(start_seed, start_seed + count):
        nominal, actual, perturbation = paired_configs(seed)
        pair_directory = output / f"seed_{seed}"
        pair_directory.mkdir()
        scene = build_v7_workbench_scene(pair_directory / "scene.xml", config=actual)
        scene_hash = sha256(scene)
        atomic_json(pair_directory / "nominal_manifest.json", nominal.manifest())
        atomic_json(pair_directory / "model_manifest.json", actual.manifest())
        for mode in ("shared", "manual"):
            started = time.perf_counter()
            directory = pair_directory / mode
            directory.mkdir()
            metadata = {"seed": seed, "mode": mode, "target_id": TARGET,
                        "shared_control": mode == "shared", "scene_sha256": scene_hash,
                        "frozen_task_spec": frozen_task, "perturbation": perturbation,
                        "nominal_target": nominal.registry.by_id(TARGET).manifest(),
                        "actual_target": actual.registry.by_id(TARGET).manifest(),
                        "source_hashes_before": sources_before, "evaluation_only_state": EVIDENCE_KEY}
            try:
                model = mujoco.MjModel.from_xml_path(str(scene))
                session = WorkbenchSession(model, nominal, shared_control=mode == "shared",
                                           tactile_retention=False if retain_disabled else None)
                metadata["tactile_retention"] = session.tactile_retention
                expected_retention = not retain_disabled
                if session.tactile_retention != expected_retention:
                    raise RuntimeError("session retention behavior does not match frozen r2 protocol")
                executor = V7TaskExecutor(session, step_callback=EvaluationInstrumentation(session))
                result = executor.run(task)
                measured = episode_metrics(result, task, nominal)
                artifacts = {
                    "trace": write_gzip(directory / "trace.json.gz", result.trace),
                    "actions": write_gzip(directory / "action_log.json.gz", result.metrics["action_log"]),
                    "events": write_gzip(directory / "events.json.gz", result.metrics["events"]),
                }
                compact = result.as_dict()
                compact.pop("trace", None)
                compact["metrics"] = {key: value for key, value in compact["metrics"].items()
                                      if key not in {"action_log", "events"}}
                atomic_json(directory / "task_result.json", compact)
                marker_body = model.body(f"v7_obj_{TARGET}").id
                marker_geoms = set(np.flatnonzero(model.geom_bodyid == marker_body).tolist())
                metadata["model_marker_mass_kg"] = float(model.body_mass[marker_body])
                metadata["model_marker_geom_friction"] = {
                    model.geom(gid).name: model.geom_friction[gid].tolist() for gid in marker_geoms}
                metadata["model_marker_contact_pairs"] = [
                    {"geom1": model.geom(int(model.pair_geom1[index])).name,
                     "geom2": model.geom(int(model.pair_geom2[index])).name,
                     "friction": model.pair_friction[index].tolist()}
                    for index in range(model.npair)
                    if int(model.pair_geom1[index]) in marker_geoms or int(model.pair_geom2[index]) in marker_geoms]
                metadata["physics_timestep_s"] = float(model.opt.timestep)
            except Exception as error:
                measured = {name: None for name in ("executor_success", "strict_physical_success", "target_held_lift",
                                                    "collision_free", "force_safe", "stable_released_in_slot")}
                measured.update(failure_reason=f"RUNTIME_ERROR:{type(error).__name__}",
                                unknown_reasons=[repr(error)], peak_hand_channel_force_n=None)
                artifacts = {}
                metadata["runtime_error"] = repr(error)
            metadata["wall_seconds"] = time.perf_counter() - started
            metadata["artifacts"] = artifacts
            atomic_json(directory / "metadata.json", metadata)
            record = {"seed": seed, "mode": mode, "path": str(directory.relative_to(output)),
                      "metrics": measured, "wall_seconds": metadata["wall_seconds"], "artifacts": artifacts}
            records.append(record)
            atomic_json(output / "summary.json", {"protocol": protocol, "results": records, **summary(records), "complete": False})
            print(json.dumps({"seed": seed, "mode": mode, "success": measured["strict_physical_success"],
                              "executor_success": measured["executor_success"], "failure": measured["failure_reason"],
                              "seconds": round(metadata["wall_seconds"], 2)}), flush=True)
    sources_after = source_snapshot()
    protocol["source_hashes_after"] = sources_after
    protocol["source_changed_during_run"] = sources_after != sources_before
    protocol["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "summary.json", {"protocol": protocol, "results": records, **summary(records), "complete": True})
    print(json.dumps({"complete": True, "source_changed": protocol["source_changed_during_run"], **summary(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=200)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--retain-disabled", action="store_true",
                        help="Disable hand retention in both modes; contact-based task phase guards remain active")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.start_seed <= 0 < args.start_seed + args.count:
        parser.error("evaluation range must exclude calibration seed 0")
    run(args.output.resolve(), args.start_seed, args.count, retain_disabled=args.retain_disabled)


if __name__ == "__main__":
    main()
