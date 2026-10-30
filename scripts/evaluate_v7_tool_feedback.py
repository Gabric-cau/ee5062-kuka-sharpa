#!/usr/bin/env python3
"""Small paired contact-feedback study for three tools in fresh six-body fixtures.

The task presets match run_v7_tool_sorting.py. Each episode constructs a new
scene; this experiment is separate from continuous sorting and RGB-D studies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_v7_robustness import atomic_json, rate, sha256, write_gzip
from run_v7_fixture_task import FixtureExecutor
from kuka_sharpa_grasp.v7_calibrate import CalibrationContactReader, EVIDENCE_KEY, score_lift_evidence
from kuka_sharpa_grasp.v7_fixture import upright_tool_fixture
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene
from kuka_sharpa_grasp.v7_session import WorkbenchSession
from kuka_sharpa_grasp.v7_task import TaskSpec, V7TaskExecutor, calibrated_task_spec


TOOLS = ("tool_marker", "tool_screwdriver", "tool_wrench")
MODES = ("shared", "no_tactile")
METRICS = ("strict_physical_success", "executor_success", "target_held_lift",
           "collision_free", "force_safe", "stable_released_in_slot")


def source_snapshot():
    paths = sorted((PROJECT / "src").rglob("*.py")) + [Path(__file__).resolve(),
        PROJECT / "scripts" / "run_v7_fixture_task.py",
        PROJECT / "scripts" / "run_v7_tool_sorting.py",
        PROJECT / "scripts" / "evaluate_v7_robustness.py"]
    return {path.relative_to(PROJECT).as_posix(): sha256(path) for path in paths}


def task_spec_for(object_id):
    if object_id == "tool_marker":
        return replace(calibrated_task_spec(object_id, slot_id="slot_marker"),
                       release_verify_seconds=3., phase_timeout_s=12., dock_tolerance_m=.025)
    if object_id not in TOOLS:
        raise ValueError(f"Unknown tool: {object_id}")
    wrench = object_id == "tool_wrench"
    return TaskSpec(object_id, slot_id="slot_wrench" if wrench else "slot_screwdriver",
                    grasp_offset_m=(-.065, .037, .233 if wrench else .210),
                    approach_axis_world=(0., 0., -1.), grasp_closure=.75 if wrench else .65,
                    lift_height_m=.10, phase_timeout_s=12., hold_seconds=.1,
                    tactile_force_min_n=.15, release_verify_seconds=3., dock_tolerance_m=.025)


class Instrumentation:
    """Record native contacts and object velocities without sending commands."""

    def __init__(self, session, object_id):
        self.session = session
        self.contacts = CalibrationContactReader(session, object_id)
        self.body_id = session.model.body(f"v7_obj_{object_id}").id
        self.samples = []

    def __call__(self, data, snapshot):
        evidence = self.contacts.read()
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.session.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.body_id, velocity, 0)
        evidence.update(target_angular_velocity_world_rad_s=velocity[:3].tolist(),
                        target_linear_velocity_world_m_s=velocity[3:].tolist(),
                        target_angular_speed_rad_s=float(np.linalg.norm(velocity[:3])),
                        target_linear_speed_m_s=float(np.linalg.norm(velocity[3:])))
        snapshot[EVIDENCE_KEY] = evidence
        self.samples.append(snapshot)


def release_evidence(result, task, config):
    phases = {phase.name: phase for phase in result.phases}
    release, verify = phases.get("RELEASE"), phases.get("VERIFY")
    if release is None or verify is None:
        return {"stable_released_in_slot": False, "stable_seconds": 0.,
                "reason": "RELEASE_OR_VERIFY_NOT_REACHED"}
    rows = [row for row in result.trace
            if verify.start_time_s < row["time_s"] <= verify.end_time_s + 1e-9]
    fields = ("target_linear_speed_m_s", "target_angular_speed_rad_s", "target_hand_force_n")
    if not rows or not all(EVIDENCE_KEY in row and all(key in row[EVIDENCE_KEY] for key in fields)
                           for row in rows):
        return {"stable_released_in_slot": None, "stable_seconds": None,
                "reason": "RELEASE_EVIDENCE_INCOMPLETE"}
    slot = next(slot for slot in config.slots if slot.slot_id == task.slot_id)
    stable_start, previous_time = None, None
    best_seconds = 0.
    for row in rows:
        timestamp = float(row["time_s"])
        evidence = row[EVIDENCE_KEY]
        position = np.asarray(row["evaluation"][task.object_id]["position"])
        valid = (
            np.linalg.norm(position - np.asarray(slot.pose)) <= task.dock_tolerance_m
            and evidence["target_linear_speed_m_s"] < .02
            and evidence["target_angular_speed_rad_s"] < .25
            and evidence["target_hand_force_n"] <= .05
            and max(row["finger_closure"].values()) <= .03
            and (previous_time is None or timestamp - previous_time <= .02 + 1e-9))
        if valid:
            if stable_start is None:
                stable_start = timestamp
            best_seconds = max(best_seconds, timestamp - stable_start)
        else:
            stable_start = None
        previous_time = timestamp
    ending_seconds = 0. if stable_start is None else rows[-1]["time_s"] - stable_start
    accepted = (release.status == verify.status == "COMPLETE"
                and ending_seconds >= task.release_verify_seconds - 1e-9)
    final = rows[-1][EVIDENCE_KEY]
    return {"stable_released_in_slot": bool(accepted), "stable_seconds": ending_seconds,
            "maximum_stable_seconds": best_seconds,
            "reason": None if accepted else "NO_STABLE_RELEASE_WINDOW",
            "final_linear_speed_m_s": final["target_linear_speed_m_s"],
            "final_angular_speed_rad_s": final["target_angular_speed_rad_s"],
            "final_target_hand_force_n": final["target_hand_force_n"]}


def episode_metrics(result, task, config):
    rows = result.trace[1:]
    complete = bool(rows) and all(EVIDENCE_KEY in row and all(
        key in row[EVIDENCE_KEY] for key in ("environment_collisions", "maximum_hand_channel_force_n"))
        for row in rows)
    lift = score_lift_evidence(result, task.object_id)
    release = release_evidence(result, task, config)
    peaks = [row[EVIDENCE_KEY]["maximum_hand_channel_force_n"]
             for row in rows if EVIDENCE_KEY in row]
    collisions = sum(bool(row[EVIDENCE_KEY]["environment_collisions"])
                     for row in rows if EVIDENCE_KEY in row)
    held = lift.accepted if complete else None
    collision_free = collisions == 0 if complete else None
    force_safe = max(peaks) <= 12. if complete else None
    parts = (bool(result.success), held, collision_free, force_safe,
             release["stable_released_in_slot"])
    strict = False if False in parts else None if None in parts else True
    reasons = [] if result.failure_reason is None else [result.failure_reason]
    reasons.extend(reason for passed, reason in (
        (held, "TARGET_HELD_LIFT_NOT_ACCEPTED"),
        (collision_free, "ENVIRONMENT_COLLISION"),
        (force_safe, "CHANNEL_FORCE_LIMIT_EXCEEDED"),
        (release["stable_released_in_slot"], release.get("reason") or "RELEASE_NOT_ACCEPTED"))
                   if passed is False)
    return {"executor_success": bool(result.success), "strict_physical_success": strict,
            "target_held_lift": held, "collision_free": collision_free,
            "force_safe": force_safe, "stable_released_in_slot": release["stable_released_in_slot"],
            "failure_reason": reasons[0] if reasons else None, "failure_reasons": reasons,
            "unknown_reasons": [] if complete else ["CONTACT_EVIDENCE_INCOMPLETE"],
            "peak_hand_channel_force_n": max(peaks) if peaks else None,
            "peak_total_tactile_force_n": result.metrics.get("peak_tactile_force_n"),
            "collision_sample_count": collisions if complete else None,
            "evidence_samples": len(peaks), "expected_evidence_samples": len(rows),
            "dock_error_m": result.metrics.get("dock_error_m"),
            "lift_evidence": lift.as_dict(), "release_evidence": release}


def full_rate(values):
    values = list(values)
    result = rate(values)
    return {**result, "success_rate_all_attempts": result["numerator"] / len(values) if values else None}


def summarize(records, tools=TOOLS, modes=MODES):
    groups = {}
    for object_id in tools:
        groups[object_id] = {}
        for mode in modes:
            selected = [row for row in records if row["object_id"] == object_id and row["mode"] == mode]
            groups[object_id][mode] = {
                "episodes": len(selected),
                "rates": {key: full_rate(row["metrics"].get(key) for row in selected) for key in METRICS},
                "failure_reasons": dict(Counter(
                    row["metrics"].get("failure_reason")
                    or ("SUCCESS" if row["metrics"].get("strict_physical_success") is True else "UNKNOWN")
                    for row in selected)),
                "peak_hand_channel_force_n": max((row["metrics"]["peak_hand_channel_force_n"]
                    for row in selected if row["metrics"].get("peak_hand_channel_force_n") is not None), default=None)}
    pairs = {}
    for object_id in tools:
        grouped = {}
        for row in records:
            if row["object_id"] == object_id:
                grouped.setdefault(row["seed"], {})[row["mode"]] = row["metrics"]["strict_physical_success"]
        complete = [pair for pair in grouped.values() if set(pair) == set(MODES)]
        known = [pair for pair in complete if None not in pair.values()]
        pairs[object_id] = {"complete_pairs": len(complete), "known_pairs": len(known),
            "unknown_pairs": len(complete) - len(known),
            "both_success": sum(pair["shared"] and pair["no_tactile"] for pair in known),
            "shared_only": sum(pair["shared"] and not pair["no_tactile"] for pair in known),
            "no_tactile_only": sum(pair["no_tactile"] and not pair["shared"] for pair in known),
            "both_failed": sum(not pair["shared"] and not pair["no_tactile"] for pair in known)}
    return {"tools": groups, "paired_comparison": pairs}


def run(output, start_seed=40, count=5, tools=TOOLS, modes=MODES):
    output.mkdir(parents=True, exist_ok=False)
    before = source_snapshot()
    frozen = {object_id: task_spec_for(object_id) for object_id in tools}
    protocol = {
        "version": "v7_three_tool_feedback_paired_r1", "seeds": list(range(start_seed, start_seed + count)),
        "tools": list(tools), "modes": list(modes), "calibration_seed": 0,
        "frozen_task_specs": {object_id: asdict(spec) for object_id, spec in frozen.items()},
        "force_limit_per_channel_n": 12.,
        "release_criteria": {"linear_speed_m_s": .02, "angular_speed_rad_s": .25,
                             "target_hand_force_n": .05, "maximum_finger_closure": .03,
                             "maximum_evidence_gap_s": .02},
        "collision_criteria": "Native CalibrationContactReader: non-target robot/environment penetration >2 mm and normal force >0.5 N",
        "scene_policy": "fresh full six-object upright fixture for every episode; same seed scene in each pair",
        "map_policy": "declared same-seed map; no RGB-D or additional map/mass/friction perturbation",
        "no_tactile_mode": "shared_control=False, tactile_retention=False; tactile phase guards zero; all other task and fixture parameters unchanged",
        "evaluation_policy": "native contact/velocity instrumentation does not send commands; task supervisory lift/dock body-pose guards are identical in both modes; localization uses the declared map; all failures and unknown attempts retained",
        "claim_scope": "small five-seed known-tool fixture comparison; not generalization, human teleoperation or continuous sorting",
        "source_hashes_before": before, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "mujoco": mujoco.__version__},
    }
    shutil.copy2(Path(__file__).resolve(), output / "frozen_evaluate_v7_tool_feedback.py")
    atomic_json(output / "protocol.json", protocol)
    records, scene_hashes = [], {}
    for seed in protocol["seeds"]:
        for object_id in tools:
            for mode in modes:
                directory = output / f"seed_{seed}" / object_id / mode
                directory.mkdir(parents=True)
                started = time.perf_counter()
                session = executor = instrument = None
                result = None
                artifacts = {}
                metadata = {"seed": seed, "object_id": object_id, "mode": mode}
                try:
                    config = upright_tool_fixture(seed)
                    if len(config.registry.objects) != 6 or len(config.registry.distractors) != 3:
                        raise ValueError("Full six-object fixture required")
                    scene = build_v7_workbench_scene(directory / "scene.xml", config=config)
                    atomic_json(directory / "scene_manifest.json", config.manifest())
                    scene_hash = sha256(scene)
                    key = (seed, object_id)
                    if scene_hashes.setdefault(key, scene_hash) != scene_hash:
                        raise ValueError("Paired modes must compile identical scenes")
                    blind = mode == "no_tactile"
                    session = WorkbenchSession(mujoco.MjModel.from_xml_path(str(scene)), config,
                                               shared_control=not blind, tactile_retention=not blind)
                    session.hand.config = replace(session.hand.config,
                        retention_minimum_force_n=1. if object_id == "tool_marker" else 3.)
                    task = replace(frozen[object_id], tactile_force_min_n=0., tactile_contacts_min=0) if blind else frozen[object_id]
                    instrument = Instrumentation(session, object_id)
                    if object_id == "tool_marker":
                        executor = V7TaskExecutor(session, reset_each_episode=False, step_callback=instrument)
                    else:
                        wrench = object_id == "tool_wrench"
                        executor = FixtureExecutor(session, dock_xy=(-.030, .010) if wrench else (.010, -.012),
                                                   dock_clearance=.006 if wrench else .014)
                        executor.reset_each_episode = False
                        executor.step_callback = instrument
                    result = executor.run(task)
                    measured = episode_metrics(result, frozen[object_id], config)
                    compact = result.as_dict()
                    compact.pop("trace")
                    compact["metrics"] = {key: value for key, value in compact["metrics"].items()
                                          if key not in {"events", "action_log"}}
                    atomic_json(directory / "task_result.json", compact)
                    metadata.update(scene_sha256=scene_hash, task_spec=asdict(task),
                                    shared_control=session.shared_control, tactile_retention=session.tactile_retention,
                                    timestep_s=float(session.model.opt.timestep), live_object_qpos_writes=0)
                except Exception as error:
                    measured = {key: None for key in METRICS}
                    measured.update(failure_reason=f"RUNTIME_ERROR:{type(error).__name__}",
                                    unknown_reasons=[repr(error)], peak_hand_channel_force_n=None)
                    metadata["runtime_error"] = repr(error)
                trace = result.trace if result is not None else ([] if instrument is None else instrument.samples)
                actions = [] if executor is None else executor.actions
                events = [] if session is None else session.events
                for name, data in (("trace", trace), ("actions", actions), ("events", events)):
                    artifacts[name] = write_gzip(directory / f"{name}.json.gz", data)
                metadata.update(artifacts=artifacts, wall_seconds=time.perf_counter() - started)
                atomic_json(directory / "metadata.json", metadata)
                record = {"seed": seed, "object_id": object_id, "mode": mode, "metrics": measured,
                          "artifacts": artifacts, "path": directory.relative_to(output).as_posix()}
                records.append(record)
                atomic_json(output / "summary.json", {"protocol": protocol, "results": records,
                    **summarize(records, tools, modes), "complete": False})
                print(f"seed={seed} object={object_id} mode={mode} strict={measured['strict_physical_success']} "
                      f"failure={measured['failure_reason']}", flush=True)
    after = source_snapshot()
    protocol.update(source_hashes_after=after, source_changed_during_run=before != after,
                    ended_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "protocol.json", protocol)
    summary = {"protocol": protocol, "results": records, **summarize(records, tools, modes), "complete": True}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=40)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--tools", nargs="+", choices=TOOLS, default=TOOLS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if args.count < 1 or args.start_seed < 0:
        parser.error("count must be positive and seed non-negative")
    if len(set(args.tools)) != len(args.tools) or len(set(args.modes)) != len(args.modes):
        parser.error("tools and modes must be unique")
    summary = run(args.output.resolve(), args.start_seed, args.count, tuple(args.tools), tuple(args.modes))
    return 2 if summary["protocol"]["source_changed_during_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
