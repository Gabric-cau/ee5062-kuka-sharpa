#!/usr/bin/env python3
"""Paired, declared held-out evaluation of the same physical known-can workcell."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / "local_cache" / "paired_matplotlib"))
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kuka_sharpa_grasp.pick_place import Scenario, reset_task, run_task
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene, WORKCELL_VERSION


MODE_LABELS = {"contact": "Contact feedback", "fixed": "Uniform fixed closure",
               "fixed_calibrated": "Calibrated fixed closure"}
MODE_COLORS = {"contact": "#278c86", "fixed": "#c97a34", "fixed_calibrated": "#b47735"}


def json_write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def source_hashes():
    paths = sorted((PROJECT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    return {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if not p.name.startswith("._")}


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z / count
    center = (p + z*z / (2*count)) / denominator
    width = z * math.sqrt(p*(1-p) / count + z*z / (4*count*count)) / denominator
    return [max(0., center-width), min(1., center+width)]


def statistics(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "maximum": max(values)}


def sample_trace(trace, timestep, stride=10):
    samples, previous_step, previous_state = [], -stride, None
    for row in trace:
        step = round(float(row["timestamp"]) / timestep)
        if step >= previous_step + stride or row["state"] != previous_state:
            samples.append(row)
            previous_step, previous_state = step, row["state"]
    if trace and (not samples or samples[-1] is not trace[-1]):
        samples.append(trace[-1])
    return samples


def measure_renderer(model, data, probe_png):
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render().copy()
        plt.imsave(probe_png, pixels)
        from OpenGL import GL
        report = {name: (GL.glGetString(enum) or b"unavailable").decode(errors="replace")
                  for name, enum in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                                      ("version", GL.GL_VERSION))}
    report.update(pixel_std=float(pixels.std()), nonzero_pixels=int(np.any(pixels != 0, axis=2).sum()),
                  scope="This process's offscreen MuJoCo OpenGL context; not GPU physics.")
    report["nvidia_rtx2060_verified"] = "NVIDIA" in report["vendor"].upper() and "2060" in report["renderer"]
    report["nonblank_image_verified"] = report["pixel_std"] > 1. and report["nonzero_pixels"] > 0
    report["probe_initialization"] = "reset_task with the first declared scenario before rendering; image retained as renderer_probe.png."
    return report


def csv_row(seed, mode, result, trace, error, wall_seconds):
    completed = result is not None
    final = result.get("final_oracle", {}) if completed else {}
    strict = bool(completed and result.get("success") and not result.get("failure") and final.get("success"))
    attempts = result.get("attempts", []) if completed else []
    first = bool(strict and len(attempts) == 1 and attempts[0].get("failure") is None)
    recovered = bool(strict and len(attempts) > 1 and any(a.get("failure") for a in attempts[:-1]))
    hand_peak = max((float(row["hand_segment_total_force"]) for row in trace), default=None)
    return {
        "seed": seed, "mode": mode, "execution_status": "completed" if completed else "runner_error",
        "strict_success": strict, "first_attempt_success": first, "recovered_success": recovered,
        "failure": (result.get("failure") or ("none" if strict else "PLACEMENT_VERIFICATION")) if completed else error["type"],
        "attempts": len(attempts) if completed else None,
        "recovery_attempts": result.get("safe_abort_attempts") if completed else None,
        "duration_s": result.get("duration") if completed else None,
        "placement_error_m": result.get("placement_error") if completed else None,
        "peak_hand_object_normal_sum_n": hand_peak,
        "peak_fingertip_normal_sum_n": result.get("maximum_fingertip_total_force") if completed else None,
        "peak_any_contact_normal_n": result.get("maximum_contact_force") if completed else None,
        "slip_adjustment_steps": result.get("slip_adjustment_steps") if completed else None,
        "cartesian_motion": result.get("cartesian_motion") if completed else None,
        "visual_transfer": result.get("visual_transfer") if completed else None,
        "support_load_sensor": result.get("support_load_sensor") if completed else None,
        "wall_seconds": wall_seconds,
    }


def summarize(rows, args):
    summaries = {}
    for mode in args.modes:
        trials = [row for row in rows if row["mode"] == mode]
        completed = [row for row in trials if row["execution_status"] == "completed"]
        successes = sum(row["strict_success"] for row in trials)
        failed = [row for row in trials if not row["strict_success"]]
        summaries[mode] = {
            "requested": args.seeds, "attempted": len(trials), "completed": len(completed),
            "pending": args.seeds-len(trials), "runner_errors": len(trials)-len(completed),
            "strict_successes": successes, "task_failures": sum(not row["strict_success"] for row in completed),
            "success_rate_over_attempted": successes / len(trials) if trials else None,
            "wilson_95_over_attempted": wilson(successes, len(trials)),
            "first_attempt_successes": sum(row["first_attempt_success"] for row in trials),
            "recovered_successes": sum(row["recovered_success"] for row in trials),
            "trials_requiring_recovery": sum((row["recovery_attempts"] or 0) > 0 for row in completed),
            "failure_counts": dict(Counter(row["failure"] for row in failed)),
            "failed_seeds": [row["seed"] for row in failed],
            "placement_error_m_all_completed": statistics(row["placement_error_m"] for row in completed),
            "placement_error_m_success_only": statistics(row["placement_error_m"] for row in completed if row["strict_success"]),
            "peak_hand_object_normal_sum_n_all_completed": statistics(row["peak_hand_object_normal_sum_n"] for row in completed),
            "peak_any_contact_normal_n_all_completed": statistics(row["peak_any_contact_normal_n"] for row in completed),
            "duration_s_all_completed": statistics(row["duration_s"] for row in completed),
        }
    by_seed = {seed: {row["mode"]: row for row in rows if row["seed"] == seed}
               for seed in range(args.seed, args.seed + args.seeds)}
    paired = [conditions for conditions in by_seed.values() if set(args.modes) <= set(conditions)]
    paired_outcomes = Counter(tuple(int(condition[mode]["strict_success"]) for mode in args.modes)
                              for condition in paired)
    return {"run_complete": len(rows) == args.seeds*len(args.modes),
            "declared_heldout_seed_start": args.seed, "declared_heldout_seed_end_inclusive": args.seed+args.seeds-1,
            "condition_order": args.modes, "fully_paired_scenarios": len(paired),
            "paired_success_patterns": {"|".join(map(str, key)): value for key, value in paired_outcomes.items()},
            "pattern_note": "Bits follow condition_order; these are matched new-workcell trials, not the historical hand-aligned baseline.",
            "denominator_note": "Rate/Wilson denominator is every attempted scenario, including runner errors; pending scenarios are reported and excluded. A complete study has all requested trials attempted.",
            "mode_summary": summaries}


def make_plots(folder, rows, summary, args):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    modes = args.modes
    fig, axis = plt.subplots(figsize=(8, 5), dpi=140)
    for index, mode in enumerate(modes):
        stats = summary["mode_summary"][mode]
        rate, interval = stats["success_rate_over_attempted"], stats["wilson_95_over_attempted"]
        if rate is None:
            continue
        axis.bar(index, rate, color=MODE_COLORS[mode], width=.55)
        axis.errorbar(index, rate, yerr=np.array([[rate-interval[0]], [interval[1]-rate]]),
                      fmt="none", color="#30363c", capsize=6)
        axis.text(index, min(1.11, interval[1]+.035),
                  f"{stats['strict_successes']}/{stats['attempted']}\nfirst try {stats['first_attempt_successes']}; recovered {stats['recovered_successes']}",
                  ha="center", fontsize=9)
    axis.set_xticks(range(len(modes)), [MODE_LABELS[mode] for mode in modes])
    axis.set_ylim(0, 1.22)
    axis.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    axis.set_ylabel("Strict full-task success rate")
    axis.set_title("Matched held-out pick-and-place scenarios | Wilson 95% intervals")
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(folder / "success_rate.png")
    fig.savefig(folder / "success_rate.pdf")
    plt.close(fig)

    reasons = sorted({row["failure"] for row in rows if not row["strict_success"]})
    codes = {reason: f"F{index+1}" for index, reason in enumerate(reasons)}
    fig = plt.figure(figsize=(12, 4.1*len(modes)+1.6), dpi=140)
    grid = fig.add_gridspec(len(modes)+1, 1, height_ratios=[1]*len(modes)+[.35], hspace=.4)
    for index, mode in enumerate(modes):
        axis = fig.add_subplot(grid[index])
        trials = [row for row in rows if row["mode"] == mode and row["placement_error_m"] is not None]
        for success in (True, False):
            group = [row for row in trials if row["strict_success"] == success]
            axis.scatter([row["seed"] for row in group], [100*row["placement_error_m"] for row in group],
                         marker="o" if success else "x", color=MODE_COLORS[mode] if success else "#b84040",
                         label="Success" if success else "Failed task", s=45)
        for row in trials:
            if not row["strict_success"]:
                axis.annotate(codes[row["failure"]], (row["seed"], 100*row["placement_error_m"]),
                              xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color="#a23333")
        axis.set_xlim(args.seed-.6, args.seed+args.seeds-.4)
        axis.set_xticks(range(args.seed, args.seed+args.seeds, max(1, args.seeds//10)))
        axis.set_xlabel("Held-out scenario seed")
        axis.set_ylabel("Final XY placement error (cm)")
        axis.set_title(MODE_LABELS[mode])
        axis.grid(alpha=.2)
        axis.legend(loc="upper right")
        axis.margins(y=.25)
    labels = fig.add_subplot(grid[-1])
    labels.set_axis_off()
    legend = [f"{codes[reason]}: {reason}" for reason in reasons]
    missing = [f"{row['mode']} seed {row['seed']}" for row in rows if row["placement_error_m"] is None]
    legend.append("Task failure can include release, retreat, settling or visual checks despite a small XY error.")
    if missing:
        legend.append("No measurement: " + ", ".join(missing))
    labels.text(0, 1, "\n".join(legend), transform=labels.transAxes, va="top", fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, top=.95, bottom=.045)
    fig.savefig(folder / "placement_error.png")
    fig.savefig(folder / "placement_error.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    metrics = [("peak_hand_object_normal_sum_n", "Peak hand-object normal force sum (N)"),
               ("peak_any_contact_normal_n", "Peak individual contact normal force (N)")]
    for axis, (metric, title) in zip(axes, metrics):
        groups = [[row[metric] for row in rows if row["mode"] == mode and row[metric] is not None] for mode in modes]
        nonempty = [(index+1, values) for index, values in enumerate(groups) if values]
        if nonempty:
            boxes = axis.boxplot([values for _, values in nonempty], positions=[index for index, _ in nonempty],
                                 widths=.45, patch_artist=True, showfliers=False)
            for patch, (index, values) in zip(boxes["boxes"], nonempty):
                patch.set_facecolor(MODE_COLORS[modes[index-1]])
                patch.set_alpha(.28)
                offsets = np.linspace(-.10, .10, len(values))
                axis.scatter(index+offsets, values, s=22, color=MODE_COLORS[modes[index-1]], alpha=.85)
        by_seed = {seed: {row["mode"]: row for row in rows if row["seed"] == seed}
                   for seed in range(args.seed, args.seed+args.seeds)}
        for pair in by_seed.values():
            if all(mode in pair and pair[mode][metric] is not None for mode in modes):
                axis.plot(range(1, len(modes)+1), [pair[mode][metric] for mode in modes],
                          color="#66717a", alpha=.15, linewidth=.7, zorder=0)
        axis.set_xticks(range(1, len(modes)+1), [MODE_LABELS[mode] for mode in modes])
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_ylim(bottom=0)
    axes[0].set_title("All finger segments and palm against the target")
    axes[1].set_title("All modeled contact pairs, including table/robot")
    fig.suptitle("Actual peak forces across all completed trials, including failures")
    fig.tight_layout()
    fig.savefig(folder / "force_peak.png")
    fig.savefig(folder / "force_peak.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--modes", choices=("contact", "fixed", "fixed_calibrated"), nargs="+", default=["contact", "fixed"])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.seeds < 1 or args.max_attempts < 1 or len(set(args.modes)) != len(args.modes):
        parser.error("Nonnegative seed, positive seeds/attempts, and distinct modes are required")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve earlier evidence")
    if "force_mode" not in inspect.signature(run_task).parameters:
        parser.error("Installed run_task does not have the required force_mode API")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial_hashes = source_hashes()
    scenarios = [Scenario(seed) for seed in range(args.seed, args.seed+args.seeds)]
    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__,
        "numpy": np.__version__, "physics_backend": "classic_mujoco_cpu", "workcell_version": WORKCELL_VERSION,
        "modes": args.modes, "maximum_attempts": args.max_attempts,
        "declared_heldout_seed_range": [args.seed, args.seed+args.seeds-1],
        "evaluation_role": "Declared held-out evaluation. Do not tune controllers on these results and relabel a rerun as unseen evaluation.",
        "scope": "One known upright YCB 005 tomato soup can. Scenario samples tabletop x/y. The perception uses the simulation's authored appearance prior and partial visible surfaces; this is not arbitrary-object or real-camera validation.",
        "scenario_manifest": [{"seed": s.seed, "object_kind": s.object_kind, "initial_position": list(s.position), "zone": list(s.zone)} for s in scenarios],
        "condition_order": "Alternates forward/reverse mode order by scenario ordinal; each mode uses a fresh model/data from the same workcell XML.",
        "comparison": "Paired new-workcell full tasks. The historical hand-aligned lift baseline is not part of this experiment.",
        "trace_sampling": "Every 10 physical steps, plus state changes and the final record. Force maxima use the full unsampled trace/result.",
        "source_sha256_at_start": initial_hashes,
        "default_optional_flags": {name: inspect.signature(run_task).parameters[name].default for name in ("cartesian_motion", "visual_transfer", "support_load_sensor") if name in inspect.signature(run_task).parameters},
        "fixed_calibrated_development_provenance": ("Frozen per-finger terminal CLOSE medians from development seeds 0..9; no contact retention. Held-out results are not used to fit these targets." if "fixed_calibrated" in args.modes else None),
    }
    json_write(args.output / "run_configuration.json", configuration)
    scene = build_pick_place_scene(args.output / "scene.xml", "ycb_can")
    probe_model = mujoco.MjModel.from_xml_path(str(scene))
    probe_data = mujoco.MjData(probe_model)
    reset_task(probe_model, probe_data, scenarios[0])
    opengl = measure_renderer(probe_model, probe_data, args.output / "renderer_probe.png")
    print(json.dumps({"renderer": opengl, "requested_trials": args.seeds*len(args.modes)}, indent=2), flush=True)
    rows = []
    controller_profiles = {}
    profile_variation = {}
    interrupted = None
    try:
        for ordinal, scenario in enumerate(scenarios):
            mode_order = args.modes if ordinal % 2 == 0 else list(reversed(args.modes))
            for mode in mode_order:
                folder = args.output / f"seed_{scenario.seed:04d}" / mode
                folder.mkdir(parents=True, exist_ok=False)
                model = mujoco.MjModel.from_xml_path(str(scene))
                data = mujoco.MjData(model)
                trial_start = time.monotonic()
                result, trace, error = None, [], None
                try:
                    result, trace = run_task(model, data, scenario, force_mode=mode, max_attempts=args.max_attempts)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                wall = time.monotonic()-trial_start
                if result is not None and result.get("hand_control") is not None:
                    profile = result["hand_control"]
                    if mode not in controller_profiles:
                        controller_profiles[mode] = profile
                    elif controller_profiles[mode] != profile:
                        profile_variation[mode] = True
                json_write(folder / "result.json", result)
                json_write(folder / "trace.json", sample_trace(trace, float(model.opt.timestep)))
                json_write(folder / "trial_metadata.json", {
                    "seed": scenario.seed, "mode": mode, "max_attempts": args.max_attempts,
                    "execution_status": "completed" if result is not None else "runner_error", "error": error,
                    "physics_timestep_s": float(model.opt.timestep), "full_trace_rows": len(trace),
                    "wall_seconds": wall, "scope": "Known upright can, randomized tabletop x/y.",
                    "actual_hand_control": result.get("hand_control") if result is not None else None,
                })
                row = csv_row(scenario.seed, mode, result, trace, error, wall)
                rows.append(row)
                print(f"[{len(rows)}/{args.seeds*len(args.modes)}] seed={scenario.seed} mode={mode} success={row['strict_success']} first={row['first_attempt_success']} recovered={row['recovered_success']} failure={row['failure']} duration={row['duration_s']} wall={wall:.2f}s", flush=True)
    except BaseException as exc:
        interrupted = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        summary = summarize(rows, args)
        with (args.output / "trials.csv").open("x", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        final_hashes = source_hashes()
        json_write(args.output / "metadata.json", {
            **configuration, "opengl": opengl, "wall_seconds": time.monotonic()-started,
            "run_complete": summary["run_complete"], "interrupted": interrupted,
            "source_sha256_at_end": final_hashes, "source_files_changed_during_run": initial_hashes != final_hashes,
            "actual_controller_profiles_from_trial_results": controller_profiles,
            "controller_profile_variation_observed": profile_variation,
        })
        json_write(args.output / "summary.json", summary)
        if rows:
            make_plots(args.output, rows, summary, args)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
