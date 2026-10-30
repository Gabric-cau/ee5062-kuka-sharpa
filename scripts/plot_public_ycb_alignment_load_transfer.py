#!/usr/bin/env python3
"""Offline, finalized R20/R22 wrist motion and vertical load-transfer diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

import analyze_public_ycb_vertical_load as load
from compare_public_ycb_close_deadline import digest, json_array, safe_path, tick
from public_ycb_evidence import load_evidence_model


PROJECT = Path(__file__).resolve().parents[1]
STEP = .002
PHASES = ("APPROACH", "ALIGN", "CLOSE", "TRIAL_LIFT")
FORMATS = {"baseline": "public_ycb_shared_normal_development_r20",
    "candidate": "public_ycb_settled_alignment_development_r22"}
CLASSES = {"baseline": "SharedNormalBudgetHand", "candidate": "ProvisionalPreloadBudgetHand"}
R20_SHA256 = "a47ab784ed6b0879f149333a08e02928b5ae5430ec93af76af67de02ef63dd7b"
FORCES = {"table_z_n": "table_world", "hand_normal_z_n": "hand_normal",
    "hand_tangent_z_n": "hand_tangent", "hand_net_z_n": "hand_world", "other_z_n": "other_world"}
COLORS = dict(APPROACH="#e7f0f7", ALIGN="#eaf3e6", CLOSE="#fff0db", TRIAL_LIFT="#f8e7eb")
SCOPE = (
    "Evaluation-only sampled diagnostics. World +z and positive force mean upward; hand forces act on "
    "the target, not on the hand. Normal/tangent are the contact-frame decomposition, not separate sensors. "
    "Native contacts, bases and trace wrist_position retain the last dynamics evaluation cache under the "
    "completed-transition timestamp; native target_qpos[2] is the integrated free-joint-origin height. "
    "Those states can differ by one dynamics evaluation. Arm target z is offline FK of the recorded "
    "post-limiter logical arm_command, not raw motor control or the final planned grasp pose. "
    "No force/acceleration residual, exact slip, clearance, causal attribution or matched physical prefix "
    "is claimed. Stage-relative curves retain each run's own duration; retries have different histories. "
    "Absent phases stay absent. Scene identities and target state must not enter the controller.")


def checked_sha(value):
    if not re.fullmatch("[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("Expected an explicit lowercase finalized SHA256")
    return value


def capture(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def captured_json(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def unchanged(bindings):
    if any(capture(Path(bound["path"])) != bound for bound in bindings.values()):
        raise ValueError("A bound diagnostic input changed during analysis")


def case_path(directory, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid recorded case directory")
    path = directory
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink evidence is not accepted")
    if not path.is_dir():
        raise ValueError("Missing recorded case directory")
    return path


def receipts(directory, summary, inputs, row):
    native, inputs["native_receipt"] = captured_json(safe_path(directory, "verification_receipt.json"))
    strict, inputs["strict_receipt"] = captured_json(safe_path(directory, "strict_native_verification.json"))
    if (any(value["summary_sha256"] != inputs["summary"]["sha256"] for value in (native, strict))
            or native["format"] != "public_ycb_development_native_verification_r1"
            or native["complete"] is not True or native["unknowns"] != 0
            or native["source_hashes_before_equal_after"] is not True
            or native["native_samples_contiguous_2ms"] is not True or native["audited_episodes"] != 1
            or len(native["rows"]) != 1 or strict["format"] != "public_ycb_strict_native_audit_20261021"
            or strict["accepted_evidence"] is not True or strict["unknowns"] != 0
            or strict["denominator"] != 1 or len(strict["rows"]) != 1):
        raise ValueError("Final native/strict receipt is absent, incomplete or binds a different summary")
    n, s = native["rows"][0], strict["rows"][0]
    count = row["native_audit"]["native_samples"]
    if (n["path"] != row["path"] or s["path"] != row["path"] or n["unknown"] is not False
            or s["unknown"] is not False or n["outcome"] != row["success"] or n["held_lift"] != row["held_lift"]
            or s["checks"]["legacy_task_success"] != row["success"] or s["checks"]["held_lift"] != row["held_lift"]
            or n["native_samples"] != count or s["native_samples"] != count or native["native_samples"] != count
            or s["result_sha256"] != inputs["result"]["sha256"] or s["scene_sha256"] != inputs["scene"]["sha256"]
            or native["frozen_source_sha256"] != inputs["source_archive"]["sha256"]):
        raise ValueError("Receipt case, outcome, native count, scene, result or source ZIP binding differs")
    return {"native_evidence_complete": True, "strict_evidence_accepted": True,
        "strict_task_success": s["strict_success"], "task_success": row["success"], "held_lift": row["held_lift"],
        "command_policy_independently_audited_here": False,
        "scope": "Accepts completed native/strict evidence, including task failure; no new command/policy audit."}


def candidate_exit(job, directory, protocol, inputs):
    status, inputs["supervisor_status"] = captured_json(safe_path(job, "status.json"))
    request, inputs["supervisor_request"] = captured_json(safe_path(job, "request.json"))
    command = request["command"]
    if command.count("--output") != 1 or command.index("--output") + 1 >= len(command):
        raise ValueError("Supervisor has no unique output argument")
    output = Path(command[command.index("--output") + 1])
    output = output if output.is_absolute() else PROJECT / output
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or type(status["returncode"]) is not int
            or status["returncode"] != 0 or output.resolve() != directory
            or command.count("scripts/develop_public_ycb_settled_alignment_r22.py") != 1
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]):
        raise ValueError("Candidate must have an observed, bound FINISHED rc0 R22 child exit")


def read_run(directory, role, expected_sha, job=None):
    directory = directory.resolve(strict=True)
    summary, summary_bound = captured_json(safe_path(directory, "summary.json"))
    protocol, protocol_bound = captured_json(safe_path(directory, "protocol.json"))
    if (summary_bound["sha256"] != expected_sha or summary["complete"] is not True
            or summary["runner_errors"] != 0 or summary["protocol"] != protocol
            or protocol["format"] != FORMATS[role] or summary["runner_revision"] != FORMATS[role]
            or len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["public_meshes_unchanged"] is not True):
        raise ValueError("Expected exact finalized one-case R20/R22 summary with unchanged sources")
    row = summary["rows"][0]
    if (row["runner_error"] is not None or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or row["controller_class"] != CLASSES[role] or not 1 <= len(row["attempts"]) <= 2
            or [item["attempt"] for item in row["attempts"]] != list(range(1, len(row["attempts"]) + 1))
            or any(row[key] != value for key, value in protocol["schedule"][0].items())
            or row["native_audit"]["external_target_qpos_writes"] != 0):
        raise ValueError("Unknown/mismatched case outcome, actual class, schedule or attempt count")
    case = case_path(directory, row["path"])
    metadata, metadata_bound = captured_json(safe_path(case, "metadata.json"))
    if metadata != row:
        raise ValueError("Final case metadata differs from summary row")
    inputs = {"summary": summary_bound, "protocol": protocol_bound, "metadata": metadata_bound}
    for key, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"),
            ("result", "result.json")):
        inputs[key] = capture(safe_path(case, filename))
    inputs["source_archive"] = capture(safe_path(directory, "frozen_source.zip"))
    if (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inputs["scene"]["sha256"]
            or row["scene_sha256_at_finalize"] != inputs["scene"]["sha256"]
            or row["derived_controller_sha256"] != inputs["controller"]["sha256"]):
        raise ValueError("Original loaded scene or executed arm source binding differs")
    with zipfile.ZipFile(inputs["source_archive"]["path"]) as archive:
        before = protocol["source_hashes_before"]
        if len(archive.namelist()) != len(before) or set(archive.namelist()) != set(before):
            raise ValueError("Frozen source ZIP inventory differs")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != expected for name, expected in before.items()):
            raise ValueError("Frozen source ZIP bytes differ")
    audit = receipts(directory, summary, inputs, row)
    if role == "candidate":
        if job is None:
            raise ValueError("Final candidate supervisor directory is required")
        candidate_exit(job.resolve(strict=True), directory, protocol, inputs)
    for name in ("native", "trace"):
        artifact = row["artifacts"][name]
        inputs[name] = capture(safe_path(case, artifact["path"]))
        if any(inputs[name][key] != artifact[key] for key in ("bytes", "sha256")):
            raise ValueError("Recorded compressed artifact differs: " + name)
    for name, record in protocol["public_meshes"].items():
        inputs["public_mesh_" + name] = capture(safe_path(PROJECT, record["path"]))
        if any(inputs["public_mesh_" + name][key] != record[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen public mesh differs")
    scene = Path(inputs["scene"]["path"])
    for ordinal, mesh in enumerate(ET.parse(scene).getroot().iter("mesh")):
        if "file" in mesh.attrib:
            path = (PROJECT / "vendor/mujoco_menagerie" / mesh.attrib["file"]).resolve(strict=True)
            inputs[f"loaded_mesh_{ordinal}"] = capture(safe_path(PROJECT, path.relative_to(PROJECT)))
    model = load_evidence_model(scene)
    if abs(float(model.opt.timestep) - STEP) > 1e-12 or not np.allclose(model.opt.gravity[:2], 0., atol=1e-12):
        raise ValueError("Expected native 2ms clock and world-vertical gravity")
    mass = float(model.body_mass[model.body("target").id])
    weight = -mass * float(model.opt.gravity[2])
    if not np.isfinite(weight) or weight <= 0.:
        raise ValueError("Invalid recorded target weight")
    return {"role": role, "directory": str(directory), "case": {key: row.get(key) for key in
        ("kind", "seed", "path", "profile", "success", "held_lift", "failure")},
        "inputs": inputs, "audit_receipts": audit, "native_samples_expected": row["native_audit"]["native_samples"],
        "duration_s": row["duration_s"], "target_mass_kg": mass, "weight_n": weight,
        "attempts_recorded": row["attempts"], "model": model}


class CommandKinematics:
    """Read command FK in scratch data; never integrate or refresh recorded contacts."""

    def __init__(self, model):
        self.model, self.data, self.body = model, mujoco.MjData(model), model.body("link7").id
        self.previous, self.height = None, None

    def height_at(self, command):
        command = load.vector(command, 7, "logical arm command")
        if self.previous is None or not np.array_equal(command, self.previous):
            self.data.qpos[:7] = command
            mujoco.mj_kinematics(self.model, self.data)
            self.height = float(self.data.xpos[self.body, 2])
            self.previous = command.copy()
        return self.height


def reduce_trace(rows, command_kinematics):
    result, previous, duplicates = {}, -1, 0
    for row in rows:
        index = tick(row["timestamp"], STEP)
        if index < previous:
            raise ValueError("Trace clocks are not chronological")
        previous = index
        if index in result:
            duplicates += 1
            continue
        # record() invokes NativeAudit.sample on its first occurrence at this clock.
        wrist = load.vector(row["wrist_position"], 3, "trace cached wrist position")
        alignment = row.get("alignment_settling")
        goal = None if alignment is None else float(load.vector(
            alignment["target_wrist_position_world_m"], 3, "ALIGN goal")[2])
        result[index] = {"timestamp_s": float(row["timestamp"]), "phase": row["state"],
            "wrist_actual_z_m": float(wrist[2]), "wrist_command_fk_z_m": command_kinematics.height_at(row["arm_command"]),
            "alignment_goal_z_m": goal,
            "_hand_command": load.vector(row["hand_command"], 22, "trace hand command").tolist()}
    return result, duplicates


def exact_native_trace(native, ordinal, trace):
    index = tick(native["timestamp"], STEP)
    if index != ordinal or index not in trace or abs(trace[index]["timestamp_s"] - native["timestamp"]) > 1e-9:
        raise ValueError("Native/first-trace clocks must match exactly at every recorded 2ms tick")
    row = trace[index]
    motors = load.vector(native["motor_commands"], 29, "native motor commands")
    if native["state"] != row["phase"] or not np.array_equal(motors[7:], row["_hand_command"]):
        raise ValueError("Native phase/hand controls differ from the first trace record at the same clock")
    return index, {key: value for key, value in row.items() if key != "_hand_command"}


def vertical_forces(sample, model, target_geom, geom_names, body_names):
    record, _ = load.force_record(sample, model, target_geom, geom_names, body_names, None)
    return {name: float(record["forces_xyz_n"][key][2]) for name, key in FORCES.items()}


def sampled_statistics(values):
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return None
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic series")
    return {"n": len(array), "first": float(array[0]), "last": float(array[-1]),
        "minimum": float(array.min()), "mean": float(array.mean()), "maximum": float(array.max())}


def phase_summary(rows, weight):
    if not rows:
        return {"executed": False, "native_samples": 0, "statistics": None, "last_250ms": None}
    def statistics(selected):
        names = ("wrist_actual_z_m", "wrist_command_fk_z_m", "object_delta_z_m", *FORCES)
        return {**{name: sampled_statistics(row[name] for row in selected) for name in names},
            "table_minus_weight_n": sampled_statistics(row["table_z_n"] - weight for row in selected),
            "hand_tangent_downward_sample_fraction": sum(row["hand_tangent_z_n"] < -1e-9 for row in selected) / len(selected)}
    tail = [row for row in rows if row["timestamp_s"] > rows[-1]["timestamp_s"] - .25 + 1e-9]
    return {"executed": True, "first_native_observation_s": rows[0]["timestamp_s"],
        "last_native_observation_s": rows[-1]["timestamp_s"], "native_samples": len(rows),
        "sampled_phase_duration_s": len(rows) * STEP, "statistics": statistics(rows),
        "last_250ms": {"native_samples": len(tail), "first_native_observation_s": tail[0]["timestamp_s"],
            "last_native_observation_s": tail[-1]["timestamp_s"], "statistics": statistics(tail)}}


def extract(run):
    model = run.pop("model")
    trace, duplicate_count = reduce_trace(json_array(Path(run["inputs"]["trace"]["path"])), CommandKinematics(model))
    names = [model.geom(g).name or "" for g in range(model.ngeom)]
    bodies = [model.body(int(model.geom_bodyid[g])).name or "" for g in range(model.ngeom)]
    target_geom = model.geom("target_geom").id
    attempts, current, previous_phase, phase_counts, count = [], None, None, Counter(), 0
    for count, native in enumerate(json_array(Path(run["inputs"]["native"]["path"])), 1):
        index, row = exact_native_trace(native, count, trace)
        phase = native["state"]
        height = float(load.vector(native["target_qpos"], 7, "integrated target qpos")[2])
        if phase == "SEARCH" and previous_phase != "SEARCH":
            ordinal = len(attempts) + 1
            if ordinal > len(run["attempts_recorded"]):
                raise ValueError("More native SEARCH epochs than recorded attempts")
            current = {"attempt": ordinal, "failure": run["attempts_recorded"][ordinal - 1].get("failure"),
                "baseline_native_timestamp_s": float(native["timestamp"]), "baseline_object_z_m": height,
                "series": {name: [] for name in PHASES}}
            attempts.append(current)
        if current is None:
            raise ValueError("Native stream has no first SEARCH attempt")
        if phase in PHASES:
            series = current["series"][phase]
            if series and tick(series[-1]["timestamp_s"], STEP) + 1 != index:
                raise ValueError("Repeated disjoint phase inside one attempt is unsupported")
            series.append({**row, "object_z_m": height, "object_delta_z_m": height - current["baseline_object_z_m"],
                **vertical_forces(native, model, target_geom, names, bodies)})
        previous_phase = phase
        phase_counts[phase] += 1
    if (count != run["native_samples_expected"] or len(attempts) != len(run["attempts_recorded"])
            or not count or abs(count * STEP - run["duration_s"]) > 1e-7
            or set(trace) - set(range(1, count + 1))):
        raise ValueError("Final native duration, count, attempts or trace coverage differs")
    for attempt in attempts:
        attempt["phases"] = {name: phase_summary(attempt["series"][name], run["weight_n"]) for name in PHASES}
    run.update(native_samples_streamed=count, trace_duplicate_clock_records=duplicate_count,
        native_phase_samples=dict(phase_counts), attempts=attempts,
        object_baseline_scope="Integrated target_qpos[2] at the first recorded native SEARCH sample of this same attempt; no interpolation or extrapolation.",
        trace_join_scope="First trace record at each native clock, with exact phase/22 hand-command equality to native motor_commands[7:]. record() immediately invokes the native sample callback. Later phase-end/final same-clock trace records are excluded.")
    return run


def draw(axes, rows, weight, relative=False):
    for axis, label in zip(axes, ("Wrist world z (m)", "Object delta z (mm)", "World vertical force (N)", "Hand vertical force (N)"), strict=True):
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Seconds into this phase" if relative else "Absolute simulation time (s)")
    if not rows:
        for axis in axes:
            axis.text(.5, .5, "Not executed", ha="center", va="center", transform=axis.transAxes)
        return
    start = rows[0]["timestamp_s"] - STEP if relative else 0.
    times = [row["timestamp_s"] - start for row in rows]
    def line(axis, field, label, color, scale=1., **kwargs):
        axis.plot(times, [np.nan if row[field] is None else row[field] * scale for row in rows],
            label=label, color=color, linewidth=1.2, **kwargs)
    line(axes[0], "wrist_command_fk_z_m", "Logical arm command FK", "#9a6e00", linestyle="--")
    line(axes[0], "wrist_actual_z_m", "Recorded cached wrist", "#176ca4")
    if any(row["alignment_goal_z_m"] is not None for row in rows):
        line(axes[0], "alignment_goal_z_m", "Recorded ALIGN final goal", "#4f7d3b", linestyle=":")
    line(axes[1], "object_delta_z_m", "Integrated object origin delta z", "#a83943", scale=1000.)
    axes[1].axhline(0., color="#888888", linewidth=.7)
    line(axes[2], "table_z_n", "Table on object", "#3a8052")
    line(axes[2], "hand_net_z_n", "Hand net on object", "#176ca4")
    line(axes[2], "other_z_n", "Other contacts on object", "#8b6ba1", linestyle=":")
    axes[2].axhline(weight, color="#333333", linestyle="--", label="Object weight")
    line(axes[3], "hand_normal_z_n", "Hand normal", "#9a6e00")
    line(axes[3], "hand_tangent_z_n", "Hand tangent", "#a83943")
    line(axes[3], "hand_net_z_n", "Hand net", "#176ca4", linestyle="--")
    axes[3].axhline(0., color="#333333", linewidth=.8)
    for axis in axes:
        axis.legend(loc="best", fontsize=8)


def figures(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for run in runs:
        for attempt in run["attempts"]:
            fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
            rows = sorted((row for phase in PHASES for row in attempt["series"][phase]), key=lambda row: row["timestamp_s"])
            draw(axes, rows, run["weight_n"])
            for phase_index, phase in enumerate(PHASES):
                series = attempt["series"][phase]
                if series:
                    left, right = series[0]["timestamp_s"] - STEP, series[-1]["timestamp_s"]
                    for axis in axes:
                        axis.axvspan(left, right, color=COLORS[phase], zorder=-1)
                    axes[0].text((left + right) / 2., .96 - (phase_index % 2) * .08, phase, ha="center", fontsize=8,
                        transform=axes[0].get_xaxis_transform())
            title = f"{run['role'].upper()} {'R20' if run['role'] == 'baseline' else 'R22'} | attempt {attempt['attempt']} | {attempt['failure'] or 'no recorded failure'}"
            fig.suptitle(title + "\nCached wrist/contact forces and integrated object height can differ by one dynamics evaluation.", fontsize=11)
            fig.text(.5, .008, f"Object delta baseline: native SEARCH sample at {attempt['baseline_native_timestamp_s']:.3f} s. Positive force is upward on the object. Evaluation only.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .025, 1, .94))
            name = f"{run['role']}_attempt{attempt['attempt']}_alignment_load.png"
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            paths.append(name)
    for index in range(max(len(run["attempts"]) for run in runs)):
        for phase in PHASES:
            fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True, sharey="row", squeeze=False)
            for column, run in enumerate(runs):
                attempt = run["attempts"][index] if index < len(run["attempts"]) else None
                rows = [] if attempt is None else attempt["series"][phase]
                draw(axes[:, column], rows, run["weight_n"], relative=True)
                clock = "not executed" if not rows else f"absolute sim {rows[0]['timestamp_s'] - STEP:.3f}-{rows[-1]['timestamp_s']:.3f} s"
                axes[0, column].set_title(f"{'R20' if column == 0 else 'R22'} | {clock}", fontsize=10)
            fig.suptitle(f"Attempt {index + 1} | {phase} | Each run's own stage clock; no matched-prefix or causal claim", fontsize=11)
            fig.text(.5, .008, "Object delta uses each attempt's first native SEARCH sample. Cached forces/wrist vs integrated object may differ by one evaluation; evaluation only.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .025, 1, .96))
            name = f"attempt{index + 1}_{phase.lower()}_r20_r22.png"
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            paths.append(name)
    return [capture(output / name) for name in paths]


def source_bindings():
    paths = (Path(__file__).resolve(), Path(load.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_close_deadline.py", PROJECT / "scripts/public_ycb_evidence.py",
        PROJECT / "scripts/evaluate_public_ycb.py", PROJECT / "scripts/develop_public_ycb_force_band.py",
        PROJECT / "tests/test_public_ycb_alignment_load_transfer.py")
    return {path.relative_to(PROJECT).as_posix(): digest(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-candidate-summary-sha256", type=checked_sha, required=True)
    parser.add_argument("--candidate-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    sources = source_bindings()
    runs = [read_run(args.baseline, "baseline", R20_SHA256), read_run(args.candidate, "candidate",
        args.expected_candidate_summary_sha256, args.candidate_job)]
    if (any(runs[0]["case"][key] != runs[1]["case"][key] for key in ("kind", "seed", "profile"))
            or runs[0]["inputs"]["scene"]["sha256"] != runs[1]["inputs"]["scene"]["sha256"]):
        raise ValueError("Comparison case/seed/profile or archived scene differs")
    runs = [extract(run) for run in runs]
    args.output.mkdir(parents=True, exist_ok=False)
    plots = figures(runs, args.output)
    for run in runs:
        path = args.output / f"{run['role']}_native_reduced.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump([{key: attempt[key] for key in ("attempt", "baseline_native_timestamp_s", "baseline_object_z_m", "series")}
                for attempt in run["attempts"]], stream, separators=(",", ":"), allow_nan=False)
        run["reduced_native_artifact"] = capture(path)
        for attempt in run["attempts"]:
            del attempt["series"]
        del run["attempts_recorded"]
        unchanged(run["inputs"])
    if source_bindings() != sources:
        raise ValueError("Diagnostic source changed during analysis")
    result = {"format": "independent_r20_r22_alignment_load_transfer_v1", "analysis_source_sha256": sources,
        "independent_postrun_analysis": True, "part_of_runtime_frozen_manifest": False,
        "physical_prefix_independently_compared": False, "runs": runs, "figures": plots,
        "inputs_unchanged_during_analysis": True, "scope": SCOPE,
        "mesh_scope": "Public target meshes are checked against the frozen inventory. All scene-resolved mesh bytes used by the portable loader are additionally hash-bound at analysis time; other robot mesh bytes are not asserted to have been archived historically.",
        "phase_statistics_scope": "All recorded native samples plus the final up-to-250ms of each executed phase. Means and downward-tangent fractions describe samples; table load above weight or downward hand tangent alone is not a causal geometry diagnosis or success predicate."}
    path = args.output / "comparison.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"comparison": str(path), "sha256": digest(path), "figures": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
