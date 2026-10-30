#!/usr/bin/env python3
"""Final, audit-bound R23 actual-wrist probe feedback and load diagnostics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import audit_public_ycb_actual_wrist_probe as probe_audit
import plot_public_ycb_alignment_load_transfer as common
from public_ycb_evidence import load_evidence_model


PROJECT = Path(__file__).resolve().parents[1]
FORMAT = "r23_actual_wrist_probe_feedback_diagnostics_v1"
RUNNER = "public_ycb_actual_wrist_probe_development_r23"
AUDIT_FORMAT = "r23_actual_wrist_probe_reference_native_and_retained_source_audit_v1"
AUDITOR = "scripts/audit_public_ycb_actual_wrist_probe.py"
DERIVED_SHA256 = "f1a1ee4c1ad8c6d4e8991c6f5a8ed931526f9165a20233fcb8c39f8d8cb4a0dd"
CHECKS = ("all_evidence_consistent", "all_probe_reference_and_native_clock_checks_consistent",
    "all_probe_termination_and_original_confirmation_checks_consistent",
    "all_retained_arm_source_and_alignment_checks_consistent")
SCOPE = (
    "Completed, independently audited R23 probes only; a confirmed 5mm probe is not full task success. "
    "World +z forces act on the object. Contact normal and tangent are a frame decomposition, not "
    "separate sensors. Recorded wrist and contacts retain the last dynamics-evaluation cache at each "
    "completed 2ms transition; no live simulation or reconstructed contact dynamics is run. "
    "IK samples retain their actual sample clock and next-segment 20ms lookahead, including the first "
    "sample at the CLOSE/admission clock. The shaded 5mm +/-0.5mm z band is a visual projection; the "
    "original endpoint predicate uses full 3D error. Parent-policy and post-R16 sample confirmation "
    "counters are retained separately. Only the independently audited exact final R16 abort permits "
    "their difference; an inhibited counter is never presented as decisive confirmation. Counters "
    "and outcomes are recorded controller values, not recomputed from plotted force curves. "
    "All target identities/contact force signs are offline evaluation only. No accuracy, unseen-object "
    "generalization, exact slip, matched-prefix comparison or causal improvement is claimed.")


def check_exit(status, request, directory, protocol):
    command = request["command"]
    if command.count("--output") != 1 or command.index("--output") + 1 >= len(command):
        raise ValueError("Expected unique supervisor output")
    output = Path(command[command.index("--output") + 1])
    output = output if output.is_absolute() else PROJECT / output
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or type(status["returncode"]) is not int
            or status["returncode"] != 0 or output.resolve() != directory
            or command.count("scripts/develop_public_ycb_actual_wrist_probe_r23.py") != 1
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]):
        raise ValueError("Expected bound observed FINISHED rc0 R23 child exit")


def check_probe_receipt(receipt, inputs, row):
    if (receipt["format"] != AUDIT_FORMAT or receipt["accepted_probe_evidence"] is not True
            or any(receipt["checks"].get(key) is not True for key in CHECKS)
            or receipt["summary_sha256"] != inputs["summary"]["sha256"]
            or receipt["protocol_sha256"] != inputs["protocol"]["sha256"]
            or receipt["script_sha256"] != inputs["probe_auditor"]["sha256"]
            or receipt["source"]["source_count"] != 167
            or receipt["source"]["inherited_r22_source_count"] != 146
            or receipt["native_samples"] != row["native_audit"]["native_samples"]
            or abs(receipt["native_timestep_s"] - common.STEP) > 1e-12
            or receipt["actual_hand_instances"] != row["effective_hand_instances"]
            or receipt["task_success"] != row["success"] or receipt["held_lift"] != row["held_lift"]):
        raise ValueError("Independent final R23 probe receipt is unaccepted or inconsistently bound")
    for key in ("summary", "protocol", "archive", "scene", "controller", "metadata", "result",
            "native", "trace", "load_transfer_probe", "bias_recovery", "model_loader", "alignment_auditor"):
        if any(receipt["inputs"][key][field] != inputs[key][field] for field in ("path", "bytes", "sha256")):
            raise ValueError("Independent R23 probe receipt input differs: " + key)


def read_run(directory, expected_sha, job, audit_path):
    directory = directory.resolve(strict=True)
    summary, summary_bound = common.captured_json(common.safe_path(directory, "summary.json"))
    protocol, protocol_bound = common.captured_json(common.safe_path(directory, "protocol.json"))
    if (summary_bound["sha256"] != expected_sha or summary["complete"] is not True
            or summary["runner_errors"] != 0 or summary["protocol"] != protocol
            or protocol["format"] != RUNNER or summary["runner_revision"] != RUNNER
            or len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or len(protocol["source_hashes_before"]) != 167 or protocol["public_meshes_unchanged"] is not True):
        raise ValueError("Expected exact finalized unchanged 167-source one-case R23 summary")
    row = summary["rows"][0]
    if (row["runner_error"] is not None or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or row["controller_class"] != "ProvisionalPreloadBudgetHand"
            or not 1 <= row["effective_hand_instances"] <= 2
            or len(row["attempts"]) != row["effective_hand_instances"]
            or any(row[key] != value for key, value in protocol["schedule"][0].items())
            or row["native_audit"]["external_target_qpos_writes"] != 0):
        raise ValueError("Unknown or mismatched finalized R23 case/outcome")
    case = common.case_path(directory, row["path"])
    metadata, metadata_bound = common.captured_json(common.safe_path(case, "metadata.json"))
    if metadata != row:
        raise ValueError("Final metadata differs from summary row")
    inputs = {"summary": summary_bound, "protocol": protocol_bound, "metadata": metadata_bound}
    for key, filename in (("scene", "scene.xml"), ("controller", "derived_controller.py"), ("result", "result.json")):
        inputs[key] = common.capture(common.safe_path(case, filename))
    inputs["archive"] = common.capture(common.safe_path(directory, "frozen_source.zip"))
    inputs["source_archive"] = inputs["archive"]
    if (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inputs["scene"]["sha256"]
            or row["scene_sha256_at_finalize"] != inputs["scene"]["sha256"]
            or row["derived_controller_sha256"] != DERIVED_SHA256
            or inputs["controller"]["sha256"] != DERIVED_SHA256):
        raise ValueError("Loaded scene or actual executed R23 arm binding differs")
    result = json.loads(Path(inputs["result"]["path"]).read_bytes())
    if (result["success"] != row["success"] or result["failure"] != row["failure"]
            or result["attempts"] != row["attempts"] or result["actual_wrist_probe"] != row["actual_wrist_probe"]):
        raise ValueError("Final result and metadata outcome/probe telemetry differ")
    with zipfile.ZipFile(inputs["archive"]["path"]) as archive:
        before = protocol["source_hashes_before"]
        if len(archive.namelist()) != len(before) or set(archive.namelist()) != set(before):
            raise ValueError("Frozen source archive inventory differs")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != sha for name, sha in before.items()):
            raise ValueError("Frozen source archive bytes differ")
    native_audit = common.receipts(directory, summary, inputs, row)
    job = job.resolve(strict=True)
    status, inputs["supervisor_status"] = common.captured_json(common.safe_path(job, "status.json"))
    request, inputs["supervisor_request"] = common.captured_json(common.safe_path(job, "request.json"))
    check_exit(status, request, directory, protocol)
    for name in ("native", "trace", "load_transfer_probe", "bias_recovery"):
        artifact = row["artifacts"][name]
        inputs[name] = common.capture(common.safe_path(case, artifact["path"]))
        if any(inputs[name][field] != artifact[field] for field in ("bytes", "sha256")):
            raise ValueError("Final compressed artifact differs: " + name)
    for key, relative in (("probe_auditor", AUDITOR), ("model_loader", "scripts/public_ycb_evidence.py"),
            ("alignment_auditor", "scripts/audit_public_ycb_settled_alignment.py")):
        inputs[key] = common.capture(common.safe_path(PROJECT, relative))
    receipt, inputs["probe_receipt"] = common.captured_json(audit_path.resolve(strict=True))
    check_probe_receipt(receipt, inputs, row)
    for name, record in protocol["public_meshes"].items():
        bound = common.capture(common.safe_path(PROJECT, record["path"]))
        if any(bound[field] != record[field] for field in ("bytes", "sha256")):
            raise ValueError("Frozen public mesh differs")
        inputs["public_mesh_" + name] = bound
    scene = Path(inputs["scene"]["path"])
    for ordinal, mesh in enumerate(ET.parse(scene).getroot().iter("mesh")):
        if "file" in mesh.attrib:
            path = (PROJECT / "vendor/mujoco_menagerie" / mesh.attrib["file"]).resolve(strict=True)
            inputs[f"loaded_mesh_{ordinal}"] = common.capture(common.safe_path(PROJECT, path.relative_to(PROJECT)))
    model = load_evidence_model(scene)
    weight = -float(model.body_mass[model.body("target").id]) * float(model.opt.gravity[2])
    if (abs(float(model.opt.timestep) - common.STEP) > 1e-12
            or not np.allclose(model.opt.gravity[:2], 0., atol=1e-12) or not np.isfinite(weight) or weight <= 0.):
        raise ValueError("Expected 2ms native clock and positive vertical target weight")
    return {"inputs": inputs, "model": model, "row": row, "weight_n": weight,
        "probe_parameters": protocol["load_transfer_probe"], "native_and_strict": native_audit,
        "independent_probe_audit_accepted": True, "audit_probe_count": len(receipt["probes"]),
        "audited_probes": receipt["probes"]}


def bind_audited_probe(record, audited):
    outcome = record["outcome"]
    if (type(audited["original_r16_pre_sample_abort_observed"]) is not bool
            or abs(audited["start_s"] - record["start_s"]) > 1e-9
            or abs(audited["end_s"] - outcome["timestamp_s"]) > 1e-9
            or audited["native_samples"] != len(record["samples"])
            or audited["ik_samples"] != len(record["ik_samples"])
            or any(audited[key] != outcome[key] for key in ("arm_exit_reason", "final_reason",
                "original_confirmation_attempted", "original_confirmation_accepted"))
            or abs(audited["final_confirmed_seconds"] - outcome["inherited_confirmed_seconds"]) > 1e-10):
        raise ValueError("Audited probe classification/outcome does not bind this exact telemetry")
    return audited


def sample_counter(probe, sample, policy, bias_aborts):
    """Keep the parent observation and post-inhibit observation distinct."""
    parent = float(policy["probe_confirmed_seconds"])
    after = float(sample["inherited_confirmed_seconds"])
    if not np.isfinite([parent, after]).all() or min(parent, after) < 0.:
        raise ValueError("Invalid original confirmation counter")
    outcome, audited = probe["outcome"], probe["audited_probe"]
    final = abs(sample["timestamp_s"] - outcome["timestamp_s"]) <= 1e-9
    aborted = audited["original_r16_pre_sample_abort_observed"] and final
    if aborted:
        event = bias_aborts.get(common.tick(sample["timestamp_s"], common.STEP))
        if (event is None or audited["classification"] != "motion_failure"
                or outcome["arm_exit_reason"] != "MOTION_EXCEPTION"
                or not isinstance(outcome["motion_failure"], str) or not outcome["motion_failure"]
                or outcome["final_reason"] != outcome["motion_failure"]
                or outcome["probe_active_at_motion_end"] is not False
                or sample["currently_probe_active"] is not False
                or outcome["original_confirmation_attempted"] is not False
                or outcome["original_confirmation_accepted"] is not None
                or abs(after) > 1e-10 or abs(outcome["inherited_confirmed_seconds"]) > 1e-10
                or parent > common.STEP + 1e-10):
            raise ValueError("Counter reset lacks its audited exact final inactive R16 failure")
        probe_audit.check_bias_abort(event, sample["timestamp_s"], outcome["motion_failure"])
    elif sample["currently_probe_active"] is not True or abs(parent - after) > 1e-10:
        raise ValueError("Original confirmation counters differ outside the audited final R16 abort")
    return parent, after, bool(aborted)


def reduce_trace(rows, kinematics):
    extras = {}

    def tapped():
        for row in rows:
            index = common.tick(row["timestamp"], common.STEP)
            if index not in extras:
                extras[index] = {"actual_wrist_probe": row.get("actual_wrist_probe"),
                    "probe_policy": row.get("probe_policy")}
            yield row

    reduced, duplicate_count = common.reduce_trace(tapped(), kinematics)
    for index, extra in extras.items():
        reduced[index].update(extra)
    return reduced, duplicate_count


def extract(run):
    model = run.pop("model")
    telemetry = run["row"]["actual_wrist_probe"]
    if telemetry["format"] != "actual_wrist_probe_policy_v1" or len(telemetry["attempts"]) != run["audit_probe_count"]:
        raise ValueError("Audited actual probe count/format differs")
    if len(run["audited_probes"]) != run["audit_probe_count"]:
        raise ValueError("Missing independently audited probe classifications")
    trace, duplicates = reduce_trace(common.json_array(Path(run["inputs"]["trace"]["path"])), common.CommandKinematics(model))
    bias_aborts = {}
    for event in common.json_array(Path(run["inputs"]["bias_recovery"]["path"])):
        if event.get("probe_abort_applied") is True:
            index = common.tick(event["post_step_timestamp_s"], common.STEP)
            if index in bias_aborts:
                raise ValueError("Repeated original R16 abort clock")
            bias_aborts[index] = event
    probes, samples = [], {}
    for ordinal, record in enumerate(telemetry["attempts"], 1):
        start = float(record["start_s"])
        position = common.load.vector(record["actual_admission_start_world_m"], 3, "probe admission wrist")
        if record["outcome"] is None or not record["samples"] or not record["ik_samples"]:
            raise ValueError("Incomplete actual probe telemetry cannot be plotted as final")
        probe = {"probe": ordinal, "start_s": start, "actual_admission_start_world_m": position.tolist(),
            "outcome": record["outcome"], "ik_samples": record["ik_samples"], "series": [],
            "audited_probe": bind_audited_probe(record, run["audited_probes"][ordinal - 1])}
        probes.append(probe)
        for sample in record["samples"]:
            index = common.tick(sample["timestamp_s"], common.STEP)
            if index in samples or not start < sample["timestamp_s"] <= start + 6. + 1e-8:
                raise ValueError("Overlapping or out-of-bounds probe sample clock")
            samples[index] = (probe, sample)
    names = [model.geom(g).name or "" for g in range(model.ngeom)]
    bodies = [model.body(int(model.geom_bodyid[g])).name or "" for g in range(model.ngeom)]
    target = model.geom("target_geom").id
    count, seen, global_attempt, previous_phase = 0, set(), 0, None
    for count, native in enumerate(common.json_array(Path(run["inputs"]["native"]["path"])), 1):
        index, joined = common.exact_native_trace(native, count, trace)
        if native["state"] == "SEARCH" and previous_phase != "SEARCH":
            global_attempt += 1
        previous_phase = native["state"]
        if index not in samples:
            continue
        probe, sample = samples[index]
        if not 1 <= global_attempt <= len(run["row"]["attempts"]):
            raise ValueError("Probe is not within a recorded actual attempt")
        if probe.setdefault("global_attempt", global_attempt) != global_attempt:
            raise ValueError("One probe cannot cross actual attempt boundaries")
        probe["attempt_failure"] = run["row"]["attempts"][global_attempt - 1].get("failure")
        policy = joined["probe_policy"]
        if (joined["phase"] != "TRIAL_LIFT" or joined["actual_wrist_probe"] != sample or policy is None
                or abs(joined["wrist_actual_z_m"] - sample["current_actual_wrist_world_m"][2]) > 1e-10):
            raise ValueError("First trace/native probe clock, wrist or original confirmation differs")
        parent_counter, post_r16_counter, aborted = sample_counter(probe, sample, policy, bias_aborts)
        actual = common.load.vector(sample["current_actual_wrist_world_m"], 3, "actual cached wrist")
        start = np.asarray(probe["actual_admission_start_world_m"])
        displacement = actual - start
        probe["series"].append({"timestamp_s": native["timestamp"], "elapsed_s": sample["elapsed_s"],
            "wrist_delta_z_m": float(displacement[2]),
            "endpoint_error_3d_m": float(np.linalg.norm(displacement - [0., 0., .005])),
            "logical_command_fk_z_m": joined["wrist_command_fk_z_m"],
            "parent_policy_confirmed_seconds": parent_counter,
            "inherited_confirmed_seconds": post_r16_counter,
            "audited_r16_abort_before_sample": aborted,
            **{key: policy[key] for key in ("actual_probe_endpoint_reached", "table_unloaded", "sampled_fixed_normal_1g_feasible")},
            **common.vertical_forces(native, model, target, names, bodies)})
        seen.add(index)
    expected = run["row"]["native_audit"]["native_samples"]
    if (count != expected or abs(count * common.STEP - run["row"]["duration_s"]) > 1e-7
            or seen != set(samples) or set(trace) - set(range(1, count + 1))
            or global_attempt != len(run["row"]["attempts"])):
        raise ValueError("Final native count/duration or actual probe/trace coverage differs")
    for probe in probes:
        series = probe["series"]
        if any(common.tick(b["timestamp_s"], common.STEP) - common.tick(a["timestamp_s"], common.STEP) != 1
                for a, b in zip(series, series[1:])):
            raise ValueError("Actual probe native coverage must be contiguous")
        if (abs(series[-1]["timestamp_s"] - probe["outcome"]["timestamp_s"]) > 1e-9
                or sum(row["audited_r16_abort_before_sample"] for row in series) !=
                    int(probe["audited_probe"]["original_r16_pre_sample_abort_observed"])):
            raise ValueError("Audited R16 abort is not uniquely bound to the last native probe sample")
        probe["statistics"] = {key: common.sampled_statistics(row[key] for row in series)
            for key in ("wrist_delta_z_m", "endpoint_error_3d_m", "parent_policy_confirmed_seconds",
                "inherited_confirmed_seconds", *common.FORCES)}
    run.update(probes=probes, native_samples_streamed=count, first_trace_duplicate_clock_records=duplicates)
    return run


def figures(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    for probe in run["probes"]:
        rows, ik, start = probe["series"], probe["ik_samples"], np.asarray(probe["actual_admission_start_world_m"])
        times = [row["elapsed_s"] for row in rows]
        ik_times = [row["timestamp_s"] - probe["start_s"] for row in ik]
        fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        axes[0].axhspan(4.5, 5.5, color="#dcebdc", label="Original endpoint z projection: 5 +/-0.5mm")
        axes[0].plot(times, [row["wrist_delta_z_m"] * 1000. for row in rows], color="#176ca4", label="Actual cached wrist dz")
        for field, label, color in (("desired_actual_path_world_m", "Desired cubic dz (20ms lookahead)", "#4d7c42"),
                ("cartesian_ik_target_world_m", "Bounded IK target dz", "#aa6633")):
            axes[0].plot(ik_times, [(row[field][2] - start[2]) * 1000. for row in ik], color=color, linestyle="--", label=label)
        for axis_index, color in enumerate(("#176ca4", "#4d7c42", "#aa3344")):
            axes[1].plot(ik_times, [row["bounded_reference_correction_world_m"][axis_index] * 1000. for row in ik],
                color=color, label=f"Reference correction {'xyz'[axis_index]}")
        axes[1].plot(ik_times, [np.linalg.norm(row["bounded_reference_correction_world_m"]) * 1000. for row in ik],
            color="#333333", linestyle="--", label="Correction norm (2mm cap)")
        axes[1].axhline(2., color="#888888", linestyle=":")
        axes[1].plot(times, [row["endpoint_error_3d_m"] * 1000. for row in rows], color="#8b6ba1", label="Actual 3D endpoint error")
        axes[1].axhline(.5, color="#8b6ba1", linestyle=":", label="Original 3D tolerance 0.5mm")
        for field, label, color in (("table_z_n", "Table on object", "#4d7c42"),
                ("hand_net_z_n", "Hand net on object", "#176ca4"), ("other_z_n", "Other contacts", "#8b6ba1")):
            axes[2].plot(times, [row[field] for row in rows], color=color, label=label)
        axes[2].axhline(run["weight_n"], color="#333333", linestyle="--", label="Object weight")
        for field, label, color in (("hand_normal_z_n", "Hand normal", "#aa6633"),
                ("hand_tangent_z_n", "Hand tangent", "#aa3344"), ("hand_net_z_n", "Hand net", "#176ca4")):
            axes[3].plot(times, [row[field] for row in rows], color=color, label=label)
        axes[3].axhline(0., color="#888888", linewidth=.7)
        axes[4].plot(times, [row["parent_policy_confirmed_seconds"] for row in rows], color="#4d7c42", linestyle="--",
            label="Parent hand policy before final R16 inhibit")
        axes[4].plot(times, [row["inherited_confirmed_seconds"] for row in rows], color="#176ca4",
            label="Recorded hand counter after native-step / R16 inhibit")
        axes[4].axhline(.25, color="#333333", linestyle="--", label="Original required 0.25s")
        outcome = probe["outcome"]
        if rows[-1]["audited_r16_abort_before_sample"]:
            axes[4].plot(times[-1], rows[-1]["inherited_confirmed_seconds"], "x", color="#aa3344", markersize=8,
                label="Audited R16 abort; no decisive confirmation")
        if outcome["original_confirmation_attempted"]:
            axes[4].plot(outcome["confirmation_timestamp_s"] - probe["start_s"], outcome["inherited_confirmed_seconds"],
                "o" if outcome["original_confirmation_accepted"] else "x", markersize=8, color="#aa3344",
                label="Original decisive confirmation")
        labels = ("Wrist dz (mm)", "Correction / 3D error (mm)", "Vertical force (N)", "Hand vertical force (N)", "Confirmation (s)")
        for axis, label in zip(axes, labels):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.legend(loc="best", fontsize=8, ncol=2)
            axis.axvline(3., color="#999999", linestyle=":", linewidth=.8)
        axes[-1].set_xlabel(f"Seconds since this probe's admission (absolute simulation time {probe['start_s']:.3f}s)")
        axes[-1].set_ylim(bottom=-.01, top=max(.3, max(row["inherited_confirmed_seconds"] for row in rows) * 1.15))
        axes[-1].set_xlim(0., max(.02, rows[-1]["elapsed_s"]) * 1.02)
        final = rows[-1]
        fig.suptitle(f"R23 probe {probe['probe']} / actual attempt {probe['global_attempt']} | "
            f"task={run['row']['success']} | held={run['row']['held_lift']}\n"
            f"{outcome['arm_exit_reason']} | {outcome['final_reason']}", fontsize=10)
        fig.text(.5, .018, f"Final original flags: endpoint={final['actual_probe_endpoint_reached']}, "
            f"table unloaded={final['table_unloaded']}, fixed-normal 1g={final['sampled_fixed_normal_1g_feasible']}.\n"
            f"Final counters: parent={final['parent_policy_confirmed_seconds']:.6f}s, "
            f"post-R16/sample={final['inherited_confirmed_seconds']:.6f}s; "
            f"decisive confirmation attempted={outcome['original_confirmation_attempted']}.\n"
            "Cached dynamics measurements; IK keeps its own sample clock. Endpoint is full 3D; forces alone do not certify success.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0., .052, 1., .95))
        for extension in ("png", "pdf"):
            path = output / f"probe{probe['probe']}_bounded_feedback.{extension}"
            fig.savefig(path, dpi=150)
            artifacts.append(common.capture(path))
        plt.close(fig)
    return artifacts


def source_bindings():
    paths = (Path(__file__).resolve(), Path(common.__file__).resolve(), Path(common.load.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_close_deadline.py", PROJECT / "scripts/public_ycb_evidence.py",
        PROJECT / AUDITOR, PROJECT / "tests/test_public_ycb_actual_wrist_probe_plots.py")
    return {path.relative_to(PROJECT).as_posix(): common.digest(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True, type=common.checked_sha)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--probe-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    sources = source_bindings()
    run = extract(read_run(args.directory, args.expected_summary_sha256, args.job, args.probe_audit))
    common.unchanged(run["inputs"])
    args.output.mkdir(parents=True, exist_ok=False)
    plots = figures(run, args.output)
    path = args.output / "probe_native_reduced.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(run["probes"], stream, separators=(",", ":"), allow_nan=False)
    reduced = common.capture(path)
    for probe in run["probes"]:
        del probe["series"], probe["ik_samples"]
    row = run.pop("row")
    run["case"] = {key: row.get(key) for key in ("kind", "seed", "path", "success", "held_lift", "failure")}
    common.unchanged(run["inputs"])
    if source_bindings() != sources:
        raise ValueError("Plotting source changed during analysis")
    result = {"format": FORMAT, "analysis_source_sha256": sources, "independent_postrun_analysis": True,
        "part_of_runtime_frozen_manifest": False, "inputs_unchanged_during_analysis": True,
        "run": run, "figures": plots, "reduced_native": reduced, "scope": SCOPE}
    receipt = args.output / "diagnostics.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"diagnostics": str(receipt), "sha256": common.digest(receipt), "figures": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
