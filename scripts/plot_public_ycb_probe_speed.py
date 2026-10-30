#!/usr/bin/env python3
"""Finalized public-YCB probe speed diagnostics from recorded cached wrist positions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import plot_public_ycb_alignment_load_transfer as evidence


PROJECT = Path(__file__).resolve().parents[1]
STEP = .002
FORMAT = "public_ycb_recorded_probe_speed_diagnostics_v1"
SCOPE = (
    "Offline diagnostics of finalized recorded probes, including failed tasks. Velocity is the "
    "difference of consecutive trace wrist_position caches divided by the native 2ms interval; "
    "the first sample uses the admission wrist cache, independently matched to the admission trace. "
    "These are last-dynamics-evaluation caches observed at completed-transition timestamps, not "
    "instantaneous velocities recomputed at integrated qpos. IK samples retain their actual clock "
    "and lookahead. Contact pairs are all recorded native geometry pairs, not an object-specific "
    "contact or force audit. Guard observations retain their original observation clock; an absent "
    "final trace contact_guard stays absent. Native/strict receipts are bound, but this tool does "
    "not independently audit LP, QP, contact association, reference policy, or task success. "
    "No simulation step, state reconstruction, or runtime parameter change is performed.")


def number(value):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("Expected a finite recorded number")
    return float(value)


def vector(value, length=3):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError("Invalid recorded vector")
    return array


def check_summary(summary, protocol, expected_runner_revision, expected_controller_class):
    before = protocol["source_hashes_before"]
    if (summary["complete"] is not True or summary["runner_errors"] != 0
            or summary["protocol"] != protocol or protocol["format"] != expected_runner_revision
            or summary["runner_revision"] != expected_runner_revision or not before
            or protocol["source_changed_during_run"] is not False
            or before != protocol["source_hashes_after"] or protocol["public_meshes_unchanged"] is not True
            or len(summary["rows"]) != 1 or len(protocol["schedule"]) != 1):
        raise ValueError("Expected an explicitly versioned, finalized unchanged one-case run")
    row = summary["rows"][0]
    count = row["native_audit"]["native_samples"]
    if (row["runner_error"] is not None or row["controller_class"] != expected_controller_class
            or type(row["success"]) is not bool or type(row["held_lift"]) is not bool
            or type(count) is not int or count <= 0
            or row["native_audit"]["integration_steps"] != count
            or row["native_audit"]["external_target_qpos_writes"] != 0
            or abs(number(row["duration_s"]) - count * STEP) > 1e-7
            or any(row[key] != value for key, value in protocol["schedule"][0].items())):
        raise ValueError("Final case, class, outcome or complete 2ms count differs")
    return row


def check_exit(status, request, directory, protocol, runner_entry):
    command = request["command"]
    if command.count("--output") != 1 or command.index("--output") + 1 >= len(command):
        raise ValueError("Expected a unique supervisor output argument")
    output = Path(command[command.index("--output") + 1])
    output = output if output.is_absolute() else PROJECT / output
    if (status["state"] != "FINISHED" or status["terminal"] is not True
            or status["child_exit_observed"] is not True or type(status["returncode"]) is not int
            or status["returncode"] != 0 or output.resolve() != directory
            or command.count(runner_entry) != 1 or runner_entry not in protocol["source_hashes_before"]
            or request["supervisor_sha256"] != protocol["source_hashes_before"]["scripts/run_supervised_experiment.py"]):
        raise ValueError("Expected the bound, observed FINISHED rc0 runner")


def read_run(directory, expected_summary_sha256, job, expected_runner_revision,
             expected_controller_class, runner_entry):
    directory = directory.resolve(strict=True)
    summary, sb = evidence.captured_json(evidence.safe_path(directory, "summary.json"))
    protocol, pb = evidence.captured_json(evidence.safe_path(directory, "protocol.json"))
    if sb["sha256"] != expected_summary_sha256:
        raise ValueError("Final summary SHA256 differs")
    row = check_summary(summary, protocol, expected_runner_revision, expected_controller_class)
    case = evidence.case_path(directory, row["path"])
    metadata, mb = evidence.captured_json(evidence.safe_path(case, "metadata.json"))
    if metadata != row:
        raise ValueError("Final metadata differs from summary")
    inputs = {"summary": sb, "protocol": pb, "metadata": mb}
    for key, filename in (("result", "result.json"), ("scene", "scene.xml"),
                          ("controller", "derived_controller.py")):
        inputs[key] = evidence.capture(evidence.safe_path(case, filename))
    result = json.loads(Path(inputs["result"]["path"]).read_bytes())
    if any(result[key] != row[key] for key in ("success", "failure", "attempts", "actual_wrist_probe")):
        raise ValueError("Final result outcome/probe telemetry differs")
    if (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != inputs["scene"]["sha256"]
            or row["scene_sha256_at_finalize"] != inputs["scene"]["sha256"]
            or row["derived_controller_sha256"] != inputs["controller"]["sha256"]):
        raise ValueError("Loaded scene or executed controller binding differs")
    option = ET.parse(inputs["scene"]["path"]).getroot().find("option")
    if option is None or abs(number(float(option.attrib["timestep"])) - STEP) > 1e-12:
        raise ValueError("Expected an explicit recorded 2ms scene timestep")
    inputs["source_archive"] = evidence.capture(evidence.safe_path(directory, "frozen_source.zip"))
    with zipfile.ZipFile(inputs["source_archive"]["path"]) as archive:
        before = protocol["source_hashes_before"]
        if len(archive.namelist()) != len(before) or set(archive.namelist()) != set(before):
            raise ValueError("Frozen source ZIP inventory differs")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != sha for name, sha in before.items()):
            raise ValueError("Frozen source ZIP bytes differ")
    audit = evidence.receipts(directory, summary, inputs, row)
    job = job.resolve(strict=True)
    status, inputs["supervisor_status"] = evidence.captured_json(evidence.safe_path(job, "status.json"))
    request, inputs["supervisor_request"] = evidence.captured_json(evidence.safe_path(job, "request.json"))
    check_exit(status, request, directory, protocol, runner_entry)
    if not {"native", "trace", "guard"} <= set(row["artifacts"]):
        raise ValueError("Missing required recorded speed/guard artifacts")
    for key, artifact in row["artifacts"].items():
        bound = evidence.capture(evidence.safe_path(case, artifact["path"]))
        if any(bound[field] != artifact[field] for field in ("bytes", "sha256")):
            raise ValueError("Final artifact bytes differ: " + key)
        inputs["artifact_" + key] = bound
    limit = number(protocol["load_transfer_probe"]["maximum_actual_wrist_speed_m_s"])
    if limit <= 0.:
        raise ValueError("Invalid recorded speed limit")
    return {"directory": str(directory), "runner_revision": expected_runner_revision,
        "controller_class": expected_controller_class, "runner_entry": runner_entry,
        "source_count": len(protocol["source_hashes_before"]), "inputs": inputs,
        "native_and_strict": audit, "row": row, "speed_limit_m_s": limit}


def trace_groups(rows, counts):
    """First same-clock record is the record sampled by the native auditor."""
    previous, first = None, None
    for row in rows:
        index = evidence.tick(row["timestamp"], STEP)
        if previous is not None and index < previous:
            raise ValueError("Trace clocks are not chronological")
        if index == previous:
            counts["duplicate_trace_clocks"] += 1
        else:
            if first is not None:
                yield first
            first = row
        previous = index
    if first is not None:
        yield first


def prepare_probes(telemetry):
    if telemetry["format"] != "actual_wrist_probe_policy_v1":
        raise ValueError("Unsupported recorded probe telemetry version")
    probes, samples, admissions, ik_clocks = [], {}, {}, {}
    for ordinal, record in enumerate(telemetry["attempts"], 1):
        start, outcome = number(record["start_s"]), record["outcome"]
        if outcome is None or not record["samples"] or not record["ik_samples"]:
            raise ValueError("Cannot plot an incomplete probe as final")
        end = number(outcome["timestamp_s"])
        start_index, end_index = evidence.tick(start, STEP), evidence.tick(end, STEP)
        if start_index >= end_index or start_index in admissions:
            raise ValueError("Invalid or repeated probe admission clock")
        probe = {"probe": ordinal, "start_s": start, "outcome": outcome,
            "actual_admission_start_world_m": vector(record["actual_admission_start_world_m"]).tolist(),
            "series": [], "ik_samples": record["ik_samples"], "events": [], "guard_observations": []}
        probes.append(probe)
        admissions[start_index] = probe
        expected = list(range(start_index + 1, end_index + 1))
        actual = [evidence.tick(row["timestamp_s"], STEP) for row in record["samples"]]
        if actual != expected or any(index in samples for index in actual):
            raise ValueError("Probe samples must cover every native tick from admission to final outcome")
        for index, sample in zip(actual, record["samples"]):
            samples[index] = (probe, sample)
        previous = start_index - 1
        previous_ik = None
        probe["ik_target_difference_rates"] = []
        for row in record["ik_samples"]:
            index = evidence.tick(row["timestamp_s"], STEP)
            if not start_index <= index < end_index or index <= previous or index in ik_clocks:
                raise ValueError("Invalid recorded IK clock")
            for key in ("desired_actual_path_world_m", "cartesian_ik_target_world_m",
                        "bounded_reference_correction_world_m", "actual_wrist_at_ik_sample_world_m"):
                vector(row[key])
            ik_clocks[index] = row
            if previous_ik is not None:
                dt = number(row["timestamp_s"]) - number(previous_ik["timestamp_s"])
                rate = (vector(row["cartesian_ik_target_world_m"]) -
                    vector(previous_ik["cartesian_ik_target_world_m"])) / dt
                probe["ik_target_difference_rates"].append({"timestamp_s": row["timestamp_s"],
                    "sample_interval_s": dt, "target_difference_velocity_world_m_s": rate.tolist()})
            previous_ik = row
            previous = index
    return probes, samples, admissions, ik_clocks


def extract(run):
    row, inputs = run["row"], run["inputs"]
    probes, samples, admissions, ik_clocks = prepare_probes(row["actual_wrist_probe"])
    counts = {"duplicate_trace_clocks": 0}
    traces = iter(trace_groups(evidence.json_array(Path(inputs["artifact_trace"]["path"])), counts))
    seen, seen_admissions, seen_ik, attempts, previous_phase = set(), set(), set(), 0, None
    previous_wrist = {}
    count = 0
    for count, native in enumerate(evidence.json_array(Path(inputs["artifact_native"]["path"])), 1):
        trace = next(traces, None)
        if (trace is None or evidence.tick(native["timestamp"], STEP) != count
                or evidence.tick(trace["timestamp"], STEP) != count
                or abs(number(native["timestamp"]) - number(trace["timestamp"])) > 1e-9
                or native["state"] != trace["state"]
                or not np.array_equal(vector(native["motor_commands"], 29),
                    np.r_[vector(trace["arm_motor_control"], 7), vector(trace["hand_command"], 22)])):
            raise ValueError("Full native/first-trace clock, phase or motor control mismatch")
        if native["state"] == "SEARCH" and previous_phase != "SEARCH":
            attempts += 1
        previous_phase = native["state"]
        position = vector(trace["wrist_position"])
        if count in admissions:
            probe = admissions[count]
            if not np.allclose(position, probe["actual_admission_start_world_m"], rtol=0., atol=1e-10):
                raise ValueError("Admission wrist differs from its actual first trace cache")
            previous_wrist[probe["probe"]] = position.copy()
            seen_admissions.add(count)
        if count in ik_clocks:
            if not np.allclose(position, ik_clocks[count]["actual_wrist_at_ik_sample_world_m"], rtol=0., atol=1e-10):
                raise ValueError("IK wrist sample differs from its actual trace cache")
            seen_ik.add(count)
        if count not in samples:
            continue
        probe, sample = samples[count]
        if not 1 <= attempts <= len(row["attempts"]):
            raise ValueError("Probe is outside the recorded actual attempts")
        if probe.setdefault("global_attempt", attempts) != attempts or trace["state"] != "TRIAL_LIFT":
            raise ValueError("One recorded probe crosses phases/attempts")
        if (trace.get("actual_wrist_probe") != sample
                or not np.allclose(position, vector(sample["current_actual_wrist_world_m"]), rtol=0., atol=1e-10)
                or abs(number(sample["elapsed_s"]) - (number(trace["timestamp"]) - probe["start_s"])) > 1e-9):
            raise ValueError("Recorded probe sample and first trace cache differ")
        velocity = (position - previous_wrist[probe["probe"]]) / STEP
        previous_wrist[probe["probe"]] = position.copy()
        speed = float(np.linalg.norm(velocity))
        policy = trace["probe_policy"]
        if (not np.allclose(vector(policy["actual_wrist_position_world_m"]), position, rtol=0., atol=1e-10)
                or abs(number(policy["actual_wrist_speed_m_s"]) - speed) > 1e-8):
            raise ValueError("Recomputed adjacent-cache speed differs from recorded probe policy")
        pairs = sorted({tuple(sorted(contact["geom_ids"])) for contact in native["native_contacts"]})
        if probe["series"] and pairs != probe["series"][-1]["native_geometry_pairs"]:
            probe["events"].append({"timestamp_s": trace["timestamp"], "kind": "native_geometry_pair_change",
                "before": probe["series"][-1]["native_geometry_pairs"], "after": pairs})
        guard = trace.get("contact_guard")
        stop = policy.get("stop_reason")
        reasons = list(policy.get("motion_safety_reasons", []))
        if stop or reasons:
            probe["events"].append({"timestamp_s": trace["timestamp"], "kind": "recorded_probe_guard",
                "stop_reason": stop, "motion_safety_reasons": reasons})
        probe["series"].append({"timestamp_s": trace["timestamp"], "elapsed_s": sample["elapsed_s"],
            "wrist_delta_world_m": (position - probe["actual_admission_start_world_m"]).tolist(),
            "velocity_world_m_s": velocity.tolist(), "speed_m_s": speed,
            "policy_speed_m_s": policy["actual_wrist_speed_m_s"], "stop_reason": stop,
            "motion_safety_reasons": reasons, "native_contact_count": len(native["native_contacts"]),
            "native_geometry_pairs": pairs, "trace_contact_guard": guard})
        seen.add(count)
    if (count != row["native_audit"]["native_samples"] or next(traces, None) is not None
            or seen != set(samples) or seen_admissions != set(admissions) or seen_ik != set(ik_clocks)
            or attempts != len(row["attempts"])):
        raise ValueError("Incomplete full native stream, trace or probe/admission/IK coverage")
    for guard in evidence.json_array(Path(inputs["artifact_guard"]["path"])):
        timestamp = number(guard["observation_timestamp_s"])
        for probe in probes:
            if probe["start_s"] <= timestamp <= probe["outcome"]["timestamp_s"]:
                probe["guard_observations"].append(guard)
                if guard.get("retreat_reason"):
                    probe["events"].append({"timestamp_s": timestamp,
                        "kind": "recorded_contact_guard_retreat", "clock": "original_observation_timestamp_s",
                        "mode": guard["mode"], "reason": guard["retreat_reason"]})
    for probe in probes:
        series = probe["series"]
        probe["events"].append({"timestamp_s": probe["outcome"]["timestamp_s"],
            "kind": "recorded_final_outcome", "reason": probe["outcome"]["final_reason"]})
        probe["events"].sort(key=lambda event: event["timestamp_s"])
        probe["statistics"] = {"speed_m_s": evidence.sampled_statistics(r["speed_m_s"] for r in series),
            "maximum_policy_speed_residual_m_s": max(abs(r["speed_m_s"] - r["policy_speed_m_s"]) for r in series),
            "speed_exceeds_limit_samples": sum(r["speed_m_s"] > run["speed_limit_m_s"] for r in series),
            "final_velocity_world_m_s": series[-1]["velocity_world_m_s"],
            "final_trace_contact_guard_present": series[-1]["trace_contact_guard"] is not None}
    run.update(probes=probes, native_samples_streamed=count, **counts)
    return run


def figures(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    colors = ("#176ca4", "#4d7c42", "#b34444")
    for probe in run["probes"]:
        rows, ik = probe["series"], probe["ik_samples"]
        times = np.asarray([row["elapsed_s"] for row in rows])
        its = [row["timestamp_s"] - probe["start_s"] for row in ik]
        start = np.asarray(probe["actual_admission_start_world_m"])
        fig, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
        for k, color in enumerate(colors):
            axes[0].plot(times, [row["velocity_world_m_s"][k] * 1000. for row in rows], color=color, label="v" + "xyz"[k])
            axes[2].plot(times, [row["wrist_delta_world_m"][k] * 1000. for row in rows], color=color, label="Actual d" + "xyz"[k])
            axes[3].plot(its, [(row["desired_actual_path_world_m"][k] - start[k]) * 1000. for row in ik],
                color=color, linestyle="--", label="Desired d" + "xyz"[k])
            axes[3].plot(its, [(row["cartesian_ik_target_world_m"][k] - start[k]) * 1000. for row in ik],
                color=color, label="IK target d" + "xyz"[k])
            axes[4].plot(its, [row["bounded_reference_correction_world_m"][k] * 1000. for row in ik],
                color=color, label="Correction " + "xyz"[k])
        rates = probe["ik_target_difference_rates"]
        axes[0].plot([row["timestamp_s"] - probe["start_s"] for row in rates],
            [row["target_difference_velocity_world_m_s"][1] * 1000. for row in rates],
            color="#4d7c42", linestyle=":", label="IK target dy/dt (its own sample clock)")
        axes[1].plot(times, [row["speed_m_s"] * 1000. for row in rows], color="#176ca4", label="Recomputed 2ms cache speed")
        axes[1].plot(times, [row["policy_speed_m_s"] * 1000. for row in rows], color="#aa6633", linestyle="--", label="Recorded hand policy speed")
        axes[1].axhline(run["speed_limit_m_s"] * 1000., color="#333333", linestyle=":", label="Original speed limit")
        axes[4].plot(its, [np.linalg.norm(row["bounded_reference_correction_world_m"]) * 1000. for row in ik],
            color="#333333", linestyle="--", label="Correction norm")
        axes[5].step(times, [row["native_contact_count"] for row in rows], color="#176ca4", where="post", label="All native contact points")
        guard_axis = axes[5].twinx()
        guards = probe["guard_observations"]
        guard_axis.plot([g["observation_timestamp_s"] - probe["start_s"] for g in guards],
            [number(g["scale"]) for g in guards], color="#aa6633", linestyle="--", label="Recorded guard scale (observation clock)")
        guard_axis.set_ylabel("Guard scale")
        guard_axis.set_ylim(-.05, 1.1)
        for axis, label in zip(axes, ("Velocity (mm/s)", "Speed norm (mm/s)", "Actual offset (mm)",
                "Recorded IK offsets (mm)", "Reference correction (mm)", "Native contacts")):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.axvline(times[-1], color="#b34444", linestyle=":", linewidth=.9)
            axis.legend(loc="best", ncol=2, fontsize=8)
        handles, labels = guard_axis.get_legend_handles_labels()
        own_handles, own_labels = axes[5].get_legend_handles_labels()
        axes[5].legend(own_handles + handles, own_labels + labels, loc="best", fontsize=8)
        axes[-1].set_xlim(0., max(.02, times[-1]) * 1.015)
        axes[-1].set_xlabel(f"Seconds since probe admission; absolute admission={probe['start_s']:.3f}s")
        final = rows[-1]
        speed = final["speed_m_s"] * 1000.
        velocity = np.asarray(final["velocity_world_m_s"]) * 1000.
        fig.suptitle(f"{run['runner_revision']} | probe {probe['probe']} / actual attempt {probe['global_attempt']}\n"
            f"task={run['row']['success']} | held={run['row']['held_lift']} | {probe['outcome']['final_reason']}", fontsize=10)
        fig.text(.5, .018, f"Final speed={speed:.6f}mm/s; xyz=[{velocity[0]:.6f}, {velocity[1]:.6f}, {velocity[2]:.6f}]mm/s. "
            f"Final trace contact_guard present={final['trace_contact_guard'] is not None}.\n"
            "Red dotted line: recorded final outcome. Cached wrist finite differences; missing guards are not synthesized.\n"
            "Final native/strict evidence bound; LP/QP/reference policy is not independently audited here.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0., .052, 1., .955))
        for extension in ("png", "pdf"):
            path = output / f"probe{probe['probe']}_speed.{extension}"
            fig.savefig(path, dpi=150)
            artifacts.append(evidence.capture(path))
        plt.close(fig)
    return artifacts


def source_bindings():
    paths = (Path(__file__).resolve(), Path(evidence.__file__).resolve(), Path(evidence.load.__file__).resolve(),
        PROJECT / "scripts/compare_public_ycb_close_deadline.py",
        PROJECT / "tests/test_public_ycb_probe_speed_plots.py")
    return {path.relative_to(PROJECT).as_posix(): evidence.digest(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", required=True, type=evidence.checked_sha)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--expected-runner-revision", required=True)
    parser.add_argument("--expected-controller-class", required=True)
    parser.add_argument("--runner-entry", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    sources = source_bindings()
    run = extract(read_run(args.directory, args.expected_summary_sha256, args.job,
        args.expected_runner_revision, args.expected_controller_class, args.runner_entry))
    evidence.unchanged(run["inputs"])
    args.output.mkdir(parents=True, exist_ok=False)
    plots = figures(run, args.output)
    path = args.output / "probe_speed_reduced.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(run["probes"], stream, separators=(",", ":"), allow_nan=False)
    for probe in run["probes"]:
        del probe["series"], probe["ik_samples"], probe["guard_observations"]
    row = run.pop("row")
    run["case"] = {key: row[key] for key in ("path", "kind", "seed", "success", "held_lift", "failure", "attempts")}
    evidence.unchanged(run["inputs"])
    if source_bindings() != sources:
        raise ValueError("Diagnostic source changed during analysis")
    receipt = {"format": FORMAT, "scope": SCOPE, "analysis_source_sha256": sources,
        "part_of_runtime_frozen_manifest": False, "independent_postrun_analysis": True,
        "inputs_unchanged_during_analysis": True, "run": run,
        "figures": plots, "reduced_probe": evidence.capture(path)}
    output = args.output / "diagnostics.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"diagnostics": str(output), "sha256": evidence.digest(output), "figures": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
