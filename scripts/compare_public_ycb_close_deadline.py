#!/usr/bin/env python3
"""Stream a hash-bound R17/R18 first-CLOSE prefix and recorded tail diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


STREAMS = ("native", "grip_feedback", "preload", "load_transfer_probe", "bias_recovery", "joint_admittance")
WINDOWS = (.25, 1., 2., 4.)
TOLERANCE = 1e-10
EXPECTED_R17_SUMMARY_SHA256 = "e2883dce787d3c9140edc59c77e415394d904a7029f9f8e4ec5559b3946d61d8"
TIME_REFERENCES = {
    "native": [("scripts/evaluate_public_ycb.py", 85, "NativeAudit integrates before recording the state"),
        ("scripts/develop_public_ycb_force_band.py", 47, "RichNativeAudit adds integrated hand encoder qpos/qvel"),
        ("scripts/develop_public_ycb_bias_budget_r17.py", 128, "Inherited R17 sample copies cached actuation without forwarding")],
    "grip_feedback": [("src/kuka_sharpa_grasp/pick_place.py", 310, "hand.close/retain precede hand.write and mj_step at lines324/326"),
        ("scripts/develop_public_ycb_force_band.py", 214, "close/retain invoke regulate"),
        ("scripts/develop_public_ycb_force_band_r4.py", 278, "regulate appends pressures at current pre-step data.time")],
    "preload": [("scripts/develop_public_ycb_constrained_preload_r14.py", 244, "after_step updates preload after the inherited integrated observation")],
    "load_transfer_probe": [("scripts/develop_public_ycb_force_band_r8.py", 215, "after_step samples current pressures and appends the probe snapshot")],
    "bias_recovery": [("scripts/develop_public_ycb_bias_recovery_r16.py", 164, "write records the current pre-integration timestamp")],
    "joint_admittance": [("scripts/develop_public_ycb_bias_recovery_r16.py", 267, "write appends applied references before the next mj_step")],
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Evidence path escapes run directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink evidence is not accepted")
    if not path.is_file():
        raise ValueError("Missing regular evidence file: " + str(path))
    return path


def json_array(path):
    decoder, buffer, position, opened = json.JSONDecoder(), "", 0, False
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if not opened:
                    if position == len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("Expected a recorded JSON array")
                    opened, position = True, position + 1
                    continue
                if position == len(buffer):
                    break
                if buffer[position] == "]":
                    if (buffer[position + 1:] + stream.read()).strip():
                        raise ValueError("Trailing recorded JSON content")
                    return
                try:
                    row, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield row
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise ValueError("Incomplete recorded JSON array")
            buffer, position = buffer[position:] + chunk, 0


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite recorded number")
    return float(value)


def timestamp(row):
    return finite(row["timestamp"] if "timestamp" in row else row["timestamp_s"])


def tick(time, step):
    value = round(finite(time) / step)
    if value < 0 or abs(time - value * step) > 1e-8:
        raise ValueError("Observation is not on the recorded native grid")
    return value


class Differences:
    def __init__(self):
        self.fields = {}
        self.first = None
        self.mismatches = 0
        self.rows = 0

    def mismatch(self, path, time, reason):
        self.mismatches += 1
        if self.first is None:
            self.first = {"time_s": time, "field": path, "reason": reason}

    def compare(self, left, right, time, path="record"):
        if type(left) in (int, float) and type(right) in (int, float):
            delta = abs(finite(left) - finite(right))
            field = self.fields.setdefault(path, {"compared_scalars": 0, "nonzero_differences": 0, "maximum_absolute_difference": 0.})
            field["compared_scalars"] += 1
            field["nonzero_differences"] += delta != 0.
            field["maximum_absolute_difference"] = max(field["maximum_absolute_difference"], delta)
            if delta > TOLERANCE:
                self.mismatch(path, time, "numeric difference exceeds fixed absolute tolerance")
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                self.mismatch(path, time, "recorded field inventory differs")
            for key in sorted(set(left) & set(right)):
                self.compare(left[key], right[key], time, path + "." + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                self.mismatch(path, time, "recorded array length differs")
            for a, b in zip(left, right):
                self.compare(a, b, time, path + "[]")
        elif type(left) is not type(right) or left != right:
            self.mismatch(path, time, "recorded Boolean/string/null differs")

    def receipt(self):
        return {"passed": self.mismatches == 0, "compared_rows": self.rows,
            "mismatching_fields": self.mismatches, "first_difference": self.first,
            "numeric_fields": self.fields}


def statistics(rows, values):
    values = [finite(value) for value in values]
    if not values:
        return None
    times = [row["time_s"] for row in rows]
    center = sum(times) / len(times)
    mean = sum(values) / len(values)
    denominator = sum((time - center) ** 2 for time in times)
    return {"n": len(values), "first": values[0], "last": values[-1], "mean": mean,
        "minimum": min(values), "maximum": max(values),
        "ols_change_per_second": sum((time - center) * (value - mean)
            for time, value in zip(times, values)) / denominator if denominator else 0.}


def flag_summary(rows, predicate, step):
    best = current = count = 0
    previous = None
    for row in rows:
        index = tick(row["time_s"], step)
        passed = bool(predicate(row))
        current = (current + 1 if previous == index - 1 else 1) if passed else 0
        best, count, previous = max(best, current), count + passed, index
    return {"passed_samples": count, "maximum_consecutive_samples": best,
        "sample_interval_coverage_s": best * step,
        "elapsed_between_first_and_last_pass_s": max(0, best - 1) * step}


def compact(key, row):
    value = {"time_s": timestamp(row)}
    if key == "preload":
        original = row["original_probe_snapshot"]
        return dict(value, groups=row["current_groups"], reasons=row["reasons"],
            ready=row["ready"], continuous_seconds=row["continuous_seconds"],
            reset=row["window_reset_reason"], recovery=row.get("r16_recovery_inhibited", False),
            fixed_normal_1g=original["sampled_fixed_normal_1g_feasible"],
            original_reasons=original["preprobe_reasons"],
            wrist_speed_m_s=original["actual_wrist_speed_m_s"],
            eligible_seconds=original["probe_eligible_seconds"])
    if key == "load_transfer_probe":
        return dict(value, reasons=row["preprobe_reasons"], active=row["probe_active"],
            eligible_seconds=row["probe_eligible_seconds"], confirmed_seconds=row["probe_confirmed_seconds"])
    if key == "bias_recovery":
        joints, modes, rejected = {}, Counter(), 0
        for finger in row["per_finger"]:
            modes[finger["mode"]] += 1
            rejected += bool(finger.get("nominal_qp") and not finger["nominal_qp"]["accepted"])
            for name, joint in finger["joints"].items():
                joints[name] = {"added_bias_rad": joint["applied_command_rad"] - joint["before_actual_rad"] - joint["fixed_capture_bias_rad"],
                    "command_velocity_rad_s": (joint["applied_command_rad"] - joint["previous_reference_rad"]) / row["dt_s"],
                    "positive_bias_margin_rad": joint["bias_interval_rad"][1] - joint["applied_command_rad"],
                    "at_rate_boundary": min(abs(joint["applied_command_rad"] - bound)
                        for bound in joint["rate_interval_rad"]) <= 1e-9}
        return dict(value, recovery=row["recovery_active"], modes=dict(modes),
            rejected_qps=rejected, joints=joints)
    return value


class ObservedStream:
    def __init__(self, path, key, step):
        self.rows, self.key, self.step = iter(json_array(path)), key, step
        self.previous = -1
        self.count = 0
        self.segments = []
        self.in_close = False
        self.excluded = Counter()

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.rows)
        index = tick(timestamp(row), self.step)
        if index < self.previous:
            raise ValueError("Recorded stream is not ordered")
        self.count += 1
        if self.key == "native" and index != self.count:
            raise ValueError("Native records must start at dt and remain contiguous")
        if self.key in ("preload", "load_transfer_probe", "bias_recovery"):
            close = row.get("phase") == "CLOSE"
            if close:
                if not self.in_close or index - self.previous > 1:
                    self.segments.append({"first_sample_s": timestamp(row), "tail": deque()})
                tail = self.segments[-1]["tail"]
                item = compact(self.key, row)
                if tail and tick(tail[-1]["time_s"], self.step) == index:
                    tail[-1] = item
                else:
                    tail.append(item)
                while tail and timestamp(row) - tail[0]["time_s"] > max(WINDOWS) + 1e-8:
                    tail.popleft()
            self.in_close = close
        self.previous = index
        return row

    def prefix(self, deadline):
        for row in self:
            time = timestamp(row)
            write_time = self.key in ("grip_feedback", "bias_recovery", "joint_admittance")
            if time > deadline + 1e-8:
                return
            if write_time and abs(time - deadline) <= 1e-8:
                self.excluded["write_at_old_deadline_next_transition"] += 1
                continue
            decisive = ((self.key == "preload" and row.get("admission_recheck") is True)
                or (self.key == "load_transfer_probe" and ("admission" in row or row.get("decisive_current_solve") is True)))
            if decisive and abs(time - deadline) <= 1e-8:
                self.excluded["original_deadline_decisive_recheck"] += 1
                continue
            yield row

    def close(self):
        self.rows.close()


def tail_windows(stream):
    output = []
    for ordinal, segment in enumerate(stream.segments, 1):
        tail = list(segment["tail"])
        end = tail[-1]["time_s"]
        windows = []
        for seconds in WINDOWS:
            rows = [row for row in tail if row["time_s"] >= end - seconds - 1e-8]
            item = {"requested_seconds": seconds, "start_s": rows[0]["time_s"],
                "end_s": end, "unique_samples": len(rows)}
            if stream.key == "preload":
                groups = {}
                for key in sorted({key for row in rows for key in row["groups"]}):
                    selected = [row for row in rows if key in row["groups"]]
                    groups[key] = {name: statistics(selected, [row["groups"][key][name + "_n"] for row in selected])
                        for name in ("target", "measured")}
                    groups[key]["signed_error_n"] = statistics(selected,
                        [row["groups"][key]["target_n"] - row["groups"][key]["measured_n"] for row in selected])
                item.update(groups=groups, reasons=dict(Counter(reason for row in rows for reason in row["reasons"])),
                    wrist_speed_m_s=statistics(rows, [row["wrist_speed_m_s"] for row in rows]),
                    maximum_reported_preload_continuous_s=max(row["continuous_seconds"] for row in rows),
                    maximum_reported_original_eligible_s=max(row["eligible_seconds"] for row in rows),
                    gates={"all_preload": flag_summary(rows, lambda row: not row["reasons"] and not row["reset"], stream.step),
                        "fixed_normal_1g": flag_summary(rows, lambda row: row["fixed_normal_1g"], stream.step),
                        "original_preprobe": flag_summary(rows, lambda row: not row["original_reasons"], stream.step)})
            elif stream.key == "load_transfer_probe":
                item.update(reasons=dict(Counter(reason for row in rows for reason in row["reasons"])),
                    maximum_reported_eligible_s=max(row["eligible_seconds"] for row in rows),
                    maximum_reported_confirmed_s=max(row["confirmed_seconds"] for row in rows),
                    active_samples=sum(row["active"] for row in rows))
            else:
                item.update(recovery_samples=sum(row["recovery"] for row in rows),
                    modes=dict(sum((Counter(row["modes"]) for row in rows), Counter())),
                    rejected_qps=sum(row["rejected_qps"] for row in rows), joints={})
                for name in sorted({name for row in rows for name in row["joints"]}):
                    selected = [row for row in rows if name in row["joints"]]
                    item["joints"][name] = {key: statistics(selected, [row["joints"][name][key] for row in selected])
                        for key in ("added_bias_rad", "command_velocity_rad_s", "positive_bias_margin_rad")}
                    item["joints"][name]["rate_boundary_samples"] = sum(row["joints"][name]["at_rate_boundary"] for row in selected)
            windows.append(item)
        output.append({"attempt_ordinal": ordinal, "first_close_sample_s": segment["first_sample_s"], "windows": windows})
    return output


def load_run(directory):
    root = Path(directory).resolve()
    summary_path, protocol_path = safe_path(root, "summary.json"), safe_path(root, "protocol.json")
    captured = {}
    def capture(path, key, *, xml=False):
        data = path.read_bytes()
        captured[key] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return ET.fromstring(data) if xml else json.loads(data)
    summary, protocol = capture(summary_path, "summary"), capture(protocol_path, "protocol")
    if (not summary.get("complete") or summary["protocol"] != protocol or len(summary["rows"]) != 1
            or protocol["source_changed_during_run"] or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("Expected one completed case with matching, unchanged recorded protocol")
    row = summary["rows"][0]
    case = root / row["path"]
    metadata = safe_path(root, Path(row["path"]) / "metadata.json")
    scene = safe_path(root, Path(row["path"]) / "scene.xml")
    if row.get("runner_error") or capture(metadata, "metadata") != row:
        raise ValueError("Incomplete runner or mismatching case metadata")
    option = capture(scene, "scene", xml=True).find("option")
    if captured["scene"]["sha256"] != row["scene_sha256_before_load"] or row["scene_unchanged_since_load"] is not True:
        raise ValueError("Scene is not bound to its original loaded input")
    step = finite(float(option.attrib["timestep"])) if option is not None else 0.
    if step <= 0.:
        raise ValueError("An explicit positive scene timestep is required")
    files = {"summary": summary_path, "protocol": protocol_path, "metadata": metadata, "scene": scene}
    for key in STREAMS:
        artifact = row["artifacts"][key]
        path = safe_path(root, Path(row["path"]) / artifact["path"])
        bound = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        if bound["bytes"] != artifact["bytes"] or bound["sha256"] != artifact["sha256"]:
            raise ValueError("Recorded compressed artifact hash/size differs: " + key)
        files[key] = path
        captured[key] = bound
    return {"root": root, "row": row, "protocol": protocol, "step": step, "files": files,
        "bindings": captured}


def compare_runs(baseline, candidate, *, identity_check=False):
    script_sha = digest(__file__)
    left, right = load_run(baseline), load_run(candidate)
    if left["bindings"]["summary"]["sha256"] != EXPECTED_R17_SUMMARY_SHA256:
        raise ValueError("Baseline differs from the exact frozen R17 summary")
    if left["protocol"]["format"] != "public_ycb_bias_budget_development_r17":
        raise ValueError("The baseline must be the declared R17 experiment")
    expected = "public_ycb_bias_budget_development_r17" if identity_check else "public_ycb_close_deadline_development_r18"
    if right["protocol"]["format"] != expected:
        raise ValueError("Candidate protocol format differs from the requested comparison")
    if not identity_check:
        declaration = right["protocol"]["r18_close_deadline_experiment"]
        if declaration["r17_comparison_summary_sha256"] != left["bindings"]["summary"]["sha256"]:
            raise ValueError("R18 is bound to a different R17 comparison summary")
    for key in ("kind", "seed"):
        if left["row"][key] != right["row"][key]:
            raise ValueError("Case identity differs")
    if (left["step"] != right["step"] or left["bindings"]["scene"]["sha256"] != right["bindings"]["scene"]["sha256"]
            or left["protocol"]["joint_admittance"] != right["protocol"]["joint_admittance"]):
        raise ValueError("Scene, timestep or joint-control parameters differ")
    step, close_seconds = left["step"], finite(left["protocol"]["phase_seconds"]["CLOSE"])
    phases_before, phases_after = dict(left["protocol"]["phase_seconds"]), dict(right["protocol"]["phase_seconds"])
    extended_close = finite(phases_after.pop("CLOSE"))
    phases_before.pop("CLOSE")
    if phases_before != phases_after or (not identity_check and extended_close <= close_seconds):
        raise ValueError("Expected only an extended CLOSE phase deadline")
    tick(close_seconds, step)
    streams, deadline, first_close = {}, None, None
    for key in STREAMS:
        a, b = ObservedStream(left["files"][key], key, step), ObservedStream(right["files"][key], key, step)
        differences = Differences()
        try:
            if key == "native":
                for ra, rb in zip_longest(a, b):
                    time = timestamp(ra) if ra is not None else timestamp(rb)
                    if ra is None or rb is None:
                        differences.mismatch("native", time, "prefix record is missing")
                        break
                    if first_close is None and ra["state"] == "CLOSE":
                        first_close = timestamp(ra)
                        deadline = (tick(first_close, step) - 1) * step + close_seconds
                    differences.compare(ra, rb, time)
                    differences.rows += 1
                    if deadline is not None and timestamp(ra) >= deadline - 1e-8:
                        break
                if deadline is None or a.previous != tick(deadline, step):
                    raise ValueError("Baseline native prefix does not reach its declared CLOSE deadline")
            else:
                for ra, rb in zip_longest(a.prefix(deadline), b.prefix(deadline)):
                    time = timestamp(ra) if ra is not None else timestamp(rb)
                    if ra is None or rb is None:
                        differences.mismatch(key, time, "prefix record is missing")
                    else:
                        differences.compare(ra, rb, time)
                        differences.rows += 1
                if key in ("preload", "load_transfer_probe", "bias_recovery"):
                    for _ in a:
                        pass
                    for _ in b:
                        pass
            if differences.rows == 0:
                raise ValueError("Required recorded prefix is empty: " + key)
            streams[key] = {**differences.receipt(),
                "cutoff_semantics": "write_time_strictly_before_deadline" if key in ("grip_feedback", "bias_recovery", "joint_admittance") else "post_integration_inclusive_deadline",
                "excluded_decisive_rechecks": [dict(a.excluded), dict(b.excluded)]}
            if key in ("preload", "load_transfer_probe", "bias_recovery"):
                streams[key]["baseline_tail_windows"] = tail_windows(a)
                streams[key]["candidate_tail_windows"] = tail_windows(b)
        finally:
            a.close()
            b.close()
    for run in (left, right):
        for key, path in run["files"].items():
            bound = run["bindings"][key]
            if path.stat().st_size != bound["bytes"] or digest(path) != bound["sha256"]:
                raise ValueError("Comparison input changed during streaming: " + key)
    if digest(__file__) != script_sha:
        raise ValueError("Comparator source changed during analysis")
    passed = all(result["passed"] for result in streams.values())
    return {"format": "public_ycb_close_deadline_prefix_comparison_v1", "passed": passed,
        "identity_check_only": identity_check, "absolute_numeric_tolerance": TOLERANCE,
        "first_baseline_close_native_sample_s": first_close, "derived_phase_start_s": first_close - step,
        "baseline_close_seconds": close_seconds, "prefix_inclusive_deadline_s": deadline,
        "native_timestep_s": step, "baseline_inputs": left["bindings"], "candidate_inputs": right["bindings"],
        "time_semantics_code_references": {key: [{"path": path, "line": line, "reason": reason,
            "recorded_source_sha256": left["protocol"]["source_hashes_before"].get(path)}
            for path, line, reason in references] for key, references in TIME_REFERENCES.items()},
        "script_sha256": script_sha, "streams": streams,
        "scope": "Offline comparison only; no simulation, controller input or success claim. All recorded native fields are compared, including motor commands, target qpos/velocity, hand encoder qpos/qvel, contacts and cached actuation. Full robot qpos/qvel are not recorded and are not claimed checked. Numeric tolerance is fixed; Boolean/string/inventory differences fail. Native and post-integration pressure samples include the old deadline; write-time grip/bias/joint events stop strictly before it, since a write at the deadline belongs to the next transition after the old branch. Only explicit decisive/admission rechecks at the old deadline are excluded, with counts. Artifact hashes cover complete compressed inputs; native and grip prefix parsing stops after the cutoff. Preload/probe/bias tails are fully streamed with bounded four-second buffers per attempt; their window endpoints are each stream's final CLOSE observation (write-time bias ends one dt earlier). Isolated passing samples are not continuous admission. Independent physical, command and task audits remain required. No causal conclusion is inferred from supported shear direction."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-check", action="store_true", help="Explicit R17 self-comparison smoke, not R18 evidence")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Comparison output already exists")
    result = compare_runs(args.baseline, args.candidate, identity_check=args.identity_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "sha256": digest(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
