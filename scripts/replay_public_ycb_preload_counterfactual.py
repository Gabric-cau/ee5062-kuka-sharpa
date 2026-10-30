#!/usr/bin/env python3
"""Offline R14 preload replay dropping one pre-motion reason, without LP/physics."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

from compare_public_ycb_close_deadline import digest, json_array, safe_path

PROJECT = Path(__file__).resolve().parents[1]
STEP = .002
CONTINUOUS_SECONDS = .25
TARGET_DRIFT_N = .35
DROP_REASON = "CURRENT_MEASURED_FIXED_NORMAL_1G_INFEASIBLE"
R19_SUMMARY_SHA256 = "644fb8713e1924b80a577f9e2a409d44398c60e9d7b65cd07f2967cd25e649c8"
FROZEN_R14_SHA256 = "32f78b5c5a8d7ddb914d6d502d2c4656bafd77423523cdf80e6a791ff17360c6"


def binding(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite recorded scalar")
    return float(value)


def tick(time):
    time = finite(time)
    result = round(time / STEP)
    if abs(result * STEP - time) > 1e-8:
        raise ValueError("Preload timestamp is not on the recorded 2ms grid")
    return result


class Replay:
    def __init__(self):
        self.start = None
        self.anchor = None

    def reset(self):
        self.start = self.anchor = None

    def advance(self, row, drop_reason=None):
        reasons = [reason for reason in row["reasons"] if reason != drop_reason]
        groups, now, reset = row["current_groups"], finite(row["timestamp_s"]), None
        if self.anchor is not None:
            if set(groups) != set(self.anchor):
                reset = "GROUP_IDENTITY_CHANGED"
            elif max((abs(finite(group["target_n"]) - self.anchor[key])
                      for key, group in groups.items()), default=0.) > TARGET_DRIFT_N:
                reset = "TARGET_MOVED_FROM_WINDOW_ORIGIN"
        if reasons or reset:
            self.reset()
        if not reasons and self.start is None:
            self.start = now
            self.anchor = {key: finite(group["target_n"]) for key, group in groups.items()}
        duration = 0. if self.start is None else now - self.start
        return {"ready": bool(not reasons and duration >= CONTINUOUS_SECONDS - 1e-9),
            "continuous_seconds": duration, "window_reset_reason_recomputed": reset,
            "window_anchor_targets_n_recomputed": copy.deepcopy(self.anchor),
            "remaining_reasons": reasons}


def compact_row(row):
    groups = row["current_groups"]
    for group in groups.values():
        finite(group["target_n"])
        finite(group["measured_n"])
    if (type(row["ready"]) is not bool or not isinstance(row["reasons"], list)
            or any(not isinstance(reason, str) for reason in row["reasons"])):
        raise ValueError("Invalid preload readiness/reason schema")
    return {"timestamp_s": finite(row["timestamp_s"]), "phase": row["phase"],
        "admission_recheck": bool(row["admission_recheck"]), "reasons": row["reasons"],
        "current_groups": groups, "recorded_ready": row["ready"],
        "recorded_continuous_seconds": finite(row["continuous_seconds"]),
        "recorded_anchor_for_validation_only": row["window_anchor_targets_n"],
        "r16_recovery_inhibited": bool(row.get("r16_recovery_inhibited", False))}


def attempt_for_time(time, attempts):
    eligible = [int(attempt["attempt"]) for attempt in attempts
        if attempt.get("belief_timestamp") is not None and attempt["belief_timestamp"] <= time + 1e-8]
    if not eligible:
        raise ValueError("CLOSE observation has no preceding attempt belief")
    return max(eligible)


def analyze(directory, output, expected_summary_sha256):
    directory, output = directory.resolve(strict=True), output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new offline output directory")
    summary_path = safe_path(directory, "summary.json")
    if digest(summary_path) != expected_summary_sha256:
        raise ValueError("Finalized summary does not match the explicit expected hash")
    summary = json.loads(summary_path.read_text())
    protocol = summary["protocol"]
    if (not summary["complete"] or summary["runner_errors"] or len(summary["rows"]) != 1
            or protocol["format"] not in {"public_ycb_extended_budget_development_r19", "public_ycb_shared_normal_development_r20"}
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or protocol["constrained_preload"]["continuous_seconds"] != CONTINUOUS_SECONDS
            or protocol["constrained_preload"]["window_target_change_limit_n"] != TARGET_DRIFT_N
            or protocol["source_hashes_before"]["scripts/develop_public_ycb_constrained_preload_r14.py"] != FROZEN_R14_SHA256):
        raise ValueError("Expected a completed unchanged R19/R20 case using the frozen R14 anchor rules")
    case = summary["rows"][0]
    if case["kind"] != "ycb_can" or case["seed"] != 3414:
        raise ValueError("This replay is scoped to the same-can seed3414 development case")
    metadata = safe_path(directory, Path(case["path"]) / "metadata.json")
    protocol_path = safe_path(directory, "protocol.json")
    if json.loads(metadata.read_text()) != case or json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Finalized summary/protocol/metadata disagree")
    archive_path = safe_path(directory, "frozen_source.zip")
    manifest = protocol["source_hashes_before"]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError("Frozen runtime source ZIP membership differs")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != sha for name, sha in manifest.items()):
            raise ValueError("Frozen runtime source hash mismatch")
    parser_path = PROJECT / "scripts/compare_public_ycb_close_deadline.py"
    if digest(parser_path) != manifest[parser_path.relative_to(PROJECT).as_posix()]:
        raise ValueError("Imported stream parser differs from the frozen source")
    artifact = case["artifacts"]["preload"]
    preload_path = safe_path(directory, Path(case["path"]) / artifact["path"])
    if binding(preload_path) != {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}:
        raise ValueError("Preload raw artifact hash/size mismatch")
    inputs = {str(path): binding(path) for path in
        (summary_path, protocol_path, metadata, archive_path, preload_path, parser_path, Path(__file__))}
    output.mkdir(parents=True)
    compact_path = output / "compact_preload_inputs.jsonl.gz"
    series_path = output / "unique_clock_replay.jsonl.gz"
    no_drop, counterfactual = Replay(), Replay()
    attempts, mismatch_examples = {}, []
    validation = {"raw_rows": 0, "unique_clock_observations": 0, "admission_recheck_rows": 0,
        "same_clock_extra_rows": 0, "ready_mismatches": 0, "duration_mismatches_over_2ms": 0,
        "maximum_duration_error_s": 0., "anchor_mismatches": 0}
    previous_tick = previous_attempt = previous_phase = None
    pending = []

    def consume(bucket, compact_stream, series_stream):
        nonlocal previous_tick, previous_attempt, previous_phase
        index = tick(bucket[0]["timestamp_s"])
        attempt = attempt_for_time(bucket[0]["timestamp_s"], case["attempts"])
        phase = bucket[0]["phase"]
        if phase != "CLOSE" or any(row["phase"] != phase for row in bucket):
            raise ValueError("Replay only accepts recorded CLOSE observations")
        new_epoch = previous_tick is None or attempt != previous_attempt or phase != previous_phase
        if previous_tick is not None and index - previous_tick != 1:
            if attempt == previous_attempt and phase == previous_phase:
                raise ValueError("Unexplained missing native CLOSE observations; continuity is unknown")
            new_epoch = True
        if new_epoch:
            no_drop.reset()
            counterfactual.reset()
        stats = attempts.setdefault(attempt, {"attempt": attempt, "first_observation_s": bucket[0]["timestamp_s"],
            "last_observation_s": None, "unique_close_observations": 0, "raw_rows": 0,
            "maximum_original_continuous_seconds": 0., "maximum_counterfactual_continuous_seconds": 0.,
            "first_hypothetical_ready": None, "counterfactual_ready_unique_clocks": 0,
            "remaining_reason_unique_clock_counts": Counter(), "recomputed_anchor_reset_unique_clock_counts": Counter(),
            "only_dropped_reason_unique_clocks": 0, "no_remaining_reason_unique_clocks": 0})
        stats["last_observation_s"] = bucket[-1]["timestamp_s"]
        stats["unique_close_observations"] += 1
        stats["raw_rows"] += len(bucket)
        union, resets, any_ready, final = set(), set(), False, None
        validation["unique_clock_observations"] += 1
        validation["same_clock_extra_rows"] += len(bucket) - 1
        for ordinal, row in enumerate(bucket):
            validation["raw_rows"] += 1
            validation["admission_recheck_rows"] += row["admission_recheck"]
            compact_stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            observed = no_drop.advance(row)
            alternate = counterfactual.advance(row, DROP_REASON)
            error = abs(observed["continuous_seconds"] - row["recorded_continuous_seconds"])
            ready_mismatch = observed["ready"] != row["recorded_ready"]
            duration_mismatch = error > STEP + 1e-9
            anchor_mismatch = observed["window_anchor_targets_n_recomputed"] != row["recorded_anchor_for_validation_only"]
            validation["ready_mismatches"] += ready_mismatch
            validation["duration_mismatches_over_2ms"] += duration_mismatch
            validation["anchor_mismatches"] += anchor_mismatch
            validation["maximum_duration_error_s"] = max(validation["maximum_duration_error_s"], error)
            if (ready_mismatch or duration_mismatch or anchor_mismatch) and len(mismatch_examples) < 10:
                mismatch_examples.append({"timestamp_s": row["timestamp_s"], "same_clock_ordinal": ordinal,
                    "recorded_ready": row["recorded_ready"], "replayed": observed,
                    "recorded_continuous_seconds": row["recorded_continuous_seconds"], "anchor_mismatch": anchor_mismatch})
            stats["maximum_original_continuous_seconds"] = max(stats["maximum_original_continuous_seconds"], observed["continuous_seconds"])
            stats["maximum_counterfactual_continuous_seconds"] = max(stats["maximum_counterfactual_continuous_seconds"], alternate["continuous_seconds"])
            union.update(alternate["remaining_reasons"])
            if alternate["window_reset_reason_recomputed"]:
                resets.add(alternate["window_reset_reason_recomputed"])
            any_ready |= alternate["ready"]
            if alternate["ready"] and stats["first_hypothetical_ready"] is None:
                stats["first_hypothetical_ready"] = {"timestamp_s": row["timestamp_s"],
                    "admission_recheck": row["admission_recheck"], "same_clock_ordinal": ordinal,
                    "continuous_seconds": alternate["continuous_seconds"]}
            final = alternate
        stats["counterfactual_ready_unique_clocks"] += any_ready
        stats["remaining_reason_unique_clock_counts"].update(union)
        stats["recomputed_anchor_reset_unique_clock_counts"].update(resets)
        stats["only_dropped_reason_unique_clocks"] += bool(not union and any(DROP_REASON in row["reasons"] for row in bucket))
        stats["no_remaining_reason_unique_clocks"] += not union
        series_stream.write(json.dumps({"timestamp_s": bucket[-1]["timestamp_s"], "attempt": attempt,
            "same_clock_observations": len(bucket), "new_close_epoch": new_epoch,
            "any_hypothetical_ready_this_clock": any_ready, "remaining_reasons_union": sorted(union),
            "recomputed_anchor_resets_this_clock": sorted(resets), "final_counterfactual": final},
            separators=(",", ":"), allow_nan=False) + "\n")
        previous_tick, previous_attempt, previous_phase = index, attempt, phase

    with gzip.open(compact_path, "xt", encoding="utf-8") as compact_stream, gzip.open(series_path, "xt", encoding="utf-8") as series_stream:
        for full in json_array(preload_path):
            row = compact_row(full)
            index = tick(row["timestamp_s"])
            if pending and index < tick(pending[-1]["timestamp_s"]):
                raise ValueError("Preload stream is not chronological")
            if pending and index != tick(pending[-1]["timestamp_s"]):
                consume(pending, compact_stream, series_stream)
                pending = []
            pending.append(row)
        if pending:
            consume(pending, compact_stream, series_stream)
    unchanged = all(binding(Path(path)) == bound for path, bound in inputs.items())
    validation["passed"] = bool(unchanged and not any(validation[key] for key in
        ("ready_mismatches", "duration_mismatches_over_2ms", "anchor_mismatches")))
    result = {"format": "offline_preload_single_reason_counterfactual_v1", "inputs_before": inputs,
        "inputs_after": {path: binding(Path(path)) for path in inputs}, "inputs_unchanged": unchanged,
        "original_runtime_source_hashes_before": manifest,
        "original_runtime_source_hashes_after": protocol["source_hashes_after"], "source_archive_verified": True,
        "dropped_reason_only": DROP_REASON, "continuous_seconds": CONTINUOUS_SECONDS,
        "target_anchor_drift_limit_n": TARGET_DRIFT_N, "no_drop_validation": validation,
        "no_drop_mismatch_examples": mismatch_examples, "attempts": list(attempts.values()),
        "preload_decompression_passes": 1,
        "same_clock_policy": "One chronological physical timestamp bucket; all recorded admission rechecks are applied in original order at zero elapsed time. Counts and reason frequencies use unique timestamps, not row counts.",
        "output_artifacts": {path.name: binding(path) for path in (compact_path, series_path)},
        "physics_rerun": False, "lp_resolved": False, "controller_changed": False,
        "scope": "Observed-prefix anchor/admission replay only. All other recorded gates and safety/recovery reasons are retained. A first hypothetical-ready time is a potential branch point, not a predicted admission, subsequent motion, measured retention or strict success. No-ready rules out this one reason-only change on the observed prefix, not all future policy changes."}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "no_drop_validation": validation,
        "attempts": [{key: row[key] for key in ("attempt", "maximum_counterfactual_continuous_seconds", "first_hypothetical_ready")}
            for row in attempts.values()]}))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", default=R19_SUMMARY_SHA256,
        help="R19 is pinned by default; a later finalized R20 requires its explicit summary hash")
    args = parser.parse_args()
    raise SystemExit(analyze(args.directory, args.output, args.expected_summary_sha256))
