#!/usr/bin/env python3
"""Independently reconstruct R16 command writes from adjacent native records."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


STEP_S = .002
BIAS_LIMIT_RAD = .02
TOLERANCE_RAD = 1e-10
ACTIVE_FINGERS = ("thumb", "middle", "ring")
EXCLUDED_PHASES = {"RELEASE", "VERIFY", "RECOVER"}
FROZEN_R14_SHA256 = "32f78b5c5a8d7ddb914d6d502d2c4656bafd77423523cdf80e6a791ff17360c6"
FROZEN_PROFILE_SHA256 = "afcbc124960136c95b1f1269dbe8dfe85974061513c5a722dc56f714da202b57"
R16_SOURCE = "scripts/develop_public_ycb_bias_recovery_r16.py"
R15_SUMMARY = "results/public_ycb_fixed_geometry_20261022_r15_seed3414/summary.json"
FROZEN_R15_SUMMARY_SHA256 = "0d1120d582e1b9b47b70ef92626a845677ec7044308710ae23510e264ad8f4a0"


@dataclass(frozen=True)
class JointSpec:
    name: str
    finger: str
    actuator: int
    lower: float
    upper: float
    ratio: float


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {label}") from error
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite {label}")
    return number


def flag(record, key):
    value = record[key]
    if type(value) is not bool:
        raise ValueError(f"Expected Boolean {key}")
    return value


def equal(actual, expected, label, tolerance=TOLERANCE_RAD):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (left.shape != right.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all() or np.any(np.abs(left - right) > tolerance)):
        raise ValueError(f"{label} differs from independently reconstructed value")


def tick(timestamp):
    time = finite(timestamp, "timestamp")
    index = round(time / STEP_S)
    if index < 0 or abs(time - index * STEP_S) > 1e-8:
        raise ValueError("Timestamp is not on the native 2ms grid")
    return index


def joint_specs(model, profile):
    specs = []
    for finger in ACTIVE_FINGERS:
        if profile["contact_closure"][finger] <= 0.:
            raise ValueError("R16 requires all three declared active fingers")
        ratios = (("MCP_FE", profile["thumb_mcp_flexion_ratio"]),
                  ("IP", profile["thumb_ip_flexion_ratio"])) if finger == "thumb" else (
            ("MCP_FE", profile.get("nonthumb_mcp_ratio", 1.)),
            ("PIP", profile.get("nonthumb_pip_ratio", 1.4)),
            ("DIP", profile.get("nonthumb_dip_ratio", .8)))
        for suffix, ratio in ratios:
            name = f"left_{finger}_{suffix}"
            joint = model.joint(name).id
            actuators = np.flatnonzero((model.actuator_trntype == 0)
                & (model.actuator_trnid[:, 0] == joint))
            if len(actuators) != 1:
                raise ValueError("Expected exactly one position actuator per audited joint")
            specs.append(JointSpec(name, finger, int(actuators[0]),
                *map(float, model.jnt_range[joint]), abs(finite(ratio, "joint ratio"))))
    return specs


def _native_index(native, specs):
    if not native:
        raise ValueError("Missing native records")
    indexed = {}
    for expected, sample in enumerate(native, 1):
        index = tick(sample["timestamp"])
        if index != expected:
            raise ValueError("Native records must start at 2ms and remain contiguous")
        commands = np.asarray(sample["motor_commands"], dtype=float)
        if commands.ndim != 1 or not np.isfinite(commands).all():
            raise ValueError("Invalid native actuator commands")
        for spec in specs:
            if spec.actuator >= len(commands):
                raise ValueError("Missing native actuator command")
            finite(sample["actual_hand_joints"][spec.name]["position_rad"], "native encoder")
        indexed[index] = sample
    return indexed


def _captures(joint_events, indexed, specs):
    names = {spec.name for spec in specs}
    captures = {}
    for event in joint_events:
        if event.get("event") != "ACTIVATION_CAPTURE":
            continue
        index = tick(event["timestamp_s"])
        if index not in indexed or index in captures:
            raise ValueError("Capture lacks a unique contemporaneous native state")
        if (set(event["captured_command_rad"]) != names
                or set(event["captured_command_minus_actual_rad"]) != names):
            raise ValueError("Incomplete activation capture")
        state = indexed[index]
        values = {}
        for spec in specs:
            command = finite(state["motor_commands"][spec.actuator], "capture command")
            actual = finite(state["actual_hand_joints"][spec.name]["position_rad"], "capture encoder")
            equal(event["captured_command_rad"][spec.name], command, "Captured command")
            equal(event["captured_command_minus_actual_rad"][spec.name], command - actual,
                "Captured fixed bias")
            values[spec.name] = command - actual
        captures[index] = values
    return captures


def _empty_gate(snapshot, label, *, allow_active_probe=False):
    if (flag(snapshot, "preload_ready") or snapshot["preload_start_s"] is not None
            or abs(finite(snapshot["probe_eligible_seconds"], label)) > 1e-12
            or abs(finite(snapshot["probe_confirmed_seconds"], label)) > 1e-12
            or snapshot["pressure_window_samples"] != 0
            or (flag(snapshot, "probe_active") and not allow_active_probe)):
        raise ValueError(f"Recovery must reset preload/eligibility and forbid probe: {label}")


def _admission_snapshot(snapshot):
    for key in ("preload_ready", "probe_active", "payload_confirmed"):
        flag(snapshot, key)
    start = snapshot["preload_start_s"]
    if start is not None:
        finite(start, "preload start")
    for key in ("probe_eligible_seconds", "probe_confirmed_seconds"):
        if finite(snapshot[key], key) < 0.:
            raise ValueError("Negative admission duration")
    count = snapshot["pressure_window_samples"]
    if type(count) is not int or count < 0:
        raise ValueError("Invalid pressure-window sample count")


def reconstruct_case(native, events, joint_events, specs, *, close_rate, relief_rate,
                     preload_events=(), probe_events=(), native_audit=None):
    """Audit fixed capture/ranges/rates/bias without trusting controller step checks."""
    specs = list(specs)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Missing or duplicate joint specification")
    close_rate, relief_rate = finite(close_rate, "closing rate"), finite(relief_rate, "relief rate")
    if close_rate <= 0. or relief_rate <= 0.:
        raise ValueError("Positive frozen rates required")
    indexed = _native_index(native, specs)
    captures = _captures(joint_events, indexed, specs)
    legacy_writes, safety_writes = {}, {}
    for event in joint_events:
        if event.get("event") in {"NOMINAL_REFERENCE_WRITE", "SAFETY_PRECEDENCE"}:
            index = tick(event["timestamp_s"])
            collection = safety_writes if event["event"] == "SAFETY_PRECEDENCE" else legacy_writes
            if index in collection:
                raise ValueError("Duplicate inherited final-write record")
            collection[index] = event
    if len(events) != len(native):
        raise ValueError("Expected exactly one final-write event per integrated native step")
    counts, peaks = Counter(), Counter()
    active_capture = None
    captured_this_close = False
    previous_phase = None
    inhibit_ticks = set()
    close_inhibit_ticks = set()
    pending_probe_ticks = set()
    explicit_infeasible = []
    failed_writes = []
    had_release = False
    for index, event in enumerate(events):
        if tick(event["timestamp_s"]) != index:
            raise ValueError("Write events must map to the next native sample without gaps")
        after = indexed[index + 1]
        if after["state"] != event["phase"]:
            raise ValueError("Write phase differs from its following native integration")
        phase = event["phase"]
        prior_phase = previous_phase
        had_release |= phase in EXCLUDED_PHASES
        if phase == "CLOSE" and previous_phase != "CLOSE":
            captured_this_close = False
        if index in captures:
            if phase != "CLOSE" or captured_this_close:
                raise ValueError("Loaded bias may be captured only once in each CLOSE epoch")
            active_capture, captured_this_close = index, True
            counts["activation_captures"] += 1
        previous_phase = phase
        dt = finite(event["dt_s"], "write interval")
        safety = flag(event, "final_safety_override")
        fresh = flag(event, "fresh_target")
        recovery = flag(event, "recovery_active")
        failure = flag(event, "failure_latched")
        inhibited = flag(event, "preload_inhibited")
        legacy = legacy_writes.get(index)
        if legacy is None:
            if not safety or index not in safety_writes:
                raise ValueError("Missing contemporaneous inherited final-write record")
            if safety_writes[index]["mode"] != event["mode"]:
                raise ValueError("Mode differs from inherited safety-precedence record")
        else:
            target_age = legacy["target_age_s"]
            legacy_fresh = (legacy["current_target"].get("feasible") is True
                and target_age is not None and abs(finite(target_age, "target age")) < 1e-9)
            if fresh != legacy_fresh:
                raise ValueError("Fresh-target declaration differs from inherited current target")
            if legacy["mode"] != event["mode"] or legacy["phase"] != phase:
                raise ValueError("Mode/phase differs from inherited final write")
            if flag(legacy, "safety_override_at_final_write") != safety:
                raise ValueError("Safety declaration differs from inherited final writer")
        fingers = event["per_finger"]
        if isinstance(fingers, list):
            if len({item["finger"] for item in fingers}) != len(fingers):
                raise ValueError("Duplicate finger telemetry")
            fingers = {item["finger"]: item for item in fingers}
        # The frozen task creates a fresh hand controller after each recovery.
        # Its first SEARCH write has no earlier controller-local timestamp.
        attempt_reset = index > 0 and abs(dt) < 1e-12
        if attempt_reset:
            if (prior_phase != "RECOVER" or phase != "SEARCH"
                    or event["mode"] != "SCALAR_ACQUISITION" or fingers
                    or safety or fresh or recovery or failure or inhibited or legacy is None
                    or legacy["current_target"] != {"feasible": False, "reason": "NOT_EVALUATED"}
                    or legacy["target_age_s"] is not None
                    or any(legacy[key] for key in ("command_rad", "candidate_command_rad",
                        "applied_command_rad", "joint_reference_steps"))):
                raise ValueError("Zero local dt requires the frozen task's new-attempt controller reset")
            equal(legacy["dt_s"], 0., "New-attempt legacy local dt")
            for state in (indexed[index], after):
                equal([state["motor_commands"][spec.actuator] for spec in specs],
                    np.zeros(len(specs)), "New-attempt native zero-open command")
            active_capture, captured_this_close = None, False
            counts["controller_attempt_resets"] += 1
        equal(dt, 0. if index == 0 or attempt_reset else STEP_S, "Write dt", 1e-8)
        scalar = event["mode"] in {"SCALAR_ACQUISITION", "SCALAR_RECONCILING"}
        if scalar and (captured_this_close or phase not in {"SEARCH", "APPROACH", "ALIGN", "CLOSE"}):
            raise ValueError("Scalar passthrough cannot conceal a captured tracking epoch")
        open_hold = event["mode"] == "OPEN_HOLD" and phase not in EXCLUDED_PHASES
        if open_hold:
            if not had_release or phase not in {"SEARCH", "APPROACH", "ALIGN"}:
                raise ValueError("Open-hold exclusion requires a preceding release epoch")
            equal([after["motor_commands"][spec.actuator] for spec in specs],
                np.zeros(len(specs)), "Native zero open-hold command")
        excluded = safety or phase in EXCLUDED_PHASES or scalar or open_hold
        if set(fingers) != set(ACTIVE_FINGERS) and not (excluded and not fingers):
            raise ValueError("Incomplete per-finger coverage")
        if excluded:
            reason = ("emergency" if safety else ("scalar_acquisition" if scalar else
                ("zero_open_hold" if open_hold else "release")))
            counts[f"excluded_{reason}_writes"] += 1
        equal(event["post_step_timestamp_s"], (index + 1) * STEP_S, "Post-step timestamp", 1e-8)
        original_failure = event["original_post_step_failure"]
        returned_failure = event["post_step_returned_failure"]
        abort_reason, resolved_abort = event["abort_reason"], event["resolved_abort_reason"]
        abort_applied = flag(event, "probe_abort_applied")
        for value in (original_failure, returned_failure, abort_reason, resolved_abort):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Invalid optional failure/abort reason")
        if abort_applied != (resolved_abort is not None):
            raise ValueError("Resolved abort reason lacks an actual R16 abort")
        if abort_applied and resolved_abort != (original_failure or abort_reason):
            raise ValueError("Resolved abort violates original-safety/queued-reason precedence")
        for key in ("admission_before", "admission_after_write", "admission_after_step"):
            _admission_snapshot(event[key])
        if original_failure and returned_failure != original_failure:
            raise ValueError("R16 replaced the original post-step safety failure")
        if failure and phase not in EXCLUDED_PHASES and not returned_failure:
            raise ValueError("Latched failure did not return a controlled-release failure")
        if recovery or failure or (active_capture is not None and not fresh and not excluded):
            if not inhibited:
                raise ValueError("Recovery or invalid target must inhibit preload")
            was_probe = flag(event["admission_before"], "probe_active")
            was_held = flag(event["admission_before"], "payload_confirmed")
            pending = bool(abort_reason) and was_probe
            if (was_probe or was_held) and not abort_reason:
                raise ValueError("Active probe/held recovery lacks a queued abort reason")
            _empty_gate(event["admission_after_write"], "final write", allow_active_probe=pending)
            _empty_gate(event["admission_after_step"], "after integration")
            if flag(event["admission_after_step"], "payload_confirmed"):
                raise ValueError("Recovery retained a confirmed held payload")
            if (was_probe or was_held) and not original_failure:
                if (not abort_applied or resolved_abort != abort_reason
                        or returned_failure != abort_reason):
                    raise ValueError("Queued probe/held recovery was not actually aborted after original safety")
            if pending:
                pending_probe_ticks.add(index + 1)
            inhibit_ticks.add(index + 1)
            if phase == "CLOSE":
                close_inhibit_ticks.add(index + 1)
        before = indexed.get(index)
        if index == 0 and (active_capture is not None or recovery or failure):
            raise ValueError("Initial passthrough cannot use a future native sample or capture")
        saw_recovery, saw_infeasible = False, False
        any_empty_box = False
        reference_mismatches = []
        final_violations = []
        if before is not None and active_capture is not None and not excluded:
            for spec in specs:
                actual = finite(before["actual_hand_joints"][spec.name]["position_rad"], "native encoder")
                previous = finite(before["motor_commands"][spec.actuator], "previous command")
                bias = captures[active_capture][spec.name]
                low = max(spec.lower, previous - relief_rate * spec.ratio * dt,
                    actual + bias - BIAS_LIMIT_RAD)
                high = min(spec.upper, previous + close_rate * spec.ratio * dt,
                    actual + bias + BIAS_LIMIT_RAD)
                any_empty_box |= low > high + TOLERANCE_RAD
            if any_empty_box and (event["mode"] != "BIAS_INFEASIBLE" or not failure):
                raise ValueError("Empty original box must produce a global infeasible failure")
            for finger_event in fingers.values():
                for name, detail in finger_event["joints"].items():
                    spec = next(item for item in specs if item.name == name)
                    previous = before["motor_commands"][spec.actuator]
                    if abs(finite(detail["previous_reference_rad"], "logged reference") - previous) > TOLERANCE_RAD:
                        reference_mismatches.append(name)
        for finger, finger_event in fingers.items():
            mode = finger_event["mode"]
            selected = [spec for spec in specs if spec.finger == finger]
            expected_names = {spec.name for spec in selected}
            names = set(finger_event["joint_names"])
            joints = finger_event["joints"]
            if names != set(joints):
                raise ValueError("Joint names and joint telemetry disagree")
            if mode == "PASSTHROUGH":
                if active_capture is not None and not excluded and phase not in {"SEARCH", "APPROACH", "ALIGN"}:
                    raise ValueError("Captured nominal/recovery/invalid-target writes cannot be excluded")
                counts["passthrough_finger_writes"] += 1
                continue
            if names != expected_names or active_capture is None or before is None:
                raise ValueError("Active write requires full joints and a contemporaneous fixed capture")
            if mode not in {"NOMINAL", "BIAS_RECOVERY", "HOLD_INVALID", "INFEASIBLE", "FAIL_PENDING"}:
                raise ValueError(f"Unknown finger mode: {mode}")
            if excluded:
                counts["excluded_joint_writes"] += len(selected)
                continue
            if mode == "NOMINAL" and not fresh:
                raise ValueError("Invalid target cannot be labelled nominal")
            if mode in {"BIAS_RECOVERY", "HOLD_INVALID"}:
                saw_recovery = True
                if finger_event["nominal_direction_satisfied"] is not None:
                    raise ValueError("Recovery cannot claim nominal direction compliance")
            if mode in {"INFEASIBLE", "FAIL_PENDING"} and not (any_empty_box or reference_mismatches):
                raise ValueError("Failure exclusion without independently observed empty box or reference mismatch")
            finger_has_empty = False
            for spec in selected:
                detail = joints[spec.name]
                actual = finite(before["actual_hand_joints"][spec.name]["position_rad"], "native encoder")
                previous = finite(before["motor_commands"][spec.actuator], "previous native command")
                applied = finite(after["motor_commands"][spec.actuator], "next native command")
                bias = captures[active_capture][spec.name]
                reference = finite(detail["previous_reference_rad"], "logged reference")
                rate = [reference - relief_rate * spec.ratio * dt,
                        reference + close_rate * spec.ratio * dt]
                window = [actual + bias - BIAS_LIMIT_RAD, actual + bias + BIAS_LIMIT_RAD]
                intersection = [max(spec.lower, rate[0], window[0]),
                                min(spec.upper, rate[1], window[1])]
                for key, expected in (("before_actual_rad", actual), ("before_command_rad", previous),
                        ("fixed_capture_bias_rad", bias),
                        ("capture_timestamp_s", active_capture * STEP_S),
                        ("range_rad", [spec.lower, spec.upper]), ("rate_interval_rad", rate),
                        ("bias_interval_rad", window), ("intersection_rad", intersection),
                        ("applied_command_rad", applied)):
                    equal(detail[key], expected, key)
                counts[f"{mode.lower()}_joint_writes"] += 1
                added = abs(applied - actual - bias)
                peaks["absolute_added_bias_rad"] = max(peaks["absolute_added_bias_rad"], added)
                if any_empty_box or reference_mismatches:
                    if mode not in {"INFEASIBLE", "FAIL_PENDING"} or not failure or not inhibited:
                        raise ValueError("Empty original intersection must latch an explicit failure")
                    equal(detail["candidate_command_rad"], reference, "Infeasible candidate hold")
                    equal(applied, reference, "Infeasible command hold")
                    if intersection[0] > intersection[1] + TOLERANCE_RAD:
                        finger_has_empty = saw_infeasible = True
                        explicit_infeasible.append({"timestamp_s": index * STEP_S, "joint": spec.name,
                            "intersection_rad": intersection, "absolute_added_bias_rad": added})
                    continue
                outside = applied < intersection[0] - TOLERANCE_RAD or applied > intersection[1] + TOLERANCE_RAD
                candidate = finite(detail["candidate_command_rad"], "candidate command")
                if outside:
                    final_violations.append(spec.name + ":APPLIED_OUTSIDE_ORIGINAL_BOX")
                if abs(candidate - applied) > TOLERANCE_RAD:
                    final_violations.append(spec.name + ":APPLIED_CANDIDATE_MISMATCH")
                if mode in {"BIAS_RECOVERY", "HOLD_INVALID"}:
                    projected = float(np.clip(previous, *intersection))
                    equal(candidate, projected, "Recovery minimum-change projection")
            if (any_empty_box or reference_mismatches) and (mode == "INFEASIBLE") != finger_has_empty:
                raise ValueError("Per-finger infeasible mode disagrees with native box")
            if mode == "NOMINAL":
                matrix = np.asarray(finger_event["compression_jacobian_m_rad"], dtype=float)
                desired = np.asarray(finger_event["desired_normal_velocity_m_s"], dtype=float)
                if (matrix.ndim != 2 or matrix.shape[1] != len(selected)
                        or desired.shape != (matrix.shape[0],)
                        or not np.isfinite(matrix).all() or not np.isfinite(desired).all()):
                    raise ValueError("Invalid recorded nominal direction inputs")
                by_name = {spec.name: spec for spec in selected}
                names = finger_event["joint_names"]
                velocity = np.array([(after["motor_commands"][by_name[name].actuator]
                    - before["motor_commands"][by_name[name].actuator]) / dt for name in names])
                predicted = matrix @ velocity
                equal(finger_event["predicted_applied_normal_velocity_m_s"], predicted,
                    "Native-command direction prediction")
                mask = np.abs(desired) > 1e-12
                direction_ok = bool(np.all(np.sign(desired[mask]) * predicted[mask] >= -1e-10))
                if flag(finger_event, "nominal_direction_satisfied") != direction_ok:
                    raise ValueError("Nominal direction diagnostic contradicts native commands")
                if not direction_ok:
                    final_violations.append(finger + ":APPLIED_NOMINAL_DIRECTION_REVERSED")
        if not excluded and recovery != saw_recovery:
            raise ValueError("Global recovery flag disagrees with per-finger modes")
        if not excluded and saw_infeasible and not failure:
            raise ValueError("Infeasible finger without latched failure")
        if not excluded and not failure:
            if event["mode"] != ("BIAS_RECOVERY" if saw_recovery else "JOINT_CONTROL"):
                raise ValueError("Global mode disagrees with independently audited finger modes")
        if not excluded:
            if sorted(event["write_verification_violations"]) != sorted(final_violations):
                raise ValueError("Write-verification declarations disagree with native command violations")
            actual_failure = bool(any_empty_box or reference_mismatches or final_violations)
            if actual_failure and (not failure or not inhibited or event["mode"] != "BIAS_INFEASIBLE"):
                raise ValueError("Native command violates range/rate/fixed-capture bounds without an explicit failure")
            if failure and not actual_failure:
                raise ValueError("Latched failure lacks independently observed native evidence")
            if flag(event, "write_verification_passed") == bool(actual_failure):
                raise ValueError("Write-verification status misrepresents a failed command")
            if actual_failure:
                failed_writes.append({"timestamp_s": index * STEP_S,
                    "reference_native_mismatches": reference_mismatches,
                    "final_write_violations": final_violations,
                    "empty_original_box": any_empty_box})
        counts["write_events"] += 1
    if not captures or counts["activation_captures"] != len(captures):
        raise ValueError("No complete audited capture, or capture outside integrated writes")
    seen_preload, seen_probe = set(), set()
    for label, stream in (("preload", preload_events), ("probe", probe_events)):
        for event in stream:
            if tick(event["timestamp_s"]) not in inhibit_ticks:
                continue
            if label == "preload":
                if flag(event, "ready") or abs(finite(event["continuous_seconds"], "preload duration")) > 1e-12:
                    raise ValueError("Recovery retained or admitted preload in independent preload log")
                counts["inhibited_preload_observations"] += 1
                seen_preload.add(tick(event["timestamp_s"]))
            else:
                if (event.get("admission") is True or event.get("preload_admission") is True
                        or (flag(event, "probe_active") and tick(event["timestamp_s"]) not in pending_probe_ticks)):
                    raise ValueError("Recovery admitted a probe in independent probe log")
                seen_probe.add(tick(event["timestamp_s"]))
    if not close_inhibit_ticks <= seen_preload or not inhibit_ticks <= seen_probe:
        raise ValueError("Missing contemporaneous recovery preload/probe observations")
    guard = {"independently_reconstructed": False,
        "scope": "Recorded frozen NativeAudit runtime guard; post-step states alone do not prove absence of external writes."}
    if native_audit is None:
        raise ValueError("Missing recorded frozen native runtime guard")
    if native_audit is not None:
        if (native_audit["integration_steps"] != len(native)
                or native_audit["native_samples"] != len(native)
                or native_audit["external_target_qpos_writes"] != 0 or native_audit["write_events"]):
            raise ValueError("Recorded native integration/qpos-write guard is incomplete or violated")
        initial = np.asarray(native_audit["initial_target_qpos"], dtype=float)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("Invalid recorded initial target pose")
        guard.update(recorded_guard_satisfied=True, initial_target_qpos=initial.tolist())
    return {"accepted_evidence_consistency": True, "counts": dict(counts), "maxima": dict(peaks),
        "original_intersection_satisfied_for_all_audited_writes": not failed_writes,
        "explicit_infeasible_writes": explicit_infeasible, "failed_writes": failed_writes,
        "native_samples": len(native),
        "additional_bias_limit_rad": BIAS_LIMIT_RAD, "native_timestep_s": STEP_S,
        "target_qpos_guard": guard,
        "scope": "Nominal/recovery/invalid-target commands reconstructed from previous and next native ctrl and same-time actual encoders. Emergency, release and scalar acquisition are counted exclusions and do not establish .02rad biased bounds. This does not prove contact force, QP optimality, grasp success or no external target writes."}


def checked_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe evidence-relative path")
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlink in evidence input")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its root")
    return resolved


def checked_stream(case, record):
    path = checked_path(case, record["path"])
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("Raw artifact size/hash differs from metadata")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise ValueError("Raw artifact must contain a nonempty record list")
    return values


def literal_assignment(source, name):
    matches = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen literal declaration: {name}")
    return ast.literal_eval(matches[0])


def frozen_inputs(directory, protocol):
    before, after = protocol["source_hashes_before"], protocol["source_hashes_after"]
    if not before or before != after or protocol["source_changed_during_run"] is not False:
        raise ValueError("Experiment source was incomplete or changed")
    if before["scripts/develop_public_ycb_constrained_preload_r14.py"] != FROZEN_R14_SHA256:
        raise ValueError("Original R14 source identity changed")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate frozen source archive entries")
        sources = {}
        for relative, expected in before.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe frozen source path")
            value = archive.read(relative)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError("Frozen source archive hash mismatch")
            sources[relative] = value
    if (before[R15_SUMMARY] != FROZEN_R15_SUMMARY_SHA256
            or protocol["r16_bias_recovery_experiment"]["r15_comparison_summary_sha256"] != FROZEN_R15_SUMMARY_SHA256):
        raise ValueError("R15 comparison identity changed")
    prior = json.loads(sources[R15_SUMMARY])
    inherited = prior["protocol"]["source_hashes_before"]
    if (prior["complete"] is not True or prior["protocol"]["source_changed_during_run"] is not False
            or inherited != prior["protocol"]["source_hashes_after"]
            or any(before.get(key) != value for key, value in inherited.items())):
        raise ValueError("R16 inherited source differs from the exact frozen R15 comparison")
    joint_parameters = literal_assignment(sources["scripts/develop_public_ycb_joint_admittance_r13.py"],
        "JOINT_PARAMETERS")
    if (joint_parameters["additional_bias_limit_rad"] != BIAS_LIMIT_RAD
            or protocol["joint_admittance"]["additional_bias_limit_rad"] != BIAS_LIMIT_RAD):
        raise ValueError("The original .02rad bias limit changed")
    rates = []
    for node in ast.walk(ast.parse(sources["scripts/develop_public_ycb_force_band.py"])):
        if isinstance(node, ast.Call):
            keywords = {item.arg: item.value for item in node.keywords}
            if all(isinstance(keywords.get(key), ast.Constant) for key in ("close_rate", "relief_rate")):
                rates.append({key: finite(keywords[key].value, key) for key in ("close_rate", "relief_rate")})
    if len(rates) != 1:
        raise ValueError("Expected exactly one original frozen acquisition-rate declaration")
    recovery = literal_assignment(sources[R16_SOURCE], "RECOVERY_PARAMETERS")
    if protocol["r16_bias_recovery_experiment"]["recovery_parameters"] != recovery:
        raise ValueError("Recovery declaration differs from frozen executed source")
    profiles = [json.loads(value) for key, value in sources.items()
                if key.endswith(".json") and before[key] == FROZEN_PROFILE_SHA256]
    if len(profiles) != 1 or profiles[0]["ycb_can"] != protocol["nominal_profiles"]["ycb_can"]:
        raise ValueError("Nominal geometry differs from the exact guarded R15 profile")
    target_assignments = []
    for node in ast.walk(ast.parse(sources[R16_SOURCE])):
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            if any(isinstance(part, ast.Attribute) and part.attr == "qpos" for part in ast.walk(target)):
                target_assignments.append(node.lineno)
    if target_assignments:
        raise ValueError("R16 source contains direct qpos assignment")
    return rates[0], {"source_archive_sha256": digest(directory / "frozen_source.zip"),
        "declared_source_files": len(before), "before_after_identical": True,
        "all_frozen_source_hashes_match": True, "r16_source_sha256": before[R16_SOURCE],
        "r15_comparison_summary_sha256": FROZEN_R15_SUMMARY_SHA256,
        "r16_direct_qpos_assignment_lines": target_assignments,
        "static_check_scope": "Direct qpos assignment syntax in the new R16 source only; not a proof about arbitrary imported code or dynamic writes."}


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path = checked_path(directory, "summary.json")
    protocol_path = checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or protocol["format"] != "public_ycb_bias_recovery_development_r16"):
        raise ValueError("Expected complete, consistently bound R16 summary/protocol")
    rates, source = frozen_inputs(directory, protocol)
    if len(summary["rows"]) != len(protocol["schedule"]) or not summary["rows"]:
        raise ValueError("Scheduled experiment denominator differs")
    cases, seen = [], set()
    for planned, row in zip(protocol["schedule"], summary["rows"], strict=True):
        if any(row[key] != value for key, value in planned.items()) or row["path"] in seen:
            raise ValueError("Duplicate or mismatching scheduled case")
        seen.add(row["path"])
        if row["runner_error"]:
            raise ValueError("Runner-error case cannot establish complete command evidence")
        case = checked_path(directory, row["path"])
        metadata = checked_path(case, "metadata.json")
        if json.loads(metadata.read_text()) != row:
            raise ValueError("Case metadata differs from its summary row")
        scene = checked_path(case, "scene.xml")
        if (row["scene_unchanged_since_load"] is not True
                or row["scene_sha256_before_load"] != digest(scene)
                or row["scene_sha256_at_finalize"] != digest(scene)):
            raise ValueError("Recorded scene does not bind the runtime-loaded joint model")
        model = load_evidence_model(scene)
        if abs(float(model.opt.timestep) - STEP_S) > 1e-12:
            raise ValueError("Native model must retain 2ms physics")
        records = {key: checked_stream(case, row["artifacts"][key]) for key in
            ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
        receipt = reconstruct_case(records["native"], records["bias_recovery"], records["joint_admittance"],
            joint_specs(model, protocol["nominal_profiles"][row["kind"]]), **rates,
            preload_events=records["preload"], probe_events=records["load_transfer_probe"],
            native_audit=row["native_audit"])
        receipt.update(case=row["path"], scene_sha256=digest(scene), metadata_sha256=digest(metadata),
            inputs={key: row["artifacts"][key] for key in records},
            task_success=row["success"], held_lift=row["held_lift"],
            task_outcome_scope="Reported original task outcomes, not independently rescored by this command auditor")
        cases.append(receipt)
    return {"format": "r16_adjacent_native_bias_recovery_audit_v1", "simulation_only": True,
        "all_evidence_consistent": all(item["accepted_evidence_consistency"] for item in cases),
        "all_command_bounds_satisfied": all(item["original_intersection_satisfied_for_all_audited_writes"]
            for item in cases), "cases": cases, "frozen_source": source, "rates_per_s": rates,
        "summary_sha256": digest(summary_path), "protocol_sha256": digest(protocol_path),
        "script_sha256": digest(__file__), "target_qpos_writes_independently_proven": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = audit_directory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"all_evidence_consistent": receipt["all_evidence_consistent"],
        "all_command_bounds_satisfied": receipt["all_command_bounds_satisfied"], "output": str(args.output)}))
    return 0 if receipt["all_command_bounds_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
