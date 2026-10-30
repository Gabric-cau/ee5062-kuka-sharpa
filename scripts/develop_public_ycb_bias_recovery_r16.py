#!/usr/bin/env python3
"""Declared per-finger bias recovery on the frozen R15 public-can geometry."""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import FunctionType, SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
import develop_public_ycb_fixed_geometry_r15 as r15

r14, r13 = r15.r14, r15.r14.r13
sha256 = r15.sha256
EXPECTED_R15_SUMMARY_SHA256 = "0d1120d582e1b9b47b70ef92626a845677ec7044308710ae23510e264ad8f4a0"
R15_SUMMARY = PROJECT / "results/public_ycb_fixed_geometry_20261022_r15_seed3414/summary.json"
EXCLUDED_MODES = {"RELEASING", "OPEN_HOLD", "SCALAR_RECONCILING", "EMERGENCY_HOLD"}
RELEASE_PHASES = {"RELEASE", "VERIFY", "RECOVER"}
RECOVERY_PARAMETERS = {
    "projection": "Euclidean minimum command change onto the unchanged separable joint box",
    "invalid_target_policy": "Project every captured joint; do not reuse stale Jacobians or pressure targets",
    "per_finger_policy": "Keep a fresh feasible R14 nominal solution for other fingers",
    "capture_policy": "Original activation capture only; never recapture during a loaded epoch",
    "empty_box_policy": "Declare failure, hold this write, then original controlled release; no feasible-write claim",
    "admission_policy": "Any recovery inhibits admission and clears continuous windows; active probe or held recovery aborts",
    "constraint_tolerance_rad": 1e-10,
    "scope": "Development controller experiment, not a success claim or a post-step invariant guarantee",
}


def command_box(reference, actual, bias, limits, ratio, dt, close_rate, relief_rate):
    values = np.asarray([reference, actual, bias, *limits, ratio, dt, close_rate, relief_rate], float)
    if not np.isfinite(values).all() or dt <= 0. or min(close_rate, relief_rate) < 0.:
        raise ValueError("Finite joint state and a positive write interval are required")
    rate = [reference - relief_rate * abs(ratio) * dt, reference + close_rate * abs(ratio) * dt]
    margin = r13.JOINT_PARAMETERS["additional_bias_limit_rad"]
    added = [actual + bias - margin, actual + bias + margin]
    interval = [max(float(limits[0]), rate[0], added[0]), min(float(limits[1]), rate[1], added[1])]
    return {"range_rad": list(map(float, limits)), "rate_interval_rad": rate,
        "bias_interval_rad": added, "intersection_rad": interval,
        "feasible": bool(interval[0] <= interval[1])}


def minimum_change_projection(reference, interval):
    low, high = map(float, interval)
    if not np.isfinite([reference, low, high]).all() or low > high:
        raise ValueError("A nonempty finite original box is required")
    return float(np.clip(reference, low, high))


class BiasRecoveryHand(r14.ConstrainedPreloadHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_recovery_events = []
        self.bias_recovery_active = False
        self.bias_failure_latched = False
        self.bias_abort_reason = None

    def phase_begin(self, phase, data):
        new_close = phase == "CLOSE" and phase != self.phase
        super().phase_begin(phase, data)
        if new_close:
            self.bias_recovery_active = self.bias_failure_latched = False
            self.bias_abort_reason = None

    def admission_state(self):
        return {"preload_ready": bool(self.preload_ready), "preload_start_s": self.preload_start_s,
            "probe_eligible_seconds": float(self.probe_eligible_seconds),
            "probe_confirmed_seconds": float(self.probe_confirmed_seconds),
            "probe_active": bool(self.probe_active),
            "payload_confirmed": bool(self.estimated.payload_confirmed),
            "pressure_window_samples": len(self.pressure_window)}

    def inhibit_admission(self):
        self.preload_ready = False
        self.preload_start_s = self.preload_anchor_targets = self.preload_previous_sample = None
        self.probe_eligible_seconds = self.probe_confirmed_seconds = 0.
        self.pressure_window.clear()
        recovery = self.bias_recovery_active or self.bias_failure_latched
        if recovery and self.probe_active:
            self.bias_abort_reason = self.bias_abort_reason or "PROBE_BIAS_RECOVERY"
        elif recovery and self.estimated.payload_confirmed:
            self.bias_abort_reason = self.bias_abort_reason or "HELD_POLICY_BIAS_RECOVERY"

    def _joint_rows(self, data, previous, dt):
        output = {}
        flexion = self.flexion_joints()
        for name, reference in previous.items():
            actual = self._actual(data, name)
            bias = self.initial_command_bias[name]
            joint = int(self.model.actuator_trnid[self.actuators[name], 0])
            box = command_box(reference, actual, bias, self.model.jnt_range[joint],
                flexion[name][1], dt, self.config.close_rate, self.config.relief_rate)
            output[name] = {"before_actual_rad": actual,
                "before_command_rad": float(data.ctrl[self.actuators[name]]),
                "previous_reference_rad": reference, "fixed_capture_bias_rad": bias,
                "capture_timestamp_s": self.joint_activation_s, **box}
        return output

    def _finish_event(self, data, event):
        safety = bool(self.emergency_active or self.emergency_phase_hold)
        event["final_safety_override"] = safety
        if safety and self.joint_references:
            self.joint_references = {name: float(data.ctrl[self.actuators[name]]) for name in self.joint_references}
            self._mode("EMERGENCY_HOLD", data, "ORIGINAL_R6_SAFETY_REFERENCES_WRITTEN")
            self.inhibit_admission()
        for finger in event["per_finger"]:
            for name, joint in finger["joints"].items():
                applied = float(data.ctrl[self.actuators[name]])
                joint["applied_command_rad"] = applied
                interval = joint.get("intersection_rad")
                joint["applied_within_original_box"] = None if interval is None or safety else bool(
                    interval[0] - 1e-10 <= applied <= interval[1] + 1e-10)
            matrix = finger.get("compression_jacobian_m_rad")
            if matrix is not None and event["dt_s"] > 0.:
                velocity = np.array([(finger["joints"][name]["applied_command_rad"] -
                    finger["joints"][name]["previous_reference_rad"]) / event["dt_s"] for name in finger["joint_names"]])
                predicted = np.asarray(matrix) @ velocity
                finger["predicted_applied_normal_velocity_m_s"] = predicted.tolist()
                desired = np.asarray(finger["desired_normal_velocity_m_s"])
                mask = abs(desired) > r14.PRELOAD_PARAMETERS["direction_error_epsilon_m_s"]
                finger["nominal_direction_satisfied"] = (bool(np.all(np.sign(desired[mask]) * predicted[mask] >=
                    -r14.PRELOAD_PARAMETERS["direction_residual_tolerance_m_s"]))
                    if finger["mode"] == "NOMINAL" and not safety else None)
            else:
                finger["predicted_applied_normal_velocity_m_s"] = None
                finger["nominal_direction_satisfied"] = None
        violations = []
        if not safety:
            for finger in event["per_finger"]:
                if finger["mode"] in ("INFEASIBLE", "FAIL_PENDING"):
                    continue
                for name, joint in finger["joints"].items():
                    if abs(joint["applied_command_rad"] - joint["candidate_command_rad"]) > 1e-10:
                        violations.append(name + ":APPLIED_CANDIDATE_MISMATCH")
                    if not joint["applied_within_original_box"]:
                        violations.append(name + ":APPLIED_OUTSIDE_ORIGINAL_BOX")
                if finger["nominal_direction_satisfied"] is False:
                    violations.append(finger["finger"] + ":APPLIED_NOMINAL_DIRECTION_REVERSED")
        if violations:
            self.bias_failure_latched = True
            self.bias_abort_reason = "BIAS_FINAL_WRITE_VERIFICATION_FAILED"
            self.joint_references = {name: float(data.ctrl[self.actuators[name]]) for name in self.joint_references}
            self._mode("BIAS_INFEASIBLE", data, self.bias_abort_reason)
            self.inhibit_admission()
        event.update(mode=self.joint_mode, recovery_active=bool(self.bias_recovery_active),
            failure_latched=bool(self.bias_failure_latched), abort_reason=self.bias_abort_reason,
            write_verification_violations=violations,
            write_verification_passed=None if safety else not (violations or self.bias_failure_latched),
            preload_inhibited=bool(self.bias_recovery_active or self.bias_failure_latched or safety),
            admission_after_write=self.admission_state())
        self.bias_recovery_events.append(event)
        return event

    def write(self, data):
        admission_before = self.admission_state()
        self.update_pressure_relief(data, self.reader.read(data))
        previous = self.joint_references.copy()
        dt = 0. if self.last_joint_write_s is None else float(data.time) - self.last_joint_write_s
        fresh = bool(self.joint_target.get("feasible") and self.target_timestamp_s is not None
            and abs(data.time - self.target_timestamp_s) < 1e-9)
        event = {"format": "r16_bias_recovery_write_v1", "timestamp_s": float(data.time),
            "phase": self.phase, "dt_s": dt, "fresh_target": fresh,
            "target_reason": self.joint_target.get("reason"), "per_finger": [],
            "admission_before": admission_before,
            "scope": "Current-write constraints only; recovery may unload and never certifies nominal direction or task success"}
        self.bias_recovery_active = False
        excluded = (self.emergency_active or self.emergency_phase_hold or not previous
            or self.joint_activation_s is None or self.joint_mode in EXCLUDED_MODES
            or self.phase in RELEASE_PHASES)
        if excluded:
            r13.JointAdmittanceProbeHand.write(self, data)
            self._finish_event(data, event)
            return
        self.last_joint_write_s = float(data.time)
        if not np.isfinite(dt) or dt <= 0.:
            self.bias_failure_latched = True
            self.bias_abort_reason = "BIAS_WRITE_INTERVAL_INVALID"
            self._mode("BIAS_INFEASIBLE", data, self.bias_abort_reason)
            self.inhibit_admission()
            r13.r11.BoundedTrackingProbeHand.write(self, data)
            self._finish_event(data, event)
            return
        joints = self._joint_rows(data, previous, dt)
        reference_mismatch = any(abs(row["before_command_rad"] - row["previous_reference_rad"]) > 1e-10
            for row in joints.values())
        infeasible = self.bias_failure_latched or reference_mismatch or any(not row["feasible"] for row in joints.values())
        nominal_by_finger = {row["finger"]: row for row in self.joint_solves} if fresh else {}
        proposed, qp_rows = {}, []
        for finger in r13.ACTIVE:
            names = [name for name in previous if self.flexion_joints()[name][0] == finger]
            if not names:
                continue
            rows = {name: joints[name] for name in names}
            row = {"finger": finger, "joint_names": names, "joints": rows,
                "nominal_qp": None, "compression_jacobian_m_rad": None,
                "desired_normal_velocity_m_s": None}
            if infeasible:
                row.update(mode="INFEASIBLE" if any(not item["feasible"] for item in rows.values()) else "FAIL_PENDING",
                    reason="EMPTY_JOINT_BOX" if any(not item["feasible"] for item in rows.values()) else "GLOBAL_FAILURE_PENDING")
                commands = {name: previous[name] for name in names}
            else:
                original = nominal_by_finger.get(finger)
                accepted = False
                if original is not None and original["joint_names"] == names:
                    low = np.array([(rows[name]["intersection_rad"][0] - previous[name]) / dt for name in names])
                    high = np.array([(rows[name]["intersection_rad"][1] - previous[name]) / dt for name in names])
                    velocity, solved = r14.constrained_joint_velocity(original["compression_jacobian_m_rad"],
                        original["desired_normal_velocity_m_s"], low, high)
                    solved.update(finger=finger, joint_names=names,
                        compression_jacobian_m_rad=original["compression_jacobian_m_rad"],
                        lower_velocity_rad_s=low.tolist(), upper_velocity_rad_s=high.tolist(),
                        desired_normal_velocity_m_s=original["desired_normal_velocity_m_s"])
                    qp_rows.append(solved)
                    row.update(nominal_qp=solved,
                        compression_jacobian_m_rad=original["compression_jacobian_m_rad"],
                        desired_normal_velocity_m_s=original["desired_normal_velocity_m_s"])
                    accepted = bool(solved["accepted"])
                    if accepted:
                        commands = {name: float(np.clip(previous[name] + dt * value,
                            *rows[name]["intersection_rad"])) for name, value in zip(names, velocity, strict=True)}
                        actual_velocity = np.array([(commands[name] - previous[name]) / dt for name in names])
                        desired = np.asarray(original["desired_normal_velocity_m_s"])
                        mask = abs(desired) > r14.PRELOAD_PARAMETERS["direction_error_epsilon_m_s"]
                        predicted = np.asarray(original["compression_jacobian_m_rad"]) @ actual_velocity
                        accepted = bool(np.all(np.sign(desired[mask]) * predicted[mask] >=
                            -r14.PRELOAD_PARAMETERS["direction_residual_tolerance_m_s"]))
                if accepted:
                    row.update(mode="NOMINAL", reason="CURRENT_R14_QP_ACCEPTED")
                else:
                    commands = {name: minimum_change_projection(previous[name], rows[name]["intersection_rad"]) for name in names}
                    moved = any(abs(commands[name] - previous[name]) > 1e-15 for name in names)
                    reason = ("INVALID_TARGET_PROJECT_MOVE" if moved else "INVALID_TARGET_PROJECT_HOLD") if not fresh else "NOMINAL_DIRECTION_OR_SOLVE_INFEASIBLE"
                    row.update(mode="BIAS_RECOVERY", reason=reason)
                    self.bias_recovery_active = True
            for name, command in commands.items():
                rows[name]["candidate_command_rad"] = command
            proposed.update(commands)
            event["per_finger"].append(row)
        if infeasible:
            self.bias_failure_latched = True
            self.bias_abort_reason = self.bias_abort_reason or ("BIAS_REFERENCE_NATIVE_MISMATCH" if reference_mismatch else "BIAS_BOX_INFEASIBLE")
            self._mode("BIAS_INFEASIBLE", data, self.bias_abort_reason)
        else:
            self.joint_references.update(proposed)
            self.joint_velocities = {name: (value - previous[name]) / dt for name, value in proposed.items()}
            self._mode("BIAS_RECOVERY" if self.bias_recovery_active else "JOINT_CONTROL", data,
                "R16_PER_FINGER_RECOVERY" if self.bias_recovery_active else "R16_ALL_FINGERS_NOMINAL")
        if self.bias_recovery_active or self.bias_failure_latched:
            self.inhibit_admission()
        # Bypass the R13 all-finger nominal rollback, retaining the original R6 final writer.
        r13.r11.BoundedTrackingProbeHand.write(self, data)
        final = self._finish_event(data, event)
        details = {name: {**item, "command_rad": item["applied_command_rad"]}
            for finger in final["per_finger"] for name, item in finger["joints"].items()}
        applied = {name: float(data.ctrl[self.actuators[name]]) for name in previous}
        self.joint_events.append({"timestamp_s": float(data.time), "event": "NOMINAL_REFERENCE_WRITE",
            "phase": self.phase, "mode": self.joint_mode, "dt_s": dt,
            "current_target": self.joint_target, "target_age_s": None if self.target_timestamp_s is None else float(data.time - self.target_timestamp_s),
            "joint_solves": self.joint_solves, "joint_reference_steps": details,
            "bounded_motion_predictions": [], "candidate_command_rad": proposed, "applied_command_rad": applied,
            "safety_override_at_final_write": final["final_safety_override"],
            "command_rad": self.joint_references.copy(), "scope": "R16 per-finger modes are in bias_recovery; legacy active-only audit is insufficient"})
        if qp_rows:
            self.qp_events.append({"timestamp_s": float(data.time), "dt_s": dt, "phase": self.phase,
                "mode": self.joint_mode, "solves": qp_rows, "expected_command_rad": proposed,
                "applied_command_rad": applied, "parent_projection_difference_rad": None,
                "applied_motion_checks": [], "original_safety_override": final["final_safety_override"],
                "scope": "R16 recovery is not nominal direction compliance; inspect bias_recovery native audit"})

    def update_preload(self, data, *, admission=False):
        row = super().update_preload(data, admission=admission)
        if self.bias_recovery_active or self.bias_failure_latched:
            self.inhibit_admission()
            row.update(ready=False, continuous_seconds=0., window_anchor_targets_n=None,
                r16_recovery_inhibited=True)
            row["reasons"].append("R16_BIAS_RECOVERY_OR_FAILURE")
        return row

    def after_step(self, state, data, dt):
        failure = super().after_step(state, data, dt)
        abort_applied, resolved_abort = False, None
        if self.bias_recovery_active or self.bias_failure_latched:
            self.inhibit_admission()
            if state.value not in RELEASE_PHASES and (self.probe_active or self.estimated.payload_confirmed):
                resolved_abort = failure or self.bias_abort_reason or "BIAS_BOX_INFEASIBLE"
                self.abort_probe(resolved_abort)
                abort_applied = True
        returned = failure
        if state.value not in RELEASE_PHASES and (self.bias_failure_latched or self.bias_abort_reason):
            returned = failure or self.bias_abort_reason or "BIAS_BOX_INFEASIBLE"
        if self.bias_recovery_events and abs(self.bias_recovery_events[-1]["timestamp_s"] + dt - data.time) < 1e-9:
            self.bias_recovery_events[-1].update(post_step_timestamp_s=float(data.time),
                admission_after_step=self.admission_state(), original_post_step_failure=failure,
                post_step_returned_failure=returned, probe_abort_applied=abort_applied,
                resolved_abort_reason=resolved_abort)
        return returned

    def admit_probe(self, data):
        if self.bias_recovery_active or self.bias_failure_latched:
            row = self.update_preload(data, admission=True)
            self.probe_events.append({**row["original_probe_snapshot"], "admission": False,
                "preload_admission": False, "preload_reasons": row["reasons"],
                "preload_continuous_seconds": 0., "r16_recovery_inhibited": True})
            return False
        return super().admit_probe(data)


def controller_for(*args, **kwargs):
    namespace = dict(r14.controller_for.__globals__, ConstrainedPreloadHand=BiasRecoveryHand)
    factory = FunctionType(r14.controller_for.__code__, namespace, r14.controller_for.__name__,
        r14.controller_for.__defaults__, r14.controller_for.__closure__)
    return factory(*args, **kwargs)


def validate_inputs(profile_path):
    declaration = r15.validate_inputs(profile_path)
    if sha256(R15_SUMMARY) != EXPECTED_R15_SUMMARY_SHA256:
        raise ValueError("Frozen R15 comparison summary SHA256 mismatch")
    baseline = json.loads(R15_SUMMARY.read_text())
    if not baseline["complete"] or baseline["protocol"]["source_changed_during_run"]:
        raise ValueError("R15 comparison must be complete and immutable")
    hashes = baseline["protocol"]["source_hashes_before"]
    for relative, expected in hashes.items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R15 inherited source mismatch: " + relative)
    declaration.update(format="r16_bias_recovery_development_protocol_v1", controller_code_modified=True,
        controller="BiasRecoveryHand", inherited_geometry_declaration=copy.deepcopy(declaration),
        r15_comparison_summary_sha256=EXPECTED_R15_SUMMARY_SHA256,
        recovery_parameters=copy.deepcopy(RECOVERY_PARAMETERS),
        runtime_target_pose_or_qpos_added=False, physics_force_and_task_gates_modified=False,
        scope="New same-seed causal experiment required; recorded R15 future states cannot establish recovery or task success")
    return declaration


def snapshot(profile_path):
    values = r15.isolated_runner(profile_path, r15.validate_inputs(profile_path)).__globals__["snapshot"](profile_path)
    for relative in ("scripts/develop_public_ycb_bias_recovery_r16.py", "tests/test_public_ycb_bias_recovery_r16.py"):
        values[relative] = sha256(PROJECT / relative)
    values[str(R15_SUMMARY.relative_to(PROJECT))] = sha256(R15_SUMMARY)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    protocol = value.get("protocol") if "protocol" in value else value
    if isinstance(protocol, dict) and "nominal_profiles" in protocol and "r16_bias_recovery_experiment" not in protocol:
        protocol["inherited_controller_protocol_format"] = protocol["format"]
        protocol["format"] = "public_ycb_bias_recovery_development_r16"
        protocol["r16_bias_recovery_experiment"] = copy.deepcopy(declaration)
        protocol["force_control"] = ("R16 explicit per-finger nominal tracking or minimum-change bias recovery. "
            "Original range/rate/fixed-capture+.02rad box and original R6 final safety writer retained. "
            "Invalid targets also project every captured joint. Recovery inhibits admission; active probe/held recovery aborts. "
            "Empty box declares failure and original controlled release, never a feasible command claim.")
        for case in protocol["schedule"]:
            case["profile"] = "bias_recovery_r16"
    return value


def isolated_runner(profile_path, declaration):
    # Extend only the existing artifact tuple in the frozen runner factory.
    tree = ast.parse(textwrap.dedent(inspect.getsource(r14._runner)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.target) == "values" and isinstance(node.iter, ast.Tuple)
        and "preload.json.gz" in ast.unparse(node.iter)]
    if len(loops) != 1:
        raise RuntimeError("Frozen R14 artifact-extension structure changed")
    loops[0].iter.elts.append(ast.Tuple(elts=[ast.Constant(value=value) for value in
        ("bias_recovery", "bias_recovery_events", "bias_recovery.json.gz")], ctx=ast.Load()))
    namespace = dict(r14._runner.__globals__, controller_for=controller_for, snapshot=snapshot)
    exec(compile(ast.fix_missing_locations(tree), "<r16_runner_artifact_factory>", "exec"), namespace)
    original = namespace["_runner"]()
    globals_ = dict(original.__globals__)
    previous = globals_["previous"]
    proxy = SimpleNamespace(**vars(previous))
    build_scene = globals_["build_pick_place_scene"]
    scene_bindings = {}

    def capture_scene_before_load(*args, **kwargs):
        scene = build_scene(*args, **kwargs)
        scene_bindings[Path(scene).resolve()] = sha256(scene)
        return scene

    def write_declared_json(path, value):
        if Path(path).name == "metadata.json":
            scene = (Path(path).parent / "scene.xml").resolve()
            if scene in scene_bindings:
                value["scene_sha256_before_load"] = scene_bindings[scene]
                value["scene_sha256_at_finalize"] = sha256(scene)
                value["scene_unchanged_since_load"] = value["scene_sha256_at_finalize"] == value["scene_sha256_before_load"]
        return previous.atomic_json(path, tagged_protocol(value, declaration))

    proxy.atomic_json = write_declared_json
    globals_["previous"] = proxy
    globals_["build_pick_place_scene"] = capture_scene_before_load
    return FunctionType(original.__code__, globals_, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases):
    declaration = validate_inputs(profile_path)
    if not cases or any(kind != "ycb_can" or seed < 0 for kind, seed in cases) or len(set(cases)) != len(cases):
        raise ValueError("R16 requires unique nonnegative public-can development cases")
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
