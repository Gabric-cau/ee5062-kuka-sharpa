#!/usr/bin/env python3
"""One declared per-instance bias-budget experiment with the frozen R16 writer."""

from __future__ import annotations

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import develop_public_ycb_bias_recovery_r16 as r16

ALLOWED_BUDGETS_RAD = (.02, .04, .10)
BASELINE_BUDGET_RAD = .02
R16_DIRECTORY = PROJECT / "results/public_ycb_bias_recovery_20261022_r16_seed3414"
R16_SUMMARY = R16_DIRECTORY / "summary.json"
R16_AUDIT_RECEIPT = PROJECT / "results/r16_bias_recovery_audit_20261022/receipt.json"
EXPECTED_R16_SUMMARY_SHA256 = "beb98722b0e9224e42f11c38bd6484d8df15091e05fcace2818553885d748757"
PREFLIGHT_HASHES = {
    "scripts/develop_public_ycb_bias_recovery_r16.py": "c24cbc678cd6c241f7c57d2a9f255dbc23c8bd6464234064789c6f97a081fe62",
    "tests/test_public_ycb_bias_recovery_r16.py": "e62846ea6483fdfc1651b4ad70e42d31b02c0aa9ae84d828730140215a5f1c7c",
    "scripts/audit_public_ycb_bias_recovery.py": "7087507aa219a8b0d7bc04022f1f49abebfe09c052371956d5049cf33b535b5e",
    "config/public_ycb_sidewall_20261021.json": "7ffffa83c4ec1b5510537b3f5b82ad113b287b4aed18d57fad6a3b382a7f203e",
    "results/public_ycb_bias_recovery_20261022_r16_seed3414/summary.json": EXPECTED_R16_SUMMARY_SHA256,
    "results/public_ycb_bias_recovery_20261022_r16_seed3414/protocol.json": "c15d32fcee290748bf123bf68e898ede397b8057d4b87589cd8dd83734fac15c",
    "results/public_ycb_bias_recovery_20261022_r16_seed3414/verification_receipt.json": "015741bc9468417b90f3d4ce1f869e2e59830de71871c687187a6803df314a27",
    "results/r16_bias_recovery_audit_20261022/receipt.json": "376d6b3da85908c203ff2cd4dcc3cbd16a2c92d34dca8f17c03e0a286abaa43c",
}
NEW_SOURCES = (
    "scripts/develop_public_ycb_bias_budget_r17.py",
    "tests/test_public_ycb_bias_budget_r17.py",
    "docs/R17_BIAS_BUDGET_20261022.md",
)
OPTIONAL_AUDIT_SOURCES = (
    "scripts/audit_public_ycb_bias_budget.py",
    "tests/test_public_ycb_bias_budget_audit.py",
)
TIMING_SCOPE = (
    "Copied evaluation-only arrays at the existing post-mj_step native sample. "
    "actuator_force, qfrc_actuator, actuator_length and actuator_velocity are cached "
    "from the most recent dynamics evaluation/transition; timestamp_s is the "
    "post-integration observation time, not a claim that these arrays were "
    "recomputed at the integrated qpos. No extra forward or integration is performed."
)


def checked_budget(value):
    budget = float(value)
    if not np.isfinite(budget) or budget not in ALLOWED_BUDGETS_RAD:
        raise ValueError("Declare exactly one additional bias budget: .02, .04 or .10 rad")
    return budget


class BiasBudgetHand(r16.BiasRecoveryHand):
    def __init__(self, *args, additional_bias_limit_rad, **kwargs):
        self._additional_bias_limit_rad = checked_budget(additional_bias_limit_rad)
        super().__init__(*args, **kwargs)

    @property
    def additional_bias_limit_rad(self):
        return self._additional_bias_limit_rad

    def _joint_rows(self, data, previous, dt):
        rows = super()._joint_rows(data, previous, dt)
        for row in rows.values():
            center = row["before_actual_rad"] + row["fixed_capture_bias_rad"]
            row["additional_bias_limit_rad"] = self.additional_bias_limit_rad
            row["bias_interval_rad"] = [center - self.additional_bias_limit_rad,
                center + self.additional_bias_limit_rad]
            intervals = [row[key] for key in ("range_rad", "rate_interval_rad", "bias_interval_rad")]
            low, high = max(part[0] for part in intervals), min(part[1] for part in intervals)
            row["intersection_rad"], row["feasible"] = [low, high], bool(low <= high)
        return rows

    def _finish_event(self, data, event):
        event.update(inherited_write_format=event["format"], format="r17_bias_budget_write_v1",
            additional_bias_limit_rad=self.additional_bias_limit_rad)
        return super()._finish_event(data, event)


_OriginalNativeAudit = r16.r13.r11.r9.r8.r4.previous.RichNativeAudit


class TorqueNativeAudit(_OriginalNativeAudit):
    """Read cached MuJoCo actuation arrays without supplying controller inputs."""

    def __init__(self, model, data):
        super().__init__(model, data)
        parameters = []
        for aid in range(model.nu):
            if int(model.actuator_trntype[aid]) != 0:
                continue
            joint = int(model.actuator_trnid[aid, 0])
            name = model.joint(joint).name or ""
            if not name.startswith("left_") or int(model.jnt_type[joint]) != 3:
                continue
            parameters.append({"actuator_id": aid, "actuator_name": model.actuator(aid).name,
                "joint_id": joint, "joint_name": name, "dof_id": int(model.jnt_dofadr[joint]),
                "trntype": int(model.actuator_trntype[aid]), "trnid": model.actuator_trnid[aid].tolist(),
                "dyntype": int(model.actuator_dyntype[aid]), "gaintype": int(model.actuator_gaintype[aid]),
                "biastype": int(model.actuator_biastype[aid]), "gainprm": model.actuator_gainprm[aid].tolist(),
                "biasprm": model.actuator_biasprm[aid].tolist(), "gear": model.actuator_gear[aid].tolist(),
                "ctrllimited": bool(model.actuator_ctrllimited[aid]),
                "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
                "forcelimited": bool(model.actuator_forcelimited[aid]),
                "forcerange": model.actuator_forcerange[aid].tolist(),
                "jointactfrclimited": bool(model.jnt_actfrclimited[joint]),
                "jointactfrcrange": model.jnt_actfrcrange[joint].tolist()})
        self.actuation_ids = [item["actuator_id"] for item in parameters]
        self.actuation_dofs = [item["dof_id"] for item in parameters]
        if not parameters or len(set(self.actuation_dofs)) != len(parameters):
            raise ValueError("Expected one joint actuator per recorded hand DOF")
        self.actuation_metadata = {"format": "r17_native_actuation_metadata_v1",
            "timing_scope": TIMING_SCOPE, "evaluation_only": True,
            "force_evaluation_state": "last_completed_dynamics_evaluation_cache",
            "observation_time_semantics": "post_integration_native_sample",
            "postintegration_qpos_recomputed": False,
            "actuator_ids": self.actuation_ids.copy(), "joint_dof_ids": self.actuation_dofs.copy(),
            "actuator_parameters": parameters}

    def sample(self, state, data):
        count = len(self.samples)
        super().sample(state, data)
        if len(self.samples) == count:
            return
        self.samples[-1]["native_actuation"] = {
            "format": "r17_native_actuation_evaluation_v1", "timestamp_s": float(data.time),
            "actuator_ids": self.actuation_ids.copy(), "joint_dof_ids": self.actuation_dofs.copy(),
            "actuator_force": data.actuator_force[self.actuation_ids].copy().tolist(),
            "qfrc_actuator": data.qfrc_actuator[self.actuation_dofs].copy().tolist(),
            "actuator_length": data.actuator_length[self.actuation_ids].copy().tolist(),
            "actuator_velocity": data.actuator_velocity[self.actuation_ids].copy().tolist()}

    def as_dict(self):
        return {**super().as_dict(), "native_actuation_metadata": copy.deepcopy(self.actuation_metadata),
            "native_actuation_samples": sum("native_actuation" in item for item in self.samples)}


def controller_for(*args, additional_bias_limit_rad, **kwargs):
    constructor = partial(BiasBudgetHand,
        additional_bias_limit_rad=checked_budget(additional_bias_limit_rad))
    namespace = dict(r16.controller_for.__globals__, BiasRecoveryHand=constructor)
    factory = FunctionType(r16.controller_for.__code__, namespace, r16.controller_for.__name__,
        r16.controller_for.__defaults__, r16.controller_for.__closure__)
    return factory(*args, **kwargs)


def validate_inputs(profile_path, additional_bias_limit_rad):
    budget = checked_budget(additional_bias_limit_rad)
    for relative, expected in PREFLIGHT_HASHES.items():
        if r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R17 preflight input SHA256 mismatch: " + relative)
    inherited = r16.validate_inputs(profile_path)
    baseline = json.loads(R16_SUMMARY.read_text())
    protocol = baseline["protocol"]
    if (not baseline["complete"] or protocol["source_changed_during_run"]
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]):
        raise ValueError("The R16 comparison must be complete and immutable")
    for relative, expected in protocol["source_hashes_before"].items():
        if r16.sha256(PROJECT / relative) != expected:
            raise ValueError("Frozen R16 inherited source mismatch: " + relative)
    if inherited != protocol["r16_bias_recovery_experiment"]:
        raise ValueError("R16 inherited declaration differs from the frozen comparison")
    verification = json.loads((R16_DIRECTORY / "verification_receipt.json").read_text())
    audit = json.loads(R16_AUDIT_RECEIPT.read_text())
    if (not verification["complete"] or verification["unknowns"]
            or verification["summary_sha256"] != EXPECTED_R16_SUMMARY_SHA256
            or not audit["all_evidence_consistent"] or not audit["all_command_bounds_satisfied"]
            or audit["summary_sha256"] != EXPECTED_R16_SUMMARY_SHA256
            or audit["script_sha256"] != PREFLIGHT_HASHES["scripts/audit_public_ycb_bias_recovery.py"]):
        raise ValueError("R16 comparison verification receipts are not bound and complete")
    declaration = copy.deepcopy(inherited)
    declaration.update(format="r17_bias_budget_development_protocol_v1", controller="BiasBudgetHand",
        additional_bias_limit_rad=budget, baseline_additional_bias_limit_rad=BASELINE_BUDGET_RAD,
        changed_controller_parameters={"additional_bias_limit_rad": {"before": BASELINE_BUDGET_RAD, "after": budget}},
        inherited_r16_declaration=copy.deepcopy(inherited),
        r16_comparison_summary_sha256=EXPECTED_R16_SUMMARY_SHA256,
        preflight_input_sha256=copy.deepcopy(PREFLIGHT_HASHES),
        instance_budget_policy="Every factory-created hand, including retries, receives this immutable public instance budget",
        telemetry={"format": "r17_native_actuation_evaluation_v1", "evaluation_only": True,
            "timing_scope": TIMING_SCOPE},
        scope="One explicit same-seed controller-budget experiment; no automatic sweep, held-out rate or task-success claim")
    declaration["joint_parameters"] = {**declaration["joint_parameters"], "additional_bias_limit_rad": budget}
    return declaration


def snapshot(profile_path):
    values = r16.snapshot(profile_path)
    for relative in (*PREFLIGHT_HASHES, *NEW_SOURCES):
        values[relative] = r16.sha256(PROJECT / relative)
    for relative in OPTIONAL_AUDIT_SOURCES:
        if (PROJECT / relative).is_file():
            values[relative] = r16.sha256(PROJECT / relative)
    return values


def tagged_protocol(value, declaration):
    if not isinstance(value, dict):
        return value
    inherited = declaration["inherited_r16_declaration"]
    r16.tagged_protocol(value, inherited)
    protocol = value.get("protocol") if "protocol" in value else value
    budget = declaration["additional_bias_limit_rad"]
    if isinstance(protocol, dict) and "nominal_profiles" in protocol:
        if "r17_bias_budget_experiment" not in protocol:
            protocol["inherited_r16_joint_admittance"] = copy.deepcopy(protocol["joint_admittance"])
            protocol["inherited_r16_force_control"] = protocol["force_control"]
            protocol["format"] = "public_ycb_bias_budget_development_r17"
            protocol["r17_bias_budget_experiment"] = copy.deepcopy(declaration)
            protocol["joint_admittance"] = {**protocol["joint_admittance"], "additional_bias_limit_rad": budget}
            protocol["force_control"] = (
                f"R17 per-instance fixed-capture plus/minus {budget:g}rad added-bias budget. "
                "Frozen R16 writer, per-finger nominal/recovery, R6 final safety, capture, "
                "rates, geometry, physics, admission and task gates are retained. "
                "Cached native actuator/joint force telemetry is evaluation-only. "
                "Legacy .02rad declarations are explicitly inherited baseline records.")
        for case in protocol["schedule"]:
            case.update(profile="bias_budget_r17", additional_bias_limit_rad=budget)
    for row in value.get("rows", []):
        row["additional_bias_limit_rad"] = budget
    if {"kind", "seed"} <= value.keys() or "controller_revision" in value:
        value["additional_bias_limit_rad"] = budget
    if "controller_revision" in value:
        value.setdefault("inherited_controller_revision", value["controller_revision"])
        value["controller_revision"] = "public_ycb_bias_budget_development_r17"
    return value


def isolated_runner(profile_path, declaration):
    original = r16.isolated_runner(profile_path, declaration["inherited_r16_declaration"])
    namespace = dict(original.__globals__)
    previous = namespace["previous"]
    proxy = SimpleNamespace(**vars(previous))

    def write_declared_json(path, value):
        # The inner R16 tagger observes its existing key and preserves the R17 tag.
        return previous.atomic_json(path, tagged_protocol(value, declaration))

    proxy.atomic_json = write_declared_json
    proxy.RichNativeAudit = TorqueNativeAudit
    namespace.update(previous=proxy, snapshot=snapshot,
        controller_for=partial(controller_for,
            additional_bias_limit_rad=declaration["additional_bias_limit_rad"]))
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def run(output, profile_path, cases, *, additional_bias_limit_rad):
    declaration = validate_inputs(profile_path, additional_bias_limit_rad)
    if tuple(cases) != (("ycb_can", 3414),):
        raise ValueError("R17 isolates one budget on the frozen ycb_can:3414 development case")
    return isolated_runner(profile_path, declaration)(output, profile_path, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--additional-bias-limit-rad", required=True, type=float, choices=ALLOWED_BUDGETS_RAD)
    parser.add_argument("--cases", nargs="+", default=["ycb_can:3414"])
    args = parser.parse_args()
    cases = tuple((value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.cases)
    summary = run(args.output.resolve(), args.profile_json.resolve(), cases,
        additional_bias_limit_rad=args.additional_bias_limit_rad)
    return 2 if summary["protocol"]["source_changed_during_run"] else int(bool(summary["runner_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
