#!/usr/bin/env python3
"""Audit declared R17 command budgets and native cached actuator evaluations."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
from types import FunctionType
import zipfile

import numpy as np

import audit_public_ycb_bias_recovery as base


ALLOWED_BUDGETS_RAD = (.02, .04, .10)
R16_SOURCE = "scripts/develop_public_ycb_bias_recovery_r16.py"
R16_SUMMARY = "results/public_ycb_bias_recovery_20261022_r16_seed3414/summary.json"
R16_CORE = "scripts/audit_public_ycb_bias_recovery.py"
R17_SOURCE = "scripts/develop_public_ycb_bias_budget_r17.py"
FROZEN_R16_SOURCE_SHA256 = "c24cbc678cd6c241f7c57d2a9f255dbc23c8bd6464234064789c6f97a081fe62"
FROZEN_R16_SUMMARY_SHA256 = "beb98722b0e9224e42f11c38bd6484d8df15091e05fcace2818553885d748757"
FROZEN_R16_CORE_SHA256 = "7087507aa219a8b0d7bc04022f1f49abebfe09c052371956d5049cf33b535b5e"
FORCE_TOLERANCE = 1e-9


def budget_value(value):
    if type(value) not in (int, float):
        raise ValueError("Bias budget must be a declared finite number")
    value = base.finite(value, "bias budget")
    if value not in ALLOWED_BUDGETS_RAD:
        raise ValueError("R17 requires a declared .02/.04/.10rad experiment")
    return value


def verify_core():
    if base.digest(base.__file__) != FROZEN_R16_CORE_SHA256 or base.BIAS_LIMIT_RAD != .02:
        raise ValueError("Frozen R16 audit core or its original .02rad limit changed")


def reconstruct_budget_case(native, events, joint_events, specs, *, budget_rad, **kwargs):
    verify_core()
    budget = budget_value(budget_rad)
    for event in events:
        if (event["format"] != "r17_bias_budget_write_v1"
                or event["inherited_write_format"] != "r16_bias_recovery_write_v1"):
            raise ValueError("Unknown or mislabeled R17 inherited write format")
        base.equal(budget_value(event["additional_bias_limit_rad"]), budget, "Write budget")
        if not isinstance(event["per_finger"], list):
            raise ValueError("R17 per-finger telemetry must be a list")
        for finger in event["per_finger"]:
            for detail in finger["joints"].values():
                base.equal(budget_value(detail["additional_bias_limit_rad"]), budget, "Joint budget")
    isolated = FunctionType(base.reconstruct_case.__code__,
        dict(base.reconstruct_case.__globals__, BIAS_LIMIT_RAD=budget),
        "reconstruct_declared_r17_budget", base.reconstruct_case.__defaults__,
        base.reconstruct_case.__closure__)
    isolated.__kwdefaults__ = base.reconstruct_case.__kwdefaults__
    receipt = isolated(native, events, joint_events, specs, **kwargs)
    receipt["baseline_additional_bias_limit_rad"] = .02
    receipt["scope"] = ("Declared R17 per-instance command budget reconstructed from adjacent native ctrl "
        "and contemporaneous encoders using the frozen R16 audit algorithm in an isolated namespace. "
        "Excluded emergency/release/scalar writes do not establish the declared biased bounds. "
        "A larger budget is a changed experiment, not compliance with the original .02rad limit. "
        "This does not prove grasp success, contact force, QP optimality or absence of external target writes.")
    return receipt


def integer_list(value, label):
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"Invalid {label}")
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate {label}")
    return value


def compiled_actuator_parameters(model, actuator):
    joint = int(model.actuator_trnid[actuator, 0])
    return {"actuator_id": actuator, "actuator_name": model.actuator(actuator).name,
        "joint_id": joint, "joint_name": model.joint(joint).name,
        "dof_id": int(model.jnt_dofadr[joint]),
        **{key: int(getattr(model, "actuator_" + key)[actuator])
            for key in ("trntype", "dyntype", "gaintype", "biastype")},
        **{key: getattr(model, "actuator_" + key)[actuator].tolist()
            for key in ("trnid", "gainprm", "biasprm", "gear", "ctrlrange", "forcerange")},
        **{key: bool(getattr(model, "actuator_" + key)[actuator])
            for key in ("ctrllimited", "forcelimited")},
        "jointactfrclimited": bool(model.jnt_actfrclimited[joint]),
        "jointactfrcrange": model.jnt_actfrcrange[joint].tolist()}


def audit_actuation(native, metadata, model, *, expected_timing_scope):
    """Recompute force from the same evaluation cache, never post-step qpos."""
    if metadata["format"] != "r17_native_actuation_metadata_v1":
        raise ValueError("Unknown native actuation metadata format")
    timing = metadata["timing_scope"]
    if (not isinstance(timing, str) or not timing.strip() or timing != expected_timing_scope
            or base.flag(metadata, "evaluation_only") is not True):
        raise ValueError("Native actuation timing/evaluation declaration differs from frozen source")
    if (metadata["force_evaluation_state"] != "last_completed_dynamics_evaluation_cache"
            or metadata["observation_time_semantics"] != "post_integration_native_sample"
            or base.flag(metadata, "postintegration_qpos_recomputed") is not False):
        raise ValueError("Native actuation evaluation state is not explicitly separated from observation time")
    import mujoco
    actuation_flags = int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL | mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    if int(model.opt.disableflags) & actuation_flags or int(model.opt.disableactuator):
        raise ValueError("Disabled actuation/control clamping is outside the frozen force reconstruction")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError("Cached-state temporal reconstruction requires the original implicitfast integrator")
    actuators = integer_list(metadata["actuator_ids"], "actuator IDs")
    dofs = integer_list(metadata["joint_dof_ids"], "joint DOF IDs")
    expected = [index for index in range(model.nu) if int(model.actuator_trntype[index]) == 0
        and (model.joint(int(model.actuator_trnid[index, 0])).name or "").startswith("left_")]
    if not expected or actuators != expected or len(dofs) != len(actuators):
        raise ValueError("Native actuator inventory differs from all compiled left-hand joint actuators")
    parameters = [compiled_actuator_parameters(model, actuator) for actuator in actuators]
    if dofs != [item["dof_id"] for item in parameters] or metadata["actuator_parameters"] != parameters:
        raise ValueError("Actuator metadata differs from the runtime-bound compiled model")
    for item in parameters:
        if (item["trntype"] != 0 or item["dyntype"] != 0 or item["gaintype"] != 0
                or item["biastype"] != 1 or int(model.jnt_type[item["joint_id"]]) != 3
                or any(abs(value) > 0. for value in item["gear"][1:])):
            raise ValueError("Only frozen fixed-gain affine-bias hinge transmissions are supported")
        transmitters = np.flatnonzero((model.actuator_trntype == 0)
            & (model.actuator_trnid[:, 0] == item["joint_id"]))
        if list(transmitters) != [item["actuator_id"]]:
            raise ValueError("Unexpected additional transmission on an audited hand joint")
    maxima = {"absolute_actuator_force": 0., "absolute_joint_actuator_torque_nm": 0.,
        "actuator_force_reconstruction_residual": 0., "joint_torque_reconstruction_residual_nm": 0.}
    by_joint = {item["joint_name"]: {"absolute_actuator_force": 0.,
        "absolute_joint_actuator_torque_nm": 0., "joint_limit_saturated_samples": 0} for item in parameters}
    for index, sample in enumerate(native, 1):
        if base.tick(sample["timestamp"]) != index:
            raise ValueError("Actuation records require contiguous native 2ms samples")
        record = sample["native_actuation"]
        if record["format"] != "r17_native_actuation_evaluation_v1":
            raise ValueError("Unknown native actuation sample format")
        base.equal(record["timestamp_s"], sample["timestamp"], "Actuation observation time", 1e-8)
        if record["actuator_ids"] != actuators or record["joint_dof_ids"] != dofs:
            raise ValueError("Actuation sample mapping changed")
        arrays = {}
        for key in ("actuator_force", "qfrc_actuator", "actuator_length", "actuator_velocity"):
            values = np.asarray(record[key], dtype=float)
            if values.shape != (len(actuators),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid native actuation vector: {key}")
            arrays[key] = values
        for offset, item in enumerate(parameters):
            if index > 1:
                previous = native[index - 2]["actual_hand_joints"][item["joint_name"]]
                base.equal(arrays["actuator_length"][offset],
                    base.finite(previous["position_rad"], "previous native encoder") * item["gear"][0],
                    "Cached actuator length from previous native state")
                base.equal(arrays["actuator_velocity"][offset],
                    base.finite(previous["velocity_rad_s"], "previous native joint velocity") * item["gear"][0],
                    "Cached actuator velocity from previous native state")
            command = base.finite(sample["motor_commands"][item["actuator_id"]], "native motor command")
            if item["ctrllimited"]:
                command = float(np.clip(command, *item["ctrlrange"]))
            gain, bias = item["gainprm"], item["biasprm"]
            force = (gain[0] * command + bias[0] + bias[1] * arrays["actuator_length"][offset]
                + bias[2] * arrays["actuator_velocity"][offset])
            if item["forcelimited"]:
                force = float(np.clip(force, *item["forcerange"]))
            base.equal(arrays["actuator_force"][offset], force,
                "Cached native actuator force", FORCE_TOLERANCE)
            raw_torque = item["gear"][0] * force
            torque = float(np.clip(raw_torque, *item["jointactfrcrange"])) if item["jointactfrclimited"] else raw_torque
            base.equal(arrays["qfrc_actuator"][offset], torque,
                "Transmitted/clamped joint actuator torque", FORCE_TOLERANCE)
            maxima["absolute_actuator_force"] = max(maxima["absolute_actuator_force"], abs(force))
            maxima["absolute_joint_actuator_torque_nm"] = max(maxima["absolute_joint_actuator_torque_nm"], abs(torque))
            maxima["actuator_force_reconstruction_residual"] = max(
                maxima["actuator_force_reconstruction_residual"], abs(arrays["actuator_force"][offset] - force))
            maxima["joint_torque_reconstruction_residual_nm"] = max(
                maxima["joint_torque_reconstruction_residual_nm"], abs(arrays["qfrc_actuator"][offset] - torque))
            stats = by_joint[item["joint_name"]]
            stats["absolute_actuator_force"] = max(stats["absolute_actuator_force"], abs(force))
            stats["absolute_joint_actuator_torque_nm"] = max(stats["absolute_joint_actuator_torque_nm"], abs(torque))
            stats["joint_limit_saturated_samples"] += int(abs(raw_torque - torque) > FORCE_TOLERANCE)
    if not native:
        raise ValueError("Missing native actuation samples")
    return {"all_cached_force_and_transmitted_torque_consistent": True,
        "native_samples": len(native), "actuators_per_sample": len(actuators),
        "cache_states_crosschecked_against_previous_native_sample": len(native) - 1,
        "initial_cache_state_excluded_samples": 1,
        "maxima": maxima, "by_joint": by_joint, "force_tolerance": FORCE_TOLERANCE,
        "timing_scope": timing,
        "scope": "Reconstructed actuator force from each same-cache actuator length/velocity, native held ctrl and compiled fixed gain/affine bias, then joint transmission and original joint actuator-force limits. For the original implicitfast hinge model, all but the initial cache length/velocity are crosschecked against the preceding native encoder/velocity, not the current postintegration qpos. Initial cache state has no preceding native sample and is explicitly excluded from that temporal check. This does not establish physical contact force, net joint torque, constraint torque or hardware motor limits."}


def frozen_inputs(directory, protocol):
    verify_core()
    declaration = protocol["r17_bias_budget_experiment"]
    if declaration["format"] != "r17_bias_budget_development_protocol_v1":
        raise ValueError("Unknown R17 experiment declaration")
    budget = budget_value(declaration["additional_bias_limit_rad"])
    base.equal(declaration["baseline_additional_bias_limit_rad"], .02, "Original R16 budget")
    base.equal(protocol["joint_admittance"]["additional_bias_limit_rad"], budget, "Current protocol budget")
    inherited_protocol = copy.deepcopy(protocol)
    inherited_protocol["joint_admittance"]["additional_bias_limit_rad"] = .02
    rates, source = base.frozen_inputs(directory, inherited_protocol)
    before = protocol["source_hashes_before"]
    if (before[R16_SOURCE] != FROZEN_R16_SOURCE_SHA256
            or before[R16_SUMMARY] != FROZEN_R16_SUMMARY_SHA256
            or before[R16_CORE] != FROZEN_R16_CORE_SHA256):
        raise ValueError("Frozen R16 execution/comparison/audit identity changed")
    with zipfile.ZipFile(directory / "frozen_source.zip") as archive:
        previous = json.loads(archive.read(R16_SUMMARY))
        source_text = archive.read(R17_SOURCE)
    prior = previous["protocol"]
    if (previous["complete"] is not True or prior["source_changed_during_run"] is not False
            or prior["source_hashes_before"] != prior["source_hashes_after"]
            or any(before.get(key) != value for key, value in prior["source_hashes_before"].items())):
        raise ValueError("R17 inherited inputs differ from exact R16 baseline")
    if (declaration["inherited_r16_declaration"] != prior["r16_bias_recovery_experiment"]
            or protocol["r16_bias_recovery_experiment"] != prior["r16_bias_recovery_experiment"]):
        raise ValueError("Original R16 declaration was relabeled")
    if (declaration["r16_comparison_summary_sha256"] != FROZEN_R16_SUMMARY_SHA256
            or before["config/public_ycb_sidewall_20261021.json"] !=
                prior["r16_bias_recovery_experiment"]["base_profile_sha256"]
            or any(before.get(key) != value for key, value in declaration["preflight_input_sha256"].items())):
        raise ValueError("R17 baseline/profile/preflight dependencies are not source-bound")
    timing = base.literal_assignment(source_text, "TIMING_SCOPE")
    if (not isinstance(timing, str) or not timing.strip()
            or declaration["telemetry"] != {"format": "r17_native_actuation_evaluation_v1",
                "evaluation_only": True, "timing_scope": timing}):
        raise ValueError("Native actuation protocol differs from frozen timing declaration")
    for node in ast.walk(ast.parse(source_text)):
        targets = (node.targets if isinstance(node, ast.Assign) else
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        if any(isinstance(part, ast.Attribute) and part.attr == "qpos"
                for target in targets for part in ast.walk(target)):
            raise ValueError("R17 source contains direct qpos assignment")
    source.update(r16_source_sha256=before[R16_SOURCE], r16_summary_sha256=before[R16_SUMMARY],
        r16_audit_core_sha256=before[R16_CORE], r17_source_sha256=before[R17_SOURCE],
        declared_additional_bias_limit_rad=budget, baseline_additional_bias_limit_rad=.02,
        native_actuation_timing_scope=timing,
        static_check_scope="Direct qpos assignment syntax in frozen R16 and new R17 sources only; not arbitrary imported/dynamic writes")
    return budget, rates, source


def audit_directory(directory):
    from public_ycb_evidence import load_evidence_model

    directory = Path(directory).resolve(strict=True)
    summary_path = base.checked_path(directory, "summary.json")
    protocol_path = base.checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or protocol["format"] != "public_ycb_bias_budget_development_r17"):
        raise ValueError("Expected a complete consistently bound R17 experiment")
    budget, rates, source = frozen_inputs(directory, protocol)
    if not summary["rows"] or len(summary["rows"]) != len(protocol["schedule"]):
        raise ValueError("Scheduled R17 experiment denominator differs")
    cases, seen = [], set()
    for planned, row in zip(protocol["schedule"], summary["rows"], strict=True):
        if any(row[key] != value for key, value in planned.items()) or row["path"] in seen:
            raise ValueError("Duplicate or mismatching R17 scheduled case")
        seen.add(row["path"])
        if row["runner_error"]:
            raise ValueError("Runner error cannot establish complete R17 command evidence")
        base.equal(budget_value(row["additional_bias_limit_rad"]), budget, "Case loaded budget")
        case = base.checked_path(directory, row["path"])
        metadata = base.checked_path(case, "metadata.json")
        if json.loads(metadata.read_text()) != row:
            raise ValueError("R17 case metadata differs from summary")
        scene = base.checked_path(case, "scene.xml")
        if (row["scene_unchanged_since_load"] is not True
                or row["scene_sha256_before_load"] != base.digest(scene)
                or row["scene_sha256_at_finalize"] != base.digest(scene)):
            raise ValueError("R17 scene does not bind the runtime-loaded model")
        model = load_evidence_model(scene)
        base.equal(model.opt.timestep, base.STEP_S, "Native R17 physics timestep", 1e-12)
        records = {key: base.checked_stream(case, row["artifacts"][key]) for key in
            ("native", "bias_recovery", "joint_admittance", "preload", "load_transfer_probe")}
        if row["native_audit"]["native_actuation_samples"] != len(records["native"]):
            raise ValueError("Native actuation denominator differs from integrated samples")
        receipt = reconstruct_budget_case(records["native"], records["bias_recovery"],
            records["joint_admittance"], base.joint_specs(model, protocol["nominal_profiles"][row["kind"]]),
            budget_rad=budget, **rates, preload_events=records["preload"],
            probe_events=records["load_transfer_probe"], native_audit=row["native_audit"])
        receipt.update(case=row["path"], scene_sha256=base.digest(scene), metadata_sha256=base.digest(metadata),
            inputs={key: row["artifacts"][key] for key in records},
            actuation=audit_actuation(records["native"], row["native_audit"]["native_actuation_metadata"], model,
                expected_timing_scope=source["native_actuation_timing_scope"]),
            task_success=row["success"], held_lift=row["held_lift"],
            task_outcome_scope="Original reported task results, not independently rescored by the budget/actuation auditor")
        cases.append(receipt)
    return {"format": "r17_declared_bias_budget_and_native_actuation_audit_v1", "simulation_only": True,
        "all_evidence_consistent": all(item["accepted_evidence_consistency"] for item in cases),
        "all_declared_command_bounds_satisfied": all(item["original_intersection_satisfied_for_all_audited_writes"]
            for item in cases), "additional_bias_limit_rad": budget, "baseline_additional_bias_limit_rad": .02,
        "cases": cases, "frozen_source": source, "rates_per_s": rates,
        "summary_sha256": base.digest(summary_path), "protocol_sha256": base.digest(protocol_path),
        "script_sha256": base.digest(__file__), "target_qpos_writes_independently_proven": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = audit_directory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"all_evidence_consistent": receipt["all_evidence_consistent"],
        "all_declared_command_bounds_satisfied": receipt["all_declared_command_bounds_satisfied"],
        "additional_bias_limit_rad": receipt["additional_bias_limit_rad"], "output": str(args.output)}))
    return 0 if receipt["all_declared_command_bounds_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
