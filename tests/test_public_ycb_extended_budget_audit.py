"""Self-contained R19 provenance/retry fixtures with real frozen audit cores."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_public_ycb_extended_budget as audit


r17, r18, base = audit.r17, audit.r18, audit.base
TIMING = "Last completed dynamics cache, not integrated-qpos recomputation."
SPECS = [base.JointSpec("left_" + finger + "_MCP_FE", finger, index, 0., 2., 1.)
    for index, finger in enumerate(base.ACTIVE_FINGERS)]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def gate():
    return {"preload_ready": False, "preload_start_s": None, "probe_eligible_seconds": 0.,
        "probe_confirmed_seconds": 0., "probe_active": False, "payload_confirmed": False,
        "pressure_window_samples": 0}


def streams_and_model():
    native, events, inherited = [], [], []
    for index, command in enumerate((.2, .2001, .2001, .2002)):
        actual = [.11 if index else .18, .18, .18]
        native.append({"timestamp": (index + 1) * .002, "state": "SEARCH" if not index else "CLOSE",
            "motor_commands": [command] * 3, "actual_hand_joints": {spec.name:
                {"position_rad": actual[i], "velocity_rad_s": 0.} for i, spec in enumerate(SPECS)}})
    inherited.append({"timestamp_s": .002, "event": "ACTIVATION_CAPTURE",
        "captured_command_rad": {spec.name: .2 for spec in SPECS},
        "captured_command_minus_actual_rad": {spec.name: .2 - .18 for spec in SPECS}})
    for index in range(4):
        fresh = index in (1, 3)
        mode = "SCALAR_ACQUISITION" if not index else "JOINT_CONTROL" if fresh else "BIAS_RECOVERY"
        event = {"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
            "additional_bias_limit_rad": .10, "timestamp_s": index * .002,
            "post_step_timestamp_s": (index + 1) * .002, "dt_s": .002 if index else 0.,
            "phase": "SEARCH" if not index else "CLOSE", "mode": mode, "fresh_target": fresh,
            "final_safety_override": False, "recovery_active": index == 2, "failure_latched": False,
            "preload_inhibited": index == 2, "write_verification_passed": True,
            "write_verification_violations": [], "abort_reason": None,
            "original_post_step_failure": None, "post_step_returned_failure": None,
            "probe_abort_applied": False, "resolved_abort_reason": None,
            "admission_before": gate(), "admission_after_write": gate(), "admission_after_step": gate(),
            "per_finger": []}
        inherited.append({"timestamp_s": index * .002, "event": "NOMINAL_REFERENCE_WRITE",
            "mode": mode, "phase": event["phase"], "current_target": {"feasible": fresh},
            "target_age_s": 0. if index else None, "safety_override_at_final_write": False})
        events.append(event)
        if not index:
            continue
        for spec in SPECS:
            before, after = native[index - 1], native[index]
            actual = before["actual_hand_joints"][spec.name]["position_rad"]
            previous, applied = before["motor_commands"][spec.actuator], after["motor_commands"][spec.actuator]
            rate, bias = [previous - .6 * .002, previous + .12 * .002], [actual + .02 - .10, actual + .02 + .10]
            detail = {"additional_bias_limit_rad": .10, "before_actual_rad": actual,
                "before_command_rad": previous, "previous_reference_rad": previous,
                "fixed_capture_bias_rad": .2 - .18, "capture_timestamp_s": .002, "range_rad": [0., 2.],
                "rate_interval_rad": rate, "bias_interval_rad": bias,
                "intersection_rad": [max(0., rate[0], bias[0]), min(2., rate[1], bias[1])],
                "candidate_command_rad": applied, "applied_command_rad": applied}
            event["per_finger"].append({"finger": spec.finger, "mode": "NOMINAL" if fresh else "BIAS_RECOVERY",
                "joint_names": [spec.name], "joints": {spec.name: detail},
                "compression_jacobian_m_rad": [[1.]] if fresh else None,
                "desired_normal_velocity_m_s": [.001] if fresh else None,
                "predicted_applied_normal_velocity_m_s": [(applied - previous) / .002] if fresh else None,
                "nominal_direction_satisfied": True if fresh else None})
    names = [spec.name for spec in SPECS]
    model = SimpleNamespace(nu=3, opt=SimpleNamespace(disableflags=0, disableactuator=0, integrator=3, timestep=.002),
        actuator_trntype=np.zeros(3, dtype=int), actuator_trnid=np.array([[0, -1], [1, -1], [2, -1]]),
        actuator_dyntype=np.zeros(3, dtype=int), actuator_gaintype=np.zeros(3, dtype=int),
        actuator_biastype=np.ones(3, dtype=int), jnt_type=np.full(3, 3), jnt_dofadr=np.arange(3),
        actuator_gainprm=np.array([[10.] + [0.] * 9] * 3),
        actuator_biasprm=np.array([[0., -10., -.2] + [0.] * 7] * 3),
        actuator_gear=np.array([[1., 0., 0., 0., 0., 0.]] * 3),
        actuator_ctrllimited=np.ones(3, dtype=bool), actuator_ctrlrange=np.array([[0., 2.]] * 3),
        actuator_forcelimited=np.zeros(3, dtype=bool), actuator_forcerange=np.zeros((3, 2)),
        jnt_actfrclimited=np.ones(3, dtype=bool), jnt_actfrcrange=np.array([[-.1, .1]] * 3))
    model.joint = lambda index: SimpleNamespace(name=names[index])
    model.actuator = lambda index: SimpleNamespace(name=names[index] + "_actuator")
    metadata = {"format": "r17_native_actuation_metadata_v1", "timing_scope": TIMING,
        "force_evaluation_state": "last_completed_dynamics_evaluation_cache",
        "observation_time_semantics": "post_integration_native_sample", "postintegration_qpos_recomputed": False,
        "evaluation_only": True, "actuator_ids": [0, 1, 2], "joint_dof_ids": [0, 1, 2],
        "actuator_parameters": [r17.compiled_actuator_parameters(model, i) for i in range(3)]}
    for index, sample in enumerate(native):
        lengths = [native[index - 1]["actual_hand_joints"][name]["position_rad"] for name in names] if index else [.18] * 3
        force = [10. * ctrl - 10. * q for ctrl, q in zip(sample["motor_commands"], lengths)]
        sample["native_actuation"] = {"format": "r17_native_actuation_evaluation_v1",
            "timestamp_s": sample["timestamp"], "actuator_ids": [0, 1, 2], "joint_dof_ids": [0, 1, 2],
            "actuator_force": force, "qfrc_actuator": np.clip(force, -.1, .1).tolist(),
            "actuator_length": lengths, "actuator_velocity": [0.] * 3}
    native_audit = {"integration_steps": 4, "native_samples": 4, "native_actuation_samples": 4,
        "native_actuation_metadata": metadata, "initial_target_qpos": [0., 0., .38, 1., 0., 0., 0.],
        "external_target_qpos_writes": 0, "write_events": []}
    return {"native": native, "bias_recovery": events, "joint_admittance": inherited,
        "preload": [{"timestamp_s": .006, "ready": False, "continuous_seconds": 0.}],
        "load_transfer_probe": [{"timestamp_s": .006, "probe_active": False, "admission": False}]}, model, native_audit


@pytest.fixture
def extended_directory(tmp_path, monkeypatch):
    directory, sources = tmp_path / "run", {}
    case = directory / "case"
    case.mkdir(parents=True)
    profile = {"ycb_can": {"contact_closure": {finger: 1. for finger in base.ACTIVE_FINGERS}}}
    sources.update({"scripts/develop_public_ycb_constrained_preload_r14.py": b"# synthetic frozen parent\n",
        "scripts/develop_public_ycb_joint_admittance_r13.py": b"JOINT_PARAMETERS = {'additional_bias_limit_rad': .02}\n",
        "scripts/develop_public_ycb_force_band.py": b"configuration = f(close_rate=.12, relief_rate=.6)\n",
        "config/profile.json": json.dumps(profile).encode()})
    hashes = {key: digest(value) for key, value in sources.items()}
    sources[base.R15_SUMMARY] = json.dumps({"complete": True, "protocol": {
        "source_changed_during_run": False, "source_hashes_before": hashes, "source_hashes_after": hashes}}).encode()
    sources[r17.R16_SOURCE] = b"RECOVERY_PARAMETERS = {'fixture': True}\n"
    base_profile = b'{"synthetic_base_profile": true}'
    r16_declaration = {"recovery_parameters": {"fixture": True},
        "r15_comparison_summary_sha256": digest(sources[base.R15_SUMMARY]),
        "base_profile_sha256": digest(base_profile)}
    hashes = {key: digest(value) for key, value in sources.items()}
    r16_protocol = {"source_changed_during_run": False, "source_hashes_before": hashes,
        "source_hashes_after": hashes, "r16_bias_recovery_experiment": r16_declaration}
    sources.update({r17.R16_SUMMARY: json.dumps({"complete": True, "protocol": r16_protocol}).encode(),
        r17.R16_CORE: Path(base.__file__).read_bytes(),
        r17.R17_SOURCE: ("TIMING_SCOPE = " + repr(TIMING) + "\n").encode(),
        audit.R17_CORE: Path(r17.__file__).read_bytes(), audit.R18_CORE: Path(r18.__file__).read_bytes(),
        audit.R18_SOURCE: b"# synthetic frozen R18 runner\n",
        audit.R18_RECOVERY_SOURCE: b"# synthetic reporting-only recovery\n",
        "config/public_ycb_sidewall_20261021.json": base_profile})
    monkeypatch.setattr(base, "FROZEN_R14_SHA256", digest(sources["scripts/develop_public_ycb_constrained_preload_r14.py"]))
    monkeypatch.setattr(base, "FROZEN_PROFILE_SHA256", digest(sources["config/profile.json"]))
    monkeypatch.setattr(base, "FROZEN_R15_SUMMARY_SHA256", digest(sources[base.R15_SUMMARY]))
    monkeypatch.setattr(r17, "FROZEN_R16_SOURCE_SHA256", digest(sources[r17.R16_SOURCE]))
    monkeypatch.setattr(r17, "FROZEN_R16_SUMMARY_SHA256", digest(sources[r17.R16_SUMMARY]))
    hashes = {key: digest(value) for key, value in sources.items()}
    old_r17 = {"format": "r17_bias_budget_development_protocol_v1", "additional_bias_limit_rad": .04,
        "baseline_additional_bias_limit_rad": .02, "inherited_r16_declaration": copy.deepcopy(r16_declaration),
        "changed_controller_parameters": {"additional_bias_limit_rad": {"before": .02, "after": .04}},
        "joint_parameters": {"additional_bias_limit_rad": .04},
        "r16_comparison_summary_sha256": digest(sources[r17.R16_SUMMARY]),
        "preflight_input_sha256": {key: hashes[key] for key in (r17.R16_SOURCE, r17.R16_SUMMARY,
            r17.R16_CORE, "config/public_ycb_sidewall_20261021.json")},
        "telemetry": {"format": "r17_native_actuation_evaluation_v1", "evaluation_only": True, "timing_scope": TIMING}}
    prior = {"format": r18.FORMAT, "source_changed_during_run": False,
        "source_hashes_before": hashes, "source_hashes_after": hashes,
        "joint_admittance": {"additional_bias_limit_rad": .04}, "r16_bias_recovery_experiment": r16_declaration,
        "r17_bias_budget_experiment": old_r17, "r18_close_deadline_experiment": {"additional_bias_limit_rad": .04},
        "nominal_profiles": profile, "maximum_attempts": 2,
        "phase_seconds": {"CLOSE": 32., "TRIAL_LIFT": 3.},
        "constrained_preload": {"close_seconds": 32., "group_error_limit_n": .35},
        "force_control": "Synthetic original R18 contract", "source_hashes_after_timing": "Explicit recovery observation",
        "schedule": [{"kind": "ycb_can", "seed": 3414, "profile": "bias_budget_r17", "additional_bias_limit_rad": .04,
            "runner_revision": r18.FORMAT, "close_seconds": 32., "arm_interpolation_seconds": 16.}]}
    for key in ("physics_profile", "grip_gate", "scoring", "latched_pressure_escape", "load_transfer_probe",
            "probe_confirmation", "wrench_evidence", "probe_hypothesis", "scalar_acquisition_tracking",
            "tracking_regulator_sha256", "tracking_regulator_source"):
        prior[key] = "synthetic unchanged " + key
    recovery = {"format": "r18_artifact_finalization_recovery_v1", "simulation_rerun": False,
        "original_complete": False, "original_runner_normal_exit_claimed": False,
        "runner_exit_code_observed": None, "observed_ssh_client_exit_code": 255,
        "script_sha256": digest(sources[audit.R18_RECOVERY_SOURCE]),
        "source_hashes_at_recovery": hashes}
    prior["finalization_recovery"] = recovery
    scene, controller = b"<mujoco/>", b"def run_task():\n    pass\n"
    (case / "scene.xml").write_bytes(scene)
    (case / "derived_controller.py").write_bytes(controller)
    arm = {**r18.TRANSFORM, "baseline_derived_controller_sha256": "earlier-R17-controller",
        "derived_controller_sha256": digest(controller)}
    baseline = {"complete": True, "artifact_finalization_recovered": True,
        "original_runner_normal_exit": False, "protocol": prior, "rows": [{"derived_controller_sha256": digest(controller),
            "scene_sha256_before_load": digest(scene), "arm_source_transform": arm}]}
    sources.update({audit.R18_SUMMARY: json.dumps(baseline).encode(), audit.R18_PROTOCOL: json.dumps(prior).encode(),
        audit.R18_RECOVERY: json.dumps(recovery).encode(), audit.R18_SCENE: scene, audit.R18_DERIVED: controller})
    baseline_hash, protocol_hash = digest(sources[audit.R18_SUMMARY]), digest(sources[audit.R18_PROTOCOL])
    sources[audit.R18_VERIFICATION] = json.dumps({"complete": True, "unknowns": 0, "summary_sha256": baseline_hash}).encode()
    sources[audit.R18_STRICT] = json.dumps({"accepted_evidence": True, "unknowns": 0, "summary_sha256": baseline_hash}).encode()
    sources[audit.R18_AUDIT] = json.dumps({"all_evidence_consistent": True, "all_declared_command_bounds_satisfied": True,
        "all_native_close_spans_within_declared_deadline": True, "script_sha256": base.digest(r18.__file__),
        "summary_sha256": baseline_hash, "protocol_sha256": protocol_hash,
        "additional_bias_limit_rad": .04, "close_seconds": 32., "arm_interpolation_seconds": 16.}).encode()
    sources[audit.R18_PREFIX] = json.dumps({"passed": False,
        "candidate_inputs": {"summary": {"sha256": baseline_hash}}}).encode()
    for attr, key in (("FROZEN_R18_SOURCE_SHA256", audit.R18_SOURCE), ("FROZEN_R18_SUMMARY_SHA256", audit.R18_SUMMARY),
            ("FROZEN_R18_PROTOCOL_SHA256", audit.R18_PROTOCOL), ("FROZEN_R18_RECOVERY_SHA256", audit.R18_RECOVERY),
            ("FROZEN_R18_VERIFICATION_SHA256", audit.R18_VERIFICATION), ("FROZEN_R18_STRICT_SHA256", audit.R18_STRICT),
            ("FROZEN_R18_AUDIT_SHA256", audit.R18_AUDIT), ("FROZEN_R18_PREFIX_SHA256", audit.R18_PREFIX),
            ("FROZEN_R18_RECOVERY_SOURCE_SHA256", audit.R18_RECOVERY_SOURCE),
            ("FROZEN_R18_SCENE_SHA256", audit.R18_SCENE), ("FROZEN_R18_DERIVED_SHA256", audit.R18_DERIVED)):
        monkeypatch.setattr(audit, attr, digest(sources[key]))
    constants = {"BUDGET_RAD": .10, "BASELINE_BUDGET_RAD": .04, "CLOSE_SECONDS": 32.,
        "ARM_INTERPOLATION_SECONDS": 16., "PROTOCOL_FORMAT": audit.FORMAT, "NATIVE_TIMESTEP_S": .002,
        "EXPECTED_R18_SUMMARY_SHA256": baseline_hash, "EXPECTED_R18_DERIVED_SHA256": digest(controller)}
    sources[audit.R19_SOURCE] = "".join(f"{key} = {value!r}\n" for key, value in constants.items()).encode()
    monkeypatch.setattr(audit, "FROZEN_R19_SOURCE_SHA256", digest(sources[audit.R19_SOURCE]))
    sources[audit.R19_AUDITOR] = Path(audit.__file__).read_bytes()
    active_r17 = copy.deepcopy(old_r17)
    active_r17["additional_bias_limit_rad"] = .10
    active_r17["changed_controller_parameters"]["additional_bias_limit_rad"]["after"] = .10
    active_r17["joint_parameters"]["additional_bias_limit_rad"] = .10
    declaration = {"format": "r19_extended_budget_development_protocol_v1", "controller": "BiasBudgetHand",
        "runner_revision": audit.FORMAT, "additional_bias_limit_rad": .10, "baseline_additional_bias_limit_rad": .04,
        "close_seconds": 32., "arm_interpolation_seconds": 16., "native_timestep_s": .002,
        "changed_parameters": {"joint_admittance.additional_bias_limit_rad": {"before": .04, "after": .10}},
        "preflight_input_sha256": audit.expected_preflight(), "r18_comparison_summary_sha256": baseline_hash,
        "active_r17_declaration": active_r17, "inherited_r18_declaration": prior["r18_close_deadline_experiment"],
        "inherited_r18_r17_declaration": old_r17,
        "inherited_r18_joint_admittance": prior["joint_admittance"],
        "inherited_r18_force_control": prior["force_control"], "inherited_r18_recovery": recovery,
        "effective_phase_seconds": prior["phase_seconds"], "effective_constrained_preload": prior["constrained_preload"],
        "baseline_source_after_timing": prior["source_hashes_after_timing"], "baseline_scene_sha256": digest(scene),
        "baseline_derived_controller_sha256": digest(controller), "baseline_task_success": False,
        "baseline_held_lift": False, "baseline_prefix_comparison_passed": False,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False}
    protocol = copy.deepcopy(prior)
    protocol.pop("finalization_recovery")
    protocol.pop("r18_close_deadline_experiment")
    hashes = {key: digest(value) for key, value in sources.items()}
    protocol.update(format=audit.FORMAT, runner_revision=audit.FORMAT, r19_extended_budget_experiment=declaration,
        source_hashes_before=hashes, source_hashes_after=hashes,
        inherited_r18_close_deadline_experiment=prior["r18_close_deadline_experiment"],
        inherited_r18_joint_admittance=prior["joint_admittance"], inherited_r18_force_control=prior["force_control"],
        r17_bias_budget_experiment=active_r17, joint_admittance={"additional_bias_limit_rad": .10})
    protocol["schedule"][0].update(runner_revision=audit.FORMAT, additional_bias_limit_rad=.10)
    streams, model, native_audit = streams_and_model()
    row = {**protocol["schedule"][0], "path": "case", "success": False, "held_lift": False, "runner_error": None,
        "attempts": [{"attempt": 1, "failure": "synthetic"}], "native_audit": native_audit,
        "scene_sha256_before_load": digest(scene), "scene_sha256_at_finalize": digest(scene),
        "scene_unchanged_since_load": True, "effective_task_phase_seconds": prior["phase_seconds"],
        "effective_runner_phase_seconds": prior["phase_seconds"], "effective_hand_instances": 1,
        "effective_hand_budgets_rad": [.10], "effective_hand_controller_types": ["BiasBudgetHand"],
        "derived_controller_sha256": digest(controller), "arm_source_transform": dict(arm,
            baseline_r18_derived_controller_sha256=digest(controller)), "artifacts": {}}
    for key, records in streams.items():
        path = case / (key + ".json.gz")
        path.write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
        row["artifacts"][key] = {"path": path.name, "sha256": base.digest(path), "bytes": path.stat().st_size}
    with zipfile.ZipFile(directory / "frozen_source.zip", "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    summary = {"complete": True, "runner_revision": audit.FORMAT, "close_seconds": 32., "arm_interpolation_seconds": 16.,
        "additional_bias_limit_rad": .10, "protocol": protocol, "rows": [row]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (case / "metadata.json").write_text(json.dumps(row))
    (case / "result.json").write_text(json.dumps({"attempts": row["attempts"], "runner_revision": audit.FORMAT,
        "controller_revision": "public_ycb_bias_budget_development_r17", "additional_bias_limit_rad": .10,
        "close_seconds": 32., "arm_interpolation_seconds": 16.}))
    import public_ycb_evidence
    monkeypatch.setattr(public_ycb_evidence, "load_evidence_model", lambda _: model)
    monkeypatch.setattr(base, "joint_specs", lambda *_: SPECS)
    return directory


def rewrite(directory, change):
    summary = json.loads((directory / "summary.json").read_text())
    change(summary)
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(summary["protocol"]))
    (directory / "case/metadata.json").write_text(json.dumps(summary["rows"][0]))


def rewrite_stream(directory, name, change):
    path = directory / "case" / (name + ".json.gz")
    values = json.loads(gzip.decompress(path.read_bytes()))
    change(values)
    path.write_bytes(gzip.compress(json.dumps(values).encode(), mtime=0))
    rewrite(directory, lambda s: s["rows"][0]["artifacts"][name].update(sha256=base.digest(path), bytes=path.stat().st_size))


def test_full_r19_reconstruction_preserves_old_protocols_and_audit_globals(extended_directory):
    prior = {path: base.digest(path) for path in (base.__file__, r17.__file__, r18.__file__)}
    protocol_before = (extended_directory / "protocol.json").read_bytes()
    receipt = audit.audit_directory(extended_directory)
    assert receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"]
    assert receipt["additional_bias_limit_rad"] == .10 and receipt["comparison_additional_bias_limit_rad"] == .04
    assert receipt["cases"][0]["maxima"]["absolute_added_bias_rad"] > .04
    assert receipt["cases"][0]["actuation"]["cache_states_crosschecked_against_previous_native_sample"] == 3
    assert receipt["cases"][0]["native_close_spans"][0]["duration_s"] == .006
    assert (extended_directory / "protocol.json").read_bytes() == protocol_before
    assert base.BIAS_LIMIT_RAD == .02 and r18.BUDGET_RAD == .04
    assert {path: base.digest(path) for path in prior} == prior


@pytest.mark.parametrize("change", [
    lambda s: s["protocol"].update(format=r18.FORMAT),
    lambda s: s.update(additional_bias_limit_rad=.04),
    lambda s: s["protocol"]["r19_extended_budget_experiment"].update(format="unknown"),
    lambda s: s["protocol"]["r19_extended_budget_experiment"].update(baseline_additional_bias_limit_rad=.02),
    lambda s: s["protocol"]["r19_extended_budget_experiment"].update(arm_interpolation_seconds=32.),
    lambda s: s["protocol"]["r19_extended_budget_experiment"]["changed_parameters"].update(force_limit={"before": 12, "after": 20}),
    lambda s: s["protocol"]["r19_extended_budget_experiment"]["inherited_r18_r17_declaration"].update(additional_bias_limit_rad=.10),
    lambda s: s["protocol"]["r19_extended_budget_experiment"]["preflight_input_sha256"].pop(audit.R18_RECOVERY),
    lambda s: s["protocol"]["joint_admittance"].update(additional_bias_limit_rad=.04),
    lambda s: s["protocol"].update(scoring="changed task"),
    lambda s: s["protocol"]["phase_seconds"].update(CLOSE=16.),
    lambda s: s["rows"][0]["effective_task_phase_seconds"].update(CLOSE=16.),
    lambda s: s["rows"][0].pop("effective_hand_budgets_rad"),
    lambda s: s["rows"][0].update(missing_runtime_controller_binding=True),
    lambda s: s["rows"][0].update(effective_hand_budgets_rad=[.04]),
    lambda s: s["rows"][0].update(effective_hand_controller_types=["OtherHand"]),
    lambda s: s["rows"][0].update(effective_hand_instances=2),
    lambda s: s["rows"][0]["arm_source_transform"].update(baseline_r18_derived_controller_sha256="wrong"),
    lambda s: s["rows"][0]["arm_source_transform"].update(hand_fraction_replacements=1),
    lambda s: s["rows"][0].update(additional_bias_limit_rad=.04),
    lambda s: s["protocol"]["source_hashes_after"].update({audit.R19_SOURCE: "changed"}),
    lambda s: s["protocol"]["source_hashes_before"].pop(audit.R18_SUMMARY),
])
def test_changed_budget_baseline_gates_effective_instances_or_bindings_rejected(extended_directory, change):
    rewrite(extended_directory, change)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(extended_directory)


@pytest.mark.parametrize("name,change", [
    ("bias_recovery", lambda r: r[2].update(format="r19_new_write")),
    ("bias_recovery", lambda r: r[2].update(additional_bias_limit_rad=.04)),
    ("bias_recovery", lambda r: r[2]["per_finger"][0]["joints"][SPECS[0].name].update(additional_bias_limit_rad=.04)),
    ("bias_recovery", lambda r: r[2]["per_finger"][0]["joints"][SPECS[0].name].update(bias_interval_rad=[.09, .17])),
    ("bias_recovery", lambda r: r[2].update(timestamp_s=.006)),
    ("bias_recovery", lambda r: r[2]["admission_after_step"].update(preload_ready=True)),
    ("native", lambda r: r[2]["motor_commands"].__setitem__(0, .4)),
    ("native", lambda r: r[2]["native_actuation"]["actuator_force"].__setitem__(0, 9.)),
    ("native", lambda r: r[2].update(timestamp=.008)),
])
def test_hash_consistent_corrupt_actual_command_bound_or_torque_rejected(extended_directory, name, change):
    rewrite_stream(extended_directory, name, change)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(extended_directory)


def test_current_scene_bound_locally_but_different_from_r18_rejected(extended_directory):
    scene = extended_directory / "case/scene.xml"
    scene.write_text("<mujoco><option gravity='0 0 -1'/></mujoco>")
    rewrite(extended_directory, lambda s: s["rows"][0].update(scene_sha256_before_load=base.digest(scene),
        scene_sha256_at_finalize=base.digest(scene)))
    with pytest.raises(ValueError, match="scene differs"):
        audit.audit_directory(extended_directory)


def test_missing_recovery_archive_entry_rejected(extended_directory):
    path = extended_directory / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist() if name != audit.R18_RECOVERY}
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(extended_directory)


def test_hash_consistent_r19_runner_replacement_rejected(extended_directory):
    path = extended_directory / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist()}
    sources[audit.R19_SOURCE] += b"# replaced runner with unchanged constants\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    rewrite(extended_directory, lambda s: [s["protocol"][key].update({audit.R19_SOURCE: digest(sources[audit.R19_SOURCE])})
        for key in ("source_hashes_before", "source_hashes_after")])
    with pytest.raises(ValueError, match="runner source identity"):
        audit.audit_directory(extended_directory)


def test_result_cannot_disagree_about_retry_instances(extended_directory):
    path = extended_directory / "case/result.json"
    value = json.loads(path.read_text())
    value["attempts"].append({"attempt": 2})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="result/retry"):
        audit.audit_directory(extended_directory)


def test_native_encoder_drift_beyond_declared_tenth_radian_is_rejected(extended_directory):
    def encoder_drift(native):
        native[1]["actual_hand_joints"][SPECS[0].name]["position_rad"] = .08
        cache = native[2]["native_actuation"]
        cache["actuator_length"][0] = .08
        cache["actuator_force"][0] = 10. * native[2]["motor_commands"][0] - .8
        cache["qfrc_actuator"][0] = .1

    def truthful_box(events):
        joint = events[2]["per_finger"][0]["joints"][SPECS[0].name]
        joint.update(before_actual_rad=.08, bias_interval_rad=[0., .2])
        joint["intersection_rad"] = [max(0., joint["rate_interval_rad"][0]),
            min(.2, joint["rate_interval_rad"][1])]

    rewrite_stream(extended_directory, "native", encoder_drift)
    rewrite_stream(extended_directory, "bias_recovery", truthful_box)
    with pytest.raises(ValueError):
        audit.audit_directory(extended_directory)


def test_retry_with_only_second_actual_instance_using_old_budget_is_rejected(extended_directory):
    def retry(summary):
        row = summary["rows"][0]
        row["attempts"].append({"attempt": 2, "failure": "synthetic"})
        row.update(effective_hand_instances=2, effective_hand_budgets_rad=[.10, .04],
            effective_hand_controller_types=["BiasBudgetHand", "BiasBudgetHand"])

    rewrite(extended_directory, retry)
    with pytest.raises(ValueError, match="instances/retries"):
        audit.audit_directory(extended_directory)
