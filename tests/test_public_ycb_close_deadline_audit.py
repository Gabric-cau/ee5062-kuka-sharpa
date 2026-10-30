"""Self-contained R18 deadline/provenance fixture using the frozen R17 cores."""

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
import audit_public_ycb_close_deadline as audit


r17, base = audit.r17, audit.base
TIMING = "Last dynamics evaluation cache, not postintegration qpos recomputation."
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
        actual = [.15 if index else .18, .18, .18]
        native.append({"timestamp": (index + 1) * .002, "state": "SEARCH" if not index else "CLOSE",
            "motor_commands": [command] * 3, "actual_hand_joints": {s.name:
                {"position_rad": actual[i], "velocity_rad_s": 0.} for i, s in enumerate(SPECS)}})
    inherited.append({"timestamp_s": .002, "event": "ACTIVATION_CAPTURE",
        "captured_command_rad": {s.name: .2 for s in SPECS},
        "captured_command_minus_actual_rad": {s.name: .2 - .18 for s in SPECS}})
    for index in range(4):
        fresh = index in (1, 3)
        mode = "SCALAR_ACQUISITION" if not index else "JOINT_CONTROL" if fresh else "BIAS_RECOVERY"
        event = {"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
            "additional_bias_limit_rad": .04, "timestamp_s": index * .002,
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
            rate, bias = [previous - .6 * .002, previous + .12 * .002], [actual + .02 - .04, actual + .02 + .04]
            detail = {"additional_bias_limit_rad": .04, "before_actual_rad": actual,
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
        velocities = [0.] * 3
        force = [10. * ctrl - 10. * q for ctrl, q in zip(sample["motor_commands"], lengths)]
        sample["native_actuation"] = {"format": "r17_native_actuation_evaluation_v1",
            "timestamp_s": sample["timestamp"], "actuator_ids": [0, 1, 2], "joint_dof_ids": [0, 1, 2],
            "actuator_force": force, "qfrc_actuator": np.clip(force, -.1, .1).tolist(),
            "actuator_length": lengths, "actuator_velocity": velocities}
    native_audit = {"integration_steps": 4, "native_samples": 4, "native_actuation_samples": 4,
        "native_actuation_metadata": metadata, "initial_target_qpos": [0., 0., .38, 1., 0., 0., 0.],
        "external_target_qpos_writes": 0, "write_events": []}
    return {"native": native, "bias_recovery": events, "joint_admittance": inherited,
        "preload": [{"timestamp_s": .006, "ready": False, "continuous_seconds": 0.}],
        "load_transfer_probe": [{"timestamp_s": .006, "probe_active": False, "admission": False}]}, model, native_audit


@pytest.fixture
def deadline_directory(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    case = directory / "case"
    case.mkdir(parents=True)
    profile = {"ycb_can": {"contact_closure": {finger: 1. for finger in base.ACTIVE_FINGERS}}}
    sources = {"scripts/develop_public_ycb_constrained_preload_r14.py": b"# synthetic frozen parent\n",
        "scripts/develop_public_ycb_joint_admittance_r13.py": b"JOINT_PARAMETERS = {'additional_bias_limit_rad': .02}\n",
        "scripts/develop_public_ycb_force_band.py": b"configuration = f(close_rate=.12, relief_rate=.6)\n",
        "config/profile.json": json.dumps(profile).encode()}
    inherited = {key: digest(value) for key, value in sources.items()}
    prior = {"complete": True, "protocol": {"source_changed_during_run": False,
        "source_hashes_before": inherited, "source_hashes_after": inherited}}
    sources[base.R15_SUMMARY] = json.dumps(prior).encode()
    sources[r17.R16_SOURCE] = b"RECOVERY_PARAMETERS = {'fixture': True}\n"
    base_profile = b'{"synthetic_base_profile": true}'
    r16_hashes = {key: digest(value) for key, value in sources.items()}
    r16_declaration = {"recovery_parameters": {"fixture": True},
        "r15_comparison_summary_sha256": digest(sources[base.R15_SUMMARY]),
        "base_profile_sha256": digest(base_profile)}
    r16_protocol = {"source_changed_during_run": False, "source_hashes_before": r16_hashes,
        "source_hashes_after": r16_hashes, "r16_bias_recovery_experiment": r16_declaration}
    sources[r17.R16_SUMMARY] = json.dumps({"complete": True, "protocol": r16_protocol}).encode()
    sources.update({r17.R16_CORE: Path(base.__file__).read_bytes(),
        r17.R17_SOURCE: ("TIMING_SCOPE = " + repr(TIMING) + "\n").encode(),
        audit.R17_CORE: Path(r17.__file__).read_bytes(),
        "config/public_ycb_sidewall_20261021.json": base_profile})
    monkeypatch.setattr(base, "FROZEN_R14_SHA256", inherited["scripts/develop_public_ycb_constrained_preload_r14.py"])
    monkeypatch.setattr(base, "FROZEN_PROFILE_SHA256", inherited["config/profile.json"])
    monkeypatch.setattr(base, "FROZEN_R15_SUMMARY_SHA256", digest(sources[base.R15_SUMMARY]))
    monkeypatch.setattr(r17, "FROZEN_R16_SOURCE_SHA256", digest(sources[r17.R16_SOURCE]))
    monkeypatch.setattr(r17, "FROZEN_R16_SUMMARY_SHA256", digest(sources[r17.R16_SUMMARY]))
    hashes = {key: digest(value) for key, value in sources.items()}
    budget_declaration = {"format": "r17_bias_budget_development_protocol_v1", "additional_bias_limit_rad": .04,
        "baseline_additional_bias_limit_rad": .02, "inherited_r16_declaration": copy.deepcopy(r16_declaration),
        "r16_comparison_summary_sha256": digest(sources[r17.R16_SUMMARY]),
        "preflight_input_sha256": {key: hashes[key] for key in (r17.R16_SOURCE, r17.R16_SUMMARY,
            r17.R16_CORE, "config/public_ycb_sidewall_20261021.json")},
        "telemetry": {"format": "r17_native_actuation_evaluation_v1", "evaluation_only": True, "timing_scope": TIMING}}
    prior = {"format": "public_ycb_bias_budget_development_r17", "source_changed_during_run": False,
        "source_hashes_before": hashes, "source_hashes_after": hashes,
        "joint_admittance": {"additional_bias_limit_rad": .04}, "r16_bias_recovery_experiment": r16_declaration,
        "r17_bias_budget_experiment": budget_declaration, "nominal_profiles": profile, "maximum_attempts": 2,
        "phase_seconds": {"CLOSE": 16., "TRIAL_LIFT": 3.},
        "constrained_preload": {"close_seconds": 16., "group_error_limit_n": .35},
        "force_control": "Synthetic unchanged R17 hand contract",
        "schedule": [{"kind": "ycb_can", "seed": 3414, "profile": "bias_budget_r17", "additional_bias_limit_rad": .04}]}
    for key in ("physics_profile", "grip_gate", "scoring", "latched_pressure_escape", "load_transfer_probe",
            "probe_confirmation", "wrench_evidence", "probe_hypothesis", "scalar_acquisition_tracking",
            "tracking_regulator_sha256", "tracking_regulator_source"):
        prior[key] = "synthetic unchanged " + key
    scene = b"<mujoco><compiler/></mujoco>"
    (case / "scene.xml").write_bytes(scene)
    controller = b"def run_task():\n    pass\n"
    (case / "derived_controller.py").write_bytes(controller)
    baseline = {"complete": True, "protocol": prior, "rows": [{"derived_controller_sha256": "synthetic-baseline-derived",
        "scene_sha256_before_load": digest(scene)}]}
    sources.update({audit.R17_SUMMARY: json.dumps(baseline).encode(), audit.R17_PROTOCOL: json.dumps(prior).encode()})
    baseline_hash = digest(sources[audit.R17_SUMMARY])
    sources[audit.R17_VERIFICATION] = json.dumps({"complete": True, "unknowns": [], "summary_sha256": baseline_hash}).encode()
    sources[audit.R17_STRICT] = json.dumps({"accepted_evidence": True, "unknowns": [], "summary_sha256": baseline_hash}).encode()
    sources[audit.R17_AUDIT] = json.dumps({"all_evidence_consistent": True, "all_declared_command_bounds_satisfied": True,
        "script_sha256": base.digest(r17.__file__), "summary_sha256": baseline_hash}).encode()
    for attr, key in (("FROZEN_R17_SOURCE_SHA256", r17.R17_SOURCE), ("FROZEN_R17_SUMMARY_SHA256", audit.R17_SUMMARY),
            ("FROZEN_R17_PROTOCOL_SHA256", audit.R17_PROTOCOL), ("FROZEN_R17_VERIFICATION_SHA256", audit.R17_VERIFICATION),
            ("FROZEN_R17_STRICT_SHA256", audit.R17_STRICT), ("FROZEN_R17_AUDIT_SHA256", audit.R17_AUDIT)):
        monkeypatch.setattr(audit, attr, digest(sources[key]))
    constants = {"CLOSE_SECONDS_BEFORE": 16., "CLOSE_SECONDS_AFTER": 32., "ARM_INTERPOLATION_SECONDS": 16.,
        "NATIVE_TIMESTEP_S": .002, "BUDGET_RAD": .04, "PROTOCOL_FORMAT": audit.FORMAT,
        "EXPECTED_R17_SUMMARY_SHA256": baseline_hash}
    sources[audit.R18_SOURCE] = "".join(f"{key} = {value!r}\n" for key, value in constants.items()).encode()
    monkeypatch.setattr(audit, "FROZEN_R18_SOURCE_SHA256", digest(sources[audit.R18_SOURCE]))
    sources[audit.R18_AUDITOR] = Path(audit.__file__).read_bytes()
    hashes = {key: digest(value) for key, value in sources.items()}
    phase, preload = dict(prior["phase_seconds"], CLOSE=32.), dict(prior["constrained_preload"], close_seconds=32.)
    declaration = {"format": "r18_close_deadline_development_protocol_v1", "controller": "BiasBudgetHand",
        "runner_revision": audit.FORMAT, "close_seconds": 32., "baseline_close_seconds": 16.,
        "arm_interpolation_seconds": 16., "native_timestep_s": .002, "additional_bias_limit_rad": .04,
        "changed_parameters": {"phase_seconds.CLOSE": {"before": 16., "after": 32.}},
        "arm_source_transform": {**audit.TRANSFORM, "baseline_close_steps": 8000, "extension_holds_arm_fraction_at_one": True},
        "preflight_input_sha256": audit.expected_preflight(), "r17_comparison_summary_sha256": baseline_hash,
        "runtime_target_pose_or_qpos_added": False, "physics_force_and_task_gates_modified": False,
        "prefix_validation": {"required": True, "status": "NOT_YET_RUN"},
        "inherited_r17_declaration": copy.deepcopy(budget_declaration), "inherited_r17_phase_seconds": prior["phase_seconds"],
        "inherited_r17_constrained_preload": prior["constrained_preload"], "inherited_r17_force_control": prior["force_control"],
        "effective_phase_seconds": phase, "effective_constrained_preload": preload}
    protocol = copy.deepcopy(prior)
    protocol.update(format=audit.FORMAT, runner_revision=audit.FORMAT, r18_close_deadline_experiment=declaration,
        source_hashes_before=hashes, source_hashes_after=hashes, phase_seconds=phase, constrained_preload=preload,
        inherited_r17_phase_seconds=prior["phase_seconds"], inherited_r17_constrained_preload=prior["constrained_preload"],
        inherited_r17_force_control=prior["force_control"])
    protocol["schedule"][0].update(runner_revision=audit.FORMAT, close_seconds=32., arm_interpolation_seconds=16.)
    streams, model, native_audit = streams_and_model()
    row = {**protocol["schedule"][0], "path": "case", "success": False, "held_lift": False, "runner_error": None,
        "native_audit": native_audit, "scene_sha256_before_load": digest(scene), "scene_sha256_at_finalize": digest(scene),
        "scene_unchanged_since_load": True, "effective_task_phase_seconds": phase, "effective_runner_phase_seconds": phase,
        "derived_controller_sha256": digest(controller), "arm_source_transform": {**audit.TRANSFORM,
            "baseline_derived_controller_sha256": "synthetic-baseline-derived", "derived_controller_sha256": digest(controller)}, "artifacts": {}}
    for key, records in streams.items():
        path = case / (key + ".json.gz")
        path.write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
        row["artifacts"][key] = {"path": path.name, "sha256": base.digest(path), "bytes": path.stat().st_size}
    with zipfile.ZipFile(directory / "frozen_source.zip", "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    summary = {"complete": True, "runner_revision": audit.FORMAT, "close_seconds": 32., "arm_interpolation_seconds": 16.,
        "protocol": protocol, "rows": [row]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (case / "metadata.json").write_text(json.dumps(row))
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


def test_full_r18_wrapper_reuses_r17_commands_forces_and_binds_real_effective_deadline(deadline_directory):
    parent_hash = base.digest(r17.__file__)
    receipt = audit.audit_directory(deadline_directory)
    assert receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"]
    assert receipt["all_native_close_spans_within_declared_deadline"]
    assert receipt["cases"][0]["native_close_spans"][0]["duration_s"] == .006
    assert receipt["cases"][0]["actuation"]["cache_states_crosschecked_against_previous_native_sample"] == 3
    assert receipt["additional_bias_limit_rad"] == .04 and receipt["close_seconds"] == 32.
    assert not receipt["target_qpos_writes_independently_proven"] and not receipt["physical_prefix_independently_compared"]
    assert base.BIAS_LIMIT_RAD == .02 and base.digest(r17.__file__) == parent_hash


@pytest.mark.parametrize("change", [
    lambda s: s["protocol"].update(format="public_ycb_bias_budget_development_r17"),
    lambda s: s.update(close_seconds=16.),
    lambda s: s["protocol"]["r18_close_deadline_experiment"].update(format="unknown"),
    lambda s: s["protocol"]["r18_close_deadline_experiment"].pop("baseline_close_seconds"),
    lambda s: s["protocol"]["r18_close_deadline_experiment"].update(native_timestep_s=.004),
    lambda s: s["protocol"]["r18_close_deadline_experiment"].update(arm_interpolation_seconds=32.),
    lambda s: s["protocol"]["r18_close_deadline_experiment"]["changed_parameters"].update(gain={"before": 1, "after": 2}),
    lambda s: s["protocol"]["r18_close_deadline_experiment"]["prefix_validation"].update(status="PASSED"),
    lambda s: s["protocol"]["r18_close_deadline_experiment"]["preflight_input_sha256"].pop(audit.R17_SUMMARY),
    lambda s: s["protocol"]["joint_admittance"].update(additional_bias_limit_rad=.1),
    lambda s: s["protocol"].update(scoring="changed task"),
    lambda s: s["protocol"]["phase_seconds"].update(TRIAL_LIFT=1.),
    lambda s: s["rows"][0]["effective_task_phase_seconds"].update(CLOSE=16.),
    lambda s: s["rows"][0]["arm_source_transform"].update(hand_fraction_replacements=1),
    lambda s: s["rows"][0]["arm_source_transform"].update(baseline_derived_controller_sha256="wrong"),
    lambda s: s["rows"][0].update(additional_bias_limit_rad=.1),
    lambda s: s["protocol"]["source_hashes_after"].update({audit.R18_SOURCE: "changed"}),
    lambda s: s["protocol"]["source_hashes_before"].pop(audit.R17_SUMMARY),
])
def test_wrong_clock_format_budget_baseline_hash_or_task_contract_rejected(deadline_directory, change):
    rewrite(deadline_directory, change)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(deadline_directory)


@pytest.mark.parametrize("name,change", [
    ("bias_recovery", lambda r: r[2].update(format="r18_new_controller_write")),
    ("bias_recovery", lambda r: r[2].update(additional_bias_limit_rad=.1)),
    ("bias_recovery", lambda r: r[2].update(timestamp_s=.006)),
    ("native", lambda r: r[2]["motor_commands"].__setitem__(0, .3)),
    ("native", lambda r: r[2]["native_actuation"]["actuator_force"].__setitem__(0, 9.)),
    ("native", lambda r: r[2].update(timestamp=.008)),
])
def test_hash_consistent_corrupt_original_r17_event_or_cache_contract_rejected(deadline_directory, name, change):
    rewrite_stream(deadline_directory, name, change)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(deadline_directory)


def test_scene_locally_bound_but_different_from_r17_is_rejected(deadline_directory):
    path = deadline_directory / "case/scene.xml"
    path.write_text("<mujoco><compiler/><option gravity='0 0 -1'/></mujoco>")
    rewrite(deadline_directory, lambda s: s["rows"][0].update(scene_sha256_before_load=base.digest(path),
        scene_sha256_at_finalize=base.digest(path)))
    with pytest.raises(ValueError, match="scene binding"):
        audit.audit_directory(deadline_directory)


def test_missing_or_unmanifested_baseline_source_archive_entry_rejected(deadline_directory):
    path = deadline_directory / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist() if name != audit.R17_SUMMARY}
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(deadline_directory)


def test_hash_consistent_changed_r18_runner_source_is_rejected(deadline_directory):
    path = deadline_directory / "frozen_source.zip"
    with zipfile.ZipFile(path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist()}
    sources[audit.R18_SOURCE] += b"\n# replaced runner with identical constants\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    replacement_hash = digest(sources[audit.R18_SOURCE])
    rewrite(deadline_directory, lambda s: [s["protocol"][key].update({audit.R18_SOURCE: replacement_hash})
        for key in ("source_hashes_before", "source_hashes_after")])
    with pytest.raises(ValueError, match="R18 runner source identity"):
        audit.audit_directory(deadline_directory)


@pytest.mark.parametrize("samples,accepted", [(16000, True), (16001, True), (16002, False)])
def test_native_close_span_has_explicit_single_step_deadline_tolerance(samples, accepted):
    native = [{"timestamp": (i + 1) * .002, "state": "CLOSE"} for i in range(samples)]
    if accepted:
        assert audit.native_close_spans(native, 2)[0]["native_samples"] == samples
    else:
        with pytest.raises(ValueError, match="exceeds"):
            audit.native_close_spans(native, 2)


def test_native_close_duration_uses_independent_phase_spans_not_whole_run_time():
    native = [{"timestamp": (i + 1) * .002, "state": state}
        for i, state in enumerate(("SEARCH", "CLOSE", "CLOSE", "RELEASE", "SEARCH", "CLOSE"))]
    spans = audit.native_close_spans(native, 2)
    assert [row["duration_s"] for row in spans] == [.004, .002]
    with pytest.raises(ValueError, match="too many"):
        audit.native_close_spans(native, 1)
