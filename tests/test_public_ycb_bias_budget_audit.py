"""R17 budget isolation and same-cache force/transmission reconstruction."""

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
import audit_public_ycb_bias_budget as audit
import audit_public_ycb_bias_recovery as base


TIMING = "Cached last completed dynamics evaluation; post-integration observation, not post-qpos recomputation."
SPECS = [base.JointSpec(f"left_{finger}_MCP_FE", finger, index, 0., 2., 1.)
    for index, finger in enumerate(base.ACTIVE_FINGERS)]


def gate():
    return {"preload_ready": False, "preload_start_s": None, "probe_eligible_seconds": 0.,
        "probe_confirmed_seconds": 0., "probe_active": False, "payload_confirmed": False,
        "pressure_window_samples": 0}


def native_sample(index, command, actual):
    return {"timestamp": index * .002, "state": "SEARCH" if index == 1 else "CLOSE",
        "motor_commands": command, "actual_hand_joints": {spec.name:
            {"position_rad": actual[i], "velocity_rad_s": 0.} for i, spec in enumerate(SPECS)}}


def budget_fixture(budget=.04, *, drift=True):
    native = [native_sample(i + 1, [command] * 3, [.15 if drift and i else .18, .18, .18])
        for i, command in enumerate((.2, .2001, .2001, .2002))]
    capture = {"timestamp_s": .002, "event": "ACTIVATION_CAPTURE",
        "captured_command_rad": {spec.name: .2 for spec in SPECS},
        "captured_command_minus_actual_rad": {spec.name: .2 - .18 for spec in SPECS}}
    events, inherited = [], [capture]
    for index in range(4):
        fresh = index in (1, 3)
        mode = "SCALAR_ACQUISITION" if index == 0 else ("BIAS_RECOVERY" if index == 2 else "JOINT_CONTROL")
        event = {"format": "r17_bias_budget_write_v1", "inherited_write_format": "r16_bias_recovery_write_v1",
            "additional_bias_limit_rad": budget, "timestamp_s": index * .002,
            "post_step_timestamp_s": (index + 1) * .002, "dt_s": .002 if index else 0.,
            "phase": "SEARCH" if not index else "CLOSE", "mode": mode,
            "fresh_target": fresh, "final_safety_override": False,
            "recovery_active": index == 2, "failure_latched": False, "preload_inhibited": index == 2,
            "write_verification_passed": True, "write_verification_violations": [], "abort_reason": None,
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
            command, applied = before["motor_commands"][spec.actuator], after["motor_commands"][spec.actuator]
            rate, window = [command - .6 * .002, command + .12 * .002], [actual + .02 - budget, actual + .02 + budget]
            detail = {"additional_bias_limit_rad": budget, "before_actual_rad": actual,
                "before_command_rad": command, "previous_reference_rad": command,
                "fixed_capture_bias_rad": .2 - .18, "capture_timestamp_s": .002, "range_rad": [0., 2.],
                "rate_interval_rad": rate, "bias_interval_rad": window,
                "intersection_rad": [max(0., rate[0], window[0]), min(2., rate[1], window[1])],
                "candidate_command_rad": applied, "applied_command_rad": applied}
            event["per_finger"].append({"finger": spec.finger, "mode": "NOMINAL" if fresh else "BIAS_RECOVERY",
                "joint_names": [spec.name], "joints": {spec.name: detail},
                "compression_jacobian_m_rad": [[1.]] if fresh else None,
                "desired_normal_velocity_m_s": [.001] if fresh else None,
                "predicted_applied_normal_velocity_m_s": [(applied - command) / .002] if fresh else None,
                "nominal_direction_satisfied": True if fresh else None})
    return {"native": native, "events": events, "joint_events": inherited, "specs": SPECS,
        "close_rate": .12, "relief_rate": .6,
        "preload_events": [{"timestamp_s": .006, "ready": False, "continuous_seconds": 0.}],
        "probe_events": [{"timestamp_s": .006, "probe_active": False, "admission": False}],
        "native_audit": {"integration_steps": 4, "native_samples": 4,
            "initial_target_qpos": [0., 0., .38, 1., 0., 0., 0.],
            "external_target_qpos_writes": 0, "write_events": []}}


@pytest.mark.parametrize("budget", [.02, .04, .10])
def test_budget_is_per_case_and_frozen_r16_globals_and_file_are_unchanged(budget):
    before = base.digest(base.__file__)
    receipt = audit.reconstruct_budget_case(**budget_fixture(budget, drift=budget > .02), budget_rad=budget)
    assert receipt["additional_bias_limit_rad"] == budget
    assert receipt["baseline_additional_bias_limit_rad"] == .02
    assert receipt["original_intersection_satisfied_for_all_audited_writes"]
    assert base.BIAS_LIMIT_RAD == .02 and base.digest(base.__file__) == before
    if budget > .02:
        assert receipt["maxima"]["absolute_added_bias_rad"] > .02


@pytest.mark.parametrize("change", [
    lambda x: x["events"][2].update(additional_bias_limit_rad=.02),
    lambda x: x["events"][2].pop("additional_bias_limit_rad"),
    lambda x: x["events"][2].update(format="r16_bias_recovery_write_v1"),
    lambda x: x["events"][2].update(inherited_write_format="unknown"),
    lambda x: x["events"][2]["per_finger"][0]["joints"][SPECS[0].name].update(additional_bias_limit_rad=.1),
    lambda x: x["events"][2]["per_finger"][0]["joints"][SPECS[0].name].update(bias_interval_rad=[.15, .19]),
    lambda x: x["events"][2].update(timestamp_s=.006),
    lambda x: x["native"][2]["motor_commands"].__setitem__(0, .3),
    lambda x: x["events"][2]["admission_after_step"].update(preload_ready=True),
])
def test_mismatched_budget_legacy_boxes_times_commands_or_admission_are_rejected(change):
    values = budget_fixture()
    change(values)
    with pytest.raises((ValueError, KeyError)):
        audit.reconstruct_budget_case(**values, budget_rad=.04)


@pytest.mark.parametrize("value", [True, None, "0.04", float("nan"), .03, -.04])
def test_undeclared_budget_is_rejected(value):
    with pytest.raises(ValueError):
        audit.budget_value(value)


def actuation_fixture(native=None):
    names = [item.name for item in SPECS]
    model = SimpleNamespace(nu=3, opt=SimpleNamespace(disableflags=0, disableactuator=0, integrator=3),
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
        "actuator_parameters": [audit.compiled_actuator_parameters(model, index) for index in range(3)]}
    native = copy.deepcopy(native) if native is not None else [native_sample(1, [.2] * 3, [.7] * 3)]
    for index, sample in enumerate(native):
        lengths = [native[index - 1]["actual_hand_joints"][name]["position_rad"] for name in names] if index else [.18] * 3
        velocities = [native[index - 1]["actual_hand_joints"][name]["velocity_rad_s"] for name in names] if index else [.1] * 3
        forces = [10. * command - 10. * length - .2 * velocity for command, length, velocity
            in zip(sample["motor_commands"], lengths, velocities, strict=True)]
        sample["native_actuation"] = {"format": "r17_native_actuation_evaluation_v1",
            "timestamp_s": sample["timestamp"], "actuator_ids": [0, 1, 2], "joint_dof_ids": [0, 1, 2],
            "actuator_force": forces, "qfrc_actuator": np.clip(forces, -.1, .1).tolist(),
            "actuator_length": lengths, "actuator_velocity": velocities}
    return {"native": native, "metadata": metadata, "model": model, "expected_timing_scope": TIMING}


def test_cached_pd_force_and_joint_limit_are_reconstructed_without_post_qpos_substitution():
    values = actuation_fixture()
    receipt = audit.audit_actuation(**values)
    assert receipt["all_cached_force_and_transmitted_torque_consistent"]
    assert receipt["maxima"]["absolute_actuator_force"] == pytest.approx(.18)
    assert receipt["maxima"]["absolute_joint_actuator_torque_nm"] == .1
    assert all(item["joint_limit_saturated_samples"] == 1 for item in receipt["by_joint"].values())
    assert receipt["cache_states_crosschecked_against_previous_native_sample"] == 0
    assert receipt["initial_cache_state_excluded_samples"] == 1


@pytest.mark.parametrize("change", [
    lambda x: x["native"][0]["native_actuation"]["actuator_force"].__setitem__(0, .19),
    lambda x: x["native"][0]["native_actuation"]["qfrc_actuator"].__setitem__(0, .18),
    lambda x: x["native"][0]["native_actuation"].update(timestamp_s=.004),
    lambda x: x["native"][0]["native_actuation"].update(format="unknown"),
    lambda x: x["native"][0]["native_actuation"]["actuator_velocity"].pop(),
    lambda x: x["native"][0]["native_actuation"]["actuator_length"].__setitem__(0, float("nan")),
    lambda x: x["native"][0]["native_actuation"].update(actuator_ids=[1, 0, 2]),
    lambda x: x["metadata"].update(evaluation_only=False),
    lambda x: x["metadata"].update(postintegration_qpos_recomputed=True),
    lambda x: x["metadata"].update(force_evaluation_state="post_step_qpos"),
    lambda x: x["metadata"].update(timing_scope="Post-qpos force recomputation"),
    lambda x: x["metadata"]["actuator_parameters"][0]["jointactfrcrange"].__setitem__(1, .2),
    lambda x: x["metadata"].update(actuator_ids=[0, 1], joint_dof_ids=[0, 1]),
    lambda x: x["model"].opt.__setattr__("disableflags", 2048),
    lambda x: x["model"].opt.__setattr__("integrator", 1),
])
def test_wrong_force_cache_transmission_mapping_model_or_timing_is_rejected(change):
    values = actuation_fixture()
    change(values)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_actuation(**values)


def test_original_actuator_force_clamping_precedes_gear_and_joint_clamping():
    values = actuation_fixture()
    model = values["model"]
    model.actuator_forcelimited[:] = True
    model.actuator_forcerange[:] = [-.05, .05]
    model.actuator_gear[:, 0] = 3.
    values["metadata"]["actuator_parameters"] = [audit.compiled_actuator_parameters(model, i) for i in range(3)]
    values["native"][0]["native_actuation"].update(actuator_force=[.05] * 3, qfrc_actuator=[.1] * 3)
    receipt = audit.audit_actuation(**values)
    assert receipt["maxima"]["absolute_actuator_force"] == .05
    assert receipt["maxima"]["absolute_joint_actuator_torque_nm"] == .1


def test_real_mujoco_cached_force_sample_matches_compiled_transmission_without_forward():
    import mujoco
    import develop_public_ycb_bias_budget_r17 as experiment
    from public_ycb_evidence import load_evidence_model

    scene = Path(__file__).resolve().parents[1] / "results/public_ycb_bias_recovery_20261022_r16_seed3414/ycb_can_bias_recovery_r16_seed3414/scene.xml"
    model = load_evidence_model(scene)
    data = mujoco.MjData(model)
    data.ctrl[:] = .1
    recorder = experiment.TorqueNativeAudit(model, data)
    for _ in range(3):
        mujoco.mj_step(model, data)
        recorder.sample("AUDIT_CACHE_SMOKE", data)
    receipt = audit.audit_actuation(recorder.samples, recorder.as_dict()["native_actuation_metadata"],
        model, expected_timing_scope=experiment.TIMING_SCOPE)
    assert receipt["native_samples"] == 3 and receipt["actuators_per_sample"] == 22
    assert receipt["all_cached_force_and_transmitted_torque_consistent"]
    assert receipt["cache_states_crosschecked_against_previous_native_sample"] == 2


def test_force_and_cache_cannot_be_jointly_forged_to_the_wrong_native_state():
    values = actuation_fixture(budget_fixture()["native"])
    sample = values["native"][2]
    record = sample["native_actuation"]
    record["actuator_length"][0] = sample["actual_hand_joints"][SPECS[0].name]["position_rad"] + .001
    record["actuator_force"][0] = (10. * sample["motor_commands"][0]
        - 10. * record["actuator_length"][0] - .2 * record["actuator_velocity"][0])
    record["qfrc_actuator"][0] = float(np.clip(record["actuator_force"][0], -.1, .1))
    with pytest.raises(ValueError, match="previous native state"):
        audit.audit_actuation(**values)


@pytest.fixture
def budget_directory(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    (directory / "case").mkdir(parents=True)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    profile = {"ycb_can": {"contact_closure": {finger: 1. for finger in base.ACTIVE_FINGERS}}}
    sources = {"scripts/develop_public_ycb_constrained_preload_r14.py": b"# test frozen parent\n",
        "scripts/develop_public_ycb_joint_admittance_r13.py": b"JOINT_PARAMETERS = {'additional_bias_limit_rad': .02}\n",
        "scripts/develop_public_ycb_force_band.py": b"configuration = f(close_rate=.12, relief_rate=.6)\n",
        "config/profile.json": json.dumps(profile).encode()}
    inherited = {key: digest(value) for key, value in sources.items()}
    prior = {"complete": True, "protocol": {"source_changed_during_run": False,
        "source_hashes_before": inherited, "source_hashes_after": inherited}}
    sources[base.R15_SUMMARY] = json.dumps(prior).encode()
    sources[audit.R16_SOURCE] = b"RECOVERY_PARAMETERS = {'fixture': True}\n"
    base_profile = b'{"fixture_base_profile": true}'
    baseline_hashes = {key: digest(value) for key, value in sources.items()}
    baseline = {"complete": True, "protocol": {"format": "public_ycb_bias_recovery_development_r16",
        "source_hashes_before": baseline_hashes, "source_hashes_after": baseline_hashes,
        "source_changed_during_run": False, "joint_admittance": {"additional_bias_limit_rad": .02},
        "r16_bias_recovery_experiment": {"recovery_parameters": {"fixture": True},
            "r15_comparison_summary_sha256": digest(sources[base.R15_SUMMARY]),
            "base_profile_sha256": digest(base_profile)},
        "nominal_profiles": profile,
        "schedule": [{"kind": "ycb_can", "seed": 3414, "profile": "bias_recovery_r16"}]}}
    monkeypatch.setattr(base, "FROZEN_R14_SHA256", inherited["scripts/develop_public_ycb_constrained_preload_r14.py"])
    monkeypatch.setattr(base, "FROZEN_PROFILE_SHA256", inherited["config/profile.json"])
    monkeypatch.setattr(base, "FROZEN_R15_SUMMARY_SHA256", digest(sources[base.R15_SUMMARY]))
    baseline_bytes = json.dumps(baseline).encode()
    sources.update({audit.R16_SUMMARY: baseline_bytes,
        audit.R16_CORE: Path(base.__file__).read_bytes(),
        audit.R17_SOURCE: ("TIMING_SCOPE = " + repr(TIMING) + "\n").encode(),
        "config/public_ycb_sidewall_20261021.json": base_profile})
    hashes = {key: digest(value) for key, value in sources.items()}
    monkeypatch.setattr(audit, "FROZEN_R16_SOURCE_SHA256", hashes[audit.R16_SOURCE])
    monkeypatch.setattr(audit, "FROZEN_R16_SUMMARY_SHA256", digest(baseline_bytes))
    with zipfile.ZipFile(directory / "frozen_source.zip", "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    protocol = copy.deepcopy(baseline["protocol"])
    protocol.update(format="public_ycb_bias_budget_development_r17",
        source_hashes_before=hashes, source_hashes_after=hashes)
    protocol["joint_admittance"]["additional_bias_limit_rad"] = .04
    protocol["schedule"][0].update(profile="bias_budget_r17", additional_bias_limit_rad=.04)
    protocol["r17_bias_budget_experiment"] = {"format": "r17_bias_budget_development_protocol_v1",
        "additional_bias_limit_rad": .04, "baseline_additional_bias_limit_rad": .02,
        "inherited_r16_declaration": copy.deepcopy(baseline["protocol"]["r16_bias_recovery_experiment"]),
        "r16_comparison_summary_sha256": digest(baseline_bytes),
        "preflight_input_sha256": {key: hashes[key] for key in (audit.R16_SOURCE, audit.R16_SUMMARY,
            audit.R16_CORE, "config/public_ycb_sidewall_20261021.json")},
        "telemetry": {"format": "r17_native_actuation_evaluation_v1", "evaluation_only": True,
            "timing_scope": TIMING}}
    values = budget_fixture()
    force_values = actuation_fixture(values["native"])
    force_values["model"].opt.timestep = .002
    scene = directory / "case/scene.xml"
    scene.write_text("<mujoco><compiler/></mujoco>")
    row = {**protocol["schedule"][0], "path": "case", "success": False, "held_lift": False,
        "runner_error": None, "native_audit": values["native_audit"],
        "scene_sha256_before_load": base.digest(scene), "scene_sha256_at_finalize": base.digest(scene),
        "scene_unchanged_since_load": True, "artifacts": {}}
    row["native_audit"].update(native_actuation_metadata=force_values["metadata"], native_actuation_samples=4)
    streams = {"native": force_values["native"], "bias_recovery": values["events"],
        "joint_admittance": values["joint_events"], "preload": values["preload_events"],
        "load_transfer_probe": values["probe_events"]}
    for key, records in streams.items():
        path = directory / "case" / f"{key}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
        row["artifacts"][key] = {"path": path.name, "sha256": base.digest(path), "bytes": path.stat().st_size}
    summary = {"complete": True, "protocol": protocol, "rows": [row]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "case/metadata.json").write_text(json.dumps(row))
    import public_ycb_evidence
    monkeypatch.setattr(public_ycb_evidence, "load_evidence_model", lambda _: force_values["model"])
    monkeypatch.setattr(base, "joint_specs", lambda *_: SPECS)
    return directory


def rewrite_summary(directory, change):
    summary = json.loads((directory / "summary.json").read_text())
    change(summary)
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "protocol.json").write_text(json.dumps(summary["protocol"]))
    (directory / "case/metadata.json").write_text(json.dumps(summary["rows"][0]))


def test_full_r17_directory_binds_budget_base_sources_raw_cache_model_and_timings(budget_directory):
    receipt = audit.audit_directory(budget_directory)
    assert receipt["all_evidence_consistent"] and receipt["all_declared_command_bounds_satisfied"]
    assert receipt["additional_bias_limit_rad"] == .04
    assert receipt["baseline_additional_bias_limit_rad"] == .02
    assert receipt["cases"][0]["actuation"]["cache_states_crosschecked_against_previous_native_sample"] == 3
    assert not receipt["target_qpos_writes_independently_proven"]


@pytest.mark.parametrize("change", [
    lambda x: x["protocol"]["joint_admittance"].update(additional_bias_limit_rad=.02),
    lambda x: x["protocol"]["r17_bias_budget_experiment"].update(baseline_additional_bias_limit_rad=.04),
    lambda x: x["protocol"]["r17_bias_budget_experiment"]["inherited_r16_declaration"].update(base_profile_sha256="wrong"),
    lambda x: x["protocol"]["r17_bias_budget_experiment"]["telemetry"].update(timing_scope="post-qpos"),
    lambda x: x["protocol"]["r17_bias_budget_experiment"]["preflight_input_sha256"].update({audit.R16_CORE: "wrong"}),
    lambda x: x["rows"][0].update(additional_bias_limit_rad=.1),
    lambda x: x["rows"][0]["native_audit"].update(native_actuation_samples=3),
    lambda x: x["rows"][0]["native_audit"]["native_actuation_metadata"].update(postintegration_qpos_recomputed=True),
    lambda x: x["rows"][0]["artifacts"]["native"].update(path="../native.json.gz"),
])
def test_full_loader_rejects_wrong_budget_provenance_denominator_timing_or_path(budget_directory, change):
    rewrite_summary(budget_directory, change)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_directory(budget_directory)


def test_r17_loader_rejects_changed_raw_bytes(budget_directory):
    path = budget_directory / "case/native.json.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size/hash"):
        audit.audit_directory(budget_directory)


def test_hash_consistent_r17_direct_qpos_assignment_is_still_rejected(budget_directory):
    archive_path = budget_directory / "frozen_source.zip"
    with zipfile.ZipFile(archive_path) as archive:
        sources = {name: archive.read(name) for name in archive.namelist()}
    sources[audit.R17_SOURCE] += b"\ndef write(data):\n    data.qpos[:] = 0\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in sources.items():
            archive.writestr(name, value)
    changed_hash = hashlib.sha256(sources[audit.R17_SOURCE]).hexdigest()
    rewrite_summary(budget_directory, lambda x: [x["protocol"][key].update({audit.R17_SOURCE: changed_hash})
        for key in ("source_hashes_before", "source_hashes_after")])
    with pytest.raises(ValueError, match="direct qpos assignment"):
        audit.audit_directory(budget_directory)
