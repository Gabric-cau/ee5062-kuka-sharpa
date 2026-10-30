"""Numerical references and composed-model contracts for the physics audit."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from validate_physics import (FixtureParameters, check_contact_convergence,
                              check_freefall, check_friction, check_static_load,
                              main, run_validation)


def test_freefall_matches_analytic_velocity_and_position_error_halves_with_step():
    coarse, _ = check_freefall(FixtureParameters(timestep_s=0.002))
    fine, _ = check_freefall(FixtureParameters(timestep_s=0.001))
    assert coarse.passed and fine.passed
    assert coarse.metrics["max_velocity_error_m_s"] < 1e-10
    assert np.isclose(coarse.metrics["max_position_error_m"] /
                      fine.metrics["max_position_error_m"], 2.0, atol=1e-10)


def test_higher_friction_shortens_measured_stopping_distance():
    low, _ = check_friction(FixtureParameters(sliding_friction=0.2))
    high, _ = check_friction(FixtureParameters(sliding_friction=0.6))
    assert low.passed and high.passed
    assert low.metrics["stopping_distance_m"] > 2.5 * high.metrics["stopping_distance_m"]


def test_contact_wrench_supports_weight_and_loaded_contact_converges():
    load, _ = check_static_load(FixtureParameters(mass_kg=0.411))
    assert load.passed
    assert np.isclose(load.metrics["mean_support_load_n"], 0.411 * 9.81, rtol=0.01)
    convergence, traces = check_contact_convergence()
    assert convergence.passed
    assert len(traces["2ms"]) == 500
    assert len(traces["1ms"]) == 1000
    assert convergence.metrics["impact_peak_2ms_n"] > 5
    assert convergence.metrics["impact_peak_1ms_n"] > 5


def test_complete_audit_records_real_compiled_models_and_preserves_existing_run(tmp_path):
    output = tmp_path / "audit"
    report = run_validation(output)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    assert report["total_checks"] == 10
    assert report["renderer"] == "none"
    assert report["physics_backend"] == "classic_mujoco_cpu"
    assert report["model_parameters"]["compiled_objects"]["ycb_can"]["mass_kg"] == 0.349
    assert len(report["model_xml_sha256"]) == 4
    assert report["model_mesh_sha256"]
    loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert loaded["source_sha256"] == report["source_sha256"]
    saved = (output / "report.json").read_bytes()
    assert main(["--output", str(output)]) == 2
    assert (output / "report.json").read_bytes() == saved


def test_validation_cli_returns_failure_for_unlimited_legacy_motor_model(tmp_path):
    output = tmp_path / "legacy"
    assert main(["--output", str(output), "--physics-profile", "legacy-v5"]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["name"] == "total_arm_motor_torque_saturation")
    assert not check["passed"]
    assert not check["metrics"]["all_joints_limited"]
