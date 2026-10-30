"""Headless numerical validation; these checks are not hardware calibration.

Primitive fixtures isolate gravity, contact load, and Coulomb sliding. The
assembled KUKA/Sharpa models are separately checked for parameter consistency
and total motor torque saturation. No renderer or privileged task state is used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.pick_place import RobotGravityCompensation
from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import OBJECT_KINDS, build_scene


@dataclass(frozen=True)
class FixtureParameters:
    mass_kg: float = 0.349
    gravity_m_s2: float = 9.81
    timestep_s: float = 0.002
    sliding_friction: float = 0.4
    half_size_m: tuple[float, float, float] = (0.02, 0.02, 0.02)
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass
class Check:
    name: str
    passed: bool
    metrics: dict
    criteria: dict
    parameters: dict
    scope: str


def _numbers(values) -> str:
    return " ".join(format(float(value), ".12g") for value in values)


def fixture_model(parameters: FixtureParameters, *, floor: bool = True,
                  rotation_locked: bool = False, initial_z: float = 0.03) -> mujoco.MjModel:
    """A box/plane fixture; optional slides isolate friction from rolling."""
    if parameters.mass_kg <= 0 or parameters.timestep_s <= 0:
        raise ValueError("mass and timestep must be positive")
    root = ET.Element("mujoco", model="physics_validation_primitive")
    ET.SubElement(root, "option", timestep=str(parameters.timestep_s),
                  gravity=_numbers((0, 0, -parameters.gravity_m_s2)),
                  integrator="implicitfast", solver="Newton", cone="elliptic",
                  iterations="100", tolerance="1e-8")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0")
    ET.SubElement(default, "geom", condim="3",
                  friction=_numbers((parameters.sliding_friction, 0, 0)),
                  solref=_numbers(parameters.solref), solimp=_numbers(parameters.solimp))
    world = ET.SubElement(root, "worldbody")
    if floor:
        ET.SubElement(world, "geom", name="support", type="plane", size="2 2 0.1")
    body = ET.SubElement(world, "body", name="block", pos=_numbers((0, 0, initial_z)))
    if rotation_locked:
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            ET.SubElement(body, "joint", type="slide", axis=_numbers(axis))
    else:
        ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", name="block_geom", type="box",
                  size=_numbers(parameters.half_size_m), mass=str(parameters.mass_kg))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _normal_load(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    support = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support")
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
    wrench = np.empty(6)
    total = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == {support, target}:
            mujoco.mj_contactForce(model, data, index, wrench)
            total += abs(float(wrench[0]))
    return total


def check_freefall(parameters: FixtureParameters = FixtureParameters()):
    model = fixture_model(parameters, floor=False, initial_z=1.0)
    data = mujoco.MjData(model)
    duration = 0.25
    samples = []
    for _ in range(round(duration / parameters.timestep_s)):
        mujoco.mj_step(model, data)
        t = float(data.time)
        expected_z = 1.0 - 0.5 * parameters.gravity_m_s2 * t * t
        expected_vz = -parameters.gravity_m_s2 * t
        samples.append(dict(time_s=t, z_m=float(data.qpos[2]),
                            analytic_z_m=expected_z, vz_m_s=float(data.qvel[2]),
                            analytic_vz_m_s=expected_vz))
    z_error = max(abs(row["z_m"] - row["analytic_z_m"]) for row in samples)
    v_error = max(abs(row["vz_m_s"] - row["analytic_vz_m_s"]) for row in samples)
    # Semi-implicit Euler position differs by g*t*dt/2 in constant gravity.
    position_bound = parameters.gravity_m_s2 * data.time * parameters.timestep_s * 0.51
    result = Check("freefall", bool(z_error <= position_bound and v_error <= 1e-9),
                   dict(max_position_error_m=z_error, max_velocity_error_m_s=v_error,
                        final_time_s=float(data.time)),
                   dict(max_position_error_m=position_bound,
                        max_velocity_error_m_s=1e-9), asdict(parameters),
                   "Analytic constant-gravity primitive; collision-free.")
    return result, samples


def check_friction(parameters: FixtureParameters = FixtureParameters()):
    model = fixture_model(parameters, rotation_locked=True)
    data = mujoco.MjData(model)
    for _ in range(round(0.5 / parameters.timestep_s)):
        mujoco.mj_step(model, data)
    initial_position = float(data.qpos[0])
    start = float(data.time)
    initial_velocity = 0.5
    data.qvel[0] = initial_velocity
    samples = []
    stop_time = None
    for _ in range(round(0.5 / parameters.timestep_s)):
        mujoco.mj_step(model, data)
        elapsed = float(data.time) - start
        if stop_time is None and abs(float(data.qvel[0])) < 1e-3:
            stop_time = elapsed
        samples.append(dict(time_s=elapsed, x_m=float(data.qpos[0]) - initial_position,
                            vx_m_s=float(data.qvel[0]), normal_force_n=_normal_load(model, data)))
    analytic_distance = initial_velocity**2 / (2 * parameters.sliding_friction * parameters.gravity_m_s2)
    analytic_time = initial_velocity / (parameters.sliding_friction * parameters.gravity_m_s2)
    distance = samples[-1]["x_m"]
    relative_distance_error = abs(distance - analytic_distance) / analytic_distance
    velocity = abs(samples[-1]["vx_m_s"])
    result = Check("friction_slide", bool(relative_distance_error <= 0.12 and velocity <= 1e-3
                                           and stop_time is not None and stop_time <= analytic_time + 0.05),
                   dict(stopping_distance_m=distance, analytic_stopping_distance_m=analytic_distance,
                        stopping_time_s=stop_time, analytic_stopping_time_s=analytic_time,
                        distance_relative_error=relative_distance_error, final_speed_m_s=velocity),
                   dict(max_distance_relative_error=0.12, max_final_speed_m_s=1e-3,
                        max_stopping_time_s=analytic_time + 0.05),
                   dict(**asdict(parameters), initial_speed_m_s=initial_velocity,
                        orientation="locked by three translational joints"),
                   "Coulomb sliding on horizontal plane; rotation deliberately isolated.")
    return result, samples


def check_static_load(parameters: FixtureParameters = FixtureParameters()):
    model = fixture_model(parameters)
    data = mujoco.MjData(model)
    samples = []
    for _ in range(round(1.0 / parameters.timestep_s)):
        mujoco.mj_step(model, data)
        samples.append(dict(time_s=float(data.time), z_m=float(data.qpos[2]),
                            normal_force_n=_normal_load(model, data), vz_m_s=float(data.qvel[2])))
    settled = [row for row in samples if row["time_s"] >= 0.75]
    expected = parameters.mass_kg * parameters.gravity_m_s2
    mean_load = float(np.mean([row["normal_force_n"] for row in settled]))
    max_load_error = max(abs(row["normal_force_n"] - expected) for row in settled)
    speed = max(abs(row["vz_m_s"]) for row in settled)
    result = Check("static_support_load", bool(max_load_error / expected < 0.01 and speed < 1e-4),
                   dict(expected_weight_n=expected, mean_support_load_n=mean_load,
                        max_settled_relative_load_error=max_load_error / expected,
                        max_settled_vertical_speed_m_s=speed),
                   dict(max_settled_relative_load_error=0.01,
                        max_settled_vertical_speed_m_s=1e-4), asdict(parameters),
                   "Actual MuJoCo contact wrench; 0.75-1.0 s settled interval.")
    return result, samples


def _loaded_contact_trace(parameters: FixtureParameters):
    model = fixture_model(parameters, initial_z=0.035)
    data = mujoco.MjData(model)
    samples = []

    def apply_load():
        data.qfrc_applied[:3] = (0.1, 0, -0.3 * (1 - np.cos(2 * np.pi * data.time)))

    for _ in range(round(1.0 / parameters.timestep_s)):
        apply_load()
        mujoco.mj_step(model, data)
        apply_load()
        mujoco.mj_forward(model, data)
        samples.append(dict(time_s=float(data.time), x_m=float(data.qpos[0]),
                            z_m=float(data.qpos[2]), normal_force_n=_normal_load(model, data),
                            downward_load_n=-float(data.qfrc_applied[2])))
    return samples


def check_contact_convergence(parameters: FixtureParameters = FixtureParameters()):
    coarse_parameters = FixtureParameters(**dict(asdict(parameters), timestep_s=0.002))
    fine_parameters = FixtureParameters(**dict(asdict(parameters), timestep_s=0.001))
    coarse = _loaded_contact_trace(coarse_parameters)
    fine = _loaded_contact_trace(fine_parameters)
    aligned_pairs = list(zip(coarse, fine[1::2]))
    settled_pairs = [(a, b) for a, b in aligned_pairs if a["time_s"] >= 0.2]
    z_rmse = float(np.sqrt(np.mean([(a["z_m"] - b["z_m"])**2 for a, b in settled_pairs])))
    force_rmse = float(np.sqrt(np.mean([(a["normal_force_n"] - b["normal_force_n"])**2
                                       for a, b in settled_pairs])))
    final_z_difference = abs(coarse[-1]["z_m"] - fine[-1]["z_m"])
    coarse_impact = [row for row in coarse if row["time_s"] <= 0.2 + 1e-9]
    fine_impact = [row for row in fine if row["time_s"] <= 0.2 + 1e-9]
    coarse_peak = max(row["normal_force_n"] for row in coarse_impact)
    fine_peak = max(row["normal_force_n"] for row in fine_impact)
    peak_relative_difference = abs(coarse_peak - fine_peak) / fine_peak
    coarse_impulse = float(np.trapezoid([row["normal_force_n"] for row in coarse_impact],
                                       [row["time_s"] for row in coarse_impact]))
    fine_impulse = float(np.trapezoid([row["normal_force_n"] for row in fine_impact],
                                     [row["time_s"] for row in fine_impact]))
    impulse_relative_difference = abs(coarse_impulse - fine_impulse) / fine_impulse
    result = Check("contact_timestep_convergence",
                   bool(z_rmse < 0.0003 and force_rmse < 0.12 and final_z_difference < 0.0001
                        and peak_relative_difference < 0.15 and impulse_relative_difference < 0.05),
                   dict(settled_z_rmse_m=z_rmse, settled_force_rmse_n=force_rmse,
                        final_z_difference_m=final_z_difference,
                        impact_peak_2ms_n=coarse_peak, impact_peak_1ms_n=fine_peak,
                        impact_peak_relative_difference=peak_relative_difference,
                        impact_impulse_2ms_n_s=coarse_impulse, impact_impulse_1ms_n_s=fine_impulse,
                        impact_impulse_relative_difference=impulse_relative_difference),
                   dict(max_settled_z_rmse_m=0.0003, max_settled_force_rmse_n=0.12,
                        max_final_z_difference_m=0.0001,
                        max_impact_peak_relative_difference=0.15,
                        max_impact_impulse_relative_difference=0.05),
                   dict(coarse=asdict(coarse_parameters), fine=asdict(fine_parameters),
                        record_every_physics_step=True, aligned_sample_interval_s=0.002,
                        settled_comparison_start_s=0.2, impact_interval_s=[0.0, 0.2],
                        lateral_force_n=0.1, downward_force="0.3*(1-cos(2*pi*t)) N"),
                   "Controlled rigid box/plane contact only; excludes full grasp convergence.")
    return result, {"2ms": coarse, "1ms": fine}


def check_assembled_models(directory: Path, physics_profile: str):
    from kuka_sharpa_grasp.physics import get_physics_profile

    profile = get_physics_profile(physics_profile)
    checks = []
    compiled_parameters = {}
    reference_model = None
    for object_kind in OBJECT_KINDS:
        path = build_scene(directory / f"{object_kind}.xml", object_kind=object_kind,
                           physics_profile=physics_profile)
        model = mujoco.MjModel.from_xml_path(str(path))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        expected = profile.object_parameters(object_kind)
        entry = dict(mass_kg=float(model.body_mass[body]), inertia_kg_m2=model.body_inertia[body].tolist(),
                     center_of_mass_m=model.body_ipos[body].tolist(), friction=model.geom_friction[geom].tolist(),
                     condim=int(model.geom_condim[geom]), solref=model.geom_solref[geom].tolist(),
                     solimp=model.geom_solimp[geom].tolist(), authored=asdict(expected),
                     timestep_s=float(model.opt.timestep), gravity_m_s2=model.opt.gravity.tolist(),
                     solver=int(model.opt.solver), integrator=int(model.opt.integrator),
                     solver_iterations=int(model.opt.iterations), solver_tolerance=float(model.opt.tolerance))
        compiled_parameters[object_kind] = entry
        passed = (np.isclose(entry["mass_kg"], expected.mass_kg, atol=1e-12)
                  and np.all(model.body_inertia[body] > 0)
                  and np.allclose(model.geom_friction[geom], expected.friction, atol=1e-12)
                  and entry["condim"] == expected.condim
                  and np.allclose(model.geom_solref[geom], expected.solref)
                  and np.allclose(model.geom_solimp[geom], expected.solimp))
        checks.append(Check(f"compiled_parameters_{object_kind}", bool(passed), entry,
                            dict(mass_kg=expected.mass_kg, positive_inertia=True,
                                 friction=list(expected.friction), condim=expected.condim),
                            dict(physics_profile=physics_profile),
                            "Parameter implementation consistency, not measured hardware fidelity."))
        if object_kind == "box":
            reference_model = model

    model = reference_model
    data = mujoco.MjData(model)
    rollout = GraspRollout(model, data)
    body = rollout.target_body_id
    mass = float(model.body_mass[body])
    inertia = model.body_inertia[body].copy()
    errors = []
    for scale in (2.0, 0.5, 1.0):
        rollout.configure_object(mass=mass * scale)
        errors.append(float(np.max(np.abs(model.body_inertia[body] / inertia - scale))))
    restored = np.allclose(model.body_inertia[body], inertia, atol=1e-14)
    checks.append(Check("object_mass_inertia_scaling", bool(max(errors) < 1e-12 and restored),
                        dict(max_ratio_error=max(errors), restored=bool(restored)),
                        dict(max_ratio_error=1e-12, restore_original=True),
                        dict(mass_scales=[2.0, 0.5, 1.0], reference_mass_kg=mass,
                             reference_inertia_kg_m2=inertia.tolist()),
                        "Mass changes preserve the same geometry and homogeneous density assumption."))

    gravity = RobotGravityCompensation(model)
    arm_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                  for i in range(1, 8)]
    limits = model.jnt_actfrcrange[arm_joints].copy()
    limited = model.jnt_actfrclimited[arm_joints].astype(bool)
    samples = []
    for sign in (-1, 1):
        rollout.reset()
        for index, aid in enumerate(rollout.arm_actuator_ids):
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = hi if sign > 0 else lo
            data.qpos[model.jnt_qposadr[arm_joints[index]]] = lo if sign > 0 else hi
        gravity.apply(data)
        first_command = data.ctrl.copy()
        gravity.apply(data)
        command_drift = float(np.max(np.abs(data.ctrl - first_command)))
        mujoco.mj_forward(model, data)
        actual = data.qfrc_actuator[:7] + data.qfrc_applied[:7]
        samples.append(dict(sign=sign, total_motor_torque_nm=actual.tolist(),
                            applied_bypass_torque_nm=data.qfrc_applied[:7].tolist(),
                            repeat_apply_command_drift=command_drift))
    saturated = all(np.allclose(sample["total_motor_torque_nm"], limits[:, 1 if sample["sign"] > 0 else 0],
                                atol=1e-8) for sample in samples)
    bypass_zero = all(np.allclose(sample["applied_bypass_torque_nm"], 0, atol=1e-12) for sample in samples)
    idempotent = all(sample["repeat_apply_command_drift"] < 1e-12 for sample in samples)
    specified = (profile.arm_torque_limits_nm is not None
                 and np.allclose(limits[:, 1], profile.arm_torque_limits_nm, atol=1e-12)
                 and np.allclose(limits[:, 0], -np.asarray(profile.arm_torque_limits_nm), atol=1e-12))
    checks.append(Check("total_arm_motor_torque_saturation",
                        bool(np.all(limited) and saturated and bypass_zero and idempotent and specified),
                        dict(all_joints_limited=bool(np.all(limited)), samples=samples,
                             bounds_nm=limits.tolist(), matches_profile_limits=bool(specified)),
                        dict(saturates_both_signs=True, no_applied_torque_bypass=True,
                             repeat_apply_command_drift_max=1e-12),
                        dict(physics_profile=physics_profile,
                             includes="position servo plus gravity feedforward"),
                        "Compiled arm motor limits; contact/environment forces are not motor torques."))
    return checks, dict(profile=asdict(profile), compiled_objects=compiled_parameters)


def _asset_hashes(model_directory: Path) -> dict[str, str]:
    paths = set()
    for path in model_directory.glob("*.xml"):
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        mesh_directory = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
        if not mesh_directory.is_absolute():
            mesh_directory = path.parent / mesh_directory
        for mesh in root.findall("./asset/mesh"):
            if mesh.get("file"):
                paths.add((mesh_directory / mesh.get("file")).resolve())
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(output: Path, traces: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "not generated: matplotlib unavailable; numerical checks and CSV remain complete"
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    free = traces["freefall"]
    axes[0, 0].plot([r["time_s"] for r in free], [r["z_m"] for r in free], label="MuJoCo")
    axes[0, 0].plot([r["time_s"] for r in free], [r["analytic_z_m"] for r in free], "--", label="Analytic")
    axes[0, 0].set(title="Freefall", ylabel="Height (m)")
    slide = traces["friction_slide"]
    axes[0, 1].plot([r["time_s"] for r in slide], [r["vx_m_s"] for r in slide], label="Sliding speed")
    axes[0, 1].plot([r["time_s"] for r in slide],
                    [max(0, 0.5 - FixtureParameters().sliding_friction *
                         FixtureParameters().gravity_m_s2 * r["time_s"]) for r in slide],
                    "--", label="Analytic Coulomb")
    axes[0, 1].set(title="Coulomb slide", ylabel="Speed (m/s)")
    load = traces["static_support_load"]
    axes[1, 0].plot([r["time_s"] for r in load], [r["normal_force_n"] for r in load], label="Contact load")
    axes[1, 0].axhline(FixtureParameters().mass_kg * FixtureParameters().gravity_m_s2,
                       linestyle="--", color="black", label="mg")
    axes[1, 0].set(title="Static support", ylabel="Normal force (N)")
    for label, rows in traces["contact_timestep_convergence"].items():
        axes[1, 1].plot([r["time_s"] for r in rows], [r["normal_force_n"] for r in rows], label=label)
    axes[1, 1].set(title="Controlled contact convergence", ylabel="Normal force (N)")
    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("MuJoCo numerical validation; not hardware calibration", fontsize=12)
    figure.savefig(output / "physics_checks.png", dpi=160)
    plt.close(figure)
    return "physics_checks.png"


def run_validation(output: Path, physics_profile: str = "physical-v6") -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = []
    traces = {}
    for function in (check_freefall, check_friction, check_static_load, check_contact_convergence):
        result, rows = function()
        checks.append(result)
        traces[result.name] = rows
        if isinstance(rows, dict):
            for label, series in rows.items():
                _write_csv(output / f"{result.name}_{label}.csv", series)
        else:
            _write_csv(output / f"{result.name}.csv", rows)
    model_directory = output / "models"
    model_directory.mkdir()
    assembled, model_parameters = check_assembled_models(model_directory, physics_profile)
    checks.extend(assembled)
    source_paths = [Path(__file__).resolve(), *sorted(path for path in (PROJECT / "src" / "kuka_sharpa_grasp").glob("*.py")
                                                    if not path.name.startswith('._')),
                    PROJECT / "vendor/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml",
                    PROJECT / "vendor/mujoco_menagerie/sharpa_wave/left_hand.xml"]
    hashes = {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    report = dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  purpose="Numerical and parameter-consistency validation; not real hardware calibration",
                  physics_backend="classic_mujoco_cpu", renderer="none",
                  platform=platform.platform(), python=sys.version, mujoco_version=mujoco.__version__,
                  numpy_version=np.__version__,
                  deterministic=True, stochastic_seed=None, source_sha256=hashes,
                  model_xml_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(model_directory.glob("*.xml"))},
                  model_mesh_sha256=_asset_hashes(model_directory),
                  model_parameters=model_parameters,
                  checks=[asdict(check) for check in checks], passed=all(check.passed for check in checks),
                  passed_checks=sum(check.passed for check in checks), total_checks=len(checks),
                  plot=_write_plot(output, traces),
                  limitations=["Primitive checks do not establish task grasp success or full-scene timestep convergence.",
                               "No material, camera, or force-sensor calibration against physical hardware.",
                               "Object inertia uses a homogeneous-density approximation."])
    _write_csv(output / "checks.csv", [dict(name=check.name, passed=check.passed,
               metrics=json.dumps(check.metrics), criteria=json.dumps(check.criteria)) for check in checks])
    report["wall_seconds"] = time.perf_counter() - started
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/physics_validation_v6")
    parser.add_argument("--physics-profile", default="physical-v6")
    args = parser.parse_args(argv)
    try:
        report = run_validation(args.output, args.physics_profile)
    except Exception as error:
        print(f"Physics validation did not complete: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(f"{report['passed_checks']}/{report['total_checks']} checks passed; {args.output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
