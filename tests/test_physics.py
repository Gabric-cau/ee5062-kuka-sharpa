"""Compiled physical contracts, parameter consistency, and real torque saturation."""

import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.physics import (
    ARM_TORQUE_LIMITS_NM, ARM_VELOCITY_LIMITS_RAD_S,
    arm_joint_acceleration_limits, arm_joint_velocity_limits, get_physics_profile,
    model_physics_version, native_contact_parameters, pair_friction,
)
from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import OBJECT_KINDS, build_scene


def compile_scene(tmp_path, kind="ycb_can", profile="physical-v6", timestep=None):
    return mujoco.MjModel.from_xml_path(str(build_scene(
        tmp_path / f"{profile}_{kind}.xml", kind, physics_profile=profile, timestep=timestep,
    )))


@pytest.mark.parametrize("kind", OBJECT_KINDS)
def test_v6_compiled_object_has_reference_mass_and_positive_consistent_inertia(tmp_path, kind):
    model = compile_scene(tmp_path, kind)
    reference = get_physics_profile("physical-v6").object_parameters(kind)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    assert model.body_mass[body] == pytest.approx(reference.mass_kg)
    assert np.all(np.isfinite(model.body_inertia[body]))
    assert np.all(model.body_inertia[body] > 0)
    moments = np.sort(model.body_inertia[body])
    assert moments[2] <= moments[0] + moments[1] + 1e-12
    np.testing.assert_allclose(model.geom_friction[geom], reference.friction)
    assert model.geom_condim[geom] == reference.condim == 3
    assert model_physics_version(model) == 6


def test_runtime_density_change_scales_inertia_from_reference_and_restores_pairs(tmp_path):
    model = compile_scene(tmp_path)
    rollout = GraspRollout(model, mujoco.MjData(model))
    body, geom = rollout.target_body_id, rollout.target_geom_id
    original_mass = float(model.body_mass[body])
    original_inertia = model.body_inertia[body].copy()
    original_pairs = model.pair_friction.copy()
    original_geom = model.geom_friction[geom].copy()
    rollout.configure_object(mass=original_mass * 2, friction=0.2)
    np.testing.assert_allclose(model.body_inertia[body], 2 * original_inertia)
    table_pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "target_table")
    assert model.pair_friction[table_pair, 0] == pytest.approx(math.sqrt(0.2 * 0.6))
    rollout.configure_object(mass=original_mass * 0.5, friction=0.9)
    np.testing.assert_allclose(model.body_inertia[body], 0.5 * original_inertia)
    rollout.configure_object()
    assert model.body_mass[body] == original_mass
    np.testing.assert_allclose(model.body_inertia[body], original_inertia)
    np.testing.assert_allclose(model.geom_friction[geom], original_geom)
    np.testing.assert_allclose(model.pair_friction, original_pairs, rtol=1e-11)


@pytest.mark.parametrize("kwargs", (
    {"mass": 0}, {"mass": -1}, {"mass": float("nan")}, {"mass": float("inf")},
    {"mass": 0.5, "friction": -1}, {"friction": float("nan")}, {"friction": float("inf")},
))
def test_invalid_runtime_parameters_leave_model_unchanged(tmp_path, kwargs):
    model = compile_scene(tmp_path)
    rollout = GraspRollout(model, mujoco.MjData(model))
    before = [value.copy() for value in (model.body_mass, model.body_inertia, model.geom_friction, model.pair_friction)]
    with pytest.raises(ValueError):
        rollout.configure_object(**kwargs)
    for expected, actual in zip(before, (model.body_mass, model.body_inertia, model.geom_friction, model.pair_friction)):
        np.testing.assert_array_equal(actual, expected)


def test_v6_actual_joint_actuator_torques_saturate_at_published_limits(tmp_path):
    model = compile_scene(tmp_path)
    data = mujoco.MjData(model)
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)])
    dofs = model.jnt_dofadr[joints]
    actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)])
    for sign in (-1, 1):
        data.ctrl[actuators] = sign * 100.0
        mujoco.mj_forward(model, data)
        assert np.all(np.abs(data.qfrc_actuator[dofs]) <= np.asarray(ARM_TORQUE_LIMITS_NM) + 1e-9)
        np.testing.assert_allclose(np.abs(data.qfrc_actuator[dofs]), ARM_TORQUE_LIMITS_NM)
    assert np.all(model.jnt_actfrclimited[joints])
    np.testing.assert_allclose(arm_joint_velocity_limits(model), ARM_VELOCITY_LIMITS_RAD_S)
    np.testing.assert_allclose(arm_joint_acceleration_limits(model), (0.8,) * 7)


def test_legacy_profile_preserves_historical_scene_physics(tmp_path):
    model = compile_scene(tmp_path, profile="legacy-v5")
    assert model.npair == 0
    assert model_physics_version(model) == 5
    assert arm_joint_velocity_limits(model) is None
    assert arm_joint_acceleration_limits(model) is None
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    assert not np.any(model.jnt_actfrclimited[joints])
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    assert model.body_mass[body] == 0.15
    np.testing.assert_allclose(model.geom_friction[geom], (0.8, 0.01, 0.001))


def test_explicit_contact_pairs_preserve_soft_fingertips_and_make_friction_effective(tmp_path):
    model = compile_scene(tmp_path)
    parameters = get_physics_profile("physical-v6")
    for name, solref in (
        ("target_table", (0.02, 1.0)),
        ("target_left_index_fingertip_collision", (0.04, 0.95)),
        ("target_left_index_PP_fit", (0.02, 1.0)),
        ("target_left_thumb_PP_fit", (0.02, 1.0)),
        ("target_palm000_collision", (0.015, 1.0)),
    ):
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        assert pair >= 0
        assert model.pair_dim[pair] == 3
        np.testing.assert_allclose(model.pair_solref[pair], solref)
    finger_pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "target_left_index_fingertip_collision")
    np.testing.assert_allclose(model.pair_friction[finger_pair], pair_friction(
        parameters.object_parameters("ycb_can").friction, parameters.hand_friction,
    ), rtol=1e-11)
    for pair in range(model.npair):
        solref, solimp = native_contact_parameters(model, int(model.pair_geom1[pair]), int(model.pair_geom2[pair]))
        np.testing.assert_allclose(model.pair_solref[pair], solref)
        np.testing.assert_allclose(model.pair_solimp[pair], solimp)
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "physics_contact_revision")
    assert model.numeric_data[model.numeric_adr[marker]] == 2


@pytest.mark.parametrize("first_solref,second_solref,first_priority,second_priority,weights", (
    ((0.01, 1.), (0.04, 0.8), 0, 0, (1., 3.)),
    ((0.01, 1.), (0.04, 0.8), 2, 0, (1., 3.)),
    ((0.01, 1.), (0.04, 0.8), 0, 2, (1., 3.)),
    ((-1000., -20.), (-2000., -10.), 0, 0, (1., 3.)),
    ((-1000., -20.), (0.04, 0.8), 0, 0, (1., 3.)),
    ((0.01, 1.), (0.04, 0.8), 0, 0, (0., 0.)),
    ((0.01, 1.), (0.04, 0.8), 0, 0, (1e-30, 3e-30)),
    ((0.01, 1.), (0.04, 0.8), 0, 0, (1e-30, 1.)),
    ((0.01, 1.), (0.04, 0.8), 0, 0, (1., 1e-30)),
))
def test_contact_parameter_mixing_matches_actual_native_engine(first_solref, second_solref,
                                                              first_priority, second_priority, weights):
    def geom(name, solref, priority, weight, extra=""):
        return (f'<geom name="{name}" type="sphere" size=".1" solref="{solref[0]} {solref[1]}" '
                f'priority="{priority}" solmix="{weight}" {extra}/>')
    xml = ('<mujoco><worldbody>' + geom("first", first_solref, first_priority, weights[0])
           + '<body pos=".19 0 0"><freejoint/>'
           + geom("second", second_solref, second_priority, weights[1]) + '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    expected_ref, expected_imp = native_contact_parameters(model, 0, 1)
    np.testing.assert_allclose(data.contact[0].solref, expected_ref)
    np.testing.assert_allclose(data.contact[0].solimp, expected_imp)


@pytest.mark.parametrize("timestep", (0, -0.001, 0.1, float("nan"), float("inf")))
def test_invalid_timestep_is_rejected_before_scene_publication(tmp_path, timestep):
    with pytest.raises(ValueError, match="timestep"):
        build_scene(tmp_path / "bad.xml", physics_profile="physical-v6", timestep=timestep)
    assert not (tmp_path / "bad.xml").exists()


def test_timestep_override_and_unknown_profile(tmp_path):
    model = compile_scene(tmp_path, timestep=0.001)
    assert model.opt.timestep == 0.001
    np.testing.assert_allclose(model.opt.gravity, (0, 0, -9.81))
    with pytest.raises(ValueError, match="physics profile"):
        build_scene(tmp_path / "bad.xml", physics_profile="unknown")


@pytest.mark.parametrize("profile,impratio,tolerance", (("legacy-v5", 10., 1e-8), ("physical-v6", 100., 1e-10)))
def test_solver_choices_are_explicit_and_static_model_stays_finite(tmp_path, profile, impratio, tolerance):
    model = compile_scene(tmp_path, kind="box", profile=profile)
    parameters = get_physics_profile(profile)
    assert parameters.solver_impratio == model.opt.impratio == impratio
    assert parameters.tolerance == model.opt.tolerance == tolerance
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert not np.any(data.warning.number)
