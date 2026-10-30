#!/usr/bin/env python3
"""Export actual compiled parameters, separately from their declared provenance."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from public_ycb_evidence import load_evidence_model
from kuka_sharpa_grasp.physics import get_physics_profile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(scene, output):
    model = load_evidence_model(scene)
    target = model.geom("target_geom").id
    body = int(model.geom_bodyid[target])
    objects = []
    for gid in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[gid])).name or ""
        if gid not in (target, model.geom("table").id) and not (
                body_name.startswith("left_") and model.geom_contype[gid]):
            continue
        objects.append({"geom": model.geom(gid).name, "body": body_name,
            "type": mujoco.mjtGeom(int(model.geom_type[gid])).name,
            "size": model.geom_size[gid].tolist(), "friction": model.geom_friction[gid].tolist(),
            "solref": model.geom_solref[gid].tolist(), "solimp": model.geom_solimp[gid].tolist(),
            "condim": int(model.geom_condim[gid]), "priority": int(model.geom_priority[gid]),
            "solmix": float(model.geom_solmix[gid])})
    pairs = [{"geoms": [model.geom(int(model.pair_geom1[p])).name,
                         model.geom(int(model.pair_geom2[p])).name],
        "dim": int(model.pair_dim[p]), "friction": model.pair_friction[p].tolist(),
        "solref": model.pair_solref[p].tolist(), "solimp": model.pair_solimp[p].tolist()}
        for p in range(model.npair) if target in (model.pair_geom1[p], model.pair_geom2[p])]
    joints = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        dof = int(model.jnt_dofadr[j])
        joints.append({"name": model.joint(j).name, "limited": bool(model.jnt_limited[j]),
            "range_rad": model.jnt_range[j].tolist(), "user_limits": model.jnt_user[j].tolist(),
            "damping": float(model.dof_damping[dof]), "armature": float(model.dof_armature[dof])})
    actuators = [{"name": model.actuator(a).name,
        "joint": model.joint(int(model.actuator_trnid[a, 0])).name,
        "gainprm": model.actuator_gainprm[a].tolist(), "biasprm": model.actuator_biasprm[a].tolist(),
        "ctrl_limited": bool(model.actuator_ctrllimited[a]), "ctrl_range": model.actuator_ctrlrange[a].tolist(),
        "force_limited": bool(model.actuator_forcelimited[a]), "force_range": model.actuator_forcerange[a].tolist()}
        for a in range(model.nu)]
    profile = get_physics_profile("physical-v6")
    document = {"format": "compiled_public_ycb_physics_v1", "mujoco": mujoco.__version__,
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "exporter_sha256": sha256(Path(__file__).resolve()),
        "physics_source_sha256": sha256(Path(__file__).resolve().parents[1] / "src/kuka_sharpa_grasp/physics.py"),
        "scope": "Model compilation only; not a new task run, hardware calibration, or validation of material assumptions. Source-declared literature provenance is not independently checked by this exporter.",
        "native_options": {"timestep_s": float(model.opt.timestep), "gravity_m_s2": model.opt.gravity.tolist(),
            "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "solver": mujoco.mjtSolver(int(model.opt.solver)).name,
            "cone": mujoco.mjtCone(int(model.opt.cone)).name,
            "iterations": int(model.opt.iterations), "tolerance": float(model.opt.tolerance),
            "impratio": float(model.opt.impratio)},
        "actual_target": {"body": model.body(body).name, "mass_kg": float(model.body_mass[body]),
            "principal_inertia_kg_m2": model.body_inertia[body].tolist(),
            "inertial_position_in_body_m": model.body_ipos[body].tolist(),
            "inertial_quaternion_in_body": model.body_iquat[body].tolist()},
        "collision_geoms": objects, "explicit_target_pairs": pairs,
        "hinge_joints": joints, "actuators": actuators, "declared_profile": asdict(profile),
        "interpretation": {"friction": "Explicit target pairs use an assumed geometric mean, not an identified material law.",
            "inertia": "Uniform-density geometry approximation, not measured physical inertia.",
            "collision": "MuJoCo convex mesh collision and authored hand collision proxies differ from detailed visual meshes.",
            "joint_limits": "Soft constraints can be temporarily crossed by actual joints; command limits are not hard state clamps.",
            "tactile": "Simulated native external contact forces, not calibrated hardware sensor output.",
            "gpu": "Native MuJoCo physics is CPU based; separate actual GL metadata identifies the renderer."}}
    if abs(model.opt.timestep - profile.timestep) > 1e-12 or not np.allclose(model.opt.gravity, profile.gravity):
        raise ValueError("Scene timestep/gravity differs from the declared physical-v6 profile")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output),
        "mass_kg": document["actual_target"]["mass_kg"],
        "principal_inertia_kg_m2": document["actual_target"]["principal_inertia_kg_m2"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.scene.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
