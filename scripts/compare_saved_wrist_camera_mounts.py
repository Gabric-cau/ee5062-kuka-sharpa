#!/usr/bin/env python3
"""Compare two ideal wrist cameras on one recorded R20 geometry, without stepping."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver, save_observation

SCENE_SHA256 = "4dd442d6a46a5141a2a8726d10bc4d0c085f8f1ebab639fb71974cf8e694ea3e"
CANDIDATE_POSITION = np.array([-.07, -.09, .02])
CANDIDATE_LOOK_AT = np.array([.06, -.04, .18])


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_record(path, timestamp, phase):
    with gzip.open(path, "rt") as stream:
        records = json.load(stream)
    selected = [(index, row) for index, row in enumerate(records)
                if abs(float(row["timestamp"]) - timestamp) < 1e-10 and row["state"] == phase]
    if len(selected) != 1:
        raise ValueError(f"Expected one {phase} record at {timestamp}: {path}, got {len(selected)}")
    index, row = selected[0]
    return row, {"index_zero_based": index, "timestamp_s": row["timestamp"],
                 "state": row["state"], "canonical_record_sha256": canonical_digest(row)}


def restore_geometry(model, native, trace):
    data = mujoco.MjData(model)
    covered_qpos = np.zeros(model.nq, dtype=bool)
    covered_qvel = np.zeros(model.nv, dtype=bool)
    arm_position = np.asarray(trace["arm_joint_position"], dtype=float)
    arm_velocity = np.asarray(trace["arm_joint_velocity_rad_s"], dtype=float)
    if arm_position.shape != (7,) or arm_velocity.shape != (7,):
        raise ValueError("Recorded arm state must have seven joints")
    joints = {f"joint{index + 1}": (arm_position[index], arm_velocity[index]) for index in range(7)}
    for name, row in native["actual_hand_joints"].items():
        if name in joints:
            raise ValueError("Duplicate robot joint")
        joints[name] = (row["position_rad"], row["velocity_rad_s"])
    for name, (position, velocity) in joints.items():
        joint = model.joint(name).id
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected recorded hinge: {name}")
        qaddress, vaddress = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[qaddress], data.qvel[vaddress] = position, velocity
        covered_qpos[qaddress], covered_qvel[vaddress] = True, True
    target = model.joint("target_free").id
    if model.jnt_type[target] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected one recorded free target joint")
    address = model.jnt_qposadr[target]
    target_qpos = np.asarray(native["target_qpos"], dtype=float)
    if target_qpos.shape != (7,):
        raise ValueError("Recorded target qpos must have seven values")
    data.qpos[address:address + 7] = target_qpos
    covered_qpos[address:address + 7] = True
    if not covered_qpos.all() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("Recorded qpos does not cover the complete finite model geometry")
    if covered_qvel.sum() != model.nv - 6:
        raise ValueError("Only the target free-joint velocity may be unrecorded")
    data.time = float(native["timestamp"])
    before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos, before_qpos) or not np.array_equal(data.qvel, before_qvel):
        raise ValueError("Forward evaluation changed restored qpos or qvel")
    return data, {"nq": model.nq, "nv": model.nv, "recorded_qpos_count": int(covered_qpos.sum()),
                  "recorded_robot_qvel_count": int(covered_qvel.sum()),
                  "unrecorded_target_free_qvel": "Six zero values; irrelevant to static RGB-D rendering. Cached COM spatial velocity is not substituted for postintegration free-joint qvel.",
                  "qpos_sha256": hashlib.sha256(data.qpos.tobytes()).hexdigest(),
                  "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                  "scope": "Recorded postintegration qpos geometry, not a complete dynamic-state reconstruction. Forward recomputes camera poses, unlike cached native derived arrays."}


def appearance_stats(observation):
    rgb, depth = observation.rgb.astype(float), observation.depth
    red = (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 1.5 * rgb[..., 2])
    red &= np.isfinite(depth) & (depth > 0.)
    return {"rgb_std": float(rgb.std()), "rgb_channel_std": rgb.std(axis=(0, 1)).tolist(),
            "valid_depth_fraction": float(np.isfinite(depth).mean()),
            "depth_std_m": float(np.nanstd(depth)), "appearance_only_red_pixels": int(red.sum()),
            "appearance_only_red_fraction": float(red.mean()),
            "nonblank_rgb": bool(rgb.std() > 5.),
            "scope": "Red color threshold only; not object segmentation truth, occlusion percentage, pose accuracy or task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--reference-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve(strict=True)
    paths = {"scene": case / "scene.xml", "native": case / "native.json.gz", "trace": case / "trace.json.gz",
             "reference_camera": args.reference_observation.resolve(strict=True) / "wrist_camera.json"}
    inputs = {name: binding(path) for name, path in paths.items()}
    if inputs["scene"]["sha256"] != SCENE_SHA256:
        raise ValueError("The finalized R20 scene differs from the pinned scene")
    reference = json.loads(paths["reference_camera"].read_text())
    timestamp = float(reference["timestamp_s"])
    native, native_binding = select_record(paths["native"], timestamp, "CLOSE")
    trace, trace_binding = select_record(paths["trace"], timestamp, "CLOSE")
    if native["timestamp"] != trace["timestamp"]:
        raise ValueError("Selected native and trace clocks differ")
    tree = ET.parse(paths["scene"]).getroot()
    meshdir = PROJECT / "vendor/mujoco_menagerie"
    original_meshdir = tree.find("compiler").get("meshdir")
    tree.find("compiler").set("meshdir", str(meshdir))
    asset_paths = sorted({(meshdir / element.get("file")).resolve(strict=True)
                          for element in tree.findall("asset/mesh") if element.get("file")})
    asset_inputs = [binding(path) for path in asset_paths]
    original_xml = ET.tostring(tree, encoding="unicode")
    camera_element = tree.find(".//camera[@name='wrist_rgbd']")
    original_camera = dict(camera_element.attrib)
    z_axis = CANDIDATE_POSITION - CANDIDATE_LOOK_AT
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    camera_element.set("pos", " ".join(map(str, CANDIDATE_POSITION)))
    camera_element.set("xyaxes", " ".join(map(str, np.r_[x_axis, y_axis])))
    candidate_camera = dict(camera_element.attrib)
    candidate_xml = ET.tostring(tree, encoding="unicode")
    models = [mujoco.MjModel.from_xml_string(xml) for xml in (original_xml, candidate_xml)]
    for attribute in ("body_mass", "body_inertia", "geom_pos", "geom_quat", "geom_friction",
                      "actuator_gainprm", "actuator_biasprm", "qpos0"):
        if not np.array_equal(getattr(models[0], attribute), getattr(models[1], attribute)):
            raise ValueError(f"Camera-only change altered {attribute}")
    config = CameraConfig(width=int(reference["shape"][1]), height=int(reference["shape"][0]))
    observations, states, render_contexts = [], [], []
    step_calls = 0

    def forbidden_step(*unused_args, **unused_kwargs):
        nonlocal step_calls
        step_calls += 1
        raise RuntimeError("Physics integration is forbidden in this saved-state comparison")

    original_step = mujoco.mj_step
    mujoco.mj_step = forbidden_step
    try:
        for model in models:
            data, state = restore_geometry(model, native, trace)
            cam = model.camera("wrist_rgbd").id
            if model.body(model.cam_bodyid[cam]).name != "link7":
                raise ValueError("Both comparison cameras must remain attached to link7")
            with CameraObserver(model, config) as observer:
                observations.append(observer.poll(data))
                from OpenGL import GL
                context = {name: GL.glGetString(token).decode("ascii", errors="replace")
                           for name, token in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                                               ("version", GL.GL_VERSION))}
                major_version = int(context["version"].split(".", 1)[0])
                if major_version >= 3:
                    extensions = {GL.glGetStringi(GL.GL_EXTENSIONS, index).decode("ascii")
                                  for index in range(int(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)))}
                else:
                    extensions = set(GL.glGetString(GL.GL_EXTENSIONS).decode("ascii").split())
                context["arb_clip_control_available"] = "GL_ARB_clip_control" in extensions
                context["depth_precision_note"] = (
                    "ARB_clip_control unavailable: MuJoCo reports limited depth accuracy on this context."
                    if not context["arb_clip_control_available"] else "ARB_clip_control is available.")
                render_contexts.append(context)
            states.append(state)
    finally:
        mujoco.mj_step = original_step
    if states[0] != states[1] or step_calls:
        raise ValueError("Views did not use identical recorded geometry without stepping")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, observation in zip(("original", "compact_candidate"), observations):
        save_observation(observation, args.output / name)
    finite_depth = np.concatenate([obs.depth[np.isfinite(obs.depth)] for obs in observations])
    low, high = np.percentile(finite_depth, [2., 98.]).tolist()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), layout="constrained")
    figure.suptitle(f"R20 recorded geometry at {native['timestamp']:.3f} s: ideal wrist camera mounts", fontsize=13)
    palette = plt.get_cmap("viridis").copy()
    palette.set_bad("#eeeeee")
    statistics = [appearance_stats(obs) for obs in observations]
    for column, (name, obs, stats) in enumerate(zip(("Original: 321 mm offset", "Compact candidate: 116 mm offset"), observations, statistics)):
        axes[0, column].imshow(obs.rgb)
        axes[0, column].set_title(name, fontsize=11)
        artist = axes[1, column].imshow(np.ma.masked_invalid(obs.depth), cmap=palette, vmin=low, vmax=high)
        axes[1, column].set_title(f"Red appearance pixels: {stats['appearance_only_red_pixels']} (not accuracy)", fontsize=10)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(artist, ax=axes[1, :].tolist(), label="Camera-axis depth (m)", shrink=.7, extend="both")
    outputs = []
    for suffix in ("png", "pdf"):
        path = args.output / ("camera_mount_comparison." + suffix)
        figure.savefig(path, dpi=180)
        outputs.append(binding(path))
    plt.close(figure)
    after = {name: binding(path) for name, path in paths.items()}
    if inputs != after or asset_inputs != [binding(path) for path in asset_paths]:
        raise ValueError("Saved inputs changed during comparison")
    observation_outputs = {name: [binding(path) for path in sorted((args.output / name).iterdir())]
                           for name in ("original", "compact_candidate")}
    receipt = {"format": "saved_r20_wrist_camera_mount_comparison_v1", "source": binding(__file__),
        "camera_observer_source": binding(PROJECT / "src/kuka_sharpa_grasp/camera.py"),
        "inputs": inputs, "assets": asset_inputs, "inputs_unchanged": True,
        "native_record": native_binding, "trace_record": trace_binding,
        "reference_camera_metadata": reference,
        "r21_physical_prefix_identity_claimed": False,
        "reference_scope": "R21 metadata selects only a timestamp and image dimensions. Both rendered views use one finalized R20 record; no R21 native state or physical prefix equality was verified.",
        "meshdir_remap_in_memory_only": {"from": original_meshdir, "to": str(meshdir)},
        "scene_written_or_modified": False, "mj_step_calls": step_calls, "mj_forward_calls": 2,
        "state": states[0], "same_recorded_qpos_and_robot_qvel_for_both_views": True,
        "camera_definitions": {"original": original_camera, "compact_candidate": candidate_camera,
            "candidate_position_link7_m": CANDIDATE_POSITION.tolist(),
            "candidate_fixed_look_at_link7_m": CANDIDATE_LOOK_AT.tolist(),
            "candidate_aim_uses_target_state": False,
            "candidate_orientation_rule": "Camera -Z toward fixed local look-at; +Y nearest link7 +Y."},
        "mujoco_version": mujoco.__version__, "requested_gl_backend": os.environ.get("MUJOCO_GL"),
        "render_contexts": render_contexts,
        "render_statistics": dict(zip(("original", "compact_candidate"), statistics)),
        "recomputed_original_vs_reference_camera_position_difference_m": float(np.linalg.norm(
            observations[0].camera_position - np.asarray(reference["camera_position_m"]))),
        "display_depth_range_m": [low, high],
        "display_depth_range_rule": "Shared pooled 2nd-98th percentile display colors only; raw metric NPY files are unchanged.",
        "outputs": outputs, "observation_outputs": observation_outputs,
        "scope": "Offline static camera comparison only. Ideal simulated sensors: no housing, mounting bracket, added mass or collision geometry. Target recorded qpos restores scene geometry for rendering, never aims either camera or feeds a runtime controller. No trajectory, collision certification, perception accuracy, grasp success or human teleoperation claim."}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "recorded_timestamp_s": native["timestamp"],
                      "mj_step_calls": step_calls, "render_statistics": receipt["render_statistics"]}))


if __name__ == "__main__":
    main()
