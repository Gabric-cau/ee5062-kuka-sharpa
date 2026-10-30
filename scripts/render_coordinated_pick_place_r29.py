#!/usr/bin/env python3
"""Render completed R29 native states without integrating or solving dynamics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite(values, shape):
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite recorded state")
    return values


def load_completed(directory):
    paths = {name: directory / name for name in (
        "compiled_model.mjb", "native.json.gz", "result.json", "summary.json", "protocol.json")}
    before = {name: digest(path) for name, path in paths.items()}
    summary, result, protocol = (read_json(paths[name]) for name in (
        "summary.json", "result.json", "protocol.json"))
    if summary.get("complete") is not True or not protocol.get("finished_at_utc"):
        raise ValueError("Render only a completed experiment; no waiting or simulation is performed")
    if len(summary["rows"]) != 1:
        raise ValueError("Expected one R29 case")
    case = summary["rows"][0]
    if case["success"] is not result["success"] or type(result["success"]) is not bool:
        raise ValueError("Summary/result task labels disagree")
    native_artifact = case["artifacts"]["native"]
    if (native_artifact["path"] != "native.json.gz"
            or native_artifact["sha256"] != before["native.json.gz"]
            or native_artifact["bytes"] != paths["native.json.gz"].stat().st_size):
        raise ValueError("Native artifact does not match its recorded hash/size")
    if protocol.get("compiled_model_sha256") != before["compiled_model.mjb"]:
        raise ValueError("Compiled model does not match its recorded hash")
    if summary["protocol"] != protocol:
        raise ValueError("Summary/protocol records disagree")
    with gzip.open(paths["native.json.gz"], "rt") as stream:
        native = json.load(stream)
    if not native or len(native) != result["native_audit"]["native_samples"]:
        raise ValueError("Native count is empty or differs from the run audit")
    times = finite([row["timestamp"] for row in native], (len(native),))
    if np.any(np.diff(times) <= 0.) or np.any(np.abs(np.diff(times) - .002) > 1e-9):
        raise ValueError("Native timestamps must retain every 2ms step in order")
    if {name: digest(path) for name, path in paths.items()} != before:
        raise ValueError("Input changed while loading completed experiment")
    return paths, before, result, native, times


def state_bindings(model):
    hinges, covered = {}, set()
    target = model.joint("target_free").id
    if model.jnt_type[target] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected target_free to be a free joint")
    target_address = int(model.jnt_qposadr[target])
    covered.update(range(target_address, target_address + 7))
    for joint in range(model.njnt):
        if joint == target:
            continue
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not model.joint(joint).name:
            raise ValueError("Unrecorded moving coordinate; refusing to invent scene state")
        hinges[model.joint(joint).name] = int(model.jnt_qposadr[joint])
        covered.add(int(model.jnt_qposadr[joint]))
    if covered != set(range(model.nq)):
        raise ValueError("Recorded joint schema does not cover the compiled qpos")
    return hinges, target_address


def restore_kinematics(model, data, sample, hinges, target_address):
    robot = sample["robot_joint_position_rad"]
    if set(robot) != set(hinges):
        raise ValueError("Recorded robot joint names differ from compiled model")
    values = finite([robot[name] for name in hinges], (len(hinges),))
    target = finite(sample["target_qpos"], (7,))
    if abs(np.linalg.norm(target[3:]) - 1.) > 1e-7:
        raise ValueError("Recorded target quaternion is not normalized")
    data.qpos[list(hinges.values())] = values
    data.qpos[target_address:target_address + 7] = target
    data.time = float(sample["timestamp"])
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def force_channels(model, native):
    hand = {}
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if body.startswith("left_"):
            hand[geom] = next((f for f in CHANNELS[:-1] if body.startswith("left_" + f + "_")), "palm")
    forces = np.zeros((len(native), len(CHANNELS)))
    for row_index, row in enumerate(native):
        for contact in row["native_contacts"]:
            geom_ids = tuple(contact["geom_ids"])
            if (len(geom_ids) != 2 or any(type(g) is not int or not 0 <= g < model.ngeom for g in geom_ids)):
                raise ValueError("Invalid recorded native geometry IDs")
            sides = [g for g in geom_ids if g in hand]
            if len(sides) == 1:
                wrench = finite(contact["force_torque_contact_frame"], (6,))
                forces[row_index, CHANNELS.index(hand[sides[0]])] += abs(wrench[0])
    return forces


def frame_indices(times, fps, start_s):
    first = int(np.searchsorted(times, start_s - 1e-9))
    if first >= len(times):
        raise ValueError("Requested start is after the recorded trajectory")
    grid = np.arange(times[first], times[-1] + 1e-10, 1. / fps)
    selected = np.searchsorted(times, grid - 1e-9)
    return np.unique(np.r_[first, np.minimum(selected, len(times) - 1), len(times) - 1]).astype(int)


def font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(overview, detail, sample, rise_mm, forces, cache_peak, outcome):
    canvas = Image.new("RGB", (1440, 900), "#13191f")
    canvas.paste(Image.fromarray(overview), (0, 64))
    canvas.paste(Image.fromarray(detail), (960, 64))
    draw = ImageDraw.Draw(canvas)
    normal, small = font(23), font(19)
    color = "#9ddca4" if outcome["success"] else "#ffb09b"
    draw.text((18, 17), f"R29 | {sample['state']} | simulation t={sample['timestamp']:.3f}s", font=normal, fill="white")
    draw.text((980, 17), "Actual hand / object state", font=small, fill="white")
    draw.text((18, 758), f"Object rise: {rise_mm:+.2f} mm   |   native window force peak: {max(forces):.3f} N   |   limit: 12 N", font=normal, fill="white")
    draw.text((18, 794), "   ".join(f"{name}: {value:.2f}N" for name, value in zip(CHANNELS, forces)), font=small, fill="white")
    draw.text((18, 826), f"Recorded final task_success: {str(outcome['success']).lower()}   |   whole-run cached peak: {cache_peak:.3f} N", font=small, fill=color)
    failure = str(outcome.get("failure") or "none")
    if len(failure) > 145:
        failure = failure[:142] + "..."
    draw.text((18, 858), f"Recorded failure: {failure}", font=small, fill=color)
    return np.asarray(canvas)


def render(directory, output, *, fps=25, start_s=0.):
    if not 20 <= fps <= 50 or not np.isfinite(start_s) or start_s < 0.:
        raise ValueError("Require 20-50fps (20-50ms sampling) and finite nonnegative start")
    paths, hashes, result, native, times = load_completed(directory)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing derived artifact directory")
    model = mujoco.MjModel.from_binary_path(str(paths["compiled_model.mjb"]))
    model.vis.headlight.ambient[:] = .35
    model.vis.headlight.diffuse[:] = .65
    model.vis.headlight.specular[:] = .2
    data = mujoco.MjData(model)
    hinges, target_address = state_bindings(model)
    forces = force_channels(model, native)
    initial_z = float(finite(result["native_audit"]["initial_target_qpos"], (7,))[2])
    cache_forces = result["native_cache_channel_peaks_n"]
    if set(cache_forces) != set(CHANNELS):
        raise ValueError("Incomplete recorded whole-run cache channel peaks")
    cache_peaks = finite([cache_forces[n] for n in CHANNELS], (6,))
    if np.any(cache_peaks < 0.):
        raise ValueError("Negative channel peak")
    indices = frame_indices(times, fps, start_s)
    model.vis.global_.offwidth = max(960, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(680, model.vis.global_.offheight)
    overview_camera, detail_camera = mujoco.MjvCamera(), mujoco.MjvCamera()
    overview_camera.lookat[:] = [.48, -.04, .61]
    overview_camera.distance, overview_camera.azimuth, overview_camera.elevation = 1.75, 135., -25.
    detail_camera.distance, detail_camera.azimuth, detail_camera.elevation = .5, 135., -15.
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    output.mkdir(parents=True)
    video_path = output / "trajectory.mp4"
    keyframes = {0: "first.png", len(indices) // 2: "mid.png", len(indices) - 1: "final.png"}
    samples, gl = [], {}
    previous = int(indices[0]) - 1
    wrist = model.body("link7").id
    with mujoco.Renderer(model, height=680, width=960) as overview_renderer, \
            mujoco.Renderer(model, height=680, width=480) as detail_renderer, \
            imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame_index, native_index in enumerate(indices):
            row = native[int(native_index)]
            restore_kinematics(model, data, row, hinges, target_address)
            overview_renderer.update_scene(data, camera=overview_camera, scene_option=options)
            overview = overview_renderer.render().copy()
            detail_camera.lookat[:] = .5 * (data.qpos[target_address:target_address + 3] + data.xpos[wrist])
            detail_renderer.update_scene(data, camera=detail_camera, scene_option=options)
            detail = detail_renderer.render().copy()
            if not gl:
                from OpenGL import GL
                gl = {label: (GL.glGetString(enum) or b"unavailable").decode("utf-8", errors="replace")
                    for label, enum in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
            window = forces[previous + 1:int(native_index) + 1].max(axis=0)
            rise_mm = 1000. * (float(row["target_qpos"][2]) - initial_z)
            frame = compose(overview, detail, row, rise_mm, window, float(cache_peaks.max()), result)
            if min(float(overview.std()), float(detail.std())) < 1.:
                raise RuntimeError("Blank scene image; rendered artifact is incomplete")
            writer.append_data(frame)
            if frame_index in keyframes:
                Image.fromarray(frame).save(output / keyframes[frame_index])
            samples.append({"frame_index": frame_index, "native_index": int(native_index),
                "timestamp_s": float(row["timestamp"]), "phase": row["state"],
                "actual_object_rise_mm": rise_mm, "native_window_channel_peaks_n": dict(zip(CHANNELS, window.tolist())),
                "pixel_std": float(frame.std()), "overview_pixel_std": float(overview.std()),
                "detail_pixel_std": float(detail.std())})
            previous = int(native_index)
    after = {name: digest(path) for name, path in paths.items()}
    if after != hashes:
        raise ValueError("Frozen raw input changed during rendering")
    outputs = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()}
    metadata = {"format": "r29_native_state_video_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(directory), "inputs_sha256": hashes, "outputs": outputs,
        "runtime": {"platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"), "opengl": gl},
        "native_sample_count": len(native), "rendered_frame_count": len(samples), "fps": fps,
        "start_s": start_s, "native_first_timestamp_s": float(times[0]), "native_final_timestamp_s": float(times[-1]),
        "rendered_final_native_index": int(indices[-1]), "recorded_task_success": result["success"],
        "recorded_failure": result.get("failure"), "native_refreshed_channel_peaks_n": dict(zip(CHANNELS, forces.max(axis=0).tolist())),
        "recorded_whole_run_cached_channel_peaks_n": cache_forces,
        "kinematics_only": True, "dynamics_integration_steps": 0,
        "sampling": "Actual native qpos at first recorded sample at/after each uniform video timestamp; no state interpolation. Final native sample retained.",
        "force_scope": "Six external hand-channel sums from recorded private-forward native contacts; window maxima retain unsampled transients. Cached peaks are separate whole-run values, not synchronized with each frame.",
        "label_scope": "Final task/failure labels copied from completed result, not independently rescored; failures retained. Geometry/lights reconstructed in scratch data using mj_kinematics/mj_comPos/mj_camlight only; no mj_step/mj_forward/contact-force recomputation.",
        "visual_overrides": {"headlight_ambient": .35, "headlight_diffuse": .65,
            "headlight_specular": .2, "physics_model_file_unchanged": True},
        "camera": {"overview_lookat": overview_camera.lookat.tolist(), "overview_distance": float(overview_camera.distance),
            "overview_azimuth": float(overview_camera.azimuth), "overview_elevation": float(overview_camera.elevation),
            "detail_lookat": "Recorded object freejoint origin / wrist midpoint", "detail_distance": float(detail_camera.distance)},
        "frames": samples}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "native_samples": len(native), "frames": len(samples),
        "final_time_s": samples[-1]["timestamp_s"], "recorded_task_success": result["success"],
        "metadata_sha256": digest(output / "metadata.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed R29 output directory")
    parser.add_argument("--output", type=Path, required=True, help="New derived artifact directory")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--start-s", type=float, default=0.)
    args = parser.parse_args()
    render(args.input.resolve(), args.output.resolve(), fps=args.fps, start_s=args.start_s)


if __name__ == "__main__":
    main()
