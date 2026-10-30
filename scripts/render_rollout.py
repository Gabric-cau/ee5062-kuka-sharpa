#!/usr/bin/env python3
"""Render a CPU rollout with synchronized robot, hand and physical signals."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from OpenGL import GL

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.visualization import (
    add_trial_arguments, configure_camera, configure_visuals, create_run_directory,
    load_trial_record, presentation_manifest, resolve_trial, stage_at, write_json,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_trial_arguments(parser)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--contact-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-vectors", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.fps <= 0 or args.hold_seconds < 0:
        parser.error("--fps must be positive and --hold-seconds nonnegative")
    record = load_trial_record(args)
    if args.output is None:
        run_dir = create_run_directory("media_runs")
        output = run_dir / "rollout.mp4"
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        run_dir = output.with_name(f"{output.stem}_evidence")
        if output.exists() or run_dir.exists():
            parser.error(f"refusing to overwrite existing video or evidence: {output}")
        run_dir.mkdir(exist_ok=False)
    keyframe_dir = (args.keyframes or output.with_name(f"{output.stem}_frames")).resolve()
    if keyframe_dir.exists():
        parser.error(f"refusing to overwrite existing keyframes: {keyframe_dir}")
    keyframe_dir.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(build_scene(run_dir / "scene.xml", object_kind=args.object)))
    data = mujoco.MjData(model)
    rollout = GraspRollout(model, data)
    trial = resolve_trial(args, rollout, record)
    rollout.configure_object(trial.object_mass, trial.object_friction)
    rollout.reset(trial.seed, trial.object_pos)
    configure_visuals(model)
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 720
    if args.fps > 1.0 / model.opt.timestep:
        parser.error("--fps cannot exceed the physics sampling frequency")
    overview_renderer = mujoco.Renderer(model, height=720, width=640)
    detail_renderer = mujoco.Renderer(model, height=480, width=640)
    overview_camera, detail_camera = mujoco.MjvCamera(), mujoco.MjvCamera()
    configure_camera(overview_camera, model, data)
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(args.contact_points)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(args.force_vectors)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
    samples: list[dict] = []
    images: list[tuple[np.ndarray, np.ndarray]] = []
    next_frame = 0
    gl_report = {}
    target_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    target_adr = model.jnt_qposadr[target_joint]

    def capture(step: int, frame_data: mujoco.MjData, readings: dict, force: bool = False) -> None:
        nonlocal next_frame, gl_report
        if not force and frame_data.time + 1e-9 < next_frame / args.fps:
            return
        if samples and abs(samples[-1]["simulation_time_s"] - float(frame_data.time)) < 1e-9:
            return
        total = rollout.reader.total(readings)
        overview_renderer.update_scene(frame_data, overview_camera, scene_option=scene_option)
        overview = overview_renderer.render().copy()
        configure_camera(detail_camera, model, frame_data, detail=True)
        detail_renderer.update_scene(frame_data, detail_camera, scene_option=scene_option)
        detail = detail_renderer.render().copy()
        if not gl_report:
            gl_report = {
                label: (GL.glGetString(enum) or b"unavailable").decode("utf-8", errors="replace")
                for label, enum in (("GL_VENDOR", GL.GL_VENDOR), ("GL_RENDERER", GL.GL_RENDERER), ("GL_VERSION", GL.GL_VERSION))
            }
            gl_report["context_scope"] = "This process's active offscreen MuJoCo renderer context; not a native-viewer-thread measurement."
        samples.append({
            "frame_index": len(samples), "simulation_step": step,
            "simulation_time_s": float(frame_data.time), "total_normal_force": total.normal,
            "total_tangential_force": total.tangential, "contacts": total.contacts,
            "object_height_m": float(frame_data.qpos[target_adr + 2]),
        })
        images.append((overview, detail))
        next_frame += 1

    try:
        capture(0, data, rollout.reader.read(data), force=True)
        result, trace_rows = rollout.run_trace(**trial.rollout_arguments(), frame_callback=capture)
        capture(result.steps, data, rollout.reader.read(data), force=True)
    finally:
        detail_renderer.close()
        overview_renderer.close()
    _write_csv(run_dir / "trace.csv", trace_rows)
    _write_csv(run_dir / "frame_samples.csv", samples)
    manifest = presentation_manifest(trial, "mujoco_offscreen_opengl", run_dir.name)
    manifest.update({
        "opengl": gl_report, "result": asdict(result), "video": str(output), "keyframes": str(keyframe_dir),
        "fps": args.fps, "frozen_final_seconds": args.hold_seconds,
        "physical_sample_count": len(samples), "physics_timestep_s": float(model.opt.timestep),
        "physical_simulation_end_s": float(data.time), "contact_points": args.contact_points,
        "force_vectors": args.force_vectors,
        "sampling": "Images and contact readings are captured together after mj_step, plus the reset state at t=0. frame_samples.csv holds actual physics timestamps.",
    })
    write_json(run_dir / "metadata.json", manifest)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig = plt.figure(figsize=(16, 7.6), dpi=100)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.0, 1.05), hspace=0.42, wspace=0.28)
    robot_axis = fig.add_subplot(grid[:, 0])
    hand_axis = fig.add_subplot(grid[0, 1])
    contact_axis = fig.add_subplot(grid[1, 1])
    force_axis = fig.add_subplot(grid[0, 2])
    height_axis = fig.add_subplot(grid[1, 2])
    fig.subplots_adjust(left=0.025, right=0.975, top=0.90, bottom=0.09)
    robot_image = robot_axis.imshow(images[0][0])
    hand_image = hand_axis.imshow(images[0][1])
    robot_axis.set_axis_off()
    hand_axis.set_axis_off()
    robot_axis.set_title("KUKA iiwa 14 + Sharpa Wave")
    hand_axis.set_title("Hand and object")
    time_values = np.asarray([row["simulation_time_s"] for row in samples])
    normals = np.asarray([row["total_normal_force"] for row in samples])
    tangentials = np.asarray([row["total_tangential_force"] for row in samples])
    contacts = np.asarray([row["contacts"] for row in samples])
    heights = np.asarray([row["object_height_m"] for row in samples])
    normal_line, = force_axis.plot([], [], color="#1672b8", label="normal")
    tangential_line, = force_axis.plot([], [], color="#d85c2b", label="tangential")
    contact_line, = contact_axis.step([], [], where="post", color="#238548")
    height_line, = height_axis.plot([], [], color="#a23b79")
    for axis in (force_axis, contact_axis, height_axis):
        axis.set_xlim(0.0, trial.duration_s)
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=0.22)
        axis.axvline(1.30, color="#7f7f7f", linestyle=":", linewidth=1)
    force_axis.set_title("Simulated fingertip contact force")
    force_axis.set_ylabel("Total force (N)")
    force_axis.set_ylim(0.0, max(5.0, float(max(normals.max(), tangentials.max())) * 1.15))
    force_axis.legend(loc="upper right", fontsize=9)
    contact_axis.set_title("Object-fingertip contact count")
    contact_axis.set_ylabel("Contacts")
    contact_axis.set_ylim(0.0, max(2, int(contacts.max()) + 1))
    contact_axis.set_yticks(range(0, max(2, int(contacts.max()) + 1) + 1))
    height_axis.set_title("Object free-joint height")
    height_axis.set_ylabel("Height (m)")
    height_axis.set_ylim(min(0.30, float(heights.min()) - 0.05), max(0.90, float(heights.max()) + 0.05))
    height_axis.axhline(0.55, color="#7f7f7f", linestyle="--", linewidth=1, label="lift threshold 0.55 m")
    height_axis.legend(loc="upper left", fontsize=8)
    title = fig.suptitle("", fontsize=14)
    targets = [(0.0, "open"), (0.50, "contact"), (1.20, "pre_lift"), (1.80, "lift"), (float(data.time), "final")]
    saved = set()
    writer = imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for index, (robot, hand) in enumerate(images):
            end = index + 1
            robot_image.set_data(robot)
            hand_image.set_data(hand)
            normal_line.set_data(time_values[:end], normals[:end])
            tangential_line.set_data(time_values[:end], tangentials[:end])
            contact_line.set_data(time_values[:end], contacts[:end])
            height_line.set_data(time_values[:end], heights[:end])
            outcome = f"{result.status} / {result.failure_reason}" if index == len(images) - 1 else stage_at(float(time_values[index]))
            title.set_text(f"{trial.object_kind} | {trial.controller} | seed {trial.seed} | t={time_values[index]:.2f} s | {outcome}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(frame)
            for target, label in targets:
                if label not in saved and time_values[index] + 1e-9 >= target:
                    fig.savefig(keyframe_dir / f"{label}.png", dpi=120)
                    saved.add(label)
        for _ in range(round(args.hold_seconds * args.fps)):
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    print(result)
    print(f"wrote {output}")
    print(f"wrote keyframes {keyframe_dir}; physical trace and GL metadata {run_dir}")


if __name__ == "__main__":
    main()
