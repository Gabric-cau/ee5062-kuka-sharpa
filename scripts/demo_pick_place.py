#!/usr/bin/env python3
"""Run and record the physical RGB-D pick-and-place task.

Run this entry point on the Windows graphical desktop. --view displays the
actual MuJoCo views; --native-viewer adds its 3D window. Frames use step callbacks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / "local_cache" / "demo_matplotlib"))
sys.path.insert(0, str(PROJECT / "src"))

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np

from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver
from kuka_sharpa_grasp.control import FINGERS
from kuka_sharpa_grasp.hand_grasp import HandContactReader
from kuka_sharpa_grasp.perception import estimate_object
from kuka_sharpa_grasp.pick_place import Scenario, run_task
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene, WORKCELL_VERSION


COLORS = {
    "thumb": "#d94a4a", "index": "#188ac0", "middle": "#8356ad",
    "ring": "#289465", "pinky": "#be922a", "palm": "#526170",
}
CONTACT_CHANNELS = (*FINGERS, "palm")
DISPLAY_GAMMA = (255. * (np.arange(256, dtype=float) / 255.) ** .72).astype(np.uint8)
STAGE_COLORS = {
    "SEARCH": "#6c8f99", "APPROACH": "#2685b5", "ALIGN": "#46a1c1",
    "CLOSE": "#936baa", "TRIAL_LIFT": "#5f9b68", "TRANSFER": "#239081",
    "PLACE": "#879948", "RELEASE": "#caa454", "VERIFY": "#50715d",
    "RECOVER": "#d68a3f", "ABORT": "#c95050",
}


def stage_name(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def source_snapshot() -> dict[str, str]:
    files = list((PROJECT / "src").rglob("*.py"))
    files.append(Path(__file__).resolve())
    return {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if not p.name.startswith("._")}


def configure_overview(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = [.52, -.02, .48]
    camera.distance = 1.85
    camera.azimuth = 135
    camera.elevation = -25


class LiveImageWindow:
    """Display the recording canvas directly without another plot renderer."""
    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Wrist RGB-D and contact feedback")
        self.root.geometry("1280x720")
        self.root.configure(background="#f4f5f6")
        self.label = tk.Label(self.root, background="#f4f5f6", borderwidth=0)
        self.label.pack(fill="both", expand=True)
        self.running = True
        self.photo = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update()

    def bind_keys(self, callback):
        def pressed(event):
            key = event.char.upper()
            if key in (" ", "P", "R"):
                callback(ord(key))
        self.root.bind("<KeyPress>", pressed)

    def pump(self):
        if self.running:
            try:
                self.root.update_idletasks()
                self.root.update()
            except self.tk.TclError:
                self.running = False

    def show(self, pixels):
        if not self.running:
            return
        from PIL import Image, ImageTk
        width, height = max(1, self.label.winfo_width()), max(1, self.label.winfo_height())
        scale = min(width / pixels.shape[1], height / pixels.shape[0])
        size = (max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale)))
        bitmap = Image.fromarray(pixels)
        if bitmap.size != size:
            bitmap = bitmap.resize(size, Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(bitmap, master=self.root)
        self.label.configure(image=self.photo)
        self.pump()

    def close(self):
        if self.running:
            self.running = False
            try:
                self.root.destroy()
            except self.tk.TclError:
                pass


class Dashboard:
    """Stable figure dimensions for live display and synchronized video."""
    def __init__(self, pyplot, fps: int, output: Path, live: bool, mode: str):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.plt = pyplot
        self.live = live
        self.mode = mode
        self.writer = imageio.get_writer(str(output), fps=fps, codec="libx264",
                                         quality=8, macro_block_size=None)
        pyplot.rcParams.update({"font.size": 10, "axes.titlesize": 11})
        self.figure = Figure(figsize=(16, 9), dpi=100, facecolor="#f4f5f6")
        FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(4, 3, width_ratios=(2.35, 1, 1),
                                       height_ratios=(1.6, 1.05, 1.15, .62),
                                       hspace=.48, wspace=.28)
        self.figure.subplots_adjust(left=.03, right=.98, top=.91, bottom=.08)
        self.overview = self.figure.add_subplot(grid[:2, 0])
        self.detail = self.figure.add_subplot(grid[2, 0])
        self.rgb = self.figure.add_subplot(grid[0, 1])
        self.depth = self.figure.add_subplot(grid[0, 2])
        self.forces = self.figure.add_subplot(grid[1:3, 1:])
        self.timeline = self.figure.add_subplot(grid[3, :])
        self.overview_image = self.overview.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.detail_image = self.detail.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
        self.rgb_image = self.rgb.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        cmap = pyplot.get_cmap("viridis").copy()
        cmap.set_bad("#171a1e")
        self.depth_image = self.depth.imshow(np.full((240, 320), np.nan), cmap=cmap,
                                             vmin=.05, vmax=.9)
        for axis in (self.overview, self.detail, self.rgb, self.depth):
            axis.set_axis_off()
        self.overview.set_title("KUKA iiwa 14 + Sharpa Wave | workcell")
        self.detail.set_title("Hand contact view")
        self.rgb.set_title("Wrist RGB")
        self.depth.set_title("Wrist depth (m)")
        self.figure.colorbar(self.depth_image, ax=self.depth, fraction=.05, pad=.04)
        self.rgb_text = self.rgb.text(.02, .03, "", transform=self.rgb.transAxes,
                                       color="white", fontsize=9,
                                       bbox={"facecolor": "#16191d", "alpha": .78, "pad": 3})
        self.lines = {finger: self.forces.plot([], [], color=COLORS[finger],
                                              label=finger)[0] for finger in CONTACT_CHANNELS}
        self.forces.set_title("Simulated hand-object normal force")
        self.forces.set_xlabel("Simulation time (s)")
        self.forces.set_ylabel("Force (N)")
        self.forces.set_xlim(0, 20)
        self.forces.set_ylim(0, 10)
        self.forces.grid(alpha=.20)
        self.forces.legend(loc="upper left", ncol=3, fontsize=8)
        self.timeline.set_xlim(0, 20)
        self.timeline.set_ylim(0, 1)
        self.timeline.set_yticks([])
        self.timeline.set_xlabel("Physical task timeline (s)")
        self.timeline.spines[["left", "right", "top"]].set_visible(False)
        self.title = self.figure.suptitle("", fontsize=14)
        self.metrics = self.figure.text(.035, .945, "", fontsize=10, color="#39414b")
        self.segments: list[dict[str, Any]] = []
        self.times: list[float] = []
        self.finger_values = {finger: [] for finger in CONTACT_CHANNELS}
        self.last_frame = None
        self.video_frames = 0
        self.live_window = LiveImageWindow() if live else None
        self.background = None
        self.horizon = 20.
        self.force_limit = 10.
        self.stage_artists = []
        self.dynamic_artists = [self.overview_image, self.detail_image, self.rgb_image,
                                self.depth_image, *self.lines.values(), self.rgb_text,
                                self.metrics, self.title]
        for artist in self.dynamic_artists:
            artist.set_animated(True)

    def track_stage(self, state, stamp):
        if not self.segments or state != self.segments[-1]["stage"]:
            if self.segments:
                self.segments[-1]["end"] = stamp
            self.segments.append({"stage": state, "start": stamp, "end": stamp})
        self.segments[-1]["end"] = stamp

    def draw(self, row, overview, detail, observation, belief, seed, result=None, *, append_video=True):
        stamp = row["simulation_time_s"]
        state = row["stage"]
        if not self.times or stamp > self.times[-1] + 1e-10:
            self.times.append(stamp)
            for finger in CONTACT_CHANNELS:
                self.finger_values[finger].append(row[f"{finger}_normal_n"])
        self.overview_image.set_data(overview)
        self.detail_image.set_data(detail)
        self.rgb_image.set_data(observation.rgb)
        self.depth_image.set_data(np.ma.masked_invalid(observation.depth))
        visible = int(len(belief.points))
        self.rgb_text.set_text(f"Visible-surface display estimate\n{visible} points | confidence {belief.confidence:.2f}")
        for finger, line in self.lines.items():
            line.set_data(self.times, self.finger_values[finger])
        peak = max(max(values, default=0) for values in self.finger_values.values())
        maximum = max(10., float(np.ceil(peak * 1.15 / 5.) * 5.))
        horizon = max(20., float(np.ceil((stamp + 1.) / 5.) * 5.))
        if maximum > self.force_limit:
            self.force_limit = maximum
            self.forces.set_ylim(0, maximum)
            self.background = None
        if horizon > self.horizon:
            self.horizon = horizon
            self.forces.set_xlim(0, horizon)
            self.timeline.set_xlim(0, horizon)
            self.background = None
        from matplotlib.patches import Rectangle
        for index, segment in enumerate(self.segments):
            width = max(segment["end"] - segment["start"], .01)
            if index >= len(self.stage_artists):
                patch = Rectangle((segment["start"], .10), width, .50,
                                  facecolor=STAGE_COLORS.get(segment["stage"], "#87919a"),
                                  animated=True)
                self.timeline.add_patch(patch)
                label = self.timeline.text(segment["start"], .35,
                                           segment["stage"].replace("TRIAL_", ""),
                                           ha="center", va="center", fontsize=8, color="white",
                                           animated=True)
                self.stage_artists.append((patch, label))
            patch, label = self.stage_artists[index]
            patch.set_width(width)
            label.set_x(segment["start"] + width / 2)
            label.set_visible(width >= .50)
        outcome = ""
        if result is not None:
            outcome = " | SUCCESS" if result.get("success") else f" | FAILED: {result.get('failure')}"
        self.title.set_text(f"RGB-D pick and place | {self.mode} | seed {seed} | {state} | t={stamp:.2f} s{outcome}")
        self.metrics.set_text(f"CPU MuJoCo physics | hand contacts {row['contacts']} | simulated hand normal total {row['total_normal_n']:.2f} N")
        if self.background is None:
            self.figure.canvas.draw()
            self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.figure.canvas.restore_region(self.background)
        for artist in self.dynamic_artists:
            self.figure.draw_artist(artist)
        for patch, label in self.stage_artists:
            self.figure.draw_artist(patch)
            if label.get_visible():
                self.figure.draw_artist(label)
        pixels = np.asarray(self.figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        if append_video:
            self.writer.append_data(pixels)
            self.video_frames += 1
        self.last_frame = pixels
        if self.live:
            self.live_window.show(pixels)
        return pixels

    def hold_final(self, frames):
        if self.last_frame is not None:
            for _ in range(frames):
                self.writer.append_data(self.last_frame)
                self.video_frames += 1

    def close(self):
        self.writer.close()
        if not self.live:
            self.plt.close(self.figure)


class EpisodeRecorder:
    def __init__(self, model, data, scenario, folder, output, args, pyplot, viewer):
        self.model, self.data = model, data
        self.scenario, self.folder, self.args = scenario, folder, args
        self.viewer = viewer
        self.reader = HandContactReader(model)
        self.overview = mujoco.Renderer(model, height=480, width=640)
        self.detail = mujoco.Renderer(model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        configure_overview(self.camera)
        self.detail_camera = mujoco.MjvCamera()
        self.detail_camera.distance = .40
        self.detail_camera.azimuth = 135
        self.detail_camera.elevation = -18
        self.hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_C_MC")
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(args.force_vectors)
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
        self.observer = CameraObserver(model, CameraConfig(seed=scenario.seed))
        self.dashboard = Dashboard(pyplot, args.fps, output, getattr(args, "view", bool(viewer)), args.force_mode)
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.next_time = 0.
        self.saved_stages = set()
        self.started = time.monotonic()
        self.paused_wall = 0.
        self.opengl = {}
        self.latest = None

    def observe(self, stage, data, *, force=False, result=None):
        stamp = float(data.time)
        self.dashboard.track_stage(stage_name(stage), stamp)
        if self.rows and abs(stamp - self.rows[-1]["simulation_time_s"]) < 1e-10:
            if result is not None and self.latest is not None:
                row, image, detail_image, observation, belief = self.latest
                pixels = self.dashboard.draw(row, image, detail_image, observation, belief, self.scenario.seed,
                                             result, append_video=False)
                imageio.imwrite(self.folder / "keyframes" / "final.png", pixels)
            return
        if not force and stamp + 1e-10 < self.next_time:
            return
        if self.viewer:
            self.viewer.sync()
        readings = self.reader.read(data)
        total = self.reader.total(readings)
        self.overview.update_scene(data, camera=self.camera, scene_option=self.option)
        image = self.overview.render().copy()
        image = DISPLAY_GAMMA[image]
        hand_rotation = data.xmat[self.hand_body].reshape(3, 3)
        self.detail_camera.lookat[:] = data.xpos[self.hand_body] + .09 * hand_rotation[:, 2]
        self.detail.update_scene(data, camera=self.detail_camera, scene_option=self.option)
        detail_image = self.detail.render().copy()
        detail_image = DISPLAY_GAMMA[detail_image]
        observation = self.observer.poll(data)
        if observation is None:
            return
        belief = estimate_object(observation, self.scenario.object_kind)
        if not self.opengl:
            from OpenGL import GL
            self.opengl = {name: (GL.glGetString(enum) or b"unavailable").decode(errors="replace")
                           for name, enum in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                                               ("version", GL.GL_VERSION))}
            self.opengl["scope"] = "This process's active offscreen renderer context."
        row = {
            "frame_index": len(self.rows), "simulation_step": round(stamp / self.model.opt.timestep),
            "simulation_time_s": stamp, "stage": stage_name(stage),
            "contacts": total.contacts, "total_normal_n": total.normal,
            "total_tangential_n": total.tangential,
            "wrist_observation_time_s": observation.timestamp,
            "wrist_depth_valid_fraction": float(np.isfinite(observation.depth).mean()),
            "display_estimate_confidence": belief.confidence,
            **{f"{finger}_normal_n": readings[finger].normal if finger in readings else 0. for finger in CONTACT_CHANNELS},
            **{f"{finger}_tangential_n": readings[finger].tangential if finger in readings else 0. for finger in CONTACT_CHANNELS},
        }
        self.rows.append(row)
        self.latest = row, image, detail_image, observation, belief
        pixels = self.dashboard.draw(row, image, detail_image, observation, belief, self.scenario.seed, result)
        if result is not None:
            imageio.imwrite(self.folder / "keyframes" / "final.png", pixels)
        stage = row["stage"]
        if stage not in self.saved_stages:
            self.saved_stages.add(stage)
            imageio.imwrite(self.folder / "keyframes" / f"{len(self.saved_stages):02d}_{stage.lower()}.png", pixels)
        self.next_time = stamp + 1. / self.args.fps

    def stage_event(self, stage, data):
        self.events.append({"stage_event": str(stage), "simulation_time_s": float(data.time)})
        if stage == "initial":
            self.observe("SEARCH", data, force=True)

    def close(self):
        self.observer.close()
        self.detail.close()
        self.overview.close()
        self.dashboard.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physics-profile", choices=("legacy-v5", "physical-v6"), default="legacy-v5")
    parser.add_argument("--support-gate", action="store_true")
    parser.add_argument("--force-mode", choices=("contact", "fixed", "fixed_calibrated"), default="contact")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--joint-space", action="store_true",
                        help="Use the historical joint-space trajectory instead of Cartesian motion.")
    parser.add_argument("--legacy-transfer", action="store_true",
                        help="Disable the default post-lift RGB-D transfer correction.")
    parser.add_argument("--placement-retention", action="store_true",
                        help="Experimental legacy finger tightening during descent.")
    parser.add_argument("--depth-noise", type=float, default=0., help="Controller depth noise standard deviation in metres.")
    parser.add_argument("--dropout", type=float, default=0., help="Controller depth pixel dropout fraction.")
    parser.add_argument("--first-grasp-bias", nargs=3, type=float,
                        help="Experimental first-candidate offset (x y z, metres) for a physical recovery demonstration.")
    parser.add_argument("--output", type=Path, help="New MP4 filename; existing output is preserved.")
    parser.add_argument("--view", action="store_true", help="Live dashboard with actual MuJoCo views and sensors.")
    parser.add_argument("--native-viewer", action="store_true", help="Also open the native MuJoCo 3D viewer; implies --view.")
    parser.add_argument("--fps", type=int, help="Physical sampling rate: default 3 with --view, otherwise 10.")
    parser.add_argument("--force-vectors", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0.)
    parser.add_argument("--view-hold-seconds", type=float,
                        help="Close interactive windows this many seconds after the episode; default keeps them open.")
    args = parser.parse_args()
    if args.native_viewer:
        args.view = True
    if args.fps is None:
        args.fps = 3 if args.view else 10
    if args.fps < 1 or args.fps > 100 or args.hold_seconds < 0:
        parser.error("--fps must be 1..100; --hold-seconds must be nonnegative")
    if args.view_hold_seconds is not None and args.view_hold_seconds < 0:
        parser.error("--view-hold-seconds must be nonnegative")
    if args.max_attempts < 1 or (args.first_grasp_bias is not None and not np.isfinite(args.first_grasp_bias).all()):
        parser.error("Positive attempts and finite first-grasp offsets are required")
    try:
        CameraConfig(depth_noise_std=args.depth_noise, dropout=args.dropout)
    except ValueError as error:
        parser.error(str(error))
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT / "results" / "demo_runs" / f"seed_{args.seed}_{stamp}.mp4"
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must be an MP4 filename")
    if args.output.exists() or args.output.with_name(f"{args.output.stem}_evidence").exists():
        parser.error("output/evidence already exists; choose a new --output")
    return args


def main():
    args = arguments()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base_folder = args.output.with_name(f"{args.output.stem}_evidence")
    base_folder.mkdir(exist_ok=False)
    scene = build_pick_place_scene(base_folder / "scene.xml", "ycb_can", physics_profile=args.physics_profile)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 480)
    pause = threading.Event()
    replay = threading.Event()
    viewer = None

    def key_callback(key):
        if key in (32, 80):
            if pause.is_set():
                pause.clear()
            else:
                pause.set()
        elif key == 82:
            replay.set()
            pause.clear()

    if args.native_viewer:
        from mujoco import viewer as mj_viewer
        from kuka_sharpa_grasp.pick_place import reset_task
        reset_task(model, data, Scenario(args.seed))
        viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback)
        configure_overview(viewer.cam)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(args.force_vectors)
        print("Native viewer: Space/P pause; R reruns after the current episode; close viewer to exit.", flush=True)
    source = source_snapshot()
    episode = 0
    recorder = None
    try:
        while True:
            output = args.output if episode == 0 else args.output.with_name(f"{args.output.stem}_replay_{episode:02d}.mp4")
            folder = base_folder if episode == 0 else output.with_name(f"{output.stem}_evidence")
            if episode:
                if output.exists() or folder.exists():
                    raise FileExistsError(f"Replay output already exists: {output}")
                folder.mkdir(exist_ok=False)
            (folder / "keyframes").mkdir(exist_ok=False)
            recorder = None
            result, trace = None, []
            error = None
            cancelled = False
            wall_start = time.monotonic()
            try:
                recorder = EpisodeRecorder(model, data, Scenario(args.seed), folder, output, args, plt, viewer)
                if args.view:
                    recorder.dashboard.live_window.bind_keys(key_callback)

                def step_callback(stage, live):
                    if viewer and not viewer.is_running():
                        raise InterruptedError("Native viewer closed during the physical episode")
                    if args.view and not recorder.dashboard.live_window.running:
                        raise InterruptedError("Live sensor dashboard closed during the physical episode")
                    if args.view and pause.is_set():
                        pause_start = time.monotonic()
                        while pause.is_set() and recorder.dashboard.live_window.running:
                            if viewer:
                                viewer.sync()
                            recorder.dashboard.live_window.pump()
                            time.sleep(.02)
                        recorder.paused_wall += time.monotonic() - pause_start
                    recorder.observe(stage, live)
                    if args.view:
                        remaining = float(live.time) - (time.monotonic() - recorder.started - recorder.paused_wall)
                        if remaining > 0:
                            time.sleep(min(remaining, .02))

                result, trace = run_task(model, data, Scenario(args.seed),
                                          frame_callback=recorder.stage_event,
                                          step_callback=step_callback, force_mode=args.force_mode,
                                          max_attempts=args.max_attempts, first_grasp_bias=args.first_grasp_bias,
                                          cartesian_motion=not args.joint_space,
                                          visual_transfer=not args.legacy_transfer,
                                          placement_retention=args.placement_retention,
                                          support_load_sensor=args.support_gate,
                                          support_required=args.support_gate,
                                          camera_config=CameraConfig(depth_noise_std=args.depth_noise,
                                              dropout=args.dropout, seed=args.seed))
                final_stage = trace[-1]["state"] if trace else "VERIFY"
                recorder.observe(final_stage, data, force=True, result=result)
                recorder.dashboard.hold_final(round(args.hold_seconds * args.fps))
            except InterruptedError as exc:
                cancelled = True
                error = {"type": type(exc).__name__, "message": str(exc)}
            except BaseException as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                if recorder is not None:
                    recorder.close()
                    if recorder.rows:
                        with (folder / "frame_samples.csv").open("x", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(recorder.rows[0]))
                            writer.writeheader()
                            writer.writerows(recorder.rows)
                    final_source = source_snapshot()
                    metadata = {
                        "entry_point": "scripts/demo_pick_place.py", "seed": args.seed,
                        "force_mode": args.force_mode, "max_attempts": args.max_attempts,
                        "experimental_first_grasp_bias_m": args.first_grasp_bias,
                        "task": ("physical_v6_pick_place" if args.physics_profile == "physical-v6" else
                                 "physical_v4_pick_place" if args.joint_space else
                                 "physical_v5_pick_place"), "physics_backend": "classic_mujoco_cpu",
                        "physics_profile": args.physics_profile, "support_required": args.support_gate,
                        "cartesian_motion": not args.joint_space,
                        "visual_transfer": not args.legacy_transfer,
                        "placement_retention": args.placement_retention,
                        "controller_depth_noise_std_m": args.depth_noise,
                        "controller_pixel_dropout": args.dropout,
                        "workcell_version": WORKCELL_VERSION, "mujoco": mujoco.__version__,
                        "python": sys.version, "opengl": recorder.opengl, "source_sha256": source,
                        "source_sha256_end": final_source,
                        "source_files_changed_during_run": source != final_source,
                        "video": str(output), "fps": args.fps,
                        "native_viewer": bool(viewer), "live_dashboard": bool(args.view),
                        "native_viewer_note": "Optional passive MuJoCo viewer. The OpenGL report above is for the offscreen recording context.",
                        "physical_sample_count": len(recorder.rows), "physics_timestep_s": float(model.opt.timestep),
                        "video_frame_count": recorder.dashboard.video_frames,
                        "physical_end_s": float(data.time), "wall_seconds": time.monotonic() - wall_start,
                        "paused_wall_seconds": recorder.paused_wall,
                        "actual_hand_control": result.get("hand_control") if result is not None else None,
                        "frozen_final_seconds": args.hold_seconds if result else 0,
                        "final_annotation": "A result annotation at an already recorded physical timestamp is saved only to final.png, without adding a video frame.",
                        "sampling": "Callback after actual mj_step/record; wrist RGB/depth and finger-segment forces share each physical timestamp. No trajectory interpolation or authored object animation.",
                        "contact_scope": "Total and plotted channels include target-object contacts on finger segments and palm, matching HandContactReader; self and table contacts are excluded.",
                        "video_dimensions": [1600, 900],
                        "display_rendering": "One fixed Agg canvas with cached artists; live Tk displays its pixels directly via PIL. Physical samples are recorded synchronously, without synthetic interpolation.",
                        "perception_overlay": "Independent visible-surface display estimate; not the controller's latest belief.",
                        "overview_display_gamma": .72,
                        "display_note": "Overview and hand-detail views use gamma tone mapping. The detail camera follows the hand body kinematics only and is for display. Wrist RGB/depth remains the original sensor observation; no controller observation or physical state is changed.",
                        "result": result, "error": error, "stage_events": recorder.events,
                        "cancelled": cancelled,
                        "execution_status": ("CANCELLED" if cancelled else "ERROR" if error else
                                             "SUCCESS" if result and result["success"] else "FAILED"),
                        "trace_available": result is not None,
                        "trace_availability_note": ("Complete controller trace returned by run_task." if result is not None else
                                                    "run_task did not return a controller trace; trace.json is empty. Actual partial samples remain in frame_samples.csv and stage_events."),
                    }
                    write_json(folder / "metadata.json", metadata)
                    write_json(folder / "trace.json", trace)
            print(json.dumps({"execution_status": "CANCELLED" if cancelled else "SUCCESS" if result and result["success"] else "FAILED",
                              "cancelled": cancelled, "error": error,
                              "success": result["success"] if result else None,
                              "failure": result["failure"] if result else None,
                              "duration_s": result["duration"] if result else float(data.time), "video": str(output),
                              "evidence": str(folder)}, indent=2), flush=True)
            if not args.view or error:
                break
            print("Episode finished; live dashboard remains open. R starts a fresh physical replay.", flush=True)
            finished_at = time.monotonic()
            while recorder.dashboard.live_window.running and (not viewer or viewer.is_running()) and not replay.is_set():
                if args.view_hold_seconds is not None and time.monotonic() - finished_at >= args.view_hold_seconds:
                    recorder.dashboard.live_window.close()
                    if viewer:
                        viewer.close()
                    break
                if viewer:
                    viewer.sync()
                recorder.dashboard.live_window.pump()
                time.sleep(.02)
            if not recorder.dashboard.live_window.running or (viewer and not viewer.is_running()):
                break
            replay.clear()
            recorder.dashboard.live_window.close()
            episode += 1
    finally:
        if viewer:
            viewer.close()
        if recorder is not None and recorder.dashboard.live_window is not None:
            recorder.dashboard.live_window.close()
        plt.close("all")


if __name__ == "__main__":
    main()
