#!/usr/bin/env python3
"""Run the V7 tactile shared-teleoperation workbench.

The graphical mode is a compact Tk desktop panel. The same physical loop is
available headlessly for reproducible videos and action-log replay:

    python scripts/run_v7_workbench.py --duration 8 --output results/v7_demo
    python scripts/run_v7_workbench.py --replay actions.json --output results/v7_replay
    python scripts/run_v7_workbench.py --view --duration 30 --output results/v7_operator

The scripted headless mode is a physical demonstration sequence, not a task
evaluator.  It intentionally records ``task_success: null``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
# Leave MuJoCo's renderer backend selection to the host.  macOS uses the
# native backend; Linux 4090 workers can select EGL in their launch wrapper.
sys.path.insert(0, str(PROJECT / "src"))

import imageio.v2 as imageio
import mujoco
import numpy as np

from kuka_sharpa_grasp.camera import CameraConfig, CameraObserver
from kuka_sharpa_grasp.v7_episode import TeleopEpisodeRecorder
from kuka_sharpa_grasp.teleop import KeyboardTeleop, TeleopActionLog, TeleopCommand
from kuka_sharpa_grasp.v7_scene import build_v7_workbench_scene, default_v7_scene_config
from kuka_sharpa_grasp.v7_session import WorkbenchSession


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
VIDEO_SIZE = (1280, 720)
OPERATOR_EVENTS_FORMAT = "kuka-sharpa-v7-operator-events"


def source_snapshot() -> dict[str, str]:
    files = sorted((PROJECT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if not path.name.startswith("._")}


def write_operator_events(path: Path, events: list[dict]) -> None:
    """Persist raw Tk input events separately from transformed commands.

    The event stream is provenance evidence only.  It does not assert that a
    human was the operator; that still requires an honest video/report claim.
    """
    payload = {"format": OPERATOR_EVENTS_FORMAT, "version": 1, "events": events}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


class OperatorEventReplay:
    """Replay GUI state changes that do not fit Cartesian TeleopCommand."""

    def __init__(self, path):
        self.path = Path(path)
        self.events, self.cursor = [], 0
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("format") != OPERATOR_EVENTS_FORMAT or payload.get("version") != 1:
            raise ValueError("unsupported operator event log")
        previous = -math.inf
        for event in payload["events"]:
            timestamp = float(event["time_s"])
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp < previous:
                raise ValueError("operator event timestamps must be monotonic and finite")
            previous = timestamp
            self.events.append(event)

    def apply_due(self, timestamp, session):
        while self.cursor < len(self.events) and self.events[self.cursor]["time_s"] <= timestamp + 1e-12:
            event = self.events[self.cursor]
            if event["event"] == "MODE_CHANGE":
                if event["mode"] not in {"shared", "manual"}:
                    raise ValueError("unknown GUI control mode")
                session.shared_control = event["mode"] == "shared"
            elif event["event"] == "TARGET_SELECT":
                session.select_target(event["target"])
            elif event["event"] == "FINGER_TARGET":
                session.set_finger_target(event["finger"], event["value"])
            elif event["event"] == "KEY_DOWN" and event["key"] == "escape":
                session.stop("OPERATOR_ESCAPE")
            # Ordinary held keys are already represented in action_log.json.
            self.cursor += 1


def _camera(lookat, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


class WorkbenchRenderer:
    """Three synchronized native MuJoCo views plus tactile annotations."""

    def __init__(self, model):
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), 480)
        self.model = model
        self.overview = mujoco.Renderer(model, height=480, width=640)
        self.hand = mujoco.Renderer(model, height=360, width=480)
        self.overview_camera = _camera((.70, -.03, .50), 1.75, 130., -24.)
        self.hand_camera = _camera((.65, -.03, .58), .43, 135., -18.)
        self.hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_C_MC")
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.camera = CameraObserver(model, CameraConfig(name="wrist_rgbd", width=320, height=240))

    def frame(self, data, snapshot):
        self.overview.update_scene(data, camera=self.overview_camera, scene_option=self.option)
        overview = self.overview.render().copy()
        rotation = data.xmat[self.hand_body].reshape(3, 3)
        self.hand_camera.lookat[:] = data.xpos[self.hand_body] + .08 * rotation[:, 2]
        self.hand.update_scene(data, camera=self.hand_camera, scene_option=self.option)
        detail = self.hand.render().copy()
        observation = self.camera.poll(data)
        if observation is None:
            rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            depth = np.zeros((240, 320), dtype=np.uint8)
        else:
            rgb = observation.rgb.copy()
            depth_data = np.asarray(observation.depth, dtype=float)
            valid = np.isfinite(depth_data)
            depth = np.zeros((*depth_data.shape, 3), dtype=np.uint8)
            if valid.any():
                scaled = np.clip((depth_data - .2) / 1.0 * 255., 0., 255.)
                depth[..., 0] = np.where(valid, scaled, 0.)
                depth[..., 1] = np.where(valid, 255. - scaled, 0.)
                depth[..., 2] = np.where(valid, 70., 0.)
        return compose_frame(overview, detail, rgb, depth, snapshot)

    def close(self):
        self.camera.close()
        self.overview.close()
        self.hand.close()


def _resize(image, size):
    from PIL import Image
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def compose_frame(overview, detail, rgb, depth, snapshot):
    """Compose stable 1280x720 output with five native tactile channels."""
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", VIDEO_SIZE, (23, 29, 35))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas.paste(Image.fromarray(_resize(overview, (720, 540))), (16, 36))
    canvas.paste(Image.fromarray(_resize(detail, (500, 375))), (744, 36))
    canvas.paste(Image.fromarray(_resize(rgb, (240, 180))), (744, 425))
    canvas.paste(Image.fromarray(_resize(depth, (240, 180))), (1000, 425))
    draw.text((18, 10), "V7 tactile shared teleoperation | native MuJoCo views", fill=(235, 240, 245), font=font)
    draw.text((760, 18), f"{snapshot.get('status', 'UNKNOWN')} | target: {snapshot.get('selected_target', '-')}",
              fill=(229, 191, 83), font=font)
    tactile = snapshot.get("tactile", {}).get("fingers", {})
    draw.text((18, 594), "TACTILE CONTACT CHANNELS  (native contact wrench; no target-geom oracle)",
              fill=(229, 235, 239), font=font)
    colors = ((220, 90, 90), (45, 160, 215), (150, 100, 195), (50, 170, 120), (200, 155, 55))
    for index, (finger, color) in enumerate(zip(FINGERS, colors)):
        reading = tactile.get(finger, {})
        x = 18 + index * 145
        normal = float(reading.get("normal_force_n", 0.))
        shear = float(reading.get("tangential_force_n", 0.))
        slip = float(reading.get("slip_score", 0.))
        draw.text((x, 616), finger, fill=color, font=font)
        draw.rectangle((x, 634, x + 112, 646), outline=(105, 115, 125))
        draw.rectangle((x, 634, x + min(112, normal * 8.), 646), fill=color)
        draw.text((x, 652), f"N {normal:5.1f} | shear {shear:5.1f}", fill=(220, 225, 230), font=font)
        draw.text((x, 668), f"contacts {int(reading.get('contacts', 0))} | slip {slip:.2f}", fill=(170, 180, 190), font=font)
    return np.asarray(canvas)


def scripted_command(timestamp: float, index: int) -> TeleopCommand:
    """Small bounded physical motion used only for headless demonstration."""
    cycle = (index // 30) % 4
    keys = ({"w"}, {"d"}, {"s", "]"}, {"a", "["})[cycle]
    return KeyboardTeleop().command(keys, timestamp, source="scripted_demo")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--duration", type=float, default=10., help="Physical seconds for headless mode")
    parser.add_argument("--fps", type=int, default=20, help="Desktop update and video sampling rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path, help="TeleopActionLog JSON to replay")
    parser.add_argument("--manual", action="store_true", help="Disable shared tactile assistance")
    parser.add_argument("--view", action="store_true", help="Open the Tk desktop panel")
    parser.add_argument("--automated-gui-check", action="store_true",
                        help="Generate labeled automated Tk events; requires --view and duration >= 3")
    args = parser.parse_args()
    if args.duration <= 0 or not math.isfinite(args.duration):
        parser.error("--duration must be finite and positive")
    if args.fps < 1 or args.fps > 60:
        parser.error("--fps must be between 1 and 60")
    if args.automated_gui_check and (not args.view or args.duration < 3):
        parser.error("--automated-gui-check requires --view and --duration >= 3")
    if args.view and args.replay:
        parser.error("--replay uses the headless renderer; omit --view")
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT / "results" / "v7_workbench" / f"run_{stamp}"
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def make_run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    scene_path = build_v7_workbench_scene(args.output / "scene.xml", config=default_v7_scene_config(args.seed))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    session = WorkbenchSession(model, default_v7_scene_config(args.seed), shared_control=not args.manual)
    return model, session


def run_headless(args, model, session):
    log = TeleopActionLog.from_json(args.replay) if args.replay else TeleopActionLog()
    gui_replay = OperatorEventReplay(args.replay.parent / "operator_events.json") if args.replay else None
    recorder = TeleopEpisodeRecorder(
        args.output,
        seed=args.seed,
        mode="replay" if args.replay else "scripted_demo",
        shared_control=session.shared_control,
        initial_commands=log.commands,
        metadata={
            "entry_point": "scripts/run_v7_workbench.py",
            "fps": args.fps,
            "physics_backend": "classic_mujoco_cpu",
            "renderer": "native_mujoco_renderer",
            "mujoco": mujoco.__version__,
            "scene": "v7_workbench_r1",
            "source_sha256": source_snapshot(),
            "operator_events_replayed": len(gui_replay.events) if gui_replay else 0,
        },
    )
    renderer = WorkbenchRenderer(model)
    writer = imageio.get_writer(str(args.output / "workbench.mp4"), fps=args.fps, codec="libx264", macro_block_size=None)
    trace, actions = [], list(log.replay())
    command_index, sample_period = 0, 1. / args.fps
    next_sample, step_index = 0., 0
    try:
        while float(session.data.time) < args.duration - 1e-12:
            timestamp = float(session.data.time)
            if args.replay:
                gui_replay.apply_due(timestamp, session)
                due = log.due(timestamp)
                for replay_command in due:
                    session.command(replay_command)
                command = None
            else:
                command = scripted_command(timestamp, command_index) if step_index % max(1, round(sample_period / model.opt.timestep)) == 0 else None
                if command is not None:
                    recorder.record_command(command)
                    actions.append(command)
                    command_index += 1
            snapshot = session.step(command)
            recorder.record_snapshot(snapshot)
            trace.append(snapshot)
            step_index += 1
            if timestamp + 1e-12 >= next_sample:
                writer.append_data(renderer.frame(session.data, snapshot))
                next_sample += sample_period
    finally:
        writer.close()
        renderer.close()
    metadata = recorder.finalize(
        task_success=None,
        task_success_note="Workbench demo has no evaluator; null is intentional.",
        events=session.events,
    )
    return metadata


class DesktopPanel:
    def __init__(self, args, model, session):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.args, self.model, self.session = args, model, session
        self.renderer = WorkbenchRenderer(model)
        self.writer = imageio.get_writer(
            str(args.output / "workbench.mp4"), fps=args.fps, codec="libx264", macro_block_size=None)
        self.keyboard = KeyboardTeleop()
        self.recorder = TeleopEpisodeRecorder(
            args.output,
            seed=args.seed,
            mode="interactive",
            shared_control=session.shared_control,
            metadata={
                "entry_point": "scripts/run_v7_workbench.py",
                "fps": args.fps,
                "input_provenance": "automated_gui_input" if args.automated_gui_check else "unverified_gui_input",
                "physics_backend": "classic_mujoco_cpu",
                "renderer": "native_mujoco_renderer",
                "mujoco": mujoco.__version__,
                "scene": "v7_workbench_r1",
                "source_sha256": source_snapshot(),
            },
        )
        self.held: set[str] = set()
        self.operator_events: list[dict] = []
        self.close_reason = "window_close"
        self.automation_cursor = 0
        self.running, self.paused = True, False
        self.root = tk.Tk()
        self.root.title("KUKA + Sharpa V7 | tactile shared teleoperation")
        self.root.geometry("1280x820")
        self.root.configure(bg="#171d23")
        self.image_label = tk.Label(self.root, bg="#171d23")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)
        controls = tk.Frame(self.root, bg="#202830")
        controls.pack(fill="x", padx=8, pady=(0, 8))
        self.mode = tk.StringVar(value="shared" if session.shared_control else "manual")
        self.mode_buttons = {}
        tk.Label(controls, text="Mode", fg="white", bg="#202830").pack(side="left", padx=5)
        for label, value in (("Shared", "shared"), ("Manual", "manual")):
            button = tk.Radiobutton(controls, text=label, variable=self.mode, value=value, command=self.change_mode,
                                   fg="white", bg="#202830", selectcolor="#344350", activebackground="#202830")
            button.pack(side="left")
            self.mode_buttons[value] = button
        self.target = tk.StringVar(value=session.selected_target)
        choices = [spec.object_id for spec in session.config.registry.actionable]
        tk.Label(controls, text="Target", fg="white", bg="#202830").pack(side="left", padx=(20, 4))
        menu = tk.OptionMenu(controls, self.target, *choices, command=self.change_target)
        menu.configure(bg="#344350", fg="white", activebackground="#526576")
        menu.pack(side="left")
        self.target_menu = menu["menu"]
        self.target_choices = choices
        self.status = tk.Label(controls, text="", fg="#e5bf53", bg="#202830", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=14)
        self.sliders = {}
        for finger in ("thumb", "index", "middle"):
            scale = tk.Scale(controls, from_=0., to=1., resolution=.01, orient="horizontal", length=115,
                             label=finger, command=lambda value, name=finger: self.set_finger(name, value),
                             fg="white", bg="#202830", highlightthickness=0)
            scale.set(0.)
            scale.pack(side="left")
            self.sliders[finger] = scale
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.bind("<FocusOut>", self.clear_keys)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.focus_force()
        self.after_id = self.root.after(round(1000 / args.fps), self.tick)

    def change_mode(self):
        self.session.shared_control = self.mode.get() == "shared"
        self.record_operator_event("MODE_CHANGE", mode=self.mode.get())

    def change_target(self, value):
        self.session.select_target(value)
        self.record_operator_event("TARGET_SELECT", target=value)

    def set_finger(self, finger, value):
        self.session.set_finger_target(finger, float(value))
        self.record_operator_event("FINGER_TARGET", finger=finger, value=float(value))

    def record_operator_event(self, event, **fields):
        self.operator_events.append({"time_s": float(self.session.data.time),
                                     "event": str(event), **fields})

    def key_down(self, event):
        key = {"bracketleft": "[", "bracketright": "]"}.get(event.keysym.lower(), event.keysym.lower())
        self.record_operator_event("KEY_DOWN", key=key,
                                   held_keys=sorted(self.held | {key}))
        if key == "escape":
            self.session.stop("OPERATOR_ESCAPE")
            self.held.clear()
        else:
            self.held.add(key)

    def key_up(self, event):
        key = {"bracketleft": "[", "bracketright": "]"}.get(event.keysym.lower(), event.keysym.lower())
        self.held.discard(key)
        self.record_operator_event("KEY_UP", key=key, held_keys=sorted(self.held))

    def clear_keys(self, _event=None):
        if self.held:
            self.record_operator_event("FOCUS_LOST", held_keys=sorted(self.held))
        self.held.clear()

    def tick(self):
        if not self.running:
            return
        if self.args.automated_gui_check:
            self.automated_gui_input()
        steps = max(1, round((1. / self.args.fps) / float(self.model.opt.timestep)))
        # Sample one operator command per 50 ms panel tick, then advance the
        # remaining MuJoCo steps without repeating that Cartesian increment.
        command = self.keyboard.command(self.held, float(self.session.data.time))
        self.recorder.record_command(command)
        snapshot = self.session.step(command)
        self.recorder.record_snapshot(snapshot)
        for _ in range(steps - 1):
            if float(self.session.data.time) >= self.args.duration - 1e-12:
                break
            snapshot = self.session.step()
            self.recorder.record_snapshot(snapshot)
        pixels = self.renderer.frame(self.session.data, snapshot)
        self.writer.append_data(pixels)
        image = self.ImageTk.PhotoImage(self.Image.fromarray(pixels))
        self.image_label.configure(image=image)
        self.image_label.image = image
        self.status.configure(text=f"{snapshot['status']} | t={snapshot['time_s']:.2f}s | interventions={snapshot['interventions']}")
        if float(self.session.data.time) >= self.args.duration - 1e-12:
            self.close(reason="duration_reached")
            return
        self.after_id = self.root.after(round(1000 / self.args.fps), self.tick)

    def automated_gui_input(self):
        """Exercise actual Tk widgets/events without claiming human input."""
        sequence = (
            (0., "target", "tool_marker"), (0., "mode", "manual"),
            (.10, "down", "w"), (.20, "up", "w"),
            (.30, "down", "e"), (.40, "up", "e"),
            (.50, "down", "bracketright"), (.65, "up", "bracketright"),
            (.70, "finger", ("index", .25)), (.70, "finger", ("thumb", .30)),
            (.80, "mode", "shared"), (1., "down", "Escape"),
            (1.05, "up", "Escape"), (1.10, "down", "r"), (1.20, "up", "r"),
            (1.30, "mode", "manual"), (1.40, "down", "d"), (1.50, "up", "d"),
            (2., "down", "w"), (2.05, "focus", None), (2.10, "up", "w"),
        )
        while self.automation_cursor < len(sequence):
            timestamp, kind, value = sequence[self.automation_cursor]
            if timestamp > float(self.session.data.time) + 1e-12:
                break
            if kind == "target":
                self.target_menu.invoke(self.target_choices.index(value))
            elif kind == "mode":
                self.mode_buttons[value].invoke()
            elif kind == "finger":
                self.sliders[value[0]].set(value[1])
                self.root.update_idletasks()
            elif kind == "focus":
                self.root.event_generate("<FocusOut>")
            else:
                self.root.focus_force()
                self.root.update_idletasks()
                self.root.event_generate(f"<Key{'Press' if kind == 'down' else 'Release'}-{value}>")
            self.automation_cursor += 1

    def close(self, reason="window_close"):
        if self.running:
            self.close_reason = str(reason)
            self.writer.close()
            write_operator_events(self.args.output / "operator_events.json", self.operator_events)
            note = ("Interactive recording reached the requested duration; no evaluator result was supplied."
                    if self.close_reason == "duration_reached"
                    else "Interactive recording ended by window close; no evaluator result was supplied.")
            self.recorder.finalize(
                task_success=None,
                task_success_note=note,
                events=self.session.events,
                extra_metadata={
                    "video": "workbench.mp4",
                    "operator_events": "operator_events.json",
                    "operator_event_count": len(self.operator_events),
                    "operator_provenance": "Tk input events captured; human identity and intent are not verified.",
                    "close_reason": self.close_reason,
                    "final_shared_control": self.session.shared_control,
                },
            )
        self.running = False
        self.renderer.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    args = parse_args()
    model, session = make_run(args)
    if args.view:
        DesktopPanel(args, model, session).run()
        return
    metadata = run_headless(args, model, session)
    print(json.dumps({"status": "recorded", "output": str(args.output), "duration_s": metadata["duration_s"],
                      "task_success": None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
