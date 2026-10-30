"""Viewer controls and reproducible presentation inputs, without physics edits."""

from __future__ import annotations

import argparse
import json
import queue
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .rollout import GraspRollout
from .scenarios import ScenarioSampler
from .scene import OBJECT_KINDS, PROJECT_ROOT


KEY_ACTIONS = {32: "pause", 80: "pause", 82: "replay", 67: "contacts", 70: "forces", 86: "camera"}
KEY_NAMES = {"SPACE": 32, "P": 80, "R": 82, "C": 67, "F": 70, "V": 86}


@dataclass(frozen=True)
class PresentationTrial:
    object_kind: str
    seed: int
    controller: str
    duration_s: float
    object_pos: tuple[float, float, float]
    object_mass: float
    object_friction: float
    source_trial_id: str | None = None
    source_trial_json: str | None = None
    source_record_sha256: str | None = None
    source_physics_sha256: str | None = None
    source_config_hash: str | None = None
    source_provenance: str = "local_inputs"

    def rollout_arguments(self) -> dict:
        return {
            "seed": self.seed, "controller": self.controller, "duration": self.duration_s,
            "object_pos": self.object_pos, "object_mass": self.object_mass,
            "object_friction": self.object_friction,
        }


def add_trial_arguments(parser: argparse.ArgumentParser, default_object: str = "ycb_can") -> None:
    parser.add_argument("--controller", choices=("fixed", "force"), default="force")
    parser.add_argument("--object", choices=OBJECT_KINDS, default=default_object)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--jitter", type=float, default=0.005)
    parser.add_argument("--friction", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--mass-range", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--trial-json", type=Path, help="Replay immutable inputs from a saved experiment trial.")


def load_trial_record(args: argparse.Namespace) -> dict | None:
    if args.trial_json is None:
        if args.duration <= 0 or not np.isfinite(args.duration):
            raise ValueError("duration must be finite and positive")
        return None
    record = json.loads(args.trial_json.read_text(encoding="utf-8"))
    if record.get("object_kind") not in OBJECT_KINDS:
        raise ValueError("trial JSON has no supported object_kind")
    if record.get("controller") not in ("fixed", "force"):
        raise ValueError("trial JSON has no supported controller")
    from .experiments import content_hash, software_versions, source_identity

    digest = record.get("record_sha256")
    if digest is not None and digest != content_hash({key: value for key, value in record.items() if key != "record_sha256"}):
        raise ValueError("saved trial record SHA256 mismatch")
    source = record.get("source_sha256")
    if source is not None and source != source_identity()["sha256"]:
        raise ValueError("saved trial physical source/assets SHA256 differs from this checkout")
    current_versions = software_versions()
    version_count = 0
    for name in ("python", "mujoco", "numpy"):
        recorded = record.get("versions", {}).get(name, record.get(f"{name}_version"))
        if recorded is not None:
            version_count += 1
            if recorded != current_versions[name]:
                raise ValueError(f"saved trial {name} version differs: {recorded} vs {current_versions[name]}")
    record["_replay_provenance"] = "verified_record_source_versions" if digest and source and version_count == 3 else "unverified_missing_provenance"
    args.object = record["object_kind"]
    return record


def resolve_trial(args: argparse.Namespace, rollout: GraspRollout, record: dict | None) -> PresentationTrial:
    if record is not None:
        position = record.get("sampled_object_pos")
        if position is None:
            position = [record[f"object_{axis}"] for axis in "xyz"]
        trial = PresentationTrial(
            object_kind=record["object_kind"], seed=int(record["seed"]),
            controller=record["controller"], duration_s=float(record["duration_s"]),
            object_pos=tuple(float(value) for value in position),
            object_mass=float(record["object_mass"]), object_friction=float(record["object_friction"]),
            source_trial_id=record.get("trial_id"), source_trial_json=str(args.trial_json.resolve()),
            source_record_sha256=record.get("record_sha256"), source_physics_sha256=record.get("source_sha256"),
            source_config_hash=record.get("config_hash"), source_provenance=record["_replay_provenance"],
        )
    else:
        sampler = ScenarioSampler(
            tuple(rollout.object_initial_pos), xy_jitter=args.jitter if args.randomize else 0.0,
            friction_range=tuple(args.friction) if args.friction else None,
            mass_range=tuple(args.mass_range) if args.mass_range else None,
        )
        sampled = sampler.sample(args.seed, args.controller, args.mass)
        trial = PresentationTrial(
            args.object, args.seed, args.controller, args.duration, sampled.object_pos,
            rollout.base_object_mass if sampled.object_mass is None else sampled.object_mass,
            float(rollout.base_object_friction[0]) if sampled.object_friction is None else sampled.object_friction,
        )
    values = (*trial.object_pos, trial.duration_s, trial.object_mass, trial.object_friction)
    if len(trial.object_pos) != 3 or not all(np.isfinite(value) for value in values):
        raise ValueError("trial inputs must contain a finite XYZ pose and finite physical values")
    if min(trial.duration_s, trial.object_mass, trial.object_friction) <= 0:
        raise ValueError("trial duration, mass and friction must be positive")
    if int(trial.duration_s / rollout.model.opt.timestep) < 1:
        raise ValueError("trial duration must cover at least one physics step")
    return trial


def create_run_directory(kind: str, requested: Path | None = None) -> Path:
    path = requested or PROJECT_ROOT / "results" / kind / uuid.uuid4().hex
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def presentation_manifest(trial: PresentationTrial, renderer_backend: str, run_label: str) -> dict:
    from .experiments import content_hash, machine_identity, software_versions, source_identity

    inputs = asdict(trial)
    configuration = {key: value for key, value in inputs.items() if not key.startswith("source_")}
    configuration_hash = content_hash(configuration)
    versions = software_versions()
    return {
        "run_id": uuid.uuid4().hex, "run_label": run_label, **machine_identity(), "config_hash": configuration_hash,
        "trial_id": content_hash({"config_hash": configuration_hash, "controller": trial.controller}),
        "source_sha256": source_identity()["sha256"], "versions": versions,
        "python_version": versions["python"], "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__, "physics_backend": "classic_mujoco_cpu",
        "renderer_backend": renderer_backend,
        "renderer_attribution": "See separate desktop GL_VENDOR/GL_RENDERER report; no attribution inferred here.",
        "task_scope": "grasp/lift/hold from authored hand-aligned pose", "trial": inputs,
        "configuration": configuration,
    }


def stage_at(sim_time: float) -> str:
    if sim_time <= 0.35:
        return "open"
    if sim_time < 1.30:
        return "grasp_settle"
    if sim_time < 2.40:
        return "lift"
    return "hold"


def configure_visuals(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.8, 0.8, 0.8)
    model.vis.headlight.specular[:] = (0.2, 0.2, 0.2)
    goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    if goal_site >= 0:
        model.site_rgba[goal_site, 3] = 0.0


def configure_camera(camera: mujoco.MjvCamera, model: mujoco.MjModel,
                     data: mujoco.MjData, detail: bool = False) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 132.0
    camera.elevation = -18.0
    if detail:
        camera.azimuth = 90.0
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_C_MC")
        target, hand = data.xpos[target_id], data.xpos[hand_id]
        camera.lookat[:] = (target + hand) / 2
        camera.distance = max(0.42, float(np.linalg.norm(target - hand)) * 1.9 + 0.18)
        camera.elevation = -12.0
    else:
        camera.lookat[:] = (0.45, 0.0, 0.62)
        camera.distance = 1.65


class ReplayRequested(Exception):
    """Interrupt the trace without taking another physics step."""


class ViewerClosed(Exception):
    """Interrupt the trace when the native window closes."""


class PlaybackControls:
    """The native key callback only queues events; the physics thread applies them."""

    def __init__(self, now: float | None = None) -> None:
        self.events: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
        self.paused = False
        self.contacts = True
        self.forces = False
        self.detail = False
        self.replay = False
        self.session_start = time.perf_counter() if now is None else now
        self.wall_anchor = self.session_start
        self.pause_start: float | None = None
        self.log: list[dict] = []

    def dispatch_key(self, key: int, source: str = "keyboard") -> None:
        if key in KEY_ACTIONS:
            self.events.put((key, source))

    def apply_pending(self, sim_time: float, now: float) -> list[str]:
        actions = []
        while not self.events.empty():
            key, source = self.events.get()
            action = KEY_ACTIONS[key]
            paused_duration = 0.0
            if action == "pause":
                self.paused = not self.paused
                if self.paused:
                    self.pause_start = now
                else:
                    paused_duration = now - (self.pause_start if self.pause_start is not None else now)
                    self.wall_anchor += paused_duration
                    self.pause_start = None
            elif action == "replay":
                self.replay = True
            elif action == "contacts":
                self.contacts = not self.contacts
            elif action == "forces":
                self.forces = not self.forces
            elif action == "camera":
                self.detail = not self.detail
            self.log.append({
                "action": action, "key": key, "source": source, "wall_s": now - self.session_start,
                "sim_time_before": sim_time, "sim_time_after": sim_time,
                "paused": self.paused, "paused_wall_s": paused_duration,
                "contact_points": self.contacts, "force_vectors": self.forces,
            })
            actions.append(action)
        return actions

    def begin_replay(self, sim_before: float, now: float) -> None:
        self.replay = False
        self.paused = False
        self.pause_start = None
        self.wall_anchor = now
        self.log.append({"action": "trial_reset", "wall_s": now - self.session_start,
                         "sim_time_before": sim_before, "sim_time_after": 0.0})

    def deadline(self, sim_time: float) -> float:
        return self.wall_anchor + sim_time


def load_validation_events(path: Path | None) -> list[tuple[float, int]]:
    if path is None:
        return []
    events = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        at = float(row["wall_s"])
        key = row["key"]
        code = KEY_NAMES.get(key.upper()) if isinstance(key, str) else int(key)
        if code not in KEY_ACTIONS or not np.isfinite(at) or at < 0:
            raise ValueError("validation events require nonnegative wall_s and one of SPACE/P/R/C/F/V")
        events.append((at, code))
    return sorted(events, key=lambda event: event[0])
