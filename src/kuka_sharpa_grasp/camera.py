"""Rendered RGB-D observations; no object state or segmentation buffer access."""
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    name: str = "wrist_rgbd"
    width: int = 320
    height: int = 240
    depth_noise_std: float = 0.0
    dropout: float = 0.0
    delay: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resolution must be positive")
        if not np.isfinite([self.depth_noise_std, self.dropout, self.delay]).all():
            raise ValueError("noise, dropout and delay must be finite")
        if self.depth_noise_std < 0 or self.delay < 0 or not 0 <= self.dropout <= 1:
            raise ValueError("invalid noise, dropout or delay")


@dataclass(frozen=True)
class CameraObservation:
    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    available_at: float
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    focal_length: float
    camera_name: str


def save_observation(observation: CameraObservation, directory: Path, *, prefix="wrist"):
    """Export RGB and lossless metric depth with an explicit invalid convention."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(observation.depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    # PNG stores millimetres; NPY retains the original metre values and NaNs.
    millimetres = np.zeros(depth.shape, dtype=np.uint16)
    millimetres[valid] = np.clip(np.rint(depth[valid] * 1000), 1, 65535).astype(np.uint16)
    rgb_path = directory / f"{prefix}_rgb.png"
    depth_path = directory / f"{prefix}_depth.png"
    raw_path = directory / f"{prefix}_depth_m.npy"
    mask_path = directory / f"{prefix}_depth_valid.png"
    Image.fromarray(observation.rgb).save(rgb_path)
    Image.fromarray(millimetres).save(depth_path)
    Image.fromarray(valid.astype(np.uint8) * 255).save(mask_path)
    np.save(raw_path, depth, allow_pickle=False)
    metadata = {
        "camera_name": observation.camera_name, "timestamp_s": observation.timestamp,
        "shape": list(depth.shape), "png_unit": "millimetres", "png_invalid_value": 0,
        "npy_unit": "metres", "npy_invalid_value": "NaN", "finite_fraction": float(valid.mean()),
        "focal_length_px": observation.focal_length,
        "camera_position_m": observation.camera_position.tolist(),
        "camera_rotation_world": observation.camera_rotation.tolist(),
        "files": {"rgb": rgb_path.name, "depth_png": depth_path.name,
                  "depth_metres": raw_path.name, "valid_mask": mask_path.name},
    }
    for path in (rgb_path, depth_path, raw_path, mask_path):
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty observation export: {path}")
    (directory / f"{prefix}_camera.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class CameraObserver:
    """Depth is optical-axis distance in metres; invalid/background is NaN.

    poll() returns None during delay warmup and on frames not yet due. Reset
    clears queued frames and restarts the seeded noise sequence after sim reset.
    """
    def __init__(self, model, config=CameraConfig()):
        self.config = config
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.name)
        if self.camera_id < 0:
            raise ValueError(f"camera missing: {config.name}")
        self.renderer = mujoco.Renderer(model, height=config.height, width=config.width)
        self.far = float(model.vis.map.zfar * model.stat.extent)
        self.focal = config.height / (2 * np.tan(np.deg2rad(model.cam_fovy[self.camera_id]) / 2))
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.config.seed)
        self.queue = deque()
        self.last_time = -np.inf

    def poll(self, data):
        now = float(data.time)
        if now < self.last_time:
            raise ValueError("simulation time moved backwards; reset observer")
        self.last_time = now
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.config.name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        depth[(depth >= self.far * .999) | (depth <= 0) | ~np.isfinite(depth)] = np.nan
        depth += self.rng.normal(0, self.config.depth_noise_std, depth.shape)
        depth[(self.rng.random(depth.shape) < self.config.dropout) | (depth <= 0)] = np.nan
        self.queue.append(CameraObservation(
            rgb, depth, now, now + self.config.delay,
            data.cam_xpos[self.camera_id].copy(),
            data.cam_xmat[self.camera_id].reshape(3, 3).copy(), self.focal, self.config.name))
        delivered = None
        while self.queue and self.queue[0].available_at <= now + 1e-12:
            delivered = self.queue.popleft()
        return delivered

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
