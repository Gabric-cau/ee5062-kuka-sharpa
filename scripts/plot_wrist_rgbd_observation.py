#!/usr/bin/env python3
"""Inspect a saved wrist observation with a metric depth color scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def binding(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Saved wrist RGB-D observation")
    args = parser.parse_args()
    source = args.observation.resolve(strict=True)
    paths = {name: source / name for name in
        ("wrist_rgb.png", "wrist_depth_m.npy", "wrist_depth_valid.png", "wrist_camera.json")}
    before = {name: binding(path) for name, path in paths.items()}
    rgb = np.asarray(Image.open(paths["wrist_rgb.png"]).convert("RGB"))
    depth = np.load(paths["wrist_depth_m.npy"], allow_pickle=False)
    mask = np.asarray(Image.open(paths["wrist_depth_valid.png"]).convert("L")) > 0
    camera = json.loads(paths["wrist_camera.json"].read_text())
    if depth.ndim != 2 or depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError("Saved RGB, metric depth and valid mask dimensions differ")
    valid = mask & np.isfinite(depth) & (depth > 0.)
    if not valid.any():
        raise ValueError("No finite positive valid depth samples")
    low, high = np.percentile(depth[valid], [2., 98.]).tolist()
    if not high > low:
        high = low + 1e-6
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
    figure.suptitle(args.label, fontsize=14)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    palette = plt.get_cmap("viridis").copy()
    palette.set_bad("#eeeeee")
    artist = axes[1].imshow(np.ma.array(depth, mask=~valid), cmap=palette, vmin=low, vmax=high)
    axes[1].set_title("Depth (2nd-98th percentile display range)", fontsize=10)
    figure.colorbar(artist, ax=axes[1], label="Camera-axis depth (m)", shrink=.76, extend="both")
    for axis in axes:
        axis.set_axis_off()
    outputs = []
    for suffix in ("png", "pdf"):
        destination = args.output / ("wrist_rgbd." + suffix)
        figure.savefig(destination, dpi=180)
        outputs.append(binding(destination))
    plt.close(figure)
    after = {name: binding(path) for name, path in paths.items()}
    if before != after:
        raise ValueError("Saved observation inputs changed during plotting")
    receipt = {"format": "saved_wrist_rgbd_observation_preview_v1", "source": binding(Path(__file__)),
        "inputs": before, "inputs_unchanged": True, "camera_metadata": camera,
        "image_size_wh": [rgb.shape[1], rgb.shape[0]], "valid_depth_pixels": int(valid.sum()),
        "valid_depth_range_m": [float(depth[valid].min()), float(depth[valid].max())],
        "display_depth_range_m": [low, high], "outputs": outputs,
        "scope": "Saved simulated-camera observation only. Percentile clipping changes display colors, not metric depth or controller input. A frame does not establish probe admission, lift, table unloading, task success or a human-operated recording."}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "valid_depth_pixels": int(valid.sum()),
        "display_depth_range_m": [low, high]}))


if __name__ == "__main__":
    main()
