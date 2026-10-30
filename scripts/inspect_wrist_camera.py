"""Save RGB-D and visible-object overlay in an isolated, new output directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.scene import build_scene, YCB_OBJECTS
from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.camera import CameraObserver
from kuka_sharpa_grasp.perception import estimate_object


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=YCB_OBJECTS, default="ycb_can")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT / "results" / "v2_vision" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(build_scene(output / "scene.xml", args.object, overhead_camera=True)))
    data = mujoco.MjData(model)
    GraspRollout(model, data).reset()
    with CameraObserver(model) as camera:
        obs = camera.poll(data)
        import OpenGL.GL as gl
        renderer = {key: gl.glGetString(value).decode() for key, value in
                    (("vendor", gl.GL_VENDOR), ("renderer", gl.GL_RENDERER), ("version", gl.GL_VERSION))}
    belief = estimate_object(obs, args.object)
    Image.fromarray(obs.rgb).save(output / "wrist_rgb.png")
    np.save(output / "depth_metres.npy", obs.depth)
    valid = np.isfinite(obs.depth)
    lo, hi = np.percentile(obs.depth[valid], [2, 98])
    gray = np.zeros(obs.depth.shape, dtype=np.uint8)
    gray[valid] = (255 * (1 - np.clip((obs.depth[valid] - lo) / max(hi-lo, 1e-6), 0, 1))).astype(np.uint8)
    Image.fromarray(gray).save(output / "depth.png")
    overlay = obs.rgb.copy()
    overlay[belief.pixel_mask] = (.45 * overlay[belief.pixel_mask] + .55 * np.array([0, 255, 70])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 31), fill="black")
    draw.text((5, 3), f"{args.object}: {len(belief.points)} points, confidence {belief.confidence:.2f}", fill="white")
    draw.text((5, 17), f"t={belief.timestamp:.3f}s; cylinder yaw unobservable" if not belief.yaw_observable else f"t={belief.timestamp:.3f}s; visible surface PCA", fill="white")
    if belief.centroid is not None:
        y, x = np.where(belief.pixel_mask)
        cx, cy = int(x.mean()), int(y.mean())
        draw.line((cx-7, cy, cx+7, cy), fill="white", width=2)
        draw.line((cx, cy-7, cx, cy+7), fill="white", width=2)
    image.save(output / "object_overlay.png")
    report = {"object": args.object, "physics_backend": "classic_mujoco_cpu",
              "rendering": renderer, "mujoco": mujoco.__version__, "python": sys.version,
              "timestamp": obs.timestamp, "depth_units": "metres, optical-axis distance; invalid NaN",
              "rgb_std": float(obs.rgb.std()), "depth_std": float(np.nanstd(obs.depth)),
              "valid_depth_pixels": int(valid.sum()), "connected_points": len(belief.points),
              "centroid": None if belief.centroid is None else belief.centroid.tolist(),
              "principal_axis": None if belief.principal_axis is None else belief.principal_axis.tolist(),
              "yaw_observable": belief.yaw_observable, "confidence": belief.confidence}
    (output / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if obs.rgb.std() < 5 or np.nanstd(obs.depth) < .01 or belief.centroid is None:
        raise SystemExit("nonblank/perception inspection failed")


if __name__ == "__main__":
    main()
