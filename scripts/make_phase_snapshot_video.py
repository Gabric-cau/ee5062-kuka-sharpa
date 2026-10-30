#!/usr/bin/env python3
"""Create a review video from recorded MuJoCo phase snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.)
    args = parser.parse_args()
    frames = sorted(args.frame_directory.glob("*.png"))
    if not frames:
        raise ValueError("No phase PNG files found")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8,
            macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    payload = {"format": "mujoco_phase_snapshot_video_r1", "fps": args.fps,
        "frames": [{"path": frame.name, "sha256": digest(frame)} for frame in frames],
        "video_sha256": digest(args.output),
        "scope": "Rendered phase snapshots from a completed MuJoCo run; not a continuous trajectory recording, human teleoperation, or hardware evidence."}
    args.metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frames": len(frames), "output": str(args.output),
        "video_sha256": payload["video_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
