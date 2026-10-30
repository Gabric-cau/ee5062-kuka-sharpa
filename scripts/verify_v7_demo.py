#!/usr/bin/env python3
"""Audit the introduced video and its recorded source clips, not task success."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    output = directory / "video_verification.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    video = directory / "EE5062_DEMONSTRATION.mp4"
    if digest(video) != manifest["video_sha256"]:
        raise ValueError("Introduced video differs from its manifest")
    for clip in manifest["clips"]:
        if digest(Path(clip["source"])) != clip["source_sha256"]:
            raise ValueError("A recorded source clip changed")
    first_clip = manifest["clips"][0]
    if first_clip["output_start_frame"] != 3 * manifest["fps"]:
        raise ValueError("Expected a three-second introduction")
    previous_end = first_clip["output_start_frame"]
    captures = {0, manifest["frames"] - 1}
    for clip in manifest["clips"]:
        start, end = clip["output_start_frame"], clip["output_end_frame"]
        if start != previous_end or end <= start:
            raise ValueError("Clip timeline has gaps or overlaps")
        captures.update((start, (start + end) // 2, end - 1))
        previous_end = end
    if previous_end != manifest["frames"]:
        raise ValueError("Clip timeline does not cover the complete video")
    frames, minimum_std, saved = 0, float("inf"), {}
    with imageio.get_reader(video) as reader:
        fps = float(reader.get_meta_data()["fps"])
        for index, frame in enumerate(reader):
            if frame.shape != (720, 1280, 3):
                raise ValueError("Unexpected frame dimensions")
            minimum_std = min(minimum_std, float(frame.std()))
            if index in captures:
                saved[index] = frame
                imageio.imwrite(directory / f"verified_frame_{index:04d}.png", frame)
            frames += 1
    if frames != manifest["frames"] or fps != manifest["fps"] or minimum_std < 5:
        raise ValueError("Encoded video is incomplete or blank")
    movement = []
    for clip in manifest["clips"]:
        first = saved[clip["output_start_frame"]].astype(float)
        last = saved[clip["output_end_frame"] - 1].astype(float)
        difference = float(np.abs(last - first).mean())
        if difference < .1:
            raise ValueError("A recorded clip shows no visible movement")
        movement.append({"title": clip["title"], "mean_first_last_pixel_difference": difference})
    result = {"accepted": True, "decoded_frames": frames, "fps": fps,
              "duration_s": frames / fps, "minimum_frame_pixel_std": minimum_std,
              "source_clips_sha256_verified": True, "clip_motion": movement,
              "video_sha256": digest(video), "manifest_sha256": digest(directory / "manifest.json"),
              "scope": "Video integrity and visible motion only; physical outcomes use separate native traces.",
              "human_operator": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    print(json.dumps({key: result[key] for key in ("accepted", "decoded_frames", "duration_s")}))


if __name__ == "__main__":
    main()
