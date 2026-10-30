#!/usr/bin/env python3
"""Introduce and join recorded simulation clips without creating task outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate(frame, title, note):
    image = Image.fromarray(frame).convert("RGB")
    if image.size != (1280, 720):
        image = image.resize((1280, 720), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=15)
    draw.rectangle((0, 0, 1280, 32), fill=(23, 29, 35))
    draw.text((18, 3), title, font=font, fill=(240, 243, 244))
    draw.rectangle((0, 690, 1280, 720), fill=(23, 29, 35))
    draw.text((18, 697), note, font=small, fill=(211, 219, 223))
    return np.asarray(image)


def make(output, sorting, *, rolling=None, obstacle=None, gui=None, fps=20):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    clips = [(sorting, "KUKA + Sharpa | continuous three-tool sorting",
              "MuJoCo simulation | six free objects | no resets between tools | 12 N channel force limit")]
    if rolling:
        clips.append((rolling, "Supported dexterous manipulation | individual index joints",
                      "Table-supported marker rolling | physical contact evidence | not in-air dexterity"))
    if obstacle:
        clips.append((obstacle, "Physical carried-tool motion | OMPL planned route and docking",
                      "Actually grasped screwdriver | actuator-driven transfer | native collision and contact audit"))
    if gui:
        clips.append((gui, "Teleoperation interface | GUI input, logging and replay",
                      "Automated GUI acceptance input | replay matched physical states | no human operator claim"))
    manifest, frame_count = [], 0
    with imageio.get_writer(str(output / "EE5062_DEMONSTRATION.mp4"), fps=fps,
                            codec="libx264", macro_block_size=None) as writer:
        for index, (path, title, note) in enumerate(clips):
            reader = imageio.get_reader(path)
            metadata = reader.get_meta_data()
            source_fps = float(metadata["fps"])
            first = reader.get_data(0)
            if index == 0:
                intro = annotate(first, "EE5062 | KUKA + Sharpa tactile-aware workcell",
                                 "Simulation-only project | tool sorting, supported rolling and teleoperation evidence")
                imageio.imwrite(output / "intro.png", intro)
                for _ in range(3 * fps):
                    writer.append_data(intro)
                    frame_count += 1
            source_frames, next_output_time = 0, 0.
            start_frame = frame_count
            for source_index, frame in enumerate(reader):
                source_frames += 1
                timestamp = source_index / source_fps
                while next_output_time <= timestamp + 1e-9:
                    writer.append_data(annotate(frame, title, note))
                    frame_count += 1
                    next_output_time += 1. / fps
            reader.close()
            manifest.append({"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_frames": source_frames, "source_fps": source_fps,
                "output_start_frame": start_frame, "output_end_frame": frame_count,
                "title": title, "scope": note})
    result = {"format": "ee5062_introduced_demonstration_r1", "fps": fps,
              "frames": frame_count, "duration_s": frame_count / fps, "clips": manifest,
              "outcome_policy": "outcomes come from independent physical traces; annotations do not infer success",
              "video_sha256": hashlib.sha256((output / "EE5062_DEMONSTRATION.mp4").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sorting", type=Path, required=True)
    parser.add_argument("--rolling", type=Path)
    parser.add_argument("--obstacle", type=Path)
    parser.add_argument("--gui", type=Path)
    args = parser.parse_args()
    result = make(args.output, args.sorting, rolling=args.rolling, obstacle=args.obstacle, gui=args.gui)
    print(json.dumps({"output": str(args.output), "frames": result["frames"],
                      "duration_s": result["duration_s"]}), flush=True)


if __name__ == "__main__":
    main()
