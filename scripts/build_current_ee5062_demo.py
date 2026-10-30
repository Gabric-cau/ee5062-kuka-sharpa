#!/usr/bin/env python3
"""Assemble an introduced review video from existing, complete simulation clips."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
FPS, WIDTH, SOURCE_HEIGHT, BAND = 20, 1280, 720, 128
HEIGHT = SOURCE_HEIGHT + BAND
SAMPLE_RATE = 24000
FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def font(size):
    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("Install a readable TrueType font or set a FONT_PATHS entry")


def fitted(draw, text, size, width):
    while size > 12:
        face = font(size)
        if draw.textbbox((0, 0), text, font=face)[2] <= width:
            return face
        size -= 1
    raise ValueError("Caption cannot fit at a legible size")


def wrap(draw, text, face, width):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=face)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def card(segment):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (21, 27, 32))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((64, 124, 72, 668), fill=(66, 175, 148))
    draw.text((106, 136), "EE5062  |  REVIEW COPY  |  22 OCT 2026", font=font(19), fill=(151, 172, 184))
    title_face = font(42)
    y = 205
    for line in wrap(draw, segment["title"], title_face, 1060):
        draw.text((104, y), line, font=title_face, fill=(241, 245, 246))
        y += 56
    y += 30
    for paragraph in segment["body"]:
        for line in wrap(draw, paragraph, font(29), 1060):
            draw.text((106, y), line, font=font(29), fill=(202, 215, 220))
            y += 42
        y += 16
    if y > HEIGHT - 112:
        raise ValueError(f"Title card overflow: {segment['id']}")
    draw.text((106, HEIGHT - 74), "Autonomous simulation footage. Synthetic narration. Human teleoperation pending.",
              font=font(19), fill=(218, 184, 100))
    return np.asarray(canvas)


def captioned(frame, segment):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (20, 27, 32))
    canvas.paste(Image.fromarray(frame), (0, BAND))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 12), segment["title"], font=fitted(draw, segment["title"], 28, WIDTH - 44), fill=(240, 245, 246))
    draw.text((22, 53), segment["subtitle"], font=fitted(draw, segment["subtitle"], 22, WIDTH - 44), fill=(202, 220, 226))
    draw.text((22, 93), "1x playback | complete source clip | autonomous, not human input | review, not final submission",
              font=font(17), fill=(218, 184, 100))
    return np.asarray(canvas)


def decoded_qa(path, *, expected_size, frame_hash_output=None, segments=()):
    count, minimum_std, moving_pairs, maximum_difference = 0, math.inf, 0, 0.
    previous, sequence_digest, frame_digests = None, hashlib.sha256(), []
    reader = imageio.get_reader(path)
    metadata = reader.get_meta_data()
    handle = gzip.open(frame_hash_output, "wt", encoding="utf-8") if frame_hash_output else None
    segment_motion = {segment["id"]: {"moving_pairs": 0, "maximum_content_mean_difference": 0.}
                      for segment in segments if segment["kind"] == "source"}
    try:
        for frame in reader:
            if frame.shape != (expected_size[1], expected_size[0], 3):
                raise ValueError(f"Unexpected frame dimensions: {path}")
            std = float(frame.std())
            if std < 5:
                raise ValueError(f"Blank frame {count}: {path}")
            minimum_std = min(minimum_std, std)
            frame_digest = hashlib.sha256(frame.tobytes()).hexdigest()
            sequence_digest.update(bytes.fromhex(frame_digest))
            frame_digests.append(frame_digest)
            thumbnail = frame[::8, ::8].astype(np.float32)
            difference = float(np.abs(thumbnail - previous).mean()) if previous is not None else 0.
            moving_pairs += int(difference > .02)
            maximum_difference = max(maximum_difference, difference)
            if handle:
                record = {"frame": count, "time_s": count / metadata["fps"],
                          "decoded_rgb_sha256": frame_digest, "pixel_std": std,
                          "mean_thumbnail_difference_from_previous": difference}
                handle.write(json.dumps(record) + "\n")
            for segment in segments:
                if segment["kind"] == "source" and segment["output_start_frame"] < count < segment["output_end_frame_exclusive"]:
                    # Ignore the added caption band when assessing original scene movement.
                    content = frame[BAND::8, ::8].astype(np.float32)
                    if previous_content is not None:
                        change = float(np.abs(content - previous_content).mean())
                        motion = segment_motion[segment["id"]]
                        motion["moving_pairs"] += int(change > .02)
                        motion["maximum_content_mean_difference"] = max(motion["maximum_content_mean_difference"], change)
            previous_content = frame[BAND::8, ::8].astype(np.float32) if expected_size[1] == HEIGHT else None
            previous = thumbnail
            count += 1
    finally:
        reader.close()
        if handle:
            handle.close()
    if count < 2 or moving_pairs == 0:
        raise ValueError(f"No decoded video movement: {path}")
    if any(item["moving_pairs"] == 0 for item in segment_motion.values()):
        raise ValueError("A source segment has no scene movement")
    return {"decoded_frames": count, "fps": metadata["fps"], "duration_s": count / metadata["fps"],
        "dimensions": list(expected_size), "minimum_frame_pixel_std": minimum_std,
        "moving_adjacent_frame_pairs": moving_pairs, "maximum_thumbnail_mean_difference": maximum_difference,
        "unique_decoded_frame_hashes": len(set(frame_digests)),
        "ordered_decoded_frame_hashes_sha256": sequence_digest.hexdigest(), "source_segment_motion": segment_motion}


def synthesize(segment, output, ffmpeg, enabled):
    speech = {"synthetic": True, "engine": "macOS say", "voice": "Samantha", "rate_words_per_minute": 172,
              "transcript": segment["narration"], "status": "text_only"}
    if not enabled or not shutil.which("say"):
        return speech, None
    text_path, aiff_path, wav_path = (output / f"{segment['id']}{suffix}" for suffix in (".txt", ".aiff", ".wav"))
    text_path.write_text(segment["narration"] + "\n", encoding="utf-8")
    try:
        subprocess.run(["say", "-v", "Samantha", "-r", "172", "-f", str(text_path), "-o", str(aiff_path)], check=True, capture_output=True)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(aiff_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le", str(wav_path)], check=True, capture_output=True)
        with wave.open(str(wav_path), "rb") as source:
            audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()
        speech.update(status="generated", duration_s=len(audio) / SAMPLE_RATE,
                      artifacts={path.name: digest(path) for path in (text_path, aiff_path, wav_path)})
        return speech, audio
    except subprocess.CalledProcessError as error:
        speech["failure"] = error.stderr.decode(errors="replace")[:500]
        return speech, None


def build(project, output, *, narration=True):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing review directory: {output}")
    paths = {
        "sorting": "results/v7_tool_sorting_current_20261021_seed0/tool_sorting.mp4",
        "index_push": "results/v7_supported_rolling_current_20261022/seed_0/index_push/supported_rolling.mp4",
        "control": "results/v7_supported_rolling_current_20261022/seed_0/no_override_control/supported_rolling.mp4",
        "carried": "results/v7_carried_obstacle_current_20261022_seed0/video/tool_screwdriver.mp4",
    }
    evidence_paths = {
        "sorting": "results/v7_tool_sorting_current_20261021_seed0/summary.json",
        "index_push": "results/v7_supported_rolling_current_20261022/seed_0/index_push/result.json",
        "control": "results/v7_supported_rolling_current_20261022/seed_0/no_override_control/result.json",
        "carried": "results/v7_carried_obstacle_current_20261022_seed0/result.json",
        "public_ycb_r14": "results/public_ycb_constrained_preload_20261021_r14/ycb_can_constrained_preload_r14_seed3414/result.json",
    }
    evidence = {key: json.loads((project / path).read_text()) for key, path in evidence_paths.items()}
    if not evidence["sorting"]["success"] or evidence["sorting"]["completed_tools"] != 3 or not evidence["carried"]["success"]:
        raise ValueError("The selected review wording requires successful sorting/carried evidence")
    if not evidence["index_push"]["success"] or evidence["control"]["success"] or evidence["public_ycb_r14"]["success"]:
        raise ValueError("The selected control/negative-result wording no longer matches evidence")
    push_angle = evidence["index_push"]["regrasp_evidence"]["relative_rotation_deg"]
    control_angle = evidence["control"]["regrasp_evidence"]["relative_rotation_deg"]
    sources = {}
    for key, relative in paths.items():
        path = project / relative
        qa = decoded_qa(path, expected_size=(WIDTH, SOURCE_HEIGHT))
        if qa["fps"] != FPS:
            raise ValueError("Source fps differs; this assembly preserves every frame at 1x")
        sources[key] = {"path": relative, "sha256": digest(path), "full_decode_qa": qa}
    output.mkdir(parents=True)
    audio_dir = output / "synthetic_narration"
    audio_dir.mkdir()
    segments = [
        {"id": "intro", "kind": "title", "title": "MuJoCo robotic tool handling",
         "body": ["KUKA arm + Sharpa hand: tool sorting, supported dexterous motion and obstacle-aware transfer.",
                  "Current autonomous simulation evidence. This is a review copy, not a final submission."],
         "narration": "This E E five zero six two project uses a Kuka arm and a Sharpa hand in MuJoCo. We review three autonomous simulation experiments. The narration is synthetic. These are not human teleoperation recordings."},
        {"id": "sorting_intro", "kind": "title", "title": "1. Continuous three-tool sorting",
         "body": ["Marker, screwdriver and wrench: one complete continuous recording at normal speed.",
                  "Known upright fixtures and declared task geometry. This is not arbitrary-pose bin picking."],
         "narration": "First, three tools are sorted in one continuous recording. The tools start in known upright fixtures, using declared geometry and tactile contact gates. This is not arbitrary pose bin picking."},
        {"id": "sorting_clip", "kind": "source", "source": "sorting", "title": "1. Continuous three-tool sorting | seed 0 | 3 tools completed",
         "subtitle": "Known upright fixtures and declared task geometry | native multi-view and tactile channels preserved"},
        {"id": "rolling_intro", "kind": "title", "title": "2. Table-supported dexterous rolling",
         "body": [f"Index-joint push: {push_angle:.2f} deg relative object rotation. No-override control: {control_angle:.2f} deg.",
                  "Same seed; fixed wrist command; continuous table support. No in-air regrasp claim."],
         "narration": f"Next, an index joint push produces {push_angle:.2f} degrees of relative object rotation, compared with {control_angle:.2f} degrees without the override. The wrist command is fixed and the table supports the object. This is not an in air regrasp."},
        {"id": "rolling_push_clip", "kind": "source", "source": "index_push", "title": f"2A. Index-joint push | relative rotation {push_angle:.2f} deg | seed 0",
         "subtitle": "Table-supported manipulation | wrist goal fixed | all source frames at normal speed"},
        {"id": "rolling_control_clip", "kind": "source", "source": "control", "title": f"2B. No-override control | relative rotation {control_angle:.2f} deg | seed 0",
         "subtitle": "Same preparation and support conditions | insufficient relative object motion under the evaluator"},
        {"id": "carried_intro", "kind": "title", "title": "3. Carry a tool around an obstacle",
         "body": ["Physical pick, OMPL transfer, docking and release. The straight joint-space path is blocked.",
                  "One seed and a declared map with a held-tool proxy. No learned perception or broad generalization claim."],
         "narration": "Finally, the hand physically picks a screwdriver and carries it around an obstacle using O M P L, then docks and releases it. This single seed experiment uses a declared map and a held tool collision proxy."},
        {"id": "carried_clip", "kind": "source", "source": "carried", "title": "3. Physical pick + carried-tool OMPL transfer + dock + release | seed 0",
         "subtitle": "Declared map and held-tool proxy | blocked direct path | single-seed evidence, not generalization"},
        {"id": "limitations", "kind": "title", "title": "What remains incomplete",
         "body": ["Public-YCB R14 still fails: no eligible trial lift and no successful held grasp.",
                  "Human teleoperation evidence is pending. No hardware, human force feedback or 85% grasp-success claim."],
         "narration": "The limits matter. Public Y C B version R fourteen still fails to qualify for a trial lift. Human teleoperation evidence is pending. These demonstrations do not establish hardware force feedback, general grasping success, or an eighty five percent success rate."},
    ]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_tracks, frame_index = [], 0
    for segment in segments:
        segment["output_start_frame"] = frame_index
        segment["output_start_s"] = frame_index / FPS
        if segment["kind"] == "title":
            speech, audio = synthesize(segment, audio_dir, ffmpeg, narration)
            segment["synthetic_narration"] = speech
            duration = max(7., len(audio) / SAMPLE_RATE + 1.) if audio is not None else 11.
            segment["output_frames"] = math.ceil(duration * FPS)
            if audio is not None:
                audio_tracks.append((frame_index / FPS, audio))
        else:
            source = sources[segment["source"]]
            segment.update(source_path=source["path"], source_sha256=source["sha256"], playback_speed=1.,
                           source_start_frame=0, source_end_frame_exclusive=source["full_decode_qa"]["decoded_frames"],
                           source_start_s=0., source_end_s=source["full_decode_qa"]["duration_s"],
                           output_frames=source["full_decode_qa"]["decoded_frames"], source_frames_dropped=0,
                           source_crop_or_resize=False)
        frame_index += segment["output_frames"]
        segment.update(output_end_frame_exclusive=frame_index, output_end_s=frame_index / FPS)
    silent_path = output / "review_video_silent.mp4"
    with imageio.get_writer(silent_path, fps=FPS, codec="libx264", quality=8, macro_block_size=None) as writer:
        for segment in segments:
            if segment["kind"] == "title":
                pixels = card(segment)
                for _ in range(segment["output_frames"]):
                    writer.append_data(pixels)
            else:
                with imageio.get_reader(project / segment["source_path"]) as reader:
                    count = 0
                    for frame in reader:
                        writer.append_data(captioned(frame, segment))
                        count += 1
                if count != segment["output_frames"]:
                    raise ValueError("A source changed between inspection and assembly")
    mixed = np.zeros(round(frame_index / FPS * SAMPLE_RATE), dtype="<i2")
    for start_s, audio in audio_tracks:
        start = round(start_s * SAMPLE_RATE)
        mixed[start:start + len(audio)] = audio
    audio_path = output / "synthetic_narration_timeline.wav"
    with wave.open(str(audio_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(mixed.tobytes())
    final_path = output / "EE5062_CURRENT_REVIEW_DEMO.mp4"
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(silent_path), "-i", str(audio_path),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(final_path)],
                   check=True, capture_output=True)
    frame_receipt = output / "decoded_frame_hashes.jsonl.gz"
    qa = decoded_qa(final_path, expected_size=(WIDTH, HEIGHT), frame_hash_output=frame_receipt, segments=segments)
    if qa["decoded_frames"] != frame_index:
        raise ValueError("Final video frame count differs from the edit manifest")
    contact_sheet = output / "contact_sheet.png"
    samples = [("Intro", segments[0]["output_start_s"] + 1),
               ("Sorting: early", segments[2]["output_start_s"] + 8),
               ("Sorting: middle", segments[2]["output_start_s"] + 40),
               ("Sorting: late", segments[2]["output_end_s"] - 2),
               ("Rolling comparison", segments[3]["output_start_s"] + 1),
               ("Index push", segments[4]["output_end_s"] - .3),
               ("No-override control", segments[5]["output_end_s"] - .3),
               ("Carried transfer introduction", segments[6]["output_start_s"] + 1),
               ("Carried: lift", segments[7]["output_start_s"] + 12),
               ("Carried: transfer", segments[7]["output_start_s"] + 27),
               ("Carried: release", segments[7]["output_end_s"] - 2),
               ("Limits and pending evidence", segments[8]["output_start_s"] + 1)]
    tile_width, tile_height = 426, 312
    sheet = Image.new("RGB", (tile_width * 3, tile_height * 4), (245, 246, 247))
    draw = ImageDraw.Draw(sheet)
    with imageio.get_reader(final_path) as reader:
        for index, (label, timestamp) in enumerate(samples):
            x, y = index % 3 * tile_width, index // 3 * tile_height
            image = Image.fromarray(reader.get_data(round(timestamp * FPS))).resize((tile_width, 282), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y + 30))
            draw.text((x + 7, y + 5), f"{timestamp:.2f}s | {label}", font=font(15), fill=(25, 31, 36))
    sheet.save(contact_sheet)
    qa.update(video_sha256=digest(final_path), per_frame_receipt=frame_receipt.name,
              per_frame_receipt_sha256=digest(frame_receipt), contact_sheet=contact_sheet.name,
              contact_sheet_sha256=digest(contact_sheet),
              audio={"synthetic": True, "sample_rate": SAMPLE_RATE, "duration_s": len(mixed) / SAMPLE_RATE,
                     "nonzero_samples": int(np.count_nonzero(mixed)), "pcm_rms": float(np.sqrt(np.mean(mixed.astype(float) ** 2)))})
    qa_path = output / "full_decode_qa_receipt.json"
    write_json(qa_path, qa)
    for source in sources.values():
        if digest(project / source["path"]) != source["sha256"]:
            raise ValueError("A source artifact changed while the review was built")
    manifest = {"format": "ee5062_introduced_review_video_r1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_only": True, "final_submission": False, "human_operator": False, "human_teleoperation_pending": True,
        "simulation_only": True, "source_video_content": "Previously recorded autonomous MuJoCo runs; no GUI input generated by this build.",
        "edit_policy": "Every source video frame appears exactly once at 1x, in original order. Only title cards and a 128px caption band above the unchanged 1280x720 source are added. Source videos remain immutable.",
        "output": {"path": final_path.name, "sha256": digest(final_path), "fps": FPS, "frames": frame_index,
                   "duration_s": frame_index / FPS, "width": WIDTH, "height": HEIGHT},
        "sources": sources, "segments": segments,
        "evidence_json": {key: {"path": value, "sha256": digest(project / value)} for key, value in evidence_paths.items()},
        "metrics": {"sorting_completed_tools": 3, "rolling_index_push_relative_rotation_deg": push_angle,
                    "rolling_control_relative_rotation_deg": control_angle, "carried_seed": 0,
                    "public_ycb_r14_success": False},
        "synthetic_narration": {"enabled": bool(audio_tracks), "not_user_voice": True, "engine": "macOS say / Samantha",
                                "timeline_wav": audio_path.name, "timeline_wav_sha256": digest(audio_path)},
        "qa_receipt": qa_path.name, "qa_receipt_sha256": digest(qa_path), "contact_sheet_samples": samples,
        "script_sha256": digest(Path(__file__)), "ffmpeg": ffmpeg}
    write_json(output / "video_manifest.json", manifest)
    print(json.dumps({"output": str(final_path), "frames": frame_index, "duration_s": frame_index / FPS,
                      "synthetic_narration": bool(audio_tracks), "sha256": digest(final_path)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/v7_introduced_current_20261022")
    parser.add_argument("--no-narration", action="store_true")
    args = parser.parse_args()
    build(args.project_root, args.output_dir, narration=not args.no_narration)


if __name__ == "__main__":
    main()
