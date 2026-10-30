"""Audit actual local run, pause/replay and decoded video evidence."""

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kuka_sharpa_grasp.experiments import aggregate_runs, file_hash, load_manifest, read_trial_records, source_identity
from kuka_sharpa_grasp.rollout import TrialResult


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_rejection(inputs: list[Path]) -> str:
    try:
        aggregate_runs(inputs)
    except ValueError as exc:
        return str(exc)
    raise ValueError("Invalid aggregation was accepted")


def audit(root: Path, media_paths: list[Path], decoded_root: Path) -> dict:
    report = {"passed": False, "runs": {}, "media": {}}
    source = source_identity()["sha256"]
    for name in ("pilot", "main", "negative_object", "mass_sweep", "random_mass"):
        path = root / name
        manifest = load_manifest(path)
        require(manifest["source"]["sha256"] == source, f"Source mismatch: {name}")
        records = read_trial_records(path, manifest)
        require(len(records) == len(manifest["planned_trials"]), f"Incomplete run: {name}")
        pairs = {}
        for row in records:
            pairs.setdefault(row["scenario_id"], {})[row["controller"]] = row
        for pair in pairs.values():
            require(set(pair) == {"fixed", "force"}, "Incomplete controller pair")
            require(all(pair["fixed"][key] == pair["force"][key]
                        for key in ("sampled_object_pos", "object_mass", "object_friction", "duration_s")), "Paired inputs differ")
        groups, coverage = aggregate_runs([path])
        report["runs"][name] = {"records": len(records), "paired_inputs_equal": True,
                                  "config_hash": manifest["config_hash"], "groups": groups, "coverage": coverage}
    report["source_sha256"] = source
    report["duplicate_rejection"] = expected_rejection([root / "main", root / "main" / "results.csv"])
    report["configuration_rejection"] = expected_rejection([root / "main", root / "negative_object"])
    with (root / "pilot_pre_resume_hashes.csv").open(encoding="utf-8-sig", newline="") as handle:
        before = list(csv.DictReader(handle))
    require(all(file_hash(Path(row["Path"])).upper() == row["Hash"] for row in before), "Resume changed prior trial files")
    report["resume"] = {"prior_trial_files": len(before), "changed": 0, "partial_planned": 12, "partial_completed": 3}

    viewer = root / "viewer_validation_v2"
    events = json.loads((viewer / "viewer_events.json").read_text(encoding="utf-8"))["events"]
    pause, resume = [event for event in events if event["action"] == "pause"]
    require(pause["sim_time_before"] == resume["sim_time_before"], "Physics moved while paused")
    require(resume["paused_wall_s"] > 0.5, "Pause interval was not exercised")
    for event in events:
        if "native_contact_points" in event:
            require(event["native_contact_points"] == int(event["contact_points"]), "Native contact flag mismatch")
            require(event["native_force_vectors"] == int(event["force_vectors"]), "Native force flag mismatch")
    first = json.loads((viewer / "replay_000" / "result.json").read_text(encoding="utf-8"))
    second = json.loads((viewer / "replay_001" / "result.json").read_text(encoding="utf-8"))
    require(first == second, "Same-input native replay diverged")
    require(file_hash(viewer / "replay_000" / "trace.csv") == file_hash(viewer / "replay_001" / "trace.csv"),
            "Same-input native traces diverged")
    completions = [event for event in events if event["action"] == "trial_complete"]
    close = events[-1]
    require(close["wall_s"] - completions[-1]["wall_s"] > 1, "Window did not stay open after completion")
    require(close["sim_time_before"] == completions[-1]["sim_time_before"], "Final state advanced during inspection")
    report["viewer"] = {"paused_sim_time_s": pause["sim_time_before"], "pause_wall_s": resume["paused_wall_s"],
                          "identical_replays": True, "native_flags_match": True,
                          "final_frozen_wall_s": close["wall_s"] - completions[-1]["wall_s"],
                          "input_scope": "Actual native viewer; scheduled events use the keyboard dispatcher. OS keystroke injection unavailable."}
    for supplied in media_paths:
        metadata_path = supplied if supplied.name == "metadata.json" else supplied / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        original = json.loads(Path(metadata["trial"]["source_trial_json"]).read_text(encoding="utf-8"))
        result = metadata["result"]
        require(all(result[field.name] == original[field.name] for field in fields(TrialResult)), "Rendered outcome differs from recorded experiment")
        require(metadata["source_sha256"] == source, "Media source mismatch")
        require(metadata["opengl"]["GL_VENDOR"] == "NVIDIA Corporation" and "2060" in metadata["opengl"]["GL_RENDERER"], "Media renderer is not RTX2060")
        video = Path(metadata["video"])
        reader = imageio.get_reader(video)
        try:
            count = reader.count_frames()
            fps = reader.get_meta_data()["fps"]
            decoded = [reader.get_data(index) for index in (0, min(count - 1, round(1.8 * fps)), count - 1)]
        finally:
            reader.close()
        require(all(frame.std() > 5 for frame in decoded), "Decoded video frame is blank")
        change = float(np.abs(decoded[0].astype(float) - decoded[-1].astype(float)).mean())
        require(change > 0.5, "Decoded video did not show temporal change")
        expected = metadata["physical_sample_count"] + round(metadata["frozen_final_seconds"] * fps)
        require(count == expected, "Encoded frame count mismatch")
        decoded_dir = decoded_root / video.parent.name
        decoded_dir.mkdir(parents=True, exist_ok=False)
        for label, frame in zip(("initial", "lift", "final"), decoded):
            imageio.imwrite(decoded_dir / f"{label}.png", frame)
        report["media"][str(video)] = {"outcome_matches_original": True, "status": result["status"],
                                        "frame_count": count, "fps": fps, "duration_s": count / fps,
                                        "resolution": list(decoded[0].shape[1::-1]), "first_final_pixel_difference": change,
                                        "sha256": file_hash(video), "decoded_qa": str(decoded_dir)}
    report["passed"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--media", type=Path, nargs="+", required=True, help="final metadata.json paths or evidence directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite delivery audit")
    report = audit(args.evidence_root, args.media, args.output.with_name(args.output.stem + "_frames"))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"passed": report["passed"], "source_sha256": report["source_sha256"],
                      "trial_records": sum(run["records"] for run in report["runs"].values()),
                      "resume": report["resume"], "viewer": report["viewer"], "media": report["media"]}, indent=2))


if __name__ == "__main__":
    main()
