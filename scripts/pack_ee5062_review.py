#!/usr/bin/env python3
"""Build a local, inspectable EE5062 review package without submitting it.

Run --inventory-only while experiments are in progress. Building requires
--source-frozen and a new output directory. This tool never marks a draft ready
for submission, infers task success, or contacts Canvas.

--compact retains review summaries, full source/assets, report and introduction;
the SHA256 inventory identifies every omitted local raw artifact explicitly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

from package_project import collect_files, file_record, package


PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "docs" / "EE5062_REPORT_DRAFT.md"
BRIEF = Path.home() / "Downloads" / "EE5110_EE5062_Segment C - CA.pdf"
EXCLUDED = {"__pycache__", ".git", ".venv", ".ompl_deps", "node_modules"}
ALLOWED = {".json", ".gz", ".csv", ".png", ".npy", ".jpg", ".jpeg", ".mp4",
           ".md", ".pdf", ".xml", ".txt", ".log", ".sha256", ".zip", ".py"}
PENDING = [
    "Student contribution/AI disclosure review and final deliverable review",
]
LIMITATIONS = [
    "Current random public-YCB baseline is 3/18; independent new-pose profile is 6/18 but <=12 N native channel reference passes 0/18",
    "Known upright tools and authored receiving fixtures; broader clutter and geometry uncertainty are not validated",
    "Rolling is table-supported; arbitrary initial poses and in-air manipulation are not demonstrated",
    "Known-cylinder RGB-D priors; noise/occlusion checked in offline replay only, not corrupted-frame robot tasks or unseen tools",
    "Automated GUI/replay is validated; a person-produced task video and outcome are not claimed",
    "Simulation-only contacts; hardware calibration and sim-to-real transfer are not validated",
]


def historical_catalogue() -> Path | None:
    candidates = sorted(PROJECT.parents[1].glob("01_*/EE5062_*/**/catalogue_18_final"))
    return candidates[0] if len(candidates) == 1 else None


def default_evidence() -> list[tuple[str, Path]]:
    paths = [
        ("physical_ycb_protocol", "results/v6_r2_object_challenge_20261013/protocol.json"),
        ("physical_ycb_summary", "results/v6_r2_object_challenge_20261013/summary.json"),
        ("current_random_public_ycb", "results/public_ycb_current_random18_20261014"),
        ("independent_public_ycb_random", "results/public_ycb_improved_random18_20261014"),
        ("public_ycb_duration_thumb_development", "results/public_ycb_isolated_development_seed0_20261014"),
        ("public_ycb_vertical_development", "results/public_ycb_box_vertical_development_seed0_20261014"),
        ("public_ycb_upper_side_development", "results/public_ycb_box_rim_development_seed0_20261014"),
        ("public_ycb_retreat_development", "results/public_ycb_box_rim_retreat_development_seed0_20261014"),
        ("public_ycb_timing_receiving_development", "results/public_ycb_timing_receiving_development_20261014"),
        ("public_ycb_registration_captures", "results/public_ycb_registration_initial_development_20261014"),
        ("public_ycb_registered_controller_development", "results/public_ycb_registered_controller_development_20261014"),
        ("public_ycb_tilt_registered_controller_development", "results/public_ycb_tilt_registered_controller_development_20261014"),
        ("public_ycb_small_tilt_offline_replay", "results/public_ycb_registration_small_tilt_replay_20261014"),
        ("public_ycb_nominal_aperture_envelope", "results/public_ycb_nominal_aperture_envelope_20261014"),
        ("public_ycb_contact_guard_negative_development", "results/public_ycb_contact_envelope_guard_development_20261014"),
        ("numerical_physics", "results/physics_validation_v6_20261013_final02/report.json"),
        ("marker_robustness", "results/v7_marker_robustness_final2_20261014/summary.json"),
        ("no_retention_ablation", "results/v7_marker_no_retention_ablation_20261014/summary.json"),
        ("fixture_development", "scratch/v7_upright/fixture_robust_summary.json"),
        ("fixture_force_relief", "results/v7_fixture_force_relief_20261014/summary.json"),
        ("supported_rolling_records", "scratch/v7_regrasp_sourcefix_lower6mm/regrasp_records.json"),
        ("supported_rolling_trace", "scratch/v7_regrasp_sourcefix_lower6mm/index_push.trace.json.gz"),
        ("supported_rolling_paired", "results/v7_supported_rolling_final_20261014"),
        ("supported_rolling_video", "results/v7_supported_rolling_video_20261014"),
        ("rgbd_probes", "results/v7_rgbd_probe_20261014/summary.json"),
        ("lab4090_paired_rgbd_feedback", "results/lab4090_v7_rgbd_paired500_529_20261014"),
        ("lab4090_paired_rgbd_precise", "results/lab4090_v7_rgbd_paired_precise600_629_20261014"),
        ("three_tool_contact_feedback", "results/v7_three_tool_feedback_paired40_44_20261014"),
        ("introduced_demonstration", "results/v7_introduced_demo_20261014"),
        ("introduced_demonstration_r2", "results/v7_introduced_demo_r2_20261014"),
        ("introduced_demonstration_r3", "results/v7_introduced_demo_r3_20261014"),
        ("development_result_figures", "results/v7_development_figures_r3_20261014"),
        ("precise_result_figures", "results/v7_precise_figures_20261014"),
        ("final_measured_result_figures", "results/v7_final_figures_r4_20261014"),
        ("offline_rgbd_sensor_stress", "results/v7_sensor_stress_r2_20261014"),
        ("fullscene_wrench_physics_sensitivity", "results/v7_fullscene_physics_sensitivity700_704_20261014"),
        ("teleop_scripted_input_check", "results/v7_teleop_headless_eventcheck_20261014_macos/metadata.json"),
        ("automated_gui_events_video", "results/v7_automated_gui_final_20261014"),
        ("automated_gui_replay", "results/v7_automated_gui_final_replay_20261014"),
        ("continuous_sorting_video", "results/v7_continuous_sorting_video_seed0_20261014"),
        ("continuous_current_motion_relief_video", "results/v7_continuous_sorting_current_motion_relief_video_seed0_20261014"),
        ("continuous_force_crossvalidation", "results/v7_continuous_force_crossval90_94_20261014"),
        ("continuous_force_holdout", "results/v7_continuous_force_holdout1000_1029_20261014"),
        ("local_obstacle_gate", "results/v7_obstacle_gate_20261014/obstacle_gate_result.json"),
        ("carried_obstacle_acceptance", "results/v7_carried_obstacle_acceptance_20261014"),
        ("lab4090_r2b_receipt", "results/lab4090_v7_r2b_deploy_20261014/deployment_receipt.json"),
        ("lab4090_main_source_r3_validation", "results/lab4090_v7_final_source_verify_r3_20261014"),
        ("final_current_source_regression", "results/v7_final_regression_20261014"),
        ("dell_native_desktop_task_validation", "results/dell_v7_final_20261014_r3c"),
        ("dell_initial_verification_failure", "results/dell_v7_initial_verify_failure_20261014"),
        ("marker_raw_demo_metadata", "results/v7_marker_live_video_seed100_20261014/video/metadata.json"),
        ("independent_report_review", "results/ee5062_report_review_20261014/verification_receipt.json"),
    ]
    paths.extend((f"physical_empty_hand_obstacle_seed{seed}",
                  f"results/v7_obstacle_execution_final_seed{seed}_20261014")
                 for seed in range(3))
    paths.extend((f"physical_carried_obstacle_seed{seed}",
                  f"results/v7_carried_obstacle_final_seed{seed}_20261014")
                 for seed in range(3))
    paths.extend((f"continuous_sorting_seed{seed}",
                  f"results/v7_continuous_sorting_seed{seed}_20261014")
                 for seed in range(30, 35))
    paths.extend((f"strict_continuous_sorting_seed{seed}",
                  f"results/v7_continuous_sorting_precise_seed{seed}_20261014")
                 for seed in range(80, 85))
    result = [(label, PROJECT / relative) for label, relative in paths]
    catalogue = historical_catalogue()
    if catalogue is not None:
        result.extend((f"historical_randomized_ycb_{name}", catalogue / f"{name}.json")
                      for name in ("protocol", "summary", "rows"))
    else:
        result.append(("historical_randomized_ycb_pending", PROJECT / "missing_catalogue_snapshot"))
    return result


def parse_evidence(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("evidence must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not re.fullmatch(r"[a-zA-Z0-9_]+", label) or not path:
        raise argparse.ArgumentTypeError("label must contain only letters, numbers and underscore")
    return label, Path(path).expanduser().absolute()


def regular_files(path: Path) -> list[Path]:
    if path.is_symlink():
        raise ValueError(f"Symlinks are not allowed in evidence: {path}")
    if path.is_file():
        if path.suffix.lower() not in ALLOWED:
            raise ValueError(f"Unsupported evidence file type: {path}")
        return [path]
    if not path.is_dir():
        return []
    result = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if any(part.startswith(".") or part in EXCLUDED for part in relative.parts):
            continue
        if item.is_symlink():
            raise ValueError(f"Symlinks are not allowed in evidence: {item}")
        if item.is_file() and item.suffix.lower() in ALLOWED:
            result.append(item)
    return result


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def copy_checked(source: Path, destination: Path) -> dict:
    expected = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    shutil.copy2(source, destination)
    if digest(source) != expected or digest(destination) != expected:
        raise ValueError(f"Evidence changed while copying: {source}")
    return {"source": str(source), "bytes": destination.stat().st_size,
            "sha256": expected}


def compact_evidence_file(path: Path) -> bool:
    """Select structured review records, not native trajectories or media."""
    if path.suffix.lower() not in {".json", ".csv", ".md", ".txt", ".log", ".sha256"}:
        return False
    name = path.name.lower()
    summary = any(token in name for token in ("protocol", "summary", "verification", "receipt",
        "metrics", "confidence", "acceptance", "metadata", "result", "manifest", "diagnosis"))
    raw = any(token in name for token in ("trace", "trajectory", "native.json", "contacts.json", "native_contacts", "contact_events",
        "force_samples", "joint_samples", "motor_samples", "guard.json"))
    if raw and not summary:
        return False
    if path.stat().st_size > 4 * 1024 * 1024:
        return False
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return isinstance(value, dict) or summary or path.stem.lower() == "rows"
    return True


def inventory(evidence: list[tuple[str, Path]], compact: bool = False) -> dict:
    labels = [label for label, _ in evidence]
    if len(labels) != len(set(labels)):
        raise ValueError("Evidence labels must be unique")
    entries = []
    for label, source in evidence:
        files = regular_files(source)
        selected = [path for path in files if not compact or compact_evidence_file(path)]
        entries.append({"label": label, "source": str(source),
                        "status": "present" if files else "pending_missing",
                        "files": len(files),
                        "bytes": sum(path.stat().st_size for path in files),
                        "selected_review_files": len(selected),
                        "selected_review_bytes": sum(path.stat().st_size for path in selected),
                        "omitted_local_raw_files": len(files) - len(selected),
                        "scope": "named artifact; existence does not imply task success"})
    return {"format": "ee5062-local-review-inventory", "status": "review_draft",
            "review_scope": "compact_summary_review" if compact else "selected_full_evidence_review",
            "submission_ready": False, "canvas_submission": "not_attempted",
            "report": str(REPORT), "brief": str(BRIEF), "evidence": entries,
            "pending_acceptance": list(PENDING), "scope_limitations": list(LIMITATIONS)}


def evidence_snapshot(evidence: list[tuple[str, Path]]) -> list[dict]:
    return [{"label": label, "source": str(item), "bytes": item.stat().st_size, "sha256": digest(item)}
            for label, source in evidence for item in regular_files(source)]


def checked_report_assets(assets: list[Path] | tuple[Path, ...]) -> list[Path]:
    result = []
    root = PROJECT.resolve()
    for value in assets:
        path = Path(value).absolute() if Path(value).is_absolute() else PROJECT / value
        if not path.is_relative_to(root):
            raise ValueError(f"Report asset must be inside PROJECT: {path}")
        for parent in (path, *path.parents):
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f"Symlinks are not allowed in report assets: {path}")
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Report asset must be a regular file inside PROJECT: {path}")
        result.append(path)
    if len(result) != len(set(result)):
        raise ValueError("Report assets must be unique")
    return result


def make_pack(output: Path, evidence: list[tuple[str, Path]], intro_video: Path | None,
              report_pdf: Path | None = None, compact: bool = False, *,
              report_assets: list[Path] | tuple[Path, ...] = (),
              original_path_assets: list[Path] | tuple[Path, ...] = (),
              status_overrides: dict | None = None, brief: Path | None = None,
              include_historical: bool = True, validate_copied_inputs=None) -> dict:
    output = output.absolute()
    brief = BRIEF if brief is None else Path(brief)
    report_assets = checked_report_assets(report_assets)
    original_path_assets = checked_report_assets(original_path_assets)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    if not REPORT.is_file():
        raise FileNotFoundError(f"Report is missing: {REPORT}")
    # The source ZIP rejects concurrent changes; the second snapshot also
    # catches edits to an earlier file after it has already been compressed.
    before = [file_record(path, PROJECT) for path in collect_files(PROJECT)]
    evidence_before = evidence_snapshot(evidence)
    output.mkdir(parents=True, exist_ok=False)
    content = output / "review"
    content.mkdir()
    copied = []
    copied.append(copy_checked(REPORT, content / "report" / REPORT.name))
    if report_pdf is not None:
        if not report_pdf.is_file() or report_pdf.suffix.lower() != ".pdf":
            raise ValueError("--report-pdf must be an existing PDF")
        with report_pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("--report-pdf does not contain a PDF header")
        copied.append(copy_checked(report_pdf, content / "report" / "EE5062_REPORT_DRAFT.pdf"))
    reference_docs = sorted(PROJECT.glob("V7*.md")) + [PROJECT / "PHYSICS_PARAMETERS.md"]
    for document in reference_docs:
        if document.is_file():
            copied.append(copy_checked(document, content / "report" / "reference_docs" / document.name))
    for name in (("measured_outcomes.png", "supported_rolling.png") if include_historical else ()):
        figure = PROJECT / "results" / "v7_final_figures_r4_20261014" / name
        if figure.is_file():
            copied.append(copy_checked(figure, content / figure.relative_to(PROJECT)))
    for asset in report_assets:
        relative = asset.relative_to(PROJECT)
        destination = (content / "report" / asset.name if relative.parts[0] == "docs"
                       else content / relative)
        record = copy_checked(asset, destination)
        record["pack_path"] = destination.relative_to(content).as_posix()
        copied.append(record)
    for asset in original_path_assets:
        destination = content / asset.relative_to(PROJECT)
        record = copy_checked(asset, destination)
        record["pack_path"] = destination.relative_to(content).as_posix()
        copied.append(record)
    if brief.is_file():
        copied.append(copy_checked(brief, content / "course_brief" / brief.name))
    source_release = package(PROJECT, content / "source" / "source_and_assets.zip")
    catalogue = historical_catalogue() if include_historical else None
    historical_source = catalogue.parents[1] / "source" / "project" if catalogue is not None else None
    historical_release = None
    if historical_source is not None and historical_source.is_dir():
        historical_release = package(historical_source, content / "source" / "historical_ycb_snapshot.zip")
    after = [file_record(path, PROJECT) for path in collect_files(PROJECT)]
    if before != after:
        raise ValueError("Source changed during packaging; this partial review directory is not a release")
    status = inventory(evidence, compact=compact)
    if status_overrides:
        status.update(status_overrides)
    status["brief"] = str(brief)
    if not brief.is_file():
        status["pending_acceptance"].append("Authoritative local course PDF unavailable at pack time")
    status["generated_utc"] = datetime.now(timezone.utc).isoformat()
    status["source_freeze_asserted_by_operator"] = True
    status["source_release"] = source_release
    status["historical_ycb_source_release"] = historical_release
    status["report_and_reference_files"] = copied
    status["report_pdf"] = "supplied for review" if report_pdf is not None else "pending rendered report"
    artifacts = []
    selected_sources = set()
    for label, source in evidence:
        for item in regular_files(source):
            if compact and not compact_evidence_file(item):
                continue
            relative = item.relative_to(source) if source.is_dir() else Path(item.name)
            target = content / "evidence" / label / relative
            record = copy_checked(item, target)
            record["pack_path"] = target.relative_to(content).as_posix()
            artifacts.append(record)
            selected_sources.add((label, str(item)))
    raw_video = PROJECT / "results" / "v7_marker_live_video_seed100_20261014" / "video" / "tool_marker.mp4"
    if include_historical and raw_video.is_file() and not compact:
        copy_checked(raw_video, content / "demonstration" / "RAW_PHYSICAL_MARKER_DEMO.mp4")
    if intro_video is not None:
        if not intro_video.is_file() or intro_video.suffix.lower() != ".mp4":
            raise ValueError("--intro-video must be an existing MP4")
        record = copy_checked(intro_video, content / "demonstration" / "INTRO_VIDEO_FOR_REVIEW.mp4")
        record["pack_path"] = "demonstration/INTRO_VIDEO_FOR_REVIEW.mp4"
        copied.append(record)
        status["intro_video"] = "supplied for review; introduction and content not automatically verified"
    else:
        status["intro_video"] = "pending; raw demo is not a completed introducing video"
    status["artifact_files"] = artifacts
    if compact:
        elsewhere = {record["source"]: record["pack_path"] for record in copied if "pack_path" in record}
        raw_inventory = [{**record, "included_in_evidence_folder":
            (record["label"], record["source"]) in selected_sources,
            "included_elsewhere_in_review": elsewhere.get(record["source"]),
            "included_anywhere_in_review": (record["label"], record["source"]) in selected_sources or record["source"] in elsewhere,
            "omission_reason": None if (record["label"], record["source"]) in selected_sources
                else ("Retained at the separately recorded review path" if record["source"] in elsewhere
                      else "Intentionally omitted from compact summary review; raw remains local")}
            for record in evidence_before]
        (content / "RAW_EVIDENCE_INVENTORY.json").write_text(json.dumps({
            "review_scope": "compact_summary_review", "full_raw_embedded": False,
            "local_raw_files": len(raw_inventory),
            "local_raw_bytes": sum(record["bytes"] for record in raw_inventory),
            "included_evidence_files": len(artifacts),
            "omitted_local_raw_files": sum(not record["included_in_evidence_folder"] for record in raw_inventory),
            "files": raw_inventory,
            "scope": "SHA256 and size of available local evidence; inventory records do not embed omitted raw data"
        }, indent=2) + "\n", encoding="utf-8")
        status["raw_evidence_inventory"] = "RAW_EVIDENCE_INVENTORY.json"
        status["full_raw_embedded"] = False
    if validate_copied_inputs is not None:
        validate_copied_inputs(content, status)
    (content / "REVIEW_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    historical_demo_note = (
        "RAW_PHYSICAL_MARKER_DEMO.mp4 retains the earlier physical marker demonstration.\n"
        if include_historical else "")
    instructions = f"""# EE5062 Local Review Package

This is a review draft. It has not been submitted to Canvas, and it does not
declare all requirements complete. Read report/EE5062_REPORT_DRAFT.md and
REVIEW_STATUS.json before interpreting the evidence.

source/source_and_assets.zip contains the current source and assets. Extract
it into a new checkout, create a virtual environment, install requirements.txt,
and use the README/V7_RUNBOOK commands. On headless Linux set MUJOCO_GL=egl.
OMPL is optional in the base project but required for the obstacle planning
gate. A task exit code 2 is a measured failure; keep that result directory.

Evidence files preserve their named source snapshots. If available,
source/historical_ycb_snapshot.zip contains the separate archived YCB controller.
Historical YCB results must not be attributed to the current controller. Summary-only entries do not
include every raw trace; provide raw evidence directories with --evidence
LABEL=PATH when preparing a complete final evidence package.

INTRO_VIDEO_FOR_REVIEW.mp4, when present, contains the introducing demonstration.
{historical_demo_note}Human teleoperation and student contribution claims require separate review.

PACK_MANIFEST.json records the SHA256 and size of every packaged file except
the manifest itself. The outer ZIP has a separate SHA256 sidecar.
"""
    (content / "README_REVIEW.md").write_text(instructions, encoding="utf-8")
    if status.get("review_notes"):
        with (content / "README_REVIEW.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + status["review_notes"] + "\n")
    if compact:
        compact_note = """
## Compact Evidence Scope

This is a compact summary review. It contains the complete current source and
assets archive, report, supplied introduction, and selected structured result
records. Native trajectories, repeated media, camera arrays, and frozen raw-run
archives are intentionally omitted; they remain in the local evidence paths.
RAW_EVIDENCE_INVENTORY.json lists the SHA256, size, local path and inclusion flag
for every available evidence artifact. An omitted artifact is not claimed to
be inside this ZIP. PACK_MANIFEST.json lists only files actually packaged.
Run without --compact to build the larger selected full-evidence review.
"""
        with (content / "README_REVIEW.md").open("a", encoding="utf-8") as handle:
            handle.write(compact_note)
    records = [{"path": path.relative_to(content).as_posix(), "bytes": path.stat().st_size,
                "sha256": digest(path)} for path in sorted(content.rglob("*")) if path.is_file()]
    if after != [file_record(path, PROJECT) for path in collect_files(PROJECT)]:
        raise ValueError("Source changed while collecting evidence; this partial directory is not a release")
    if evidence_before != evidence_snapshot(evidence):
        raise ValueError("Evidence changed during packaging; wait for complete retrieval and frozen experiments")
    if any(digest(Path(record["source"])) != record["sha256"] for record in copied):
        raise ValueError("Report or reference input changed during packaging; this partial directory is not a release")
    (content / "PACK_MANIFEST.json").write_text(json.dumps({"files": records,
        "status": "review_draft", "submission_ready": False,
        "review_scope": "compact_summary_review" if compact else "selected_full_evidence_review",
        "full_selected_evidence_embedded": not compact}, indent=2) + "\n", encoding="utf-8")
    archive = output / "EE5062_LOCAL_REVIEW.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
        for path in sorted(content.rglob("*")):
            if path.is_file():
                writer.write(path, path.relative_to(content).as_posix())
    with zipfile.ZipFile(archive) as reader:
        if reader.testzip() is not None:
            raise ValueError("Review ZIP failed CRC validation")
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    return {"status": "review_draft", "submission_ready": False,
            "review_scope": "compact_summary_review" if compact else "selected_full_evidence_review",
            "canvas_submission": "not_attempted", "output": str(output),
            "zip": str(archive), "zip_sha256": checksum,
            "packed_files": len(records) + 1,
            "pending_evidence": [item["label"] for item in status["evidence"]
                                 if item["status"] != "present"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new local review directory")
    parser.add_argument("--source-frozen", action="store_true", help="operator asserts experiment source is frozen")
    parser.add_argument("--inventory-only", action="store_true", help="print artifact inventory without writing files")
    parser.add_argument("--compact", action="store_true", help="summary review with explicit omitted-raw SHA256 inventory; full source and supplied intro retained")
    parser.add_argument("--evidence", action="append", type=parse_evidence, default=[], metavar="LABEL=PATH")
    parser.add_argument("--intro-video", type=Path, help="optional MP4 to include for review")
    parser.add_argument("--report-pdf", type=Path, help="optional rendered report PDF to include for review")
    args = parser.parse_args()
    evidence = default_evidence() + args.evidence
    try:
        if args.inventory_only:
            report = inventory(evidence, compact=args.compact)
        else:
            if args.output is None or not args.source_frozen:
                parser.error("build requires --output and --source-frozen; use --inventory-only during experiments")
            report = make_pack(args.output, evidence, args.intro_video, args.report_pdf, compact=args.compact)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Review package failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
