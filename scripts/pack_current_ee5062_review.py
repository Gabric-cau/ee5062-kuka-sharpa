#!/usr/bin/env python3
"""Inventory or package the explicit 21-22 October evidence; never submit it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pack_ee5062_review as review
from package_project import EXCLUDED_NAMES, EXCLUDED_SUFFIXES


PROJECT = Path(__file__).resolve().parents[1]
CURRENT_PATHS = (
    ("current_continuous_sorting", "results/v7_tool_sorting_current_20261021_seed0"),
    ("current_three_tool_tactile", "results/v7_three_tool_feedback_current_20261021_40_44"),
    ("current_randomized_rgbd", "results/v7_rgbd_randomized_current_20261022_1600_1629"),
    ("current_supported_rolling", "results/v7_supported_rolling_current_20261022"),
    ("current_carried_dependency_failure", "results/v7_carried_obstacle_current_20261021_seed0"),
    ("current_carried_success", "results/v7_carried_obstacle_current_20261022_seed0"),
    ("current_public_ycb_r14_negative", "results/public_ycb_constrained_preload_20261021_r14"),
    ("current_public_ycb_r15_nominal_static", "results/public_ycb_geometry_20261022_r15_final"),
    ("current_public_ycb_r15_dynamic_negative", "results/public_ycb_fixed_geometry_20261022_r15_seed3414"),
    ("current_independent_v7_audit", "results/current_evidence_audit_20261022"),
    ("current_independent_rgbd_audit", "results/current_rgbd_evidence_audit_20261022"),
    ("current_compiled_physics", "results/compiled_physics_20261021"),
    ("current_dell_operator_deployment", "results/dell_operator_deployment_20261022/deployment_receipt.json"),
    *((f"current_introduced_video_{label}", f"results/v7_introduced_current_20261022/{name}")
      for label, name in (("manifest", "video_manifest.json"), ("decode_qa", "full_decode_qa_receipt.json"),
                          ("audio_qa", "encoded_audio_qa_receipt.json"), ("frame_hashes", "decoded_frame_hashes.jsonl.gz"),
                          ("contact_sheet", "contact_sheet.png"), ("visual_review", "VISUAL_REVIEW.md"))),
)
REPORT_ASSETS = (
    "PHYSICS_AND_CONTACT_CONTROL.md", "V7_RUNBOOK.md",
    "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md", "docs/V7_OPERATOR_RECORDING_AUDIT.md",
    "docs/R15_NOMINAL_GEOMETRY_DIAGNOSIS_20261022.md",
    "results/v7_carried_obstacle_current_20261022_seed0/video/tool_screwdriver_peak_tactile.png",
)
R15_PROVENANCE_ASSETS = (
    "results/public_ycb_geometry_20261022_r15_final/ranked.json",
    "results/public_ycb_geometry_20261022_r15_final/fixed_orientation_candidate_profile.json",
)
ORIGINAL_PATH_ASSETS = (
    "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md", "ACTIVE_DEVELOPMENT.md",
    "results/compiled_physics_20261021/ycb_can.json",
    "results/compiled_physics_20261021/ycb_cracker_box.json",
    "results/compiled_physics_20261021/ycb_potted_meat_can.json",
    "results/public_ycb_constrained_preload_20261021_r14/native_command_bounds.json",
)
REPORT_PDF = "docs/EE5062_REPORT_DRAFT.pdf"
INTRO_VIDEO = "results/v7_introduced_current_20261022/EE5062_CURRENT_REVIEW_DEMO.mp4"
LIMITATIONS = [
    "Current public-YCB R14 and fixed-geometry R15 seed3414 both fail task/held-lift; static R15 feasibility is not measured grasp success. No current randomized public-dataset batch meets the requested performance.",
    "Known upright fixture sorting completes three tools in one seed. Three-tool tactile comparisons contain 30 episodes: shared 15/15, control 10/15; this is not arbitrary-pose bin picking.",
    "Known upright-marker RGB-D cohort has 90 episodes: map+tactile 25/30, RGB-D+tactile 30/30, RGB-D without tactile 23/30. The independent audit is trace-derived, without a full raw per-contact stream or runtime object-write counter.",
    "Supported rolling has five paired seeds, index push 5/5 versus control 0/5; continuous table support is required, with no in-air regrasp claim.",
    "Carried-tool OMPL success is one development seed on a declared map and authored corridor; the earlier missing-OMPL failure remains included.",
    "Introduced video is autonomous simulation with explicitly synthetic narration. Dell installation/dependency/shortcut checks are not a new GUI or rendering test. Human teleoperation remains pending.",
    "CPU MuJoCo physics and simulated contacts are not hardware force feedback, sensor calibration, neural training or sim-to-real validation.",
]
PENDING = ["Current randomized public-3D-dataset grasping baseline with complete raw evidence",
           "Actual operator-produced simulation recording and honest contribution/AI acknowledgement review",
           "Student review of report, video and final submission contents"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_file(path, *, project_only=True):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Symlinks are not allowed in current review inputs: {path}")
    resolved = path.resolve(strict=True)
    if project_only and not resolved.is_relative_to(PROJECT.resolve()):
        raise ValueError(f"Current evidence must be inside PROJECT: {path}")
    if not resolved.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    if resolved.name in EXCLUDED_NAMES or resolved.suffix.lower() in EXCLUDED_SUFFIXES:
        raise ValueError(f"Credential-like file is not review evidence: {path}")
    return resolved


def current_evidence():
    return [(label, PROJECT / relative) for label, relative in CURRENT_PATHS]


def check_completeness(evidence):
    """Check saved denominators/receipts, without repeating native-log inflation."""
    checks = []
    for label, path in evidence:
        missing, issues, required = [], [], []
        if path.is_symlink():
            raise ValueError(f"Symlink evidence root: {path}")
        files = review.regular_files(path)
        for item in files:
            safe_file(item)
        if path.is_dir():
            summary = read_json(path / "summary.json") if (path / "summary.json").is_file() else None
            if summary is not None and "complete" in summary and summary["complete"] is not True:
                issues.append("summary.complete is not true")
            if label in {"current_three_tool_tactile", "current_randomized_rgbd"}:
                required += ["summary.json", "protocol.json"]
                if (path / "protocol.json").is_file() and summary is not None:
                    protocol = read_json(path / "protocol.json")
                    if protocol.get("source_changed_during_run") is not False:
                        issues.append("source_changed_during_run is not false")
                    expected = len(protocol["seeds"]) * len(protocol["modes"]) * len(protocol.get("tools", [None]))
                    if len(summary.get("results", [])) != expected:
                        issues.append("summary episode count differs from declared denominator")
                    for seed in protocol["seeds"]:
                        for tool in protocol.get("tools", [None]):
                            for mode in protocol["modes"]:
                                prefix = Path(f"seed_{seed}") / tool / mode if tool else Path(f"seed_{seed}") / mode
                                required += [(prefix / name).as_posix() for name in
                                             ("metadata.json", "task_result.json", "trace.json.gz", "events.json.gz")]
                                required.append((prefix / "scene.xml").as_posix() if tool else f"seed_{seed}/scene.xml")
            elif label == "current_supported_rolling":
                required += ["summary.json"]
                if summary:
                    if len(summary.get("episodes", [])) != 2 * summary.get("seeds", -1):
                        issues.append("rolling paired denominator is incomplete")
                    for row in summary.get("episodes", []):
                        prefix = Path(f"seed_{row['seed']}") / row["arm"]
                        required += [(prefix / name).as_posix() for name in
                                     ("result.json", "scene.xml", "physical_trace.jsonl.gz", "native_contacts.jsonl.gz", "supported_rolling.mp4")]
            elif label in {"current_public_ycb_r14_negative", "current_public_ycb_r15_dynamic_negative"}:
                required += ["summary.json", "protocol.json", "verification_receipt.json", "strict_native_verification.json", "native_command_bounds.json", "frozen_source.zip"]
                if summary:
                    if not summary.get("rows") or summary.get("runner_errors") != 0:
                        issues.append("missing public-object rows or runner errors")
                    for row in summary.get("rows", []):
                        prefix = Path(row["path"])
                        if prefix.is_absolute() or ".." in prefix.parts:
                            raise ValueError("Unsafe public-object case path")
                        required += [(prefix / name).as_posix() for name in
                                     ("result.json", "metadata.json", "scene.xml", "native.json.gz", "trace.json.gz", "preload.json.gz", "constrained_joint.json.gz")]
                    for name in ("verification_receipt.json", "strict_native_verification.json", "native_command_bounds.json"):
                        if (path / name).is_file() and read_json(path / name).get("summary_sha256") != review.digest(path / "summary.json"):
                            issues.append(f"{name} does not bind this summary")
            elif label in {"current_carried_dependency_failure", "current_carried_success"}:
                required += ["result.json", "scene.xml", "physical_trace.jsonl.gz", "native_contacts.jsonl.gz", "action_log.json"]
                if label.endswith("success"):
                    required += ["video/tool_screwdriver.mp4", "video/metadata.json"]
            elif label == "current_continuous_sorting":
                required += ["summary.json", "scene.xml", "trace.json.gz", "events.json.gz", "action_log.json.gz", "tool_sorting.mp4"]
            elif label == "current_public_ycb_r15_nominal_static":
                required += ["summary.json", "ranked.json", "fixed_orientation_candidate_profile.json", "frozen_geometry_source.zip"]
            elif label == "current_independent_v7_audit":
                required += ["receipt.json", "three_tools.json", "rolling.json", "carried.json"]
            elif label == "current_independent_rgbd_audit":
                required += ["receipt.json", "statistics.json"]
            elif label == "current_compiled_physics":
                required += ["ycb_can.json", "ycb_cracker_box.json", "ycb_potted_meat_can.json"]
            for relative in required:
                if not (path / relative).is_file():
                    missing.append(relative)
            partials = [str(item.relative_to(path)) for item in path.rglob("*")
                        if item.name.endswith((".part", ".partial", ".tmp"))]
            issues.extend(f"partial transfer artifact: {name}" for name in partials)
        marker = path / ".complete" if path.is_dir() else None
        checks.append({"label": label, "complete": bool(files) and not missing and not issues,
            "missing_required_files": missing, "issues": issues,
            "completion_marker": ({"sha256": review.digest(safe_file(marker)), "bytes": marker.stat().st_size}
                                  if marker is not None and marker.is_file() else None),
            "outcome_policy": "A recorded false task outcome is complete evidence, not a missing artifact."})
    return checks


def inventory_current(brief, report_pdf, intro_video, *, compact):
    brief = safe_file(brief, project_only=False)
    report_pdf, intro_video = safe_file(report_pdf), safe_file(intro_video)
    if brief.suffix.lower() != ".pdf" or brief.read_bytes()[:5] != b"%PDF-":
        raise ValueError("--brief must be the local authoritative PDF")
    if report_pdf.suffix.lower() != ".pdf" or report_pdf.read_bytes()[:5] != b"%PDF-":
        raise ValueError("--report-pdf must contain a PDF header")
    if intro_video.suffix.lower() != ".mp4":
        raise ValueError("--intro-video must be an MP4")
    review.checked_report_assets([PROJECT / name for name in REPORT_ASSETS + R15_PROVENANCE_ASSETS])
    review.checked_report_assets([PROJECT / name for name in ORIGINAL_PATH_ASSETS])
    evidence = current_evidence()
    status = review.inventory(evidence, compact=compact)
    completeness = check_completeness(evidence)
    roots = dict(evidence)
    if "current_independent_rgbd_audit" in roots and "current_randomized_rgbd" in roots:
        audit = read_json(roots["current_independent_rgbd_audit"] / "receipt.json")
        bindings = all(audit.get(f"{name}_sha256") == review.digest(roots["current_randomized_rgbd"] / f"{name}.json")
                       for name in ("summary", "protocol"))
        completeness.append({"label": "rgbd_independent_receipt_binding", "complete": bindings and audit.get("accepted_trace_derived_audit") is True})
    if "current_independent_v7_audit" in roots:
        audit = read_json(roots["current_independent_v7_audit"] / "receipt.json")
        completeness.append({"label": "v7_independent_receipt_completion", "complete":
            audit.get("all_requested_audits_completed") is True and audit.get("all_available_evidence_consistent") is True})
    if "current_introduced_video_manifest" in roots:
        manifest = read_json(roots["current_introduced_video_manifest"])
        qa = read_json(roots["current_introduced_video_decode_qa"])
        completeness.append({"label": "introduced_video_manifest_binding", "complete":
            manifest["output"]["sha256"] == review.digest(intro_video) == qa["video_sha256"]
            and manifest["qa_receipt_sha256"] == review.digest(roots["current_introduced_video_decode_qa"])
            and qa["per_frame_receipt_sha256"] == review.digest(roots["current_introduced_video_frame_hashes"])})
    status.update(brief=str(brief), report=str(PROJECT / "docs/EE5062_REPORT_DRAFT.md"),
        scope_limitations=LIMITATIONS, pending_acceptance=PENDING, evidence_selection="explicit_current_20261021_20261022",
        evidence_completeness=completeness,
        declared_inputs={"brief": {"path": str(brief), "sha256": review.digest(brief)},
                         "report_pdf": {"path": str(report_pdf), "sha256": review.digest(report_pdf)},
                         "intro_video": {"path": str(intro_video), "sha256": review.digest(intro_video)}},
        review_notes="Current evidence is selected explicitly; no legacy default evidence or archived source ZIP is added. Current report links, the contact addendum at its original docs path, compiled-physics JSON records, ACTIVE_DEVELOPMENT.md and the R14 command-bounds record are retained at their original relative paths. Supplementary development links in these notes to R9/R11/R12/R13 audit, contact-distribution, probe and force-target studies, R14 prelaunch review and R14 phase-snapshot video depend on the original checkout or recovered research records; these supplementary paths are intentionally absent from this compact ZIP. Other old historical paths in development notes refer to the recovered historical review package only. Compact omitted raw artifacts are not embedded in this ZIP; RAW_EVIDENCE_INVENTORY.json records their hashes and exact inclusion locations, including report figures retained separately. The 214-second introduced video is autonomous simulation with synthetic narration. Human teleoperation and a successful current public-YCB randomized benchmark remain pending.\n\nTo reproduce the exact R15 guard inputs, first copy the review-root results/public_ycb_geometry_20261022_r15_final directory into the extracted source checkout at results/public_ycb_geometry_20261022_r15_final, preserving ranked.json and fixed_orientation_candidate_profile.json byte-for-byte. Then run the frozen wrapper. The source ZIP alone does not contain these results-derived exact-hash inputs; regenerating a nominal screen is not claimed to reconstruct the exact guarded bytes.")
    return evidence, status


def validate_current_copied_inputs(content, status):
    declared = status["declared_inputs"]
    destinations = {"brief": Path("course_brief") / Path(declared["brief"]["path"]).name,
                    "report_pdf": Path("report/EE5062_REPORT_DRAFT.pdf"),
                    "intro_video": Path("demonstration/INTRO_VIDEO_FOR_REVIEW.mp4")}
    for name, relative in destinations.items():
        if review.digest(content / relative) != declared[name]["sha256"]:
            raise ValueError(f"Copied {name} differs from validated declared input")
    _, fresh = inventory_current(*(Path(declared[name]["path"]) for name in
                                   ("brief", "report_pdf", "intro_video")),
                                 compact=status["review_scope"] == "compact_summary_review")
    if fresh["declared_inputs"] != declared or not all(item["complete"] for item in fresh["evidence_completeness"]):
        raise ValueError("Current evidence bindings changed or became incomplete after inventory")
    for record in status["artifact_files"]:
        if review.digest(Path(record["source"])) != record["sha256"] or review.digest(content / record["pack_path"]) != record["sha256"]:
            raise ValueError(f"Copied evidence differs from revalidated input: {record['source']}")
    status["evidence_completeness"] = fresh["evidence_completeness"]
    status["final_current_bindings_revalidated"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", required=True, type=Path)
    parser.add_argument("--report-pdf", type=Path, default=PROJECT / REPORT_PDF)
    parser.add_argument("--intro-video", type=Path, default=PROJECT / INTRO_VIDEO)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--source-frozen", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evidence, status = inventory_current(args.brief, args.report_pdf, args.intro_video, compact=args.compact)
        if args.inventory_only:
            result = status
        else:
            if args.output is None or not args.source_frozen:
                parser.error("build requires --output and --source-frozen; inventory-only does not freeze inputs")
            incomplete = [item["label"] for item in status["evidence_completeness"] if not item["complete"]]
            if incomplete:
                raise ValueError(f"Current evidence is missing or incomplete: {incomplete}")
            result = review.make_pack(args.output, evidence, args.intro_video, args.report_pdf, compact=args.compact,
                report_assets=[PROJECT / name for name in REPORT_ASSETS + R15_PROVENANCE_ASSETS], status_overrides=status,
                original_path_assets=[PROJECT / name for name in ORIGINAL_PATH_ASSETS],
                brief=args.brief, include_historical=False, validate_copied_inputs=validate_current_copied_inputs)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Current review package failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
