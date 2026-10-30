#!/usr/bin/env python3
"""Build a compact, hash-bound EE5062 submission review ZIP.

The package contains the current report, source archive, demonstrations and
small evidence receipts. Large native traces remain outside the ZIP and are
identified by the included R29 reports and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CHUNK = 1024 * 1024

FILES = {
    "report/EE5062_SUBMISSION_REPORT_20261030.md": "docs/EE5062_SUBMISSION_REPORT_20261030.md",
    "report/EE5062_SUBMISSION_REPORT_20261030.pdf": "docs/EE5062_SUBMISSION_REPORT_20261030.pdf",
    "report/EE5062_SUBMISSION_READINESS_20261030.md": "docs/EE5062_SUBMISSION_READINESS_20261030.md",
    "report/V7_REQUIREMENTS_MATRIX.md": "V7_REQUIREMENTS_MATRIX.md",
    "report/V7_RUNBOOK.md": "V7_RUNBOOK.md",
    "report/V7_DELL_SETUP.md": "V7_DELL_SETUP.md",
    "source/source_and_assets_20261030.zip": "results/submission_source_20261030_v2/source_and_assets_20261030.zip",
    "source/source_and_assets_20261030.zip.manifest.json": "results/submission_source_20261030_v2/source_and_assets_20261030.zip.manifest.json",
    "source/source_and_assets_20261030.zip.inventory.txt": "results/submission_source_20261030_v2/source_and_assets_20261030.zip.inventory.txt",
    "source/source_and_assets_20261030.zip.sha256": "results/submission_source_20261030_v2/source_and_assets_20261030.zip.sha256",
    "demonstration/R29_native_pick_place.mp4": "results/r29_native_video_review_20261025/trajectory.mp4",
    "demonstration/V7_tool_sorting.mp4": "results/v7_tool_sorting_current_20261021_seed0/tool_sorting.mp4",
    "demonstration/V7_carried_screwdriver_obstacle.mp4": "results/v7_carried_obstacle_current_20261022_seed0/video/tool_screwdriver.mp4",
    "evidence/r29/result.json": "results/r29_event_arrival_full_task_20261024/result.json",
    "evidence/r29/summary.json": "results/r29_event_arrival_full_task_20261024/summary.json",
    "evidence/r29/protocol.json": "results/r29_event_arrival_full_task_20261024/protocol.json",
    "evidence/r29/independent_audit_macos.json": "results/r29_event_arrival_full_task_20261024/independent_audit_macos_20261024.json",
    "evidence/r29/independent_audit_linux.json": "results/r29_event_arrival_full_task_20261024/independent_audit_linux_20261025.json",
    "evidence/r29/independent_sequence_audit.json": "results/r29_event_arrival_full_task_20261024/independent_sequence_audit_20261024.json",
    "evidence/r29/trajectory_force.pdf": "results/r29_native_plot_20261024/trajectory_force.pdf",
    "evidence/r29/trajectory_force.png": "results/r29_native_plot_20261024/trajectory_force.png",
    "evidence/r29/plot_manifest.json": "results/r29_native_plot_20261024/manifest.json",
    "evidence/r29/video_metadata.json": "results/r29_native_video_review_20261025/metadata.json",
    "evidence/r29/result_notes.md": "docs/R29_FULL_TASK_RESULTS_20261025.md",
    "evidence/v7_tool_sorting/summary.json": "results/v7_tool_sorting_current_20261021_seed0/summary.json",
    "evidence/v7_obstacle/result.json": "results/v7_carried_obstacle_current_20261022_seed0/result.json",
    "evidence/v7_tactile/summary.json": "results/v7_three_tool_feedback_current_20261021_40_44/summary.json",
    "evidence/v7_rolling/summary.json": "results/v7_supported_rolling_current_20261022/summary.json",
    "evidence/v7_rgbd/summary.json": "results/v7_rgbd_randomized_current_20261022_1600_1629/summary.json",
}


def digest(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def info(name: str) -> zipfile.ZipInfo:
    result = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    result.create_system = 3
    result.external_attr = (stat.S_IFREG | 0o644) << 16
    result.compress_type = zipfile.ZIP_DEFLATED
    return result


def package(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    missing = [source for source in FILES.values() if not (PROJECT / source).is_file()]
    if missing:
        raise FileNotFoundError("missing package inputs: " + ", ".join(missing))
    records = []
    for target, source in sorted(FILES.items()):
        sha256, size = digest(PROJECT / source)
        records.append({"path": target, "source": source, "bytes": size, "sha256": sha256})
    readme = """# EE5062 submission review package\n\nThis is a current pre-submission package for the KUKA-Sharpa MuJoCo project.\nIt contains the updated report, source archive, demonstrations and compact evidence receipts.\n\nThe strongest result is R29: one reused YCB can development seed completes pick, lift, transfer, place and release, with independent native/command audits. The package states the remaining open requirements explicitly: randomized public-object trials under one frozen protocol, a person-produced teleoperation recording, and final team contribution review.\n\nThe source archive is reproducible on a compatible MuJoCo installation. The R29 raw native/controller streams are retained outside this compact ZIP because they are large; the included reports contain their hashes and paths. Do not describe this package as an official submission until the student/team has reviewed the report, AI acknowledgement, contribution statement and course policy.\n\nMain entry after extracting `source/source_and_assets_20261030.zip`: `scripts/run_coordinated_pick_place_r29.py`.\n""".encode()
    manifest = {"format": "ee5062_submission_review_package_v1", "files": records,
                "readme_sha256": hashlib.sha256(readme).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(info("README_SUBMISSION.md"), readme)
        for record in records:
            with (PROJECT / record["source"]).open("rb") as source, archive.open(info(record["path"]), "w") as target:
                for block in iter(lambda: source.read(CHUNK), b""):
                    target.write(block)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        archive.writestr(info("PACKAGE_MANIFEST.json"), manifest_bytes)
    zip_sha256, zip_size = digest(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha256}  {output.name}\n", encoding="utf-8")
    receipt = {"output": str(output), "bytes": zip_size, "sha256": zip_sha256,
               "file_count": len(records) + 2, "manifest": manifest}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
