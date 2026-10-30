"""Create a reproducible source-and-assets ZIP for another project checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "requirements.txt",
    "README.md",
    "V3_RUNBOOK.md",
    "START_PICK_PLACE.cmd",
    "vendor/mujoco_menagerie/LICENSE",
    "vendor/mujoco_menagerie/README.md",
    "vendor/mujoco_menagerie/CITATION.cff",
)
OPTIONAL_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "V3_VALIDATION.md",
                  "ACTIVE_DEVELOPMENT.md", "PHYSICS_AND_CONTACT_CONTROL.md", "CONNECTIONS_20261021.md",
                  "\u5f53\u524d\u5165\u53e3\u4e0e\u5b9e\u9a8c\u72b6\u6001_20261021.md",
                  "docs/R14_CONSTRAINED_JOINT_DESIGN_20261021.md",
                  "docs/R15_NOMINAL_GEOMETRY_DIAGNOSIS_20261022.md",
                  "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md",
                  "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021_FINAL2.pdf",
                  "docs/V7_OPERATOR_RECORDING_AUDIT.md",
                  "docs/EE5062_REPORT_DRAFT.md", "docs/EE5062_REPORT_DRAFT.pdf",
                  "requirements-motion.txt", "requirements-report.txt", "requirements-public-ycb.txt", "START_V7_SORTING.cmd", "SETUP_V7.cmd",
                  "V7_GUI_ACCEPTANCE.md", "V7_OBSTACLE_EXECUTION.md", "V7_SUPPORTED_ROLLING.md", "V7_DELL_SETUP.md",
                  "PUBLIC_YCB_DEVELOPMENT.md",
                  "PHYSICS_PARAMETERS.md", "V6_RUNBOOK.md", "V6_VALIDATION.md", "START_PHYSICS_PICK_PLACE.cmd",
                  "V7_DESIGN.md", "V7_RUNBOOK.md", "V7_REQUIREMENTS_MATRIX.md", "START_V7_WORKBENCH.cmd",
                  "V7_PROTOCOL_RESULTS.md", "scripts/run_v7_rgbd_marker.py",
                  "scratch/v6_grip/README.md", "scratch/v6_grip/tune_grip.py",
                  "scratch/v6_grip/diagnose_release.py",
                  "scratch/v6_friction/tune_solver.py",
                  "scratch/v7_physics_sensitivity/evaluate.py", "scratch/v7_physics_sensitivity/verify.py",
                  "scratch/v7_physics_sensitivity/crossvalidate_continuous.py",
                  "scratch/v7_dell_deploy/audit.py",
                  "V4_RUNBOOK.md", "V4_VALIDATION.md", "V4_DIAGNOSIS.md", "DELIVERY.md",
                  "V5_RUNBOOK.md", "V5_VALIDATION.md", "V5_DIAGNOSIS.md",
                  "RESUME_EVIDENCE.md", "requirements-dell.txt", ".gitignore",
                  "START_V7_WORKBENCH.cmd",
                  "scratch/geometry_objects/README.md",
                  "scratch/geometry_objects/objects_experiment.py",
                  "scratch/geometry_objects/place_controller.py",
                  "scratch/geometry_objects/summarize_results.py",
                  "scratch/geometry_objects/study_summary.py",
                  "scratch/geometry_objects/study_summary.json",
                  "scratch/can_place_timing/README.md",
                  "scratch/can_place_timing/controller.py",
                  "scratch/can_place_timing/compare.py",
                  "scratch/can_place_timing/place5_comparison.json",
                  "scratch/v4_release/evaluate_upgrade.py",
                  "scratch/v4_release/plot_upgrade.py",
                  "scratch/v5_validation/evaluate_final.py",
                  "scratch/v5_validation/conditions_dell.json",
                  "scratch/v5_validation/conditions_v5_only.json")
SOURCE_TREES = ("src", "scripts", "tests", "config")
ASSET_TREES = (
    "vendor/mujoco_menagerie/kuka_iiwa_14",
    "vendor/mujoco_menagerie/sharpa_wave",
    "vendor/ycb/003_cracker_box/003_cracker_box/poisson",
    "vendor/ycb/005_tomato_soup_can/005_tomato_soup_can/poisson",
    "vendor/ycb/010_potted_meat_can/010_potted_meat_can/poisson",
)
REQUIRED_ASSETS = (
    "vendor/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml",
    "vendor/mujoco_menagerie/kuka_iiwa_14/LICENSE",
    "vendor/mujoco_menagerie/sharpa_wave/left_hand.xml",
    "vendor/mujoco_menagerie/sharpa_wave/right_hand.xml",
    "vendor/mujoco_menagerie/sharpa_wave/LICENSE",
    *(
        f"vendor/ycb/{name}/{name}/poisson/nontextured.stl"
        for name in ("003_cracker_box", "005_tomato_soup_can", "010_potted_meat_can")
    ),
)
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", ".env", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "results", "scratch", "archive",
    ".ssh", ".aws", ".config", "__MACOSX",
}
EXCLUDED_NAMES = {
    ".DS_Store", "Thumbs.db", "desktop.ini", "MUJOCO_LOG.TXT",
    "authorized_keys", "known_hosts", "id_rsa", "id_ed25519", "id_ecdsa",
    "credentials", "credentials.json", "secrets.json", "viewer_validation_events.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx"}
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CHUNK_SIZE = 1024 * 1024


def excluded(path: Path) -> bool:
    return (
        path.name in EXCLUDED_NAMES
        or path.name.startswith(".")
        or path.suffix.lower() in EXCLUDED_SUFFIXES
        or any(part in EXCLUDED_DIRS for part in path.parts)
    )


def reject_symlink(path: Path, project: Path) -> None:
    for part in (path, *path.parents):
        if part == project:
            return
        if part.is_symlink():
            raise ValueError(f"Symlinks are not allowed in a release: {part.relative_to(project)}")
    raise ValueError(f"Path is outside the project: {path}")


def collect_files(project: Path) -> list[Path]:
    files: set[Path] = set()
    for name in REQUIRED_FILES + REQUIRED_ASSETS:
        path = project / name
        reject_symlink(path, project)
        if not path.is_file():
            raise ValueError(f"Required regular file is missing: {name}")
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        path = project / name
        reject_symlink(path, project)
        if path.is_file():
            files.add(path)
    for name in SOURCE_TREES + ASSET_TREES:
        root = project / name
        reject_symlink(root, project)
        if not root.is_dir():
            raise ValueError(f"Required regular directory is missing: {name}")
        for directory, directories, names in os.walk(root, followlinks=False):
            parent = Path(directory)
            kept = []
            for dirname in sorted(directories):
                path = parent / dirname
                if dirname.startswith(".") or excluded(path.relative_to(project)):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Symlinks are not allowed in a release: {path.relative_to(project)}")
                kept.append(dirname)
            directories[:] = kept
            for filename in sorted(names):
                path = parent / filename
                relative = path.relative_to(project)
                if excluded(relative):
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Only regular files are allowed in a release: {relative}")
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(project).as_posix())


def file_record(path: Path, project: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
            size += len(block)
    return {"path": path.relative_to(project).as_posix(), "size_bytes": size, "sha256": digest.hexdigest()}


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def write_source(archive: zipfile.ZipFile, path: Path, expected: dict) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source, archive.open(zip_info(expected["path"]), "w") as target:
        for block in iter(lambda: source.read(CHUNK_SIZE), b""):
            target.write(block)
            digest.update(block)
            size += len(block)
    if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Source changed while packaging; retry after edits finish: {expected['path']}")


def package(project: Path, output: Path) -> dict:
    project = project.resolve(strict=True)
    output = output.absolute()
    destinations = (
        output,
        output.with_name(output.name + ".manifest.json"),
        output.with_name(output.name + ".inventory.txt"),
        output.with_name(output.name + ".sha256"),
    )
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    files = collect_files(project)
    records = [file_record(path, project) for path in files]
    manifest = {
        "format_version": 1,
        "zip_timestamps": "1980-01-01T00:00:00",
        "source_file_count": len(records),
        "source_size_bytes": sum(record["size_bytes"] for record in records),
        "included_trees": list(SOURCE_TREES + ASSET_TREES),
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    inventory = (
        "SHA256  BYTES  PATH\n"
        + "".join(f"{record['sha256']}  {record['size_bytes']}  {record['path']}\n" for record in records)
        + f"\nSOURCE_FILES {manifest['source_file_count']}\nSOURCE_BYTES {manifest['source_size_bytes']}\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("xb") as handle:
            created.append(output)
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path, record in zip(files, records):
                    write_source(archive, path, record)
                archive.writestr(zip_info("PACKAGE_MANIFEST.json"), manifest_bytes)
                archive.writestr(zip_info("PACKAGE_INVENTORY.txt"), inventory)
        for path, content in zip(destinations[1:3], (manifest_bytes, inventory)):
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
        digest = hashlib.sha256()
        with output.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        with destinations[3].open("x", encoding="utf-8", newline="\n") as handle:
            created.append(destinations[3])
            handle.write(f"{checksum}  {output.name}\n")
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return {
        "output": str(output),
        "zip_size_bytes": output.stat().st_size,
        "zip_sha256": checksum,
        "source_file_count": manifest["source_file_count"],
        "source_size_bytes": manifest["source_size_bytes"],
        "manifest": str(destinations[1]),
        "inventory": str(destinations[2]),
        "checksum": str(destinations[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new ZIP path; existing output or sidecars are never overwritten")
    parser.add_argument("--project-root", type=Path, default=PROJECT, help="checkout to package (default: parent of this scripts directory)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip":
        parser.error("--output must end in .zip")
    try:
        report = package(args.project_root, args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
