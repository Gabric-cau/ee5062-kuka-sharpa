#!/usr/bin/env python3
"""Install or verify exact R29 replay inputs using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe manifest/archive path: " + repr(name))
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name:
        raise ValueError("Noncanonical input path: " + name)
    return relative


def records_from_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != "r29_runtime_inputs_v1":
        raise ValueError("Unsupported runtime manifest format")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Runtime manifest has no files")
    names = set()
    for record in records:
        safe_relative(record["path"])
        if record["path"] in names:
            raise ValueError("Duplicate manifest input: " + record["path"])
        names.add(record["path"])
        if (type(record["bytes"]) is not int or record["bytes"] < 0
                or type(record["archive"]) is not bool
                or not isinstance(record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            raise ValueError("Invalid manifest size/hash/archive flag")
    if sum(r["bytes"] for r in records if r["archive"]) != manifest["archive_input_bytes"]:
        raise ValueError("Manifest archive byte total differs")
    return records


def destination(root, name):
    relative = safe_relative(name)
    path = root
    for part in (None, *relative.parts):
        if part is not None:
            path /= part
        if path.is_symlink():
            raise ValueError("Refusing symlink destination: " + str(path))
        if path != root.joinpath(*relative.parts) and path.exists() and not path.is_dir():
            raise ValueError("Destination parent is not a directory: " + str(path))
    return path


def file_matches(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == record["sha256"]


def check_inputs(root, records):
    for record in records:
        path = destination(root, record["path"])
        if not file_matches(path, record):
            raise ValueError("Missing or mismatched input: " + record["path"])
    return {"verified_files": len(records), "verified_bytes": sum(r["bytes"] for r in records)}


def install_inputs(root, records, archive_path):
    expected = {r["path"]: r for r in records if r["archive"]}
    installed, already_present = [], []
    for record in records:
        path = destination(root, record["path"])
        if path.exists():
            if not file_matches(path, record):
                raise ValueError("Existing input differs; refusing overwrite: " + record["path"])
            if record["archive"]:
                already_present.append(record["path"])
        elif not record["archive"]:
            raise ValueError("Vendored input missing: " + record["path"])

    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="r29-inputs-") as temporary:
        members = archive.infolist()
        for member in members:
            safe_relative(member.filename)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if member.is_dir() or kind not in (0, stat.S_IFREG):
                raise ValueError("Only regular files are accepted: " + member.filename)
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError("Archive members differ from the exact runtime input inventory")

        # Verify every payload in temporary storage before creating project files.
        staged = {}
        for index, member in enumerate(members):
            record = expected[member.filename]
            if member.file_size != record["bytes"]:
                raise ValueError("Archive input size differs: " + member.filename)
            path = Path(temporary) / str(index)
            digest, total = hashlib.sha256(), 0
            with archive.open(member) as source, path.open("xb") as target:
                for block in iter(lambda: source.read(CHUNK), b""):
                    total += len(block)
                    if total > record["bytes"]:
                        raise ValueError("Archive input exceeds declared size: " + member.filename)
                    digest.update(block)
                    target.write(block)
            if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("Archive input checksum differs: " + member.filename)
            staged[member.filename] = path

        for name, source in staged.items():
            path = destination(root, name)
            if path.exists():
                if not file_matches(path, expected[name]):
                    raise ValueError("Destination changed; refusing overwrite: " + name)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path = destination(root, name)
            with source.open("rb") as stream, path.open("xb") as target:
                shutil.copyfileobj(stream, target, CHUNK)
            installed.append(name)
    return {"installed": installed, "already_present": already_present}


def prepare(root, manifest_path, archive_path=None):
    root = Path(root).absolute()
    records = records_from_manifest(manifest_path)
    installation = install_inputs(root, records, archive_path) if archive_path is not None else {}
    return {"format": "r29_runtime_input_check_v1", "ok": True,
            **installation, **check_inputs(root, records),
            "scope": "Input identity only; no simulation or task-success verification performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Downloaded r29-runtime-inputs.zip")
    parser.add_argument("--check", action="store_true", help="Verify all recorded inputs and vendored mesh")
    args = parser.parse_args()
    if args.archive is None and not args.check:
        parser.error("Use --archive PATH to install, or --check to verify")
    try:
        report = prepare(PROJECT, PROJECT / "config/r29_runtime_inputs.json", args.archive)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, "R29 input check failed: " + str(error) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
