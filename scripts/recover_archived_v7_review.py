#!/usr/bin/env python3
"""Recover one exact V7 review ZIP into a new directory, verifying manifest bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


SUFFIX = "/kuka_sharpa_force_grasp/releases/EE5062_V7_Review_20261014/EE5062_LOCAL_REVIEW.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = [entry for entry in manifest["entries"]
        if entry["type"] == "file" and entry["path"].endswith(SUFFIX)]
    if len(entries) != 1:
        raise ValueError("Expected exactly one known V7 review archive in manifest")
    expected = entries[0]
    archive_before = args.archive.stat()
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "EE5062_LOCAL_REVIEW.zip"
    found = False
    # Stream once and stop at the exact member; never extract arbitrary paths.
    with tarfile.open(args.archive, mode="r|*") as archive:
        for member in archive:
            if member.name.removeprefix("./") != expected["path"]:
                continue
            if not member.isfile() or member.size != expected["size"]:
                raise ValueError("Archive member metadata differs from verified manifest")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Regular archive member has no byte stream")
            digest = hashlib.sha256()
            total = 0
            with source, destination.open("xb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    total += len(block)
            if total != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Recovered review archive failed exact byte/hash verification")
            found = True
            break
    if not found:
        raise ValueError("Exact review archive is absent from backup")
    archive_after = args.archive.stat()
    if (archive_before.st_size, archive_before.st_mtime_ns) != (archive_after.st_size, archive_after.st_mtime_ns):
        raise ValueError("Backup metadata changed during read-only recovery")
    receipt = {"scope": "Exact historical V7 review recovery, not a new result or submission",
        "archive_path": str(args.archive), "archive_bytes": archive_before.st_size,
        "archive_mtime_ns": archive_before.st_mtime_ns,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "member_path": expected["path"], "output": destination.name,
        "bytes": total, "sha256": digest.hexdigest(),
        "source_backup_metadata_unchanged": True,
        "recovery_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (args.output / "recovery_receipt.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, ensure_ascii=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
