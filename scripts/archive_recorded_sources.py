#!/usr/bin/env python3
"""Archive current files only when they match a recorded source manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def archive_sources(project, protocol, output, receipt):
    project, protocol = Path(project).resolve(), Path(protocol).resolve()
    output, receipt = Path(output), Path(receipt)
    if output.exists() or receipt.exists():
        raise FileExistsError("Archive and receipt must be new files")
    recorded = protocol.read_bytes()
    sources = json.loads(recorded)["source_hashes_before"]
    if not sources:
        raise ValueError("Empty source manifest")
    payloads = {}
    for relative, expected in sorted(sources.items()):
        path = (project / relative).resolve()
        if path == project or not path.is_relative_to(project):
            raise ValueError("Source path escapes project")
        payload = path.read_bytes()
        if digest(payload) != expected:
            raise ValueError("Current source differs from recorded hash: " + relative)
        payloads[relative] = payload
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, payload in payloads.items():
            archive.writestr(relative, payload)
    with zipfile.ZipFile(output) as archive:
        if sorted(archive.namelist()) != sorted(sources):
            raise ValueError("Archive member mismatch")
        for relative, expected in sources.items():
            if digest(archive.read(relative)) != expected:
                raise ValueError("Archive hash mismatch: " + relative)
            if digest((project / relative).read_bytes()) != expected:
                raise ValueError("Source changed during archiving: " + relative)
    result = {
        "format": "post_run_recorded_source_archive_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol), "protocol_sha256": digest(recorded),
        "archive_path": str(output.resolve()), "archive_sha256": digest(output.read_bytes()),
        "source_count": len(sources), "source_hashes": sources,
        "verified_current_and_archive_sources_match_recorded_manifest": True,
        "scope": "Created after the run from files matching its recorded manifest; not a runtime capture.",
    }
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = archive_sources(args.project, args.protocol, args.output, args.receipt)
    print(json.dumps({key: result[key] for key in ("source_count", "archive_sha256")}))


if __name__ == "__main__":
    main()
