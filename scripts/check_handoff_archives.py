#!/usr/bin/env python3
"""Verify release archives and write their local SHA-256 checksum list."""

import hashlib
import json
from pathlib import Path
import zipfile


def main():
    folder = Path(__file__).resolve().parents[1] / "artifacts"
    files = sorted([*folder.glob("*.zip"), folder / "PROJECT_REPORT.pdf"])
    records = []
    for path in files:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError(f"Corrupt archive member: {path.name}:{bad}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    (folder / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['name']}\n" for r in records), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
