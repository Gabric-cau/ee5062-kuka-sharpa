"""Hash the retained vendor assets and legacy experiment files."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NEW_RESULTS = {"dell_20261007", "local_runs", "viewer_runs", "render_runs", "trace_runs", "launcher_checks", "local_cache"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.capture) == bool(args.verify):
        parser.error("choose exactly one of --capture or --verify")
    if args.capture:
        files = sorted((PROJECT / "vendor").rglob("*"))
        files += sorted(path for path in (PROJECT / "results").rglob("*")
                        if path.relative_to(PROJECT / "results").parts[0] not in NEW_RESULTS
                        and path.name != "dell_pre_development_20261007.zip")
        record = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                  "files": {path.relative_to(PROJECT).as_posix(): {"size": path.stat().st_size, "sha256": digest(path)}
                            for path in files if path.is_file()}}
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        with args.capture.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"Captured {len(record['files'])} retained files: {args.capture}")
    else:
        record = json.loads(args.verify.read_text(encoding="utf-8"))
        failed = []
        for name, expected in record["files"].items():
            path = PROJECT / name
            if not path.is_file() or path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
                failed.append(name)
        report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "files_checked": len(record["files"]),
                  "unchanged": not failed, "mismatches": failed}
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
