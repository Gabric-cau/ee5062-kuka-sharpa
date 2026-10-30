#!/usr/bin/env python3
"""Validate and summarize manifest-backed local runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.experiments import _write_json, aggregate_runs, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize compatible runs, retaining failed and incomplete denominators")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="run directories or their results.csv")
    parser.add_argument("--output", type=Path, help="fresh summary CSV; existing output rejected")
    parser.add_argument("--json", type=Path, help="fresh summary JSON; existing output rejected")
    args = parser.parse_args()
    try:
        summary, coverage = aggregate_runs(args.input)
        if args.output:
            write_csv(args.output, summary, immutable=True)
        if args.json:
            _write_json(args.json, {"coverage": coverage, "groups": summary}, immutable=True)
        if summary:
            fields = [key for key in summary[0] if key != "incomplete_trial_ids"]
            writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        print(json.dumps(coverage, sort_keys=True))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
