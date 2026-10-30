#!/usr/bin/env python3
"""Run paired fixed/force trials on this host with resumable evidence."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.experiments import experiment_cli


if __name__ == "__main__":
    experiment_cli()
