#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.scene import OBJECT_KINDS, build_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=OBJECT_KINDS, default="box")
    args = parser.parse_args()
    print(build_scene(object_kind=args.object))
