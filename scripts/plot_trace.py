#!/usr/bin/env python3
"""Plot force and contact count from a trace CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT / "results" / "force_trace.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "force_trace.png")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    time = [float(row["time"]) for row in rows]
    normal = [float(row["total_normal_force"]) for row in rows]
    tangential = [float(row["total_tangential_force"]) for row in rows]
    target = [float(row["target_force"]) for row in rows]
    contacts = [float(row["contacts"]) for row in rows]
    target_total = [force * max(1.0, count) for force, count in zip(target, contacts)]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
    axes[0].plot(time, normal, label="normal contact force")
    axes[0].plot(time, tangential, label="tangential contact force", alpha=0.8)
    axes[0].plot(time, target_total, "--", label="per-fingertip target x contacts")
    axes[0].set_ylabel("total force (N)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    axes[1].step(time, contacts, where="post", label="active fingertip contacts")
    axes[1].set_xlabel("simulation time (s)")
    axes[1].set_ylabel("contacts")
    axes[1].grid(alpha=0.25)
    fig.suptitle("KUKA-Sharpa simulated contact-force trace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
