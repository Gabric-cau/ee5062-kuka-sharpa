"""Plot recorded success denominators and paired outcomes for one local run."""

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / "local_cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kuka_sharpa_grasp.experiments import aggregate_runs


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a plot")
    groups, coverage = aggregate_runs([args.input])
    if coverage["incomplete_trials"]:
        parser.error("finish or resume this run before plotting completed-outcome rates")
    mass_groups = list(dict.fromkeys(row["mass_group"] for row in groups))
    fig, (rates, pairs) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"fixed": "#59656f", "force": "#007d72"}
    locations = np.arange(len(mass_groups))
    for offset, controller in ((-0.18, "fixed"), (0.18, "force")):
        rows = [next(row for row in groups if row["mass_group"] == mass and row["controller"] == controller)
                for mass in mass_groups]
        values = [row["success_rate_completed"] for row in rows]
        intervals = [wilson(row["successes"], row["completed_trials"]) for row in rows]
        errors = np.array([[value - low, high - value] for value, (low, high) in zip(values, intervals)]).T
        rates.bar(locations + offset, values, width=0.34, label=controller, color=colors[controller],
                  yerr=errors, capsize=4, error_kw={"linewidth": 1})
        for x, interval, row in zip(locations + offset, intervals, rows):
            rates.text(x, interval[1] + 0.025, f"{row['successes']}/{row['completed_trials']}", ha="center", fontsize=9)
    rates.set_xticks(locations, mass_groups)
    rates.set_xlabel("Object mass group (kg)")
    rates.set_ylabel("Recorded success fraction")
    rates.set_ylim(0, 1.12)
    rates.set_title("Grasp / lift / final contact")
    rates.legend(loc="upper left")
    rates.grid(axis="y", alpha=0.2)
    rates.set_axisbelow(True)
    paired = coverage["paired_outcomes"]
    totals = [sum(row[name] for row in paired) for name in ("both_succeed", "fixed_only", "force_only", "neither")]
    pairs.bar(["Both succeed", "Fixed only", "Force only", "Neither"], totals,
              color=["#3f7f4f", "#59656f", "#007d72", "#bb394b"])
    for index, total in enumerate(totals):
        pairs.text(index, total + 0.15, str(total), ha="center")
    pairs.set_ylim(0, max(totals) * 1.25 + 1)
    pairs.set_ylabel("Paired scenarios")
    pairs.set_title("Identical pose / mass / friction per pair")
    pairs.grid(axis="y", alpha=0.2)
    pairs.set_axisbelow(True)
    fig.suptitle(f"{groups[0]['object_kind']} | {groups[0]['host']} | CPU MuJoCo | {coverage['completed_trials']} trials", fontsize=13)
    fig.supxlabel("Left error bars: marginal Wilson 95% intervals. Failed trials remain in the denominator.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
