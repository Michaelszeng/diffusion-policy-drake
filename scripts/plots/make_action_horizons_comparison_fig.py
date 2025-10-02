"""
Plot success rates across action horizons using evaluation summaries.

This utility scans an evaluation experiment directory that contains sub-folders
of the form ``T_a_<horizon>``. Each of those folders is expected to contain one
or more checkpoint directories (for example ``epoch=0030...`` or
``latest.ckpt``). Every checkpoint directory must include a ``summary.pkl``
file. The script selects the checkpoint with the highest success rate for each
horizon and produces a single-trace figure that mimics the reference style
shared by the user.

Example usage
-------------

   python scripts/plots/make_action_horizons_comparison_fig.py \
       --experiment-path eval/sim_sim/baseline \
       --output outputs/action_horizon_success.png

Don't set --output to not save the figure to disk.
Set --show to show the figure in an interactive window after saving.

"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

NAVY = "#1f3b6f"
GRID_COLOR = "#bdbdbd"


@dataclass
class HorizonResult:
    horizon: float
    success_rate: float
    num_trials: int
    checkpoint_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create a success-rate vs. action-horizon plot using evaluation summaries.")
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        default=Path("eval/sim_sim/baseline"),
        help=("Path to the experiment directory that contains T_a_<horizon> sub-folders."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure (PNG, PDF, etc.).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in an interactive window after saving",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI when saving to disk.",
    )
    return parser.parse_args()


def iter_action_horizon_dirs(experiment_path: Path) -> Iterable[Path]:
    for candidate in sorted(experiment_path.iterdir()):
        if candidate.is_dir() and candidate.name.startswith("T_a_"):
            yield candidate


def parse_horizon_from_dirname(dirname: str) -> float:
    _, _, value = dirname.partition("T_a_")
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(f"Could not parse horizon from '{dirname}'.") from err


def load_success_rate(summary_path: Path) -> tuple[float, int]:
    with summary_path.open("rb") as handle:
        summary: Dict = pickle.load(handle)

    trial_results: Sequence[str] = summary.get("trial_result", [])
    if trial_results:
        successes = sum(1 for result in trial_results if result == "success")
        total = len(trial_results)
    else:
        successes = len(summary.get("successful_trials", []))
        total = len(summary.get("final_error", []))

    if total == 0:
        return float("nan"), 0

    return successes / total, total


def find_best_checkpoint(horizon_dir: Path) -> Optional[HorizonResult]:
    best_result: Optional[HorizonResult] = None

    summary_paths = sorted(horizon_dir.glob("*/summary.pkl"))
    for summary_path in summary_paths:
        success_rate, total_trials = load_success_rate(summary_path)
        if math.isnan(success_rate):
            continue

        checkpoint_dir = summary_path.parent
        horizon_value = parse_horizon_from_dirname(horizon_dir.name)
        candidate = HorizonResult(horizon_value, success_rate, total_trials, checkpoint_dir)

        if best_result is None:
            best_result = candidate
            continue

        if success_rate > best_result.success_rate + 1e-6:
            best_result = candidate
            continue

        are_equal = math.isclose(success_rate, best_result.success_rate, rel_tol=1e-6)
        if are_equal and total_trials > best_result.num_trials:
            best_result = candidate
            continue

        if are_equal and best_result.checkpoint_dir.name != "latest.ckpt" and checkpoint_dir.name == "latest.ckpt":
            best_result = candidate

    return best_result


def collect_best_results(experiment_path: Path) -> List[HorizonResult]:
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

    results: List[HorizonResult] = []
    for horizon_dir in iter_action_horizon_dirs(experiment_path):
        best = find_best_checkpoint(horizon_dir)
        if best is None:
            print(f"Skipping {horizon_dir} (no valid summary.pkl found).")
            continue
        results.append(best)

    results.sort(key=lambda res: res.horizon)
    return results


def make_plot(results: Sequence[HorizonResult], dpi: int) -> plt.Figure:
    horizons = np.array([res.horizon for res in results], dtype=float)
    success_rates = np.array([res.success_rate for res in results], dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.set_facecolor("white")

    ax.plot(
        horizons,
        success_rates,
        color=NAVY,
        linewidth=2.5,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(val)) if float(val).is_integer() else f"{val:g}" for val in horizons])

    y_min = 0.8
    ax.set_ylim(y_min, 1.0)
    ax.set_yticks(np.linspace(y_min, 1.0, 6))

    ax.set_title("Action Horizon", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Action Horizon (steps)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", labelsize=10, length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.8)
    ax.tick_params(axis="y", which="minor", left=False)

    fig.tight_layout()
    fig.set_dpi(dpi)
    return fig


def main() -> None:
    args = parse_args()
    results = collect_best_results(args.experiment_path)
    if not results:
        raise RuntimeError(f"No valid summary.pkl files found under {args.experiment_path}.")

    print("Best checkpoints per horizon:")
    for res in results:
        print(
            f"  Horizon {res.horizon:g}: success_rate={res.success_rate:.3f}"
            f" ({res.num_trials} trials) -> {res.checkpoint_dir}"
        )

    fig = make_plot(results, dpi=args.dpi)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
