"""
Plot success rates across action horizons using evaluation summaries.

This utility scans one or more evaluation experiment directories that contain 
sub-folders of the form ``T_a_<horizon>``. Each of those folders is expected to 
contain one or more checkpoint directories (for example ``epoch=0030...`` or
``latest.ckpt``). Every checkpoint directory must include a ``summary.pkl`` file.
The script selects the checkpoint with the highest success rate for each
horizon and produces a figure with overlaid traces for multiple experiments.

Example usage
-------------

Single experiment:
   python scripts/plots/make_action_horizons_comparison_fig.py \
       --experiment-path eval/sim_sim/baseline \
       --output outputs/action_horizon_success.png

Multiple experiments with custom legend labels and title:
   python scripts/plots/make_action_horizons_comparison_fig.py \
       --experiment-path eval/sim_sim/baseline eval/sim_sim/friction_0_3 eval/sim_sim/friction_0_1 \
       --experiment-name "Baseline" "mu=0.3" "mu=0.1" \
       --plot-name "Action Horizon Comparison" \
       --output outputs/action_horizon_comparison.png

Don't set --output to not save the figure to disk.
Set --show to show the figure in an interactive window after saving.

Notes:
- --experiment-name is used for legend labels (optional, defaults to directory names)
- --plot-name is used for the plot title (optional, defaults to auto-generated title)
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
import statsmodels.stats.proportion as smp
from matplotlib.ticker import ScalarFormatter

NAVY = "#1f3b6f"
GRID_COLOR = "#bdbdbd"
# Color palette for multiple experiments
# COLOR_PALETTE = [
#     "#1f3b6f",  # Navy (original)
#     "#d62728",  # Red
#     "#2ca02c",  # Green
#     "#ff7f0e",  # Orange
#     "#9467bd",  # Purple
#     "#8c564b",  # Brown
#     "#e377c2",  # Pink
#     "#17becf",  # Cyan
# ]

COLOR_PALETTE = [
    "#2ca02c",  # Green
    # "#5cb830",  # Yellow-green
    # "#8bcf34",  # Lime
    "#bae738",  # Yellow-lime
    # "#e8ff3c",  # Yellow
    # "#ffc940",  # Orange-yellow
    "#ff9344",  # Orange
    # "#ff5d48",  # Red-orange
    # "#ff274c",  # Red-pink
    "#d62728",  # Red
]

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
        nargs='+',
        default=[Path("eval/sim_sim/baseline")],
        help=("Path(s) to experiment directories that contain T_a_<horizon> sub-folders. "
              "Can specify multiple paths to overlay multiple experiments."),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        nargs='+',
        default=None,
        help=("Name(s) of the experiment(s) for the legend labels. "
              "If not provided, labels will be derived from directory names. "
              "Number of names should match number of experiment paths."),
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help=("Name for the plot title. "
              "If not provided, defaults to 'Action Horizon Comparison' for multiple experiments "
              "or 'Action Horizon: <experiment_name>' for single experiment."),
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




def make_plot(
    experiments: List[tuple[str, Sequence[HorizonResult], str]], 
    dpi: int,
    plot_name: Optional[str] = None
) -> plt.Figure:
    """
    Create plot with multiple overlaid experiments.
    
    Args:
        experiments: List of (experiment_name, results, color) tuples
        dpi: Figure DPI
        plot_name: Optional title for the plot
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_facecolor("white")

    # Collect all unique horizons across all experiments for x-axis
    all_horizons = set()
    for _, results, _ in experiments:
        all_horizons.update(res.horizon for res in results)
    all_horizons = sorted(all_horizons)

    # Plot each experiment
    for experiment_name, results, color in experiments:
        if not results:
            continue

        horizons = np.array([res.horizon for res in results], dtype=float)
        success_rates = np.array([res.success_rate for res in results], dtype=float)
        num_trials = np.array([res.num_trials for res in results], dtype=int)

        # Compute Wilson 95% CI per point
        ci_bounds = np.array([smp.proportion_confint(int(p * n), n, alpha=0.05, method='wilson') for p, n in zip(success_rates, num_trials)], dtype=float)
        ci_lo = ci_bounds[:, 0]
        ci_hi = ci_bounds[:, 1]
        # yerr expects distances from the central value
        yerr = np.vstack([
            np.clip(success_rates - ci_lo, 0, 1),   # lower distances
            np.clip(ci_hi - success_rates, 0, 1),   # upper distances
        ])

        # Main line with markers
        ax.plot(
            horizons,
            success_rates,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=experiment_name,
            zorder=3,
        )

        # Thin vertical error bars with horizontal caps (95% Wilson CI)
        ax.errorbar(
            horizons,
            success_rates,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=4.0,      # horizontal caps at top/bottom
            capthick=1.0,
            alpha=0.9,
            zorder=2,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    if all_horizons:
        ax.set_xticks(all_horizons)
        ax.set_xticklabels([str(int(val)) if float(val).is_integer() else f"{val:g}" for val in all_horizons])

    y_min = 0.6
    ax.set_ylim(y_min, 1.0)
    ax.set_yticks(np.linspace(y_min, 1.0, 6))

    # Set title
    if plot_name is not None:
        title = plot_name
    elif len(experiments) == 1:
        title = f"Action Horizon: {experiments[0][0]}"
    else:
        title = "Action Horizon Comparison"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    
    ax.set_xlabel("Action Horizon (steps)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", labelsize=8, length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.8)
    ax.tick_params(axis="y", which="minor", left=False)

    # Add legend if multiple experiments
    if len(experiments) > 1:
        ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#4f4f4f")

    fig.tight_layout()
    fig.set_dpi(dpi)
    return fig


def main() -> None:
    args = parse_args()
    
    # Handle experiment paths
    experiment_paths = args.experiment_path if isinstance(args.experiment_path, list) else [args.experiment_path]
    
    # Generate legend labels: use provided names or fall back to directory names
    if args.experiment_name is None:
        experiment_labels = [path.name for path in experiment_paths]
    else:
        experiment_labels = args.experiment_name
        if len(experiment_labels) != len(experiment_paths):
            raise ValueError(
                f"Number of experiment names ({len(experiment_labels)}) must match "
                f"number of experiment paths ({len(experiment_paths)})"
            )
    
    # Collect results for each experiment
    experiments: List[tuple[str, Sequence[HorizonResult], str]] = []
    for idx, (exp_path, exp_label) in enumerate(zip(experiment_paths, experiment_labels)):
        results = collect_best_results(exp_path)
        if not results:
            print(f"Warning: No valid summary.pkl files found under {exp_path}. Skipping.")
            continue
        
        # Assign color from palette (cycle if more experiments than colors)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        experiments.append((exp_label, results, color))
        
        print(f"\n{exp_label} - Best checkpoints per horizon:")
        for res in results:
            print(
                f"  Horizon {res.horizon:g}: success_rate={res.success_rate:.3f}"
                f" ({res.num_trials} trials) -> {res.checkpoint_dir}"
            )
    
    if not experiments:
        raise RuntimeError("No valid experiments found to plot.")
    
    fig = make_plot(experiments, dpi=args.dpi, plot_name=args.plot_name)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
