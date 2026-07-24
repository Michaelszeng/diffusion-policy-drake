"""
Plot average trial duration across action horizons using evaluation summaries.

This utility scans one or more evaluation experiment directories that contain 
sub-folders of the form ``T_a_<horizon>``. It selects the best checkpoint (highest
success rate) for each horizon, then computes the average trial duration and 
standard deviation for that checkpoint.

The script includes a hard-coded parameter ONLY_SUCCESSFUL_TRIALS to toggle 
whether to only consider duration from successful trials.

Example usage
-------------

If running on SuperCloud: first run:
module load anaconda/Python-ML-2025a

python scripts/plots/make_action_horizons_duration_comparison_fig.py \
    --experiment-path eval/sim_sim/10_obs_32_horizon_idle_frames_pruned/v2 \
    --plot-name "Task Completion Time Comparison" \
    --output outputs/duration_comparison.png
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from plotting_utils import (
    GRID_COLOR,
    NAVY,
    HorizonResult,
    collect_all_checkpoint_results,
    collect_best_results,
    generate_color_palette,
    load_duration_stats,
    parse_args,
)

ONLY_SUCCESSFUL_TRIALS = True  # Toggle whether to filter for successful trials only


def make_plot(
    experiments: List[tuple[str, Sequence[Tuple[HorizonResult, float, float]], str]], 
    dpi: int, 
    plot_name: Optional[str] = None
) -> plt.Figure:
    """
    Create plot with multiple overlaid experiments.

    Args:
        experiments: List of (experiment_name, results_with_stats, color) tuples.
                     results_with_stats is a list of (HorizonResult, mean_duration, std_duration).
        dpi: Figure DPI
        plot_name: Optional title for the plot
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.set_facecolor("white")

    # Collect all unique horizons
    all_horizons = set()
    for _, results, _ in experiments:
        all_horizons.update(res.horizon for res, _, _ in results)
    all_horizons = sorted(all_horizons)

    # Plot each experiment
    for experiment_name, results, color in experiments:
        if not results:
            continue

        horizons = np.array([res.horizon for res, _, _ in results], dtype=float)
        means = np.array([mean for _, mean, _ in results], dtype=float)
        stds = np.array([std for _, _, std in results], dtype=float)

        # Main line with markers
        ax.plot(
            horizons,
            means,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=experiment_name,
            zorder=3,
        )

        # Vertical error bars (standard deviation)
        ax.errorbar(
            horizons,
            means,
            yerr=stds,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=4.0,
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

    # Set title
    if plot_name is not None:
        title = plot_name
    else:
        if len(experiments) == 1:
            title = f"Duration vs Horizon: {experiments[0][0]}"
        else:
            title = "Action Horizon Duration Comparison"
            
        if ONLY_SUCCESSFUL_TRIALS:
            title += "\n(Successful Trials Only)"
        else:
            title += "\n(All Trials)"
        
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    ax.set_xlabel("Action Horizon (steps)", fontsize=12)
    ax.set_ylabel("Average Trial Duration (s)", fontsize=12)

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
    description = "Create a duration comparison plot across action horizons using evaluation summaries."
    parser = parse_args(description)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Plot all checkpoints instead of just the best per horizon."
    )
    args = parser.parse_args()

    # Handle experiment paths
    experiment_paths = args.experiment_path if isinstance(args.experiment_path, list) else [args.experiment_path]

    # Generate legend labels
    if args.experiment_name is None:
        experiment_labels = [path.name for path in experiment_paths]
    else:
        experiment_labels = args.experiment_name
        if len(experiment_labels) != len(experiment_paths):
            raise ValueError(
                f"Number of experiment names ({len(experiment_labels)}) must match "
                f"number of experiment paths ({len(experiment_paths)})"
            )

    # Collect results
    # Each entry is (label, list of (HorizonResult, mean, std), color)
    experiments: List[tuple[str, Sequence[Tuple[HorizonResult, float, float]], str]] = []

    if args.all_checkpoints:
        if len(experiment_paths) > 1:
            raise ValueError("--all-checkpoints is only supported for single experiment paths")

        exp_path = experiment_paths[0]
        checkpoint_results_map = collect_all_checkpoint_results(exp_path)

        if not checkpoint_results_map:
            raise RuntimeError(f"No valid checkpoints found under {exp_path}.")

        color_palette = generate_color_palette(len(checkpoint_results_map))

        for idx, (checkpoint_name, results) in enumerate(sorted(checkpoint_results_map.items())):
            color = color_palette[idx % len(color_palette)]
            
            stats_results = []
            for res in results:
                if res.summary_path:
                     mean, std = load_duration_stats(res.summary_path, success_only=ONLY_SUCCESSFUL_TRIALS)
                     if not np.isnan(mean):
                         stats_results.append((res, mean, std))
            
            if stats_results:
                experiments.append((checkpoint_name, stats_results, color))
                print(f"\n{checkpoint_name}:")
                for res, mean, std in stats_results:
                    print(f"  Horizon {res.horizon:g}: duration={mean:.2f}±{std:.2f}s")

    else:
        if len(experiment_paths) == 1:
            color_palette = [NAVY]
        else:
            color_palette = generate_color_palette(len(experiment_paths))

        for idx, (exp_path, exp_label) in enumerate(zip(experiment_paths, experiment_labels)):
            results = collect_best_results(exp_path)
            if not results:
                print(f"Warning: No valid summary.pkl files found under {exp_path}. Skipping.")
                continue

            color = color_palette[idx % len(color_palette)]
            
            stats_results = []
            for res in results:
                if res.summary_path:
                     mean, std = load_duration_stats(res.summary_path, success_only=ONLY_SUCCESSFUL_TRIALS)
                     if not np.isnan(mean):
                         stats_results.append((res, mean, std))
            
            if stats_results:
                experiments.append((exp_label, stats_results, color))
                print(f"\n{exp_label} - Best checkpoints per horizon:")
                for res, mean, std in stats_results:
                    print(
                        f"  Horizon {res.horizon:g}: duration={mean:.2f}±{std:.2f}s "
                        f"(SR={res.success_rate:.2f}) -> {res.checkpoint_dir.name}"
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

