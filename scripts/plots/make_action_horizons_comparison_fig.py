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
       --plot-name "Action Horizon Comparison" \
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

import math
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import ScalarFormatter

from plotting_utils import (
    GRID_COLOR,
    NAVY,
    parse_args,
    HorizonResult,
    collect_best_results,
    collect_all_checkpoint_results,
    generate_color_palette,
)

# Note: the original script defined its own colour constants and helper functions.
# Those have been moved to 'plotting_utils' to avoid duplication.

# -----------------------------------------------------------------------------
# The HorizonResult dataclass is now imported from plotting_utils
# -----------------------------------------------------------------------------

# The helper functions iter_action_horizon_dirs, parse_horizon_from_dirname, ...
# have been removed from this file and are now imported above.


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
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.set_facecolor("white")

    # Collect all unique horizons across all experiments for x-axis
    all_horizons = set()
    for _, results, _ in experiments:
        all_horizons.update(res.horizon for res in results)
    all_horizons = sorted(all_horizons)
    
    # Determine if we should show checkpoint labels (only for single experiment plots)
    show_checkpoint_labels = len(experiments) == 1

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
        
        # Add checkpoint labels if this is a single experiment and the horizon has multiple checkpoints
        if show_checkpoint_labels:
            for res in results:
                if res.num_checkpoints_available > 1:
                    # Truncate checkpoint name to first 10 chars
                    ckpt_name = res.checkpoint_dir.name
                    label_text = ckpt_name[:10] + "..." if len(ckpt_name) > 10 else ckpt_name
                    
                    # Position label slightly above the data point
                    ax.annotate(
                        label_text,
                        xy=(res.horizon, res.success_rate),
                        xytext=(0, 5),  # 5 points above
                        textcoords='offset points',
                        fontsize=6,
                        color=color,
                        ha='center',
                        va='bottom',
                        alpha=0.8,
                    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    if all_horizons:
        ax.set_xticks(all_horizons)
        ax.set_xticklabels([str(int(val)) if float(val).is_integer() else f"{val:g}" for val in all_horizons])

    # y_min = 0.35
    # ax.set_ylim(y_min, 1.0)
    # ax.set_yticks(np.linspace(y_min, 1.0, 6))

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
    description = "Create a success-rate comparison plot across action horizons using evaluation summaries."
    args = parse_args(description)
    
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
    
    # Handle --all-checkpoints mode
    if args.all_checkpoints:
        if len(experiment_paths) > 1:
            raise ValueError("--all-checkpoints is only supported for single experiment paths")
        
        exp_path = experiment_paths[0]
        checkpoint_results = collect_all_checkpoint_results(exp_path)
        
        if not checkpoint_results:
            raise RuntimeError(f"No valid checkpoints found under {exp_path}.")
        
        color_palette = generate_color_palette(len(checkpoint_results))
        
        for idx, (checkpoint_name, results) in enumerate(sorted(checkpoint_results.items())):
            color = color_palette[idx % len(color_palette)]
            experiments.append((checkpoint_name, results, color))
            
            print(f"\n{checkpoint_name}:")
            for res in results:
                print(f"  Horizon {res.horizon:g}: success_rate={res.success_rate:.3f} ({res.num_trials} trials)")
    else:  # Plot just the best checkpoint per action horizon
        # Use appropriate color palette based on number of experiments
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
            experiments.append((exp_label, results, color))
            
            print(f"\n{exp_label} - Best checkpoints per horizon:")
            for res in results:
                ckpt_info = f" [{res.num_checkpoints_available} checkpoints available]" if res.num_checkpoints_available > 1 else ""
                print(
                    f"  Horizon {res.horizon:g}: success_rate={res.success_rate:.3f}"
                    f" ({res.num_trials} trials) -> {res.checkpoint_dir}{ckpt_info}"
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
