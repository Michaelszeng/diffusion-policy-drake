"""
Plot *relative* success rate (delta w.r.t. baseline) across action horizons.

The CLI and directory structure expectations are identical to
``make_action_horizons_comparison_fig.py`` except that the first
``--experiment-path`` acts as the *baseline*.  For every subsequent
experiment, the y-value is the success-rate difference compared to the
baseline at matching horizons.  Positive values indicate improvement over the
baseline, negative values indicate worse performance.

Example
-------
python scripts/plots/make_action_horizons_delta_fig.py \
    --experiment-path eval/sim_sim/baseline eval/sim_sim/friction_0_3 \
    --experiment-name "Baseline" "mu=0.3" \
    --plot-name "Relative Success Rate" \
    --output outputs/action_horizon_delta.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from plotting_utils import (
    GRID_COLOR,
    NAVY,
    parse_args,
    HorizonResult,
    collect_best_results,
    generate_color_palette,
)

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def make_delta_plot(
    baseline_results: Sequence[HorizonResult],
    comparisons: List[Tuple[str, Sequence[HorizonResult], str]],
    dpi: int,
    plot_name: Optional[str] = None,
) -> plt.Figure:
    """Create a plot showing success-rate deltas relative to *baseline_results*."""

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_facecolor("white")

    # Build a mapping from horizon -> baseline success-rate for quick lookup
    baseline_map = {res.horizon: res.success_rate for res in baseline_results}
    baseline_horizons = sorted(baseline_map.keys())

    # Plot each comparison experiment
    for exp_name, results, color in comparisons:
        if not results:
            continue

        # Only evaluate horizons that exist in *both* baseline and this experiment
        shared = [res for res in results if res.horizon in baseline_map]
        if not shared:
            print(f"Warning: 0 overlapping horizons between baseline and {exp_name}. Skipping.")
            continue

        horizons = np.array([res.horizon for res in shared], dtype=float)
        deltas = np.array([
            res.success_rate - baseline_map[res.horizon] for res in shared
        ])

        ax.plot(
            horizons,
            deltas,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=exp_name,
            zorder=3,
        )

        # Small horizontal line at y=0
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)

    # Axis scaling / labels
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    if baseline_horizons:
        ax.set_xticks(baseline_horizons)
        ax.set_xticklabels([str(int(h)) if float(h).is_integer() else f"{h:g}" for h in baseline_horizons])

    ax.set_ylabel("Δ Success Rate", fontsize=12)
    ax.set_xlabel("Action Horizon (steps)", fontsize=12)

    y_abs_max = max(0.05, abs(deltas).max(initial=0)) + 0.05
    ax.set_ylim(-y_abs_max, y_abs_max)

    # Grid and style
    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", labelsize=8, length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.8)
    ax.tick_params(axis="y", which="minor", left=False)

    if comparisons:
        ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#4f4f4f")

    if plot_name:
        ax.set_title(plot_name, fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.set_dpi(dpi)
    return fig


# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main() -> None:
    description = "Create a success-rate delta vs. action-horizon plot using evaluation summaries."
    args = parse_args(description)

    experiment_paths: List[Path] = (
        args.experiment_path if isinstance(args.experiment_path, list) else [args.experiment_path]
    )
    if len(experiment_paths) < 2:
        raise ValueError("At least two --experiment-path values are required (baseline + comparison).")

    # Labels
    if args.experiment_name is None:
        labels = [p.name for p in experiment_paths]
    else:
        labels = args.experiment_name
        if len(labels) != len(experiment_paths):
            raise ValueError("Number of --experiment-name values must match --experiment-path.")

    # Collect results
    baseline_results = collect_best_results(experiment_paths[0])
    if not baseline_results:
        raise RuntimeError("Baseline experiment contains no valid summary.pkl files.")

    # Use Navy if there is only one comparison experiment
    num_comparisons = len(experiment_paths) - 1
    if num_comparisons == 1:
        cmp_palette = [NAVY]
    else:
        cmp_palette = generate_color_palette(num_comparisons+1)[1:]  # skip bright green corresponding to baseline
        print(f"Comparison palette: {cmp_palette}")

    comparisons: List[Tuple[str, Sequence[HorizonResult], str]] = []
    for idx, (path, label) in enumerate(zip(experiment_paths[1:], labels[1:])):
        results = collect_best_results(path)
        if not results:
            print(f"Warning: No valid summary.pkl in {path}. Skipping.")
            continue
        color = cmp_palette[idx % len(cmp_palette)]
        comparisons.append((label, results, color))

        print(f"\n{label} - Δ success rate relative to baseline:")
        for res in results:
            if res.horizon in {b.horizon for b in baseline_results}:
                delta = res.success_rate - next(b.success_rate for b in baseline_results if b.horizon == res.horizon)
                print(f"  Horizon {res.horizon:g}: Δ={delta:+.3f}")

    if not comparisons:
        raise RuntimeError("No comparison experiments contain valid data overlapping with baseline.")

    fig = make_delta_plot(baseline_results, comparisons, dpi=args.dpi, plot_name=args.plot_name)

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
