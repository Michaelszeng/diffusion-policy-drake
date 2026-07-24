"""
Interactive 3D surface plot of success rates vs action horizon and effective memory.

This script compares the trade-off between action horizon and observation horizon
across different "effective memory" values (defined as action_horizon + observation_horizon).

Creates an interactive 3D surface plot (HTML) with:
- X-axis: Action horizon
- Y-axis: Effective memory
- Z-axis: Success rate

Scans multiple eval experiment directories, each with a known observation horizon,
and looks for matching action horizons to satisfy the effective memory constraint.
If the required action horizon doesn't exist, that data point is skipped.

The output is a self-contained HTML file that can be opened in any web browser
with full rotation, zoom, and hover capabilities.

Example usage
-------------

If running on SuperCloud: first run:
module load anaconda/Python-ML-2025a

python scripts/plots/make_fixed_effective_memory_comparison_fig.py \
    --experiment-path eval/sim_sim/baseline/v4 eval/sim_sim/6_obs_32_horizon/v2 eval/sim_sim/10_obs_32_horizon/v2 eval/sim_sim/14_obs_32_horizon/v2 eval/sim_sim/18_obs_32_horizon/v2 \
    --experiment-name "T_o=2" "T_o=6" "T_o=10" "T_o=14" "T_o=18" \
    --observation-horizons 2 6 10 14 18 \
    --effective-memories 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 \
    --plot-name "Action Horizon vs Effective Memory" \
    --output outputs/fixed_effective_memory_comparison.html

python scripts/plots/make_fixed_effective_memory_comparison_fig.py \
    --experiment-path eval/sim_sim/baseline/v4 eval/sim_sim/6_obs_32_horizon_idle_frames_pruned/v2 eval/sim_sim/10_obs_32_horizon_idle_frames_pruned/v2 eval/sim_sim/14_obs_32_horizon_idle_frames_pruned/v2 eval/sim_sim/18_obs_32_horizon_idle_frames_pruned/v2 \
    --experiment-name "T_o=2" "T_o=6" "T_o=10" "T_o=14" "T_o=18" \
    --observation-horizons 2 6 10 14 18 \
    --effective-memories 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 \
    --plot-name "Action Horizon vs Effective Memory (Idle Frames Pruned)" \
    --output outputs/fixed_effective_memory_comparison_idle_frames_pruned.html
"""

from __future__ import annotations

from typing import Dict, List, Optional

import argparse
import numpy as np
from pathlib import Path
from plotting_utils import (
    HorizonResult,
    collect_best_results,
    parse_args,
)


def collect_effective_memory_data(
    experiment_paths: List[Path],
    observation_horizons: List[int],
    effective_memories: List[int],
    experiment_names: Optional[List[str]] = None,
) -> Dict[int, List[tuple[float, HorizonResult]]]:
    """
    Collect data points for each effective memory value.
        
    Returns:
        Dictionary mapping effective_memory -> list of (action_horizon, HorizonResult) tuples
    """
    # Collect results for each experiment
    experiment_results: List[tuple[int, List[HorizonResult], str]] = []
    
    for idx, (exp_path, obs_horizon) in enumerate(zip(experiment_paths, observation_horizons)):
        exp_name = experiment_names[idx] if experiment_names else str(exp_path)
        results = collect_best_results(exp_path)
        if results:
            experiment_results.append((obs_horizon, results, exp_name))
            print(f"\n[{exp_name}] Loaded {len(results)} action horizons for observation_horizon={obs_horizon}")
        else:
            print(f"Warning: No valid results found for {exp_path} ({exp_name}, obs_horizon={obs_horizon})")
    
    # Organize data by effective memory
    effective_memory_data: Dict[int, List[tuple[float, HorizonResult]]] = {
        em: [] for em in effective_memories
    }
    
    for effective_memory in effective_memories:
        print(f"\nEffective Memory = {effective_memory}:")
        
        for obs_horizon, results, exp_name in experiment_results:
            # Calculate required action horizon
            required_action_horizon = effective_memory - obs_horizon
            
            if required_action_horizon <= 0:
                print(f"  [{exp_name}] obs_horizon={obs_horizon}: required action_horizon={required_action_horizon} (invalid, skipping)")
                continue
            
            # Look for this action horizon in the results
            matching_result = None
            for result in results:
                if result.horizon == required_action_horizon:
                    matching_result = result
                    break
            
            if matching_result:
                effective_memory_data[effective_memory].append(
                    (required_action_horizon, matching_result)
                )
                print(
                    f"  [{exp_name}] obs_horizon={obs_horizon} -> action_horizon={required_action_horizon}: "
                    f"success_rate={matching_result.success_rate:.3f} ({matching_result.num_trials} trials)"
                )
            else:
                available = [r.horizon for r in results]
                print(
                    f"  [{exp_name}] obs_horizon={obs_horizon} -> action_horizon={required_action_horizon}: NOT FOUND "
                    f"(available: {available})"
                )
        
        # Sort by action horizon
        effective_memory_data[effective_memory].sort(key=lambda x: x[0])
    
    return effective_memory_data


def make_plot(
    effective_memory_data: Dict[int, List[tuple[float, HorizonResult]]],
    plot_name: Optional[str] = None,
):
    """
    Create interactive 3D surface plot using Plotly.
    
    Args:
        effective_memory_data: Dictionary mapping effective_memory -> list of (action_horizon, result) tuples
        plot_name: Optional title for the plot
        
    Returns:
        Plotly Figure object (interactive HTML-compatible)
    """
    import plotly.graph_objects as go
    from scipy.interpolate import griddata
    
    # Filter out empty effective memories
    effective_memories = sorted([
        em for em, data in effective_memory_data.items() if data
    ])
    
    if not effective_memories:
        raise RuntimeError("No data points found for any effective memory value")
    
    # Collect all action horizons
    all_action_horizons = set()
    for data_points in effective_memory_data.values():
        all_action_horizons.update(action_horizon for action_horizon, _ in data_points)
    all_action_horizons = sorted(all_action_horizons)
    
    # Create a mapping from (action_horizon, effective_memory) -> success_rate
    data_map = {}
    for effective_memory, data_points in effective_memory_data.items():
        for action_horizon, result in data_points:
            data_map[(action_horizon, effective_memory)] = result.success_rate
    
    # Build 2D grid for surface plot
    X = []  # Action horizons
    Y = []  # Effective memories
    Z = []  # Success rates
    
    for em in effective_memories:
        for ah in all_action_horizons:
            if (ah, em) in data_map:
                X.append(ah)
                Y.append(em)
                Z.append(data_map[(ah, em)])
    
    if not X:
        raise RuntimeError("No valid data points found for plotting")
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    # Create regular grid for interpolation
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate Z values on the grid
    Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic', fill_value=np.nan)
    
    # Create the surface plot
    surface = go.Surface(
        x=Xi,
        y=Yi,
        z=Zi,
        colorscale='Viridis',
        opacity=0.8,
        name='Interpolated Surface',
        colorbar=dict(title='Success Rate', titleside='right', tickmode='linear', tick0=0, dtick=0.1),
    )
    
    # Create scatter plot for actual data points
    scatter = go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode='markers',
        marker=dict(
            size=5,
            color=Z,
            colorscale='Viridis',
            opacity=0.9,
            line=dict(color='white', width=0.5),
        ),
        name='Actual Data',
        hovertemplate='Action Horizon: %{x}<br>Effective Memory: %{y}<br>Success Rate: %{z:.3f}<extra></extra>',
    )
    
    # Create figure
    fig = go.Figure(data=[surface, scatter])
    
    # Update layout
    if plot_name is not None:
        title = plot_name
    else:
        title = "Success Rate vs Action Horizon and Effective Memory"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial, sans-serif')),
        scene=dict(
            xaxis=dict(title='Action Horizon (steps)', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Effective Memory', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='Success Rate', range=[0, 1], backgroundcolor="rgb(230, 230,230)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        width=1000,
        height=800,
    )
    
    return fig


def main() -> None:
    description = "Create a success-rate comparison plot for fixed effective memory values."
    parser = parse_args(description)  # Base arguments
    
    # Script-specific arguments
    parser.add_argument(
        "--observation-horizons",
        type=int,
        nargs="+",
        required=True,
        help="Observation horizon value for each experiment path (must match number of paths).",
    )
    parser.add_argument(
        "--effective-memories",
        type=int,
        nargs="+",
        required=True,
        help="Effective memory values to plot (action_horizon + observation_horizon).",
    )
    args = parser.parse_args()
    
    # Validate inputs
    experiment_paths = args.experiment_path if isinstance(args.experiment_path, list) else [args.experiment_path]
    observation_horizons = args.observation_horizons
    effective_memories = sorted(args.effective_memories)
    experiment_names = args.experiment_name
    if len(experiment_names) != len(experiment_paths):
        raise ValueError(
            f"Number of experiment names ({len(experiment_names)}) must match "
            f"number of experiment paths ({len(experiment_paths)})"
        )

    if len(observation_horizons) != len(experiment_paths):
        raise ValueError(
            f"Number of observation horizons ({len(observation_horizons)}) must match "
            f"number of experiment paths ({len(experiment_paths)})"
        )
    
    print(f"\nExperiment paths: {len(experiment_paths)}")
    print(f"Experiment names: {experiment_names}")
    print(f"Observation horizons: {observation_horizons}")
    print(f"Effective memories to evaluate: {effective_memories}")
    
    # Collect data
    effective_memory_data = collect_effective_memory_data(
        experiment_paths,
        observation_horizons,
        effective_memories,
        experiment_names,
    )
    
    # Create interactive 3D plot
    print("\nCreating interactive 3D plot...")
    fig = make_plot(effective_memory_data, plot_name=args.plot_name)
    
    # Save as HTML
    if args.output is not None:
        # Change extension to .html if it's not already
        output_path = args.output
        if output_path.suffix.lower() not in ['.html', '.htm']:
            output_path = output_path.with_suffix('.html')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"\nSaved interactive HTML figure to {output_path}")
        print("Download this file and open it in your web browser to explore the 3D plot.")
    else:
        print("\nWarning: No output path specified. Use --output to save the plot.")
    
    if args.show:
        print("\nNote: --show is not supported. Open the HTML file in your web browser instead.")


if __name__ == "__main__":
    main()

