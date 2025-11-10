from __future__ import annotations


import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

NAVY = "#1f3b6f"
GRID_COLOR = "#bdbdbd"

def generate_color_palette(num_colors: int) -> List[str]:
    """
    Generate a color palette by smoothly interpolating between green and red.
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of hex color strings
    """
    if num_colors <= 0:
        return []
    elif num_colors == 1:
        return ["#00ff00"]  # Pure green
    
    # Define start (green) and end (red) colors in RGB
    start_rgb = (0, 255, 0)    # Green
    end_rgb = (255, 0, 0)     # Red
    
    colors = []
    for i in range(num_colors):
        # Linear interpolation between start and end
        t = i / (num_colors - 1) if num_colors > 1 else 0
        
        # Interpolate each RGB component
        r = int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))
        g = int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1]))
        b = int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2]))
        
        # Convert to hex
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append(hex_color)
    
    return colors

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class HorizonResult:
    """Container holding the best checkpoint statistics for a given horizon."""

    horizon: float
    success_rate: float
    num_trials: int
    checkpoint_dir: Path
    num_checkpoints_available: int = 1  # Number of checkpoints evaluated for this horizon


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Path(s) to experiment directories that contain T_a_<horizon> sub-folders. "
            "The *first* path is treated as the baseline; the others are compared against it."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Name(s) of the experiment(s) for the legend labels. If not provided, "
            "labels will be derived from directory names. Number of names should match "
            "number of experiment paths."
        ),
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="Relative Action Horizon Performance",
        help="Title for the plot (optional).",
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
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    parser.add_argument("--all-checkpoints", action="store_true", help="Plot all checkpoints instead of just the best per horizon.")
    return parser.parse_args()


def iter_action_horizon_dirs(experiment_path: Path) -> Iterable[Path]:
    """Yield sub-directories of *experiment_path* that match the ``T_a_*`` pattern."""

    for candidate in sorted(experiment_path.iterdir()):
        if candidate.is_dir() and candidate.name.startswith("T_a_"):
            yield candidate


def parse_horizon_from_dirname(dirname: str) -> float:
    """Extract the numeric horizon value from a directory name of the form ``T_a_<h>``."""

    _, _, value = dirname.partition("T_a_")
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(f"Could not parse horizon from '{dirname}'.") from err


def load_success_rate(summary_path: Path) -> Tuple[float, int]:
    """Return (success_rate, num_trials) extracted from a ``summary.pkl`` file."""

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
    """Return the *HorizonResult* corresponding to the best checkpoint in *horizon_dir*."""

    best_result: Optional[HorizonResult] = None
    seen_checkpoints = set()  # Track unique checkpoint directories

    for summary_path in sorted(horizon_dir.glob("**/summary.pkl")):
        success_rate, total_trials = load_success_rate(summary_path)
        if math.isnan(success_rate):
            continue

        # Get the checkpoint directory name (first subdirectory under horizon_dir)
        relative_path = summary_path.relative_to(horizon_dir)
        checkpoint_dir = horizon_dir / relative_path.parts[0]
        
        # Track unique checkpoints
        seen_checkpoints.add(checkpoint_dir.name)
        
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

    # Update the final count of checkpoints available
    if best_result is not None:
        best_result.num_checkpoints_available = len(seen_checkpoints)

    return best_result


def collect_best_results(experiment_path: Path) -> List[HorizonResult]:
    """Return a list of *HorizonResult* objects for each horizon in *experiment_path*."""

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


def collect_all_checkpoint_results(experiment_path: Path) -> Dict[str, List[HorizonResult]]:
    """Return all checkpoints grouped by checkpoint name across all horizons."""
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")
    
    checkpoint_results: Dict[str, List[HorizonResult]] = {}
    
    for horizon_dir in iter_action_horizon_dirs(experiment_path):
        horizon_value = parse_horizon_from_dirname(horizon_dir.name)
        
        for summary_path in sorted(horizon_dir.glob("**/summary.pkl")):
            success_rate, total_trials = load_success_rate(summary_path)
            if math.isnan(success_rate):
                continue
            
            relative_path = summary_path.relative_to(horizon_dir)
            checkpoint_dir = horizon_dir / relative_path.parts[0]
            checkpoint_name = checkpoint_dir.name
            
            result = HorizonResult(horizon_value, success_rate, total_trials, checkpoint_dir)
            
            if checkpoint_name not in checkpoint_results:
                checkpoint_results[checkpoint_name] = []
            checkpoint_results[checkpoint_name].append(result)
    
    # Sort each checkpoint's results by horizon
    for results in checkpoint_results.values():
        results.sort(key=lambda res: res.horizon)
    
    return checkpoint_results
