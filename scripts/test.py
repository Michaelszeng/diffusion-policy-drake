import os
import pickle
import numpy as np
from pathlib import Path

def reconstruct_summary_pkl_from_txt(output_dir):
    """Reconstruct summary.pkl from summary.txt for resume capability."""
    summary_txt_path = os.path.join(output_dir, "summary.txt")
    
    summary = {
        "successful_trials": [],
        "trial_times": [],
        "initial_conditions": [],
        "final_error": [],
        "trial_result": [],
        "total_eval_sim_time": 0.0,
        "total_eval_wall_time": 0.0,
    }
    
    try:
        with open(summary_txt_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ❌ Failed to read {summary_txt_path}: {e}")
        return False
    
    if not lines:
        print(f"  ❌ {summary_txt_path} is empty")
        return False
    
    trial_idx = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for trial start (skips header lines automatically)
        if line.startswith("Trial "):
            try:
                # Parse trial block
                i += 1  # Skip "--------------------"
                
                # Result
                i += 1
                result_line = lines[i].strip()
                result_value = result_line.split(": ", 1)[1]
                summary["trial_result"].append(result_value)
                if result_value == "success":
                    summary["successful_trials"].append(trial_idx)
                
                # Trial time
                i += 1
                trial_time_line = lines[i].strip()
                trial_time = float(trial_time_line.split(": ", 1)[1])
                summary["trial_times"].append(trial_time)
                
                # Initial slider pose
                i += 1
                initial_pose_line = lines[i].strip()
                initial_pose_str = initial_pose_line.split(": ", 1)[1]
                # Parse numpy array string like "[ 0.6418 -0.1046 -2.875 ]"
                initial_pose = np.fromstring(initial_pose_str.strip("[]"), sep=" ")
                summary["initial_conditions"].append(initial_pose)
                
                # Final pusher error
                i += 1
                pusher_error_line = lines[i].strip()
                pusher_error_str = pusher_error_line.split(": ", 1)[1]
                pusher_error = np.fromstring(pusher_error_str.strip("[]"), sep=" ")
                
                # Final slider error
                i += 1
                slider_error_line = lines[i].strip()
                slider_error_str = slider_error_line.split(": ", 1)[1]
                slider_error = np.fromstring(slider_error_str.strip("[]"), sep=" ")
                
                summary["final_error"].append({
                    "pusher_error": pusher_error,
                    "slider_error": slider_error,
                })
                
                trial_idx += 1
            except (IndexError, ValueError) as e:
                print(f"  ⚠️  Warning: Failed to parse trial {trial_idx + 1}: {e}")
                break
        
        i += 1
    
    if trial_idx == 0:
        print(f"  ❌ No valid trial data found in {summary_txt_path}")
        return False
    
    # Save reconstructed summary
    summary_pkl_path = os.path.join(output_dir, "summary.pkl")
    try:
        with open(summary_pkl_path, "wb") as f:
            pickle.dump(summary, f)
        print(f"  ✓ Reconstructed summary.pkl from {len(summary['trial_times'])} trials")
        return True
    except Exception as e:
        print(f"  ❌ Failed to save summary.pkl: {e}")
        return False


def find_and_reconstruct_missing_pkl_files(base_dirs):
    """Find all directories with summary.txt but missing summary.pkl and reconstruct them."""
    
    total_found = 0
    total_reconstructed = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"⚠️  Directory not found: {base_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Scanning: {base_dir}")
        print('='*80)
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_dir):
            # Check if this directory has summary.txt
            if "summary.txt" in files:
                summary_pkl = os.path.join(root, "summary.pkl")
                
                # Check if summary.pkl is missing
                if not os.path.exists(summary_pkl):
                    total_found += 1
                    rel_path = os.path.relpath(root, base_dir)
                    print(f"\n[{total_found}] Found: {rel_path}")
                    
                    # Reconstruct
                    if reconstruct_summary_pkl_from_txt(root):
                        total_reconstructed += 1
    
    return total_found, total_reconstructed


if __name__ == "__main__":
    # Directories to scan
    base_directories = [
        "eval/sim_sim/dynamic_0_01/v3",
        "eval/sim_sim/dynamic_0_02/v3",
        "eval/sim_sim/dynamic_0_005/v3",
    ]
    
    print("Starting reconstruction of missing summary.pkl files...")
    print("="*80)
    
    found, reconstructed = find_and_reconstruct_missing_pkl_files(base_directories)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total directories with missing summary.pkl: {found}")
    print(f"Successfully reconstructed: {reconstructed}")
    if found > reconstructed:
        print(f"Failed: {found - reconstructed}")
    print("="*80)