#!/bin/bash
#SBATCH --job-name=gcs_data_collect
#SBATCH --account=locomotion
#SBATCH --partition=locomotion-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Run 1 task per node (which spawns the accelerate launcher)
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH --cpus-per-task=40          # CPUs per task
#SBATCH --mem=384G                  # Memory per node
#SBATCH --time=24:00:00             # Time limit
#SBATCH --output=logs/slurm-%j.out  # Standard output log
#SBATCH --error=logs/slurm-%j.err   # Standard error log
#SBATCH --requeue                   # Re-queue the job if it fails/is pre-empted

# Setup:
# python3 -m venv env --without-pip
# curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
# python3 /tmp/get-pip.py --no-warn-script-location
# pip install -r requirements.txt

# Usage: sbatch submit_gcs_data_collection.sh

# ============================================================================
# Configuration
# ============================================================================
NUM_RUNS=500
SEED=3
ZARR_PATH="/home/michzeng/diffusion-policy/data/diffusion_experiments/planar_pushing/sim_sim_tee_GCS_data_carbon_large.zarr"

DATE=$(date +"%Y.%m.%d")
TIME=$(date +"%H.%M.%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment Setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
# Load venv environment
source env/bin/activate

export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=========================================="
echo "GCS Data Collection"
echo "  NUM_RUNS: $NUM_RUNS"
echo "  SEED: $SEED"
echo "  ZARR_PATH: $ZARR_PATH"
echo "=========================================="

python scripts/run_sim_sim_gcs_planner.py \
    collect_data=true \
    SEED=$SEED \
    multi_run_config.num_runs=$NUM_RUNS \
    multi_run_config.num_trials_to_record=0 \
    use_realtime=false \
    save_gcs_videos=false \
    data_collection_config.zarr_path="$ZARR_PATH"
