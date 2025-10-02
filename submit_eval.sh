#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment."
source /etc/profile
module load anaconda/2023b

# Assume current directory is diffusion-policy-drake
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline."
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export cHYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running eval code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

python scripts/launch_eval.py \
    --csv-path config/all_action_horizons_launch_eval.txt \
    --max-concurrent-jobs 8 \
    --num-trials 50 50 100 \
    --drop-threshold 0.05