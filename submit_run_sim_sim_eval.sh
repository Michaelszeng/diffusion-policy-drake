#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment."
source /etc/profile
module load anaconda/Python-ML-2025a

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline."
wandb offline

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Expose the diffusion-policy repo to Python without Poetry
export PYTHONPATH="/home/gridsan/mzeng/diffusion-policy-experiments${PYTHONPATH:+:${PYTHONPATH}}"

# Export MOSEK license file
export MOSEKLM_LICENSE_FILE=/home/gridsan/mzeng/mosek.lic

# Assume current directory is diffusion-policy-drake
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export cHYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running eval code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

python scripts/run_sim_sim_eval.py \
    --config-dir=config/sim_config/sim_sim \
    --config-name=gamepad_teleop_carbon \
    'diffusion_policy_config.checkpoint="/home/gridsan/mzeng/diffusion-policy-experiments/data/outputs/planar_pushing/2_obs/checkpoints/latest.ckpt"'
