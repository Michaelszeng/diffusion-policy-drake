#!/bin/bash

# Usage
# LLsub ./submit_run_sim_sim_eval.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_run_sim_sim_eval.sh] Loading modules and virtual environment."
source /etc/profile
module load anaconda/Python-ML-2025a

# Set wandb to offline since Supercloud has no internet access
echo "[submit_run_sim_sim_eval.sh] Setting wandb to offline."
wandb offline

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Expose the diffusion-policy repo to Python without Poetry
export PYTHONPATH="/home/gridsan/mzeng/diffusion-policy-experiments${PYTHONPATH:+:${PYTHONPATH}}"

# Export MOSEK license file
export MOSEKLM_LICENSE_FILE=/home/gridsan/mzeng/mosek.lic
# Expose MOSEK shared library file
export LD_LIBRARY_PATH=/home/gridsan/mzeng/mosek/11.0/tools/platform/linux64x86/bin:$LD_LIBRARY_PATH

# Assume current directory is diffusion-policy-drake
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export HYDRA_FULL_ERROR=1

# Silence LCM error when running on Supercloud
export LCM_DEFAULT_URL=memq://null

# Fix lack of X server when running on Supercloud
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export GALLIUM_DRIVER=llvmpipe
Xvfb "$DISPLAY" -screen 0 1400x900x24 -nolisten tcp > /tmp/xvfb.log 2>&1 &  # silence Xvfb output
xvfb_pid=$!
trap "kill $xvfb_pid" EXIT

echo "[submit_run_sim_sim_eval.sh] Running eval code..."
echo "[submit_run_sim_sim_eval.sh] Date: $DATE"
echo "[submit_run_sim_sim_eval.sh] Time: $TIME"

python scripts/run_sim_sim_eval.py \
    --config-dir=config/sim_config/sim_sim \
    --config-name=gamepad_teleop_carbon \
    'diffusion_policy_config.checkpoint="/home/gridsan/mzeng/diffusion-policy-experiments/data/outputs/planar_pushing/2_obs/checkpoints/latest.ckpt"'
