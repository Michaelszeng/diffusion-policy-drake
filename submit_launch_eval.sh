#!/bin/bash

# Usage
# ./submit_launch_eval.sh [path/to/config.txt]
# Example (LLsub): LLsub ./submit_launch_eval.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_launch_eval.sh] Loading modules and virtual environment."
source /etc/profile
module load anaconda/Python-ML-2025a

# Set wandb to offline since Supercloud has no internet access
echo "[submit_launch_eval.sh] Setting wandb to offline."
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
XKB_DISABLE=1 Xvfb $DISPLAY -screen 0 1400x900x24 -nolisten tcp &
xvfb_pid=$!
trap "kill $xvfb_pid" EXIT

# Read config path from first argument or use default
DEFAULT_CONFIG_PATH="config/all_action_horizons_launch_eval_supercloud.txt"
CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"

# Validate config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[submit_launch_eval.sh] ERROR: Config file not found: $CONFIG_PATH"
    echo "[submit_launch_eval.sh] Provide a valid .txt config path as the first argument."
    exit 1
fi

DEFAULT_CONCURRENT_JOBS=15
CONCURRENT_JOBS="${2:-$DEFAULT_CONCURRENT_JOBS}"

echo "[submit_launch_eval.sh] Running eval code..."
echo "[submit_launch_eval.sh] Date: $DATE"
echo "[submit_launch_eval.sh] Time: $TIME"
echo "[submit_launch_eval.sh] Config: $CONFIG_PATH"
echo "[submit_launch_eval.sh] Concurrent jobs: $CONCURRENT_JOBS"

python scripts/launch_eval.py \
    --csv-path "$CONFIG_PATH" \
    --max-concurrent-jobs $CONCURRENT_JOBS \
    --num-trials 50 50 100 \
    --drop-threshold 0.05 \
    --force