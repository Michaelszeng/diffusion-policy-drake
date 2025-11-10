#!/bin/bash

# Usage
# LLsub ./submit_launch_eval.sh -s 20 -g volta:1 -- config/all_action_horizons_launch_eval_supercloud.txt "50,100,350"
# LLsub ./submit_launch_eval.sh -s 20 -g volta:1 -- config/all_action_horizons_launch_eval_supercloud.txt "50,100,350" 10 2 --resume
# Parameters:
#   $1: Config path (default: config/all_action_horizons_launch_eval_supercloud.txt)
#   $2: Num trials, comma-separated (default: "45,55,100")
#   $3: Concurrent jobs per GPU (default: 10)
#   $4: Number of GPUs (default: 2)
#   $5: Additional flags (e.g., --resume) (default: none)

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

# Limit OpenMP threads to prevent thread exhaustion with many concurrent jobs
export OMP_NUM_THREADS=2

# Fix lack of X server when running on Supercloud
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export GALLIUM_DRIVER=llvmpipe
Xvfb "$DISPLAY" -screen 0 1400x900x24 -nolisten tcp > /tmp/xvfb.log 2>&1 &  # silence Xvfb output
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

# Accept comma-separated list (e.g. "50,100,350") and convert to space-separated
DEFAULT_NUM_TRIALS="500"
# If the user supplied a second positional parameter, use it; otherwise use default
RAW_NUM_TRIALS="${2:-$DEFAULT_NUM_TRIALS}"
# Replace commas with spaces so that launch_eval.py receives a space-delimited list
NUM_TRIALS="${RAW_NUM_TRIALS//,/ }"

DEFAULT_CONCURRENT_JOBS_PER_GPU=10
CONCURRENT_JOBS_PER_GPU="${3:-$DEFAULT_CONCURRENT_JOBS_PER_GPU}"

DEFAULT_NUM_GPUS=2
NUM_GPUS="${4:-$DEFAULT_NUM_GPUS}"

# Accept additional flags like --resume
ADDITIONAL_FLAGS="${5:-}"

echo "[submit_launch_eval.sh] Running eval code..."
echo "[submit_launch_eval.sh] Date: $DATE"
echo "[submit_launch_eval.sh] Time: $TIME"
echo "[submit_launch_eval.sh] Config: $CONFIG_PATH"
echo "[submit_launch_eval.sh] Num trials: $NUM_TRIALS"
echo "[submit_launch_eval.sh] Concurrent jobs per GPU: $CONCURRENT_JOBS_PER_GPU"
echo "[submit_launch_eval.sh] Number of GPUs to use: $NUM_GPUS"
echo "[submit_launch_eval.sh] Additional flags: $ADDITIONAL_FLAGS"

# -u option to unbuffer the stdout and stderr outputs
python -u scripts/launch_eval.py \
    --csv-path "$CONFIG_PATH" \
    --max-concurrent-jobs-per-gpu $CONCURRENT_JOBS_PER_GPU \
    --num-gpus $NUM_GPUS \
    --num-trials-per-round $NUM_TRIALS \
    --drop-threshold 0.05 \
    --non-interactive \
    $ADDITIONAL_FLAGS