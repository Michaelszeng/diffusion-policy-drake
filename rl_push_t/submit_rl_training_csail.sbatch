#!/bin/bash
#SBATCH --job-name=rl_push_t_training
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx2080ti
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage: sbatch rl_push_t/submit_rl_training_csail.sbatch

# ============================================================================
# Training hyperparameters
# ============================================================================
N_ENVS=76
TOTAL_TIMESTEPS=500000000
LR=1e-4
NUM_STEPS=200  # Full episode length for each step
NUM_MINIBATCHES=25
UPDATE_EPOCHS=4
EVAL_FREQ=100
NUM_EVAL_EPISODES=10
EXP_NAME="push_t_ppo"
# ============================================================================

DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment Setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
source env/bin/activate

mkdir -p logs

# --- WANDB Authentication ---
export WANDB_USERNAME="michzeng"
SECRETS_FILE="$SLURM_SUBMIT_DIR/.secrets"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
    echo "Loaded secrets from $SECRETS_FILE"
else
    echo "ERROR: $SECRETS_FILE not found. Create it from .secrets.template before submitting." >&2
    exit 1
fi
echo "WANDB_USERNAME and WANDB_API_KEY set for authentication."

export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "N_ENVS:           $N_ENVS"
echo "TOTAL_TIMESTEPS:  $TOTAL_TIMESTEPS"
echo "LR:               $LR"
echo "NUM_STEPS:        $NUM_STEPS"
echo "NUM_MINIBATCHES:  $NUM_MINIBATCHES"
echo "UPDATE_EPOCHS:    $UPDATE_EPOCHS"
echo "EXP_NAME:         $EXP_NAME"
echo "=========================================="

python rl_push_t/ppo.py \
    --exp_name=$EXP_NAME \
    --n_envs=$N_ENVS \
    --total_timesteps=$TOTAL_TIMESTEPS \
    --lr=$LR \
    --num_steps=$NUM_STEPS \
    --num_minibatches=$NUM_MINIBATCHES \
    --update_epochs=$UPDATE_EPOCHS \
    --eval_freq=$EVAL_FREQ \
    --num_eval_episodes=$NUM_EVAL_EPISODES \
    --cuda
