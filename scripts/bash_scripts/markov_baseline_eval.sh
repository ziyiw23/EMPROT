#!/bin/bash

#SBATCH --job-name=markov_baseline
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --output=output/logs/eval/%x_%j.out
#SBATCH --error=output/logs/eval/%x_%j.err

set -euo pipefail

echo "   EMPROT:"
echo "   Markov baseline training"
echo "   Date: $(date)"
echo ""

# --- Load modules and environment ---
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

# --- Optional W&B (no effect if not used) ---
export WANDB_MODE="offline"

# --- Path Setup ---
export PYTHONPATH="/oak/stanford/groups/rbaltman/ziyiw23/EMPROT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# --- Arguments (env-overridable) ---
DATA_ROOT=${DATA_ROOT:-"/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings"}
META_CSV=${META_CSV:-"/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv"}
FIT_SPLIT=${FIT_SPLIT:-"train"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_FRAMES=${NUM_FRAMES:-5}
STRIDE=${STRIDE:-1}
FUTURE_H=${FUTURE_H:-1}
MAX_BATCHES=${MAX_BATCHES:-500}
SEED=${SEED:-42}
TOPK=${TOPK:-256}
ALPHA=${ALPHA:-1e-6}
MODEL_CKPT=${MODEL_CKPT:-"/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/markov_ckpts/markov_${FIT_SPLIT}.pkl"}

echo " [Markov Baseline] training_split=$FIT_SPLIT batch_size=$BATCH_SIZE num_frames=$NUM_FRAMES stride=$STRIDE future_h=$FUTURE_H topk=$TOPK alpha=$ALPHA max_batches=$MAX_BATCHES"
echo " [Markov Baseline] checkpoint -> $MODEL_CKPT"

ARGS=(
  scripts/baselines/markov_baseline.py
  --data_dir "$DATA_ROOT"
  --metadata_path "$META_CSV"
  --fit_split "$FIT_SPLIT"
  --batch_size "$BATCH_SIZE"
  --num_full_res_frames "$NUM_FRAMES"
  --stride "$STRIDE"
  --future_horizon "$FUTURE_H"
  --max_batches "$MAX_BATCHES"
  --seed "$SEED"
  --topk "$TOPK"
  --alpha "$ALPHA"
  --model_ckpt "$MODEL_CKPT"
)

python "${ARGS[@]}"
