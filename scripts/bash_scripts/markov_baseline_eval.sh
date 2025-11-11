#!/bin/bash

#SBATCH --job-name=markov_baseline
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --output=output/logs/eval/%x_%j.out
#SBATCH --error=output/logs/eval/%x_%j.err

set -euo pipefail

echo "   EMPROT:"
echo "   Markov baseline evaluation"
echo "   Date: $(date)"
echo ""

# --- Load modules and environment ---
module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

# --- Optional W&B (no effect if not used) ---
export WANDB_MODE="offline"

# --- Path Setup ---
export PYTHONPATH="/oak/stanford/groups/rbaltman/ziyiw23/EMPROT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# --- Arguments (env-overridable) ---
DATA_ROOT=${DATA_ROOT:-"/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/data"}
META_CSV=${META_CSV:-"$DATA_ROOT/traj_metadata.csv"}
EVAL_SPLIT=${EVAL_SPLIT:-"test"}
PROTEIN_ID=${PROTEIN_ID:-""}
T_START=${T_START:-0}
T_STEPS=${T_STEPS:-200}
MAX_BATCHES=${MAX_BATCHES:-500}
NUM_FRAMES=${NUM_FRAMES:-5}
STRIDE=${STRIDE:-1}
FUTURE_H=${FUTURE_H:-1}
SEED=${SEED:-42}
TOPK=${TOPK:-256}
ALPHA=${ALPHA:-1e-6}
OUT_DIR=${OUT_DIR:-""}

echo " [Markov Baseline] data_root=$DATA_ROOT eval_split=$EVAL_SPLIT protein_id=${PROTEIN_ID:-<first>} t_start=$T_START steps=$T_STEPS topk=$TOPK max_batches=$MAX_BATCHES"

ARGS=(
  scripts/baselines/markov_baseline.py
  --data_dir "$DATA_ROOT"
  --metadata_path "$META_CSV"
  --eval_split "$EVAL_SPLIT"
  --time_start "$T_START"
  --time_steps "$T_STEPS"
  --max_batches "$MAX_BATCHES"
  --num_full_res_frames "$NUM_FRAMES"
  --stride "$STRIDE"
  --future_horizon "$FUTURE_H"
  --seed "$SEED"
  --topk "$TOPK"
  --alpha "$ALPHA"
)

if [[ -n "${PROTEIN_ID}" ]]; then
  ARGS+=(--protein_id "$PROTEIN_ID")
fi

if [[ -n "${OUT_DIR}" ]]; then
  ARGS+=(--output_dir "$OUT_DIR")
fi

python "${ARGS[@]}"
