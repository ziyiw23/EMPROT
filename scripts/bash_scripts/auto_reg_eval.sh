#!/bin/bash

#SBATCH --job-name=AR_eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --output=output/logs/autoregressive_eval.out
#SBATCH --error=output/logs/autoregressive_eval.err

# Fail fast and be explicit
set -euo pipefail
mkdir -p logs

# --- Load modules and environment ---
module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

# --- Path Setup ---
export PYTHONPATH="/oak/stanford/groups/rbaltman/ziyiw23/EMPROT:$(pwd):${PYTHONPATH:-}"

CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/ablations/sweep_K5_D768_future_horizon1_L8_CE1.0_JS0.0_SS0.5_5393/best.pt}
DATA_ROOT=${DATA_ROOT:-/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings}
OUT_DIR=${OUT_DIR:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/evaluation_results/sweep_K5_D768_future_horizon1_L8_CE1.0_JS0.0_SS0.5_5393/autoregressive_eval}
SPLIT=${SPLIT:-train}
T_START=${T_START:-500}
T_STEPS=${T_STEPS:-100}
K_RECENT=${K_RECENT:-5}
K_RES=${K_RES:-5}
RES_MODE=${RES_MODE:-random}
PROTEIN_ID=${PROTEIN_ID:-}

# Sampling / decode controls (pure nucleus sampling)
DECODE_MODE=${DECODE_MODE:-sample}
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-0.98}
HIST_TOPK=${HIST_TOPK:-30}
MARKOV_CKPT=${MARKOV_CKPT:-"/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/markov_ckpts/markov_${SPLIT}.pkl"}
MARKOV_LABEL=${MARKOV_LABEL:-"Markov baseline"}

echo "[AR-EVAL] ckpt=$CKPT"
echo "[AR-EVAL] data_root=$DATA_ROOT split=$SPLIT"
echo "[AR-EVAL] t_start=$T_START t_steps=$T_STEPS K=$K_RECENT k_res=$K_RES mode=$RES_MODE"
if [ -n "$OUT_DIR" ]; then
  echo "[AR-EVAL] out_dir=$OUT_DIR"
fi
echo "[AR-EVAL] decode=$DECODE_MODE temp=$TEMPERATURE top_p=$TOP_P hist_topk=$HIST_TOPK markov_ckpt=${MARKOV_CKPT:-<none>}"
echo "[AR-EVAL] extras: plot_hist=1 markov_label=$MARKOV_LABEL"

ARGS=(
  scripts/attn_rollout_min.py
  --ckpt "$CKPT"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --time_start "$T_START"
  --time_steps "$T_STEPS"
  --recent_full_frames "$K_RECENT"
  --k_residues "$K_RES"
  --residue_select "$RES_MODE"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --hist_topk "$HIST_TOPK"
  --plot_hist
  --plot_corr
  --device cuda
)

if [ -n "$PROTEIN_ID" ]; then
  ARGS+=( --protein_id "$PROTEIN_ID" )
fi

if [ -n "$OUT_DIR" ]; then
  ARGS+=( --output_dir "$OUT_DIR" )
fi

if [ -n "$MARKOV_CKPT" ]; then
  ARGS+=( --markov_ckpt "$MARKOV_CKPT" --markov_label "$MARKOV_LABEL" )
fi


python "${ARGS[@]}"
