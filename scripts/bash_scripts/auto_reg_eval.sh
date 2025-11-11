#!/bin/bash

#SBATCH --job-name=AR_eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
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
export PLOT_STEP_ATTN=${PLOT_STEP_ATTN:-0}
export ATTN_STEP=${ATTN_STEP:-0}
export PLOT_STEPS_GRID=${PLOT_STEPS_GRID:-0}
export ATTN_STEPS=${ATTN_STEPS:-}
export ATTN_RANDOM_K=${ATTN_RANDOM_K:-5}
export ANALYZE_STEP_ATTENTION=${ANALYZE_STEP_ATTENTION:-0}

# python scripts/autoregressive_eval.py \
#   --model_path checkpoints/hybrid_ctx_cls_no_bias/best_model_epoch_113.pt \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir /scratch/groups/rbaltman/ziyiw23/EMPROT/evaluation_results/hybrid_ctx_cls_no_bias/autoregressive_eval \
#   --context_frames 5 \
#   --stride 1 \
#   --time_step 0.2 \
#   --device cuda \
#   --plot_trajectories 1 \
#   --max_residues_plot 50 \
#   --horizon 30 \
#   --tica_lag 1

# python scripts/autoregressive_eval.py \
#   --model_path checkpoints/hybrid_ctx_cls/best_model_epoch_193.pt \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir /scratch/groups/rbaltman/ziyiw23/EMPROT/evaluation_results/hybrid_ctx_cls/autoregressive_eval \
#   --context_frames 5 \
#   --stride 1 \
#   --time_step 0.2 \
#   --device cuda \
#   --plot_trajectories 1 \
#   --max_residues_plot 50 \
#   --horizon 30 \
#   --tica_lag 1

# python scripts/autoregressive_eval.py \
#   --ckpt checkpoints/context_cbce_unlike_cace/best_model_epoch_14.pt \
#   --data_root /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --split train \
#   --recent_full_frames 5 \
#   --context_latents 32 \
#   --time_start 5 \
#   --time_steps 500 \
#   --residue_select random \
#   --k_residues 5 \
#   --seed 42 \
#   --batch_size 1 \
#   --device cuda \
#   --output_dir /oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/evaluation_results/context_cbce_unlike_cace/autoregressive_eval

# python scripts/autoregressive_eval.py \
#   --ckpt checkpoints/context_cbce_unlike_cace/best_model_epoch_14.pt --data_root /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ --split val \
#   --time_start 5 --time_steps 250 --output_dir output/evaluation_results/context_cbce_unlike_cace/sampling_decode \
#   --decode_mode sample --temperature 0.90 --top_p 0.9\
#   --temp_anneal_gamma 1.0 --min_temperature 0.9 --copy_bias 0.0 --min_dwell 1 \
#   --device cuda --seed 42 --residue_select random --context_latents 32 \
#   --k_residues 12 \
#   --plot_distributions

# python scripts/autoregressive_eval.py \
#   --ckpt checkpoints/2KL_1CE/best_model_epoch_0.pt --data_root /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ --split val \
#   --time_start 5 --time_steps 50 --output_dir output/evaluation_results/depth_ctx_teacher_forcing/autoregressive_eval \
#   --decode_mode sample --temperature 0.90 --top_p 0.9\
#   --temp_anneal_gamma 1.0 --min_temperature 0.9 --copy_bias 0.0 --min_dwell 1 \
#   --device cuda --seed 42 --residue_select random --context_latents 32 \
#   --k_residues 12 \
#   --plot_distributions

#
# Minimal, parameterized invocation (set envs or rely on defaults)
#   CKPT: checkpoint path (e.g., output/checkpoints/<run_name>/best.pt)
#   DATA_ROOT: LMDB root directory
#   SPLIT: train|val|test (default: val)
#   T_START: rollout start index (default: 500)
#   T_STEPS: number of predicted steps (default: 100)
#   K_RECENT: K recent full-res frames (default: 8)
#   K_RES: number of residues to plot (default: 5)
#   RES_MODE: random|most_change|uniform (default: most_change)
#   OUT_DIR: output directory

# CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/cta_sharedkv_K5_nolatents/best.pt}
# CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/emprot_f1_smoothed_jsaux/best.pt}
CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/st_gumbel_F8/best.pt}
# CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/cta_sharedkv_K1_f1_jsaux/best.pt}
# CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/cta_sharedkv_K10_nolatents/best.pt}
# CKPT=${CKPT:-/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/cta_persourcekv_K5_with_latents/best.pt}
DATA_ROOT=${DATA_ROOT:-/scratch/groups/rbaltman/ziyiw23/traj_embeddings}
OUT_DIR=${OUT_DIR:-""}
SPLIT=${SPLIT:-test}
T_START=${T_START:-500}
T_STEPS=${T_STEPS:-100}
K_RECENT=${K_RECENT:-5}
K_RES=${K_RES:-5}
RES_MODE=${RES_MODE:-most_change}
PROTEIN_ID=${PROTEIN_ID:-}

# Sampling / decode controls (pure nucleus sampling)
DECODE_MODE=${DECODE_MODE:-sample}
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-1}
HIST_TOPK=${HIST_TOPK:-30}

# Simple nucleus default

# Optional single-step attention visualization
PLOT_STEP_ATTN=${PLOT_STEP_ATTN:-0}
ATTN_STEP=${ATTN_STEP:-0}

PLOT_STEP_ATTN=1 
ATTN_STEP=5
TEMPERATURE=0.75
TOP_P=1

echo "[AR-EVAL] ckpt=$CKPT"
echo "[AR-EVAL] data_root=$DATA_ROOT split=$SPLIT"
echo "[AR-EVAL] t_start=$T_START t_steps=$T_STEPS K=$K_RECENT k_res=$K_RES mode=$RES_MODE"
if [ -n "$OUT_DIR" ]; then
  echo "[AR-EVAL] out_dir=$OUT_DIR"
fi
echo "[AR-EVAL] decode=$DECODE_MODE temp=$TEMPERATURE top_p=$TOP_P hist_topk=$HIST_TOPK"
echo "[AR-EVAL] plots: plot_hist=1 step_raw=$PLOT_STEP_ATTN step=$ATTN_STEP steps_grid=$PLOT_STEPS_GRID steps='$ATTN_STEPS' rand_k=$ATTN_RANDOM_K analyze=$ANALYZE_STEP_ATTENTION"

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
  --device cuda
)

if [ -n "$PROTEIN_ID" ]; then
  ARGS+=( --protein_id "$PROTEIN_ID" )
fi

if [ -n "$OUT_DIR" ]; then
  ARGS+=( --output_dir "$OUT_DIR" )
fi

if [ "$PLOT_STEP_ATTN" = "1" ]; then
  ARGS+=( --plot_step_attn --attn_step "$ATTN_STEP" )
fi

if [ "$PLOT_STEPS_GRID" = "1" ]; then
  ARGS+=( --plot_steps_grid )
  if [ -n "$ATTN_STEPS" ]; then
    ARGS+=( --attn_steps "$ATTN_STEPS" )
  else
    ARGS+=( --attn_steps random --attn_random_k "$ATTN_RANDOM_K" )
  fi
fi

if [ "$ANALYZE_STEP_ATTENTION" = "1" ]; then
  ARGS+=( --analyze_step_attention )
fi


python "${ARGS[@]}"
