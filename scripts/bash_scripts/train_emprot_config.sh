#!/bin/bash

#SBATCH --job-name=v2_st_gumbel_F8
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --output=output/logs/training/%x_%j.out
#SBATCH --error=output/logs/training/%x_%j.err


echo "   EMPROT:"
echo "   ablation studies using YAML configs"
echo "   Date: $(date)"
echo ""

# --- Load modules and environment ---
module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

# --- Weights & Biases (non-interactive auth) ---
# Prefer env override from sbatch: --export=ALL,WANDB_API_KEY=...  
# Falls back to the default below if not provided.
export WANDB_API_KEY="${WANDB_API_KEY:-795763e9b7eb6ed670708ab2c2f392e9b25b0af2}"
# Optional: ensure online mode (not offline)
export WANDB_MODE="online"


# --- Path Setup ---
export PYTHONPATH="/oak/stanford/groups/rbaltman/ziyiw23/EMPROT:$PYTHONPATH"
# Reduce CUDA fragmentation when near OOM
# Use PyTorch segmented allocator in near-OOM regimes (requires True/False)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# ============================================================================
# CONFIGURATION SELECTION
# ============================================================================
# Choose your experiment configuration by setting CONFIG_NAME
# All hyperparameters are defined in the corresponding YAML file

# === BASELINE EXPERIMENTS ===
# CONFIG_NAME="baseline_mse_only.yaml"          # MSE regression only
# CONFIG_NAME="depth_context_hybrid.yaml"       # distributional hybrid context classification
# Minimal, classification-only config compatible with new trainer
# CONFIG_NAME="emprot.yaml"

# Change-aware sampling + loss weighting config
# CONFIG_NAME="emprot_changeaware.yaml"
# CONFIG_NAME=emprot_hist.yaml
# CONFIG_NAME="cta_sharedkv_K10_nolatents.yaml"
# CONFIG_NAME="cta_sharedkv_K5_nolatents.yaml"
# CONFIG_NAME="cta_persourcekv_K5_with_latents.yaml"
# CONFIG_NAME="emprot_f1_smoothed_jsaux.yaml"
# CONFIG_NAME="cta_sharedkv_K5_f1_jsaux.yaml"
# CONFIG_NAME="cta_sharedkv_K5_f1_jsaux_reco.yaml"
CONFIG_NAME="cta_sharedkv_K5_f8_stgumbel.yaml"
# CONFIG_NAME="hybrid_context_classification.yaml"       # hybrid context classification
# CONFIG_NAME="ctx_single_step_rollout.yaml"
# CONFIG_NAME="depth_context_hybrid_nextk.yaml"       # distributional KL only
# CONFIG_NAME="depth_context_hybrid_teacher_forcing_tuned.yaml"       # distributional KL only

# === ABLATION STUDIES ===
# CONFIG_NAME="amplification_study.yaml"       # Test magnitude amplification
# CONFIG_NAME="curriculum_data.yaml"           # Test data curriculum
# CONFIG_NAME="full_features.yaml"             # All features enabled

# === CUSTOM CONFIG ===
# CONFIG_NAME="my_custom_config.yaml"          # Your custom configuration

echo " Selected Configuration: $CONFIG_NAME"
echo ""

# ============================================================================
# EXECUTE TRAINING
# ============================================================================
echo " Starting EMPROT training with config-based system..."
echo "   All hyperparameters loaded from: configs/$CONFIG_NAME"
echo ""

# Counts precompute removed; loss uses masked CE with optional class weights computed separately

# ============================================================================
#  CHECKPOINT CLEANUP
# ============================================================================
# Extract run_name from config to determine checkpoint directory
RUN_NAME=$(python -c "
import yaml, os
config_name = os.environ.get('CONFIG_NAME', '$CONFIG_NAME')
with open(f'configs/{config_name}', 'r') as f:
    config = yaml.safe_load(f)
    base_config = config.get('base_config')
    if base_config:
        with open(f'configs/{base_config}', 'r') as bf:
            base = yaml.safe_load(bf)
            base.update(config)
            config = base
    print(config.get('experiment', {}).get('run_name', 'default_run'))
" CONFIG_NAME="$CONFIG_NAME")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$APPEND_TIME" ]; then
  RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
fi
echo " Detected run_name: $RUN_NAME"

CHECKPOINT_DIR="output/checkpoints/$RUN_NAME"

if [ -d "$CHECKPOINT_DIR" ]; then
    if [ "$CLEAN" = "1" ]; then
        echo " Cleaning existing checkpoint directory: $CHECKPOINT_DIR"
        rm -rf "$CHECKPOINT_DIR"
        echo " Removed old checkpoints"
    else
        echo " Using existing checkpoint directory (no cleanup): $CHECKPOINT_DIR"
    fi
else
    echo " Creating new checkpoint directory: $CHECKPOINT_DIR"
fi

echo ""

# The config system automatically handles all parameters
python scripts/train_transformer.py \
    --config "$CONFIG_NAME" \
    --config_dir "configs"

echo ""
echo " EMPROT training completed!"
