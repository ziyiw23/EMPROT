#!/bin/bash

#SBATCH --job-name=train_emprot_config
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --output=output/logs/ablations/%x_%j.out
#SBATCH --error=output/logs/ablations/%x_%j.err


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

CONFIG_NAME=${1:-"residue_centric_f3.yaml"}

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

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"output/checkpoints/$RUN_NAME"}

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
    --config_dir "configs" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --run_name "$RUN_NAME"

echo ""
echo " EMPROT training completed!"
