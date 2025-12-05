#!/bin/bash
#SBATCH --job-name=align_lookup
#SBATCH --output=output/logs/align/align_%j.out
#SBATCH --error=output/logs/align/align_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=rbaltman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

CONFIG_NAME=${1:-"residue_centric_full.yaml"}
SAVE_PATH=${2:-"output/checkpoints/aligned_lookup.pt"}

echo "Starting Offline Alignment Training..."
echo "Config: $CONFIG_NAME"
echo "Save Path: $SAVE_PATH"

mkdir -p output/logs/align
mkdir -p output/checkpoints

python scripts/train_lookup_table.py \
    --config "$CONFIG_NAME" \
    --config_dir "configs" \
    --save_path "$SAVE_PATH" \
    --batch_size 128 \
    --lr 1e-3 \
    --epochs 1

echo "Alignment Training Finished"

