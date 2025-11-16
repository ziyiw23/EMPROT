#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=prep_cluster_ids
#SBATCH --partition=rbaltman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=4-00:00:00
#SBATCH --output=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/prep_cluster_ids/preprocess_cluster_ids.log
#SBATCH --error=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/prep_cluster_ids/preprocess_cluster_ids.err

module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

set -e  # Exit on any error

# --- Configuration ---
DATA_DIR="/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings"
CLUSTER_MODEL_PATH="/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl"
BATCH_SIZE=1000
DEVICE="cuda"

# --- Safety Check ---
echo "========================"
echo "Safety Check:"
echo "========================"
echo "   Data directory: $DATA_DIR"
echo "   Cluster model: $CLUSTER_MODEL_PATH"
echo "   Batch size: $BATCH_SIZE"
echo "   Device: $DEVICE"
echo ""

# Check if files exist
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$CLUSTER_MODEL_PATH" ]; then
    echo "ERROR: Cluster model file does not exist: $CLUSTER_MODEL_PATH"
    exit 1
fi

echo "SUCCESS: All paths verified"
echo ""

# --- Preprocessing Commands ---

echo "START: Processing cluster IDs (this will take a while)..."
python scripts/preprocess/preprocess_cluster_ids.py \
    --data_dir "$DATA_DIR" \
    --cluster_model_path "$CLUSTER_MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo ""
echo "SUCCESS: Cluster ID preprocessing completed successfully!"
echo ""
echo "   Your dataset now contains pre-computed cluster IDs for every frame"
echo "   This will fix the classification head receiving zero gradients"
echo ""
echo "INFO: You can now run training with: bash bash_scripts/train_single_gpu.sh"
