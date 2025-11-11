#!/bin/bash

# Low memory evaluation script for EMPROT models
# This script runs model evaluation with aggressive memory optimization

module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /scratch/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

echo "🚀 Starting EMPROT model evaluation with low memory settings..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Run evaluation with minimal batch size
srun --partition=rbaltman \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=64G \
     python test/test_model_performance.py \
     --run_name emprot_baseline_MSE_regression_ONLY \
     --batch_size 1 \
     --model_type auto

echo "✅ Evaluation completed!"
