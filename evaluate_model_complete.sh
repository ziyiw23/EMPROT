#!/bin/bash

# Complete Model Evaluation Workflow
# This script runs inference first, then cluster conversion separately

RUN_NAME="emprot_baseline_MSE_regression_ONLY"

echo "🚀 Starting complete model evaluation workflow..."
echo "   Run name: $RUN_NAME"
echo ""

# Step 1: Run inference and save results
echo "📊 Step 1: Running inference and saving results..."
srun --partition=rbaltman \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=64G \
     python test/test_model_performance.py \
     --run_name $RUN_NAME \
     --batch_size 1 \
     --model_type auto

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed: Inference results saved"
    echo ""
    
    # Step 2: Convert to clusters (run separately)
    echo "🔄 Step 2: Converting regression predictions to clusters..."
    srun --partition=rbaltman \
         --cpus-per-task=4 \
         --gres=gpu:1 \
         --mem=32G \
         python scripts/convert_regression_to_clusters.py \
         --results_dir evaluation_results/$RUN_NAME \
         --gpu_batch_size 500 \
         --chunk_size 50
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 2 completed: Cluster conversion finished"
        echo ""
        echo "🎉 Complete evaluation workflow finished!"
        echo "📁 Results are in: evaluation_results/$RUN_NAME"
    else
        echo "⚠️  Step 2 failed, but inference results are saved"
        echo "📁 Inference results are in: evaluation_results/$RUN_NAME/inference_results.pkl"
    fi
else
    echo "❌ Step 1 failed: Inference did not complete"
fi
