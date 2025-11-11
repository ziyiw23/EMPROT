#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --partition=rbaltman
#SBATCH --output=output/logs/evaluate_emprot_checkpoint.out
#SBATCH --error=output/logs/evaluate_emprot_checkpoint.err

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

# regression
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_baseline_MSE_regression_ONLY/best_model_epoch_103.pt \
#   --model_name reg_only \
#   --model_type regression \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_baseline_MSE_regression_ONLY_regression \
#   --cluster_model_path /oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl \
#   --topk 10 --eval_topN -1 \
#   --batch_size 4 --device cuda

# # classification only
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_ONLY/best_model_epoch_198.pt \
#   --model_name cls_only \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_ONLY \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# # classification only 8attn
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_attn8/best_model_epoch_96.pt \
#   --model_name cls_attn8 \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_attn8 \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# # # dual head (evaluates both heads)
# # python scripts/evaluate_single_model.py \
# #   --model_path checkpoints/emprot_dual_head_baseline/best_model_epoch_191.pt \
# #   --model_name my_dual \
# #   --model_type dual_head \
# #   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
# #   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
# #   --output_dir evaluation_results/emprot_dual_head_baseline_dual \
# #   --cluster_model_path /oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl \
# #   --topk 10 --eval_topN -1 \
# #   --batch_size 4 --device cuda

# # # classification only axial
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_axial/best_model_epoch_198.pt \
#   --model_name cls_axial \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_axial \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# # # classification only hierarchical
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_hierarchical/best_model_epoch_187.pt \
#   --model_name cls_hierarchical \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_hierarchical \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# # # classification only attn4
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_attn4/best_model_epoch_243.pt \
#   --model_name cls_attn4 \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_attn4 \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda


# # classification only lower smooth
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_lower_smooth/best_model_epoch_243.pt \
#   --model_name cls_lower_smooth \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_lower_smooth \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# # classification only no bias
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_no_bias/best_model_epoch_243.pt \
#   --model_name cls_no_bias \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_no_bias \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# classification only cross balanced
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_cls_cross_balanced/best_model_epoch_198.pt \
#   --model_name cls_cross_balanced \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_cross_balanced \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# classification only cross margin
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_cls_cross_margin/best_model_epoch_198.pt \
#   --model_name cls_cross_margin \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/emprot_classification_cross_margin \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# classification only unlikelihood with tuned unlikelihood weight
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_unlikelihood/best_model_epoch_152.pt \
#   --model_name UL_lW \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/UL_lW \
#   --topk 1 --eval_topN 50000 \
#   --batch_size 4 --device cuda``

# # classification only unlikelihood without tuned unlikelihood weight
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_unlikelihood/best_model_epoch_299.pt \
#   --model_name UL_hW \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/UL_hW \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# classification only change aware ce
# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/emprot_classification_unlikelihood_change_aware_ce/best_model_epoch_198.pt \
#   --model_name UL+ChangeAW \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/UL+ChangeAW \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/context_cbce_unlike_cace/best_model_epoch_9.pt \
#   --model_name cls_ctx_cbce_unlike_cace \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir output/evaluation_results/context_cbce_unlike_cace \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

# python scripts/evaluate_single_model.py \
#   --model_path checkpoints/hybrid_ctx_cls_no_bias/best_model_epoch_113.pt \
#   --model_name hybrid_ctx_cls_no_bias \
#   --model_type classification \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir evaluation_results/hybrid_ctx_cls_no_bias \
#   --topk 10 --eval_topN 50000 \
#   --batch_size 4 --device cuda

  --model_path checkpoints/prev_token_dropout/best_model_epoch_4.pt \
  --model_name prev_token_dropout \
  --model_type classification \
  --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
  --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
  --output_dir output/evaluation_results/prev_token_dropout \
  --topk 10 --eval_topN 50000 \
  --batch_size 4 --device cuda

# H/M/T using eval-defined bins
python scripts/analysis/advanced_eval.py \
  --output evaluation_results/advanced \
  --hmt_results \
    output/evaluation_results/emprot_classification_ONLY/cluster_results \
    output/evaluation_results/context_cbce_unlike_cace/cluster_results \
    output/evaluation_results/prev_token_dropout/cluster_results \
  --hmt_labels baseline cls_ctx_cbce_unlike_cace prev_token_dropout \
  --hmt_bins_source eval \
  --hmt_bins_metric traj_local \
  --hmt_quantiles 0.9 0.99

# H/M/T using baseline models
# python scripts/analysis/advanced_eval.py \
#   --output evaluation_results/advanced_base \
#   --hmt_results \
#     evaluation_results/emprot_baseline_MSE_regression_ONLY_regression/cluster_results \
#     evaluation_results/emprot_dual_head_baseline_dual/classification/cluster_results \
#     evaluation_results/emprot_dual_head_baseline_dual/regression/cluster_results \
#     evaluation_results/emprot_classification_ONLY/cluster_results \
#     evaluation_results/emprot_classification_hierarchical/cluster_results \
#     evaluation_results/emprot_classification_axial/cluster_results \
#     evaluation_results/emprot_classification_lower_smooth/cluster_results \
#     evaluation_results/emprot_classification_no_bias/cluster_results \
#   --hmt_labels reg_only dual_cls dual_reg cls_only cls_hierarchical cls_axial cls_lower_smooth cls_no_bias \
#   --hmt_bins_source eval \
#   --hmt_bins_metric traj_local \
#   --hmt_quantiles 0.9 0.99

# echo "[INFO] Running autoregressive evaluation"

# Comprehensive autoregressive-focused analysis
# python scripts/autoregressive_eval.py \
#   --model_path checkpoints/emprot_classification_ONLY/best_model_epoch_198.pt \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir /scratch/groups/rbaltman/ziyiw23/EMPROT/evaluation_results/emprot_classification_ONLY/autoregressive_eval \
#   --context_frames 5 \
#   --stride 1 \
#   --time_step 0.2 \
#   --device cuda \
#   --plot_trajectories 1 \
#   --max_residues_plot 200 \
#   --tica_lag 1

# python scripts/autoregressive_eval.py \
#   --model_path checkpoints/change_aware_ce_prev_frame_dropout_scheduled_sampling/best_model_epoch_198.pt \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir /scratch/groups/rbaltman/ziyiw23/EMPROT/evaluation_results/ScheSamp+ChangeAW+PrevDrop+UL/autoregressive_eval \
#   --context_frames 5 \
#   --stride 1 \
#   --horizon 30 \
#   --time_step 0.2 \
#   --device cuda \
#   --plot_trajectories 1 \
#   --max_residues_plot 50 \
#   --tica_lag 1
# echo "Finished Running autoregressive evaluation with no temporal bias"

# python scripts/autoregressive_eval.py \
#   --model_path checkpoints/change_aware_ce_prev_frame_dropout_scheduled_sampling/best_model_epoch_198.pt \
#   --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata_path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --output_dir /scratch/groups/rbaltman/ziyiw23/EMPROT/evaluation_results/emprot_classification_ONLY/autoregressive_eval \
#   --context_frames 5 \
#   --stride 1 \
#   --horizon 200 \
#   --time_step 0.2 \
#   --device cuda \
#   --plot_trajectories 1 \
#   --max_residues_plot 200 \
#   --tica_lag 1

# --- Cross-temporal attention visualization ---
# echo "[ATTN-VIZ] Visualizing attention for classification-only best checkpoint"
# python scripts/analysis/visualize_cross_temporal_attention.py \
#   --checkpoint checkpoints/emprot_classification_ONLY/best_model_epoch_198.pt \
#   --output-dir evaluation_results/emprot_classification_ONLY/attention_viz \
#   --data-dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata-path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --mode both --batch-size 1 --device cuda --query-index 0

# echo "[ATTN-VIZ] Visualizing attention for attn8 classification best checkpoint"
# python scripts/analysis/visualize_cross_temporal_attention.py \
#   --checkpoint checkpoints/emprot_classification_attn8/best_model_epoch_96.pt \
#   --output-dir evaluation_results/emprot_classification_attn8/attention_viz \
#   --data-dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata-path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --mode both --batch-size 1 --device cuda --query-index 0

# echo "[ATTN-VIZ] Visualizing attention for no bias classification best checkpoint"
# python scripts/analysis/visualize_cross_temporal_attention.py \
#   --checkpoint checkpoints/emprot_classification_no_bias/best_model_epoch_243.pt \
#   --output-dir evaluation_results/emprot_classification_no_bias/attention_viz \
#   --data-dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
#   --metadata-path /scratch/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
#   --mode both --batch-size 1 --device cuda --query-index 0
