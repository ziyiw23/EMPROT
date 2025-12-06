#!/bin/bash
#SBATCH --job-name=cluster_dyn
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rbaltman
#SBATCH --output=output/logs/cluster_by_dynamics.out
#SBATCH --error=output/logs/cluster_by_dynamics.err

set -euo pipefail

module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate
export PYTHONPATH="/oak/stanford/groups/rbaltman/ziyiw23/EMPROT:${PYTHONPATH:-}"

python scripts/analysis/cluster_proteins_by_dynamics.py \
  --data_dir /oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings \
  --num_clusters 8 \
  --top_k 500 \
  --markov_ckpt output/markov_ckpts/markov_train.pkl \
  --output protein_clusters.json