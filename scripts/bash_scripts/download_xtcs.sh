#!/bin/bash
#SBATCH --job-name=pdb_download
#SBATCH --partition=rbaltman
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/pdb_download.out
#SBATCH --error=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/pdb_download.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu

set -euo pipefail

module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate

cd /oak/stanford/groups/rbaltman/ziyiw23/EMPROT

echo "[INFO] Starting GPCRmd downloads..."
# Allow override via env MAX_WORKERS; default to SLURM_CPUS_PER_TASK or 16
# MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
# declare -a EXTRA_ARGS
# EXTRA_ARGS=()
# if [[ "${FORCE:-0}" == "1" ]]; then
#   EXTRA_ARGS+=(--force)
# fi
# if [[ -n "${DYN_IDS:-}" ]]; then
#   EXTRA_ARGS+=(--dyn_ids "${DYN_IDS}")
# fi
# if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
#   python -u scripts/preprocess/download_xtcs.py --max_workers "${MAX_WORKERS}" "${EXTRA_ARGS[@]}"
# else
#   python -u scripts/preprocess/download_xtcs.py --max_workers "${MAX_WORKERS}"
# fi

# MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
# RETRY_MAX=$(( MAX_WORKERS / 2 )); [ "$RETRY_MAX" -lt 1 ] && RETRY_MAX=1

# python scripts/preprocess/download_xtcs.py \
#   --max_workers "${MAX_WORKERS}" \
#   --retry_passes 2 \
#   --retry_sleep 120 \
#   --retry_max_workers "${RETRY_MAX}"

# echo "[INFO] Done."

python -u scripts/preprocess/download_xtcs.py \
  --pdb_only \
  --max_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --retry_passes 2 \
  --retry_sleep 120 \
  --retry_max_workers "$(( ${SLURM_CPUS_PER_TASK:-8} / 2 ))"