#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=sample_trajectories
#SBATCH --partition=rbaltman
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --output=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/out_sample_traj_gromacs.log
#SBATCH --error=/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/logs/err_sample_traj_gromacs.log

set -uo pipefail

# --- Module Loading ---
module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0
module load chemistry
module load gromacs/2025.1

# --- Environment Setup ---
source /oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/activate
PYTHON="$(which python)"

# --- Paths ---
# INPUT_DIR="/oak/stanford/groups/rbaltman/ziyiw23/GPCR_trajectories/xtc"
# OUTPUT_DIR="/oak/stanford/groups/rbaltman/ziyiw23/traj_sampled_pdbs"
# mkdir -p "$OUTPUT_DIR"

# --- Run Script ---
# Use all allocated CPUs; incremental mode will skip already-complete outputs.
# NUM_WORKERS="${SLURM_CPUS_PER_TASK:-32}"
# echo "Starting trajectory sampling with $NUM_WORKERS workers..."
# python -u scripts/preprocess/sample_trajectories.py \
#   --input-dir "$INPUT_DIR" \
#   --output-dir "$OUTPUT_DIR" \
#   --num-workers "$NUM_WORKERS"
# echo "Files in output directory after running: $(find $OUTPUT_DIR -type f | wc -l)"

IN="/oak/stanford/groups/rbaltman/ziyiw23/GPCR_trajectories/xtc"
OUT="/oak/stanford/groups/rbaltman/ziyiw23/traj_sampled_pdbs"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$OUT"

echo "Starting trajectory sampling with $NUM_WORKERS workers..."

# --- Helpers ---
qc_xtc () {
  # Args: xtc pdb
  local xtc="$1"
  local pdb="$2"
  local check_out
  if ! check_out=$(gmx check -f "$xtc" 2>&1); then
    echo "[QC] gmx check failed for $(basename "$xtc")"
    return 1
  fi
  if echo "$check_out" | grep -qi "Incomplete frame"; then
    echo "[QC] Incomplete frame in $(basename "$xtc"); marking invalid"
    return 1
  fi
  echo "[QC] OK $(basename "$xtc")"
  return 0
}

process_one_protein () {
  local pdb="$1"
  local base dyn

  base=$(basename "$pdb")
  dyn=$(sed -E 's/.*_dyn_([0-9]+)\.pdb/\1/' <<< "$base")

  # all XTCs for this dynamics
  local xtcs=()
  mapfile -t xtcs < <(ls "$IN"/d${dyn}_tr{j,aj}_*.xtc 2>/dev/null || true)
  if [ ${#xtcs[@]} -eq 0 ]; then
    echo "[MISS] no XTC for dyn $dyn"
    return 0
  fi

  # QC each candidate; keep only valid
  local valid_xtcs=()
  local x
  for x in "${xtcs[@]}"; do
    if qc_xtc "$x" "$pdb"; then
      valid_xtcs+=("$x")
    fi
  done
  if [ ${#valid_xtcs[@]} -eq 0 ]; then
    echo "[SKIP] dyn $dyn: no valid XTC after QC"
    return 0
  fi

  # pick one valid at random
  local xtc tid out
  xtc=$(printf '%s\n' "${valid_xtcs[@]}" | shuf -n1)
  tid=$(basename "$xtc" | sed -E 's/^d[0-9]+_(trj|traj)_([0-9]+)\.xtc/\2/')

  out="$OUT/${base%.pdb}_traj_${tid}"
  if [ -d "$out" ] && [ "$(ls -1 "$out"/frame_*.pdb 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[SKIP] $out already has frames"
    return 0
  fi
  mkdir -p "$out"
  if ! printf "System\nSystem\n" | gmx trjconv \
      -f "$xtc" -s "$pdb" -o "$out/frame_.pdb" \
      -sep -pbc atom -ur rect -center -quiet -nobackup; then
    echo "[ERR] trjconv failed for dyn $dyn XTC $(basename "$xtc")" >&2
    return 1
  fi
  echo "[DONE] Finished ${base%.pdb} using $(basename "$xtc") -> $out"
}

pdbs=( "$IN"/*_dyn_*.pdb )
if [ ${#pdbs[@]} -eq 0 ]; then
  echo "No PDBs found in $IN"
else
  echo "Processing ${#pdbs[@]} proteins with up to $NUM_WORKERS parallel jobs..."
  active=0
  for pdb in "${pdbs[@]}"; do
    process_one_protein "$pdb" || echo "[ERR] protein $(basename "$pdb") failed" >&2 &
    ((active++))
    if [ "$active" -ge "$NUM_WORKERS" ]; then
      wait -n
      ((active--))
    fi
  done
  wait
fi

echo "Job finished." 