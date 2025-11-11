#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=sample_trajectories
#SBATCH --partition=rbaltman
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/EMPROT/logs/out_sample_traj.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/EMPROT/logs/err_sample_traj.log

# --- Module Loading ---
module load cuda/11.7.1
module load gcc/12.4.0
module load python/3.9.0

# --- Environment Setup ---
source /scratch/groups/rbaltman/ziyiw23/EMPROT/emprot_venv/bin/activate

# --- Clear Output Directory ---
OUTPUT_DIR="/scratch/groups/rbaltman/ziyiw23/traj_sampled_pdbs"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Clearing output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"/*
    echo "Output directory cleared."
else
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# --- Run Script ---
# Run the python script, using the number of allocated CPUs for the number of workers.
# This ensures we take full advantage of the requested resources.
echo "Starting trajectory sampling..."
python -u scripts/preprocess/sample_trajectories.py --num-workers 32 --debug
echo "Files in output directory after running: $(find $OUTPUT_DIR -type f | wc -l)"

echo "Job finished." 