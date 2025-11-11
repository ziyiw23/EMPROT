#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=prep_embed_array
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --array=0-179
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=4:00:00 # time for a single protein
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/EMPROT/logs/prep_embed/out_%A_%a.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/EMPROT/logs/prep_embed/err_%A_%a.log

# --- Configuration ---
SAMPLED_PDB_ROOT="/scratch/groups/rbaltman/ziyiw23/traj_sampled_pdbs/"
EMBEDDINGS_ROOT="/scratch/groups/rbaltman/ziyiw23/traj_embeddings/"
COLLAPSE_DIR="/oak/stanford/groups/rbaltman/ziyiw23/opt_collapse"
CONDA_PATH="/scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3"
CONDA_ENV_NAME="collapse"
SCRIPT="gen_embed.py"

# --- Environment Setup ---
cd $COLLAPSE_DIR || exit 1
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
module load devel cuda/11.7.1 gcc/12.4.0 
export LD_LIBRARY_PATH=/scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/envs/collapse/lib:$LD_LIBRARY_PATH
conda activate /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/envs/collapse

# --- Job Array Logic ---
# Create an array of all the protein directories from the input path
mapfile -t PROTEIN_DIRS < <(find "$SAMPLED_PDB_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

# Select the specific protein directory for this job using the SLURM task ID
CURRENT_PROTEIN_DIR=${PROTEIN_DIRS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$CURRENT_PROTEIN_DIR" ]; then
    echo "Error: No protein directory found for SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Get the protein's name from its directory path (e.g., "10224_dyn_15")
PROTEIN_NAME=$(basename "$CURRENT_PROTEIN_DIR")

# Define the final output path for this protein, creating a subdirectory for it
DATA_OUT_FINAL="$EMBEDDINGS_ROOT/$PROTEIN_NAME"
mkdir -p "$DATA_OUT_FINAL"

# --- Run Script ---
echo "SLURM Array Job ID: $SLURM_ARRAY_TASK_ID"
echo "Processing input directory: $CURRENT_PROTEIN_DIR"
echo "Output will be saved to: $DATA_OUT_FINAL"

python3 "$SCRIPT" "$CURRENT_PROTEIN_DIR" "$DATA_OUT_FINAL" --filetype pdb --num_workers 9 --compile_model

echo "Embedding generation complete for $PROTEIN_NAME."
