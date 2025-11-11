### EMPROT preprocessing: sampling → embeddings → cluster IDs

This pipeline prepares trajectory data for the EMPROT dataloader. It:
- extracts all frames from XTCs into per-trajectory PDBs,
- generates per-frame embeddings,
- assigns per-residue cluster IDs used as classification targets.

### Inputs and outputs

- **Inputs**:
  - PDB/XTC pairs organized so that filenames match `..._dyn_<ID>.pdb` and `..._trj_<ID>.xtc`.
- **Outputs**:
  - Sampled frames: `/scratch/.../traj_sampled_pdbs/<pdb_basename>_traj_<unique_traj_id>/frame_<idx>.pdb`
  - Embeddings: `/scratch/.../traj_embeddings/<protein_dir>/...` (consumed by the dataloader)
  - Cluster IDs added into each frame record under the embeddings root

### Step 1 — Sample trajectories (PDB frames)

- Script: `scripts/preprocess/sample_trajectories.py`
- SLURM wrapper: `scripts/bash_scripts/run_sampling.sh`

What it does:
- Groups PDB/XTC by common `<ID>`.
- Validates XTCs have exactly 2500 frames.
- Randomly selects one valid XTC per group and saves every frame as a PDB (multiprocessing).

Run:
- Edit `scripts/bash_scripts/run_sampling.sh` as needed (modules, venv), and remove `--debug` to process all groups.
- Submit:
```bash
sbatch scripts/bash_scripts/run_sampling.sh
```

Defaults inside the Python script:
- `--input-dir`: `/scratch/groups/rbaltman/hkrupkin/gpcr_database_v3`
- `--output-dir`: `/scratch/groups/rbaltman/ziyiw23/traj_sampled_pdbs`

### Step 2 — Generate embeddings

- SLURM array wrapper: `scripts/bash_scripts/prep_embeddings.sh`
- Iterates over each per-trajectory folder created in Step 1 and runs `gen_embed.py` (external) to produce embeddings under the embeddings root.

Key variables (edit in the script if needed):
- `SAMPLED_PDB_ROOT="/scratch/groups/rbaltman/ziyiw23/traj_sampled_pdbs/"`
- `EMBEDDINGS_ROOT="/scratch/groups/rbaltman/ziyiw23/traj_embeddings/"`

Run:
```bash
sbatch scripts/bash_scripts/prep_embeddings.sh
```

### Step 3 — Assign cluster IDs (classification targets)

- SLURM wrapper: `scripts/bash_scripts/preprocess_cluster_ids.sh`
- Applies a pretrained clustering model to embeddings and writes `cluster_ids` per frame, which the dataloader requires.

Key variables (edit in the script if needed):
- `DATA_DIR="/scratch/groups/rbaltman/ziyiw23/traj_embeddings"`
- `CLUSTER_MODEL_PATH="/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl"`

Run:
```bash
sbatch scripts/bash_scripts/preprocess_cluster_ids.sh
```

### How the dataloader consumes this

- The dataloader expects trajectory directories under `data_dir` (e.g., `/scratch/.../traj_embeddings/`) where each frame includes `cluster_ids`.
- It samples variable-length sequences using a stride in frames (default `stride=5`, with `time_step=0.2` ns → 1.0 ns between sequence steps).

Batch contents (shapes):
- `input_cluster_ids` / `ids`: `[B, T, N]` long
- `targets['target_cluster_ids']`: `[B, N]` long
- `times`, `delta_t`: `[B, T]` float
- Masks: `residue_mask` `[B, N]`, `history_mask` `[B, T, N]`
- Optional: `future_cluster_ids` `[B, F, N]`, `future_times` `[B, F]`, `msm_pi`, `msm_P`

Minimal test after preprocessing:
```python
from emprot.data.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="/scratch/groups/rbaltman/ziyiw23/traj_embeddings",
    metadata_path="PATH/TO/metadata.csv",
    batch_size=2,
    min_sequence_length=3,
    max_sequence_length=5,
    stride=5,
)

batch = next(iter(train_loader))
print({k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()})
print({k: v.shape for k, v in batch['targets'].items()})
```

### Tips

- Step 1 validation is strict (2500 frames); adjust only if your dataset differs.
- Remove `--debug` in `run_sampling.sh` to process all groups.
- Ensure `DATA_DIR` in the cluster-ID step points to the embeddings root from Step 2.


