## emprot.data

Data loading for EMPROT: building fixed‑length temporal sequences of residue‑level cluster IDs from LMDB trajectory data with an L/K/F split.

Notation:
- L = number of older history frames (summarized context)
- K = number of recent full‑resolution frames
- F = number of future frames (targets)

Scope: multi-step, classification-only. The dataset emits integer cluster IDs only and provides multi-step futures by default. Non-classification targets (e.g., regression on embeddings/coordinates) and single-step-only code paths have been removed.

### Key modules
- `dataset.py`
  - `ProteinTrajectoryDataset`: builds fixed‑length sequences consisting of L+K input frames of per‑residue `cluster_ids` using frames at `start + i·stride` (i=0..L+K−1); at each epoch start it shuffles the set of valid start positions (uses all windows; no per‑protein downsampling).
  - `create_dataloaders(...)`: creates train/val/test `DataLoader`s using `ProteinBucketBatchSampler` and `collate_variable_length`.
- `data_loader.py`
  - `LMDBLoader`: efficient LMDB reader per trajectory; `get_metadata()`, `load_frames(indices)`; auto‑detects gzip/lz4/raw pickles.
- `sampling.py`
  - `ProteinBucketBatchSampler`: buckets items by protein size (num residues, binned) and optionally enforces per‑batch protein diversity.
  - `collate_variable_length`: pads to `(B, T_max, N_max)`, builds `input_mask` (code key: `history_mask`), `residue_mask`, targets, and optional futures/MSM stats.
// moved to scripts/preprocess/cluster_lookup.py
- `cluster_lookup.py` (moved out)
  - `ClusterCentroidLookup` now lives in `scripts/preprocess/cluster_lookup.py` since it is used only for preprocessing/evaluation.
- `metadata.py`
  - `MetadataManager`: load and query CSV metadata (e.g., `get_protein_info(dynamic_id)`).
  - `TrajectoryCatalog`: helper to map between PDB/trajectory naming schemes.

### Conceptual dataset and labels (multi‑step by default; L/K/F)

- Training example (xᵢ): a fixed‑length sequence of L+K input frames from one trajectory, spaced by frame stride s.
  - L = `history_prefix_frames` (summarized context), K = `num_full_res_frames` (recent full‑res context).
  - Each frame is a length‑N vector of integer cluster IDs (one per residue).
  - Shape of xᵢ: [L+K, N] (dtype long). N depends on the protein.
- Multi‑step labels (yᵢ): next F steps after the input window, spaced by the same stride.
  - `future_cluster_ids`: shape [F, N] (dtype long), where F = `future_horizon`.
  - Single‑step targets are not emitted. Use `future_cluster_ids[:, :]` for supervision.

Example (L=5, K=4, F=3, stride=5, start=10):
- x uses frames [10, 15, 20, 25, 30, 35, 40, 45, 50] (L+K=9 frames)
- y uses future frames [55, 60, 65] → `future_cluster_ids` with shape [3, N]

### How a sample is formed

1) Choose a protein and a valid start frame. 2) Set inputs T = L+K, ensure frames exist up to `start + (L+K+F−1)·stride`. 3) Build inputs at `start + i·stride` for `i=0..L+K−1`. 4) Multi‑step targets are the F frames at `start + (L+K + j − 1)·stride` for `j=1..F` (equivalently, start at `target_frame = start + (L+K)·stride` then `target_frame + i·stride` for `i=0..F−1`). Only `cluster_ids` are loaded in this classification‑only pipeline.

### Batch structure (after collation)

- Inputs
  - `input_cluster_ids` (alias `ids`): (B, T_max, N_max) long where `T_max = L+K` within a batch
  - `times`, `delta_t`: (B, T_max) float
- Targets (multi‑step)
  - `future_cluster_ids`: (B, F_max, N_max)
  - `future_times`: (B, F_max)
- Masks/metadata
  - `input_mask` (code key: `history_mask`): (B, T_max, N_max)
  - `residue_mask`: (B, N_max)
  - Optional `change_mask`, `run_length`: (B, T_max, N_max)
  - `sequence_lengths`: (B,)
  - IDs: `traj_name`, `protein_indices`, `start_frames`, `uniprot_ids`, `pdb_ids`, etc.
  - `temporal_info.history_prefix_frames` (L), `temporal_info.recent_full_frames` (K)

Mask semantics:
- `input_mask` (code key: `history_mask`): True where an input token is real (not time padding). Use to mask losses/metrics over time.
- `residue_mask`: True where a residue exists for that protein (not residue padding). Use to ignore padded residues.
- `change_mask`: True where the cluster ID at time t differs from t−1 for that residue. Useful for change‑focused objectives.
- `run_length`: Number of consecutive steps since the last change at (t, residue), computed along the stride‑spaced sequence within a window (frames are at `start + i*stride`). Useful for modeling dwell times/persistence.
- `sequence_lengths`: Actual (unpadded) T per item; often used to derive masks or slice outputs.
- `future_step_mask`: (B, F_max) True where a future step exists (not future padding). Use to mask multistep losses over F.

### Sampling, padding, and efficiency

- At each epoch start (`on_epoch_start`), valid start positions are shuffled so examples vary by start; all windows are eligible each epoch (no per‑protein downsampling).
- `ProteinBucketBatchSampler` groups items by residue count (binned) to reduce padding waste.
- `collate_variable_length` pads time (T) and residues (N) within each batch and emits masks; with fixed K, `T_max == K`.

### Hyperparameters

- `history_prefix_frames` (L) → number of summarized prefix frames
- `num_full_res_frames` (K) → recent full‑resolution sequence length
- `stride` (frames) → temporal spacing; with `time_step=0.2` ns, `stride=5` ≈ 1.0 ns
- `future_horizon` (F) → number of future steps; recommended to set F > 0 (default practice)

### Design simplifications

- Removed per‑protein epoch downsampling: there is no `max_windows_per_protein_per_epoch`. To shorten epochs, adjust dataset size externally (e.g., filter proteins) or reduce steps per epoch/num_batches.

### API changes and removals (for maintainers)

- Replace variable‑length controls with fixed L/K lengths:
  - Remove `min_sequence_length` and `max_sequence_length`; use `history_prefix_frames` (L) and `num_full_res_frames` (K).
  - Update signatures and call sites:
    - `ProteinTrajectoryDataset.__init__(..., history_prefix_frames: int, num_full_res_frames: int, stride: int, future_horizon: int, ...)`
    - `create_dataloaders(..., history_prefix_frames: int, num_full_res_frames: int, ...)`
- Remove per‑protein window downsampling:
  - Delete `max_windows_per_protein_per_epoch` from all constructors and call sites.
  - In `ProteinTrajectoryDataset.on_epoch_start`, use all precomputed windows (shuffle only), remove selection logic based on `max_windows_per_protein_per_epoch`.
  - Ensure `ProteinBucketBatchSampler` relies on `_epoch_indices` produced from the full window list.

- Remove bucketed batching by protein size (optional simplification):
  - Remove `ProteinBucketBatchSampler` and related options (e.g., `enforce_protein_diversity`, bucket shuffling).
  - In `create_dataloaders`, stop passing a `batch_sampler`; instead:
    - Train: `DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_variable_length, ...)`
    - Val/Test: `DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length, ...)`
  - Delete bucket construction code in `sampling.py` and keep only `collate_variable_length`.
  - Rationale: with fixed batch_size and narrow residue-count variance, bucketing doesn’t reduce peak VRAM; simplifying improves maintainability.

- Remove single‑step target field from dataset batches:
  - Delete emission of `targets['target_cluster_ids']` in the dataset and all downstream references in training/eval code.
  - Standardize exclusively on `future_cluster_ids` (multi‑step). If a final‑step target is needed, derive it as `future_cluster_ids[:, 0]` at use‑site.

- Classification‑only scope:
  - Remove data/model/loss code paths that depend on continuous/regression targets (e.g., direct embedding regression, coordinate losses).
  - Do not emit raw embeddings/coordinates as supervision targets from the dataloader; only `input_cluster_ids` and `future_cluster_ids` are required.
  - Remove on‑the‑fly clustering during training; all cluster IDs must be precomputed in preprocessing.

### Train/val/test loaders

Use `create_dataloaders(data_dir, metadata_path, ...)` in `dataset.py`. It splits by protein and uses `collate_variable_length` for batching. Batches contain L+K inputs and F targets.

### Notes

- The dataloader requires `cluster_ids` per frame. If missing, run preprocessing to add them to the LMDB entries.
