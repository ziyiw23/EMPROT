# emprot (package)

Top‑level package for EMPROT.

Contents
- `data/` — data access and batching
- `models/` — transformer model and attention blocks
- `losses/` — composite loss and components
- `training/` — trainer and training loop utilities
- `utils/` — reusable utilities (checkpointing, EMA, metrics, evaluation)

Conventions
- Batches are dicts containing at least: `input_cluster_ids: (B,T,N)`, `sequence_lengths: (B,)`, `history_mask: (B,T,N)`; optional: `future_cluster_ids`, `future_step_mask`, `times`, `delta_t`, `run_length`, `change_mask`.
- Targets for CE: `targets['target_cluster_ids']` shaped `(B,N)` (last step) or `(B,T,N)` (we use last frame).
- Invalid cluster IDs are `-1` and masked out.

Logging
- All losses are normalized to mean per residue for comparability.
- Train and val prefixes are applied in the trainer; equilibrium regs are logged under `eq/*`.

