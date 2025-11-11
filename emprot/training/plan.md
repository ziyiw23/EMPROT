## emprot.training

Compact, classification-only training loop aligned with the simplified data and model modules.

### Scope

- Task: multi-step classification on precomputed cluster IDs
- Inputs: fixed-length history `K`, predict `F` future steps
- Model output: `cluster_logits (B, F, N, C)`
- Loss: masked cross-entropy over future steps and residues
- No regression targets, no hybrid context rebuilds, no on-the-fly clustering

### Public API

- `EMPROTTrainer(config, cluster_lookup=None)`
  - `train_epoch(loader) -> Dict[str, float]`
  - `validate(loader) -> Dict[str, float]`
  - `train(train_loader=None, val_loader=None, test_loader=None) -> float`
  - `load_checkpoint(path) -> bool`

### Expected batch contract (from data)

- Inputs
  - `input_cluster_ids`: (B, T, N) long — history IDs
  - `times`: (B, T) float (optional; zeros OK)
  - `sequence_lengths`: (B,) long
  - `history_mask`: (B, T, N) bool
  - `residue_mask`: (B, N) bool
- Targets
  - `future_cluster_ids`: (B, F, N) long, -1 where invalid
  - `future_step_mask`: (B, F) bool
- Optional
  - `delta_t`, `change_mask`, `run_length` (ignored by default)

### Model interface

- `ProteinTransformerClassificationOnly`
  - Inputs: `input_cluster_ids, times, sequence_lengths, history_mask`
  - Outputs:
    - `cluster_logits`: (B, F, N, C) float
    - `context`: (B, N, D) float
    - `new_state`: (B, N, D) optional (not required for baseline training)

### Loss (masked CE)

- Define a combined mask:
  - `M = future_step_mask[:, :, None] & residue_mask[:, None, :]` → (B, F, N)
- Index logits and targets by `M` and `targets >= 0`:
  - `logits_sel = logits[M]` → (M_pos, C)
  - `targets_sel = future_cluster_ids[M]` → (M_pos,)
- Compute `F.cross_entropy(logits_sel, targets_sel)`; average over valid elements
- No single-step target field; no KL or distributional losses in baseline

### Training step

For each batch:
1. Move tensors to device
2. Forward:
   - `outputs = model(...)`
   - `logits = outputs['cluster_logits']`
3. Masked CE as above
4. Backward, grad clip (`max_grad_norm`), step optimizer, step scheduler (optional)
5. Log scalar metrics: total_loss, token_ce, (optional) token_accuracy

Accuracy (optional):
- Take argmax over logits, mask with `M`, compare to targets ≥ 0, average

### Validation

- Same forward + masked CE
- Report `val/total_loss`, optional token accuracy
- Early stopping on `val/total_loss` with patience

### Optimizer / Scheduler / AMP

- Optimizer: AdamW
  - `learning_rate`, `weight_decay`, `betas` config keys
- Scheduler (optional and simple):
  - Cosine with warmup if `transformers` available; else no scheduler (or StepLR)
- AMP: enable when CUDA available (`use_amp: true`); bfloat16 if supported, else float16

### Checkpointing

- Save:
  - `model_state_dict`, `optimizer`, `scheduler` (if any), `epoch`, `config`
- Best checkpoint on lowest `val/total_loss`; final checkpoint at end of training

### Distributed / Logging

- DDP (optional): rely on torchrun; reduce only scalar logs
- Logging: Python logging; optional `wandb` gated by `use_wandb`

### Slim-down directives (what to remove)

- Remove hybrid context / latent builder usage in training
- Remove multi-step rollout, scheduled sampling, teacher forcing paths
- Remove equilibrium regularizers (`eq/*`), distributional next-k, and related metrics
- Remove copy-gate logic; use dense logits only
- Remove EMA weights by default (allow optional if needed)
- Remove reliance on `faiss` or heavy extras from training path
- Keep only: masked CE, basic metrics, optional AMP/scheduler/DDP

### Minimal config keys

- Model: `d_embed, num_heads, num_layers, future_horizon`
- Optim: `learning_rate, weight_decay, betas`
- Train: `max_epochs, grad_accum_steps, max_grad_norm, use_amp (bool)`
- Scheduler (optional): `use_scheduler, warmup_proportion, min_lr` (if cosine)
- Early stop: `patience, early_stopping_min_delta`

### Pseudocode

