## emprot.losses (slim)

Goal: Keep a minimal, robust objective for classification on cluster IDs with fixed K and multi-horizon F, aligned to the simplified data/model/trainer.

### Scope

- Token-level classification only (no regression, no hybrid/context losses)
- Primary loss: masked cross-entropy (CE) over future steps and residues
- Stability: label smoothing (ε) and optional horizon-wise weights
- Optional: class-balanced CE (CBCE) via static class weights if needed
- No KL/JS sequence losses; no equilibrium/distributional next-k paths

### Inputs and masking

- logits: (B, F, N, C) float
- targets: (B, F, N) long, -1 where invalid
- future_step_mask: (B, F) bool
- residue_mask: (B, N) bool

Define valid mask M = (future_step_mask[:, f] & residue_mask[:, n]) and ignore targets < 0.

### Primary loss (Masked CE)

- Select logits/targets where valid
- CE(logits_sel, targets_sel, label_smoothing=ε, reduction='mean')
- If horizon_weights is provided (length F), weight tokens by the per-horizon scalar before averaging

Config keys (trainer consumes):
- label_smoothing: float in [0, 0.1] (default 0.0)
- horizon_weights: Optional[List[float]] length F (default None)

### Optional: Class-balanced CE (CBCE)

Motivation: mitigate long-tail bias if frequent clusters dominate.

- class_weights[c] from training counts (recommended: Effective Number of Samples)
  - w_c = (1 − β) / (1 − β^{n_c}), β≈0.999
  - Optionally normalize and/or clip weights to [w_min, w_max]
- Apply weights in CE as the per-class weight vector

Config keys (optional):
- use_cbce: bool (default False)
- class_weights_path: Optional[str] to a JSON/pt file with weights (loaded at runtime)
- cbce_beta: float (default 0.999) when computing effective number (if a builder is provided)

### API expectations

- Loss function in `emprot.losses.masked_ce.masked_cross_entropy(logits, targets, *, future_step_mask, residue_mask, label_smoothing=0.0, horizon_weights=None, class_weights=None)` → scalar loss
- Prior utilities in `emprot.losses.prior_utils`:
  - `compute_class_counts(loader, horizon_mode='f1'|'all')` → counts (C,)
  - `compute_effective_number_weights(counts, beta=0.999, normalize=True, clip=(0.1,10.0))` → weights (C,)
  - `save_vector/ load_vector`

CLI helper:
- `scripts/compute_class_prior.py` computes counts and effective-number weights from train split and saves JSON/PT

### Defaults

- Start: label_smoothing = 0.05, no CBCE, no horizon_weights
- If tail underfits: enable CBCE
- If far horizons unstable: horizon_weights like [1.0, 1.0, 0.9, 0.8, ...]

### Integration

- `trainer.py` consumes config keys:
  - `label_smoothing: float`
  - `horizon_weights: Optional[List[float]]`
  - `class_weights: Optional[List[float]]` (direct list/tensor)
  - if `class_weights_path` is set in the launch script, it is loaded and injected as `class_weights`


### Tests

- Unit: verify loss decreases on a small synthetic batch after one optimizer step
- Masking: tokens with target=-1 or masked steps/residues do not contribute
- Horizon weights: changing weights changes the contribution per horizon
- CBCE: class_weights scale per-class contribution (sanity with two-class toy)


