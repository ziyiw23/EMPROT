# EMPROT

EMPROT is a classification‑only transformer for protein trajectory modeling. It predicts the next cluster ID per residue given recent context and optional latent summaries. Training and evaluation utilities, modular losses, and analysis scripts are provided to run experiments end‑to‑end at scale.

## Quick Start

1) Train with a config

```
bash scripts/bash_scripts/train_emprot_config.sh configs/depth_context_hybrid.yaml
```

2) Mid‑epoch validation every N steps (set in config)

- Configure `val_check_steps` to run validation in the middle of training steps.
- Train/val use the same forward (single‑step CE or multi‑step rollout) so losses are comparable.

3) Evaluate autoregressively

```
python scripts/autoregressive_eval.py \
  --ckpt checkpoints/<run>/best_model_epoch_X.pt \
  --data_root /path/to/lmdb_root --split val \
  --time_start 200 --time_steps 100 \
  --output_dir output/evaluation_results/<run>/autoregressive_eval \
  --decode_mode argmax
```

4) Visualize attention

```
python scripts/analysis/viz_attention_all_blocks.py --ckpt checkpoints/<run>/best_model_epoch_X.pt
```

## Logging (Weights & Biases)

Training and validation log per‑component losses with identical scales (mean per residue) and keys:

- `train/loss_total`, `val/loss_total`
- `train/loss_ce` (and `train/token_ce` alias), `val/loss_ce` (and `val/token_ce`)
- `train/loss_rollout_kl` (and `train/batch_kl_hist`), `val/loss_rollout_kl` (and `val/batch_kl_hist`)
- `train/loss_nextk_kl` (teacher‑forced next‑K)
- `train/loss_unlikelihood`
- Equilibrium diagnostics are logged under `eq/*`: `eq/pi_kl`, `eq/P_row_kl`, plus `eq/val_*` during validation.

DDP‑safe means are computed via `ddp_reduce_dict`.

## Repository Layout

- `emprot/` — core Python package
  - `data/` — dataset/loader, curriculum, LMDB helpers
  - `models/` — transformer backbone and attention modules
  - `losses/` — `CompositeLoss` and components
  - `training/` — trainer and orchestration
  - `utils/` — checkpointing, EMA, metrics, eval helpers
- `scripts/` — training, evaluation, and analysis entry points
- `scripts/bash_scripts/` — SLURM/CLI wrappers for common runs
- `configs/` — YAML configs; see `configs/readme.md` for knobs
- `test/` — unit/integration tests

See the per‑directory readme.md files for details.

