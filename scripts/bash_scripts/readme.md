# scripts/bash_scripts

Convenience wrappers for training, evaluation, and visualization on clusters (e.g., SLURM).

- `train_emprot_config.sh` — run a training job from a YAML config; sets up run name, logging, and checkpoint paths.
- `evaluate_emprot_checkpoint.sh` — evaluate a saved checkpoint.
- `attn_viz.sh` — generate attention heatmaps for a checkpoint using the analysis tool.
- `auto_reg_eval.sh` — batch wrapper around `scripts/autoregressive_eval.py`.
- `prep_embeddings.sh` / `preprocess_cluster_ids.sh` — helpers for preparing LMDB data.
- `run_sampling.sh` — sampling demo using trained logits.

Tip: keep per‑run logs and outputs under `output/` for easy browsing and cleanup.

