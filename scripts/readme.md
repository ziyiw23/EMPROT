# scripts (entry points)

Training and evaluation entry points with supporting analysis tools.

Train
- `train_transformer.py` — Python entry to train via `EMPROTTrainer` directly.
- `bash_scripts/train_emprot_config.sh` — wrapper to run experiment configs reproducibly.

Evaluate
- `evaluate_single_model.py` — lightweight evaluation on a saved checkpoint.
- `autoregressive_eval.py` — autoregressive rollout for a chosen trajectory; produces plots and distributional diagnostics.
  - Memory tips for large C (≈50K): prefer `--decode_mode argmax`, reduce `--time_steps`, and keep `--recent_full_frames` / `--context_latents` modest to avoid OOM.

Analysis
- `analysis/` contains plotting and diagnostics (attention viz, class distributions, curriculum analysis, etc.). See its readme for details.

Utilities
- `utils/` and `preprocess/` contain helpers for config preview, preprocessing, and shared tools.

