# scripts/analysis

Plotting and diagnostics to understand model behavior and datasets.

Notable tools
- `viz_attention_all_blocks.py` — captures and saves attention heatmaps for each decoder block and the latent summarizer. Works with or without a checkpoint; uses synthetic data by default.
- `class_distribution_analysis.py` — explores class visitation distributions and imbalance.
- `advanced_eval.py` — extended evaluation comparing distributions and transitions.
- `analyze_train_changes.py` — tracks drift in training metrics across runs.
- `compute_train_class_counts.py` — computes class counts for CBCE/logit adjustment; saves JSON.

Usage
- All scripts are standalone and import the package via a project‑root sys.path insertion.
- Prefer running from the project root so relative output paths match the training outputs.

