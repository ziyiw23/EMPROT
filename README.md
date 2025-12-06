# EMPROT

Transformer models for protein trajectory modeling. The codebase trains and evaluates transformers with cross temporal and hierarchical attention.

## Repository Layout
- `emprot/` --- models, data loaders, samplers, training loop.
- `deployment/` --- Gradio User interface
- `configs/` --- YAML configs for training.
- `scripts/train_transformer.py` --- main training entrypoint.
- `scripts/analysis/` --- clustering, eval, and aggregation utilities.
- `scripts/bash_scripts/` --- Slurm helpers.
- `scripts/preprocess/` --- data download and LMDB prep helpers.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python scripts/train_transformer.py \
  --config configs/residue_centric_full.yaml \
  --run_name my_run \
  --checkpoint_dir output/checkpoints/my_run \
  --embedding_lr_scale 0.1 \
  --use_output_projector
```
Key flags:
- `--embedding_lr_scale`: scale LR for alignment layers.
- `--freeze_alignment_weights`: freeze lookup/projector if set.
- `--max_train_proteins`: optionally subsample proteins for faster iteration.

## Evaluation
Autoregressive eval with optional attention plots:
```bash
python scripts/analysis/autoregressive_eval.py \
  --config configs/residue_centric_full.yaml \
  --checkpoint output/checkpoints/my_run/ckpt.pt \
  --plot_attention
```

## Metrics & Sweeps
- Collect metrics: `scripts/analysis/collect_eval_metrics.py`
- Decode sweeps: `scripts/run_sweep.py` or bash helpers under `scripts/bash_scripts/`.

