# emprot.utils

Reusable helpers and lightweight utilities.

Key modules
- `checkpointing.py` — save/load checkpoints. Checkpoints contain at least `model_state_dict` and a `config` snapshot; the evaluator understands this layout.
- `ema.py` — exponential moving average of weights.
- `metrics.py` — aggregation helpers for training/validation.
- `evaluation.py` — embedding metrics and simple plotting helpers used by analysis scripts.
- `sampling.py`, `logit_sampling.py` — helper policies for sampling from logits.
- `mask.py` — small utilities for token masks.

Notes
- `checkpointing` provides best/regular/final checkpoints and is robust to `DataParallel` prefixes.
- Many utilities are CPU‑safe; GPU is only required where tensors are created from CUDA data.

