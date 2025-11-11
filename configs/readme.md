### EMPROT config cheat sheet

### Model
- model.attention_type: cross_temporal | axial_temporal | hierarchical_temporal
- model.temporal_bias: true | false  (per‑head time‑decay prior)
- model.num_layers: 1 | 4 | N
- model.dropout: 0.1 | 0.15 | 0.2
- model.classifier_type: linear | cosine
- model.classifier_scale: 30.0  (for cosine head)

### Loss (classification)
- loss.label_smoothing: 0.0–0.1
- loss.balanced_softmax: false | true  (needs train_class_counts)
- loss.logit_adjustment_tau: 0.0 | 0.5–1.0  (needs train_class_counts)
- loss.use_margin: false | true  (AM‑Softmax style)
- loss.margin_m: 0.2–0.35
- loss.margin_s: 30.0

### Mode selection
- loss.classification_weight: 1.0  (classification‑only)
- loss.regression_weight: 0.0  (classification‑only)

### Training
- training.learning_rate: 1e‑4 | 7e‑5 | 3e‑5
- training.max_grad_norm: 0.5 | 1.0
- training.warmup_proportion: 0.043 | 0.1 | 0.2
- training.grad_accum_steps: 8

### Evaluation (inference‑only)
- --eval_topN: 50000 for global metrics
- --topk: 10  (saved predictions)
- --logit_adjust_tau: 0.0–1.0
- --rerank_centroid: off | on
- --rerank_topk: 10
- --rerank_alpha: 0.8
- --rerank_beta: 20.0

### Data & curriculum
- curriculum.disable_data_curriculum: true
- curriculum.disable_loss_curriculum: true
- data.sequence_length: 5 (typical)
- data.stride: 10 (0.2 ns × stride)

### Class counts & seeds
- train_class_counts: list[int]  (optional)
- train_class_counts_path: "counts_train.json"  (preferred; auto‑loaded)
- scripts/analysis/compute_train_class_counts.py: supports --seed (default 42) and passes it into create_dataloaders for reproducible splits.