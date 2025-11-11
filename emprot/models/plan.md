## emprot.models

Model implementation for classification on fixed‑length history windows with multi‑step futures, using an L/K/F split.

Notation:
- L = number of older history frames (summarized context)
- K = number of recent full‑resolution frames
- F = number of future frames (targets)

### Public APIs

- `ProteinTransformerClassificationOnly`
  - Constructor: `(d_embed: int, num_heads: int, dropout: float = 0.1, use_gradient_checkpointing: bool = True, min_context_frames: int = 2, num_clusters: int = 50000, num_layers: int = 1, attention_type: str = 'cross_temporal', future_horizon: int = 1, recent_full_frames: Optional[int] = None, **backbone_kwargs)`
  - Forward (supports both direct multi‑horizon and AR unroll with SS):
    - `forward(input_cluster_ids, times, sequence_lengths, history_mask, change_mask=None, run_length=None, delta_t=None, t_scalar=None, state=None, teacher_future_ids=None, future_step_mask=None, scheduled_sampling_p=0.0) -> dict`
  - Returns: `{ 'cluster_logits': (B, F, N, C), 'context': (B, N, D), ['new_state': (B, N, D)] }`

- `TemporalBackbone`
  - `encode(embeddings, times, sequence_lengths, history_mask, t_scalar=None, change_mask=None, run_length=None, delta_t=None, state=None) -> (query0, h_last, new_state)`
  - Single, standard path with cross‑temporal attention layers (no hybrid context path).

### Data shapes and masks

- Inputs
  - `input_cluster_ids`: `(B, T, N)` long — cluster IDs per residue per history frame, where `T = L+K` if the dataloader provides L prefix frames.
  - `times`: `(B, T)` float — temporal stamps (optional; zeros if unused)
  - `sequence_lengths`: `(B,)` long — valid time steps per sample
  - `history_mask`: `(B, T, N)` bool — True where tokens are valid

- Outputs
  - `cluster_logits`: `(B, F, N, C)` float — logits for `future_horizon = F` steps
  - `context`: `(B, N, D)` float — last‑frame contextualized residue features
  - `new_state` (optional): `(B, N, D)` float — per‑residue rolling summary for streaming

### Module breakdown

1) Embedding and temporal encoding
- `cluster_embedding`: `nn.Embedding(C+1, D)` maps IDs to vectors. Invalid IDs are clamped to padding.
- `TemporalEncoder`: sinusoidal time encodings added to embeddings, indexed by `times`. Kept lightweight.
- Optional `TemporalFeatureProjector` exists but is disabled by default; `change_mask`/`run_length` are deprecated.

2) TemporalBackbone (single path)
- Query tokens: last valid frame per sample gathered from encoded embeddings → `(B, N, D)`.
- History keys/values: flatten `(B, T, N, D)` → `(B, T·N, D)` and build one key‑padding mask (True=PAD) once.
- Cross‑temporal attention stack: N layers of `CrossTemporalAttention` + FFN (pre/post LayerNorms), no hybrid context, no spatial decoders.
- Rolling per‑residue state (optional): if `state` `(B, N, D)` is provided, prepend it as a “virtual past frame” before the K full‑resolution frames for K/V; update via EMA after encoding.

3) Attention primitive
- `CrossTemporalAttention`: scaled dot‑product attention over flattened history with future masking and key‑padding mask. Axial/hierarchical variants are removed from the default build.

4) Multi‑horizon decoding (with L/K/F)
- Parallel direct multi‑horizon (teacher forcing, SS=0):
  - Given input history of length `T = L+K`, and teacher futures `F`, build F windows of length K using the last K frames as full‑resolution context.
  - Window f uses indices `[s : s+K)` where `s = base + f` and `base = T − K` (so f=0 uses the last K frames). The prefix before s (length up to L) is summarized:
    - If `latent_summary_enabled=False`, use a per‑window EMA state gathered at index `s−1`.
    - If `latent_summary_enabled=True`, use `StreamingLatentPool` to summarize frames `[: s)` capped by `latent_summary_max_prefix` (L).
  - Batch all F windows: shape `(B·F, K, N, D)` with optional per‑window summary; run one backbone.encode; classify → reshape to `(B, F, N, C)`.
- Autoregressive unroll (teacher forcing + SS>0): step through horizons; at each step use the last K frames (sliding window) as full‑res context, and summarize the older‑than‑K prefix into latents; swap teacher vs predicted IDs by probability p; update latents with the frame leaving the K window.
  - Both modes share a single `ClassificationHead` producing `(B, F, N, C)`.

### Forward flow (end‑to‑end)

1. Preprocess
   - Clamp negative/invalid IDs to padding, embed IDs to `(B, T, N, D)`.
   - Add sinusoidal time encodings (if supplied).

2. Encode
   - Build `history` `(B, T·N, D)` and key‑padding mask from `history_mask`.
   - Gather last‑frame queries `(B, N, D)`.
   - If `state` provided, prepend it to history as a virtual frame and extend time/mask accordingly.
   - Run cross‑temporal attention stack → `h_last` `(B, N, D)`.
   - Update `new_state = α·state + (1−α)·stopgrad(h_last)` if `state` was provided.

3. Decode
   - Project `h_last` to F step‑specific features; apply `ClassificationHead` per step to produce `(B, F, N, C)` logits.

### Training and inference

- Training
  - Targets: `future_cluster_ids (B, F, N)` (integers). Use `future_step_mask (B, F)` and `residue_mask (B, N)` to mask CE loss.
  - Fixed K history; optionally compute `new_state` but do not carry it across batches (detach).

- Streaming inference (optional)
  - Maintain `state (B, N, D)` across calls. For each new frame/chunk, feed `state` into forward and receive `new_state` to carry forward. No rescanning of older frames.

### Removed legacy paths

- Hybrid context builder (`ContextBuilder`/`LatentPool`/`FramePool`) and spatial decoder blocks are removed.
- Axial/hierarchical temporal attention variants are dropped from the default factory; only `CrossTemporalAttention` is used.
- Copy‑gate and sparse/cosine classification heads are removed.
- `change_mask`/`run_length` are deprecated and unused by default.

### Extensibility

- Streaming global latents: `StreamingLatentPool (B, Lz, D)` is available. If enabled, concatenate `[Z_t || history]` for K/V and update `Z` online per plan. Use `latent_summary_max_prefix` to cap L.
- Positional encoding: can swap sinusoidal with learned positional embeddings if `times` are uniform.

### Testing

- Unit: `(B,T=L+K,N)` → `(B,F,N,C)`; verify optional `new_state.shape == (B,N,D)`, no NaNs.
- Serialization: save/load `state_dict`; forward equivalence with fixed seed.
