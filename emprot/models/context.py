from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Context utilities used by the model.

Scope and data contracts
- StreamingLatentPool: online context summarizer (Perceiver-style) that updates a small set of
  latent tokens from the current full-resolution frame. Designed for streaming long histories
  without rescanning older frames.
  Inputs/outputs:
    - Z_t: (B, L, D) float or None — latent tokens at step t; if None, initialized from a learned
      template and expanded to batch.
    - X_t: (B, N, D) float — token features for the current frame (one vector per residue).
    - Returns Z_{t+1}: (B, L, D) float — updated latents after cross-attention and MLP.
  Update rule (within forward):
    1) LayerNorm on Z_t and X_t; linear projections to Q/K/V.
    2) Scaled-dot-product attention with latents as queries over residue tokens as keys/values.
    3) Residual add with dropout and output projection; then a residual MLP block.
  Usage w/ backbone:
    - Concatenate latents Z_t before the recent K full-resolution tokens as K/V, use last-frame
      residues as queries, then update Z with the produced last-frame features. In the current
      code, the backbone defaults to a per-residue rolling state and does not wire Z by default;
      Z can be enabled later if desired.

- TemporalFeatureProjector: optional feature augmentor (change/run/delta) kept for legacy
  compatibility; it is disabled by default in the backbone and can be removed if not needed.
"""


def ema_update_state(prev: torch.Tensor, frame: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-residue EMA update for a single frame.

    prev: (B, N, D) previous EMA state; frame: (B, N, D) features of current frame.
    Returns: (B, N, D) updated EMA state.
    """
    return float(alpha) * prev + (1.0 - float(alpha)) * frame


def prefix_ema_sequence(all_emb: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute per-residue EMA prefix states over an example timeline.

    all_emb: (B, L, N, D) embeddings for all frames in the example timeline.
    Returns (B, L, N, D) where output[:, t] is the EMA state after consuming frames up to t.
    """
    B, L, N, D = all_emb.shape
    out = []
    prev = torch.zeros(B, N, D, device=all_emb.device, dtype=all_emb.dtype)
    for t in range(L):
        prev = ema_update_state(prev, all_emb[:, t, :, :], alpha)
        out.append(prev)
    return torch.stack(out, dim=1)


def build_parallel_windows(all_ids: torch.Tensor, K: int, Fh: int) -> torch.Tensor:
    """Construct Fh teacher-forced windows of length K from a timeline of IDs.

    all_ids: (B, L, N) where L >= K+Fh-1; returns (B, Fh, K, N) with window f at [f-1 : f-1+K].
    """
    B, L, N = all_ids.shape
    windows = []
    for f in range(1, Fh + 1):
        s = f - 1
        e = s + K
        windows.append(all_ids[:, s:e, :])
    return torch.stack(windows, dim=1)


def build_prefix_states_from_ema(ema_seq: torch.Tensor, K: int, Fh: int) -> torch.Tensor:
    """Gather prefix EMA states for each window start (index s-1).

    ema_seq: (B, L, N, D) EMA after each timeline index; returns (B, Fh, N, D).
    If s-1 < 0, a zero state is used.
    """
    B, L, N, D = ema_seq.shape
    states = []
    zero = torch.zeros(B, N, D, device=ema_seq.device, dtype=ema_seq.dtype)
    for f in range(1, Fh + 1):
        s = f - 1
        states.append(ema_seq[:, s - 1] if s - 1 >= 0 else zero)
    return torch.stack(states, dim=1)


class TemporalFeatureProjector(nn.Module):
    """Augment token embeddings with lightweight temporal features."""

    def __init__(self, d_model: int, d_extra: int = 16, max_run_length: int = 32) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_extra = int(max(1, d_extra))
        self.max_run_length = int(max(1, max_run_length))
        self.change_embed = nn.Embedding(2, self.d_extra)
        self.run_embed = nn.Embedding(self.max_run_length + 1, self.d_extra)
        self.delta_proj = nn.Linear(1, self.d_extra)
        self.proj = nn.Linear(self.d_model + 3 * self.d_extra, self.d_model)

    def forward(
        self,
        base: torch.Tensor,
        change_mask: Optional[torch.Tensor] = None,
        run_length: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base.dim() != 4:
            raise ValueError("TemporalFeatureProjector expects base embeddings of shape (B,T,N,D)")
        B, T, N, _ = base.shape
        device = base.device

        if change_mask is None:
            change_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
        else:
            change_mask = change_mask.to(device=device, dtype=torch.bool)
            if change_mask.dim() == 3:
                change_mask = change_mask
            elif change_mask.dim() == 2:
                change_mask = change_mask.unsqueeze(1).expand(-1, T, -1)
            else:
                raise ValueError("change_mask must have shape (B,T,N) or (B,N)")

        if run_length is None:
            run_length = torch.ones(B, T, N, dtype=torch.long, device=device)
        else:
            run_length = run_length.to(device=device, dtype=torch.long)
            if run_length.dim() == 3:
                run_length = run_length
            elif run_length.dim() == 2:
                run_length = run_length.unsqueeze(1).expand(-1, T, -1)
            else:
                raise ValueError("run_length must have shape (B,T,N) or (B,N)")

        run_length = run_length.clamp(min=0, max=self.max_run_length)

        if delta_t is None:
            delta_t = torch.zeros(B, T, dtype=base.dtype, device=device)
        else:
            delta_t = delta_t.to(device=device, dtype=base.dtype)
            if delta_t.dim() == 1:
                delta_t = delta_t.unsqueeze(0).expand(B, -1)
            elif delta_t.dim() != 2:
                raise ValueError("delta_t must have shape (B,T) or (T,)")
        delta_expanded = delta_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1)

        change_feat = self.change_embed(change_mask.long())
        run_feat = self.run_embed(run_length)
        delta_feat = self.delta_proj(delta_expanded)

        concat = torch.cat([base, change_feat, run_feat, delta_feat], dim=-1)
        return self.proj(concat)


class ContextBuilder(nn.Module):
    """
    Build KV context using two strategies:
      - latent_cap: recent K frames at full resolution + fixed number of latents summarizing the tail
      - hier_pool:  recent K frames at full resolution + pooled tokens per frame, then summarize with latents
    """
    def __init__(self,
                 d_model: int,
                 recent_full_frames: int = 5,
                 context_mode: str = 'latent_cap',
                 train_context: Optional[Dict] = None,
                 summarizer: Optional[Dict] = None,
                 hier_pool: Optional[Dict] = None,
                 memory: Optional[Dict] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.recent_full_frames = int(max(0, recent_full_frames))
        self.context_mode = (context_mode or 'latent_cap').lower()
        self.train_cfg = train_context or {}
        self.sum_cfg = summarizer or {}
        self.hier_cfg = hier_pool or {}
        self.mem_cfg = memory or {}

        self.num_latents = int(self.sum_cfg.get('num_latents', 48))
        self.latent_heads = int(self.sum_cfg.get('heads', 8))
        self.latent_layers = int(self.sum_cfg.get('layers', 1))
        self.latent_dropout = float(self.sum_cfg.get('dropout', 0.0))
        self.sum_mask_true_is_valid = bool(self.sum_cfg.get('mask_true_is_valid', True))
        self.store_attn_eval_only = bool(self.mem_cfg.get('store_attn_eval_only', True))
        self.summarizer = LatentPool(
            d_model=self.d_model,
            num_latents=self.num_latents,
            heads=self.latent_heads,
            layers=self.latent_layers,
            dropout=self.latent_dropout,
            mask_true_is_valid=self.sum_mask_true_is_valid,
            store_attn_eval_only=self.store_attn_eval_only,
        )

        self.R = int(self.hier_cfg.get('R', 3))
        self.scorer_hidden = int(self.hier_cfg.get('scorer_hidden', 256))
        self.frame_pool = FramePool(self.d_model, R=self.R, scorer_hidden=self.scorer_hidden)

        self.kmin = int(self.train_cfg.get('kmin', 3))
        self.kmax = int(self.train_cfg.get('kmax', 8))
        self.variable_k = bool(self.train_cfg.get('variable_k', True))
        self.stopgrad_tail = bool(self.train_cfg.get('stopgrad_tail', True))

    def _choose_K(self, T: int) -> int:
        K = self.recent_full_frames
        if self.training and self.variable_k:
            low = max(1, min(self.kmin, self.kmax))
            high = max(low, self.kmax)
            K = int(torch.randint(low, high + 1, (1,)).item())
        return int(max(1, min(K, T)))

    def build(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Context:
        B, T, N, D = x.shape
        device = x.device
        K = self._choose_K(T)

        # Recent window
        recent = x[:, max(0, T - K):, :, :]
        recent_mask = mask[:, max(0, T - K):, :] if mask is not None else None
        recent_flat = recent.reshape(B, K * N, D)
        recent_kpm = normalize_key_padding_mask(recent_mask, true_is_valid=True) if recent_mask is not None else None
        if recent_kpm is None:
            recent_kpm = torch.zeros(B, recent_flat.size(1), dtype=torch.bool, device=device)

        # Tail
        tail_T = max(0, T - K)
        if self.context_mode in ('hier_pool', 'hierarchical', 'hier_pooling') and tail_T > 0:
            tail = x[:, :tail_T, :, :].reshape(B * tail_T, N, D)
            tail_mask = None if mask is None else mask[:, :tail_T, :].reshape(B * tail_T, N)
            tail_pad = None if tail_mask is None else normalize_key_padding_mask(tail_mask, true_is_valid=True)
            pooled = self.frame_pool(tail, frame_pad_mask=tail_pad)
            pooled = pooled.reshape(B, tail_T * self.R, D)
            if tail_mask is None:
                pooled_mask = torch.zeros(B, tail_T * self.R, dtype=torch.bool, device=device)
            else:
                frame_all_pad = tail_pad.view(B, tail_T, N).all(dim=2)
                pooled_mask = frame_all_pad.unsqueeze(-1).expand(-1, -1, self.R).reshape(B, tail_T * self.R)
            pooled = pooled.detach() if (self.training and self.stopgrad_tail) else pooled
            latents = self.summarizer(pooled, pooled_mask)
            lat_mask = torch.zeros(B, latents.size(1), dtype=torch.bool, device=device)
        else:
            if tail_T > 0:
                tail = x[:, :tail_T, :, :].reshape(B, tail_T * N, D)
                tail_mask = None if mask is None else mask[:, :tail_T, :].reshape(B, tail_T * N)
                tail_pad = normalize_key_padding_mask(tail_mask, true_is_valid=True) if tail_mask is not None else None
                tail = tail.detach() if (self.training and self.stopgrad_tail) else tail
                latents = self.summarizer(tail, tail_pad)
                lat_mask = torch.zeros(B, latents.size(1), dtype=torch.bool, device=device)
            else:
                latents = x.new_zeros(B, 0, D)
                lat_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)

        kv = torch.cat([latents, recent_flat], dim=1)
        kv_mask = torch.cat([lat_mask, recent_kpm], dim=1)
        return Context(kv=kv, kv_mask=kv_mask)


class StreamingLatentPool(nn.Module):
    """Online latent tokens updated from the current frame (Perceiver-style)."""

    def __init__(self, d_model: int, num_latents: int = 32, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_latents = int(max(1, num_latents))
        assert d_model % max(1, heads) == 0
        self.heads = heads
        self.d_k = d_model // heads
        self.latents_init = nn.Parameter(torch.randn(self.num_latents, d_model) / (d_model ** 0.5))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, Z_t: Optional[torch.Tensor], X_t: torch.Tensor) -> torch.Tensor:
        B = X_t.size(0)
        if Z_t is None or Z_t.numel() == 0:
            Z_t = self.latents_init.unsqueeze(0).expand(B, -1, -1)
        q = self.w_q(self.ln_q(Z_t)).view(B, Z_t.size(1), self.heads, self.d_k).transpose(1, 2)
        kv = self.ln_kv(X_t)
        k = self.w_k(kv).view(B, X_t.size(1), self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(kv).view(B, X_t.size(1), self.heads, self.d_k).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        upd = attn.transpose(1, 2).contiguous().view(B, Z_t.size(1), self.d_model)
        Z_mid = Z_t + self.dropout(self.proj_o(upd))
        Z_next = Z_mid + self.mlp(Z_mid)
        return Z_next
