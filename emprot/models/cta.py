"""
End-to-end forward propagation flow and tensor contracts

Notation and shapes
- B: batch size; T: history length; F: future horizon; N: residues; D: model dim; C: classes

Inputs (from dataloader)
- input_cluster_ids: (B, T, N) long — integer cluster IDs
- times: (B, T) float — per-frame timestamps (optional)
- sequence_lengths: (B,) long — valid history steps per item (≤ T)
- history_mask: (B, T, N) bool — True where tokens are real
- optional state: (B, N, D) float — per-residue rolling summary in feature space

Front end (in transformer)
1) Embedding: IDs → (B, T, N, D) via nn.Embedding(C+1, D)
2) Temporal encoding: add sinusoidal encodings gathered from times (B,T,D) → expand to (B,T,N,D) and add
3) Queries: gather last valid frame per item → (B, N, D)
4) History K/V: flatten frames → (B, T·N, D); if state exists, prepend (B,N,D) to get (B,(1+T)·N,D)
   - Build key_padding_mask (B, T·N) from history_mask (True=valid → False; PAD → True); prepend last-frame residue validity if state used
   - Build query_times (B,N) and key_times (B,T·N); prepend earlier time for state

CrossTemporalAttention (this module)
5) Project to Q/K/V: (B,N,D) and (B,S,D) → (B,H,N,D/H), (B,H,S,D/H)
6) Masks: future_mask from time_diff; padding mask from key_padding_mask; OR both to final attn_mask
7) Attention: scaled_dot_product_attention(Q,K,V, attn_mask), merge heads, output projection → (B, N, D)

Back end
8) Stack attention + FFN layers in transformer backbone; return last-frame features (B, N, D)
9) Multi-step head: map (B,N,D) → (B,F,N,C) logits (shared classifier across steps)
10) Optional: update streaming state via EMA of (B,N,D)
11) Loss masking: apply future_step_mask (B,F) and residue_mask (B,N) to CE on integer targets future_cluster_ids (B,F,N)
"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAxialAttention(nn.Module):
    """Within-frame self-attention over residue tokens with temperature, dropout, and entropy."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        mask_true_is_valid: bool = True,
        attn_temperature: float = 1.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mask_true_is_valid = mask_true_is_valid
        self.temperature = attn_temperature
        self.attn_dropout = attn_dropout
        # For compatibility with visualization hooks
        self.store_attention_weights: bool = False
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        residue_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, N, D = x.shape
        
        q = self.proj_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.proj_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.proj_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply temperature scaling
        q = q / (self.temperature * math.sqrt(self.d_k))
        
        # Create attention mask from residue mask
        if residue_mask is not None:
            if self.mask_true_is_valid:
                attn_mask = ~residue_mask.bool()
            else:
                attn_mask = residue_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        else:
            attn_mask = None
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj_o(attn_output)
        
        # Compute attention entropy for regularization; optionally store weights for viz
        with torch.no_grad():
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            if (not self.training) and self.store_attention_weights and need_weights:
                # Store mean over batch for downstream visualization
                try:
                    self.last_attention_weights = attn_probs.detach().mean(dim=0).cpu()
                except Exception:
                    self.last_attention_weights = None
            else:
                self.last_attention_weights = None
            attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-12), dim=-1).mean()
        
        out = self.dropout(out)
        return out, {"spatial_attn_entropy": attn_entropy}


class CrossTemporalAttention(nn.Module):
    """
    Cross-temporal attention over flattened residue sequences.
    Uses scaled_dot_product_attention (FlashAttention when available).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 store_attention_weights: bool = False,
                 per_source_kv: bool = False,
                 max_source_buckets: int = 0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout
        self.store_attention_weights = store_attention_weights
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.per_source_kv = bool(per_source_kv)
        self.max_source_buckets = int(max(0, max_source_buckets))
        if self.per_source_kv and self.max_source_buckets > 0:
            self.w_k_buckets = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(self.max_source_buckets)])
            self.w_v_buckets = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(self.max_source_buckets)])
            # Fallback shared (used if bucket ids absent)
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.last_attention_weights = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_times: torch.Tensor, key_times: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                source_bucket_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = query.shape
        S = key.shape[1]
        Q = self.w_q(query).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        if self.per_source_kv and (source_bucket_ids is not None) and (self.max_source_buckets > 0):
            # Project K,V per source bucket
            key_flat = key.reshape(B * S, D)
            val_flat = value.reshape(B * S, D)
            buckets = source_bucket_ids.reshape(B * S).to(key.device)
            K_flat = torch.empty_like(key_flat)
            V_flat = torch.empty_like(val_flat)
            for b in range(self.max_source_buckets):
                mask = (buckets == b)
                if mask.any():
                    K_flat[mask] = self.w_k_buckets[b](key_flat[mask])
                    V_flat[mask] = self.w_v_buckets[b](val_flat[mask])
            K = K_flat.view(B, S, D).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
            V = V_flat.view(B, S, D).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        else:
            K = self.w_k(key).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        time_diff = query_times.unsqueeze(-1) - key_times.unsqueeze(1)
        future_mask = (time_diff < 0).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Create attention mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # True=PAD
            mask = mask.expand(-1, self.num_heads, L, -1)
        else:
            mask = torch.zeros((B, self.num_heads, L, S), dtype=torch.bool, device=query.device)
        
        mask = mask | future_mask

        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Store attention weights for visualization (evaluation only)
        if self.store_attention_weights and not self.training:
            with torch.no_grad():
                attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                attn_weights = attn_weights.masked_fill(mask, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                self.last_attention_weights = attn_weights.detach().cpu()
        
        return self.w_o(context)


## Axial and hierarchical temporal attention removed from default build


def build_attention(attention_type: str, d_model: int, num_heads: int, dropout: float = 0.1,
                    store_attention_weights: bool = False,
                    per_source_kv: bool = False,
                    max_source_buckets: int = 0) -> nn.Module:
    return CrossTemporalAttention(d_model, num_heads, dropout, store_attention_weights,
                                  per_source_kv=per_source_kv,
                                  max_source_buckets=max_source_buckets)
