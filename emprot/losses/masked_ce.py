#!/usr/bin/env python3
from typing import Optional

import torch
import torch.nn as nn


def masked_cross_entropy(
    logits: torch.Tensor,                 # (B,F,N,C)
    targets: torch.Tensor,                # (B,F,N)
    *,
    future_step_mask: Optional[torch.Tensor] = None,  # (B,F) bool
    residue_mask: Optional[torch.Tensor] = None,      # (B,N) bool
    label_smoothing: float = 0.0,
    horizon_weights: Optional[torch.Tensor] = None,   # (F,) or None
    class_weights: Optional[torch.Tensor] = None,     # (C,) or None
    input_cluster_ids: Optional[torch.Tensor] = None, # (B,T,N) history, used to detect changes at f=0
    change_upweight: float = 1.0,                     # >1.0 to upweight change tokens
) -> torch.Tensor:
    """Masked CE over (B,F,N) with optional smoothing, horizon weights, and CBCE.

    - Valid tokens are those where future_step_mask & residue_mask & targets>=0.
    - horizon_weights (if provided) scales tokens per horizon before reduction.
    - class_weights (if provided) is passed to CE as per-class weighting.
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B,F,N,C); got {tuple(logits.shape)}")
    if targets.dim() != 3:
        raise ValueError(f"targets must be (B,F,N); got {tuple(targets.shape)}")

    B, F, N, C = logits.shape
    device = logits.device

    if future_step_mask is None:
        future_step_mask = torch.ones(B, F, dtype=torch.bool, device=device)
    if residue_mask is None:
        residue_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    M = future_step_mask[:, :, None] & residue_mask[:, None, :]
    valid = M & (targets >= 0)
    if not valid.any():
        return logits.new_zeros(())

    # Guard against out-of-range targets to avoid CUDA device asserts
    targets_sel = targets[valid]
    max_id = int(targets_sel.max().item())
    if max_id >= C:
        raise ValueError(f"Target id {max_id} >= num_classes {C}; check num_clusters/config vs data.")
    min_id = int(targets_sel.min().item())
    if min_id < 0:
        raise ValueError(f"Negative target ids present after masking; check padding/masks.")

    if horizon_weights is not None:
        if isinstance(horizon_weights, (list, tuple)):
            horizon_weights = torch.as_tensor(horizon_weights, dtype=logits.dtype, device=device)
        if horizon_weights.dim() != 1 or horizon_weights.numel() != F:
            raise ValueError("horizon_weights must be shape (F,)")
        w = horizon_weights.view(1, F, 1).expand(B, F, N)
        token_weights = (w * valid.to(logits.dtype))
    else:
        token_weights = valid.to(logits.dtype)

    if float(change_upweight) > 1.0:
        try:
            change_mask = torch.zeros(B, F, N, dtype=torch.bool, device=device)
            if input_cluster_ids is not None and torch.is_tensor(input_cluster_ids) and input_cluster_ids.dim() == 3:
                last_hist = input_cluster_ids[:, -1, :].to(device=device)
                f0 = targets[:, 0, :].to(device=device)
                mask_f0 = (f0 >= 0)
                change_mask[:, 0, :] = (f0 != last_hist) & mask_f0
            if F > 1:
                t_prev = targets[:, :-1, :]
                t_curr = targets[:, 1:, :]
                valid_prev = t_prev >= 0
                valid_curr = t_curr >= 0
                cm = (t_curr != t_prev) & valid_prev & valid_curr
                change_mask[:, 1:, :] = cm
            token_weights = torch.where(change_mask & valid, token_weights * float(change_upweight), token_weights)
        except Exception:
            pass

    logits_sel = logits[valid]            # (K,C)
    targets_sel = targets[valid].long()   # (K,)
    weights_sel = token_weights[valid]    # (K,)

    if class_weights is not None:
        if isinstance(class_weights, (list, tuple)):
            class_weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=device)
        class_weights = class_weights.to(device=device, dtype=logits.dtype)
        if class_weights.numel() != C:
            raise ValueError("class_weights must have length C")
    ls = float(max(0.0, label_smoothing))

    ce_per = nn.functional.cross_entropy(
        logits_sel, targets_sel,
        reduction='none',
        label_smoothing=ls,
        weight=class_weights,
    )  # (K,)

    loss = (ce_per * weights_sel).sum() / weights_sel.sum().clamp_min(1.0)
    return loss
