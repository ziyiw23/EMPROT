#!/usr/bin/env python3
from typing import Optional

import torch


def histogram_ce_loss(
    logits: torch.Tensor,                 # (B,F,N,C)
    targets: torch.Tensor,                # (B,F,N)
    *,
    future_step_mask: Optional[torch.Tensor] = None,  # (B,F) bool
    residue_mask: Optional[torch.Tensor] = None,      # (B,N) bool
    eps: float = 1e-6,
    random_horizon: bool = False,
) -> torch.Tensor:
    """Cross-entropy between empirical target histogram q and predicted marginal p̂.

    - Works over all valid tokens in (F,N) per sample (or a random single horizon if random_horizon=True).
    - Ignores targets < 0.
    - Order-free: matches distributions without requiring per-token correctness.
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B,F,N,C); got {tuple(logits.shape)}")
    if targets.dim() != 3:
        raise ValueError(f"targets must be (B,F,N); got {tuple(targets.shape)}")

    B, F, N, C = logits.shape
    device = logits.device

    probs = torch.softmax(logits, dim=-1)  # (B,F,N,C)

    valid = (targets >= 0)  # (B,F,N)
    if future_step_mask is not None:
        valid = valid & future_step_mask.to(dtype=torch.bool, device=targets.device)[:, :, None]
    if residue_mask is not None:
        valid = valid & residue_mask.to(dtype=torch.bool, device=targets.device)[:, None, :]

    if random_horizon and F > 1:
        j = torch.randint(low=0, high=F, size=(B,), device=device)
        fmask = torch.zeros(B, F, 1, dtype=torch.bool, device=device)
        fmask.scatter_(1, j.view(B, 1, 1), True)
        valid = valid & fmask.expand(-1, -1, N)

    total_loss = 0.0
    count = 0
    for b in range(B):
        vb = valid[b]  # (F,N)
        if not vb.any():
            continue
        tb = targets[b].clamp_min(0)  # (F,N)
        pb = probs[b]  # (F,N,C)

        mask_bn = vb.unsqueeze(-1).to(dtype=pb.dtype)
        p_hat = (pb * mask_bn).sum(dim=(0, 1))  # (C,)
        denom = mask_bn.sum(dim=(0, 1)).clamp_min(1.0)  # scalar
        p_hat = (p_hat / denom).clamp_min(eps)

        t_flat = tb[vb]
        q = torch.bincount(t_flat, minlength=C).to(dtype=pb.dtype, device=device)
        q = q / q.sum().clamp_min(eps)

        ce = -(q * p_hat.log()).sum()
        total_loss += ce
        count += 1

    if count == 0:
        return logits.new_zeros(())
    return total_loss / float(count)

