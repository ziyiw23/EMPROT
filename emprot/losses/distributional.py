#!/usr/bin/env python3
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def per_residue_histogram_from_ids(
    future_ids: torch.Tensor,
    num_classes: int,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    B, F, N = future_ids.shape
    device = future_ids.device
    valid = (future_ids >= 0)
    if future_step_mask is not None:
        valid = valid & future_step_mask[:, :, None].to(dtype=torch.bool, device=device)
    if residue_mask is not None:
        valid = valid & residue_mask[:, None, :].to(dtype=torch.bool, device=device)

    q = torch.zeros(B, N, num_classes, device=device, dtype=torch.float32)
    if valid.any():
        idx_flat = future_ids.clamp_min(0)
        one_hot = F.one_hot(idx_flat, num_classes).to(q.dtype)
        one_hot = torch.where(valid[..., None], one_hot, torch.zeros_like(one_hot))
        q = one_hot.sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1)
        q = q / denom[..., None]

    if label_smoothing > 0.0:
        u = torch.full_like(q, 1.0 / float(num_classes))
        q = (1.0 - label_smoothing) * q + label_smoothing * u

    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q


def aggregated_probability_kl_loss(
    logits: torch.Tensor,
    future_ids: torch.Tensor,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
    reduce: str = "mean",
) -> torch.Tensor:
    B, F, N, C = logits.shape
    p = torch.softmax(logits, dim=-1)
    p_avg = p.mean(dim=1)
    with torch.no_grad():
        q = per_residue_histogram_from_ids(
            future_ids,
            C,
            future_step_mask=future_step_mask,
            residue_mask=residue_mask,
            label_smoothing=label_smoothing,
            eps=eps,
        )
    kl = (q * (q.clamp_min(eps).log() - p_avg.clamp_min(eps).log())).sum(dim=-1)
    mask = torch.ones(B, N, dtype=torch.bool, device=logits.device)
    if residue_mask is not None:
        mask = mask & residue_mask.to(dtype=torch.bool, device=logits.device)
    if future_step_mask is not None:
        valid_r = (future_ids >= 0) & future_step_mask[:, :, None].to(dtype=torch.bool, device=logits.device)
        valid_r = valid_r.any(dim=1)
        mask = mask & valid_r
    vals = kl[mask]
    if vals.numel() == 0:
        return kl.new_tensor(0.0)
    if reduce == "sum":
        return vals.sum()
    return vals.mean()


def kl_from_histograms(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def straight_through_gumbel_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    noise = torch.rand_like(logits).clamp_min(1e-8)
    g = -torch.log(-torch.log(noise))
    y_soft = ((logits + g) / tau).softmax(dim=-1)
    y_hard = torch.nn.functional.one_hot(y_soft.argmax(dim=-1), logits.size(-1)).to(y_soft.dtype)
    return (y_hard - y_soft).detach() + y_soft


def _st_histogram_partial_teacher(
    logits: torch.Tensor,
    future_ids: torch.Tensor,
    future_step_mask: Optional[torch.Tensor],
    residue_mask: Optional[torch.Tensor],
    tau: float,
    M: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, F, N, C = logits.shape
    device = logits.device
    valid = (future_ids >= 0)
    if future_step_mask is not None:
        valid = valid & future_step_mask[:, :, None].to(dtype=torch.bool, device=device)
    if residue_mask is not None:
        valid = valid & residue_mask[:, None, :].to(dtype=torch.bool, device=device)
    valid_f = valid.to(dtype=logits.dtype)
    hist = torch.zeros(B, N, C, dtype=logits.dtype, device=device)
    for _ in range(int(M)):
        y = straight_through_gumbel_softmax(logits, tau)
        y = y * valid_f[..., None]
        hist = hist + y.sum(dim=1)
    counts = valid_f.sum(dim=1).clamp_min(1.0)
    p_hat = hist / (counts[..., None] * float(M))
    p_hat = p_hat.clamp_min(eps)
    p_hat = p_hat / p_hat.sum(dim=-1, keepdim=True).clamp_min(eps)
    mask = counts > 0
    return p_hat, mask, valid_f


def st_gumbel_hist_kl_loss(
    model,
    batch: Dict[str, torch.Tensor],
    tau: float,
    M: int = 3,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
    logits: Optional[torch.Tensor] = None,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    partial_tf: bool = True,
    use_scheduled_sampling: bool = False,
    scheduled_sampling_p: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    future_ids = batch['future_cluster_ids']
    B, F, N = future_ids.shape
    if logits is None or not partial_tf:
        if not partial_tf:
            raise NotImplementedError("Full on-policy ST rollouts are not yet implemented.")
        outputs = model(
            input_cluster_ids=batch['input_cluster_ids'],
            times=batch.get('times'),
            sequence_lengths=batch.get('sequence_lengths'),
            history_mask=batch.get('history_mask'),
            teacher_future_ids=future_ids,
            scheduled_sampling_p=(scheduled_sampling_p if use_scheduled_sampling else 0.0),
        )
        logits = outputs['cluster_logits']
    C = logits.size(-1)
    with torch.no_grad():
        q = per_residue_histogram_from_ids(
            future_ids,
            C,
            future_step_mask=future_step_mask,
            residue_mask=residue_mask,
            label_smoothing=label_smoothing,
            eps=eps,
        )
    p_hat, mask, valid_f = _st_histogram_partial_teacher(
        logits,
        future_ids,
        future_step_mask,
        residue_mask,
        tau=tau,
        M=M,
        eps=eps,
    )
    kl = kl_from_histograms(p_hat, q, eps=eps)
    if residue_mask is not None:
        mask = mask & residue_mask.to(dtype=torch.bool, device=mask.device)
    if future_step_mask is not None:
        step_valid = valid_f.sum(dim=1) > 0
        mask = mask & step_valid
    vals = kl[mask]
    loss = vals.mean() if vals.numel() > 0 else kl.new_tensor(0.0)
    with torch.no_grad():
        entropy = -(p_hat.clamp_min(eps) * p_hat.clamp_min(eps).log()).sum(dim=-1)
        if mask.any():
            entropy_mean = float(entropy[mask].mean().item())
        else:
            entropy_mean = 0.0
    return loss, {"p_hat_entropy": entropy_mean}

