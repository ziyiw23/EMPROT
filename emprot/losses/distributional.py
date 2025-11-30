#!/usr/bin/env python3
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def js_divergence(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Elementwise Jensen-Shannon divergence over the last dimension."""
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    M = 0.5 * (P + Q)
    return 0.5 * (P * (P.log() - M.log())).sum(dim=-1) + 0.5 * (Q * (Q.log() - M.log())).sum(dim=-1)


def per_residue_histogram_from_ids(
    future_ids: torch.Tensor,
    num_classes: int,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    B, num_frames, N = future_ids.shape
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


def js_from_histograms(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between histograms over the last dimension.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (kl_from_histograms(m, p, eps=eps) + kl_from_histograms(m, q, eps=eps))


def residue_centric_loss(
    logits: torch.Tensor,
    future_ids: torch.Tensor,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    num_samples: int = 32,
    ce_weight: float = 1.0,
    js_weight: float = 1.0,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Residue-centric composite loss that samples residues per protein and matches
    both token-level CE and visitation JS divergences.
    """
    B, num_frames, N, C = logits.shape
    device = logits.device
    valid = (future_ids >= 0)
    if future_step_mask is not None:
        valid = valid & future_step_mask[:, :, None].to(dtype=torch.bool, device=device)
    if residue_mask is not None:
        valid = valid & residue_mask[:, None, :].to(dtype=torch.bool, device=device)

    has_valid = valid.any(dim=1)
    # Use all valid residues (no subsampling)
    sampled_mask = has_valid  # Using all valid residues for JS

    if not sampled_mask.any():
        zero = logits.new_tensor(0.0)
        return zero, {
            'res_ce_mean': 0.0,
            'res_js_mean': 0.0,
            'res_num_used': 0.0,
        }

    # Use sparse sampling ONLY for JS part to save compute
    token_mask = valid  # Use dense mask for CE (stable training)
    if token_mask.any():
        flat_logits = logits[token_mask]
        flat_targets = future_ids[token_mask].long()
        if label_smoothing > 0.0:
            with torch.no_grad():
                target_one_hot = F.one_hot(flat_targets, num_classes=C).to(flat_logits.dtype)
                target_smooth = (1.0 - label_smoothing) * target_one_hot + label_smoothing / float(C)
            log_probs = torch.log_softmax(flat_logits, dim=-1)
            ce_per = -(target_smooth * log_probs).sum(dim=-1)
        else:
            ce_per = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        ce_loss = ce_per.mean()
    else:
        ce_loss = logits.new_tensor(0.0)

    with torch.no_grad():
        q = per_residue_histogram_from_ids(
            future_ids,
            num_classes=C,
            future_step_mask=future_step_mask,
            residue_mask=residue_mask,
            label_smoothing=0.0,
            eps=eps,
        )
    probs = torch.softmax(logits, dim=-1)
    valid_f = valid.to(dtype=logits.dtype)
    counts = valid_f.sum(dim=1).clamp_min(1.0)[..., None]
    p_avg = (probs * valid_f[..., None]).sum(dim=1) / counts
    js_all = js_from_histograms(p_avg, q, eps=eps)
    js_mask = sampled_mask & has_valid
    js_vals = js_all[js_mask]
    js_loss = js_vals.mean() if js_vals.numel() > 0 else logits.new_tensor(0.0)

    loss = ce_weight * ce_loss + js_weight * js_loss
    debug = {
        'res_ce_mean': float(ce_loss.detach().item()),
        'res_js_mean': float(js_loss.detach().item()),
        'res_num_used': float(js_mask.sum().item()),
    }
    return loss, debug


def dwell_rate_loss_from_logits(
    logits: torch.Tensor,
    future_ids: torch.Tensor,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Match ground-truth stay rate vs predicted stay probability per residue over teacher-forced steps.
    """
    B, F, N, _ = logits.shape
    device = logits.device

    if F <= 1:
        return logits.new_tensor(0.0), {}

    fmask = future_step_mask if future_step_mask is not None else torch.ones(B, F, dtype=torch.bool, device=device)
    fmask = fmask.to(dtype=torch.bool, device=device)
    rmask = residue_mask if residue_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
    rmask = rmask.to(dtype=torch.bool, device=device)

    valid_h = fmask[:, 1:] & fmask[:, :-1]
    gt_curr = future_ids[:, 1:, :]
    gt_prev = future_ids[:, :-1, :]
    valid_pairs = (gt_curr >= 0) & (gt_prev >= 0)
    valid = valid_pairs & valid_h[:, :, None] & rmask[:, None, :]
    if not valid.any():
        return logits.new_tensor(0.0), {'dwell_rate_gt': 0.0, 'dwell_rate_pred': 0.0, 'dwell_mse': 0.0}

    stay_gt = (gt_curr == gt_prev) & valid
    probs = torch.softmax(logits[:, 1:, :, :], dim=-1)
    gather_idx = gt_prev.clamp_min(0).unsqueeze(-1)
    pred_stay_prob = probs.gather(dim=-1, index=gather_idx).squeeze(-1)
    pred_stay_prob = torch.where(valid, pred_stay_prob, torch.zeros_like(pred_stay_prob))

    denom = valid.to(logits.dtype).sum(dim=1).clamp_min(1.0)
    gt_stay_rate = stay_gt.to(logits.dtype).sum(dim=1) / denom
    pred_stay_rate = pred_stay_prob.sum(dim=1) / denom

    rvalid = (denom > 0) & rmask
    if rvalid.any():
        mse = ((gt_stay_rate - pred_stay_rate) ** 2)[rvalid].mean()
        dbg = {
            'dwell_rate_gt': float(gt_stay_rate[rvalid].mean().item()),
            'dwell_rate_pred': float(pred_stay_rate[rvalid].mean().item()),
            'dwell_mse': float(mse.item()),
        }
        return mse, dbg
    return logits.new_tensor(0.0), {'dwell_rate_gt': 0.0, 'dwell_rate_pred': 0.0, 'dwell_mse': 0.0}


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


def transition_row_js_loss_from_logits(
    logits: torch.Tensor,
    future_ids: torch.Tensor,
    min_count: int = 5,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Align predicted next-state distributions with empirical ground truth rows per residue/state.
    """
    B, num_frames, N, C = logits.shape
    device = logits.device
    if num_frames <= 1:
        return logits.new_tensor(0.0), {'row_js_mean': 0.0, 'row_js_rows': 0}

    fmask = future_step_mask if future_step_mask is not None else torch.ones(B, num_frames, dtype=torch.bool, device=device)
    fmask = fmask.to(dtype=torch.bool, device=device)
    rmask = residue_mask if residue_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
    rmask = rmask.to(dtype=torch.bool, device=device)

    valid_h = fmask[:, 1:] & fmask[:, :-1]
    curr = future_ids[:, :-1, :]
    next_ids = future_ids[:, 1:, :]
    valid = (curr >= 0) & (next_ids >= 0) & valid_h[:, :, None] & rmask[:, None, :]
    if not valid.any():
        return logits.new_tensor(0.0), {'row_js_mean': 0.0, 'row_js_rows': 0}

    probs = torch.softmax(logits[:, 1:, :, :], dim=-1)
    b_idx, f_idx, n_idx = torch.nonzero(valid, as_tuple=True)
    curr_states = curr[b_idx, f_idx, n_idx].clamp_min(0)
    next_states = next_ids[b_idx, f_idx, n_idx].clamp_min(0)
    probs_vals = probs[b_idx, f_idx, n_idx]

    num_rows = B * N * C
    device_dtype = logits.dtype

    flat_index = (b_idx * N + n_idx) * C + curr_states

    counts_flat = torch.zeros(num_rows, dtype=device_dtype, device=device)
    counts_flat.index_add_(0, flat_index, torch.ones_like(curr_states, dtype=device_dtype))

    P_rows_flat = torch.zeros(num_rows, C, dtype=device_dtype, device=device)
    P_rows_flat.index_add_(0, flat_index, probs_vals)

    Q_rows_flat = torch.zeros(num_rows, C, dtype=device_dtype, device=device)
    one_next = F.one_hot(next_states, num_classes=C).to(device_dtype)
    Q_rows_flat.index_add_(0, flat_index, one_next)

    counts = counts_flat.clamp_min(1.0).view(B, N, C)
    row_valid = (counts >= float(min_count)) & rmask[:, :, None]

    P_rows = (P_rows_flat / counts_flat.clamp_min(1.0).unsqueeze(-1)).view(B, N, C, C)
    Q_rows = (Q_rows_flat / counts_flat.clamp_min(1.0).unsqueeze(-1)).view(B, N, C, C)

    js_vals = js_divergence(P_rows, Q_rows, eps=eps)
    if row_valid.any():
        loss = js_vals[row_valid].mean()
        dbg = {
            'row_js_mean': float(loss.detach().item()),
            'row_js_rows': int(row_valid.sum().item()),
        }
        return loss, dbg
    return logits.new_tensor(0.0), {'row_js_mean': 0.0, 'row_js_rows': 0}


def coverage_hinge_loss(
    p_hist: torch.Tensor,
    q_hist: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    thr: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Encourage predicted support coverage to match ground truth support width.
    """
    p = p_hist.clamp_min(eps)
    q = q_hist.clamp_min(eps)
    cov_p = (p > thr).to(p.dtype).mean(dim=-1)
    cov_q = (q > thr).to(q.dtype).mean(dim=-1)
    gap = (cov_q - cov_p).clamp_min(0.0)

    if mask is not None:
        weights = mask.to(p.dtype)
        denom = weights.sum().clamp_min(1.0)
        loss = (gap * weights).sum() / denom
        cov_p_mean = (cov_p * weights).sum() / denom
        cov_q_mean = (cov_q * weights).sum() / denom
        gap_mean = (gap * weights).sum() / denom
    else:
        loss = gap.mean()
        cov_p_mean = cov_p.mean()
        cov_q_mean = cov_q.mean()
        gap_mean = gap.mean()

    dbg = {
        'coverage_pred_mean': float(cov_p_mean.detach().item()),
        'coverage_gt_mean': float(cov_q_mean.detach().item()),
        'coverage_gap_mean': float(gap_mean.detach().item()),
    }
    return loss, dbg


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

