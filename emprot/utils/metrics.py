#!/usr/bin/env python3
"""Metrics for training."""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from emprot.losses import aggregated_probability_kl_loss


def _valid_mask(targets: torch.Tensor,
                future_step_mask: Optional[torch.Tensor],
                residue_mask: Optional[torch.Tensor]) -> torch.Tensor:
    B, F, N = targets.shape
    device = targets.device
    if future_step_mask is None:
        future_step_mask = torch.ones(B, F, dtype=torch.bool, device=device)
    if residue_mask is None:
        residue_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    M = future_step_mask[:, :, None] & residue_mask[:, None, :]
    return M & (targets >= 0)


def _acc_from_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> float:
    if not valid.any():
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & valid
    return float(correct.sum().item() / max(1, int(valid.sum().item())))


def _topk_from_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, k: int = 5) -> float:
    if not valid.any():
        return 0.0
    topk = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1).indices  # (..., k)
    tgt = targets.unsqueeze(-1)
    hit = (topk == tgt).any(dim=-1) & valid
    return float(hit.sum().item() / max(1, int(valid.sum().item())))


def _mtp_entropy(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    if not valid.any():
        return {'mtp': 0.0, 'entropy': 0.0}
    probs = F.softmax(logits, dim=-1)
    # True-class prob
    idx = targets.clamp_min(0).unsqueeze(-1)
    gather = torch.gather(probs, dim=-1, index=idx).squeeze(-1)
    mtp = float(gather[valid].mean().item()) if valid.any() else 0.0
    # Entropy
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    entropy = float(ent[valid].mean().item()) if valid.any() else 0.0
    return {'mtp': mtp, 'entropy': entropy}


def _ece(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, bins: int = 10) -> float:
    if not valid.any():
        return 0.0
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).to(torch.float32)
    conf_v = conf[valid]
    corr_v = correct[valid]
    if conf_v.numel() == 0:
        return 0.0
    edges = torch.linspace(0, 1, steps=bins + 1, device=conf_v.device)
    idx = torch.bucketize(conf_v, edges, right=True) - 1  # [0..bins-1]
    ece = 0.0
    total = float(conf_v.numel())
    for b in range(bins):
        mask = (idx == b)
        if not mask.any():
            continue
        conf_mean = float(conf_v[mask].mean().item())
        acc_mean = float(corr_v[mask].mean().item())
        ece += abs(acc_mean - conf_mean) * (float(mask.sum().item()) / total)
    return float(ece)


def compute_classification_metrics(
    logits: torch.Tensor,           # (B,F,N,C)
    targets: torch.Tensor,          # (B,F,N)
    *,
    input_cluster_ids: Optional[torch.Tensor] = None,  # (B,T,N)
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    compute_ece: bool = True,
) -> Dict[str, Union[float, torch.Tensor]]:
    B, F, N, C = logits.shape
    valid = _valid_mask(targets, future_step_mask, residue_mask)

    # Per-horizon accuracy
    acc_f = []
    for f in range(F):
        acc_f.append(_acc_from_logits(logits[:, f, :, :], targets[:, f, :], valid[:, f, :]))
    acc_per_horizon = torch.tensor(acc_f)

    # f=1 (index 0) metrics
    f0_logits = logits[:, 0, :, :]
    f0_targets = targets[:, 0, :]
    f0_valid = valid[:, 0, :]
    acc_f1 = _acc_from_logits(f0_logits, f0_targets, f0_valid)
    top5_f1 = _topk_from_logits(f0_logits, f0_targets, f0_valid, k=5)
    mm = _mtp_entropy(f0_logits, f0_targets, f0_valid)
    ece_f1 = _ece(f0_logits, f0_targets, f0_valid) if compute_ece else 0.0

    # Change vs stay
    acc_change_f1 = 0.0
    acc_stay_f1 = 0.0
    if input_cluster_ids is not None and input_cluster_ids.dim() == 3:
        last_inputs = input_cluster_ids[:, -1, :]
        change = (f0_targets != last_inputs) & f0_valid
        stay = (f0_targets == last_inputs) & f0_valid
        if change.any():
            acc_change_f1 = _acc_from_logits(f0_logits, f0_targets, change)
        if stay.any():
            acc_stay_f1 = _acc_from_logits(f0_logits, f0_targets, stay)

    return {
        'acc_f1': acc_f1,
        'top5_f1': top5_f1,
        'mtp_f1': mm['mtp'],
        'entropy_f1': mm['entropy'],
        'ece_f1': ece_f1,
        'acc_change_f1': acc_change_f1,
        'acc_stay_f1': acc_stay_f1,
        'acc_per_horizon': acc_per_horizon,
    }


@torch.no_grad()
def compute_histogram_metrics(
    logits: torch.Tensor,           # (B,F,N,C)
    targets: torch.Tensor,          # (B,F,N)
    *,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    input_cluster_ids: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Order-free distribution metrics between empirical q and predicted marginal p̂.

    Returns mean over batch of:
    - kl_hist: KL(q || p̂)
    - js_hist: Jensen-Shannon(q, p̂)
    - l1_hist: 0.5 * L1 distance between q and p̂
    - pred_entropy: entropy of p̂
    - true_entropy: entropy of q
    - change_ratio_f0: fraction of tokens at f=0 where target != last input
    - change_ratio_any: fraction of tokens across all horizons where target changes vs previous step (approx)
    """
    B, F, N, C = logits.shape
    device = logits.device
    probs = F_softmax = torch.nn.functional.softmax  # alias
    P = probs(logits, dim=-1)  # (B,F,N,C)

    # Build valid mask
    valid = (targets >= 0)  # (B,F,N)
    if future_step_mask is not None:
        valid = valid & future_step_mask.to(dtype=torch.bool, device=targets.device)[:, :, None]
    if residue_mask is not None:
        valid = valid & residue_mask.to(dtype=torch.bool, device=targets.device)[:, None, :]

    kl_list = []
    js_list = []
    l1_list = []
    pe_list = []
    qe_list = []
    cr0_list = []
    cran_list = []

    for b in range(B):
        vb = valid[b]
        if not vb.any():
            continue
        tb = targets[b].clamp_min(0)
        pb = P[b]

        # p̂: predicted marginal over valid tokens
        mask_bn = vb.unsqueeze(-1).to(dtype=pb.dtype)
        p_hat = (pb * mask_bn).sum(dim=(0, 1))  # (C,)
        denom = mask_bn.sum(dim=(0, 1)).clamp_min(1.0)
        p_hat = (p_hat / denom).clamp_min(eps)

        # q: empirical histogram from targets
        t_flat = tb[vb]
        q = torch.bincount(t_flat, minlength=C).to(dtype=pb.dtype, device=device)
        q = (q / q.sum().clamp_min(eps)).clamp_min(eps)

        # KL and JS
        kl = (q * (q.log() - p_hat.log())).sum()
        m = 0.5 * (q + p_hat)
        js = 0.5 * (q * (q.log() - m.log())).sum() + 0.5 * (p_hat * (p_hat.log() - m.log())).sum()
        l1 = 0.5 * (q - p_hat).abs().sum()
        pe = -(p_hat * p_hat.log()).sum()
        qe = -(q * q.log()).sum()

        kl_list.append(float(kl.item()))
        js_list.append(float(js.item()))
        l1_list.append(float(l1.item()))
        pe_list.append(float(pe.item()))
        qe_list.append(float(qe.item()))

        # Change ratios (diagnostic only)
        # f0 vs last input
        if input_cluster_ids is not None and input_cluster_ids.dim() == 3:
            last_hist = input_cluster_ids[b, -1, :]  # (N,)
            f0_valid = vb[0]
            if f0_valid.any():
                f0 = tb[0]
                cr0 = float(((f0 != last_hist) & f0_valid).float().mean().item())
                cr0_list.append(cr0)
        # any change across horizons vs previous step
        if F > 1:
            t_prev = tb[:-1]
            t_curr = tb[1:]
            v_prev = vb[:-1]
            v_curr = vb[1:]
            vpair = v_prev & v_curr
            if vpair.any():
                cran = float(((t_curr != t_prev) & vpair).float().mean().item())
                cran_list.append(cran)

    def _mean(x):
        return float(sum(x) / max(1, len(x)))

    return {
        'kl_hist': _mean(kl_list),
        'js_hist': _mean(js_list),
        'l1_hist': _mean(l1_list),
        'pred_entropy': _mean(pe_list),
        'true_entropy': _mean(qe_list),
        'change_ratio_f0': _mean(cr0_list),
        'change_ratio_any': _mean(cran_list),
    }


@torch.no_grad()
def compute_brier_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
) -> float:
    probs = torch.softmax(logits, dim=-1)
    num_classes = probs.size(-1)
    tgt = targets.clamp_min(0)
    one_hot = torch.nn.functional.one_hot(tgt, num_classes).to(probs.dtype)
    valid = (targets >= 0)
    if future_step_mask is not None:
        valid = valid & future_step_mask[:, :, None].to(dtype=torch.bool, device=targets.device)
    if residue_mask is not None:
        valid = valid & residue_mask[:, None, :].to(dtype=torch.bool, device=targets.device)
    diff = (probs - one_hot) ** 2
    if not valid.any():
        return 0.0
    return float(diff[valid].mean().item())


@torch.no_grad()
def compute_aggregated_kl_metric(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    future_step_mask: Optional[torch.Tensor] = None,
    residue_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> float:
    val = aggregated_probability_kl_loss(
        logits,
        targets,
        future_step_mask=future_step_mask,
        residue_mask=residue_mask,
        label_smoothing=label_smoothing,
        eps=eps,
        reduce="mean",
    )
    return float(val.detach().item())


class MetricsManager:
    def __init__(self):
        pass

    def compute(self, outputs: Dict, batch: Dict) -> Dict:
        return compute_classification_metrics(
            outputs['cluster_logits'],
            batch['future_cluster_ids'],
            input_cluster_ids=batch.get('input_cluster_ids'),
            future_step_mask=batch.get('future_step_mask'),
            residue_mask=batch.get('residue_mask'),
            compute_ece=True,
        )
