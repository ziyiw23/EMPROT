#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple
import json
import os

import torch


def compute_class_counts(
    loader,
    *,
    num_classes: Optional[int] = None,
    horizon_mode: str = "f1",  # "f1" or "all"
) -> torch.Tensor:
    device = torch.device("cpu")
    counts: Optional[torch.Tensor] = None
    for batch in loader:
        targets = batch.get("future_cluster_ids")  # (B,F,N)
        if targets is None:
            continue
        step_mask = batch.get("future_step_mask")  # (B,F)
        res_mask = batch.get("residue_mask")       # (B,N)
        t = targets
        if horizon_mode == "f1":
            t = t[:, :1, :]  # (B,1,N)
            if step_mask is not None:
                step_mask = step_mask[:, :1]
        valid = (t >= 0)
        if step_mask is not None:
            valid = valid & step_mask[:, :, None].to(valid.device)
        if res_mask is not None:
            valid = valid & res_mask[:, None, :].to(valid.device)
        if not valid.any():
            continue
        flat_t = t[valid].to(torch.long)
        if flat_t.numel() == 0:
            continue
        cmax = int(flat_t.max().item()) if flat_t.numel() > 0 else -1
        k = int(max(num_classes or 0, cmax + 1))
        if counts is None:
            counts = torch.zeros(k, dtype=torch.long, device=device)
        elif counts.numel() < k:
            new_c = torch.zeros(k, dtype=torch.long, device=device)
            new_c[: counts.numel()] = counts
            counts = new_c
        counts += torch.bincount(flat_t, minlength=counts.numel())
    if counts is None:
        counts = torch.zeros(int(num_classes or 0), dtype=torch.long)
    return counts.cpu()


def compute_effective_number_weights(
    counts: torch.Tensor,
    *,
    beta: float = 0.999,
    normalize: bool = True,
    clip: Optional[Tuple[float, float]] = (0.1, 10.0),
) -> torch.Tensor:
    counts = counts.to(torch.float32)
    beta = float(beta)
    eps = 1e-12
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / torch.clamp(eff_num, min=eps)
    if normalize and weights.numel() > 0:
        weights = weights * (weights.numel() / torch.clamp(weights.sum(), min=eps))
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        weights = torch.clamp(weights, min=lo, max=hi)
    return weights


def save_vector(vec: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.endswith(".pt"):
        torch.save(vec.cpu(), path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vec.cpu().tolist(), f)


def load_vector(path: str) -> torch.Tensor:
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return torch.tensor(data)


