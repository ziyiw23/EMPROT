#!/usr/bin/env python3
"""
Train a first-order Markov baseline (cluster-ID transition model) and save it as a
pickled checkpoint. The resulting checkpoint can be consumed by attn_rollout_min.py
to compare transformer rollouts against the Markov baseline and ground truth.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from emprot.data.dataset import create_dataloaders


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fit_markov_from_split(
    data_dir: str,
    metadata_path: str,
    split: str,
    *,
    batch_size: int = 16,
    num_full_res_frames: int = 5,
    stride: int = 1,
    future_horizon: int = 1,
    max_batches: int = -1,
    seed: int = 42,
) -> Tuple[Dict[int, Dict[int, int]], np.ndarray]:
    """Return raw transition counts and initial-state counts for the requested split."""
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_full_res_frames=num_full_res_frames,
        history_prefix_frames=0,
        stride=stride,
        future_horizon=future_horizon,
        num_workers=2,
        seed=seed,
    )

    split_key = split.lower()
    if split_key in ('train',):
        loader = train_loader
    elif split_key in ('val', 'valid', 'validation'):
        loader = val_loader
    elif split_key in ('test', 'eval'):
        loader = test_loader
    else:
        raise ValueError(f"Unsupported fit_split '{split}'. Use train/val/test.")

    trans = defaultdict(lambda: defaultdict(int))  # type: ignore[arg-type]
    pi_counts = np.zeros(1, dtype=np.int64)

    def _grow(arr: np.ndarray, new_max: int) -> np.ndarray:
        if new_max < arr.size:
            return arr
        new_size = new_max + 1
        new_arr = np.zeros(new_size, dtype=arr.dtype)
        new_arr[: arr.size] = arr
        return new_arr

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        ids = batch['input_cluster_ids']  # (B,T,N)
        B, T, N = ids.shape
        fut = batch.get('future_cluster_ids', None)
        F = int(fut.shape[1]) if (fut is not None and fut.dim() == 3) else 0

        seq = torch.full((B, T + F, N), -1, dtype=torch.long)
        seq[:, :T, :] = ids
        if F > 0:
            seq[:, T:T + F, :] = fut

        first = ids[:, 0, :].reshape(-1)
        first_np = first[first >= 0].cpu().numpy().astype(np.int64)
        if first_np.size:
            pi_counts = _grow(pi_counts, int(first_np.max()))
            binc = np.bincount(first_np, minlength=pi_counts.size)
            pi_counts[: binc.size] += binc

        prev = seq[:, :-1, :].reshape(-1)
        nex = seq[:, 1:, :].reshape(-1)
        valid = (prev >= 0) & (nex >= 0)
        if valid.any():
            pv = prev[valid].cpu().numpy().astype(np.int64)
            nv = nex[valid].cpu().numpy().astype(np.int64)
            base = int(np.max([pv.max() if pv.size else 0, nv.max() if nv.size else 0]) + 1)
            codes = pv * base + nv
            uniq, cnts = np.unique(codes, return_counts=True)
            for code, c in zip(uniq.tolist(), cnts.tolist()):
                s = code // base
                d = code % base
                trans[int(s)][int(d)] += int(c)

    return trans, pi_counts


def build_transition_model(trans_counts: Dict[int, Dict[int, int]], alpha: float = 1e-6, topk: int = 256) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Normalize transition counts into per-source distributions."""
    model: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for s, row in trans_counts.items():
        if not row:
            continue
        items = sorted(row.items(), key=lambda x: x[1], reverse=True)
        if topk > 0:
            items = items[:topk]
        dests = np.asarray([d for d, _ in items], dtype=np.int64)
        counts = np.asarray([c for _, c in items], dtype=np.float64)
        probs = counts + float(alpha)
        probs = probs / probs.sum().clip(min=1e-12)
        model[int(s)] = (dests, probs)
    return model


def save_markov_checkpoint(path: Path,
                           transitions: Dict[int, Tuple[np.ndarray, np.ndarray]],
                           pi_counts: np.ndarray,
                           *,
                           alpha: float,
                           topk: int,
                           fit_split: str,
                           num_full_res_frames: int,
                           stride: int,
                           future_horizon: int,
                           max_batches: int,
                           seed: int) -> None:
    payload = {
        'transitions': {
            int(s): {'dests': dests.tolist(), 'probs': probs.tolist()}
            for s, (dests, probs) in transitions.items()
        },
        'pi_counts': pi_counts.astype(np.int64).tolist(),
        'alpha': float(alpha),
        'topk': int(topk),
        'fit_split': fit_split,
        'num_full_res_frames': int(num_full_res_frames),
        'stride': int(stride),
        'future_horizon': int(future_horizon),
        'max_batches': int(max_batches),
        'seed': int(seed),
    }
    _ensure_dir(path.parent)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a Markov baseline and save a checkpoint.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--metadata_path', required=True)
    ap.add_argument('--fit_split', type=str, default='train', choices=['train', 'val', 'valid', 'validation', 'test', 'eval'])
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_full_res_frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--future_horizon', type=int, default=1)
    ap.add_argument('--max_batches', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topk', type=int, default=256)
    ap.add_argument('--alpha', type=float, default=1e-6)
    ap.add_argument('--model_ckpt', type=str, default='')
    args = ap.parse_args()

    trans_counts, pi_counts = fit_markov_from_split(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split=args.fit_split,
        batch_size=int(args.batch_size),
        num_full_res_frames=int(args.num_full_res_frames),
        stride=int(args.stride),
        future_horizon=int(args.future_horizon),
        max_batches=int(args.max_batches),
        seed=int(args.seed),
    )
    model = build_transition_model(trans_counts, alpha=float(args.alpha), topk=int(args.topk))

    ckpt_path = args.model_ckpt.strip()
    if not ckpt_path:
        ckpt_path = Path('output') / 'markov_ckpts' / f'markov_{args.fit_split}.pkl'
    else:
        ckpt_path = Path(ckpt_path)

    save_markov_checkpoint(
        ckpt_path,
        model,
        pi_counts,
        alpha=float(args.alpha),
        topk=int(args.topk),
        fit_split=args.fit_split,
        num_full_res_frames=int(args.num_full_res_frames),
        stride=int(args.stride),
        future_horizon=int(args.future_horizon),
        max_batches=int(args.max_batches),
        seed=int(args.seed),
    )

    num_states = len(model)
    avg_branch = np.mean([dests.size for dests, _ in model.values()]) if model else 0.0
    print(f"[Markov] trained on split='{args.fit_split}' | states={num_states} | avg_branch={avg_branch:.1f}")
    print(f"[Markov] checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()

