#!/usr/bin/env python3
"""
Simple first-order Markov baseline for cluster-ID sequences.

- Fits a global transition model T[s -> s'] and initial distribution π from the TRAIN split
  using the same data pipeline as training (via create_dataloaders).
- Evaluates on a selected trajectory by propagating distributions p_{t+1} = p_t · T
  (expectation-based, no sampling) and compares visitation histograms vs GT.

Outputs (under output/evaluation_results/markov_baseline/<traj_name>/):
  - distribution_metrics.json: js_hist, l1_hist, pred_entropy, true_entropy
  - occupancy_top30.png, occupancy_bottom30.png

Usage example:
  python scripts/baselines/markov_baseline.py \
    --data_dir /path/to/lmdb_root \
    --metadata_path /path/to/traj_metadata.csv \
    --eval_split test --protein_id 11755 \
    --time_start 100 --time_steps 200 --max_batches 500
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from emprot.data.dataset import create_dataloaders
from emprot.data.data_loader import LMDBLoader
from emprot.data.metadata import MetadataManager


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_split_proteins(data_root: str, metadata_path: str, split: str, seed: int) -> Tuple[list, list]:
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_root,
        metadata_path=metadata_path,
        batch_size=8,
        num_full_res_frames=5,
        history_prefix_frames=0,
        stride=1,
        future_horizon=1,
        num_workers=0,
        seed=seed,
    )
    if split == 'train':
        ds = train_loader.dataset
    elif split in ('val', 'valid', 'validation'):
        ds = val_loader.dataset
    else:
        ds = test_loader.dataset
    # Also return full metadata for resolving other splits if needed
    return list(ds.protein_metadata), list(train_loader.dataset.all_protein_metadata)


def _select_protein_by_id(metas: list, protein_id: str) -> Dict:
    pid = str(protein_id)
    for m in metas:
        name = m.get('traj_name') or os.path.basename(m.get('path', ''))
        if name == pid:
            return m
    # Try dynamic id (3rd token)
    for m in metas:
        name = m.get('traj_name') or os.path.basename(m.get('path', ''))
        parts = name.split('_')
        if len(parts) >= 3 and parts[2] == pid:
            return m
    raise ValueError(f"protein_id '{protein_id}' not found in split metadata")


def _load_trajectory(traj_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    with LMDBLoader(traj_path) as loader:
        meta = loader.get_metadata()
        T = int(meta['num_frames'])
        Ys = []
        times = []
        for t in range(T):
            fr = loader.load_frame(t)
            if 'cluster_ids' not in fr:
                raise KeyError(f"cluster_ids missing in frame {t}")
            Ys.append(fr['cluster_ids'].astype(np.int32))
            times.append(float(t))
        traj_name = meta.get('traj_name', os.path.basename(traj_path))
    Y_all = np.stack(Ys, axis=0)
    times_all = np.asarray(times, dtype=np.float32)
    return traj_name, Y_all, times_all


def fit_markov_from_train(
    data_dir: str,
    metadata_path: str,
    *,
    batch_size: int = 16,
    num_full_res_frames: int = 5,
    stride: int = 1,
    future_horizon: int = 1,
    max_batches: int = -1,
    seed: int = 42,
) -> Tuple[Dict[int, Dict[int, int]], np.ndarray]:
    """Return (transition_counts, pi_counts).

    transition_counts: dict[src][dst] = count (sparse)
    pi_counts: counts over initial states (1D array sized to max seen id + 1)
    """
    train_loader, _, _ = create_dataloaders(
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

    trans = defaultdict(lambda: defaultdict(int))  # type: ignore
    pi_counts = np.zeros(1, dtype=np.int64)  # will grow dynamically

    def _grow(arr: np.ndarray, new_max: int) -> np.ndarray:
        if new_max < arr.size:
            return arr
        new_size = new_max + 1
        new_arr = np.zeros(new_size, dtype=arr.dtype)
        new_arr[: arr.size] = arr
        return new_arr

    processed = 0
    for bi, batch in enumerate(train_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        ids = batch['input_cluster_ids']  # (B,T,N)
        B, T, N = ids.shape
        fut = batch.get('future_cluster_ids', None)  # (B,F,N) or None
        F = int(fut.shape[1]) if (fut is not None and fut.dim() == 3) else 0

        # Build combined sequence along time: [T inputs] + [F futures]
        seq = torch.full((B, T + F, N), -1, dtype=torch.long)
        seq[:, :T, :] = ids
        if F > 0:
            seq[:, T:T + F, :] = fut

        # Initial state counts (use first valid frame per sample)
        first = ids[:, 0, :].view(-1)
        first_np = first[first >= 0].cpu().numpy().astype(np.int64)
        if first_np.size:
            pi_counts = _grow(pi_counts, int(first_np.max()))
            binc = np.bincount(first_np, minlength=pi_counts.size)
            pi_counts[: binc.size] += binc

        # Transitions across time axis
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
        processed += 1

    return trans, pi_counts


def build_transition_model(trans_counts: Dict[int, Dict[int, int]], alpha: float = 1e-6, topk: int = 256) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return sparse row-normalized transitions per source: {s: (dests, probs)}.

    Keeps at most topk destinations per source by count and applies add-alpha smoothing
    within the kept set.
    """
    model: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for s, row in trans_counts.items():
        if not row:
            continue
        # sort by count desc
        items = sorted(row.items(), key=lambda x: x[1], reverse=True)
        if topk > 0:
            items = items[:topk]
        dests = np.asarray([d for d, _ in items], dtype=np.int64)
        counts = np.asarray([c for _, c in items], dtype=np.float64)
        probs = counts + float(alpha)
        probs = probs / probs.sum().clip(min=1e-12)
        model[int(s)] = (dests, probs)
    return model


def predict_expectation_over_horizon(
    model: Dict[int, Tuple[np.ndarray, np.ndarray]],
    last_ids: np.ndarray,  # (N,)
    steps: int,
) -> np.ndarray:
    """Compute expected per-step distributions over classes aggregated across residues and steps.

    Returns p_hat (C_sparse,) as a sparse accumulator over seen classes only, implemented
    as a dict-like dense vector sized to max class id encountered in transitions.
    """
    # Use dictionary accumulator to avoid allocating large dense C
    agg: Dict[int, float] = defaultdict(float)

    # Current per-residue distribution represented as current state ids (for step 0)
    curr_states = last_ids.copy()

    for _ in range(int(steps)):
        # For each residue, distribute its mass according to the row of its current state
        # Since we propagate expectations, each residue contributes the same row dist.
        for s in curr_states.tolist():
            if s < 0:
                continue
            if s in model:
                dests, probs = model[s]
                for d, p in zip(dests.tolist(), probs.tolist()):
                    agg[int(d)] += float(p)
        # Update curr_states for next step by most likely transition (cheap heuristic)
        # This keeps the chain moving; exact expectation would keep full distributions per residue
        # but is O(N * topk) anyway; choosing argmax row is a pragmatic compromise.
        next_states = curr_states.copy()
        for i, s in enumerate(curr_states.tolist()):
            if s in model:
                dests, probs = model[s]
                if dests.size > 0:
                    next_states[i] = int(dests[int(np.argmax(probs))])
        curr_states = next_states

    # Convert agg dict to dense vector sized to max key + 1
    if not agg:
        return np.zeros((0,), dtype=np.float64)
    Cmax = max(agg.keys()) + 1
    out = np.zeros((Cmax,), dtype=np.float64)
    for k, v in agg.items():
        out[int(k)] = float(v)
    return out


def compute_js_l1(q_counts: np.ndarray, p_agg: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    # Equalize lengths
    Cq = q_counts.size
    Cp = p_agg.size
    C = max(Cq, Cp)
    if Cq < C:
        q_counts = np.pad(q_counts, (0, C - Cq))
    if Cp < C:
        p_agg = np.pad(p_agg, (0, C - Cp))
    q = q_counts.astype(np.float64)
    q = q / max(q.sum(), eps)
    p = p_agg.astype(np.float64)
    p = p / max(p.sum(), eps)
    m = 0.5 * (q + p)
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    js = 0.5 * (kl_qm + kl_pm)
    l1 = 0.5 * np.abs(q - p).sum()
    pe = -np.sum(p * np.log(p + eps))
    qe = -np.sum(q * np.log(q + eps))
    return {
        'js_hist': float(js),
        'l1_hist': float(l1),
        'pred_entropy': float(pe),
        'true_entropy': float(qe),
    }


def plot_occupancy(q_counts: np.ndarray, p_agg: np.ndarray, out_dir: Path, traj_name: str):
    # Normalize to distributions
    C = max(q_counts.size, p_agg.size)
    if q_counts.size < C:
        q_counts = np.pad(q_counts, (0, C - q_counts.size))
    if p_agg.size < C:
        p_agg = np.pad(p_agg, (0, C - p_agg.size))
    q = q_counts.astype(np.float64)
    p = p_agg.astype(np.float64)
    q = q / max(q.sum(), 1e-12)
    p = p / max(p.sum(), 1e-12)

    order_desc = np.argsort(-q)
    order_asc = np.argsort(q)

    def _plot(indices: np.ndarray, title: str, fname: str):
        k = min(30, indices.size)
        idx = indices[:k]
        x = np.arange(k)
        plt.figure(figsize=(10, 4))
        plt.bar(x - 0.2, q[idx], width=0.4, label='GT')
        plt.bar(x + 0.2, p[idx], width=0.4, label='Pred')
        plt.xticks(x, [str(i) for i in idx], rotation=90)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    _plot(order_desc, f"Occupancy Top-30 | {traj_name}", f"occupancy_top30.png")
    _plot(order_asc, f"Occupancy Bottom-30 | {traj_name}", f"occupancy_bottom30.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--metadata_path', required=True)
    ap.add_argument('--fit_split', type=str, default='train')
    ap.add_argument('--eval_split', type=str, default='test')
    ap.add_argument('--protein_id', type=str, default=None)
    ap.add_argument('--time_start', type=int, default=0)
    ap.add_argument('--time_steps', type=int, default=200)
    ap.add_argument('--max_batches', type=int, default=500)
    ap.add_argument('--num_full_res_frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--future_horizon', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topk', type=int, default=256)
    ap.add_argument('--alpha', type=float, default=1e-6)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    # Fit transitions and initial distribution from TRAIN
    trans_counts, pi_counts = fit_markov_from_train(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=16,
        num_full_res_frames=args.num_full_res_frames,
        stride=args.stride,
        future_horizon=args.future_horizon,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    trans_model = build_transition_model(trans_counts, alpha=float(args.alpha), topk=int(args.topk))

    # Resolve eval protein
    metas_eval, all_metas = _list_split_proteins(args.data_dir, args.metadata_path, args.eval_split, args.seed)
    m = _select_protein_by_id(metas_eval, args.protein_id) if args.protein_id is not None else metas_eval[0]
    traj_path = m['path']
    traj_name = m.get('traj_name') or os.path.basename(traj_path)

    # Load full trajectory for evaluation
    traj_name, Y_all, _ = _load_trajectory(traj_path)
    T_total, N = Y_all.shape
    t0 = int(args.time_start)
    steps = int(min(max(1, args.time_steps), max(0, T_total - t0)))
    if t0 <= 0:
        last_ids = np.zeros((N,), dtype=np.int64)
    else:
        last_ids = Y_all[t0 - 1].copy().astype(np.int64)
        last_ids[last_ids < 0] = 0
    gt_window = Y_all[t0:t0 + steps]

    # Predicted aggregate distribution across steps and residues (sparse dense vector)
    p_agg = predict_expectation_over_horizon(trans_model, last_ids=last_ids, steps=steps)

    # Ground-truth histogram over the same window
    gt_flat = gt_window.reshape(-1)
    gt_flat = gt_flat[gt_flat >= 0]
    q_counts = np.bincount(gt_flat.astype(np.int64), minlength=max(gt_flat.max() + 1 if gt_flat.size else 0, p_agg.size))

    # Metrics
    metrics = compute_js_l1(q_counts=q_counts, p_agg=p_agg)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path('output') / 'evaluation_results' / 'markov_baseline' / traj_name / 'autoregressive_eval'
    _ensure_dir(out_dir)

    # Save metrics
    with open(out_dir / 'distribution_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_occupancy(q_counts=q_counts, p_agg=p_agg, out_dir=out_dir, traj_name=traj_name)

    # Params (for bookkeeping)
    params = {
        'fit_split': args.fit_split,
        'eval_split': args.eval_split,
        'protein_id': args.protein_id,
        'time_start': t0,
        'time_steps': steps,
        'topk': int(args.topk),
        'alpha': float(args.alpha),
        'num_full_res_frames': int(args.num_full_res_frames),
        'stride': int(args.stride),
        'future_horizon': int(args.future_horizon),
        'max_batches': int(args.max_batches),
    }
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved Markov baseline results to {out_dir}")


if __name__ == '__main__':
    main()


