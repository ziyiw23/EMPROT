#!/usr/bin/env python3
"""
Autoregressive rollout comparison between transformer model and Markov baseline.

Inputs:
- ckpt: path to trained checkpoint (best.pt or final.pt)
- data_root: LMDB root directory
- split: which split to sample protein from (val/test/train)
- time_start: index of first predicted frame (history is [0..time_start-1])
- time_steps: number of autoregressive steps to roll out

Outputs in --output_dir:
- <traj>_residue_panel.png           (GT vs Pred vs Markov cluster-id lines for selected residues)
- <traj>_residue_visitation.png     (per-residue visitation bars for GT vs Transformer vs Markov)

This script wraps utilities defined in scripts/autoregressive_eval.py to keep logic small.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.autoregressive_eval import (
    load_model,
    load_sequence,
    select_residues,
    plot_residue_trajectories_pretty,
    rollout_autoregressive,
    _remap_neighbors_to_col_space,
    plot_residue_visitation_bars,
    compute_distributional_metrics,
    compute_per_residue_metrics,
    plot_correlation_matrices,
)
from scripts.evaluate_dynamics import evaluate_batch_dynamics


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_markov_checkpoint(path: str) -> Optional[dict]:
    if not path:
        return None
    ckpt = Path(path).expanduser()
    if not ckpt.is_file():
        print(f"[WARN] Markov checkpoint not found at {ckpt}")
        return None
    try:
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load Markov checkpoint {ckpt}: {exc}")
        return None
    transitions = {}
    argmax = {}
    for key, entry in (payload.get('transitions') or {}).items():
        try:
            state = int(key)
            dests = np.asarray(entry.get('dests', []), dtype=np.int64)
            probs = np.asarray(entry.get('probs', []), dtype=np.float64)
            transitions[state] = (dests, probs)
            if dests.size > 0:
                argmax[state] = int(dests[int(np.argmax(probs))])
        except Exception:
            continue
    if not transitions:
        print(f"[WARN] No transitions found in Markov checkpoint {ckpt}")
        return None
    return {'transitions': transitions, 'argmax': argmax, 'path': str(ckpt)}


def _run_markov_rollout(markov_model: Optional[dict],
                        Y_all: np.ndarray,
                        time_start: int,
                        time_steps: int,
                        seed: Optional[int] = None) -> Optional[np.ndarray]:
    if markov_model is None or time_steps <= 0:
        return None
    transitions = markov_model.get('transitions', {})
    if not transitions:
        return None
    rng = np.random.default_rng(seed)
    T_total, N = Y_all.shape
    if time_start < 0 or time_start > T_total:
        return None
    steps = int(time_steps)
    preds = np.full((steps, N), -1, dtype=np.int32)
    if time_start <= 0:
        curr = Y_all[0].copy()
    else:
        curr = Y_all[min(time_start - 1, T_total - 1)].copy()
    curr = np.where(curr < 0, 0, curr).astype(np.int64)
    for step in range(steps):
        next_ids = curr.copy()
        for i, state in enumerate(curr):
            row = transitions.get(int(state))
            if row is None:
                continue
            dests, probs = row
            if dests.size == 0:
                continue
            idx = int(rng.choice(dests.size, p=probs))
            next_ids[i] = int(dests[idx])
        preds[step] = next_ids.astype(np.int32)
        curr = next_ids
    return preds


def main():
    ap = argparse.ArgumentParser(description='Autoregressive rollout with Markov baseline comparison')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--time_start', type=int, required=True)
    ap.add_argument('--time_steps', type=int, required=True)
    ap.add_argument('--recent_full_frames', type=int, default=8)
    ap.add_argument('--k_residues', type=int, default=5)
    ap.add_argument('--residue_select', type=str, default='most_change', help="Mode: random, most_change, uniform, or manual:1,2,3")
    ap.add_argument('--protein_id', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default='', help='If empty, derived from ckpt: output/evaluation_results/{run}/autoregressive_eval')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    # decoding (nucleus-only path)
    ap.add_argument('--decode_mode', type=str, default='sample', choices=['sample'])
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.98)
    # extra plots/analysis
    ap.add_argument('--plot_hist', action='store_true', help='Plot GT vs rollout occupancy histograms')
    ap.add_argument('--plot_corr', action='store_true', help='Plot correlation matrices of state changes')
    ap.add_argument('--hist_topk', type=int, default=30, help='Top-K clusters by GT freq to show')
    # markov baseline comparison
    ap.add_argument('--markov_ckpt', type=str, default='', help='Path to serialized Markov baseline checkpoint (.pkl)')
    ap.add_argument('--markov_label', type=str, default='Markov baseline')

    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    # Derive default output dir from ckpt if not provided
    if isinstance(args.output_dir, str) and args.output_dir.strip():
        out_dir = Path(args.output_dir)
    else:
        ckpt_p = Path(args.ckpt).resolve()
        run_name = ckpt_p.parent.name
        base = None
        for p in ckpt_p.parents:
            if p.name == 'output':
                base = p
                break
        if base is None:
            base = ckpt_p.parent.parent if ckpt_p.parent.parent is not None else ckpt_p.parent
        out_dir = base / 'evaluation_results' / run_name / 'autoregressive_eval'
    ensure_dir(out_dir)
    markov_model = _load_markov_checkpoint(args.markov_ckpt.strip()) if isinstance(args.markov_ckpt, str) and args.markov_ckpt.strip() else None
    markov_label = args.markov_label if isinstance(args.markov_label, str) and args.markov_label.strip() else 'Markov baseline'

    # Load model
    model, _cfg, id2col, col2id, col2id_array = load_model(args.ckpt, device, use_sparse_logits=True)
    # Respect requested K
    try:
        setattr(model, 'recent_full_frames', int(args.recent_full_frames))
    except Exception:
        pass

    # Load a single trajectory (Y_all: [T,N])
    traj_name, Y_all, _ = load_sequence(args.data_root, args.split, protein_id=args.protein_id, seed=int(args.seed))
    T_total, N = Y_all.shape
    if args.time_start + args.time_steps > T_total:
        raise ValueError(f"Requested window [{args.time_start}, {args.time_start+args.time_steps-1}] exceeds length {T_total}")

    # Autoregressive rollout (transformer)
    eval_out = rollout_autoregressive(
        model=model,
        Y_all=Y_all,
        time_start=int(args.time_start),
        time_steps=int(args.time_steps),
        device=device,
        recent_full_frames=int(args.recent_full_frames),
        col2id_array=col2id_array,
        col2id=col2id or {},
        decode_mode='sample',
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        simple_nucleus=True,
    )
    markov_pred = _run_markov_rollout(markov_model, Y_all, int(args.time_start), int(args.time_steps), seed=int(args.seed)) if markov_model is not None else None
    if markov_model is not None and markov_pred is None:
        print("[WARN] Markov baseline rollout unavailable; skipping comparison.")
    ridxs = select_residues(eval_out.gt, int(args.k_residues), args.residue_select, int(args.seed))
    times_abs = np.arange(args.time_start, args.time_start + args.time_steps, dtype=np.float32)
    times_ns = times_abs * 0.2
    plot_residue_trajectories_pretty(
        traj_name,
        times_ns,
        eval_out.gt,
        eval_out.pred,
        ridxs,
        out_dir / f'{traj_name}_residue_panel.png',
        pred_label='Transformer autoregressive',
    )
    # Save arrays for downstream tools (e.g., UI live plotting)
    try:
        import numpy as _np  # noqa: F401
        from pathlib import Path as _Path  # noqa: F401
        payload = {
            'gt': eval_out.gt.astype(np.int32),
            'pred': eval_out.pred.astype(np.int32),
            'times_abs': times_abs.astype(np.float32),
            'times_ns': times_ns.astype(np.float32),
            'ridxs': np.asarray(ridxs, dtype=np.int32),
        }
        if markov_pred is not None:
            payload['markov_pred'] = markov_pred.astype(np.int32)
            payload['markov_label'] = np.asarray(markov_label)
        _np.savez(out_dir / f'{traj_name}_rollout_arrays.npz', **payload)
    except Exception:
        pass

    # Summary per-residue visitation (grid) for the same residues
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        if markov_pred is not None and markov_pred.size > 0:
            valid_markov = markov_pred[markov_pred >= 0]
            if valid_markov.size > 0:
                num_classes = int(max(num_classes, int(valid_markov.max()) + 1))
        plot_residue_visitation_bars(
            traj_name=traj_name,
            Y_gt=eval_out.gt,
            Y_pred=eval_out.pred,
            residue_indices=ridxs,
            num_classes=int(num_classes),
            out_path=out_dir / f'{traj_name}_residue_visitation.png',
            topk=int(getattr(args, 'hist_topk', 20)),
            Y_baseline=markov_pred,
            baseline_label=markov_label,
        )
    except Exception:
        pass

    # 3) Optional: GT vs prediction occupancy histogram + metrics
    if bool(getattr(args, 'plot_corr', False)):
        try:
            plot_correlation_matrices(
                eval_out.gt,
                eval_out.pred,
                out_dir / f'{traj_name}_correlation_matrices.png',
                title_suffix=f'({traj_name})'
            )
        except Exception as e:
            print(f"[WARN] Failed to plot correlation matrices: {e}")

    if bool(getattr(args, 'plot_hist', False)):
        try:
            from scripts.autoregressive_eval import plot_histograms
            plot_histograms(
                eval_out.gt, 
                eval_out.pred, 
                out_dir / f'{traj_name}_visitation_histograms.png',
                top_k=args.hist_topk,
                pred_label='Transformer',
                baseline=markov_pred,
                baseline_label=markov_label
            )
        except Exception as e:
            print(f"[WARN] Failed to plot histograms: {e}")

    # Distributional metrics JSON (parity with full evaluator)
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        if markov_pred is not None and markov_pred.size > 0:
            valid_markov = markov_pred[markov_pred >= 0]
            if valid_markov.size > 0:
                num_classes = int(max(num_classes, int(valid_markov.max()) + 1))
        dist = compute_distributional_metrics(eval_out.gt, eval_out.pred, int(num_classes))
        dist_per = compute_per_residue_metrics(eval_out.gt, eval_out.pred, int(num_classes), top_k=10)
        dist['per_residue'] = dist_per
        
        # Compute dynamics metrics
        dyn_metrics = evaluate_batch_dynamics(eval_out.gt, eval_out.pred)
        dist['dynamics'] = dyn_metrics
        
        if markov_pred is not None:
            dist_markov = compute_distributional_metrics(eval_out.gt, markov_pred, int(num_classes))
            dist_markov['per_residue'] = compute_per_residue_metrics(eval_out.gt, markov_pred, int(num_classes), top_k=10)
            dist_markov['dynamics'] = evaluate_batch_dynamics(eval_out.gt, markov_pred)
            dist_markov['label'] = markov_label
            dist['markov_baseline'] = dist_markov
        import json as _json
        with open(out_dir / 'distribution_metrics.json', 'w') as f:
            _json.dump(dist, f, indent=2)

        # Generate comparison plot
        try:
            from scripts.visualize_metrics_comparison import plot_comparison
            plot_comparison(str(out_dir / 'distribution_metrics.json'), str(out_dir / 'metrics_comparison.png'))
        except Exception as e:
            print(f"[WARN] Failed to generate metrics comparison plot: {e}")
            
    except Exception as e:
        print(f"[ERROR] Failed to compute/save distribution metrics: {e}")
        import traceback
        traceback.print_exc()

    print(f"Wrote attention + rollout plots to: {out_dir}")


def _ensure_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

if __name__ == '__main__':
    main()
