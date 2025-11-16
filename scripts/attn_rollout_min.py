#!/usr/bin/env python3
"""
Minimal attention visualization + autoregressive rollout (cluster-id line plots).

Inputs:
- ckpt: path to trained checkpoint (best.pt or final.pt)
- data_root: LMDB root directory
- split: which split to sample protein from (val/test/train)
- time_start: index of first predicted frame (history is [0..time_start-1])
- time_steps: number of autoregressive steps to roll out

Outputs in --output_dir:
- <traj>_attention_over_frames.png   (attention mass per history frame)
- <traj>_residue_panel.png           (GT vs Pred cluster-id lines for selected residues)

This script wraps utilities defined in scripts/autoregressive_eval.py to keep logic small.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

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
    capture_temporal_attention_per_frame,
    select_residues,
    plot_temporal_attention_over_frames,
    plot_residue_trajectories_pretty,
    rollout_autoregressive,
    _remap_neighbors_to_col_space,
    plot_residue_visitation_bars,
    plot_residue_visitation_single,
    compute_distributional_metrics,
    compute_per_residue_metrics,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description='Minimal attention viz + autoregressive rollout')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--time_start', type=int, required=True)
    ap.add_argument('--time_steps', type=int, required=True)
    ap.add_argument('--recent_full_frames', type=int, default=8)
    ap.add_argument('--k_residues', type=int, default=5)
    ap.add_argument('--residue_select', type=str, default='most_change', choices=['random', 'most_change', 'uniform'])
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
    ap.add_argument('--hist_topk', type=int, default=30, help='Top-K clusters by GT freq to show')
    ap.add_argument('--analyze_step_attention', action='store_true', help='Analyze stepwise attention over K frames + latents during rollout')
    # single-step raw attention visualization
    ap.add_argument('--plot_step_attn', action='store_true', help='Plot raw attention matrix at a specific rollout step')
    ap.add_argument('--attn_step', type=int, default=0, help='Which rollout step to inspect (0 = first predicted)')
    # multi-step grid (no averaging over steps)
    ap.add_argument('--plot_steps_grid', action='store_true', help='Plot multiple raw attention steps in one grid')
    ap.add_argument('--attn_steps', type=str, default='', help='Comma-separated step indices, or "random"')
    ap.add_argument('--attn_random_k', type=int, default=5, help='How many random steps if attn_steps=random')
    # softmax-at-step visualization (per-residue prob bars with GT marker)
    ap.add_argument('--plot_softmax_step', action='store_true', help='Plot per-residue softmax distribution at a rollout step vs GT')
    ap.add_argument('--softmax_step', type=int, default=0, help='Which rollout step to visualize (0-based)')
    ap.add_argument('--softmax_topk', type=int, default=20, help='Top-K classes by prob to show per residue')

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

    # 1) Attention visualization on history [0..time_start-1]
    if args.time_start >= 2:
        hist_ids = torch.from_numpy(np.where(Y_all[:args.time_start] < 0, 0, Y_all[:args.time_start])).long().unsqueeze(0).to(device)
        times = torch.arange(args.time_start, dtype=torch.float32, device=device).view(1, -1) * 0.2
        hist_mask = torch.ones(1, args.time_start, N, dtype=torch.bool, device=device)
        seq_lens = torch.tensor([args.time_start], dtype=torch.long, device=device)
        attn_pf = capture_temporal_attention_per_frame(model, hist_ids, times, hist_mask, seq_lens)
        ridxs_attn: List[int] = select_residues(Y_all[max(0, args.time_start - args.recent_full_frames): args.time_start], int(args.k_residues), args.residue_select, int(args.seed))
        times_hist_ns = np.arange(args.time_start, dtype=np.float32) * 0.2
        plot_temporal_attention_over_frames(times_hist_ns, attn_pf, ridxs_attn, out_dir / f'{traj_name}_attention_over_frames.png', k_recent=int(args.recent_full_frames))
        # Mean attention over frames (single line summary)
        try:
            _plot_mean_attention_over_frames(times_hist_ns, attn_pf, out_dir / f'{traj_name}_attention_mean_over_frames.png')
        except Exception:
            pass

    # 2) Simple autoregressive rollout (nucleus-only)
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
    ridxs = select_residues(eval_out.gt, int(args.k_residues), args.residue_select, int(args.seed))
    times_abs = np.arange(args.time_start, args.time_start + args.time_steps, dtype=np.float32)
    times_ns = times_abs * 0.2
    plot_residue_trajectories_pretty(traj_name, times_ns, eval_out.gt, eval_out.pred, ridxs, out_dir / f'{traj_name}_residue_panel.png')
    # Save arrays for downstream tools (e.g., UI live plotting)
    try:
        import numpy as _np  # noqa: F401
        from pathlib import Path as _Path  # noqa: F401
        _np.savez(out_dir / f'{traj_name}_rollout_arrays.npz',
                  gt=eval_out.gt.astype(np.int32),
                  pred=eval_out.pred.astype(np.int32),
                  times_abs=times_abs.astype(np.float32),
                  times_ns=times_ns.astype(np.float32),
                  ridxs=np.asarray(ridxs, dtype=np.int32))
    except Exception:
        pass

    # Summary per-residue visitation (grid) for the same residues
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        plot_residue_visitation_bars(
            traj_name=traj_name,
            Y_gt=eval_out.gt,
            Y_pred=eval_out.pred,
            residue_indices=ridxs,
            num_classes=int(num_classes),
            out_path=out_dir / f'{traj_name}_residue_visitation.png',
            topk=int(getattr(args, 'hist_topk', 20)),
        )
    except Exception:
        pass

    # 3) Optional: GT vs prediction occupancy histogram + metrics
    if bool(getattr(args, 'plot_hist', False)):
        _plot_histograms_and_metrics(traj_name, eval_out.gt, eval_out.pred, out_dir, topk=int(args.hist_topk))

    # 4) Optional: Stepwise attention analysis across rollout using K frames + latents
    if bool(getattr(args, 'analyze_step_attention', False)):
        _analyze_step_attention(model, Y_all, int(args.time_start), int(args.time_steps), int(args.recent_full_frames), out_dir, ridxs)

    if bool(getattr(args, 'plot_step_attn', False)):
        _plot_single_step_attention(
            model,
            Y_all,
            time_start=int(args.time_start),
            step=int(args.attn_step),
            K_recent=int(args.recent_full_frames),
            out_dir=out_dir,
            residue_indices=ridxs,
        )

    # Multi-step grid (raw attention for several steps on one figure)
    if bool(getattr(args, 'plot_steps_grid', False)):
        steps_list: List[int] = []
        if isinstance(args.attn_steps, str) and args.attn_steps.strip():
            if args.attn_steps.strip().lower() == 'random':
                import random as _rand
                k = max(1, int(args.attn_random_k))
                pool = list(range(max(0, int(args.time_steps))))
                steps_list = sorted(_rand.sample(pool, k=min(k, len(pool))))
            else:
                try:
                    steps_list = sorted({int(s) for s in args.attn_steps.split(',') if str(s).strip() != ''})
                except Exception:
                    steps_list = []
        if not steps_list:
            # default: first min(5, time_steps) steps
            steps_list = list(range(0, min(5, int(args.time_steps))))
        _plot_steps_attention_grid(
            model,
            Y_all,
            time_start=int(args.time_start),
            steps=steps_list,
            K_recent=int(args.recent_full_frames),
            out_dir=out_dir,
            residue_indices=ridxs,
        )

    # 2c) Optional: per-residue softmax distribution at a given step
    if bool(getattr(args, 'plot_softmax_step', False)):
        try:
            _plot_softmax_step_distribution(
                model=model,
                Y_all=Y_all,
                time_start=int(args.time_start),
                step=int(args.softmax_step),
                out_dir=out_dir,
                residue_indices=ridxs,
                id2col=id2col,
                col2id=col2id,
                topk=int(args.softmax_topk),
                traj_name=traj_name,
            )
        except Exception:
            pass

    # Distributional metrics JSON (parity with full evaluator)
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        dist = compute_distributional_metrics(eval_out.gt, eval_out.pred, int(num_classes))
        dist_per = compute_per_residue_metrics(eval_out.gt, eval_out.pred, int(num_classes), top_k=10)
        dist['per_residue'] = dist_per
        import json as _json
        with open(out_dir / 'distribution_metrics.json', 'w') as f:
            _json.dump(dist, f, indent=2)
    except Exception:
        pass

    print(f"Wrote attention + rollout plots to: {out_dir}")


# -------------------------
# Extras: histograms and attention analysis
# -------------------------

import numpy as _np
import matplotlib.pyplot as _plt

def _safe_hist(vec: _np.ndarray, num_classes: int) -> _np.ndarray:
    h = _np.bincount(vec[vec >= 0], minlength=num_classes).astype(_np.float64)
    s = float(h.sum())
    return (h / s) if s > 0 else h

def _js_div(p: _np.ndarray, q: _np.ndarray, eps: float = 1e-12) -> float:
    p = _np.asarray(p, dtype=_np.float64); q = _np.asarray(q, dtype=_np.float64)
    p = _np.clip(p, eps, 1.0); q = _np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = _np.sum(p * _np.log(p / m))
    kl_qm = _np.sum(q * _np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)

def _plot_histograms_and_metrics(traj_name: str, gt: _np.ndarray, pr: _np.ndarray, out_dir: Path, topk: int = 30) -> None:
    C = int(max(int(gt.max(initial=0)), int(pr.max(initial=0))) + 1)
    h_gt = _safe_hist(gt.reshape(-1), C)
    h_pr = _safe_hist(pr.reshape(-1), C)
    js = float(_js_div(h_gt, h_pr))
    l1 = float(0.5 * _np.abs(h_gt - h_pr).sum())
    # top-K clusters by GT (legacy view)
    idx = _np.argsort(-h_gt)[:min(topk, C)]
    x = _np.arange(idx.size)
    _plt.figure(figsize=(12, 4.5))
    _plt.bar(x - 0.2, h_gt[idx], width=0.4, label='GT')
    _plt.bar(x + 0.2, h_pr[idx], width=0.4, label='Pred')
    _plt.xticks(x, [str(int(i)) for i in idx], rotation=90)
    _plt.ylabel('Occupancy')
    _plt.title(f'{traj_name} — Occupancy (top-{idx.size}) | JS={js:.4f}, L1={l1:.4f}')
    _plt.legend()
    _plt.tight_layout()
    (_ensure_dir(out_dir) or True)
    _plt.savefig(out_dir / f'{traj_name}_occupancy_top{idx.size}.png', dpi=300, bbox_inches='tight')
    _plt.close()

    # bottom-K clusters by GT (smallest non-zero GT occupancy)
    nonzero = _np.where(h_gt > 0)[0]
    if nonzero.size > 0:
        order_low = _np.argsort(h_gt[nonzero])[:min(topk, nonzero.size)]
        idx_low = nonzero[order_low]
        x2 = _np.arange(idx_low.size)
        _plt.figure(figsize=(12, 4.5))
        _plt.bar(x2 - 0.2, h_gt[idx_low], width=0.4, label='GT')
        _plt.bar(x2 + 0.2, h_pr[idx_low], width=0.4, label='Pred')
        _plt.xticks(x2, [str(int(i)) for i in idx_low], rotation=90)
        _plt.ylabel('Occupancy')
        _plt.title(f'{traj_name} — Occupancy (bottom-{idx_low.size} non-zero by GT)')
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_dir / f'{traj_name}_occupancy_bottom{idx_low.size}.png', dpi=300, bbox_inches='tight')
        _plt.close()

def _plot_mean_attention_over_frames(times_ns: _np.ndarray,
                                     attn_per_frame: _np.ndarray,
                                     out_path: Path) -> None:
    if attn_per_frame is None or attn_per_frame.size == 0:
        return
    mean_curve = attn_per_frame.mean(axis=0)
    _plt.figure(figsize=(10, 3.6))
    _plt.plot(times_ns, mean_curve, color='#1f77b4', linewidth=2)
    _plt.xlabel('Time (ns; history prior to rollout start)')
    _plt.ylabel('Mean attention')
    _plt.title('Mean temporal attention over frames')
    _plt.grid(alpha=0.25)
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=300, bbox_inches='tight')
    _plt.close()

    # Presence analysis + two-proportion z test for frequency differences
    n_gt = int((gt >= 0).sum())
    n_pr = int((pr >= 0).sum())
    c_gt = _np.bincount(gt.reshape(-1)[gt.reshape(-1) >= 0], minlength=C).astype(_np.int64)
    c_pr = _np.bincount(pr.reshape(-1)[pr.reshape(-1) >= 0], minlength=C).astype(_np.int64)
    present_gt = set(_np.where(c_gt > 0)[0].tolist())
    present_pr = set(_np.where(c_pr > 0)[0].tolist())
    only_gt = sorted(list(present_gt - present_pr))
    only_pr = sorted(list(present_pr - present_gt))
    both = _np.array(sorted(list(present_gt & present_pr)), dtype=_np.int64)

    # Two-proportion z test per common cluster
    eps = 1e-12
    p_gt = c_gt[both] / max(1, n_gt)
    p_pr = c_pr[both] / max(1, n_pr)
    p_pool = (c_gt[both] + c_pr[both]) / max(1, n_gt + n_pr)
    se = _np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / max(1, n_gt) + 1.0 / max(1, n_pr)) + eps)
    z = (p_pr - p_gt) / _np.maximum(se, eps)
    # two-sided p-values (normal approx)
    from math import erf, sqrt
    pvals = 2.0 * (1.0 - 0.5 * (1.0 + _np.array([erf(abs(zz) / sqrt(2.0)) for zz in z])))
    # Pick top-K by absolute difference
    diff = p_pr - p_gt
    order = _np.argsort(-_np.abs(diff))[:min(topk, diff.size)]
    sel = both[order]
    diff_sel = diff[order]
    se_sel = se[order]
    p_sel = pvals[order]
    lab = [str(int(i)) for i in sel]
    # Plot differences with 95% CI; color significant ones
    _plt.figure(figsize=(12, 4.5))
    x2 = _np.arange(len(sel))
    ci = 1.96 * se_sel
    colors = ['#d62728' if pv < 0.05 else '#1f77b4' for pv in p_sel]
    _plt.bar(x2, diff_sel, yerr=ci, color=colors, alpha=0.8)
    _plt.axhline(0.0, color='k', linewidth=1)
    _plt.xticks(x2, lab, rotation=90)
    _plt.ylabel('Pred - GT occupancy')
    _plt.title(f'{traj_name} — Freq diff (top-{len(sel)}) | #GT-only={len(only_gt)} #Pred-only={len(only_pr)}')
    _plt.tight_layout()
    _plt.savefig(out_dir / f'{traj_name}_freqdiff_top{len(sel)}.png', dpi=300, bbox_inches='tight')
    _plt.close()

    # Union-of-appeared clusters: side-by-side histograms (GT vs Pred)
    union_ids = _np.array(sorted(list(present_gt | present_pr)), dtype=_np.int64)
    if union_ids.size > 0:
        # Rank by max occupancy between GT/Pred to surface important clusters on either side
        score = _np.maximum(h_gt[union_ids], h_pr[union_ids])
        order_u = _np.argsort(-score)[:min(topk, union_ids.size)]
        sel_u = union_ids[order_u]
        x = _np.arange(sel_u.size)
        _plt.figure(figsize=(12, 4.5))
        _plt.bar(x - 0.2, h_gt[sel_u], width=0.4, color='#1f77b4', label='GT')
        _plt.bar(x + 0.2, h_pr[sel_u], width=0.4, color='#ff7f0e', label='Pred')
        _plt.xticks(x, [str(int(i)) for i in sel_u], rotation=90)
        _plt.ylabel('Occupancy')
        _plt.title(f'{traj_name} — Union occupancy (top-{sel_u.size}) | #GT-only={len(only_gt)} #Pred-only={len(only_pr)}')
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_dir / f'{traj_name}_occupancy_union_top{sel_u.size}.png', dpi=300, bbox_inches='tight')
        _plt.close()

    # Save a small JSON summary
    try:
        import json
        summary = {
            'n_gt_tokens': n_gt,
            'n_pred_tokens': n_pr,
            'num_clusters': C,
            'js': js,
            'l1': l1,
            'gt_only_count': len(only_gt),
            'pred_only_count': len(only_pr),
            'gt_only_ids': only_gt[:200],
            'pred_only_ids': only_pr[:200],
        }
        with open(out_dir / 'histogram_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

def _ensure_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

@torch.no_grad()
def _plot_softmax_step_distribution(model: torch.nn.Module,
                                    Y_all: _np.ndarray,
                                    time_start: int,
                                    step: int,
                                    out_dir: Path,
                                    residue_indices: List[int],
                                    id2col: dict | None,
                                    col2id: dict | None,
                                    topk: int = 20,
                                    traj_name: str = "") -> None:
    device = next(model.parameters()).device
    T_total, N = int(Y_all.shape[0]), int(Y_all.shape[1])
    s = int(max(0, step))
    t_hist = int(time_start) + s
    if t_hist <= 0 or t_hist >= T_total:
        return
    # Build history up to t_hist and get next-step logits (which align with GT at t_hist)
    hist_ids = torch.from_numpy(_np.where(Y_all[:t_hist] < 0, 0, Y_all[:t_hist])).long().unsqueeze(0).to(device)
    times = torch.arange(t_hist, dtype=torch.float32, device=device).view(1, -1) * 0.2
    hist_mask = torch.ones(1, t_hist, N, dtype=torch.bool, device=device)
    seq_lens = torch.tensor([t_hist], dtype=torch.long, device=device)
    out = model(
        input_cluster_ids=hist_ids,
        times=times,
        sequence_lengths=seq_lens,
        history_mask=hist_mask,
    )
    if 'cluster_logits' in out:
        logits = out['cluster_logits']
        if logits.dim() == 4:
            logits = logits[:, 0, :, :]
    else:
        ctx = out['context']
        logits = model.classification_head(ctx)
    probs = torch.softmax(logits, dim=-1)[0]  # (N, C)
    C = int(probs.size(-1))
    # Ground-truth at this next step
    gt_raw = Y_all[t_hist]  # (N,)
    # Map raw GT to column space if mapping available
    if id2col is not None and isinstance(id2col, dict) and len(id2col) > 0:
        gt_cols = _np.array([int(id2col.get(int(x), -1)) for x in gt_raw], dtype=_np.int64)
    else:
        gt_cols = gt_raw.astype(_np.int64)
    # Prepare residues
    ridxs = [int(r) for r in residue_indices if 0 <= int(r) < N]
    if not ridxs:
        ridxs = list(range(min(8, N)))
    k = int(max(1, min(int(topk), C)))
    # Plot grid of per-residue top-k bars; highlight GT class
    import math as _math
    ncols = 1 if len(ridxs) <= 3 else 2
    nrows = int(_math.ceil(len(ridxs) / ncols))
    fig, axes = _plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows), constrained_layout=True)
    axes = _np.array(axes).reshape(-1)
    for i, ridx in enumerate(ridxs):
        ax = axes[i]
        p = probs[ridx]  # (C,)
        topv, topi = torch.topk(p, k=k, dim=-1)
        topv_np = topv.detach().cpu().numpy()
        topi_np = topi.detach().cpu().numpy()
        # Labels in raw ID space if col2id is available
        if col2id is not None and isinstance(col2id, dict) and len(col2id) > 0:
            labels = [str(int(col2id.get(int(c), int(c)))) for c in topi_np]
            gt_label = str(int(col2id.get(int(gt_cols[ridx]), int(gt_cols[ridx]))))
        else:
            labels = [str(int(c)) for c in topi_np]
            gt_label = str(int(gt_cols[ridx]))
        x = _np.arange(k)
        colors = ['#1f77b4'] * k
        # Highlight GT if present in top-k
        try:
            if int(gt_cols[ridx]) >= 0:
                match_idx = int(_np.where(topi_np == int(gt_cols[ridx]))[0][0]) if (topi_np == int(gt_cols[ridx])).any() else None
            else:
                match_idx = None
        except Exception:
            match_idx = None
        if match_idx is not None:
            colors[match_idx] = '#d62728'
        ax.bar(x, topv_np, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0.0, max(1.0 / max(1, C), float(topv_np.max()) * 1.05))
        ax.set_ylabel('P(class)')
        ax.set_title(f'Residue {int(ridx)} — step {s} (GT={gt_label})')
        # Annotate GT prob even if not in top-k
        if int(gt_cols[ridx]) >= 0 and match_idx is None:
            gt_prob = float(p[int(gt_cols[ridx])].detach().cpu().item()) if int(gt_cols[ridx]) < C else 0.0
            ax.text(0.98, 0.92, f'GT p={gt_prob:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.grid(alpha=0.2)
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    title = traj_name if isinstance(traj_name, str) and len(traj_name) > 0 else 'trajectory'
    fig.suptitle(f'{title} — per-residue softmax at step {s}')
    fig.savefig(out_dir / f'{title}_softmax_step{s}.png', dpi=300, bbox_inches='tight')
    _plt.close(fig)

@torch.no_grad()
def _analyze_step_attention(model: torch.nn.Module,
                            Y_all: _np.ndarray,
                            time_start: int,
                            time_steps: int,
                            K_recent: int,
                            out_dir: Path,
                            residue_indices: List[int]) -> None:
    """Analyze per-step attention over K frames + latents during rollout.

    Produces a heatmap (steps × sources) and optional small residue-level plots.
    """
    device = next(model.parameters()).device
    N = int(Y_all.shape[1])
    # Initialize history buffer and latent summary Z
    hist_ids = torch.from_numpy(_np.where(Y_all[:time_start] < 0, 0, Y_all[:time_start])).long().unsqueeze(0).to(device)
    times = torch.arange(time_start, dtype=torch.float32, device=device).view(1, -1) * 0.2
    hist_mask = torch.ones(1, time_start, N, dtype=torch.bool, device=device)
    # Latent pool
    Z = None
    if getattr(model, 'latent_pool', None) is not None and hist_ids.size(1) > K_recent:
        base = hist_ids.size(1) - K_recent
        older_init = model.cluster_embedding(hist_ids[:, :base, :].clamp_min(0)).reshape(1, base * N, -1)
        Z = model.latent_pool(None, older_init)
    step_group = []  # list per step of attention fractions over [K frames] + [latents]
    # Toggle attention weight capture
    from scripts.autoregressive_eval import _set_store_attention
    _set_store_attention(model, True)
    for s in range(int(time_steps)):
        # Build K recent slice
        if hist_ids.size(1) < K_recent:
            K_eff = hist_ids.size(1)
        else:
            K_eff = K_recent
        step_ids = hist_ids[:, -K_eff:, :]
        step_times = times[:, -K_eff:]
        step_mask = torch.ones(1, K_eff, N, dtype=torch.bool, device=device)
        # Encode with optional latents
        if Z is not None and Z.numel() > 0:
            extra_kv = Z
            extra_kv_mask = torch.ones(1, extra_kv.size(1), dtype=torch.bool, device=device)
            extra_kv_time = step_times[:, -1]
        else:
            extra_kv = None; extra_kv_mask = None; extra_kv_time = None
        _, _, _ = model.backbone.encode(
            model.cluster_embedding(step_ids.clamp_min(0)),
            step_times,
            torch.full((1,), K_eff, dtype=torch.long, device=device),
            step_mask,
            t_scalar=None,
            change_mask=None,
            run_length=None,
            delta_t=None,
            state=None,
            extra_kv=extra_kv,
            extra_kv_mask=extra_kv_mask,
            extra_kv_time=extra_kv_time,
        )
        attn = getattr(model.backbone, 'last_attention_weights', None)
        if attn is None:
            break
        # attn: (H, N, S) or (B,H,N,S) averaged already; reduce to (N,S)
        if attn.dim() == 4:
            attn = attn.mean(dim=0)
        if attn.dim() == 3:
            attn = attn.mean(dim=0)
        S = attn.shape[1]
        # Token layout: [K_eff * N] + [Lz] (if any)
        Lz = extra_kv.size(1) if (extra_kv is not None and extra_kv.numel() > 0) else 0
        # Sum per frame across its N tokens
        frame_mass = []
        for f in range(K_eff):
            start = f * N
            frame_mass.append(attn[:, start:start + N].sum(dim=1))  # (N_query,)
        if Lz > 0:
            lat_mass = attn[:, K_eff * N : K_eff * N + Lz].sum(dim=1)
            frame_mass.append(lat_mass)
        per_src = torch.stack(frame_mass, dim=1)  # (N_query, K_eff + (1 if Lz>0 else 0))
        per_src = (per_src / per_src.sum(dim=1, keepdim=True).clamp_min(1e-12)).mean(dim=0)  # (K+Lz,)
        step_group.append(per_src.detach().cpu().numpy())
        # Predict next token (argmax for stability here) and update buffers/latents
        logits = model.classification_head(model.backbone.encode(
            model.cluster_embedding(step_ids.clamp_min(0)),
            step_times,
            torch.full((1,), K_eff, dtype=torch.long, device=device),
            step_mask,
        )[1]).argmax(dim=-1)  # (1,N)
        next_ids = logits.view(1, 1, N)
        hist_ids = torch.cat([hist_ids, next_ids], dim=1)
        times = torch.cat([times, step_times[:, -1:] + 0.2], dim=1)
        if getattr(model, 'latent_pool', None) is not None and K_eff > 0:
            leaving = step_ids[:, 0, :]
            Z = model.latent_pool(Z, model.cluster_embedding(leaving.clamp_min(0)))
    _set_store_attention(model, False)
    if not step_group:
        return
    A = _np.stack(step_group, axis=0)  # (S, K_or_less + lat)
    _plt.figure(figsize=(12, 5))
    _plt.imshow(A.T, aspect='auto', interpolation='nearest', cmap='magma')
    k_cols = min(K_recent, A.shape[1])
    has_lat = (A.shape[1] > k_cols)
    labels = [f'F-{k_cols - i}' for i in range(k_cols)] + (['latents'] if has_lat else [])
    _plt.yticks(_np.arange(A.shape[1]), labels)
    _plt.xlabel('Rollout step')
    _plt.ylabel('Source (frames →, latents)')
    _plt.title('Attention over K frames and latents (avg over residues)')
    _plt.colorbar(label='Attention mass')
    _plt.tight_layout()
    _plt.savefig(out_dir / 'step_attention_heatmap.png', dpi=300, bbox_inches='tight')
    _plt.close()


@torch.no_grad()
def _plot_single_step_attention(model: torch.nn.Module,
                                Y_all: _np.ndarray,
                                time_start: int,
                                step: int,
                                K_recent: int,
                                out_dir: Path,
                                residue_indices: List[int]) -> None:
    """Plot raw attention weights for a single rollout step.

    - y-axis: selected query residues
    - x-axis: keys laid out as [F-1: all residues] → ... → [F-K: all residues] → [latents]
    Also writes a per-frame mean bar plot and prints uniformity stats.
    """
    device = next(model.parameters()).device
    N = int(Y_all.shape[1])
    t_hist = max(0, int(time_start) + int(step))
    if t_hist <= 0:
        return
    hist_ids_full = torch.from_numpy(_np.where(Y_all[:t_hist] < 0, 0, Y_all[:t_hist])).long().unsqueeze(0).to(device)
    times_full = torch.arange(t_hist, dtype=torch.float32, device=device).view(1, -1) * 0.2

    # Build K recent slice
    K_eff = min(int(K_recent), hist_ids_full.size(1))
    step_ids = hist_ids_full[:, -K_eff:, :]
    step_times = times_full[:, -K_eff:]
    step_mask = torch.ones(1, K_eff, N, dtype=torch.bool, device=device)

    # Optional latent summary from older history
    Z = None
    if getattr(model, 'latent_pool', None) is not None and hist_ids_full.size(1) > K_eff:
        base = hist_ids_full.size(1) - K_eff
        older_init = model.cluster_embedding(hist_ids_full[:, :base, :].clamp_min(0)).reshape(1, base * N, -1)
        Z = model.latent_pool(None, older_init)

    # Toggle attention collection and run encode
    from scripts.autoregressive_eval import _set_store_attention
    _set_store_attention(model, True)
    _ = model.backbone.encode(
        model.cluster_embedding(step_ids.clamp_min(0)),
        step_times,
        torch.full((1,), K_eff, dtype=torch.long, device=device),
        step_mask,
        t_scalar=None,
        change_mask=None,
        run_length=None,
        delta_t=None,
        state=None,
        extra_kv=Z,
        extra_kv_mask=(torch.ones(1, Z.size(1), dtype=torch.bool, device=device) if (Z is not None and Z.numel() > 0) else None),
        extra_kv_time=(step_times[:, -1] if (Z is not None and Z.numel() > 0) else None),
    )
    attn = getattr(model.backbone, 'last_attention_weights', None)
    _set_store_attention(model, False)
    if attn is None:
        return
    if attn.dim() == 4:
        attn = attn.mean(dim=0)  # (H,N,S) -> average over batch
    if attn.dim() == 3:
        attn = attn.mean(dim=0)  # (N,S) averaged over heads
    # Select queries
    ridxs = [int(r) for r in residue_indices if 0 <= int(r) < N]
    if not ridxs:
        ridxs = list(range(min(8, N)))
    A_qk = attn[ridxs, :]  # (R, S)
    Lz = 0
    if Z is not None and Z.numel() > 0:
        Lz = Z.size(1)
    S = A_qk.shape[1]
    # Plot raw matrix
    _plt.figure(figsize=(12, max(3.0, 0.4 * len(ridxs) + 2)))
    _plt.imshow(A_qk, aspect='auto', interpolation='nearest', cmap='magma')
    # Frame boundaries
    for f in range(K_eff):
        x = f * N
        _plt.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.7)
    if Lz > 0:
        _plt.axvline(K_eff * N - 0.5, color='white', linewidth=1.0, alpha=0.9)
    _plt.yticks(_np.arange(len(ridxs)), [str(int(r)) for r in ridxs])
    _plt.xlabel('Keys (frames→residues, then latents)')
    _plt.ylabel('Query residue idx')
    _plt.title(f'Raw attention at step {step} (K={K_eff}, latents={Lz})')
    _plt.tight_layout()
    _plt.savefig(out_dir / f'step_attention_raw_step{step}.png', dpi=300, bbox_inches='tight')
    _plt.close()

    # Per-frame mean attention (single step) as a simple line graph
    masses = []
    for f in range(K_eff):
        start = f * N
        masses.append(A_qk[:, start:start + N].sum(axis=1))
    M = _np.stack(masses, axis=1)  # (R, K)
    M_mean = M.mean(axis=0)
    denom = float(M_mean.sum()) if float(M_mean.sum()) > 0 else 1.0
    M_mean = M_mean / denom
    _plt.figure(figsize=(10, 3.6))
    x = _np.arange(K_eff)
    labels = [f'F-{K_eff - i}' for i in range(K_eff)]
    _plt.plot(x, M_mean, marker='o', linewidth=2, color='#1f77b4')
    _plt.xticks(x, labels)
    _plt.ylabel('Mean attention (normalized)')
    _plt.xlabel('History frames (recent → left)')
    _plt.title(f'Per-frame mean at step {step}')
    _plt.grid(alpha=0.25)
    _plt.tight_layout()
    _plt.savefig(out_dir / f'step_attention_frame_mean_step{step}.png', dpi=300, bbox_inches='tight')
    _plt.close()


@torch.no_grad()
def _plot_steps_attention_grid(model: torch.nn.Module,
                               Y_all: _np.ndarray,
                               time_start: int,
                               steps: List[int],
                               K_recent: int,
                               out_dir: Path,
                               residue_indices: List[int]) -> None:
    device = next(model.parameters()).device
    N = int(Y_all.shape[1])
    if not steps:
        return
    # Precompute full history tensors up to max step time
    t_hist_max = max(0, int(time_start) + max(steps))
    hist_ids_full = torch.from_numpy(_np.where(Y_all[:t_hist_max] < 0, 0, Y_all[:t_hist_max])).long().unsqueeze(0).to(device)
    times_full = torch.arange(t_hist_max, dtype=torch.float32, device=device).view(1, -1) * 0.2
    ridxs = [int(r) for r in residue_indices if 0 <= int(r) < N]
    if not ridxs:
        ridxs = list(range(min(8, N)))

    import math as _math
    cols = min(5, max(1, len(steps)))
    rows = int(_math.ceil(len(steps) / cols))
    _plt.figure(figsize=(6 * cols, 3.8 * rows))

    for idx, s in enumerate(steps):
        t_hist = max(0, int(time_start) + int(s))
        if t_hist <= 0:
            continue
        K_eff = min(int(K_recent), max(1, t_hist))
        step_ids = hist_ids_full[:, t_hist - K_eff: t_hist, :]
        step_times = times_full[:, t_hist - K_eff: t_hist]
        step_mask = torch.ones(1, K_eff, N, dtype=torch.bool, device=device)
        from scripts.autoregressive_eval import _set_store_attention
        _set_store_attention(model, True)
        _ = model.backbone.encode(
            model.cluster_embedding(step_ids.clamp_min(0)),
            step_times,
            torch.full((1,), K_eff, dtype=torch.long, device=device),
            step_mask,
        )
        attn = getattr(model.backbone, 'last_attention_weights', None)
        _set_store_attention(model, False)
        if attn is None:
            continue
        if attn.dim() == 4:
            attn = attn.mean(dim=0)
        if attn.dim() == 3:
            attn = attn.mean(dim=0)
        A_qk = attn[ridxs, :]
        ax = _plt.subplot(rows, cols, idx + 1)
        im = ax.imshow(A_qk, aspect='auto', interpolation='nearest', cmap='magma')
        for f in range(K_eff):
            x = f * N
            ax.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.7)
        ax.set_title(f'step {int(s)} (K={K_eff})')
        ax.set_xlabel('keys')
        ax.set_ylabel('query residues')
    _plt.tight_layout()
    _plt.savefig(out_dir / 'step_attention_grid.png', dpi=300, bbox_inches='tight')
    _plt.close()

    # Per-frame mean (average over query residues), normalized
    masses = []
    for f in range(K_eff):
        start = f * N
        masses.append(A_qk[:, start:start + N].sum(axis=1))  # (R,)
    if Lz > 0:
        masses.append(A_qk[:, K_eff * N : K_eff * N + Lz].sum(axis=1))
    M = _np.stack(masses, axis=1)  # (R, K(+lat))
    M_mean = M.mean(axis=0)
    M_mean = M_mean / max(1e-12, M_mean.sum())
    # Uniformity stats across frames (ignore latents for H/logK)
    import math as _math
    if K_eff > 0:
        p_frames = M_mean[:K_eff] / max(1e-12, M_mean[:K_eff].sum())
        H = -_np.sum(_np.clip(p_frames, 1e-12, 1.0) * _np.log(_np.clip(p_frames, 1e-12, 1.0)))
        H_norm = float(H / _math.log(K_eff))
    else:
        H_norm = float('nan')
    _plt.figure(figsize=(10, 3.6))
    x = _np.arange(K_eff + (1 if Lz > 0 else 0))
    labels = [f'F-{K_eff - i}' for i in range(K_eff)] + (['latents'] if Lz > 0 else [])
    _plt.bar(x, M_mean, color='#1f77b4')
    _plt.xticks(x, labels)
    _plt.ylabel('Mean attention (avg over queries)')
    _plt.title(f'Per-frame mean at step {step} | H_norm={H_norm:.3f}')
    _plt.tight_layout()
    _plt.savefig(out_dir / f'step_attention_frame_mean_step{step}.png', dpi=300, bbox_inches='tight')
    _plt.close()

if __name__ == '__main__':
    main()
