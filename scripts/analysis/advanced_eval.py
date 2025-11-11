#!/usr/bin/env python3
"""
Advanced evaluation analyses for EMPROT runs.

Features (all optional and can be run independently):
1) Head/Med/Tail Recall@{1,5,10} from saved evaluator arrays
2) Coverage@N (unique classes predicted) vs step across multiple checkpoints
3) Transition confusion heatmap (top-50 recent→predicted) [requires model+data]
4) No-change vs change ROC/PRC [requires model+data]
5) Attention-lag profiles per head [requires model+data]
6) Loss component shares over epochs (stacked area) from a CSV/JSON log

This script intentionally does not re-run full evaluation; it either reads the
cluster_results saved by scripts/evaluate_single_model.py, or runs small targeted
forward passes when model+data are provided.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# --- Temporal binning helper metrics ---
def _metric_global_freq(targs_ids: np.ndarray, valid_mask: np.ndarray) -> Counter:
    return Counter(targs_ids[valid_mask].tolist())


# Note: Removed global traj_support metric; use 'traj_local' for per-trajectory binning.


def _metric_median_gap(targs_ids: np.ndarray, traj_ids: np.ndarray, time_idx: np.ndarray, valid_mask: np.ndarray):
    # classes → list of gaps across trajs; metric = median gap (bigger = rarer)
    from collections import defaultdict
    per_tc = defaultdict(list)  # (traj, c) -> list of times
    v = np.where(valid_mask & (targs_ids >= 0))[0]
    for i in v:
        per_tc[(int(traj_ids[i]), int(targs_ids[i]))].append(int(time_idx[i]))
    # compute gaps
    cls2gaps = defaultdict(list)
    for (traj, c), ts in per_tc.items():
        ts = np.sort(np.array(ts))
        if ts.size >= 2:
            gaps = np.diff(ts)
            cls2gaps[c].extend(gaps.tolist())
    # median gap; if no gaps (singleton), treat as very rare (use +inf sentinel)
    metric = {}
    classes_present = set([int(t) for t in targs_ids[v].tolist() if int(t) >= 0])
    for c in classes_present:
        gaps = cls2gaps.get(c, [])
        metric[c] = (np.median(gaps) if len(gaps) > 0 else np.inf)
    return metric


def _load_cls_results(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load classification results: preds (cols), targets (cols), optional top-k scores."""
    preds = np.load(results_dir / 'predicted_cluster_cols.npy')
    targs = np.load(results_dir / 'target_cluster_cols.npy')
    scores_path = results_dir / 'prediction_scores.npz'
    scores = None
    if scores_path.exists():
        scores = np.load(scores_path)['scores']  # shape (K, eval_classes)
    return preds, targs, scores


def _load_reg_results(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load regression results (mapped): preds (ids), targets (ids), optional scores."""
    preds = np.load(results_dir / 'predicted_cluster_ids.npy')
    # prefer mapped if present
    tmap = results_dir / 'target_cluster_ids.npy'
    traw = results_dir / 'target_cluster_ids_raw.npy'
    if tmap.exists():
        targs = np.load(tmap)
    else:
        targs = np.load(traw)
    scores_path = results_dir / 'prediction_scores.npz'
    scores = None
    if scores_path.exists():
        scores = np.load(scores_path)['scores']
    return preds, targs, scores


# Optional: new compact top-k file loader (works for cls+reg)
def _load_topk(results_dir: Path):
    ids = np.load(results_dir / 'topk_ids.npy')  # [N,k] or [B,T,k]
    sc = None
    p = results_dir / 'topk_scores.npy'
    if p.exists():
        sc = np.load(p)
    # Flatten time if 3D
    if ids.ndim == 3:
        ids = ids.reshape(-1, ids.shape[-1])
        if sc is not None:
            sc = sc.reshape(-1, sc.shape[-1])
    return ids, sc


def topk_hits_from_topk_arrays(topk_ids: np.ndarray, targets: np.ndarray, ks=(1, 5, 10)):
    # targets: [N]; topk_ids: [N, K]
    hits = {}
    valid = targets >= 0
    K = topk_ids.shape[1]
    for k in ks:
        kk = min(k, K)
        hits[k] = (topk_ids[:, :kk] == targets[:, None]).any(axis=1) & valid
    return hits, valid


def _topk_hits_from_scores(scores: np.ndarray, eval_class_ids: np.ndarray, targets: np.ndarray, ks=(1,5,10)):
    """Compute top-k hits given class scores and targets in the same space.
    scores: (N, Ck), eval_class_ids: (Ck,), targets: (N,)
    Returns dict{k: hits (bool array)}.
    """
    # map target to local column when eval_class_ids is used
    id2col = {cid: i for i, cid in enumerate(eval_class_ids)}
    mask = np.array([t in id2col for t in targets])
    mapped = np.full_like(targets, -1)
    mapped[mask] = np.array([id2col[t] for t in targets[mask]])
    out = {}
    # argsort descending and take top-k columns
    order = np.argpartition(-scores, kth=np.minimum(scores.shape[1]-1, 10), axis=1)
    for k in ks:
        k = min(k, scores.shape[1])
        topk = order[:, :k]
        hits = np.any(topk == mapped[:, None], axis=1) & (mapped >= 0)
        out[k] = hits
    return out, mask


def _compute_train_class_counts(data_dir: str, metadata_path: str, seed: int = 42,
                                batch_size: int = 32, sequence_length: int = 5,
                                stride: int = 10, num_workers: int = 2,
                                max_batches: int = None) -> Counter:
    """Count class frequencies on the TRAIN split (last-frame targets)."""
    from emprot.data.dataset import create_dataloaders
    train_loader, _, _ = create_dataloaders(
        data_dir=data_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        max_sequence_length=sequence_length,
        min_sequence_length=2,
        stride=stride,
        train_split=0.8,
        val_split=0.1,
        num_workers=num_workers,
        seed=seed,
    )
    counts = Counter()
    for i, batch in enumerate(train_loader):
        tgt = None
        tdict = batch.get('targets', {}) if 'targets' in batch else {}
        if isinstance(tdict, dict) and 'target_cluster_ids' in tdict:
            tgt = tdict['target_cluster_ids']
        elif 'target_cluster_ids' in batch:
            tgt = batch['target_cluster_ids']
        if tgt is None:
            continue
        arr = tgt.numpy() if hasattr(tgt, 'numpy') else np.asarray(tgt)
        valid = arr >= 0
        if valid.any():
            counts.update(arr[valid].ravel().tolist())
        if (i + 1) % 100 == 0:
            print(f"   Counted {i+1} train batches...")
        if max_batches is not None and (i + 1) >= max_batches:
            print(f"   Stopping early at max_batches={max_batches}")
            break
    return counts


def head_med_tail_recall(results_dirs: List[str], labels: List[str], output: str,
                         train_counts: Counter = None, bins_source: str = "eval",
                         bins_metric: str = "global_freq", quantiles: Tuple[float, float] = (0.1, 0.3)):
    """Compute Recall@{1,5,10} for head/med/tail bins per run.
    Uses frequency deciles to define bins.
    bins_source: 'train' → use train_counts; 'eval' → use current targs.
    """
    ks = (1, 5, 10)
    recalls = {
        lab: (
            {
                b: ({k: 0.0 for k in ks} | {"support": 0, "hits_at_k": {k: 0 for k in ks}})
                for b in ["head", "med", "tail"]
            } | {"bucket_coverage": {}}
        )
        for lab in labels
    }

    for lab, rd in zip(labels, results_dirs):
        rd = Path(rd)
        # Try classification first; fallback to regression
        preds = targs = scores = None
        # eval_class_ids.npy (columns) and eval_class_raw_ids.npy (global IDs) saved by evaluator
        eval_ids_path = rd / 'eval_class_ids.npy'
        # Backward-compat: also check parent dir
        if not eval_ids_path.exists():
            compat_path = rd.parent / 'eval_class_ids.npy'
            if compat_path.exists():
                eval_ids_path = compat_path
        eval_raw_path = rd / 'eval_class_raw_ids.npy'

        # Prefer new compact top-k if present
        use_topk = (rd / 'topk_ids.npy').exists()
        if (rd / 'predicted_cluster_cols.npy').exists():
            preds, targs, scores = _load_cls_results(rd)
        else:
            preds, targs, scores = _load_reg_results(rd)
        # Prefer global raw IDs if available; else fall back to column indices
        eval_ids_are_global = False
        if eval_raw_path.exists():
            eval_ids = np.load(eval_raw_path)
            eval_ids_are_global = True
        elif eval_ids_path.exists():
            eval_ids = np.load(eval_ids_path)
        else:
            eval_ids = None

        # Normalize to global ID space for binning/coverage
        # After loading preds, targs, scores, eval_ids
        is_cls = (rd / 'predicted_cluster_cols.npy').exists()
        if is_cls:
            # Prefer raw global ID files if present (avoid mismatched eval_ids length)
            targs_raw = rd / 'target_cluster_ids_raw.npy'
            preds_raw = rd / 'predicted_cluster_ids_raw.npy'
            if targs_raw.exists():
                targs_ids = np.load(targs_raw)
            elif eval_ids is not None and eval_ids_are_global:
                # Guard against bad indexing
                if targs.max(initial=-1) >= len(eval_ids):
                    raise RuntimeError(f"targs index {int(targs.max())} >= eval_ids length {len(eval_ids)}")
                targs_ids = np.where(targs >= 0, eval_ids[targs], -1)
            else:
                targs_ids = targs
            if preds_raw.exists():
                preds_ids = np.load(preds_raw)
            elif eval_ids is not None and eval_ids_are_global:
                if preds.max(initial=-1) >= len(eval_ids):
                    raise RuntimeError(f"preds index {int(preds.max())} >= eval_ids length {len(eval_ids)}")
                preds_ids = np.where(preds >= 0, eval_ids[preds], -1)
            else:
                preds_ids = preds
        else:
            # Regression paths are already in global ID space
            targs_ids = targs
            preds_ids = preds

        # Try to load temporal alignment (optional)
        traj_path = rd / 'eval_traj_ids.npy'
        time_path = rd / 'eval_time_idx.npy'
        traj_ids = np.load(traj_path) if traj_path.exists() else None
        time_idx = np.load(time_path) if time_path.exists() else None
        if traj_ids is not None and traj_ids.ndim == 2:
            traj_ids = traj_ids.reshape(-1)
        if time_idx is not None and time_idx.ndim == 2:
            time_idx = time_idx.reshape(-1)

        # Define bins by selected metric
        valid_targs_mask = (targs_ids >= 0)
        metric_name = bins_metric
        use_traj_local = False
        if bins_metric == 'traj_local':
            # Per-trajectory head/med/tail bins computed within each trajectory separately
            if traj_ids is None:
                raise RuntimeError("traj_local metric requires eval_traj_ids.npy aligned with targets.")
            use_traj_local = True
            q1, q2 = quantiles
            assert 0.0 < q1 < q2 < 1.0, f"Bad hmt_quantiles: {quantiles}"
            traj_bins = {}
            head_union = set()
            med_union = set()
            tail_union = set()
            # Build per-trajectory class frequency and split into H/M/T
            unique_traj = np.unique(traj_ids)
            for tid in unique_traj:
                idx = (traj_ids == tid) & valid_targs_mask
                if not np.any(idx):
                    continue
                cls_counts = Counter(targs_ids[idx].tolist())
                if not cls_counts:
                    continue
                # Sort classes by descending frequency and split by cumulative coverage thresholds (q1, q2)
                cls_arr = np.array(list(cls_counts.keys()))
                freq_arr = np.array([cls_counts[c] for c in cls_arr], dtype=float)
                order_loc = np.argsort(-freq_arr)
                cls_sorted = cls_arr[order_loc]
                freq_sorted = freq_arr[order_loc]
                total = float(freq_sorted.sum())
                if total <= 0:
                    continue
                cum = np.cumsum(freq_sorted) / total
                # head: up to q1 coverage; med: (q1, q2]; tail: remainder
                head_mask = cum <= q1
                med_mask = (cum > q1) & (cum <= q2)
                tail_mask = cum > q2
                # Ensure non-empty head
                if not head_mask.any():
                    head_mask[0] = True
                    med_mask = (cum > q1) & (cum <= q2)
                    tail_mask = cum > q2
                head = set(cls_sorted[head_mask].tolist())
                med = set(cls_sorted[med_mask].tolist())
                tail = set(cls_sorted[tail_mask].tolist())
                traj_bins[int(tid)] = {'head': head, 'med': med, 'tail': tail}
                head_union.update(head)
                med_union.update(med)
                tail_union.update(tail)
            # Use unions for coverage calculations later
            classes = np.array(list(head_union | med_union | tail_union))
            metric_vals = np.ones_like(classes, dtype=float)
            order = np.arange(len(classes))
            head_set, med_set, tail_set = head_union, med_union, tail_union
            metric_name = 'traj_local'
        else:
            print(f"[WARN] bins_metric='{bins_metric}' is deprecated; using traj_local coverage bins instead.")
            if traj_ids is None:
                raise RuntimeError("traj_local metric requires eval_traj_ids.npy aligned with targets.")
            use_traj_local = True
            q1, q2 = quantiles
            assert 0.0 < q1 < q2 < 1.0, f"Bad hmt_quantiles: {quantiles}"
            traj_bins = {}
            head_union = set()
            med_union = set()
            tail_union = set()
            unique_traj = np.unique(traj_ids)
            for tid in unique_traj:
                idx = (traj_ids == tid) & valid_targs_mask
                if not np.any(idx):
                    continue
                cls_counts = Counter(targs_ids[idx].tolist())
                if not cls_counts:
                    continue
                cls_arr = np.array(list(cls_counts.keys()))
                freq_arr = np.array([cls_counts[c] for c in cls_arr], dtype=float)
                order_loc = np.argsort(-freq_arr)
                cls_sorted = cls_arr[order_loc]
                freq_sorted = freq_arr[order_loc]
                total = float(freq_sorted.sum())
                if total <= 0:
                    continue
                cum = np.cumsum(freq_sorted) / total
                head_mask = cum <= q1
                med_mask = (cum > q1) & (cum <= q2)
                tail_mask = cum > q2
                if not head_mask.any():
                    head_mask[0] = True
                head = set(cls_sorted[head_mask].tolist())
                med = set(cls_sorted[med_mask].tolist())
                tail = set(cls_sorted[tail_mask].tolist())
                traj_bins[int(tid)] = {'head': head, 'med': med, 'tail': tail}
                head_union.update(head)
                med_union.update(med)
                tail_union.update(tail)
            classes = np.array(list(head_union | med_union | tail_union))
            metric_vals = np.ones_like(classes, dtype=float)
            order = np.arange(len(classes))
            head_set, med_set, tail_set = head_union, med_union, tail_union
            metric_name = 'traj_local'
        # No other metrics supported; traj_local is enforced above

        classes = classes[order]
        metric_vals = metric_vals[order]
        # head_set/med_set/tail_set already computed for traj_local coverage; keep as-is
        # Bucket sizes: head={len(head_set)} med={len(med_set)} tail={len(tail_set)}

        # Intersect buckets with evaluated classes (scores space) if available
        if eval_ids is not None:
            eval_class_set = set(eval_ids.tolist())
            head_eval = head_set & eval_class_set
            med_eval  = med_set & eval_class_set
            tail_eval = tail_set & eval_class_set
        else:
            head_eval, med_eval, tail_eval = head_set, med_set, tail_set

        # Compute hits
        if use_topk:
            # Prefer raw-ID top-k if present (global ID space)
            tk_raw_path = rd / 'topk_ids_raw.npy'
            if tk_raw_path.exists():
                tk_ids = np.load(tk_raw_path)
                hits_k, valid_mask = topk_hits_from_topk_arrays(tk_ids, targs_ids)
                # Normalize shape to [N,K]
                if tk_ids.ndim == 3:
                    tk_ids = tk_ids.reshape(-1, tk_ids.shape[-1])
                topk_ids_for_topk = tk_ids
                targs_for_topk = targs_ids
            else:
                # Fall back to column-space top-k; compare in column space
                tk_ids, tk_scores = _load_topk(rd)
                hits_k, valid_mask = topk_hits_from_topk_arrays(tk_ids, targs)
                # Map to global space if eval_ids available; else keep in column space
                if eval_ids is not None and eval_ids_are_global:
                    try:
                        topk_ids_for_topk = eval_ids[tk_ids]
                        targs_for_topk = targs_ids
                    except Exception:
                        topk_ids_for_topk = tk_ids
                        targs_for_topk = targs
                else:
                    topk_ids_for_topk = tk_ids
                    targs_for_topk = targs
        elif scores is not None and eval_ids is not None:
            hits_k, valid_mask = _topk_hits_from_scores(scores, eval_ids, targs)
            # Build explicit top-k id lists from scores for further metrics
            try:
                # Compute top-10 columns, then map to global if available
                kmax = min(10, scores.shape[1])
                order = np.argpartition(-scores, kth=kmax-1, axis=1)[:, :kmax]
                if is_cls:
                    if eval_ids is not None and eval_ids_are_global:
                        topk_ids_for_topk = eval_ids[order]
                        targs_for_topk = targs_ids
                    else:
                        topk_ids_for_topk = order
                        targs_for_topk = targs
                else:
                    # Regression scores are already over eval_ids columns
                    if eval_ids is not None:
                        topk_ids_for_topk = eval_ids[order]
                        targs_for_topk = targs_ids
                    else:
                        topk_ids_for_topk = order
                        targs_for_topk = targs
            except Exception:
                topk_ids_for_topk = None
                targs_for_topk = None
        else:
            # fall back: top-1 only from predictions
            hits_k = {1: (preds == targs)}
            valid_mask = np.ones_like(targs, dtype=bool)
            # Build a top-1 list in the same space as targs_ids mapping used above
            if is_cls and (eval_ids is not None and eval_ids_are_global):
                # Use global space
                preds_top1 = preds_ids.reshape(-1, 1)
                targs_for_topk = targs_ids
                topk_ids_for_topk = preds_top1
            else:
                preds_top1 = preds.reshape(-1, 1)
                targs_for_topk = targs
                topk_ids_for_topk = preds_top1

        # Denominators per bucket before applying valid_mask
        for bname, bset in [('head', head_eval), ('med', med_eval), ('tail', tail_eval)]:
            _ = np.array([t in bset for t in targs_ids]).sum()

        # Aggregate by bins with supports and hits
        if use_traj_local:
            # Build per-sample masks according to its trajectory-local bins
            bmask_head = np.zeros_like(valid_mask, dtype=bool)
            bmask_med  = np.zeros_like(valid_mask, dtype=bool)
            bmask_tail = np.zeros_like(valid_mask, dtype=bool)
            # Iterate per trajectory for vectorized set membership
            for tid, sets in traj_bins.items():
                idx = (traj_ids == tid)
                if not np.any(idx):
                    continue
                th = np.isin(targs_ids[idx], list(sets['head']))
                tm = np.isin(targs_ids[idx], list(sets['med']))
                tt = np.isin(targs_ids[idx], list(sets['tail']))
                bmask_head[idx] = th
                bmask_med[idx]  = tm
                bmask_tail[idx] = tt
            for bname, bmask in [('head', bmask_head), ('med', bmask_med), ('tail', bmask_tail)]:
                bmask_final = bmask & valid_mask
                denom = int(bmask_final.sum())
                recalls[lab][bname]['support'] = denom
                # Top-1 metrics within bucket
                metrics_mask = bmask_final & (targs_ids >= 0) & (preds_ids >= 0)
                if int(metrics_mask.sum()) > 0:
                    vp = preds_ids[metrics_mask]
                    vt = targs_ids[metrics_mask]
                    try:
                        acc = float((vp == vt).mean())
                        prec_m = float(precision_score(vt, vp, average='macro', zero_division=0))
                        rec_m = float(recall_score(vt, vp, average='macro', zero_division=0))
                        f1_m = float(f1_score(vt, vp, average='macro', zero_division=0))
                    except Exception:
                        acc = prec_m = rec_m = f1_m = 0.0
                    recalls[lab][bname]['accuracy'] = acc
                    recalls[lab][bname]['precision_macro'] = prec_m
                    recalls[lab][bname]['recall_macro'] = rec_m
                    recalls[lab][bname]['f1_macro'] = f1_m
                else:
                    recalls[lab][bname]['accuracy'] = 0.0
                    recalls[lab][bname]['precision_macro'] = 0.0
                    recalls[lab][bname]['recall_macro'] = 0.0
                    recalls[lab][bname]['f1_macro'] = 0.0
                for k in ks:
                    if k in hits_k:
                        num = int(hits_k[k][bmask_final].sum()) if denom > 0 else 0
                        recalls[lab][bname]['hits_at_k'][k] = num
                        recalls[lab][bname][k] = float(num) / max(1, denom)
        else:
            for bname, bset in [('head',head_eval), ('med',med_eval), ('tail',tail_eval)]:
                bmask = np.array([t in bset for t in targs_ids]) & valid_mask
                denom = int(bmask.sum())
                recalls[lab][bname]['support'] = denom
                # Top-1 metrics within bucket
                metrics_mask = bmask & (targs_ids >= 0) & (preds_ids >= 0)
                if int(metrics_mask.sum()) > 0:
                    vp = preds_ids[metrics_mask]
                    vt = targs_ids[metrics_mask]
                    try:
                        acc = float((vp == vt).mean())
                        prec_m = float(precision_score(vt, vp, average='macro', zero_division=0))
                        rec_m = float(recall_score(vt, vp, average='macro', zero_division=0))
                        f1_m = float(f1_score(vt, vp, average='macro', zero_division=0))
                    except Exception:
                        acc = prec_m = rec_m = f1_m = 0.0
                    recalls[lab][bname]['accuracy'] = acc
                    recalls[lab][bname]['precision_macro'] = prec_m
                    recalls[lab][bname]['recall_macro'] = rec_m
                    recalls[lab][bname]['f1_macro'] = f1_m
                else:
                    recalls[lab][bname]['accuracy'] = 0.0
                    recalls[lab][bname]['precision_macro'] = 0.0
                    recalls[lab][bname]['recall_macro'] = 0.0
                    recalls[lab][bname]['f1_macro'] = 0.0
                for k in ks:
                    if k in hits_k:
                        num = int(hits_k[k][bmask].sum()) if denom>0 else 0
                        recalls[lab][bname]['hits_at_k'][k] = num
                        recalls[lab][bname][k] = float(num) / max(1, denom)

        # Also compute bucket coverage of predicted classes (unique coverage)
        pred_class_set = set(preds_ids.tolist())
        cov = {}
        for bname, full_bset in [('head', head_set), ('med', med_set), ('tail', tail_set)]:
            cov[bname] = float(len(pred_class_set & full_bset)) / max(1, len(full_bset))
        # Attach to JSON under special key
        recalls[lab]['bucket_coverage'] = cov

        # --- Compute Top-K macro metrics per bucket (precision/recall/F1, accuracy) ---
        if topk_ids_for_topk is not None and targs_for_topk is not None:
            # Ensure shapes
            if topk_ids_for_topk.ndim == 1:
                topk_ids_for_topk = topk_ids_for_topk.reshape(-1, 1)
            # Valid samples for this computation
            valid_topk_mask = (targs_for_topk >= 0)
            # Prepare bucket class sets in the same id space as targs_for_topk/topk_ids_for_topk
            # If targs_for_topk is global, head_set/med_set/tail_set are already global
            # If targs_for_topk is column-space, then earlier targs_ids == targs and bucket sets were created in that space
            bucket_sets_for_space = {
                'head': head_set,
                'med': med_set,
                'tail': tail_set,
            }

            Kmax_avail = topk_ids_for_topk.shape[1]
            for bname in ['head', 'med', 'tail']:
                recalls[lab][bname].setdefault('topk', {})
                class_set = bucket_sets_for_space[bname]
                # Restrict to classes that actually appear anywhere to avoid empty denominators explosion
                class_list = list(class_set)
                if len(class_list) == 0:
                    continue
                targs_vec = targs_for_topk[valid_topk_mask]
                topk_mat = topk_ids_for_topk[valid_topk_mask]
                # Precompute per-k membership
                for k in (1, 5, 10):
                    if k > Kmax_avail:
                        continue
                    kslice = topk_mat[:, :k]
                    # Build per-class counts
                    tp_list = []
                    predpos_list = []
                    supp_list = []
                    for c in class_list:
                        # Boolean arrays
                        supp_mask = (targs_vec == c)
                        predpos_mask = np.any(kslice == c, axis=1)
                        tp = int((supp_mask & predpos_mask).sum())
                        predpos = int(predpos_mask.sum())
                        supp = int(supp_mask.sum())
                        tp_list.append(tp)
                        predpos_list.append(predpos)
                        supp_list.append(supp)
                    # Compute per-class precision/recall/F1
                    precisions = []
                    recalls_cls = []
                    f1s = []
                    for tp, pp, sp in zip(tp_list, predpos_list, supp_list):
                        prec = (tp / pp) if pp > 0 else 0.0
                        rec = (tp / sp) if sp > 0 else 0.0
                        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                        precisions.append(prec)
                        recalls_cls.append(rec)
                        f1s.append(f1)
                    # Macro-averages over classes with any support or predictions
                    precision_macro_k = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
                    recall_macro_k = float(np.mean(recalls_cls)) if len(recalls_cls) > 0 else 0.0
                    f1_macro_k = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
                    # Accuracy@k within bucket equals mean hit rate among samples whose true class is in bucket
                    # Build bucket mask using targs_vec and class_set
                    bmask_vec = np.isin(targs_vec, class_list)
                    acc_k = float((np.any(kslice[bmask_vec] == targs_vec[bmask_vec][:, None], axis=1).mean())) if bmask_vec.any() else 0.0
                    recalls[lab][bname]['topk'][f'@{k}'] = {
                        'accuracy': acc_k,
                        'precision_macro': precision_macro_k,
                        'recall_macro': recall_macro_k,
                        'f1_macro': f1_macro_k,
                        'classes_considered': len(class_list)
                    }

    # Plot bar chart
    xbins = ['head','med','tail']
    for k in ks:
        plt.figure(figsize=(6,4))
        width = 0.8/len(labels)
        xs = np.arange(len(xbins))
        for i, lab in enumerate(labels):
            vals = [recalls[lab][b][k] for b in xbins]
            plt.bar(xs + i*width, vals, width=width, label=lab)
        plt.xticks(xs + width*(len(labels)-1)/2, xbins)
        plt.ylim(0,1)
        plt.ylabel(f'Recall@{k}')
        plt.title(f'H/M/T Recall (metric={bins_metric}, q={quantiles})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output) / f'head_med_tail_recall_at_{k}.png', dpi=200)
        plt.close()

    # Save table JSON
    with open(Path(output)/'head_med_tail_recall.json','w') as f:
        json.dump(recalls, f, indent=2)

    # Additional H/M/T bar charts for top-1 metrics
    xbins = ['head','med','tail']
    for metric_key, display_name, file_tag in [
        ('accuracy', 'Accuracy', 'accuracy'),
        ('precision_macro', 'Precision (Macro)', 'precision_macro'),
        ('f1_macro', 'F1 (Macro)', 'f1_macro'),
    ]:
        plt.figure(figsize=(6,4))
        width = 0.8/len(labels)
        xs = np.arange(len(xbins))
        for i, lab in enumerate(labels):
            vals = []
            for b in xbins:
                supp = recalls[lab][b].get('support', 0)
                vals.append(recalls[lab][b].get(metric_key, 0.0) if supp > 0 else 0.0)
            plt.bar(xs + i*width, vals, width=width, label=lab)
        plt.xticks(xs + width*(len(labels)-1)/2, xbins)
        plt.ylim(0,1)
        plt.ylabel(display_name)
        plt.title(f'H/M/T {display_name} (metric={bins_metric}, q={quantiles})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output) / f'head_med_tail_{file_tag}.png', dpi=200)
        plt.close()

    # Per-K H/M/T charts for Accuracy, Precision (Macro), and F1 (Macro)
    xbins = ['head','med','tail']
    for k in (1, 5, 10):
        # Accuracy@k
        plt.figure(figsize=(6,4))
        width = 0.8/len(labels)
        xs = np.arange(len(xbins))
        for i, lab in enumerate(labels):
            vals = []
            for b in xbins:
                # Prefer topk accuracy if available; else fall back to existing recall@k
                topk_dict = recalls[lab][b].get('topk', {})
                if f'@{k}' in topk_dict:
                    vals.append(topk_dict[f'@{k}'].get('accuracy', 0.0))
                else:
                    vals.append(recalls[lab][b].get(k, 0.0))
            plt.bar(xs + i*width, vals, width=width, label=lab)
        plt.xticks(xs + width*(len(labels)-1)/2, xbins)
        plt.ylim(0,1)
        plt.ylabel(f'Accuracy@{k}')
        plt.title(f'H/M/T Accuracy@{k} (metric={bins_metric}, q={quantiles})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output) / f'head_med_tail_accuracy_at_{k}.png', dpi=200)
        plt.close()

        # Precision@k (Macro)
        plt.figure(figsize=(6,4))
        for i, lab in enumerate(labels):
            vals = []
            for b in xbins:
                topk_dict = recalls[lab][b].get('topk', {})
                vals.append(topk_dict.get(f'@{k}', {}).get('precision_macro', 0.0))
            plt.bar(xs + i*width, vals, width=width, label=lab)
        plt.xticks(xs + width*(len(labels)-1)/2, xbins)
        plt.ylim(0,1)
        plt.ylabel(f'Precision@{k} (Macro)')
        plt.title(f'H/M/T Precision@{k} (Macro)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output) / f'head_med_tail_precision_macro_at_{k}.png', dpi=200)
        plt.close()

        # F1@k (Macro)
        plt.figure(figsize=(6,4))
        for i, lab in enumerate(labels):
            vals = []
            for b in xbins:
                topk_dict = recalls[lab][b].get('topk', {})
                vals.append(topk_dict.get(f'@{k}', {}).get('f1_macro', 0.0))
            plt.bar(xs + i*width, vals, width=width, label=lab)
        plt.xticks(xs + width*(len(labels)-1)/2, xbins)
        plt.ylim(0,1)
        plt.ylabel(f'F1@{k} (Macro)')
        plt.title(f'H/M/T F1@{k} (Macro)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output) / f'head_med_tail_f1_macro_at_{k}.png', dpi=200)
        plt.close()


def comprehensive_metrics_analysis(results_dirs: List[str], labels: List[str], output: str,
                                 train_counts: Counter = None, bins_source: str = "eval",
                                 bins_metric: str = "global_freq", quantiles: Tuple[float, float] = (0.1, 0.3),
                                 autoregressive_focused: bool = True):
    """Comprehensive evaluation analysis optimized for autoregressive generation.
    
    When autoregressive_focused=True, prioritizes top-1 accuracy, calibration, and per-class
    analysis over ranking metrics that aren't relevant for single-prediction generation.
    """
    print("Running comprehensive metrics analysis...")
    if autoregressive_focused:
        print("Autoregressive mode: focusing on top-1 prediction quality")
    
    all_metrics = {}
    
    for lab, rd in zip(labels, results_dirs):
        print(f"Analyzing {lab}...")
        rd = Path(rd)
    
        # Load basic results
        preds = targs = scores = None
        eval_ids_path = rd / 'eval_class_ids.npy'
        eval_raw_path = rd / 'eval_class_raw_ids.npy'
        
        # Load predictions and targets
        if (rd / 'predicted_cluster_cols.npy').exists():
            preds, targs, scores = _load_cls_results(rd)
        else:
            preds, targs, scores = _load_reg_results(rd)
            
        # Load eval class IDs
        eval_ids_are_global = False
        if eval_raw_path.exists():
            eval_ids = np.load(eval_raw_path)
            eval_ids_are_global = True
        elif eval_ids_path.exists():
            eval_ids = np.load(eval_ids_path)
        else:
            eval_ids = None
            
        # Convert to global ID space for binning
        is_cls = (rd / 'predicted_cluster_cols.npy').exists()
        if is_cls:
            targs_raw = rd / 'target_cluster_ids_raw.npy'
            preds_raw = rd / 'predicted_cluster_ids_raw.npy'
            if targs_raw.exists():
                targs_ids = np.load(targs_raw)
            elif eval_ids is not None and eval_ids_are_global:
                targs_ids = np.where(targs >= 0, eval_ids[targs], -1)
            else:
                targs_ids = targs
            if preds_raw.exists():
                preds_ids = np.load(preds_raw)
            elif eval_ids is not None and eval_ids_are_global:
                preds_ids = np.where(preds >= 0, eval_ids[preds], -1)
            else:
                preds_ids = preds
        else:
            targs_ids = targs
            preds_ids = preds
            
        # Load temporal data if available
        traj_path = rd / 'eval_traj_ids.npy'
        time_path = rd / 'eval_time_idx.npy'
        traj_ids = np.load(traj_path) if traj_path.exists() else None
        time_idx = np.load(time_path) if time_path.exists() else None
        if traj_ids is not None and traj_ids.ndim == 2:
            traj_ids = traj_ids.reshape(-1)
        if time_idx is not None and time_idx.ndim == 2:
            time_idx = time_idx.reshape(-1)
            
        # Define H/M/T bins
        valid_targs_mask = (targs_ids >= 0)
        if bins_metric == 'global_freq':
            base = _metric_global_freq(targs_ids, valid_targs_mask)
            if bins_source == "train" and train_counts is not None and len(train_counts) > 0:
                base = dict(train_counts)
            classes = np.array(list(base.keys()))
            metric_vals = np.array([base[c] for c in classes], dtype=float)
            order = np.argsort(-metric_vals)
        elif bins_metric == 'traj_support':
            if traj_ids is None:
                raise RuntimeError("traj_support metric requires eval_traj_ids.npy")
            base = _metric_traj_support(targs_ids, traj_ids, valid_targs_mask)
            classes = np.array(list(base.keys()))
            metric_vals = np.array([base[c] for c in classes], dtype=float)
            order = np.argsort(-metric_vals)
        elif bins_metric == 'median_gap':
            if traj_ids is None or time_idx is None:
                raise RuntimeError("median_gap metric requires eval_traj_ids.npy and eval_time_idx.npy")
            base = _metric_median_gap(targs_ids, traj_ids, time_idx, valid_targs_mask)
            classes = np.array(list(base.keys()))
            metric_vals = np.array([base[c] for c in classes], dtype=float)
            order = np.argsort(metric_vals)
        else:
            raise ValueError(f"Unknown bins_metric: {bins_metric}")
            
        classes = classes[order]
        n = len(classes)
        q1, q2 = quantiles
        cut1 = int(np.floor(n * q1))
        cut2 = int(np.floor(n * q2))
        head_set = set(classes[:max(1, cut1)])
        med_set = set(classes[max(1, cut1): max(1, cut2)])
        tail_set = set(classes[max(1, cut2):])
        
        # Initialize metrics for this model
        metrics = {
            'overall': {},
            'head': {},
            'med': {},
            'tail': {},
            'temporal': {},
            'calibration': {},
            'ranking': {}
        }
        
        # Compute overall metrics
        valid_mask = (targs_ids >= 0) & (preds_ids >= 0)
        if valid_mask.sum() > 0:
            valid_preds = preds_ids[valid_mask]
            valid_targs = targs_ids[valid_mask]
            
            # Basic metrics
            accuracy = (valid_preds == valid_targs).mean()
            unique_targs = len(np.unique(valid_targs))
            unique_preds = len(np.unique(valid_preds))
            coverage = len(set(valid_preds) & set(valid_targs)) / max(1, unique_targs)
            
            # Precision, Recall, F1 (macro and micro)
            try:
                precision_macro = precision_score(valid_targs, valid_preds, average='macro', zero_division=0)
                recall_macro = recall_score(valid_targs, valid_preds, average='macro', zero_division=0)
                f1_macro = f1_score(valid_targs, valid_preds, average='macro', zero_division=0)
                precision_micro = precision_score(valid_targs, valid_preds, average='micro', zero_division=0)
                recall_micro = recall_score(valid_targs, valid_preds, average='micro', zero_division=0)
                f1_micro = f1_score(valid_targs, valid_preds, average='micro', zero_division=0)
            except Exception:
                precision_macro = recall_macro = f1_macro = 0.0
                precision_micro = recall_micro = f1_micro = 0.0
                
            metrics['overall'] = {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'precision_micro': float(precision_micro),
                'recall_macro': float(recall_macro),
                'recall_micro': float(recall_micro),
                'f1_macro': float(f1_macro),
                'f1_micro': float(f1_micro),
                'unique_targets': int(unique_targs),
                'unique_predictions': int(unique_preds),
                'coverage': float(coverage),
                'support': int(valid_mask.sum())
            }
            
            # Per-bucket metrics
            for bucket_name, bucket_set in [('head', head_set), ('med', med_set), ('tail', tail_set)]:
                bucket_mask = np.array([t in bucket_set for t in valid_targs])
                if bucket_mask.sum() > 0:
                    bucket_preds = valid_preds[bucket_mask]
                    bucket_targs = valid_targs[bucket_mask]
                    
                    bucket_acc = (bucket_preds == bucket_targs).mean()
                    try:
                        bucket_prec = precision_score(bucket_targs, bucket_preds, average='macro', zero_division=0)
                        bucket_rec = recall_score(bucket_targs, bucket_preds, average='macro', zero_division=0)
                        bucket_f1 = f1_score(bucket_targs, bucket_preds, average='macro', zero_division=0)
                    except Exception:
                        bucket_prec = bucket_rec = bucket_f1 = 0.0
                        
                    metrics[bucket_name] = {
                        'accuracy': float(bucket_acc),
                        'precision_macro': float(bucket_prec),
                        'recall_macro': float(bucket_rec),
                        'f1_macro': float(bucket_f1),
                        'support': int(bucket_mask.sum()),
                        'unique_targets': int(len(np.unique(bucket_targs))),
                        'unique_predictions': int(len(np.unique(bucket_preds)))
                    }
                else:
                    metrics[bucket_name] = {'support': 0}
            
            # Ranking metrics (only if not autoregressive-focused)
            if not autoregressive_focused and scores is not None and eval_ids is not None:
                print("Computing ranking metrics...")
                metrics['ranking'] = compute_ranking_metrics(scores, targs, eval_ids, is_cls)
            elif autoregressive_focused:
                print("Skipping ranking metrics (not relevant for autoregressive generation)")
                
            # Autoregressive-specific metrics (always compute for generation quality)
            if autoregressive_focused:
                print("Computing autoregressive quality metrics...")
                scores_for_ar = scores if scores is not None else None
                metrics['autoregressive'] = compute_autoregressive_quality_metrics(
                    preds_ids, targs_ids, scores_for_ar
                )
                
            # Calibration metrics (ALWAYS important for autoregressive confidence)
            if scores is not None:
                print("Computing calibration metrics...")
                metrics['calibration'] = compute_calibration_metrics(scores, targs, eval_ids, is_cls)
                
            # Temporal metrics (if time data available)
            if time_idx is not None:
                print("Computing temporal metrics...")
                metrics['temporal'] = compute_temporal_metrics(
                    preds_ids, targs_ids, time_idx, valid_mask
                )
                
        all_metrics[lab] = metrics
        
    # Create comprehensive plots
    create_comprehensive_plots(all_metrics, labels, output, quantiles, bins_metric)
    
    # Save comprehensive results
    with open(Path(output) / 'comprehensive_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Comprehensive analysis saved to {output}")


def compute_ranking_metrics(scores: np.ndarray, targets: np.ndarray, eval_ids: np.ndarray, is_cls: bool) -> Dict:
    """Compute MRR, median rank, NDCG@K within evaluated class set."""
    try:
        # Map targets to score columns
        if is_cls:
            # For classification: targets are already column indices
            valid_mask = (targets >= 0) & (targets < len(eval_ids))
            valid_targets_cols = targets[valid_mask]
            valid_scores = scores[valid_mask]
        else:
            # For regression: map raw IDs to columns
            id2col = {cid: i for i, cid in enumerate(eval_ids)}
            valid_mask = np.array([t in id2col for t in targets]) & (targets >= 0)
            valid_targets_cols = np.array([id2col[t] for t in targets[valid_mask]])
            valid_scores = scores[valid_mask]
            
        if len(valid_targets_cols) == 0:
            return {'mrr': 0.0, 'median_rank': float('inf'), 'ndcg_at_5': 0.0, 'ndcg_at_10': 0.0}
            
        # Compute ranks (1-indexed)
        ranks = []
        ndcg_scores_5 = []
        ndcg_scores_10 = []
        
        for i, (target_col, score_row) in enumerate(zip(valid_targets_cols, valid_scores)):
            # Sort scores descending, get rank of target
            sorted_indices = np.argsort(-score_row)
            rank = np.where(sorted_indices == target_col)[0][0] + 1  # 1-indexed
            ranks.append(rank)
            
            # NDCG@K computation
            for k in [5, 10]:
                if k <= len(score_row):
                    dcg = 1.0 / np.log2(rank + 1) if rank <= k else 0.0
                    idcg = 1.0 / np.log2(2)  # Perfect ranking (target at position 1)
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    if k == 5:
                        ndcg_scores_5.append(ndcg)
                    else:
                        ndcg_scores_10.append(ndcg)
                        
        # Compute metrics
        ranks = np.array(ranks)
        mrr = np.mean(1.0 / ranks) if len(ranks) > 0 else 0.0
        median_rank = float(np.median(ranks)) if len(ranks) > 0 else float('inf')
        ndcg_5 = np.mean(ndcg_scores_5) if ndcg_scores_5 else 0.0
        ndcg_10 = np.mean(ndcg_scores_10) if ndcg_scores_10 else 0.0
        
        return {
            'mrr': float(mrr),
            'median_rank': float(median_rank),
            'ndcg_at_5': float(ndcg_5),
            'ndcg_at_10': float(ndcg_10),
            'evaluated_samples': int(len(ranks)),
            'mean_rank': float(np.mean(ranks)) if len(ranks) > 0 else float('inf')
        }
        
    except Exception as e:
        print(f"Warning: Ranking metrics failed: {e}")
        return {'error': str(e)}


def compute_calibration_metrics(scores: np.ndarray, targets: np.ndarray, eval_ids: Optional[np.ndarray], is_cls: bool) -> Dict:
    """Compute ECE, reliability curves, confidence vs accuracy."""
    try:
        # Convert scores to probabilities
        max_scores = scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)
        
        # Get predictions and confidence
        pred_cols = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # Map targets to column space if needed
        if is_cls:
            valid_mask = (targets >= 0) & (targets < scores.shape[1])
            target_cols = targets[valid_mask]
            pred_cols = pred_cols[valid_mask]
            confidence = confidence[valid_mask]
        else:
            if eval_ids is not None:
                id2col = {cid: i for i, cid in enumerate(eval_ids)}
                valid_mask = np.array([t in id2col for t in targets]) & (targets >= 0)
                target_cols = np.array([id2col[t] for t in targets[valid_mask]])
                pred_cols = pred_cols[valid_mask]
                confidence = confidence[valid_mask]
            else:
                raise RuntimeError('No eval_ids for regression calibration')
        
        if len(target_cols) == 0:
            return {'ece': 0.0, 'max_calibration_error': 0.0}
        
        # Compute correctness
        correct = (pred_cols == target_cols)
        
        # ECE computation (10 bins)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        max_cal_error = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.sum() / len(confidence)
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                cal_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * cal_error
                max_cal_error = max(max_cal_error, cal_error)
                
                bin_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'count': int(in_bin.sum()),
                    'calibration_error': float(cal_error)
                })
            else:
                bin_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'count': 0,
                    'calibration_error': 0.0
                })
        
        # Confidence vs accuracy deciles
        conf_deciles = np.percentile(confidence, np.linspace(0, 100, 11))
        conf_acc_data = []
        for i in range(10):
            mask = (confidence >= conf_deciles[i]) & (confidence < conf_deciles[i+1])
            if mask.sum() > 0:
                conf_acc_data.append({
                    'decile': i + 1,
                    'conf_range': [float(conf_deciles[i]), float(conf_deciles[i+1])],
                    'accuracy': float(correct[mask].mean()),
                    'count': int(mask.sum()),
                    'mean_confidence': float(confidence[mask].mean())
                })
        
        return {
            'ece': float(ece),
            'max_calibration_error': float(max_cal_error),
            'overall_accuracy': float(correct.mean()),
            'mean_confidence': float(confidence.mean()),
            'reliability_bins': bin_data,
            'confidence_accuracy_deciles': conf_acc_data,
            'evaluated_samples': int(len(correct))
        }
    except Exception as e:
        raise


def compute_autoregressive_quality_metrics(preds: np.ndarray, targets: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict:
    """Compute metrics specifically relevant for autoregressive generation."""
    try:
        valid_mask = (targets >= 0) & (preds >= 0)
        if valid_mask.sum() == 0:
            return {'error': 'No valid predictions'}
            
        valid_preds = preds[valid_mask]
        valid_targets = targets[valid_mask]
        
        # Core autoregressive metrics
        top1_accuracy = (valid_preds == valid_targets).mean()
        
        # Error analysis: what types of mistakes?
        class_counts = Counter(valid_targets)
        total_classes = len(class_counts)
        
        # Prediction diversity (are we always predicting the same few classes?)
        pred_counts = Counter(valid_preds)
        entropy_preds = -sum((count/len(valid_preds)) * np.log2(count/len(valid_preds) + 1e-8) 
                           for count in pred_counts.values())
        max_entropy = np.log2(min(total_classes, 50000))  # theoretical max
        normalized_entropy = entropy_preds / max_entropy if max_entropy > 0 else 0
        
        # Class coverage: how many different classes do we predict?
        pred_coverage = len(pred_counts) / total_classes if total_classes > 0 else 0
        
        # Frequency bias: do we over-predict common classes?
        common_classes = set([cls for cls, count in class_counts.most_common(int(0.1 * total_classes))])
        common_pred_rate = sum(1 for p in valid_preds if p in common_classes) / len(valid_preds)
        common_target_rate = sum(1 for t in valid_targets if t in common_classes) / len(valid_targets)
        frequency_bias = common_pred_rate / (common_target_rate + 1e-8)
        
        # Confidence-based metrics (if scores available)
        confidence_metrics = {}
        if scores is not None:
            # Convert to probabilities and get confidence
            max_scores = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)
            confidence = np.max(probs, axis=1)[valid_mask]
            correct = (valid_preds == valid_targets)
            
            # Confidence on correct vs incorrect predictions
            conf_correct = confidence[correct].mean() if correct.sum() > 0 else 0
            conf_incorrect = confidence[~correct].mean() if (~correct).sum() > 0 else 0
            confidence_gap = conf_correct - conf_incorrect
            
            # Threshold for "high confidence" predictions
            high_conf_thresh = np.percentile(confidence, 80)
            high_conf_mask = confidence >= high_conf_thresh
            high_conf_accuracy = correct[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0
            
            confidence_metrics = {
                'mean_confidence': float(confidence.mean()),
                'confidence_on_correct': float(conf_correct),
                'confidence_on_incorrect': float(conf_incorrect), 
                'confidence_gap': float(confidence_gap),
                'high_confidence_accuracy': float(high_conf_accuracy),
                'high_confidence_threshold': float(high_conf_thresh),
                'high_confidence_fraction': float(high_conf_mask.mean())
            }
        
        return {
            'top1_accuracy': float(top1_accuracy),
            'prediction_entropy': float(entropy_preds),
            'normalized_prediction_entropy': float(normalized_entropy),
            'class_coverage': float(pred_coverage),
            'frequency_bias': float(frequency_bias),
            'total_classes_in_targets': int(total_classes),
            'total_classes_predicted': int(len(pred_counts)),
            'evaluated_samples': int(valid_mask.sum()),
            **confidence_metrics
        }
        
    except Exception as e:
        print(f"Warning: Autoregressive quality metrics failed: {e}")
        return {'error': str(e)}


def compute_temporal_metrics(preds: np.ndarray, targets: np.ndarray, time_idx: np.ndarray, valid_mask: np.ndarray) -> Dict:
    """Compute accuracy vs time bins."""
    try:
        valid_preds = preds[valid_mask]
        valid_targets = targets[valid_mask]
        valid_times = time_idx[valid_mask]
        
        if len(valid_times) == 0:
            return {'error': 'No valid temporal data'}
            
        # Bin by time quantiles
        time_quantiles = np.percentile(valid_times, [0, 25, 50, 75, 100])
        temporal_data = []
        
        for i in range(4):
            mask = (valid_times >= time_quantiles[i]) & (valid_times < time_quantiles[i+1])
            if i == 3:  # Include upper bound for last bin
                mask = (valid_times >= time_quantiles[i]) & (valid_times <= time_quantiles[i+1])
                
            if mask.sum() > 0:
                bin_acc = (valid_preds[mask] == valid_targets[mask]).mean()
                temporal_data.append({
                    'quartile': i + 1,
                    'time_range': [float(time_quantiles[i]), float(time_quantiles[i+1])],
                    'accuracy': float(bin_acc),
                    'count': int(mask.sum())
                })
                
        return {
            'temporal_quartiles': temporal_data,
            'time_range': [float(valid_times.min()), float(valid_times.max())],
            'evaluated_samples': int(len(valid_times))
        }
        
    except Exception as e:
        print(f"Warning: Temporal metrics failed: {e}")
        return {'error': str(e)}


def create_comprehensive_plots(all_metrics: Dict, labels: List[str], output: str, quantiles: Tuple[float, float], bins_metric: str):
    """Create comprehensive visualization plots."""
    output_path = Path(output)
    
    # 1. Precision/Recall/F1 comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comprehensive Metrics Comparison', fontsize=14)
    
    # Overall metrics
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[i // 2, i % 2]
        values = []
        for lab in labels:
            if lab in all_metrics and 'overall' in all_metrics[lab]:
                values.append(all_metrics[lab]['overall'].get(metric, 0))
            else:
                values.append(0)
        
        bars = ax.bar(labels, values, alpha=0.7)
        ax.set_title(name)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comprehensive_metrics_overall.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Head/Med/Tail breakdown
    buckets = ['head', 'med', 'tail']
    bucket_colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'H/M/T Performance Breakdown (metric={bins_metric}, q={quantiles})', fontsize=14)
    
    for i, metric in enumerate(['accuracy', 'precision_macro', 'f1_macro']):
        ax = axes[i]
        x_pos = np.arange(len(labels))
        width = 0.25
        
        for j, bucket in enumerate(buckets):
            values = []
            for lab in labels:
                if lab in all_metrics and bucket in all_metrics[lab] and 'support' in all_metrics[lab][bucket]:
                    if all_metrics[lab][bucket]['support'] > 0:
                        values.append(all_metrics[lab][bucket].get(metric, 0))
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            ax.bar(x_pos + j*width, values, width, label=bucket.title(), 
                  color=bucket_colors[j], alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path / 'hmt_performance_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Ranking metrics (if available)
    ranking_metrics = ['mrr', 'median_rank', 'ndcg_at_5', 'ndcg_at_10']
    ranking_names = ['MRR', 'Median Rank', 'NDCG@5', 'NDCG@10']
    
    # Check if any model has ranking metrics
    has_ranking = any(
        lab in all_metrics and 'ranking' in all_metrics[lab] and 'mrr' in all_metrics[lab]['ranking']
        for lab in labels
    )
    
    if has_ranking:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Ranking Metrics', fontsize=14)
        
        for i, (metric, name) in enumerate(zip(ranking_metrics, ranking_names)):
            ax = axes[i // 2, i % 2]
            values = []
            for lab in labels:
                if (lab in all_metrics and 'ranking' in all_metrics[lab] and 
                    metric in all_metrics[lab]['ranking']):
                    val = all_metrics[lab]['ranking'][metric]
                    # Handle infinite median ranks
                    if metric == 'median_rank' and np.isinf(val):
                        val = len(labels) * 1000  # Large number for plotting
                    values.append(val)
                else:
                    values.append(0)
            
            bars = ax.bar(labels, values, alpha=0.7)
            ax.set_title(name)
            ax.tick_params(axis='x', rotation=45)
            
            # Special handling for median rank (lower is better)
            if metric == 'median_rank':
                ax.set_ylabel('Rank (lower is better)')
            else:
                ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if metric == 'median_rank' and val >= 1000:
                    label = '∞'
                else:
                    label = f'{val:.3f}' if val < 10 else f'{val:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'ranking_metrics.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # 4. Calibration plots (if available)
    has_calibration = any(
        lab in all_metrics and 'calibration' in all_metrics[lab] and 'ece' in all_metrics[lab]['calibration']
        for lab in labels
    )
    
    if has_calibration:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Calibration', fontsize=14)
        
        # ECE comparison
        ax = axes[0]
        ece_values = []
        for lab in labels:
            if (lab in all_metrics and 'calibration' in all_metrics[lab] and 
                'ece' in all_metrics[lab]['calibration']):
                ece_values.append(all_metrics[lab]['calibration']['ece'])
            else:
                ece_values.append(0)
        
        bars = ax.bar(labels, ece_values, alpha=0.7, color='coral')
        ax.set_title('Expected Calibration Error')
        ax.set_ylabel('ECE (lower is better)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, ece_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ece_values)*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Reliability diagram for first model (if available)
        ax = axes[1]
        first_model = labels[0]
        if (first_model in all_metrics and 'calibration' in all_metrics[first_model] and 
            'reliability_bins' in all_metrics[first_model]['calibration']):
            
            rel_data = all_metrics[first_model]['calibration']['reliability_bins']
            bin_centers = [(bin_data['bin_lower'] + bin_data['bin_upper']) / 2 for bin_data in rel_data]
            accuracies = [bin_data['accuracy'] for bin_data in rel_data]
            counts = [bin_data['count'] for bin_data in rel_data]
            
            # Plot reliability diagram
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            scatter = ax.scatter(bin_centers, accuracies, s=[c/10 for c in counts], 
                               alpha=0.7, label=f'{first_model} (size=count/10)')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Reliability Diagram: {first_model}')
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'calibration_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Comprehensive plots saved to {output_path}")


def create_confusion_heatmap(results_dirs: List[str], labels: List[str], output: str, top_n: int = 20):
    """Create confusion heatmap for top-N most frequent classes."""
    print(f"Creating confusion heatmap for top-{top_n} classes...")
    
    for lab, rd in zip(labels, results_dirs):
        rd = Path(rd)
        
        # Load results
        if (rd / 'predicted_cluster_cols.npy').exists():
            preds, targs, _ = _load_cls_results(rd)
        else:
            preds, targs, _ = _load_reg_results(rd)
            
        # Convert to global ID space if possible
        is_cls = (rd / 'predicted_cluster_cols.npy').exists()
        if is_cls:
            eval_raw_path = rd / 'eval_class_raw_ids.npy'
            if eval_raw_path.exists():
                eval_ids = np.load(eval_raw_path)
                valid_mask = (targs >= 0) & (targs < len(eval_ids)) & (preds >= 0) & (preds < len(eval_ids))
                if valid_mask.sum() > 0:
                    targs_ids = eval_ids[targs[valid_mask]]
                    preds_ids = eval_ids[preds[valid_mask]]
                else:
                    continue
            else:
                valid_mask = (targs >= 0) & (preds >= 0)
                targs_ids = targs[valid_mask]
                preds_ids = preds[valid_mask]
        else:
            valid_mask = (targs >= 0) & (preds >= 0)
            targs_ids = targs[valid_mask]
            preds_ids = preds[valid_mask]
            
        if len(targs_ids) == 0:
            continue
            
        # Get top-N most frequent target classes
        class_counts = Counter(targs_ids)
        top_classes = [cls for cls, _ in class_counts.most_common(top_n)]
        
        # Filter to only these classes
        mask = np.isin(targs_ids, top_classes)
        if mask.sum() == 0:
            continue
            
        filtered_targs = targs_ids[mask]
        filtered_preds = preds_ids[mask]
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(filtered_targs, filtered_preds, labels=top_classes)
        
        # Normalize by rows (true classes)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.title(f'Confusion Matrix: {lab} (Top-{top_n} Classes, Row-Normalized)')
        plt.colorbar(label='Recall')
        
        # Add text annotations for significant values
        thresh = 0.1
        for i in range(len(top_classes)):
            for j in range(len(top_classes)):
                if cm_norm[i, j] > thresh:
                    plt.text(j, i, f'{cm_norm[i, j]:.2f}', 
                           horizontalalignment="center", color="white" if cm_norm[i, j] > 0.5 else "black")
        
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks(range(len(top_classes)), [f'{cls}' for cls in top_classes], rotation=45)
        plt.yticks(range(len(top_classes)), [f'{cls}' for cls in top_classes])
        plt.tight_layout()
        
        plt.savefig(Path(output) / f'confusion_matrix_{lab.replace(" ", "_")}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    print(f"Confusion heatmaps saved to {output}")


def coverage_vs_step(results_dirs: List[str], labels: List[str], output: str):
    """Plot unique predicted classes vs step index across checkpoints for each label series.
    results_dirs: list of dirs; each list element may be a comma-separated series for a label.
    """
    plt.figure(figsize=(6,4))
    for lab, series in zip(labels, results_dirs):
        dirs = [Path(x) for x in series.split(',')]
        cov = []
        for d in dirs:
            preds = None
            if (d / 'predicted_cluster_cols.npy').exists():
                preds = np.load(d / 'predicted_cluster_cols.npy')
            elif (d / 'predicted_cluster_ids.npy').exists():
                preds = np.load(d / 'predicted_cluster_ids.npy')
            else:
                continue
            cov.append(len(set(preds.tolist())))
        if cov:
            plt.plot(range(len(cov)), cov, marker='o', label=lab)
    plt.xlabel('Checkpoint index')
    plt.ylabel('Unique predicted classes')
    plt.title('Coverage@N (unique) vs checkpoint')
    plt.legend()
    plt.tight_layout()
    Path(output).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output)/'coverage_vs_step.png', dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Advanced EMPROT evaluation analyses')
    ap.add_argument('--output', type=str, required=True, help='Output directory for plots')

    # Head/Med/Tail recall
    ap.add_argument('--hmt_results', nargs='*', default=None,
                    help='Cluster_results dirs for head/med/tail recall (one per label)')
    ap.add_argument('--hmt_labels', nargs='*', default=None, help='Labels for H/M/T plot')
    ap.add_argument('--hmt_bins_source', type=str, choices=['train','eval'], default='eval',
                    help="Use 'train' frequencies or 'eval' (test) targets to define H/M/T bins")
    ap.add_argument('--hmt_bins_metric', type=str,
                    choices=['global_freq','median_gap','traj_local'],
                    default='traj_local',
                    help='How to score classes for H/M/T binning.')
    ap.add_argument('--hmt_quantiles', nargs=2, type=float, default=[0.9, 0.99],
                    help='Two cut points (e.g., 0.1 0.3). Head=top q1, Med=next q2-q1, Tail=rest.')
    ap.add_argument('--hmt_train_data_dir', type=str, default=None,
                    help='If set, define buckets from TRAIN frequencies at this data_dir')
    ap.add_argument('--hmt_train_metadata_path', type=str, default=None,
                    help='Metadata path for training counts')
    ap.add_argument('--hmt_seed', type=int, default=42)
    ap.add_argument('--hmt_stride', type=int, default=10)
    ap.add_argument('--hmt_seq_len', type=int, default=5)
    ap.add_argument('--hmt_num_workers', type=int, default=2)
    ap.add_argument('--hmt_max_train_batches', type=int, default=200,
                    help='Limit train batches scanned for class counts (None=all)')

    # Comprehensive metrics analysis
    ap.add_argument('--comprehensive_results', nargs='*', default=None,
                    help='Cluster_results dirs for comprehensive analysis (one per label)')
    ap.add_argument('--comprehensive_labels', nargs='*', default=None,
                    help='Labels for comprehensive analysis')
    ap.add_argument('--autoregressive_focused', action='store_true', default=True,
                    help='Focus on top-1 accuracy and generation quality metrics (default: True)')
    ap.add_argument('--ranking_metrics', action='store_true', default=False,
                    help='Include ranking metrics (MRR, NDCG) - mainly for retrieval tasks')

    # Confusion heatmaps
    ap.add_argument('--confusion_results', nargs='*', default=None,
                    help='Cluster_results dirs for confusion matrices (one per label)')
    ap.add_argument('--confusion_labels', nargs='*', default=None,
                    help='Labels for confusion matrices')
    ap.add_argument('--confusion_top_n', type=int, default=20,
                    help='Number of top classes to include in confusion matrix')

    # Coverage vs step (each item may be a comma-separated series per label)
    ap.add_argument('--cov_series', nargs='*', default=None,
                    help='Comma-separated series of cluster_results dirs per label')
    ap.add_argument('--cov_labels', nargs='*', default=None, help='Labels for coverage plot')

    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Shared train counts for binning
    train_counts = None
    if args.hmt_bins_source == 'train' and args.hmt_train_data_dir and args.hmt_train_metadata_path:
        print("Computing training class counts for binning...")
        train_counts = _compute_train_class_counts(
            args.hmt_train_data_dir,
            args.hmt_train_metadata_path,
            seed=args.hmt_seed,
            batch_size=32,
            sequence_length=args.hmt_seq_len,
            stride=args.hmt_stride,
            num_workers=args.hmt_num_workers,
            max_batches=(None if args.hmt_max_train_batches <= 0 else args.hmt_max_train_batches),
        )

    # Head/Med/Tail recall
    if args.hmt_results and args.hmt_labels:
        assert len(args.hmt_results) == len(args.hmt_labels)
        print("Running Head/Med/Tail recall analysis...")
        head_med_tail_recall(args.hmt_results, args.hmt_labels, str(out),
                             train_counts=train_counts,
                             bins_source=args.hmt_bins_source,
                             bins_metric=args.hmt_bins_metric,
                             quantiles=tuple(args.hmt_quantiles))

    # Comprehensive metrics analysis
    if args.comprehensive_results and args.comprehensive_labels:
        assert len(args.comprehensive_results) == len(args.comprehensive_labels)
        print("Running comprehensive metrics analysis...")
        comprehensive_metrics_analysis(
            args.comprehensive_results, args.comprehensive_labels, str(out),
            train_counts=train_counts,
            bins_source=args.hmt_bins_source,
            bins_metric=args.hmt_bins_metric,
            quantiles=tuple(args.hmt_quantiles),
            autoregressive_focused=args.autoregressive_focused and not args.ranking_metrics
        )

    # Confusion matrices
    if args.confusion_results and args.confusion_labels:
        assert len(args.confusion_results) == len(args.confusion_labels)
        print("Creating confusion heatmaps...")
        create_confusion_heatmap(
            args.confusion_results, args.confusion_labels, str(out),
            top_n=args.confusion_top_n
        )

    # Coverage vs step
    if args.cov_series and args.cov_labels:
        assert len(args.cov_series) == len(args.cov_labels)
        print("Running coverage vs step analysis...")
        coverage_vs_step(args.cov_series, args.cov_labels, str(out))

    print(f"Saved advanced analysis to {out}")


if __name__ == '__main__':
    main()
