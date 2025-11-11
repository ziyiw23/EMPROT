#!/usr/bin/env python3
"""
Analyze change vs. stale statistics on the TRAIN split used for EMPROT training.

This script reproduces the exact train/val/test split via the project's
`create_dataloaders` utility and then computes:
  1) Window-level stats (sequence windows): how many examples have any change
     at the last step (target vs previous step) vs. entirely stale at the last step.
  2) Token-level stats (individual time transitions): across all consecutive pairs
     inside each window, how many tokens change vs. stay the same.

Notes:
  - We rely on `batch['input_cluster_ids']` of shape (B, T, N) and
    `batch['targets']['target_cluster_ids']` of shape (B, N) when available.
  - Valid tokens are identified by target IDs >= 0 for last-step metrics,
    and by having non-negative cluster IDs at both t and t+1 for pairwise metrics.
  - Sequence length is typically 5; the script will report if it observes a
    different value.

Examples:
  python scripts/analysis/analyze_train_changes.py \
    --data-dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings/ \
    --metadata-path /oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
    --sequence-length 5 --stride 1 --batch-size 8 --seed 42

  # Force fixed sequence length (min=max=sequence_length)
  python scripts/analysis/analyze_train_changes.py \
    --config configs/depth_context_hybrid.yaml --seed 42 --fixed-length

  # Or read from a config YAML (fields under data: data_dir, metadata_path, sequence_length, stride):
  python scripts/analysis/analyze_train_changes.py --config configs/depth_context_hybrid.yaml --seed 42
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add project root to import emprot
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml  # Optional; used only if --config is provided
except Exception:
    yaml = None

from emprot.data.dataset import create_dataloaders


def _load_data_args_from_config(config_path: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Load (data_dir, metadata_path, sequence_length, stride, batch_size) from a YAML config.
    Returns None for any missing fields. Does not raise if yaml isn't available.
    """
    if yaml is None:
        return None, None, None, None, None
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None, None, None, None, None

    data_cfg = cfg.get('data', cfg) if isinstance(cfg, dict) else {}
    data_dir = data_cfg.get('data_dir')
    metadata_path = data_cfg.get('metadata_path')
    sequence_length = data_cfg.get('sequence_length')
    stride = data_cfg.get('stride')
    batch_size = data_cfg.get('batch_size')
    return data_dir, metadata_path, sequence_length, stride, batch_size


def analyze_changes(train_loader, max_batches: int = -1) -> dict:
    """
    Iterate over the TRAIN loader and accumulate change/stale statistics.

    Returns a dictionary with raw counts and percentages.
    """
    total_windows = 0
    windows_with_change = 0
    windows_all_stale = 0
    windows_skipped_no_targets = 0

    tokens_changed_last = 0
    tokens_stayed_last = 0
    tokens_valid_last = 0

    tokens_changed_all_pairs = 0
    tokens_stayed_all_pairs = 0
    tokens_valid_all_pairs = 0

    observed_T: Optional[int] = None

    # Per-window residue-level fraction changed (last step only)
    frac_changed_last_per_window = []  # list[float]

    for bi, batch in enumerate(train_loader):
        if max_batches > 0 and bi >= max_batches:
            break

        # Expect (B, T, N) input IDs
        ids = batch.get('input_cluster_ids', None)
        if ids is None or not isinstance(ids, torch.Tensor) or ids.dim() != 3:
            continue

        B, T, N = ids.shape
        if observed_T is None:
            observed_T = int(T)

        # 1) Window-level: last-step change vs. stay (requires targets)
        targets = batch.get('targets', {})
        t_last = None
        if isinstance(targets, dict) and ('target_cluster_ids' in targets):
            t_last = targets['target_cluster_ids']  # (B, N)
        elif 'target_cluster_ids' in batch:
            # Fallback if dataset surfaces targets at top-level
            t_last = batch['target_cluster_ids']

        if isinstance(t_last, torch.Tensor) and t_last.dim() == 2 and T >= 2:
            prev_last = ids[:, T - 2, :]  # (B, N)
            valid_last = (t_last >= 0)
            changed_last = (t_last != prev_last) & valid_last

            # Per-window decision: any change at last step?
            changed_per_window = changed_last.any(dim=1)
            valid_per_window = valid_last.any(dim=1)

            total_windows += int(valid_per_window.sum().item())
            windows_with_change += int((changed_per_window & valid_per_window).sum().item())
            windows_all_stale += int(((~changed_per_window) & valid_per_window).sum().item())

            # Token-level (last step only)
            vcount = int(valid_last.sum().item())
            ccount = int(changed_last.sum().item())
            tokens_valid_last += vcount
            tokens_changed_last += ccount
            tokens_stayed_last += int(vcount - ccount)

            # Per-window residue fraction changed (only for windows with any valid residues)
            per_window_valid_counts = valid_last.sum(dim=1)  # (B,)
            per_window_changed_counts = changed_last.sum(dim=1)  # (B,)
            for b in range(per_window_valid_counts.numel()):
                v_b = int(per_window_valid_counts[b].item())
                if v_b > 0:
                    c_b = int(per_window_changed_counts[b].item())
                    frac_changed_last_per_window.append(float(c_b / v_b))
        else:
            windows_skipped_no_targets += B

        # 2) Token-level across all internal consecutive pairs (t -> t+1) within window
        if T >= 2:
            for t in range(T - 1):
                ids_t = ids[:, t, :]
                ids_tp1 = ids[:, t + 1, :]
                valid_pair = (ids_t >= 0) & (ids_tp1 >= 0)
                changes = (ids_tp1 != ids_t) & valid_pair
                vcount = int(valid_pair.sum().item())
                ccount = int(changes.sum().item())
                tokens_valid_all_pairs += vcount
                tokens_changed_all_pairs += ccount
                tokens_stayed_all_pairs += int(vcount - ccount)

    # Avoid division by zero
    def _pct(num: int, den: int) -> float:
        return (float(num) / float(max(1, den))) * 100.0

    # Compute simple stats for per-window last-step fractions
    def _quantiles(arr: list) -> dict:
        try:
            import numpy as _np
            a = _np.asarray(arr, dtype=_np.float64)
            if a.size == 0:
                return {'count': 0}
            return {
                'count': int(a.size),
                'mean': float(a.mean()),
                'median': float(_np.median(a)),
                'p10': float(_np.quantile(a, 0.10)),
                'p25': float(_np.quantile(a, 0.25)),
                'p75': float(_np.quantile(a, 0.75)),
                'p90': float(_np.quantile(a, 0.90)),
                'min': float(a.min()),
                'max': float(a.max()),
            }
        except Exception:
            return {'count': len(arr)}

    summary = {
        'observed_sequence_length': int(observed_T) if observed_T is not None else None,
        'window_level': {
            'total_windows_with_targets': int(total_windows),
            'windows_with_change_last_step': int(windows_with_change),
            'windows_all_stale_last_step': int(windows_all_stale),
            'windows_skipped_no_targets': int(windows_skipped_no_targets),
            'pct_windows_with_change_last_step': _pct(windows_with_change, total_windows),
            'pct_windows_all_stale_last_step': _pct(windows_all_stale, total_windows),
            'per_window_fraction_changed_last_step_stats': _quantiles(frac_changed_last_per_window),
        },
        'token_level_last_step': {
            'tokens_valid_last': int(tokens_valid_last),
            'tokens_changed_last': int(tokens_changed_last),
            'tokens_stayed_last': int(tokens_stayed_last),
            'pct_tokens_changed_last': _pct(tokens_changed_last, tokens_valid_last),
            'pct_tokens_stayed_last': _pct(tokens_stayed_last, tokens_valid_last),
        },
        'token_level_all_pairs': {
            'tokens_valid_all_pairs': int(tokens_valid_all_pairs),
            'tokens_changed_all_pairs': int(tokens_changed_all_pairs),
            'tokens_stayed_all_pairs': int(tokens_stayed_all_pairs),
            'pct_tokens_changed_all_pairs': _pct(tokens_changed_all_pairs, tokens_valid_all_pairs),
            'pct_tokens_stayed_all_pairs': _pct(tokens_stayed_all_pairs, tokens_valid_all_pairs),
        },
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description='Analyze change vs. stale stats on TRAIN split')
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--metadata-path', type=str, default=None)
    ap.add_argument('--sequence-length', type=int, default=5)
    ap.add_argument('--min-sequence-length', type=int, default=None, help='Override min sequence length (default follows training behavior)')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42, help='Seed to match trainer splits')
    ap.add_argument('--max-batches', type=int, default=-1, help='Limit batches for a quick estimate (-1 for full)')
    ap.add_argument('--config', type=str, default=None, help='Optional YAML config to pull data.* fields')
    ap.add_argument('--fixed-length', action='store_true', help='Force min_sequence_length=sequence_length (fixed T)')
    args = ap.parse_args()

    # Optionally read data args from YAML
    if args.config:
        ddir, mpath, slen, stride, bsz = _load_data_args_from_config(args.config)
        if args.data_dir is None and ddir is not None:
            args.data_dir = ddir
        if args.metadata_path is None and mpath is not None:
            args.metadata_path = mpath
        if args.sequence_length in (None, 0) and slen is not None:
            args.sequence_length = int(slen)
        if args.stride in (None, 0) and stride is not None:
            args.stride = int(stride)
        if args.batch_size in (None, 0) and bsz is not None:
            args.batch_size = int(bsz)

    assert args.data_dir and args.metadata_path, '--data-dir and --metadata-path are required (or provide --config)'

    # Build dataloaders using the same split function
    # Determine min sequence length policy
    if args.min_sequence_length is not None:
        min_seq_len = int(args.min_sequence_length)
    elif args.fixed_length:
        min_seq_len = int(args.sequence_length)
    else:
        # Match typical training behavior: allow variability, but at least 2–3
        min_seq_len = max(2, min(3, args.sequence_length))

    train_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        max_sequence_length=args.sequence_length,
        # Control min length according to flags
        min_sequence_length=min_seq_len,
        stride=args.stride,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    summary = analyze_changes(train_loader, max_batches=args.max_batches)

    # Print concise human-readable summary first
    obs_T = summary.get('observed_sequence_length', None)
    print('=== Train Split Change/Stale Analysis ===')
    if obs_T is not None and obs_T != args.sequence_length:
        print(f' Note: observed sequence length T={obs_T} (requested={args.sequence_length})')
    elif obs_T is not None:
        print(f' Observed sequence length T={obs_T}')

    wl = summary['window_level']
    print(f" Windows (with targets): total={wl['total_windows_with_targets']}")
    print(f"  - with change at last step: {wl['windows_with_change_last_step']} ({wl['pct_windows_with_change_last_step']:.2f}%)")
    print(f"  - all stale at last step:  {wl['windows_all_stale_last_step']} ({wl['pct_windows_all_stale_last_step']:.2f}%)")
    if wl['windows_skipped_no_targets'] > 0:
        print(f"  - skipped (no targets available): {wl['windows_skipped_no_targets']}")

    tl = summary['token_level_last_step']
    print(f" Tokens (last step only): valid={tl['tokens_valid_last']}")
    print(f"  - changed: {tl['tokens_changed_last']} ({tl['pct_tokens_changed_last']:.2f}%)")
    print(f"  - stayed:  {tl['tokens_stayed_last']} ({tl['pct_tokens_stayed_last']:.2f}%)")

    apair = summary['token_level_all_pairs']
    print(f" Tokens (all consecutive pairs in windows): valid={apair['tokens_valid_all_pairs']}")
    print(f"  - changed: {apair['tokens_changed_all_pairs']} ({apair['pct_tokens_changed_all_pairs']:.2f}%)")
    print(f"  - stayed:  {apair['tokens_stayed_all_pairs']} ({apair['pct_tokens_stayed_all_pairs']:.2f}%)")

    # Also emit JSON for downstream use
    try:
        import json
        print('\n-- JSON --')
        print(json.dumps(summary, indent=2))
    except Exception:
        pass


if __name__ == '__main__':
    main()


