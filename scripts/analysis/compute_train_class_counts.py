#!/usr/bin/env python3
"""
Compute training class counts for Balanced Softmax / Logit Adjustment.

Outputs a JSON file with a length-C array of counts (C≈50000) that you can
reference in your training config as `train_class_counts`.

Usage:
  python scripts/analysis/compute_train_class_counts.py \
    --data-dir /path/to/lmdb_root \
    --metadata-path /path/to/traj_metadata.csv \
    --sequence-length 5 --stride 10 --batch-size 8 \
    --output counts_train.json

This iterates only the TRAIN split created by create_dataloaders and counts
target_cluster_ids where available.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from emprot.data.dataset import create_dataloaders


def _update_unique_from_tensor(tensor, accumulator):
    if tensor is None:
        return
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.numel() == 0:
        return
    flat = tensor.view(-1)
    valid = flat >= 0
    if not torch.any(valid):
        return
    unique_vals = torch.unique(flat[valid])
    accumulator.update(int(v) for v in unique_vals.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--metadata-path', required=True)
    ap.add_argument(
        '--num-full-res-frames',
        type=int,
        default=None,
        help='Number of recent full-resolution frames (K). Defaults to legacy --sequence-length value.',
    )
    ap.add_argument(
        '--sequence-length',
        type=int,
        default=None,
        help='[DEPRECATED] Alias for --num-full-res-frames.',
    )
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--output', type=str, default='train_class_counts.json')
    ap.add_argument('--seed', type=int, default=42, help='Seed for reproducible train/val/test split')
    ap.add_argument('--max-batches', type=int, default=-1, help='Limit batches for a quick estimate (-1 for full)')
    ap.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bar.')
    ap.add_argument('--max-train-proteins', type=int, default=0, help='Sample up to this many proteins from the train split (0 = all).')
    args = ap.parse_args()

    num_full_res_frames = args.num_full_res_frames
    if num_full_res_frames is None:
        if args.sequence_length is not None:
            num_full_res_frames = args.sequence_length
        else:
            num_full_res_frames = 5

    train_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        num_full_res_frames=num_full_res_frames,
        stride=args.stride,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_proteins=int(args.max_train_proteins) if args.max_train_proteins and args.max_train_proteins > 0 else None,
    )

    # Lazy allocate counts on first batch when we see max class id
    counts = None
    seen = 0
    history_ids = set()
    future_ids = set()
    target_ids = set()
    pbar = None
    loader_iter = train_loader
    if not args.no_progress:
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        pbar = tqdm(loader_iter, total=total_batches, desc='Counting train classes', dynamic_ncols=True)
        loader_iter = pbar
    for bi, batch in enumerate(loader_iter):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        _update_unique_from_tensor(batch.get('input_cluster_ids'), history_ids)
        _update_unique_from_tensor(batch.get('future_cluster_ids'), future_ids)
        targets = batch.get('targets', {})
        t = None
        if isinstance(targets, dict) and 'target_cluster_ids' in targets:
            t = targets['target_cluster_ids']  # (B, N)
        elif 'short_term_target' in batch:
            # No discrete IDs; skip
            continue
        if t is None:
            continue
        _update_unique_from_tensor(t, target_ids)
        t = t.view(-1)
        valid = t >= 0
        tv = t[valid]
        if tv.numel() == 0:
            continue
        max_id = int(tv.max().item())
        if counts is None:
            # Allocate conservatively to max seen so far; will grow if needed
            counts = torch.zeros(max_id + 1, dtype=torch.long)
        elif max_id >= counts.numel():
            new_counts = torch.zeros(max_id + 1, dtype=torch.long)
            new_counts[:counts.numel()] = counts
            counts = new_counts
        binc = torch.bincount(tv, minlength=counts.numel())
        counts[:binc.numel()] += binc
        seen += int(tv.numel())
        if (bi + 1) % 50 == 0:
            print(f"Processed {bi+1} batches, examples: {seen:,}")
        if pbar is not None:
            pbar.set_postfix({'batches': bi + 1, 'examples': f'{seen:,}'}, refresh=False)

    if pbar is not None:
        pbar.close()

    unique_history = len(history_ids)
    unique_future = len(future_ids)
    unique_targets = len(target_ids)
    residue_unique = len(history_ids | future_ids)
    total_unique = len(history_ids | future_ids | target_ids)
    print(f"Unique history cluster IDs: {unique_history:,}")
    print(f"Unique future cluster IDs: {unique_future:,}")
    print(f"Unique target cluster IDs: {unique_targets:,}")
    print(f"Unique residue cluster IDs (history ∪ future): {residue_unique:,}")
    print(f"Unique clusters overall (history ∪ future ∪ targets): {total_unique:,}")

    if counts is None:
        print("No discrete targets found; counts not computed.")
        return

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        json.dump([int(x) for x in counts.tolist()], f)
    print(f"Saved train_class_counts to {outp} (length={counts.numel()})")


if __name__ == '__main__':
    main()
