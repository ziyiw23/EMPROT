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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from emprot.data.dataset import create_dataloaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--metadata-path', required=True)
    ap.add_argument('--sequence-length', type=int, default=5)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--output', type=str, default='train_class_counts.json')
    ap.add_argument('--seed', type=int, default=42, help='Seed for reproducible train/val/test split')
    ap.add_argument('--max-batches', type=int, default=-1, help='Limit batches for a quick estimate (-1 for full)')
    args = ap.parse_args()

    train_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        max_sequence_length=args.sequence_length,
        stride=args.stride,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Lazy allocate counts on first batch when we see max class id
    counts = None
    seen = 0
    for bi, batch in enumerate(train_loader):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        targets = batch.get('targets', {})
        t = None
        if isinstance(targets, dict) and 'target_cluster_ids' in targets:
            t = targets['target_cluster_ids']  # (B, N)
        elif 'short_term_target' in batch:
            # No discrete IDs; skip
            continue
        if t is None:
            continue
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
