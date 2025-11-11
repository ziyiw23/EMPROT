#!/usr/bin/env python3
"""
Compute class counts and class-balanced weights from the training split.

Usage (cluster):
  python scripts/compute_class_prior.py \
    --data_dir /scratch/groups/rbaltman/ziyiw23/traj_embeddings \
    --metadata /oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv \
    --batch_size 8 --K 3 --F 5 --out counts.json --weights weights.json
"""
import argparse
import os
import sys

import torch

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emprot.data.dataset import create_dataloaders
from emprot.losses.prior_utils import (
    compute_class_counts,
    compute_effective_number_weights,
    save_vector,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--K', type=int, default=3, help='num_full_res_frames')
    ap.add_argument('--F', type=int, default=5, help='future_horizon')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='class_counts.json')
    ap.add_argument('--weights', type=str, default='class_weights.json')
    ap.add_argument('--beta', type=float, default=0.999)
    args = ap.parse_args()

    train_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        num_full_res_frames=args.K,
        stride=args.stride,
        future_horizon=args.F,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    counts = compute_class_counts(train_loader, horizon_mode='f1')
    save_vector(counts, args.out)
    print(f"Saved counts -> {args.out} (C={counts.numel()}, total={int(counts.sum().item())})")

    weights = compute_effective_number_weights(counts, beta=args.beta)
    save_vector(weights, args.weights)
    print(f"Saved weights -> {args.weights} (min={float(weights.min()):.4f}, max={float(weights.max()):.4f})")


if __name__ == '__main__':
    main()


