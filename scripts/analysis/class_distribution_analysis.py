#!/usr/bin/env python3
"""
Class/Distribution Analysis for EMPROT evaluation outputs

Reads saved arrays from scripts/evaluate_single_model.py and produces:
 - Target class frequency histogram (log scale)
 - Target class frequency CDF
 - Accuracy vs. target class frequency bins (deciles)
 - Coverage@K curve (how coverage grows with K)

Works for both classification and regression evaluations.
Inputs are the files saved under evaluation_results/.../cluster_results/.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import sys, os

import numpy as np
import matplotlib.pyplot as plt

# For dataset-based analysis
from emprot.data.dataset import create_dataloaders

seed = 42


def load_predictions_and_targets(results_dir: Path):
    """Load predictions and targets from the evaluator's cluster_results directory.

    Returns:
        predictions (np.ndarray): 1D array of predicted class indices/IDs
        targets (np.ndarray): 1D array of target class indices/IDs
        mode (str): 'classification' or 'regression'
    """
    # Classification outputs
    cls_pred_cols = results_dir / 'predicted_cluster_cols.npy'
    cls_tgt_cols = results_dir / 'target_cluster_cols.npy'

    # Regression outputs
    reg_pred_ids = results_dir / 'predicted_cluster_ids.npy'
    reg_tgt_ids = results_dir / 'target_cluster_ids.npy'
    reg_tgt_raw = results_dir / 'target_cluster_ids_raw.npy'

    if cls_pred_cols.exists() and cls_tgt_cols.exists():
        predictions = np.load(cls_pred_cols)
        targets = np.load(cls_tgt_cols)
        return predictions, targets, 'classification'

    if reg_pred_ids.exists():
        predictions = np.load(reg_pred_ids)
        # Prefer mapped target IDs if present; fall back to raw
        if reg_tgt_ids.exists():
            targets = np.load(reg_tgt_ids)
        elif reg_tgt_raw.exists():
            targets = np.load(reg_tgt_raw)
        else:
            raise FileNotFoundError('No regression target file found')
        return predictions, targets, 'regression'

    raise FileNotFoundError('No recognizable prediction/target files found in cluster_results')


def compute_coverage_at_k(predictions, targets, ks=(10, 50, 100, 500, 1000)):
    pred_counts = Counter(predictions)
    tgt_counts = Counter(targets)
    tgt_classes = set(tgt_counts.keys())
    pred_classes_by_freq = [cls for cls, _ in pred_counts.most_common()]

    cov = {}
    for k in ks:
        topk = set(pred_classes_by_freq[:min(k, len(pred_classes_by_freq))])
        covered = topk & tgt_classes
        cov[int(k)] = len(covered) / max(1, len(tgt_classes))
    return cov


def accuracy_vs_frequency_bins(predictions, targets, bins=10):
    """Compute accuracy vs. target-class frequency deciles.

    Returns a list of dicts with bin label, count and accuracy.
    """
    correct = (predictions == targets)
    tgt_counts = Counter(targets)
    freqs = np.array([tgt_counts[c] for c in targets])

    if len(freqs) == 0:
        return []

    quantiles = np.quantile(freqs, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            out.append({
                'bin': f'{int(lo)}-{int(hi)}',
                'count': int(mask.sum()),
                'accuracy': float(correct[mask].mean())
            })
        else:
            out.append({'bin': f'{int(lo)}-{int(hi)}', 'count': 0, 'accuracy': None})
    return out


def plot_histogram(targets, output_dir: Path, max_classes=2000, quantiles=(0.9, 0.99)):
    tgt_counts = Counter(targets)
    counts = np.array(sorted(tgt_counts.values(), reverse=True))
    # For visibility, limit to top classes if extremely long tail
    top_counts = counts[:max_classes]

    plt.figure(figsize=(8, 4))
    plt.hist(top_counts, bins=50, log=True)
    plt.xlabel('Class frequency (top classes)')
    plt.ylabel('Count (log scale)')
    plt.title('Target Class Frequency Histogram (log scale)')
    # Compute quantiles on the full original distribution and draw on log-scaled hist
    if quantiles and len(quantiles) > 0:
        qvals = []
        for q in quantiles:
            try:
                qv = float(np.quantile(counts, q))
                qvals.append((q, qv))
                plt.axvline(qv, color='red' if q >= 0.99 else 'orange', linestyle='--', alpha=0.8,
                            label=f"Q{int(q*100)} = {int(qv)}")
            except Exception:
                continue
        if qvals:
            plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'class_frequency_hist.png', dpi=200)
    plt.close()


def plot_cdf(targets, output_dir: Path, quantiles=(0.9, 0.99)):
    tgt_counts = Counter(targets)
    counts = np.array(sorted(tgt_counts.values(), reverse=True))
    cum = np.cumsum(counts)
    cdf = cum / cum[-1]
    idx = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, cdf)
    plt.xscale('log')
    plt.xlabel('Top N classes (log scale)')
    plt.ylabel('Cumulative fraction of residues')
    plt.title('CDF of Target Class Frequencies')
    # Mark coverage quantiles on the CDF (fraction of residues covered)
    if quantiles and len(quantiles) > 0:
        for q in quantiles:
            try:
                # smallest N such that cdf >= q
                n_idx = int(np.searchsorted(cdf, q)) + 1
                n_idx = min(max(1, n_idx), len(idx))
                plt.axhline(q, color='orange', linestyle='--', alpha=0.6)
                plt.axvline(n_idx, color='orange', linestyle='--', alpha=0.6,
                            label=f"Top-{n_idx} cover {int(q*100)}%")
            except Exception:
                continue
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'class_frequency_cdf.png', dpi=200)
    plt.close()


def plot_accuracy_vs_frequency_bins(acc_bins, output_dir: Path):
    labels = [b['bin'] for b in acc_bins]
    values = [b['accuracy'] if b['accuracy'] is not None else 0.0 for b in acc_bins]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.xlabel('Target class frequency bin (deciles)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Target Class Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_frequency_bins.png', dpi=200)
    plt.close()


def plot_coverage_curve(cov_dict, output_dir: Path):
    ks = sorted(cov_dict.keys())
    vals = [cov_dict[k] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker='o')
    plt.xscale('log')
    plt.xlabel('K (log scale)')
    plt.ylabel('Coverage@K')
    plt.title('Coverage vs K')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_at_k.png', dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='EMPROT class/distribution analysis')
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--results_dir', type=str,
                            help='Path to evaluation_results/.../(classification|regression)/cluster_results')
    mode_group.add_argument('--analyze_dataset', action='store_true', default=False,
                            help='Analyze test dataset distribution directly (no predictions needed)')

    # Common
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Directory to save plots (default: parent of results_dir or a derived dir)')
    ap.add_argument('--quantiles', nargs='*', type=float, default=[0.9, 0.99],
                    help='Quantiles to overlay (computed on original distribution, drawn on plots)')

    # Dataset mode args
    ap.add_argument('--data_dir', type=str, default=None,
                    help='Path to LMDB root (required for --analyze_dataset)')
    ap.add_argument('--metadata_path', type=str, default=None,
                    help='Path to metadata CSV (required for --analyze_dataset)')
    ap.add_argument('--seed', type=int, default=42, help='Seed to match trainer splits (default: 42)')
    ap.add_argument('--batch_size', type=int, default=32, help='Batch size for dataset analysis')
    ap.add_argument('--sequence_length', type=int, default=5, help='Sequence length for dataset analysis')
    ap.add_argument('--stride', type=int, default=10, help='Stride for dataset analysis (ns=0.2*stride)')
    ap.add_argument('--num_workers', type=int, default=2, help='Data loader workers for dataset analysis')

    args = ap.parse_args()

    if args.analyze_dataset:
        # Validate inputs
        assert args.data_dir and args.metadata_path, "--data_dir and --metadata_path are required for --analyze_dataset"
        # Create loaders with same seed/stride to match trainer splits
        _, _, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            batch_size=args.batch_size,
            max_sequence_length=args.sequence_length,
            min_sequence_length=2,
            stride=args.stride,
            train_split=0.8,
            val_split=0.1,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        # Collect target cluster IDs across test set
        collected_targets = []
        total_batches = 0
        for batch in test_loader:
            total_batches += 1
            # Handle nested targets
            tgt = None
            if isinstance(batch, dict):
                tdict = batch.get('targets', {}) if 'targets' in batch else {}
                if isinstance(tdict, dict) and 'target_cluster_ids' in tdict:
                    tgt = tdict['target_cluster_ids']
                elif 'target_cluster_ids' in batch:
                    tgt = batch['target_cluster_ids']
            if tgt is None:
                # Skip batch if missing cluster IDs
                continue
            tgt_np = tgt.numpy() if hasattr(tgt, 'numpy') else np.asarray(tgt)
            valid = tgt_np >= 0
            if valid.any():
                collected_targets.append(tgt_np[valid].ravel())
        if not collected_targets:
            raise RuntimeError("No target_cluster_ids found in test dataset. Ensure cluster IDs are present in LMDB.")
        targets = np.concatenate(collected_targets, axis=0)
        # Output dir default
        output_dir = Path(args.output_dir) if args.output_dir else Path('evaluation_results/dataset_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Basic stats
        stats = {
            'mode': 'dataset_only',
            'num_samples': int(len(targets)),
            'num_unique_targets': int(len(set(targets.tolist()))),
            'seed': args.seed,
            'stride': args.stride,
            'sequence_length': args.sequence_length,
            'batches_scanned': total_batches,
        }

        # Plots (histogram, CDF)
        qs = tuple(args.quantiles) if args.quantiles else ()
        plot_histogram(targets, output_dir, quantiles=qs)
        plot_cdf(targets, output_dir, quantiles=qs)

        # Save summary
        summary = {
            'stats': stats,
        }
        with open(output_dir / 'dataset_distribution.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Saved dataset analysis to: {output_dir}")
        return

    # Default: analyze evaluator outputs
    results_dir = Path(args.results_dir)
    assert results_dir.is_dir(), f"results_dir not found: {results_dir}"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, targets, mode = load_predictions_and_targets(results_dir)

    # Basic stats
    stats = {
        'mode': mode,
        'num_samples': int(len(targets)),
        'num_unique_targets': int(len(set(targets.tolist()))),
        'num_unique_predictions': int(len(set(predictions.tolist()))),
    }

    # Plots
    qs = tuple(args.quantiles) if args.quantiles else ()
    plot_histogram(targets, output_dir, quantiles=qs)
    plot_cdf(targets, output_dir, quantiles=qs)

    # Coverage@K and accuracy vs frequency
    cov = compute_coverage_at_k(predictions, targets)
    plot_coverage_curve(cov, output_dir)
    acc_bins = accuracy_vs_frequency_bins(predictions, targets)
    if acc_bins:
        plot_accuracy_vs_frequency_bins(acc_bins, output_dir)

    # Save analysis summary
    summary = {
        'stats': stats,
        'coverage_at_k': cov,
        'accuracy_vs_frequency_bins': acc_bins,
    }
    with open(output_dir / 'distribution_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved analysis to: {output_dir}")


if __name__ == '__main__':
    main()
