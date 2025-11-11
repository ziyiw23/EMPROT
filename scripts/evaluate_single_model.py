#!/usr/bin/env python3
"""
Unified Single Model Evaluation Script

Evaluates a single model (regression or classification) and produces ROC/PRC curves.
Handles the complexity of converting different model outputs to cluster predictions.
"""

import argparse
import torch
import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
from torch.serialization import add_safe_globals, safe_globals
try:
    from numpy.core.multiarray import _reconstruct as _np_reconstruct
    from numpy import ndarray as _np_ndarray
    add_safe_globals([_np_reconstruct, _np_ndarray])
except Exception:
    pass
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Add project root to path (idempotent)
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from emprot.models.transformer import ProteinTransformerClassificationOnly
from emprot.data.dataset import create_dataloaders
from emprot.data.cluster_lookup import ClusterCentroidLookup

# -------------------------
# Logging + Plot style
# -------------------------

log = logging.getLogger("emprot.eval_single")

def _maybe_set_plot_style() -> None:
    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(context="talk", style="whitegrid")
    except Exception:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            pass


def map_targets_to_eval_cols(targets_raw: torch.Tensor, id2col: dict, device: torch.device) -> torch.Tensor:
    """Map raw target IDs to column indices using id2col; returns -1 for unmappable.

    Args:
        targets_raw: tensor of raw target IDs (any shape)
        id2col: mapping raw_id -> column_index
        device: device for the returned tensor
    Returns:
        Tensor of same shape as targets_raw with column indices (or -1 for unmappable)
    """
    target_cols = torch.full_like(targets_raw, -1, device=device)
    flat_targets = targets_raw.view(-1)
    flat_target_cols = target_cols.view(-1)
    # Mask of mappable targets
    mappable_mask = torch.tensor([rid.item() in id2col for rid in flat_targets], device=device, dtype=torch.bool)
    if mappable_mask.any():
        mappable_rids = flat_targets[mappable_mask].cpu().numpy()
        mapped_cols = np.array([id2col[int(rid)] for rid in mappable_rids])
        flat_target_cols[mappable_mask] = torch.from_numpy(mapped_cols).to(device)
    return target_cols


def js_divergence(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)



# Attention visualization removed as per request


# Attention visualization removed


def extract_target_tensor(batch):
    """
    Returns (targets_tensor, path_str) or raises KeyError with helpful message.
    Prefers nested targets > top-level to avoid ambiguity.
    """
    # 1) Nested dict under 'targets'
    t = batch.get('targets', None)
    if isinstance(t, dict):
        for k in ('target_cluster_ids', 'cluster_ids', 'target_ids'):
            if k in t:
                return t[k], f"targets/{k}"

    # 2) Top-level fallbacks
    for k in ('target_cluster_ids', 'cluster_ids'):
        if k in batch and isinstance(batch[k], torch.Tensor):
            return batch[k], k

    # 3) Nothing found — produce a helpful error
    avail = list(batch.keys())
    nested = list(t.keys()) if isinstance(t, dict) else None
    msg = [f"No target tensor found. Top-level keys: {avail}"]
    if nested is not None:
        msg.append(f"Available target keys: {nested}")
    raise KeyError(" | ".join(msg))


def compute_metrics_with_curves(predicted_clusters, target_clusters, prediction_scores=None, eval_class_ids=None):
    """Compute comprehensive metrics with ROC/PRC curves - self-contained version."""
    log.info("Computing classification metrics…")
    
    # Filter out invalid targets
    valid_mask = target_clusters >= 0
    valid_predictions = predicted_clusters[valid_mask]
    valid_targets = target_clusters[valid_mask]
    
    log.info("Valid samples: %s / %s", f"{len(valid_targets):,}", f"{len(target_clusters):,}")
    
    if len(valid_targets) == 0:
        return {}, {'roc_curves': [], 'prc_curves': [], 'class_names': [], 'auroc_scores': [], 'auprc_scores': []}
    
    # Basic accuracy
    correct = (valid_predictions == valid_targets).sum()
    accuracy = correct / len(valid_targets)
    
    # Count unique classes
    unique_targets = np.unique(valid_targets)
    unique_predictions = np.unique(valid_predictions)
    
    log.info("Classes in targets=%s, predictions=%s", f"{len(unique_targets):,}", f"{len(unique_predictions):,}")
    
    # Real per-sample top-K accuracy if scores available
    if prediction_scores is not None and eval_class_ids is not None:
        class_id_to_col = {cid: i for i, cid in enumerate(eval_class_ids)}
        log.info("Computing per-sample top-K accuracy…")
        
        # Filter scores to match valid samples
        valid_scores = prediction_scores[valid_mask]
        
        # Filter samples where target is in evaluated classes
        valid_eval_mask = np.array([target in class_id_to_col for target in valid_targets])
        eval_targets = valid_targets[valid_eval_mask]
        eval_scores = valid_scores[valid_eval_mask]
        
        if len(eval_targets) > 0:
            # Map targets to local column indices  
            eval_targets_mapped = np.array([class_id_to_col[target] for target in eval_targets])
            
            # Real per-sample top-K accuracy - critical fix: proper guard for K > columns
            topk_values = [1, 5, 10]
            topk_accs = {}
            K = eval_scores.shape[1]  # Number of evaluated classes
            
            # Compute NLL/perplexity (macro over evaluated samples)
            try:
                logits = eval_scores.astype(np.float64)
                n = logits.shape[0]
                row_max = logits.max(axis=1, keepdims=True)
                logsumexp = row_max + np.log(np.exp(logits - row_max).sum(axis=1, keepdims=True))
                target_logits = logits[np.arange(n), eval_targets_mapped]
                nll_vec = -(target_logits - logsumexp.squeeze(1))
                ce_loss = float(nll_vec.mean())
                perplexity = float(np.exp(ce_loss))
            except Exception:
                ce_loss = float('nan')
                perplexity = float('nan')

            for k in topk_values:
                if k <= K:
                    # Get top-k predictions per sample
                    topk_idx = np.argpartition(-eval_scores, k-1, axis=1)[:, :k]
                    # Check if target is in top-k
                    hit = (topk_idx == eval_targets_mapped[:, None]).any(axis=1)
                    topk_accs[f'top_{k}'] = float(hit.mean())
                    log.info("Top-%d accuracy: %.4f", k, topk_accs[f'top_{k}'])
                else:
                    log.info("Top-%d accuracy: N/A (k > evaluated_classes=%d)", k, K)
            
            log.info("Evaluated %s/%s samples in top-%d classes", f"{len(eval_targets):,}", f"{len(valid_targets):,}", len(eval_class_ids))
            # Set metrics only if computed (avoid overly optimistic defaults)
            top5_accuracy = float(topk_accs['top_5']) if 'top_5' in topk_accs else np.nan
            top10_accuracy = float(topk_accs['top_10']) if 'top_10' in topk_accs else np.nan
            topk_mode = 'per_sample'
        else:
            log.warning("No samples found in evaluated classes for top-K")
            top5_accuracy = 0.0
            top10_accuracy = 0.0
            topk_mode = 'per_sample'
            ce_loss = float('nan')
            perplexity = float('nan')
    else:
        # Fallback: Global top-k
        log.info("Using global top-K accuracy (no per-sample scores available) [approx]…")
        prediction_counts = Counter(valid_predictions)
        most_common_preds = [cls for cls, _ in prediction_counts.most_common()]
        
        top5_classes = set(most_common_preds[:min(5, len(most_common_preds))])
        top5_correct = sum(1 for target in valid_targets if target in top5_classes)
        top5_accuracy = top5_correct / len(valid_targets)
        
        top10_classes = set(most_common_preds[:min(10, len(most_common_preds))])
        top10_correct = sum(1 for target in valid_targets if target in top10_classes)
        top10_accuracy = top10_correct / len(valid_targets)
        topk_mode = 'global_approx'
    
    # Analyze prediction distribution
    target_counts = Counter(valid_targets)
    pred_counts = Counter(valid_predictions)
    
    # Coverage: how many target classes does the model predict?
    covered_classes = len(set(valid_targets) & set(valid_predictions))
    coverage_ratio = covered_classes / len(unique_targets)
    
    # Concentration: how concentrated are the predictions?
    total_preds = len(valid_predictions)
    prediction_counts = Counter(valid_predictions)
    most_common_preds = [cls for cls, _ in prediction_counts.most_common()]
    top_10_pred_ratio = sum(pred_counts[cls] for cls in most_common_preds[:10]) / total_preds
    # Entropy of prediction distribution
    pred_probs = np.array([cnt / total_preds for _, cnt in prediction_counts.most_common()])
    pred_entropy = float(-(pred_probs * np.log(pred_probs + 1e-12)).sum())

    js_pi = np.nan
    if len(valid_predictions) > 0 and len(valid_targets) > 0:
        max_cls = int(max(valid_predictions.max(), valid_targets.max())) + 1
        pred_counts_vec = np.bincount(valid_predictions, minlength=max_cls).astype(np.float64)
        tgt_counts_vec = np.bincount(valid_targets, minlength=max_cls).astype(np.float64)
        if pred_counts_vec.sum() > 0 and tgt_counts_vec.sum() > 0:
            pi_pred = pred_counts_vec / pred_counts_vec.sum()
            pi_tgt = tgt_counts_vec / tgt_counts_vec.sum()
            js_pi = js_divergence(pi_pred, pi_tgt)

    # Compute AUROC and AUPRC
    log.info("Computing AUROC/AUPRC…")
    auroc_macro, auprc_macro, used_classes, used_samples = compute_auroc_auprc(
        valid_targets, valid_predictions, prediction_scores[valid_mask] if prediction_scores is not None else None,
        eval_class_ids, max_classes=100, max_samples=50000
    )
    
    # Compute ROC and PRC curves for visualization
    log.info("Computing ROC/PRC curves for top classes…")
    curves_data = compute_roc_prc_curves(
        valid_targets, valid_predictions, 
        prediction_scores[valid_mask] if prediction_scores is not None else None, 
        eval_class_ids, max_classes=5, max_samples=10000
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'top_5_accuracy': float(top5_accuracy),
        'top_10_accuracy': float(top10_accuracy),
        'topk_mode': topk_mode,
        'auroc_macro': float(auroc_macro),
        'auprc_macro': float(auprc_macro),
        'cross_entropy': float(ce_loss) if 'ce_loss' in locals() else float('nan'),
        'perplexity': float(perplexity) if 'perplexity' in locals() else float('nan'),
        'macro_auroc_max_classes': 100,
        'macro_auroc_max_samples': 50000,
        'macro_auroc_used_classes': int(used_classes),
        'macro_auroc_used_samples': int(used_samples),
        'num_classes_in_targets': int(len(unique_targets)),
        'num_classes_predicted': int(len(unique_predictions)),
        'total_samples': int(len(valid_targets)),
        'total_raw_samples': int(len(target_clusters)),
        'valid_sample_ratio': float(len(valid_targets) / len(target_clusters)),
        'coverage_ratio': float(coverage_ratio),
        'top_10_prediction_concentration': float(top_10_pred_ratio),
        'prediction_entropy': pred_entropy,
        'correct_predictions': int(correct),
        'js_pi': float(js_pi) if not np.isnan(js_pi) else float('nan'),
    }
    
    log.info("Metrics:")
    log.info("  Accuracy: %.4f (%s/%s)", accuracy, f"{correct:,}", f"{len(valid_targets):,}")
    log.info("  Top-5 Accuracy: %s", f"{top5_accuracy:.4f}" if not np.isnan(top5_accuracy) else "N/A")
    log.info("  Top-10 Accuracy: %s", f"{top10_accuracy:.4f}" if not np.isnan(top10_accuracy) else "N/A")
    log.info("  AUROC (macro): %.4f", auroc_macro)
    log.info("  AUPRC (macro): %.4f", auprc_macro)
    log.info("  Coverage: %.4f (%s/%s classes)", coverage_ratio, f"{covered_classes:,}", f"{len(unique_targets):,}")
    log.info("  Prediction concentration (top 10): %.4f", top_10_pred_ratio)
    if not np.isnan(js_pi):
        log.info("  JS(pi_pred || pi_target): %.4f", js_pi)

    return metrics, curves_data


def compute_auroc_auprc(valid_targets, valid_predictions, prediction_scores=None, eval_class_ids=None, max_classes=100, max_samples=50000):
    """Compute AUROC and AUPRC with proper score handling."""
    try:
        # Set seed for reproducibility  
        np.random.seed(42)

        # Sample classes by frequency - critical fix: force intersection with eval_class_ids
        class_id_to_col = None
        if eval_class_ids is not None:
            eval_set = set(eval_class_ids.tolist())
            class_counts = Counter(valid_targets)
            candidate_classes = [cls for cls, _ in class_counts.most_common(max_classes * 2)]  # Get more candidates
            sample_classes = np.array([c for c in candidate_classes if c in eval_set])[:min(max_classes, len(eval_set))]
            class_id_to_col = {cid: i for i, cid in enumerate(eval_class_ids)}
        else:
            class_counts = Counter(valid_targets)
            sample_classes = np.array([cls for cls, _ in class_counts.most_common(max_classes)])

        used_classes = int(len(sample_classes))
        if used_classes == 0:
            log.warning("No overlapping classes available for AUROC/AUPRC computation")
            return 0.0, 0.0, 0, 0

        # Sample data points if too many
        if len(valid_targets) > max_samples:
            indices = np.random.choice(len(valid_targets), max_samples, replace=False)
            sample_targets = valid_targets[indices]
            sample_predictions = valid_predictions[indices]
            sample_scores = prediction_scores[indices] if prediction_scores is not None else None
        else:
            sample_targets = valid_targets
            sample_predictions = valid_predictions
            sample_scores = prediction_scores

        used_samples = int(len(sample_targets))

        # Create binary matrices
        y_true_binary = np.zeros((used_samples, used_classes))
        y_proba_sampled = np.zeros((used_samples, used_classes))

        for i, cls in enumerate(sample_classes):
            y_true_binary[:, i] = (sample_targets == cls).astype(int)

            if sample_scores is not None and class_id_to_col is not None and cls in class_id_to_col:
                # Always use mapping - guaranteed to work now
                y_proba_sampled[:, i] = sample_scores[:, class_id_to_col[cls]]
            elif sample_scores is not None:
                # Fallback for direct indexing (shouldn't happen with eval_class_ids)
                y_proba_sampled[:, i] = sample_scores[:, cls] if cls < sample_scores.shape[1] else 0
            else:
                # Fallback: Use one-hot encoding
                y_proba_sampled[:, i] = (sample_predictions == cls).astype(float)

        # Compute macro-averaged AUROC and AUPRC
        auroc_scores = []
        auprc_scores = []

        for i in range(used_classes):
            if y_true_binary[:, i].sum() > 0:  # Only if positive examples exist
                try:
                    auroc = roc_auc_score(y_true_binary[:, i], y_proba_sampled[:, i])
                    auprc = average_precision_score(y_true_binary[:, i], y_proba_sampled[:, i])
                    auroc_scores.append(auroc)
                    auprc_scores.append(auprc)
                except Exception:
                    continue

        if auroc_scores:
            auroc_macro = float(np.mean(auroc_scores))
            auprc_macro = float(np.mean(auprc_scores))
        else:
            auroc_macro = 0.0
            auprc_macro = 0.0

        log.info("Using %d evaluated classes for AUROC/AUPRC", used_classes)
        return auroc_macro, auprc_macro, used_classes, used_samples

    except Exception as e:
        log.warning("Error computing AUROC/AUPRC: %s", e)
        return 0.0, 0.0, 0, 0


def compute_roc_prc_curves(valid_targets, valid_predictions, prediction_scores=None, eval_class_ids=None, max_classes=5, max_samples=10000):
    """Compute ROC and PRC curves for top classes."""
    try:
        # Sample data if too large
        if len(valid_targets) > max_samples:
            indices = np.random.choice(len(valid_targets), max_samples, replace=False)
            sample_targets = valid_targets[indices]
            sample_predictions = valid_predictions[indices]
            sample_scores = prediction_scores[indices] if prediction_scores is not None else None
        else:
            sample_targets = valid_targets
            sample_predictions = valid_predictions
            sample_scores = prediction_scores
        
        # Get top classes by frequency in targets
        target_counts = Counter(sample_targets)
        top_classes = [cls for cls, _ in target_counts.most_common(max_classes)]
        
        # Critical fix: restrict to intersection with eval_class_ids to prevent indexing bugs
        if sample_scores is not None and eval_class_ids is not None:
            allowed = set(eval_class_ids.tolist())
            top_classes = [c for c in top_classes if c in allowed]
            class_id_to_col = {cid: i for i, cid in enumerate(eval_class_ids)}
            print(f"         Restricted to {len(top_classes)} classes in evaluated set")
        
        curves_data = {
            'roc_curves': [],
            'prc_curves': [], 
            'class_names': [],
            'auroc_scores': [],
            'auprc_scores': []
        }
        
        for cls in top_classes:
            # Create binary classification problem
            y_true = (sample_targets == cls).astype(int)
            
            # Use actual scores if available, otherwise fall back to one-hot
            if sample_scores is not None and eval_class_ids is not None:
                # Always use mapping - guaranteed to be in the mapping now
                y_score = sample_scores[:, class_id_to_col[cls]]
            elif sample_scores is not None:
                # Fallback for direct indexing (shouldn't happen with eval_class_ids)
                y_score = sample_scores[:, cls] if cls < sample_scores.shape[1] else (sample_predictions == cls).astype(float)
            else:
                y_score = (sample_predictions == cls).astype(float)  # One-hot fallback
            
            # Skip if no positive examples
            if y_true.sum() == 0:
                continue
                
            try:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auroc = roc_auc_score(y_true, y_score)
                
                # PRC Curve  
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                auprc = average_precision_score(y_true, y_score)
                
                curves_data['roc_curves'].append({'fpr': fpr, 'tpr': tpr})
                curves_data['prc_curves'].append({'precision': precision, 'recall': recall})
                curves_data['class_names'].append(f'Class {cls}')
                curves_data['auroc_scores'].append(auroc)
                curves_data['auprc_scores'].append(auprc)
                
            except Exception as e:
                log.info("Skipping class %s: %s", cls, e)
                continue
        
        return curves_data
        
    except Exception as e:
        log.warning("Error computing curves: %s", e)
        return {'roc_curves': [], 'prc_curves': [], 'class_names': [], 'auroc_scores': [], 'auprc_scores': []}


def create_visualization(output_dir, model_name, model_type, metrics, curves_data):
    """Create ROC/PRC curve visualization with a clean style."""
    log.info("Creating ROC/PRC visualizations…")

    _maybe_set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_type.title()} Model Performance: {model_name}', fontsize=16, fontweight='bold')

    # Common color cycle
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

    # ROC (top-left)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')
    if curves_data['roc_curves']:
        for i, (roc_data, class_name, auroc) in enumerate(zip(
            curves_data['roc_curves'], curves_data['class_names'], curves_data['auroc_scores']
        )):
            axes[0, 0].plot(roc_data['fpr'], roc_data['tpr'], color=colors[i % len(colors)], linewidth=2,
                            label=f'{class_name} (AUC = {auroc:.3f})')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves')
    axes[0, 0].legend(loc='lower right', fontsize=8, frameon=True)
    axes[0, 0].grid(alpha=0.3)

    # PRC (top-right)
    if curves_data['prc_curves']:
        for i, (prc_data, class_name, auprc) in enumerate(zip(
            curves_data['prc_curves'], curves_data['class_names'], curves_data['auprc_scores']
        )):
            axes[0, 1].plot(prc_data['recall'], prc_data['precision'], color=colors[i % len(colors)], linewidth=2,
                            label=f'{class_name} (AUC = {auprc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curves')
    axes[0, 1].legend(loc='lower left', fontsize=8, frameon=True)
    axes[0, 1].grid(alpha=0.3)

    # Behavior (bottom-left)
    behavior_values = [metrics.get('coverage_ratio', 0), metrics.get('top_10_prediction_concentration', 0)]
    behavior_labels = ['Class\nCoverage', 'Top-10\nConcentration']
    bars = axes[1, 0].bar(behavior_labels, behavior_values, color=['#1f77b4', '#ff7f0e'])
    axes[1, 0].set_title('Model Behavior Metrics')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_ylim(0, 1)
    for bar, v in zip(bars, behavior_values):
        axes[1, 0].annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v), xytext=(0, 5), textcoords='offset points', ha='center')

    # Summary text (bottom-right)
    axes[1, 1].text(0.1, 0.9, f"Model Type: {model_type.title()}", fontsize=12, transform=axes[1, 1].transAxes, weight='bold')
    axes[1, 1].text(0.1, 0.8, f"Total Samples: {metrics.get('total_samples', 0):,}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Accuracy: {metrics.get('accuracy', 0):.4f}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"AUROC (Macro): {metrics.get('auroc_macro', 0):.4f}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"AUPRC (Macro): {metrics.get('auprc_macro', 0):.4f}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f"Class Coverage: {metrics.get('coverage_ratio', 0):.3f}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f"Target Classes: {metrics.get('num_classes_in_targets', 0):,}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, f"Predicted Classes: {metrics.get('num_classes_predicted', 0):,}", fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Model Performance Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()
    out_path = Path(output_dir) / 'performance_summary_final.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info("Visualization saved: %s", out_path)


def evaluate_single_model(model_path: str, model_name: str, model_type: str, 
                         data_dir: str, metadata_path: str, output_dir: str, 
                         cluster_model_path: str = '/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl',
                         batch_size: int = 8, device: str = 'cuda', no_plots: bool = False,
                         eval_topN: int = 2000, use_faiss: bool = False,
                         topk: int = 10, logit_adjust_tau: float = 0.0,
                         temperature: float = 1.0,
                         rerank_centroid: bool = False,
                         rerank_topk: int = 10,
                         rerank_alpha: float = 0.8,
                         rerank_beta: float = 20.0):
    """
    Evaluate a single model and produce ROC/PRC curves.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name for the model (for plots)
        model_type: 'regression' or 'classification'
        data_dir: Path to test data
        metadata_path: Path to metadata file
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to run on
    """
    
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    log.info("Evaluating %s model '%s' → %s", model_type, model_name, output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Save inference options for reproducibility
    try:
        inf_opts = {
            'rerank_centroid': bool(rerank_centroid),
            'rerank_topk': int(rerank_topk),
            'rerank_alpha': float(rerank_alpha),
            'rerank_beta': float(rerank_beta),
            'logit_adjust_tau': float(logit_adjust_tau),
        }
        with open(Path(output_dir) / 'inference_options.json', 'w') as f:
            json.dump(inf_opts, f, indent=2)
    except Exception:
        pass
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    log.info("Using device: %s", device)
    
    # Load test data (using standard dataset for both types)
    log.info("Loading test data…")
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        max_sequence_length=5,
        stride=10,
        num_workers=2
    )
    log.info("Test batches: %d", len(test_loader))
    
    # Load model
    log.info("Loading %s model…", model_type)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})
    
    # Extract model configuration from checkpoint
    cfg_model = model_config.get('model', {}) if isinstance(model_config.get('model', {}), dict) else {}
    
    d_embed = int(model_config.get('d_embed', cfg_model.get('d_embed', 512)))
    num_heads = int(model_config.get('num_heads', cfg_model.get('num_heads', 8)))
    dropout = float(model_config.get('dropout', cfg_model.get('dropout', 0.1)))
    
    # Extract latent summary configuration to avoid shape mismatches
    latent_cfg = {
        'enabled': bool(model_config.get('latent_summary_enabled', cfg_model.get('latent_summary_enabled', False))),
        'num_latents': int(model_config.get('latent_summary_num_latents', cfg_model.get('latent_summary_num_latents', 0)) or 0),
        'layers': int(model_config.get('latent_summary_layers', cfg_model.get('latent_summary_layers', 1)) or 1),
        'd_model': int(model_config.get('latent_summary_d_model', cfg_model.get('latent_summary_d_model', d_embed)) or d_embed),
        'heads': int(model_config.get('latent_summary_heads', cfg_model.get('latent_summary_heads', num_heads)) or num_heads),
        'dropout': float(model_config.get('latent_summary_dropout', cfg_model.get('latent_summary_dropout', dropout))),
        'context_mode': model_config.get('context_mode', cfg_model.get('context_mode', 'latent_cap')),
        'train_context': model_config.get('train_context', cfg_model.get('train_context', {})),
        'summarizer': model_config.get('summarizer', cfg_model.get('summarizer', {})),
        'hier_pool': model_config.get('hier_pool', cfg_model.get('hier_pool', {})),
        'memory': model_config.get('memory', cfg_model.get('memory', {})),
    }
    
    # If checkpoint stores a specific latent count, honor it to avoid shape mismatches
    state_dict = checkpoint['model_state_dict']
    latent_param_keys = [
        'backbone.context_builder.summarizer.latents',
        'module.backbone.context_builder.summarizer.latents',
        'backbone.latent_summarizer.latents',
        'module.backbone.latent_summarizer.latents',
    ]
    ck_num_latents = None
    for k in latent_param_keys:
        if k in state_dict and hasattr(state_dict[k], 'shape') and len(state_dict[k].shape) == 3:
            ck_num_latents = int(state_dict[k].shape[1])
            log.info("Found %d latent tokens in checkpoint at %s", ck_num_latents, k)
            break
    if ck_num_latents is not None and ck_num_latents > 0:
        latent_cfg['num_latents'] = ck_num_latents
        if isinstance(latent_cfg.get('summarizer'), dict):
            latent_cfg['summarizer'] = dict(latent_cfg['summarizer'])
            latent_cfg['summarizer']['num_latents'] = ck_num_latents
    
    # Extract other backbone configuration
    hybrid_context = bool(model_config.get('hybrid_context', cfg_model.get('hybrid_context', False)))
    recent_full_frames = int(model_config.get('recent_full_frames', cfg_model.get('recent_full_frames', 5)))
    decoder_layout = model_config.get('decoder_layout', cfg_model.get('decoder_layout', None))
    stochastic_depth = float(model_config.get('stochastic_depth', cfg_model.get('stochastic_depth', 0.0)))
    
    backbone_kwargs = dict(
        hybrid_context=hybrid_context,
        recent_full_frames=recent_full_frames,
        decoder_layout=decoder_layout,
        stochastic_depth=stochastic_depth,
        latent_summary_config=latent_cfg,
    )

    # Create model with proper configuration
    model = ProteinTransformerClassificationOnly(
        d_embed=d_embed,
        num_heads=num_heads,
        dropout=dropout,
        use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', True),
        min_context_frames=model_config.get('min_context_frames', 2),
        num_layers=int(model_config.get('num_layers', 1)),
        attention_type=str(model_config.get('attention_type', 'cross_temporal')),
        classifier_type=str(model_config.get('classifier_type', 'linear')),
        classifier_scale=float(model_config.get('classifier_scale', 30.0)),
        num_clusters=50000,
        **backbone_kwargs
    ).to(device)
    
    # Load model state dict (already loaded above)
    
    # If checkpoint latent tokens do not match instantiation, resize before loading to avoid mismatch
    target_latent_shape = None
    for k in latent_param_keys:
        if k in state_dict and hasattr(state_dict[k], 'shape'):
            target_latent_shape = tuple(state_dict[k].shape)
            break
    if target_latent_shape:
        cb = getattr(getattr(model, 'backbone', None), 'context_builder', None)
        summarizer = getattr(cb, 'summarizer', None) if cb is not None else None
        if summarizer is not None and hasattr(summarizer, 'latents'):
            latents_param = summarizer.latents
            if tuple(latents_param.shape) != target_latent_shape:
                with torch.no_grad():
                    new_latents = torch.nn.Parameter(torch.zeros(*target_latent_shape, dtype=latents_param.dtype, device=latents_param.device))
                    summarizer.latents = new_latents

    # Filter out keys that don't exist in current model (like the reference does)
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    
    # Load with strict=False to ignore missing/unexpected keys (like the reference)
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    
    if unexpected_keys:
        log.warning("Ignored unexpected keys: %s", unexpected_keys)
    if missing_keys:
        log.warning("Missing keys (random init): %s", missing_keys)
    
    # Initialize dynamic components if they exist in original checkpoint
    # Heads exist by construction in ProteinTransformerDualHead
        
    if any('cluster_embedding' in key for key in state_dict.keys()):
        if not hasattr(model, 'cluster_embedding'):
            model.cluster_embedding = torch.nn.Embedding(
                num_embeddings=50001,
                embedding_dim=model.d_embed,
                padding_idx=0
            ).to(device)
        log.info("Initialized cluster embedding (from checkpoint)")
    model.eval()
    log.info("Loaded model from epoch %s", checkpoint.get('epoch', 'unknown'))
    
    # BULLETPROOF label mapping loading for classification models
    label_map = None
    id2col = None
    col2id = None
    
    if model_type == 'classification':
        log.info("Loading label mapping for classification model…")
        
        # Check multiple possible keys for label mapping (in priority order)
        mapping_found = False
        for key in ['id2col', 'label_map', 'cluster_mapping', 'class_to_idx']:
            if key in checkpoint:
                label_map = checkpoint[key]
                log.info("Found mapping in checkpoint['%s']", key)
                if isinstance(label_map, dict) and len(label_map) > 0:
                    mapping_found = True
                    break
                else:
                    log.warning("Invalid mapping in '%s': %s", key, type(label_map))
        
        # Check for inverse mapping if direct mapping not found
        if not mapping_found and 'col2id' in checkpoint:
            col2id_checkpoint = checkpoint['col2id']
            if isinstance(col2id_checkpoint, dict) and len(col2id_checkpoint) > 0:
                log.info("Found inverse mapping in checkpoint['col2id']")
                # Create id2col from col2id
                label_map = {v: k for k, v in col2id_checkpoint.items()}
                mapping_found = True
        
        if mapping_found and label_map is not None:
            # Build id2col (raw_id -> column_index) 
            id2col = {int(k): int(v) for k, v in label_map.items()}
            
            # Build col2id (column_index -> raw_id) efficiently
            max_col = max(id2col.values())
            col2id = {}
            for rid, col in id2col.items():
                col2id[col] = rid
            
            log.info("Loaded label mapping: %d classes", len(id2col))
            
            # Validate mapping integrity
            if max_col >= len(id2col):
                log.warning("Non-contiguous columns: max_col=%s, num_classes=%s", max_col, len(id2col))
                
        else:
            log.warning("No valid label mapping found in checkpoint; results may be unreliable.")
            
            # Check model size to confirm this is the issue
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                for key in state_dict.keys():
                    if 'classification_head' in key and 'weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) >= 2:
                            log.info("Found %s: %s → suggests %s classes", key, weight_shape, weight_shape[-1])
            
            log.info("Tip: save 'id2col'/'col2id' in checkpoint during training or reconstruct mapping from dataset.")
    
    # For regression models, load cluster lookup
    cluster_lookup = None
    if model_type == 'regression':
        log.info("Loading cluster centroids for regression model…")
        import pickle
        with open(cluster_model_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        
        cluster_lookup = ClusterCentroidLookup(
            num_clusters=kmeans_model.n_clusters,
            embedding_dim=kmeans_model.n_features_in_,
            device=device
        )
        cluster_lookup.load_centroids_from_sklearn(cluster_model_path)
        log.info("Loaded cluster model: %s clusters", f"{kmeans_model.n_clusters:,}")
    
    # Pre-scan to find top classes by frequency for memory optimization
    log.info("Pre-scanning to find most frequent classes…")
    class_counts = {}
    sample_count = 0
    max_classes_to_eval = 100  # Only evaluate top 100 classes
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 50:  # Sample first 50 batches to get class distribution
                break
                
            # Get targets using bulletproof extraction
            try:
                targets, _ = extract_target_tensor(batch)
            except KeyError:
                continue
                
            # Count classes
            unique_classes, counts = torch.unique(targets[targets >= 0], return_counts=True)
            for cls, count in zip(unique_classes.cpu().numpy(), counts.cpu().numpy()):
                class_counts[cls] = class_counts.get(cls, 0) + count
            sample_count += targets.numel()
    
    # Seed eval set by frequency; will expand later
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:max_classes_to_eval]
    seed_eval_class_ids = np.array([cls for cls, _ in top_classes])
    log.info("Found %d unique classes in %s samples", len(class_counts), f"{sample_count:,}")
    log.info("Evaluating top %d classes by frequency", len(seed_eval_class_ids))
    log.info("Top 5 classes: %s (counts: %s)", seed_eval_class_ids[:5], [class_counts[c] for c in seed_eval_class_ids[:5]])
    
    # For classification: prepare eval_cols tensor for efficient logits slicing
    eval_cols_tensor = None
    if model_type == 'classification':
        if id2col is not None:
            # Map seed set to column indices for initial slicing
            eval_cols = [id2col[c] for c in seed_eval_class_ids if c in id2col]
            eval_cols_tensor = torch.tensor(eval_cols, dtype=torch.long)
            log.info("Prepared eval columns tensor: %d/%d mappable", len(eval_cols), len(seed_eval_class_ids))
            log.info("Will slice logits to (..., %d) for efficiency", len(eval_cols))
            
            seed_mappable = [c for c in seed_eval_class_ids if c in id2col]
            log.info("Filtered eval_class_ids: %d/%d classes", len(seed_mappable), len(seed_eval_class_ids))
        else:
            # Identity mapping: slice logits by actual raw class IDs
            eval_cols_tensor = torch.tensor(seed_eval_class_ids, dtype=torch.long)
            log.warning("No mapping - using raw class IDs for slicing: %d classes (assumes IDs are valid column indices)", len(seed_eval_class_ids))
    
    # Sanity check: compute CE on one batch to catch mapping/loader issues
    log.info("Running sanity check…")
    with torch.no_grad():
        try:
            sample_batch = next(iter(test_loader))
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    sample_batch[key] = value.to(device)
            
            if model_type == 'classification':
                sample_outputs = model(
                    input_cluster_ids=sample_batch.get('input_cluster_ids'),
                    times=sample_batch['times'],
                    sequence_lengths=sample_batch['sequence_lengths'],
                    history_mask=sample_batch['history_mask']
                )
                if isinstance(sample_outputs, dict) and 'cluster_logits' in sample_outputs:
                    logits = sample_outputs['cluster_logits']
                    # Optional inference-time logit adjustment based on training priors
                    if logit_adjust_tau and 'train_class_counts' in checkpoint:
                        try:
                            counts_np = checkpoint['train_class_counts']
                            counts = torch.as_tensor(counts_np, dtype=torch.float32, device=logits.device)
                            priors = (counts / counts.sum()).clamp_min(1e-12)
                            logits = logits - float(logit_adjust_tau) * priors.log()
                        except Exception:
                            pass
                    # Get targets using bulletproof extraction
                    try:
                        targets_raw, target_path = extract_target_tensor(sample_batch)
                        log.info("   Found targets at: %s", target_path)
                    except KeyError as e:
                        log.warning("   Target extraction failed: %s", e)
                        log.warning("   Skipping sanity check due to target extraction failure")
                        targets_raw = None
                    
                    if targets_raw is not None:
                        log.info("   Target field shape: %s", targets_raw.shape)
                        log.info("   Target dtype: %s", targets_raw.dtype)
                        
                        # === THREE BULLETPROOF SANITY CHECKS ===
                        
                        # 1. PADDING/MASKING CHECK
                        targets_raw = targets_raw.to(device)  # Ensure targets on correct device
                        m = sample_batch.get('residue_mask', torch.ones_like(targets_raw, dtype=torch.bool)).to(device)
                        valid = (targets_raw >= 0) & m
                        log.info("   Valid positions: %d/%d (%.1f%%)", valid.sum().item(), valid.numel(), 100*valid.float().mean().item())
                        
                        if valid.sum() > 0:
                            vals, cnts = torch.unique(targets_raw[valid], return_counts=True)
                            zero_frac = (targets_raw[valid] == 0).float().mean().item()
                            log.info("   Unique targets in valid positions: %d", vals.numel())
                        else:
                            zero_frac = 0.0
                            log.info("   Unique targets in valid positions: 0")
                        log.info("   Fraction zeros: %.4f", zero_frac)
                        if zero_frac > 0.5:
                            log.warning("   >50%% targets are 0 - padding issue suspected")
                        elif zero_frac > 0.1:
                            log.warning("   %.1f%% targets are 0", zero_frac*100)
                        else:
                            log.info("   Low zero fraction (%.1f%%)", zero_frac*100)
                            
                        if vals.numel() > 0:
                            log.info("   Target range: %d - %d", vals.min().item(), vals.max().item())
                            top_targets = vals[:5].tolist()
                            top_counts = cnts[:5].tolist()
                            log.info("   Top targets: %s", list(zip(top_targets, top_counts)))
                    else:
                        print(f"   ❌ FAIL: No valid target positions found!")
                        
                    # 2. MAPPING COVERAGE CHECK
                    print(f"   📍 CHECK 2 - Mapping coverage:")
                    if id2col is not None and valid.sum() > 0:
                        unique_targets_in_batch = targets_raw[valid].unique()
                        mappable_targets = torch.tensor([tid.item() in id2col for tid in unique_targets_in_batch])
                        coverage = mappable_targets.float().mean().item()
                        print(f"   Mappable targets: {mappable_targets.sum().item()}/{len(unique_targets_in_batch)} ({coverage*100:.1f}%)")
                        
                        if coverage >= 0.95:
                            print(f"   ✅ PASS: Excellent mapping coverage ({coverage*100:.1f}%)")
                        elif coverage >= 0.8:
                            print(f"   ⚠️  WARNING: Some unmappable targets ({coverage*100:.1f}%)")
                        else:
                            print(f"   ❌ FAIL: Poor mapping coverage ({coverage*100:.1f}%) - many OOV targets!")
                            
                        if coverage < 1.0:
                            unmappable = [tid.item() for tid in unique_targets_in_batch if tid.item() not in id2col]
                            print(f"   Unmappable target IDs: {unmappable[:10]}{'...' if len(unmappable) > 10 else ''}")
                    else:
                        if id2col is None:
                            print(f"   ⚠️  SKIP: No mapping available")
                        else:
                            print(f"   ❌ FAIL: No valid targets to check")
                    
                    # BULLETPROOF TARGET MAPPING
                    print(f"   📍 Applying bulletproof target mapping...")
                    if id2col is not None:
                        # Use the bulletproof mapping approach you outlined
                        target_cols = torch.full_like(targets_raw, -1)
                        flat_targets = targets_raw.view(-1)
                        flat_target_cols = target_cols.view(-1)
                        
                        # Create mask for mappable targets
                        mappable_mask = torch.tensor([rid.item() in id2col for rid in flat_targets], 
                                                   device=targets_raw.device, dtype=torch.bool)
                        
                        # Map only the mappable targets
                        if mappable_mask.sum() > 0:
                            mappable_rids = flat_targets[mappable_mask].cpu().numpy()
                            mapped_cols = np.array([id2col[int(rid)] for rid in mappable_rids])
                            flat_target_cols[mappable_mask] = torch.from_numpy(mapped_cols).to(targets_raw.device)
                        
                        targets_for_ce = target_cols
                        valid_for_ce = (targets_for_ce >= 0) & valid
                        print(f"   Mapped targets: {mappable_mask.sum().item()}/{len(flat_targets)} total positions")
                        print(f"   Valid mapped targets: {valid_for_ce.sum().item()}/{valid.sum().item()} valid positions")
                    else:
                        targets_for_ce = targets_raw
                        valid_for_ce = valid
                        print(f"   Using raw targets (no mapping available)")
                    
                    # 3. CROSS-ENTROPY VALIDATION CHECK
                    print(f"   📍 CHECK 3 - Cross-entropy validation:")
                    if valid_for_ce.sum() > 0:
                        # Get valid logits and targets for CE computation
                        valid_logits = logits.view(-1, logits.size(-1))[valid_for_ce.view(-1)]
                        valid_targets_ce = targets_for_ce.view(-1)[valid_for_ce.view(-1)].long()
                        
                        ce_loss = torch.nn.functional.cross_entropy(valid_logits, valid_targets_ce, reduction='mean')
                        expected_random_ce = torch.log(torch.tensor(float(logits.size(-1)))).item()
                        
                        print(f"   CE loss: {ce_loss.item():.4f}")
                        print(f"   Expected random CE: {expected_random_ce:.4f}")
                        print(f"   CE ratio (actual/random): {ce_loss.item()/expected_random_ce:.3f}")
                        
                        if ce_loss.item() > expected_random_ce * 0.95:  # Within 5% of random
                            print(f"   ❌ FAIL: CE ≈ random ({ce_loss.item():.3f} vs {expected_random_ce:.3f}) - suggests severe misalignment!")
                        elif ce_loss.item() > expected_random_ce * 0.7:  # Within 30% of random  
                            print(f"   ⚠️  WARNING: CE higher than expected - possible alignment issues")
                        else:
                            print(f"   ✅ PASS: CE well below random - alignment looks correct")
                    else:
                        log.warning("   No valid targets for CE computation!")
                    
                    # Diagnostic: Check target range vs logits shape
                    if id2col is not None:
                        # Show both raw and mapped target info
                        valid_raw = targets_raw.view(-1)[targets_raw.view(-1) >= 0]
                        valid_mapped = targets_for_ce.view(-1)[targets_for_ce.view(-1) >= 0]
                        log.info("   Raw target range: %d - %d", valid_raw.min().item(), valid_raw.max().item())
                        log.info("   Mapped target range: %d - %d", valid_mapped.min().item(), valid_mapped.max().item())
                        log.info("   Logits shape: %s (classes: 0-%d)", tuple(logits.shape), logits.size(-1)-1)
                        log.info("   Mapped targets in valid range: %.3f", ((valid_mapped < logits.size(-1)) & (valid_mapped >= 0)).float().mean().item())
                        
                        # Check overlap with mapped targets
                        preds_flat = logits.argmax(-1).view(-1)[targets_for_ce.view(-1) >= 0]
                        target_set = set(valid_mapped.cpu().numpy())
                        pred_set = set(preds_flat.cpu().numpy())
                        overlap = len(target_set & pred_set)
                        log.info("   Mapped class overlap: %d/%d targets appear in predictions", overlap, len(target_set))
                    else:
                        # Original diagnostics for identity mapping
                        valid_targets_flat = targets_for_ce.view(-1)[targets_for_ce.view(-1) >= 0]
                        log.info("   Target range: %d - %d", valid_targets_flat.min().item(), valid_targets_flat.max().item())
                        log.info("   Logits shape: %s (classes: 0-%d)", tuple(logits.shape), logits.size(-1)-1)
                        log.info("   Targets in valid range: %.3f", ((valid_targets_flat < logits.size(-1)) & (valid_targets_flat >= 0)).float().mean().item())
                        
                        preds_flat = logits.argmax(-1).view(-1)[targets_for_ce.view(-1) >= 0]
                        target_set = set(valid_targets_flat.cpu().numpy())
                        pred_set = set(preds_flat.cpu().numpy())
                        overlap = len(target_set & pred_set)
                        log.info("   Class overlap: %d/%d targets appear in predictions", overlap, len(target_set))
                        
                    # Final distribution check
                    log.info("   === DISTRIBUTION ANALYSIS ===")
                    if id2col is not None:
                        raw_targets_in_batch = set(targets_raw[valid].cpu().numpy())
                        mappable_in_batch = {t for t in raw_targets_in_batch if t in id2col}
                        log.info("   Raw targets in batch: %d", len(raw_targets_in_batch))
                        log.info("   Mappable targets in batch: %d/%d (%.1f%%)", len(mappable_in_batch), len(raw_targets_in_batch), 100*len(mappable_in_batch)/max(1,len(raw_targets_in_batch)))
                        
                        if len(mappable_in_batch) < len(raw_targets_in_batch):
                            unmappable = raw_targets_in_batch - mappable_in_batch
                            log.info("   Unmappable targets: %s…", sorted(list(unmappable))[:10])
                    log.info("   ==============================")
        except Exception as e:
            log.warning("   Sanity check failed: %s", e)
    
    # Optional centroid lookup for classification re-ranking
    centroid_lookup = None
    if model_type == 'classification' and rerank_centroid:
        try:
            emb_dim = model_config.get('d_embed', 512)
            centroid_lookup = ClusterCentroidLookup(num_clusters=50000, embedding_dim=int(emb_dim), device=str(device))
            centroid_lookup.load_centroids_from_sklearn(cluster_model_path)
            log.info("Loaded centroids from %s for re-ranking", cluster_model_path)
        except Exception as e:
            log.warning("Re-ranking disabled (failed to load centroids): %s", e)
            centroid_lookup = None

    # Run inference
    log.info("Running inference…")
    all_predictions = []  # For metrics (column indices for classification, raw IDs for regression)
    all_targets_raw = []  # Raw cluster IDs (straight from dataset)
    # Optional aligned arrays for temporal binning in analysis
    all_eval_traj_ids = []  # per-position trajectory integer IDs
    all_eval_time_idx = []  # per-position target frame indices
    all_residue_idx = []    # per-position residue index within N
    # Persistent mapping from trajectory names to small integer IDs
    ___traj_name_to_id = {}
    ___next_tid = 0
    all_targets_mapped = []  # Column indices for classification, same as raw for regression
    all_scores = []       # For proper ROC/PRC curves (only top classes)
    # New: collect compact top-k predictions across full class space
    all_topk_ids = []
    all_topk_scores = []
    # New: track union of IDs for expanded eval set
    all_target_ids_set = set()
    all_pred_ids_set = set()
    
    # For classification: also track raw IDs separately for human-readable artifacts
    all_predictions_raw = [] if model_type == 'classification' else None
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Initialize to avoid stale locals from previous iterations
            pred_raw = None
            
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(test_loader)} batches")
            
            # Move batch to device (following reference approach)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Forward pass - handle multi-scale model output (like reference)
            # For regression models, explicitly exclude input_cluster_ids to avoid classification mode
            if model_type == 'regression':
                outputs = model(
                    input_cluster_ids=batch.get('input_cluster_ids'),
                    times=batch['times'],
                    sequence_lengths=batch['sequence_lengths'],
                    history_mask=batch['history_mask']
                    # Note: regression mode determined by output processing, not input parameters
                )
            else:
                # For classification models, include input_cluster_ids if available
                outputs = model(
                    input_cluster_ids=batch.get('input_cluster_ids'),
                    times=batch['times'],
                    sequence_lengths=batch['sequence_lengths'],
                    history_mask=batch['history_mask']
                )
            
            # Extract predictions and scores based on model type and output format
            scores = None  # Will hold probabilities/distances for ROC/PRC
            
            if isinstance(outputs, dict):
                # Multi-scale model output
                if model_type == 'classification' and 'cluster_logits' in outputs:
                    # Keep a copy of native logits for accuracy/top-1 predictions
                    logits_native = outputs['cluster_logits']  # (B, N, full_classes)
                    # Compute native predictions BEFORE any post-processing
                    try:
                        pred_cols_native = logits_native.argmax(dim=-1)
                    except Exception:
                        pred_cols_native = None
                    # Post-processed logits for optional scoring/top-k and curves
                    logits = logits_native
                    # Temperature scaling (configurable)
                    if temperature is not None and abs(float(temperature) - 1.0) > 1e-6:
                        logits = logits / float(temperature)
                    # Optional inference-time logit adjustment based on training priors (post-processing only)
                    if logit_adjust_tau and 'train_class_counts' in checkpoint:
                        try:
                            counts_np = checkpoint['train_class_counts']
                            counts = torch.as_tensor(counts_np, dtype=torch.float32, device=logits.device)
                            priors = (counts / counts.sum()).clamp_min(1e-12)
                            logits = logits - float(logit_adjust_tau) * priors.log()
                        except Exception:
                            pass
                    # (A) Always compute top-k on FULL class space for predictions
                    tkK = min((rerank_topk if rerank_centroid else topk), logits.shape[-1])
                    try:
                        tk_scores_full, tk_ids_full = torch.topk(logits, k=tkK, dim=-1)
                    except Exception as e:
                        print(f"   ⚠️ Failed topk over logits: {e}")
                        tk_scores_full, tk_ids_full = None, None

                    # Use NATIVE full-space top-1 as predictions for metrics (column indices)
                    if pred_cols_native is not None:
                        pred_cols_full = pred_cols_native
                    elif tk_ids_full is not None:
                        pred_cols_full = tk_ids_full[..., 0]
                    else:
                        pred_cols_full = logits.argmax(dim=-1)

                    # (B) Slice ONLY for fast ROC/PRC curves
                    scores = None
                    if eval_cols_tensor is not None:
                        eval_cols_device = eval_cols_tensor.to(logits.device)
                        scores = logits.index_select(-1, eval_cols_device).float()  # (B, N, eval_classes)

                    # (C) For human-readable artifacts: map column -> raw id if mapping exists
                    if id2col is not None and col2id is not None:
                        pred_raw_full = torch.from_numpy(
                            np.array([col2id.get(col.item(), -1) for col in pred_cols_full.cpu().flatten()])
                        ).view_as(pred_cols_full).to(pred_cols_full.device)
                    else:
                        pred_raw_full = pred_cols_full

                    # Optional centroid-based re-ranking over top-k (classification only)
                    if (centroid_lookup is not None) and (tk_ids_full is not None) and (outputs.get('context', None) is not None):
                        try:
                            ctx = outputs['context']  # (B, N, E)
                            Bc, Nc, E = ctx.shape
                            # Normalize context
                            ctx_n = torch.nn.functional.normalize(ctx, dim=-1)
                            # Map column indices to raw IDs via col2id
                            if col2id is not None and len(col2id) > 0:
                                max_col = int(max(col2id.keys()))
                                col2raw = torch.full((max_col + 1,), -1, dtype=torch.long, device=logits.device)
                                for c, rid in col2id.items():
                                    if int(c) >= 0 and int(c) <= max_col:
                                        col2raw[int(c)] = int(rid)
                                topk_cols = tk_ids_full  # (B,N,K)
                                topk_raw = col2raw.index_select(0, topk_cols.view(-1)).view_as(topk_cols)
                                # Gather centroids for these raw IDs
                                cents = centroid_lookup.centroids  # (C,E)
                                cents_sel = cents.index_select(0, topk_raw.view(-1).clamp_min(0)).view(Bc, Nc, tkK, E)
                                cents_n = torch.nn.functional.normalize(cents_sel, dim=-1)
                                # Cosine similarity
                                cos = (ctx_n.unsqueeze(2) * cents_n).sum(dim=-1)  # (B,N,K)
                                # Blend
                                alpha = float(rerank_alpha)
                                beta = float(rerank_beta)
                                blended = alpha * tk_scores_full + (1.0 - alpha) * (beta * cos)
                                # Reorder top-k by blended
                                new_scores, new_ord = torch.sort(blended, dim=-1, descending=True)
                                new_ids = torch.gather(tk_ids_full, -1, new_ord)
                                tk_scores_full, tk_ids_full = new_scores, new_ids
                                pred_cols_full = new_ids[..., 0]
                                # Update human-readable raw pred if mapping exists
                                if col2id is not None and len(col2id) > 0:
                                    pred_raw_full = torch.from_numpy(
                                        np.array([col2id.get(col.item(), -1) for col in pred_cols_full.cpu().flatten()])
                                    ).view_as(pred_cols_full).to(pred_cols_full.device)
                            else:
                                pass
                        except Exception as e:
                            print(f"   ⚠️ Re-ranking failed (skipping): {e}")

                    # (D) Use native full-space predictions going forward for metrics
                    predictions = pred_cols_full
                    pred_raw = pred_raw_full
                elif 'short_term' in outputs:
                    embeddings = outputs['short_term']  # (B, N, 512)
                    if model_type == 'regression':
                        # Compute top-k nearest centroids across ALL classes (cosine similarity)
                        B, N, E = embeddings.shape
                        embeddings_flat = embeddings.view(-1, E)
                        # Normalize inputs/centroids
                        emb = torch.nn.functional.normalize(embeddings_flat, dim=-1)
                        cents = torch.from_numpy(kmeans_model.cluster_centers_.astype(np.float32)).to(emb.device)
                        cents = torch.nn.functional.normalize(cents, dim=-1)
                        M = emb.shape[0]
                        C = cents.shape[0]
                        k_eff = min(topk, C)
                        # Accumulate global top-k in a streaming fashion
                        tk_vals = torch.full((M, k_eff), -1e9, device=emb.device)
                        tk_idx = torch.full((M, k_eff), -1, device=emb.device, dtype=torch.long)
                        c_chunk = max(2000, min(20000, C))
                        for j in range(0, C, c_chunk):
                            j_end = min(j + c_chunk, C)
                            sim = emb @ cents[j:j_end].T  # (M, c_chunk)
                            vals_j, idx_j = torch.topk(sim, k=min(k_eff, sim.shape[1]), dim=1)
                            idx_j = idx_j + j
                            merged_vals = torch.cat([tk_vals, vals_j], dim=1)
                            merged_idx = torch.cat([tk_idx, idx_j], dim=1)
                            new_vals, new_order = torch.topk(merged_vals, k=k_eff, dim=1)
                            new_idx = torch.gather(merged_idx, 1, new_order)
                            tk_vals, tk_idx = new_vals, new_idx
                            del sim, vals_j, idx_j, merged_vals, merged_idx, new_vals, new_order, new_idx
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        # Set predictions/top-k tensors
                        predictions = tk_idx.view(B, N, k_eff)[..., 0]
                        tk_scores_full = tk_vals.view(B, N, k_eff)
                        tk_ids_full = tk_idx.view(B, N, k_eff)
                        # Dense scores over evaluated subset (seed_eval_class_ids) for ROC/PRC
                        Ce = int(len(seed_eval_class_ids))
                        if Ce > 0:
                            eval_ids_tensor = torch.tensor(seed_eval_class_ids, device=emb.device, dtype=torch.long)
                            cents_eval = cents.index_select(0, eval_ids_tensor)  # (Ce, E)
                            scores = (emb @ cents_eval.T).view(B, N, Ce).float()
                        else:
                            scores = None
                    else:
                        print("Warning: classification model returned embeddings instead of logits")
                        continue
                else:
                    print(f"Warning: unexpected output keys: {list(outputs.keys())}")
                    continue
            else:
                # Single output model (fallback)
                if model_type == 'regression':
                    embeddings = outputs  # (B, N, 512)
                    B, N, E = embeddings.shape
                    embeddings_flat = embeddings.view(-1, E)  # (B*N, 512)
                    cluster_ids_flat = cluster_lookup.batch_assign_to_clusters(
                        embeddings_flat
                    )
                    predictions = cluster_ids_flat.view(B, N)  # (B, N)
                else:
                    print("Warning: single output not supported for classification models")
                    continue
            
            # Get targets using bulletproof extraction
            try:
                targets_raw, target_path = extract_target_tensor(batch)
                targets_raw = targets_raw.to(device)  # Ensure targets on correct device
            except KeyError as e:
                print(f"Warning: {e}")
                continue
            
            # BULLETPROOF TARGET MAPPING (same as sanity check)
            if model_type == 'classification' and id2col is not None:
                # Map raw targets to column indices using bulletproof approach
                target_cols = torch.full_like(targets_raw, -1)
                flat_targets = targets_raw.view(-1)
                flat_target_cols = target_cols.view(-1)
                
                # Create mask for mappable targets
                mappable_mask = torch.tensor([rid.item() in id2col for rid in flat_targets], 
                                           device=targets_raw.device, dtype=torch.bool)
                
                # Map only the mappable targets
                if mappable_mask.sum() > 0:
                    mappable_rids = flat_targets[mappable_mask].cpu().numpy()
                    mapped_cols = np.array([id2col[int(rid)] for rid in mappable_rids])
                    flat_target_cols[mappable_mask] = torch.from_numpy(mapped_cols).to(targets_raw.device)
                
                targets = target_cols  # Use mapped column indices for metrics
            else:
                targets = targets_raw  # Use raw targets
            
            # Apply mask if available (use original batch, not model_batch)
            if 'residue_mask' in batch:
                mask = batch['residue_mask'].to(device).bool()  # (B, N)
                # Ensure targets are on the same device as mask for indexing
                targets = targets.to(device)
                valid_predictions = predictions[mask]
                valid_targets = targets[mask]
            else:
                valid_predictions = predictions.flatten()
                valid_targets = targets.flatten()
            
            # Filter valid targets (>= 0)
            valid_mask = valid_targets >= 0
            valid_predictions = valid_predictions[valid_mask]
            valid_targets = valid_targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Track ID unions for expanded eval space
                try:
                    all_target_ids_set.update(valid_targets.detach().cpu().numpy().ravel().tolist())
                except Exception:
                    pass
                # Store predictions (column indices for classification, raw IDs for regression)
                all_predictions.extend(valid_predictions.cpu().numpy())
                
                # Store both raw and mapped targets for audit trails
                targets_raw_for_masking = targets_raw.to(device)  # Ensure on same device as mask
                valid_targets_raw_batch = targets_raw_for_masking if 'residue_mask' in batch else targets_raw_for_masking.flatten()
                if 'residue_mask' in batch:
                    valid_targets_raw_batch = valid_targets_raw_batch[mask][valid_mask]
                else:
                    valid_targets_raw_batch = valid_targets_raw_batch[valid_mask]
                
                all_targets_raw.extend(valid_targets_raw_batch.cpu().numpy())

                # Collect aligned traj/time for temporal binning (if available)
                try:
                    # Determine joint valid mask over (B,N) grid
                    if 'residue_mask' in batch:
                        base_mask = batch['residue_mask'].to(device).bool()
                        joint = (targets_raw >= 0).to(device) & base_mask
                        # Build (B,N) traj ids and time idx arrays
                        if isinstance(batch.get('traj_name', None), list) or isinstance(batch.get('traj_name', None), tuple):
                            traj_names = list(batch['traj_name'])
                        else:
                            # If a single string, replicate to batch size inferred from targets
                            B = targets_raw.shape[0] if targets_raw.ndim == 2 else 1
                            traj_names = [batch.get('traj_name', 'unknown') for _ in range(B)]
                        # Map traj names to stable integer IDs across evaluation
                        batch_traj_ids_rows = []
                        B = targets_raw.shape[0] if targets_raw.ndim == 2 else 1
                        N = targets_raw.shape[1] if targets_raw.ndim == 2 else targets_raw.shape[0]
                        for i in range(B):
                            name = str(traj_names[i]) if i < len(traj_names) else str(traj_names[-1])
                            if name not in ___traj_name_to_id:
                                ___traj_name_to_id[name] = ___next_tid
                                ___next_tid += 1
                            tid = ___traj_name_to_id[name]
                            batch_traj_ids_rows.append([tid] * N)
                        batch_traj_ids = torch.tensor(batch_traj_ids_rows, device=device)
                        # time index: use target_frame from temporal_info if present; else zeros
                        tframes = []
                        tinfos = batch.get('temporal_info', None)
                        if isinstance(tinfos, (list, tuple)) and len(tinfos) == B:
                            for i in range(B):
                                tf = tinfos[i].get('target_frame', 0) if isinstance(tinfos[i], dict) else 0
                                tframes.append(int(tf))
                        else:
                            tframes = [0] * B
                        batch_time_idx = torch.tensor([ [tframes[i]] * N for i in range(B) ], device=device)
                        # Select valid positions and append to global lists
                        sel_traj = batch_traj_ids[joint].detach().cpu().numpy().ravel()
                        sel_time = batch_time_idx[joint].detach().cpu().numpy().ravel()
                        all_eval_traj_ids.extend(sel_traj.tolist())
                        all_eval_time_idx.extend(sel_time.tolist())
                        # Residue indices for valid positions
                        residue_idx_grid = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
                        sel_res = residue_idx_grid[joint].detach().cpu().numpy().ravel()
                        all_residue_idx.extend(sel_res.tolist())
                    else:
                        # No residue_mask; treat flattened targets as positions
                        flat_valid = (targets_raw.view(-1) >= 0).to(device)
                        B = targets_raw.shape[0] if targets_raw.ndim == 2 else 1
                        N = targets_raw.shape[1] if targets_raw.ndim == 2 else targets_raw.shape[0]
                        if isinstance(batch.get('traj_name', None), list) or isinstance(batch.get('traj_name', None), tuple):
                            traj_names = list(batch['traj_name'])
                        else:
                            traj_names = [batch.get('traj_name', 'unknown') for _ in range(B)]
                        # Map traj names to stable integer IDs across evaluation
                        batch_traj_ids_rows = []
                        for i in range(B):
                            name = str(traj_names[i]) if i < len(traj_names) else str(traj_names[-1])
                            if name not in ___traj_name_to_id:
                                ___traj_name_to_id[name] = ___next_tid
                                ___next_tid += 1
                            tid = ___traj_name_to_id[name]
                            batch_traj_ids_rows.append([tid] * N)
                        batch_traj_ids = torch.tensor(batch_traj_ids_rows, device=device).view(-1)
                        tinfos = batch.get('temporal_info', None)
                        tframes = []
                        if isinstance(tinfos, (list, tuple)) and len(tinfos) == B:
                            for i in range(B):
                                tf = tinfos[i].get('target_frame', 0) if isinstance(tinfos[i], dict) else 0
                                tframes.append(int(tf))
                        else:
                            tframes = [0] * B
                        batch_time_idx = torch.tensor([ [tframes[i]] * N for i in range(B) ], device=device).view(-1)
                        sel_traj = batch_traj_ids[flat_valid].detach().cpu().numpy().ravel()
                        sel_time = batch_time_idx[flat_valid].detach().cpu().numpy().ravel()
                        all_eval_traj_ids.extend(sel_traj.tolist())
                        all_eval_time_idx.extend(sel_time.tolist())
                        # Residue indices for valid positions
                        residue_idx_flat = torch.arange(N, device=device).repeat(B)
                        sel_res = residue_idx_flat[flat_valid].detach().cpu().numpy().ravel()
                        all_residue_idx.extend(sel_res.tolist())
                except Exception as _e:
                    # Non-fatal: temporal arrays are optional for analysis
                    pass
                all_targets_mapped.extend(valid_targets.cpu().numpy())
                
                # For classification: also store raw ID predictions for human-readable artifacts
                if model_type == 'classification' and pred_raw is not None:
                    if 'residue_mask' in batch:
                        valid_pred_raw = pred_raw[mask][valid_mask]
                    else:
                        valid_pred_raw = pred_raw.flatten()[valid_mask]
                    all_predictions_raw.extend(valid_pred_raw.cpu().numpy())
                
                # Store scores for ROC/PRC curves (only valid positions)
                if scores is not None:
                    if 'residue_mask' in batch:
                        # Move scores to GPU to match mask device, then apply masks
                        scores_gpu = scores.to(device)
                        valid_scores = scores_gpu[mask][valid_mask].cpu()  # Apply both masks
                    else:
                        valid_scores = scores.flatten(0, 1)[valid_mask].cpu()  # (valid_positions, top_classes)
                    all_scores.append(valid_scores)  # Store as list of tensors to save memory

                # Save compact top-k arrays when available
                if 'tk_ids_full' in locals() and tk_ids_full is not None:
                    if 'residue_mask' in batch:
                        valid_topk_ids = tk_ids_full[mask][valid_mask].detach().cpu()
                        valid_topk_scores = (tk_scores_full[mask][valid_mask].detach().cpu()
                                             if 'tk_scores_full' in locals() and tk_scores_full is not None
                                             else torch.zeros_like(valid_topk_ids, dtype=torch.float32))
                    else:
                        valid_topk_ids = tk_ids_full.flatten(0, 1)[valid_mask].detach().cpu()
                        valid_topk_scores = (tk_scores_full.flatten(0, 1)[valid_mask].detach().cpu()
                                             if 'tk_scores_full' in locals() and tk_scores_full is not None
                                             else torch.zeros_like(valid_topk_ids, dtype=torch.float32))
                    all_topk_ids.append(valid_topk_ids)
                    all_topk_scores.append(valid_topk_scores)

                    # Update predicted ID set using global top-1 (map to raw IDs if mapping available)
                    try:
                        top1_cols = valid_topk_ids[:, 0].numpy().ravel().tolist()
                        if col2id is not None:
                            all_pred_ids_set.update([col2id.get(int(c), int(c)) for c in top1_cols])
                        else:
                            all_pred_ids_set.update([int(c) for c in top1_cols])
                    except Exception:
                        pass
    
    print(f"Collected {len(all_predictions):,} predictions")
    
    if len(all_predictions) == 0:
        print("ERROR: No valid predictions collected! Saving empty results and exiting.")
        os.makedirs(output_dir, exist_ok=True)
        eval_results = {
            "run_name": model_name,
            "model_type": model_type,
            "config": model_config,
            "metrics": {},
            "evaluation_timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            "note": "No valid predictions collected"
        }
        with open(Path(output_dir) / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        return
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets_raw = np.array(all_targets_raw)
    all_targets_mapped = np.array(all_targets_mapped)
    
    # Concatenate scores if available
    all_scores_array = None
    if all_scores:
        print("Processing scores for proper ROC/PRC curves...")
        all_scores_array = torch.cat(all_scores, dim=0).numpy()  # (total_predictions, num_clusters)
        print(f"   Scores shape: {all_scores_array.shape}")
    
    # Save raw results
    results_file = Path(output_dir) / 'cluster_results'
    results_file.mkdir(exist_ok=True)
    
    # Save predictions and targets (both raw and mapped forms for audit)
    if model_type == 'classification':
        # For classification: save both column indices (for metrics) and raw IDs (for humans)
        np.save(results_file / 'predicted_cluster_cols.npy', all_predictions)  # Column indices
        if all_predictions_raw is not None and len(all_predictions_raw) > 0:
            np.save(results_file / 'predicted_cluster_ids_raw.npy', np.array(all_predictions_raw))  # Raw IDs
        np.save(results_file / 'target_cluster_cols.npy', all_targets_mapped)  # Column indices (for metrics)
        np.save(results_file / 'target_cluster_ids_raw.npy', all_targets_raw)  # Raw IDs (from dataset)
        print(f"   Saved classification: pred_cols, pred_raw, target_cols, target_raw")
    else:
        # For regression: predictions are already raw IDs, targets are same raw/mapped
        np.save(results_file / 'predicted_cluster_ids.npy', all_predictions)
        np.save(results_file / 'target_cluster_ids_raw.npy', all_targets_raw)
        np.save(results_file / 'target_cluster_ids.npy', all_targets_mapped)  # Same as raw for regression
        print(f"   Saved regression predictions: raw IDs")
    
    # Save scores and class mapping for proper ROC/PRC computation
    if all_scores_array is not None:
        # Convert to float32 to save memory/disk space (cut file size by ~50%)
        all_scores_array = all_scores_array.astype(np.float32, copy=False)
        
        # Use compressed format for large arrays to save disk space
        np.savez_compressed(results_file / 'prediction_scores.npz', scores=all_scores_array)
        # Save eval_class_ids that ALIGN with prediction_scores columns
        if model_type == 'classification' and eval_cols_tensor is not None:
            eval_ids_for_scores = eval_cols_tensor.cpu().numpy().astype(np.int32)
        else:
            eval_ids_for_scores = np.asarray(seed_eval_class_ids, dtype=np.int32)
        np.save(results_file / 'eval_class_ids.npy', eval_ids_for_scores)
        # Optional: save raw IDs corresponding to score columns for clarity
        try:
            if model_type == 'classification' and eval_cols_tensor is not None and col2id is not None:
                eval_cols = eval_cols_tensor.cpu().numpy().astype(np.int32)
                eval_raw_for_scores = np.array([col2id.get(int(c), -1) for c in eval_cols], dtype=np.int32)
                np.save(results_file / 'eval_class_raw_ids.npy', eval_raw_for_scores)
        except Exception as e:
            print(f"   ⚠️ Could not save eval_class_raw_ids.npy: {e}")
        print(f"   Saved prediction scores: {all_scores_array.shape} (float32, compressed)")
        print(f"   Saved evaluation class IDs (for scores): {eval_ids_for_scores.shape}")
        
        # Save evaluation metadata with safe JSON serialization
        eval_metadata = {
            'num_evaluated_classes': int(len(seed_eval_class_ids)),
            'total_unique_classes': int(len(class_counts)),
            'evaluated_class_coverage': float(len(seed_eval_class_ids) / max(1, len(class_counts))),
            'model_type': str(model_type),
            'normalization_applied': bool(model_type == 'regression'),  # L2 norm for regression
            'top_classes': [int(c) for c in seed_eval_class_ids[:10].tolist()],
            'class_frequencies': [int(class_counts[int(c)]) for c in seed_eval_class_ids[:10].tolist()]
        }
        with open(results_file / 'eval_metadata.json', 'w') as f:
            json.dump(eval_metadata, f, indent=2)
        print(f" Saved evaluation metadata: {len(seed_eval_class_ids)}/{len(class_counts)} classes")

    # Save new compact top-k outputs (ids+scores) if computed
    if all_topk_ids:
        tk_ids_np = torch.cat(all_topk_ids, dim=0).numpy().astype(np.int32)
        np.save(results_file / 'topk_ids.npy', tk_ids_np)
        # Also save raw-ID top-k for classification when mapping is available
        try:
            if model_type == 'classification' and col2id is not None and len(col2id) > 0:
                max_col = int(max(col2id.keys())) if len(col2id) else -1
                col2raw = np.full(max_col + 1, -1, dtype=np.int32)
                for c, rid in col2id.items():
                    if int(c) >= 0:
                        col2raw[int(c)] = int(rid)
                tk_ids_raw = col2raw[tk_ids_np]
                np.save(results_file / 'topk_ids_raw.npy', tk_ids_raw.astype(np.int32))
                print(f"   Saved topk_ids_raw.npy {tk_ids_raw.shape} (global IDs)")
        except Exception as e:
            print(f"   ⚠️ Could not save topk_ids_raw.npy: {e}")
        if all_topk_scores:
            tk_scores_np = torch.cat(all_topk_scores, dim=0).numpy().astype(np.float32)
            np.save(results_file / 'topk_scores.npy', tk_scores_np)
            print(f"   Saved topk_ids.npy {tk_ids_np.shape} and topk_scores.npy {tk_scores_np.shape}")
        else:
            print(f"   Saved topk_ids.npy {tk_ids_np.shape}")

    # Save optional temporal alignment arrays if collected
    try:
        if len(all_eval_traj_ids) > 0 and len(all_eval_time_idx) == len(all_eval_traj_ids):
            np.save(results_file / 'eval_traj_ids.npy', np.asarray(all_eval_traj_ids, dtype=np.int32))
            np.save(results_file / 'eval_time_idx.npy', np.asarray(all_eval_time_idx, dtype=np.int32))
            if len(all_residue_idx) == len(all_eval_traj_ids):
                np.save(results_file / 'eval_residue_idx.npy', np.asarray(all_residue_idx, dtype=np.int32))
            print(f"   Saved eval_traj_ids.npy and eval_time_idx.npy with {len(all_eval_traj_ids)} entries")
    except Exception:
        pass

    # Save expanded eval_class_ids = union(targets ∪ preds ∪ topN_by_frequency)
    try:
        topN_from_counts = [c for c, _ in sorted(class_counts.items(), key=lambda x: -x[1])[:max(1, eval_topN)]]
    except Exception:
        topN_from_counts = []
    expanded_eval_union = sorted(set(topN_from_counts) | set(all_target_ids_set) | set(all_pred_ids_set))
    if expanded_eval_union:
        np.save(results_file / 'eval_class_ids_expanded.npy', np.array(expanded_eval_union, dtype=np.int32))
        print(f"   Saved eval_class_ids_expanded.npy with {len(expanded_eval_union)} classes")
    
    # Save a small ID-space metadata file for downstream readers
    try:
        with open(results_file / 'id_space.json', 'w') as f:
            json.dump({
                "predicted_cluster_cols": "column",
                "predicted_cluster_ids_raw": "global",
                "target_cluster_cols": "column",
                "target_cluster_ids_raw": "global",
                "topk_ids": "column",
                "topk_ids_raw": "global",
                "prediction_scores": "column_subspace",
                "eval_class_ids": "column",
                "eval_class_raw_ids": "global"
            }, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Could not write id_space.json: {e}")

    print(f"Saved cluster results to: {results_file}")
    
    # Compute metrics and generate ROC/PRC curves (self-contained)
    print("Computing metrics and generating ROC/PRC curves...")
    
    # Prepare eval_class_ids for metrics computation
    if model_type == 'classification' and id2col is not None:
        # For classification: targets are already mapped to column indices during inference
        # If we computed scores, align eval_class_ids to those score columns
        if all_scores_array is not None and 'eval_cols_tensor' in locals() and eval_cols_tensor is not None:
            eval_class_ids_for_metrics = eval_cols_tensor.cpu().numpy().astype(np.int64)
            print(f"✅ Using eval columns from scores: {len(eval_class_ids_for_metrics)} classes")
        else:
            # Expand eval set: seed ∪ unique targets ∪ unique predictions (limited by --eval_topN)
            expanded_eval_ids = np.unique(np.concatenate([
                seed_eval_class_ids,
                np.array(list(set(all_targets_raw.tolist())), dtype=np.int64),
                np.array(list(set(all_predictions.flatten().tolist())), dtype=np.int64)
            ]))
            if expanded_eval_ids.size > eval_topN:
                from collections import Counter
                tgt_counts = Counter(all_targets_raw.tolist())
                expanded_eval_ids = np.array(sorted(expanded_eval_ids, key=lambda c: tgt_counts.get(int(c),0), reverse=True)[:eval_topN])
            eval_class_ids_for_metrics = np.array([id2col[c] for c in expanded_eval_ids if c in id2col])
            print(f"✅ Targets already mapped during inference - using expanded set {len(eval_class_ids_for_metrics)} eval classes")
    else:
        # For regression or no-mapping
        if all_scores_array is not None:
            # Align eval_class_ids to the score columns (seed subset of raw IDs)
            eval_class_ids_for_metrics = np.asarray(seed_eval_class_ids, dtype=np.int64)
            print(f"✅ Using seed eval_class_ids for scores: {len(eval_class_ids_for_metrics)} classes")
        else:
            # Use expanded union for coverage-only metrics
            expanded_eval_ids = np.unique(np.concatenate([
                seed_eval_class_ids,
                np.array(list(set(all_targets_raw.tolist())), dtype=np.int64),
                np.array(list(set(all_predictions.flatten().tolist())), dtype=np.int64)
            ]))
            if expanded_eval_ids.size > eval_topN:
                from collections import Counter
                tgt_counts = Counter(all_targets_raw.tolist())
                expanded_eval_ids = np.array(sorted(expanded_eval_ids, key=lambda c: tgt_counts.get(int(c),0), reverse=True)[:eval_topN])
            eval_class_ids_for_metrics = expanded_eval_ids
            print(f"✅ Using raw expanded eval_class_ids - {len(eval_class_ids_for_metrics)} classes")
        
    # Final distribution analysis before metrics
    print("\n=== FINAL DISTRIBUTION CHECK ===")
    pred_set = set(all_predictions)
    tgt_mapped_set = set(all_targets_mapped)
    tgt_raw_set = set(all_targets_raw)
    
    if model_type == 'classification' and id2col is not None and all_predictions_raw:
        pred_raw_set = set(all_predictions_raw)
        print(f"   Predictions (cols): {len(pred_set)} unique")
        print(f"   Targets (cols): {len(tgt_mapped_set)} unique") 
        print(f"   Column overlap: {len(pred_set & tgt_mapped_set)}/{max(1,len(tgt_mapped_set))} ({100*len(pred_set & tgt_mapped_set)/max(1,len(tgt_mapped_set)):.1f}%)")
        print(f"   Predictions (raw): {len(pred_raw_set)} unique")
        print(f"   Targets (raw): {len(tgt_raw_set)} unique")
        print(f"   Raw overlap: {len(pred_raw_set & tgt_raw_set)}/{max(1,len(tgt_raw_set))} ({100*len(pred_raw_set & tgt_raw_set)/max(1,len(tgt_raw_set)):.1f}%)")
        print(f"   ✅ Using column-space overlap for metrics: {100*len(pred_set & tgt_mapped_set)/max(1,len(tgt_mapped_set)):.1f}%")
    else:
        print(f"   Predictions: {len(pred_set)} unique")
        print(f"   Targets: {len(tgt_mapped_set)} unique")
        print(f"   Overlap: {len(pred_set & tgt_mapped_set)}/{max(1,len(tgt_mapped_set))} ({100*len(pred_set & tgt_mapped_set)/max(1,len(tgt_mapped_set)):.1f}%)")
    
    if len(pred_set & tgt_mapped_set) / max(1, len(tgt_mapped_set)) < 0.3:
        print("   ⚠️  WARNING: <30% overlap suggests systematic issues!")
    print("=================================\n")
    
    # Compute metrics with properly aligned data
    metrics, curves_data = compute_metrics_with_curves(
        np.array(all_predictions), np.array(all_targets_mapped), all_scores_array, eval_class_ids_for_metrics
    )
    # Clarify that accuracy is computed from native logits (pre-prior/temperature)
    metrics['accuracy_logits'] = 'native'
    
    # Create evaluation results structure
    eval_results = {
        "run_name": model_name,
        "model_type": model_type,
        "config": model_config,
        "metrics": metrics,
        "evaluation_timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    }

    # -----------------------------
    # Extra diagnostics and analysis
    # -----------------------------
    extras = {}

    # 1) Coverage@K (how many target classes are covered by the K most frequent predictions)
    try:
        pred_counts = Counter(all_predictions)
        tgt_counts = Counter(all_targets_mapped)
        tgt_classes = set(tgt_counts.keys())
        pred_classes_by_freq = [cls for cls, _ in pred_counts.most_common()]

        def coverage_at_k(k: int) -> float:
            covered = set(pred_classes_by_freq[:min(k, len(pred_classes_by_freq))]) & tgt_classes
            return float(len(covered) / max(1, len(tgt_classes)))

        ks = [10, 50, 100, 500, 1000]
        extras['coverage_at_k'] = {int(k): coverage_at_k(k) for k in ks}
        extras['num_unique_targets'] = int(len(tgt_classes))
        extras['num_unique_predictions'] = int(len(pred_counts))
    except Exception as e:
        extras['coverage_error'] = str(e)

    # 2) Accuracy vs. class-frequency bins (deciles over target frequency)
    try:
        # Build per-sample correctness
        correct = (all_predictions == all_targets_mapped)
        # Get target frequencies per class
        freqs = np.array([tgt_counts[c] for c in all_targets_mapped])
        # Bin by deciles
        if len(freqs) > 0:
            quantiles = np.quantile(freqs, np.linspace(0, 1, 11))
            acc_bins = []
            for i in range(10):
                lo, hi = quantiles[i], quantiles[i+1]
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    acc_bins.append({
                        'bin': f'{int(lo)}-{int(hi)}',
                        'count': int(mask.sum()),
                        'accuracy': float(correct[mask].mean())
                    })
                else:
                    acc_bins.append({'bin': f'{int(lo)}-{int(hi)}', 'count': 0, 'accuracy': None})
            extras['accuracy_vs_frequency_bins'] = acc_bins
    except Exception as e:
        extras['acc_vs_freq_error'] = str(e)

    # 3) Logit entropy diagnostics (if scores available)
    try:
        if all_scores_array is not None and all_scores_array.size > 0:
            # Softmax over evaluated classes then entropy per-sample
            max_scores = all_scores_array.max(axis=1, keepdims=True)
            exp_scores = np.exp(all_scores_array - max_scores)
            probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)
            entropy = -(probs * (np.log(probs + 1e-8))).sum(axis=1)
            extras['logit_entropy_mean'] = float(entropy.mean())
            extras['logit_entropy_std'] = float(entropy.std())
    except Exception as e:
        extras['entropy_error'] = str(e)

    # 4) Continuous regression metrics (cosine/MSE vs target and copy-last baseline)
    try:
        if model_type == 'regression':
            # Re-run a light pass to compute continuous metrics without storing huge tensors
            cosine_sums = 0.0
            mse_sums = 0.0
            baseline_mse_sums = 0.0
            count = 0
            self_device = device  # use the same device
            # Reload a small iterator over the same test loader for a pass
            _, _, tmp_loader = create_dataloaders(
                data_dir=data_dir,
                metadata_path=metadata_path,
                batch_size=batch_size,
                max_sequence_length=5,
                stride=10,
                num_workers=2
            )
            model = ProteinTransformerClassificationOnly(
                d_embed=model_config.get('d_embed', 512),
                num_heads=model_config.get('num_heads', 8),
                dropout=model_config.get('dropout', 0.1),
                use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', True),
                min_context_frames=model_config.get('min_context_frames', 2),
                num_clusters=50000
            ).to(self_device)
            # Load state dict (filtered)
            filtered_state = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            model.load_state_dict(filtered_state, strict=False)
            model.eval()
            with torch.no_grad():
                for i, b in enumerate(tmp_loader):
                    for k, v in b.items():
                        if isinstance(v, torch.Tensor):
                            b[k] = v.to(self_device)
                    out = model(
                        input_cluster_ids=b.get('input_cluster_ids'),
                        times=b['times'],
                        sequence_lengths=b['sequence_lengths'],
                        history_mask=b['history_mask']
                    )
                    if 'short_term' not in out:
                        continue
                    pred = out['short_term']         # (B, N, E)
                    tgt = b.get('targets', {}).get('short_term', None)
                    if tgt is None:
                        continue
                    # Mask (last frame)
                    mask = b.get('residue_mask', torch.ones(pred.shape[:2], device=pred.device))
                    if mask.dim() == 3:
                        mask = mask[:, -1, :]
                    m = mask.bool().unsqueeze(-1)  # (B,N,1)
                    pred_v = pred[m.expand_as(pred)].view(-1, pred.shape[-1])
                    tgt_v = tgt[m.expand_as(tgt)].view(-1, tgt.shape[-1])
                    if pred_v.numel() == 0:
                        continue
                    # Cosine and MSE
                    cos = torch.nn.functional.cosine_similarity(pred_v, tgt_v, dim=-1).mean().item()
                    mse = torch.nn.functional.mse_loss(pred_v, tgt_v).item()
                    # Copy-last baseline
                    # Extract last frame current embeddings
                    B, T, N, E = b['embeddings'].shape
                    idx = (b['sequence_lengths'] - 1).view(B, 1, 1, 1).expand(-1, -1, N, E)
                    cur = torch.gather(b['embeddings'], 1, idx).squeeze(1)
                    cur_v = cur[m.expand_as(cur)].view(-1, E)
                    base_mse = torch.nn.functional.mse_loss(cur_v, tgt_v).item()
                    cosine_sums += cos
                    mse_sums += mse
                    baseline_mse_sums += base_mse
                    count += 1
                    if i >= 50:  # cap for speed
                        break
            if count > 0:
                extras['regression_cosine_mean'] = float(cosine_sums / count)
                extras['regression_mse_mean'] = float(mse_sums / count)
                extras['regression_baseline_mse_mean'] = float(baseline_mse_sums / count)
    except Exception as e:
        extras['regression_metrics_error'] = str(e)

    # Save extras
    with open(Path(output_dir) / 'metrics_extra.json', 'w') as f:
        json.dump(extras, f, indent=2)

    # Granular CSVs: per-time and per-trajectory accuracy, plus mispredictions dump
    try:
        import csv
        results_dir = Path(output_dir) / 'cluster_results'
        results_dir.mkdir(exist_ok=True)
        # Per-time accuracy
        if len(all_eval_time_idx) == len(all_predictions):
            times_np = np.asarray(all_eval_time_idx, dtype=np.int32)
            correct_np = (all_predictions == all_targets_mapped).astype(np.int8)
            per_time = {}
            for t, c in zip(times_np, correct_np):
                n, k = per_time.get(t, (0, 0))
                per_time[t] = (n + 1, k + int(c))
            with open(results_dir / 'per_time_accuracy.csv', 'w', newline='') as fcsv:
                w = csv.writer(fcsv)
                w.writerow(['time_idx', 'count', 'num_correct', 'accuracy'])
                for t in sorted(per_time.keys()):
                    n, k = per_time[t]
                    w.writerow([int(t), int(n), int(k), float(k / max(1, n))])
        # Per-trajectory accuracy
        if len(all_eval_traj_ids) == len(all_predictions):
            traj_np = np.asarray(all_eval_traj_ids, dtype=np.int32)
            correct_np = (all_predictions == all_targets_mapped).astype(np.int8)
            per_traj = {}
            for tid, c in zip(traj_np, correct_np):
                n, k = per_traj.get(tid, (0, 0))
                per_traj[tid] = (n + 1, k + int(c))
            with open(results_dir / 'per_traj_accuracy.csv', 'w', newline='') as fcsv:
                w = csv.writer(fcsv)
                w.writerow(['traj_id', 'count', 'num_correct', 'accuracy'])
                for tid in sorted(per_traj.keys()):
                    n, k = per_traj[tid]
                    w.writerow([int(tid), int(n), int(k), float(k / max(1, n))])
        # Mispredictions dump (sampled for size)
        if len(all_eval_traj_ids) == len(all_predictions):
            mis_mask = (all_predictions != all_targets_mapped)
            if mis_mask.any():
                # sample up to 1e6 rows to keep file manageable
                idx = np.where(mis_mask)[0]
                max_rows = 1000000
                if idx.size > max_rows:
                    idx = np.random.default_rng(0).choice(idx, size=max_rows, replace=False)
                with open(results_dir / 'mispredictions_sample.csv', 'w', newline='') as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(['traj_id', 'time_idx', 'residue_idx', 'target', 'pred', 'correct'])
                    traj_np = np.asarray(all_eval_traj_ids, dtype=np.int32)
                    time_np = np.asarray(all_eval_time_idx, dtype=np.int32)
                    res_np = np.asarray(all_residue_idx, dtype=np.int32) if len(all_residue_idx) == len(all_predictions) else np.full_like(traj_np, -1)
                    for i in idx:
                        w.writerow([
                            int(traj_np[i]), int(time_np[i]), int(res_np[i]),
                            int(all_targets_raw[i]) if i < len(all_targets_raw) else int(all_targets_mapped[i]),
                            int(all_predictions[i]), 0
                        ])
    except Exception:
        pass
    
    # Save evaluation results
    with open(Path(output_dir) / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Create visualization with ROC/PRC curves
    if not no_plots and HAS_SEABORN:
        create_visualization(output_dir, model_name, model_type, metrics, curves_data)

    
    log.info("Evaluation complete! Results saved to: %s", output_dir)
    log.info("Key Results:")
    log.info("   Accuracy: %.4f", metrics.get('accuracy', 0))
    log.info("   AUROC: %.4f", metrics.get('auroc_macro', 0))
    log.info("   AUPRC: %.4f", metrics.get('auprc_macro', 0))
    log.info("   Class Coverage: %.4f", metrics.get('coverage_ratio', 0))


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    _maybe_set_plot_style()
    parser = argparse.ArgumentParser(description='Evaluate a single model with ROC/PRC curves')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name for the model (for plots)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['regression', 'classification', 'dual_head'],
                       help='Type of model to evaluate')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    parser.add_argument('--cluster_model_path', type=str, 
                       default='/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl',
                       help='Path to cluster model for regression conversion')
    parser.add_argument('--no_plots', action='store_true', default=False,
                       help='Disable plotting (useful on headless systems or without seaborn)')
    parser.add_argument('--eval_topN', type=int, default=2000,
                       help='Seed top-N classes by frequency; expanded with predicted+target classes')
    parser.add_argument('--use_faiss', action='store_true', default=False,
                       help='Use FAISS for nearest-centroid retrieval in regression (if available)')
    parser.add_argument('--topk', type=int, default=10,
                       help='Top-k predictions to save per sample (ids+scores)')
    parser.add_argument('--logit_adjust_tau', type=float, default=0.0,
                       help='Apply inference-time logit adjustment with tau (0 disables)')
    # Phase 1 configurable inference features
    parser.add_argument('--rerank_centroid', action='store_true', default=False,
                       help='Enable centroid-based re-ranking on top-k classes (classification only)')
    parser.add_argument('--rerank_topk', type=int, default=10,
                       help='Top-k size for centroid-based re-ranking (<= topk)')
    parser.add_argument('--rerank_alpha', type=float, default=0.8,
                       help='Blend weight: score = alpha*logit + (1-alpha)*beta*cosine')
    parser.add_argument('--rerank_beta', type=float, default=20.0,
                       help='Cosine scale factor used in centroid re-ranking')
    
    args = parser.parse_args()
    
    if args.model_type == 'dual_head':
        # Evaluate classification head
        cls_dir = str(Path(args.output_dir) / 'classification')
        evaluate_single_model(
            model_path=args.model_path,
            model_name=f"{args.model_name}_cls",
            model_type='classification',
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            output_dir=cls_dir,
            cluster_model_path=args.cluster_model_path,
            batch_size=args.batch_size,
            device=args.device,
            no_plots=args.no_plots,
            eval_topN=args.eval_topN,
            use_faiss=args.use_faiss,
            topk=args.topk,
            logit_adjust_tau=args.logit_adjust_tau,
            rerank_centroid=args.rerank_centroid,
            rerank_topk=args.rerank_topk,
            rerank_alpha=args.rerank_alpha,
            rerank_beta=args.rerank_beta,
        )

        # Evaluate regression head
        reg_dir = str(Path(args.output_dir) / 'regression')
        evaluate_single_model(
            model_path=args.model_path,
            model_name=f"{args.model_name}_reg",
            model_type='regression',
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            output_dir=reg_dir,
            cluster_model_path=args.cluster_model_path,
            batch_size=max(1, args.batch_size // 2),  # be conservative for 50k logits memory
            device=args.device,
            no_plots=args.no_plots,
            topk=args.topk,
        )
    else:
        evaluate_single_model(
            model_path=args.model_path,
            model_name=args.model_name,
            model_type=args.model_type,
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            cluster_model_path=args.cluster_model_path,
            batch_size=args.batch_size,
            device=args.device,
            no_plots=args.no_plots,
            topk=args.topk,
            logit_adjust_tau=args.logit_adjust_tau,
            rerank_centroid=args.rerank_centroid,
            rerank_topk=args.rerank_topk,
            rerank_alpha=args.rerank_alpha,
            rerank_beta=args.rerank_beta,
        )


if __name__ == "__main__":
    main()
