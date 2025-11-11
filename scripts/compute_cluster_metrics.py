#!/usr/bin/env python3
"""
Simple Memory-Efficient Cluster Metrics Computation

Computes basic classification metrics without memory-intensive operations.
"""

# srun --partition=rbaltman --cpus-per-task=2 --mem=4G python scripts/compute_cluster_metrics.py --results_dir evaluation_results/emprot_baseline_MSE_regression_ONLY

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_cluster_results(results_dir: Path):
    """Load cluster predictions and targets."""
    cluster_results_dir = results_dir / 'cluster_results'
    
    pred_file = cluster_results_dir / 'predicted_cluster_ids.npy'
    target_file = cluster_results_dir / 'target_cluster_ids.npy'
    
    print(f"📁 Loading cluster results from: {cluster_results_dir}")
    
    predicted_clusters = np.load(pred_file)
    target_clusters = np.load(target_file)
    
    print(f"✅ Loaded cluster results:")
    print(f"   Predicted clusters: {predicted_clusters.shape}")
    print(f"   Target clusters: {target_clusters.shape}")
    
    return predicted_clusters, target_clusters


def compute_auroc_auprc_aligned(valid_targets, valid_predictions, prediction_scores=None, eval_class_ids=None, max_classes=100, max_samples=50000):
    """
    Compute AUROC and AUPRC using the SAME method as ClassificationEvaluator.
    
    This ensures fair comparison with classification and dual-head models.
    Since we only have hard predictions (not probabilities), we simulate the 
    ClassificationEvaluator approach using one-hot encoding.
    
    Strategy (aligned with ClassificationEvaluator):
    1. Sample most common classes from targets (like ClassificationEvaluator line 264)
    2. Create binary matrices for sampled classes  
    3. Use one-hot predictions as "probabilities"
    4. Compute macro-averaged AUROC/AUPRC exactly like ClassificationEvaluator
    """
    try:
        # Set seed for reproducibility  
        np.random.seed(42)
        
        # Step 1: Sample classes by frequency (not first-appearing)
        if eval_class_ids is not None:
            # Use pre-computed top classes
            sample_classes = eval_class_ids[:min(max_classes, len(eval_class_ids))]
            print(f"      Using pre-computed top {len(sample_classes)} classes")
        else:
            # Fallback: compute from targets
            from collections import Counter
            class_counts = Counter(valid_targets)
            sample_classes = np.array([cls for cls, _ in class_counts.most_common(max_classes)])
            print(f"      Computing top {len(sample_classes)} classes by frequency")
        
        print(f"      Evaluating {len(sample_classes)} classes (aligned with ClassificationEvaluator)...")
        
        # Step 2: Sample data points if too many
        if len(valid_targets) > max_samples:
            indices = np.random.choice(len(valid_targets), max_samples, replace=False)
            sample_targets = valid_targets[indices]
            sample_predictions = valid_predictions[indices]
            sample_scores = prediction_scores[indices] if prediction_scores is not None else None
            print(f"      Sampling {max_samples:,} data points...")
        else:
            sample_targets = valid_targets
            sample_predictions = valid_predictions
            sample_scores = prediction_scores
        
        # Step 3: Create binary matrices and probability matrices
        y_true_binary = np.zeros((len(sample_targets), len(sample_classes)))
        y_proba_sampled = np.zeros((len(sample_targets), len(sample_classes)))
        
        # Create class ID to column mapping if using eval_class_ids
        if sample_scores is not None and eval_class_ids is not None:
            class_id_to_col = {cls_id: i for i, cls_id in enumerate(eval_class_ids)}
        
        for i, cls in enumerate(sample_classes):
            y_true_binary[:, i] = (sample_targets == cls).astype(int)
            
            if sample_scores is not None:
                if eval_class_ids is not None and cls in class_id_to_col:
                    # Map class ID to score column
                    score_col = class_id_to_col[cls]
                    y_proba_sampled[:, i] = sample_scores[:, score_col]
                else:
                    # Direct indexing (fallback)
                    y_proba_sampled[:, i] = sample_scores[:, cls] if cls < sample_scores.shape[1] else 0
            else:
                # Fallback: Use one-hot encoding as "probabilities"
                y_proba_sampled[:, i] = (sample_predictions == cls).astype(float)
        
        # Step 4: Compute macro-averaged AUROC and AUPRC (same as ClassificationEvaluator lines 277-289)
        auroc_scores = []
        auprc_scores = []
        
        for i in range(len(sample_classes)):
            if y_true_binary[:, i].sum() > 0:  # Only if positive examples exist
                try:
                    auroc = roc_auc_score(y_true_binary[:, i], y_proba_sampled[:, i])
                    auprc = average_precision_score(y_true_binary[:, i], y_proba_sampled[:, i])
                    auroc_scores.append(auroc)
                    auprc_scores.append(auprc)
                except:
                    pass
        
        # Step 5: Average across classes (same as ClassificationEvaluator lines 291-296)
        if auroc_scores:
            auroc_macro = float(np.mean(auroc_scores))
            auprc_macro = float(np.mean(auprc_scores))
            print(f"      AUROC computed on {len(auroc_scores)} classes: {auroc_macro:.4f}")
            print(f"      AUPRC computed on {len(auprc_scores)} classes: {auprc_macro:.4f}")
        else:
            auroc_macro = 0.0
            auprc_macro = 0.0
            print("      Could not compute AUROC/AUPRC (insufficient valid classes)")
            
        return auroc_macro, auprc_macro
        
    except Exception as e:
        print(f"      Error computing AUROC/AUPRC: {e}")
        return 0.0, 0.0


def compute_roc_prc_curves(valid_targets, valid_predictions, prediction_scores=None, max_classes=5, max_samples=10000):
    """
    Compute ROC and PRC curves for top classes to create actual curve plots.
    
    Args:
        valid_targets: Array of true cluster IDs
        valid_predictions: Array of predicted cluster IDs  
        max_classes: Number of top classes to compute curves for
        max_samples: Maximum samples to use (for memory efficiency)
    
    Returns:
        Dictionary containing curve data for plotting
    """
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
        
        curves_data = {
            'roc_curves': [],
            'prc_curves': [], 
            'class_names': [],
            'auroc_scores': [],
            'auprc_scores': []
        }
        
        print(f"      Computing ROC/PRC curves for top {len(top_classes)} classes...")
        
        for cls in top_classes:
            # Create binary classification problem
            y_true = (sample_targets == cls).astype(int)
            
            # Use actual scores if available, otherwise fall back to one-hot
            if sample_scores is not None:
                y_score = sample_scores[:, cls]  # Actual probabilities/similarities
                print(f"         Using actual scores for class {cls}")
            else:
                y_score = (sample_predictions == cls).astype(float)  # One-hot fallback
                print(f"         Using one-hot encoding for class {cls}")
            
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
                print(f"        Skipping class {cls}: {e}")
                continue
        
        return curves_data
        
    except Exception as e:
        print(f"      Error computing curves: {e}")
        return {'roc_curves': [], 'prc_curves': [], 'class_names': [], 'auroc_scores': [], 'auprc_scores': []}


def compute_simple_metrics(predicted_clusters: np.ndarray, target_clusters: np.ndarray, prediction_scores: np.ndarray = None, eval_class_ids: np.ndarray = None):
    """Compute memory-efficient classification metrics."""
    print("📊 Computing classification metrics (memory-efficient)...")
    
    # Filter out invalid targets
    valid_mask = target_clusters >= 0
    valid_predictions = predicted_clusters[valid_mask]
    valid_targets = target_clusters[valid_mask]
    
    print(f"   Valid samples: {len(valid_targets):,} / {len(target_clusters):,}")
    
    if len(valid_targets) == 0:
        return {}
    
    # Basic accuracy
    correct = (valid_predictions == valid_targets).sum()
    accuracy = correct / len(valid_targets)
    
    # Count unique classes
    unique_targets = np.unique(valid_targets)
    unique_predictions = np.unique(valid_predictions)
    
    print(f"   Classes in targets: {len(unique_targets):,}")
    print(f"   Classes in predictions: {len(unique_predictions):,}")
    
    # Compute top-k accuracy - use real per-sample top-k if scores available
    if prediction_scores is not None and eval_class_ids is not None:
        print("   Computing real per-sample top-K accuracy...")
        
        # Create mapping from original class IDs to local indices
        class_id_to_col = {cls_id: i for i, cls_id in enumerate(eval_class_ids)}
        
        # Filter samples where target is in evaluated classes
        valid_eval_mask = np.array([target in class_id_to_col for target in valid_targets])
        eval_targets = valid_targets[valid_eval_mask]
        eval_scores = valid_scores[valid_eval_mask] if valid_scores is not None else None
        
        if len(eval_targets) > 0 and eval_scores is not None:
            # Map targets to local column indices  
            eval_targets_mapped = np.array([class_id_to_col[target] for target in eval_targets])
            
            # Real per-sample top-K accuracy
            topk_values = [1, 5, 10]
            topk_accs = {}
            
            for k in topk_values:
                if k <= eval_scores.shape[1]:
                    # Get top-k predictions per sample
                    topk_idx = np.argpartition(-eval_scores, k-1, axis=1)[:, :k]
                    # Check if target is in top-k
                    hit = (topk_idx == eval_targets_mapped[:, None]).any(axis=1)
                    topk_accs[f'top_{k}'] = float(hit.mean())
                else:
                    topk_accs[f'top_{k}'] = float((eval_targets_mapped >= 0).mean())  # All valid
            
            print(f"      Evaluated {len(eval_targets):,}/{len(valid_targets):,} samples in top-{len(eval_class_ids)} classes")
            top5_accuracy = topk_accs.get('top_5', 0.0)
            top10_accuracy = topk_accs.get('top_10', 0.0)
        else:
            print("      No samples found in evaluated classes for top-K")
            top5_accuracy = 0.0
            top10_accuracy = 0.0
    else:
        # Fallback: Global top-k (previous method)
        print("   Using global top-K accuracy (no scores available)...")
        prediction_counts = Counter(valid_predictions)
        most_common_preds = [cls for cls, _ in prediction_counts.most_common()]
        
        top5_classes = set(most_common_preds[:min(5, len(most_common_preds))])
        top5_correct = sum(1 for target in valid_targets if target in top5_classes)
        top5_accuracy = top5_correct / len(valid_targets)
        
        top10_classes = set(most_common_preds[:min(10, len(most_common_preds))])
        top10_correct = sum(1 for target in valid_targets if target in top10_classes)
        top10_accuracy = top10_correct / len(valid_targets)
    
    # Analyze prediction distribution
    target_counts = Counter(valid_targets)
    pred_counts = Counter(valid_predictions)
    
    # Coverage: how many target classes does the model predict?
    covered_classes = len(set(valid_targets) & set(valid_predictions))
    coverage_ratio = covered_classes / len(unique_targets)
    
    # Concentration: how concentrated are the predictions?
    total_preds = len(valid_predictions)
    top_10_pred_ratio = sum(pred_counts[cls] for cls in most_common_preds[:10]) / total_preds
    
    # Compute AUROC and AUPRC using the SAME method as ClassificationEvaluator
    print("   Computing AUROC/AUPRC (aligned with ClassificationEvaluator)...")
    
    # Filter scores to match valid samples if available
    valid_scores = None
    if prediction_scores is not None:
        valid_scores = prediction_scores[valid_mask]
        print(f"   Using actual prediction scores: {valid_scores.shape}")
    
    auroc_macro, auprc_macro = compute_auroc_auprc_aligned(
        valid_targets, valid_predictions, valid_scores, eval_class_ids, max_classes=100, max_samples=50000
    )
    
    # Compute ROC and PRC curves for visualization
    print("   Computing ROC/PRC curves for top classes...")
    curves_data = compute_roc_prc_curves(valid_targets, valid_predictions, valid_scores, max_classes=5, max_samples=10000)
    
    metrics = {
        'accuracy': float(accuracy),
        'top_5_accuracy': float(top5_accuracy),
        'top_10_accuracy': float(top10_accuracy),
        'auroc_macro': float(auroc_macro),
        'auprc_macro': float(auprc_macro),
        'num_classes_in_targets': int(len(unique_targets)),
        'num_classes_predicted': int(len(unique_predictions)),
        'total_samples': int(len(valid_targets)),
        'total_raw_samples': int(len(target_clusters)),
        'valid_sample_ratio': float(len(valid_targets) / len(target_clusters)),
        'coverage_ratio': float(coverage_ratio),
        'top_10_prediction_concentration': float(top_10_pred_ratio),
        'correct_predictions': int(correct)
    }
    
    print(f"✅ Classification metrics computed:")
    print(f"   Accuracy: {accuracy:.4f} ({correct:,}/{len(valid_targets):,})")
    print(f"   Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"   Top-10 Accuracy: {top10_accuracy:.4f}")
    print(f"   AUROC (macro): {auroc_macro:.4f}")
    print(f"   AUPRC (macro): {auprc_macro:.4f}")
    print(f"   Coverage: {coverage_ratio:.4f} ({covered_classes:,}/{len(unique_targets):,} classes)")
    print(f"   Prediction concentration: {top_10_pred_ratio:.4f} (top 10 classes)")
    
    return metrics, curves_data


def update_evaluation_results(results_dir: Path, cluster_metrics: dict):
    """Update the evaluation results with cluster metrics."""
    results_file = results_dir / 'evaluation_results.json'
    
    # Load existing results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Update metrics
    results['metrics'].update(cluster_metrics)
    results['metrics']['cluster_conversion_skipped'] = False
    results['metrics']['cluster_conversion_completed'] = True
    results['metrics']['memory_efficient_computation'] = True
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Updated evaluation results: {results_file}")


def create_roc_prc_visualization(results_dir: Path, metrics: dict, curves_data: dict):
    """Create visualization with ROC and PRC curves."""
    print("📈 Creating ROC and PRC curve visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Performance with ROC/PRC Curves: {results_dir.name}', fontsize=16, fontweight='bold')
    
    # ROC Curves (Top-Left)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#845EC2']
    
    if curves_data['roc_curves']:
        for i, (roc_data, class_name, auroc) in enumerate(zip(
            curves_data['roc_curves'], 
            curves_data['class_names'], 
            curves_data['auroc_scores']
        )):
            if i < len(colors):
                axes[0, 0].plot(roc_data['fpr'], roc_data['tpr'], 
                               color=colors[i], linewidth=2,
                               label=f'{class_name} (AUC = {auroc:.3f})')
    
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves')
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # PRC Curves (Top-Right)
    if curves_data['prc_curves']:
        for i, (prc_data, class_name, auprc) in enumerate(zip(
            curves_data['prc_curves'], 
            curves_data['class_names'], 
            curves_data['auprc_scores']
        )):
            if i < len(colors):
                axes[0, 1].plot(prc_data['recall'], prc_data['precision'], 
                               color=colors[i], linewidth=2,
                               label=f'{class_name} (AUC = {auprc:.3f})')
    
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision') 
    axes[0, 1].set_title('Precision-Recall Curves')
    axes[0, 1].legend(loc='lower left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coverage and model behavior
    behavior_values = [metrics.get('coverage_ratio', 0), metrics.get('top_10_prediction_concentration', 0)]
    behavior_labels = ['Class\nCoverage', 'Top-10\nConcentration']
    
    axes[1, 0].bar(behavior_labels, behavior_values, color=['#2E86AB', '#F18F01'])
    axes[1, 0].set_title('Model Behavior Metrics')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(behavior_values):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Model information  
    axes[1, 1].text(0.1, 0.9, f"Model Type: Regression→Classification", fontsize=12, transform=axes[1, 1].transAxes, weight='bold')
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
    plt.savefig(results_dir / 'performance_summary_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Final visualization saved: {results_dir}/performance_summary_final.png")


def main():
    parser = argparse.ArgumentParser(description='Compute simple classification metrics from cluster results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing cluster results')
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    # Load cluster results
    predicted_clusters, target_clusters = load_cluster_results(results_dir)
    
    # Compute metrics
    metrics, curves_data = compute_simple_metrics(predicted_clusters, target_clusters)
    
    # Update evaluation results
    update_evaluation_results(results_dir, metrics)
    
    # Create visualization with ROC/PRC curves
    create_roc_prc_visualization(results_dir, metrics, curves_data)
    
    print("🎉 Simple cluster metrics computation completed!")
    print(f"📊 Key Results:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   Class Coverage: {metrics.get('coverage_ratio', 0):.4f}")
    print(f"   Prediction Diversity: {metrics.get('num_classes_predicted', 0):,} classes")


if __name__ == '__main__':
    main()
