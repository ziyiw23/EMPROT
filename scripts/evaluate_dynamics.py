import numpy as np
from typing import Dict, Tuple, List

def compute_dwell_times(traj: np.ndarray) -> np.ndarray:
    """
    Compute the sequence of dwell times for a trajectory.
    traj: (T,) integer array
    Returns: array of dwell lengths (e.g. [3, 1, 5] for A,A,A,B,C,C,C,C,C)
    """
    if len(traj) == 0:
        return np.array([])
    
    # Find indices where value changes
    # diff != 0 gives boolean mask of changes
    # We prepend True to count the first segment
    changes = np.concatenate(([True], traj[1:] != traj[:-1], [True]))
    
    # The indices of changes
    change_indices = np.nonzero(changes)[0]
    
    # Dwell times are the differences between change indices
    dwells = np.diff(change_indices)
    return dwells

def compute_change_rate(traj: np.ndarray) -> float:
    """Fraction of steps that are different from the previous step."""
    if len(traj) < 2:
        return 0.0
    changes = np.sum(traj[1:] != traj[:-1])
    return float(changes) / (len(traj) - 1)

def compute_ngram_overlap(gt: np.ndarray, pred: np.ndarray, n: int = 2) -> float:
    """
    Compute Jaccard similarity of N-grams (transitions) between GT and Pred.
    """
    def get_ngrams(seq):
        return set(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1))
    
    gt_ngrams = get_ngrams(gt)
    pred_ngrams = get_ngrams(pred)
    
    if not gt_ngrams and not pred_ngrams:
        return 1.0
    
    intersection = len(gt_ngrams.intersection(pred_ngrams))
    union = len(gt_ngrams.union(pred_ngrams))
    
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(s1, s2):
    """Simple edit distance for integer sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def evaluate_trajectory_dynamics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute dynamic metrics for a single residue trajectory.
    gt, pred: (T,) integer arrays
    """
    # 1. Dwell Time Distribution Matching (Wasserstein/EMD approx via means)
    gt_dwells = compute_dwell_times(gt)
    pred_dwells = compute_dwell_times(pred)
    
    dwell_mean_diff = abs(np.mean(gt_dwells) - np.mean(pred_dwells)) if len(pred_dwells) > 0 else 0.0
    
    # 2. Change Rate Error
    gt_change = compute_change_rate(gt)
    pred_change = compute_change_rate(pred)
    change_rate_error = abs(gt_change - pred_change)
    
    # 3. Transition Overlap (Bigram Jaccard)
    transition_overlap = compute_ngram_overlap(gt, pred, n=2)
    
    # 4. Edit Distance (Normalized)
    edit_dist = levenshtein_distance(gt, pred)
    norm_edit_dist = edit_dist / max(len(gt), 1)
    
    return {
        "dwell_mean_diff": dwell_mean_diff,
        "change_rate_error": change_rate_error,
        "transition_overlap": transition_overlap,
        "norm_edit_dist": norm_edit_dist,
        "gt_change_rate": gt_change,
        "pred_change_rate": pred_change
    }

def compute_correlation_error(gt_batch: np.ndarray, pred_batch: np.ndarray) -> float:
    """
    Compute Frobenius norm of difference between GT and Pred change-event correlation matrices.
    Returns normalized error (Frobenius norm / N).
    """
    T, N = gt_batch.shape
    if T < 2 or N < 2:
        return 0.0

    # 1. Identify change events (binary mask): 1 if state changed, 0 if stay
    # shape (T-1, N)
    gt_changes = (gt_batch[1:] != gt_batch[:-1]).astype(float)
    pred_changes = (pred_batch[1:] != pred_batch[:-1]).astype(float)
    
    def safe_corr(X):
        # X: (T, N)
        # Centered
        X_centered = X - X.mean(axis=0)
        # Covariance: (N, N)
        cov = X_centered.T @ X_centered / (max(1, X.shape[0] - 1))
        # Std: (N,)
        std = np.sqrt(np.diag(cov))
        # Outer product of std
        std_outer = np.outer(std, std)
        # Correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / std_outer
        # Replace NaNs (const columns) with 0 correlation
        corr[~np.isfinite(corr)] = 0.0
        return corr

    gt_corr = safe_corr(gt_changes)
    pred_corr = safe_corr(pred_changes)
    
    # 3. Frobenius distance normalized by size
    diff = gt_corr - pred_corr
    return float(np.linalg.norm(diff) / N)

def evaluate_batch_dynamics(gt_batch: np.ndarray, pred_batch: np.ndarray) -> Dict[str, float]:
    """
    Aggregate metrics over a batch of residues.
    gt_batch, pred_batch: (T, N) arrays
    """
    T, N = gt_batch.shape
    metrics_sum = {
        "dwell_mean_diff": 0.0,
        "change_rate_error": 0.0,
        "transition_overlap": 0.0,
        "norm_edit_dist": 0.0,
        "gt_change_rate": 0.0,
        "pred_change_rate": 0.0
    }
    
    # Compute correlation error (global metric)
    corr_error = compute_correlation_error(gt_batch, pred_batch)
    
    valid_count = 0
    for i in range(N):
        # Skip if GT is all padding (-1)
        if np.all(gt_batch[:, i] < 0):
            continue
            
        # Mask out padding
        mask = gt_batch[:, i] >= 0
        g = gt_batch[mask, i]
        p = pred_batch[mask, i]
        
        m = evaluate_trajectory_dynamics(g, p)
        for k, v in m.items():
            metrics_sum[k] += v
        valid_count += 1
        
    if valid_count == 0:
        metrics_sum["correlation_error"] = corr_error
        return metrics_sum
        
    avg_metrics = {k: v / valid_count for k, v in metrics_sum.items()}
    avg_metrics["correlation_error"] = corr_error
    return avg_metrics

