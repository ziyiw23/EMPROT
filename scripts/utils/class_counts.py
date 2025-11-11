import torch
from collections import Counter
from typing import Dict, Optional


def build_class_counts_from_train(train_loader,
                                  id2col: Dict[int, int],
                                  device: str = 'cpu',
                                  max_batches: Optional[int] = None) -> torch.Tensor:
    """
    Returns a LongTensor [C] with class counts in column-index space.
    Unseen classes get count=0. Counts are clamped to at least 1 for stability.

    Args:
        train_loader: DataLoader over training data yielding batches with targets
        id2col: Mapping from raw class id -> column index (0..C-1)
        device: Unused (kept for API symmetry); computation occurs on CPU
        max_batches: Optional cap on number of batches to scan
    """
    if not isinstance(id2col, dict) or len(id2col) == 0:
        raise ValueError("id2col mapping must be a non-empty dict")

    C = max(id2col.values()) + 1
    counts = torch.zeros(C, dtype=torch.long)
    seen = Counter()

    for i, batch in enumerate(train_loader):
        # Targets may be nested under 'targets'
        tdict = batch.get('targets', {}) if isinstance(batch, dict) else {}
        if isinstance(tdict, dict) and 'target_cluster_ids' in tdict:
            targets_raw = tdict['target_cluster_ids']
        elif isinstance(batch, dict) and 'target_cluster_ids' in batch:
            targets_raw = batch['target_cluster_ids']
        elif isinstance(tdict, dict) and 'cluster_ids' in tdict:
            targets_raw = tdict['cluster_ids']
        else:
            targets_raw = None

        if targets_raw is None:
            continue  # no labels in this batch

        flat = targets_raw.view(-1).cpu().tolist()
        for rid in flat:
            if rid is None or int(rid) < 0:
                continue
            col = id2col.get(int(rid), None)
            if col is not None:
                seen[col] += 1

        if max_batches is not None and (i + 1) >= max_batches:
            break

    for col, n in seen.items():
        if 0 <= col < C:
            counts[col] = n

    # safety: avoid zeros in denominators later
    counts = counts.clamp_min(1)
    return counts

