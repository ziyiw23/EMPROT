import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import BatchSampler


class DynamicClusterBatchSampler(BatchSampler):
    """
    BatchSampler that yields batches drawn from a single kinetic cluster at a time.

    Assumptions:
    - The dataset exposes:
        * protein_metadata: list of dicts with 'traj_name' entries
        * _epoch_indices: list of (pid, widx) tuples mapping dataset index -> protein id
    - A JSON file mapping traj_name -> cluster_id (int) exists.
    """

    def __init__(
        self,
        protein_clusters_path: str,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        path = Path(protein_clusters_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"protein_clusters.json not found at {path}")
        with open(path, "r") as f:
            cluster_map: Dict[str, int] = json.load(f)

        # Build buckets: cluster_id -> list of dataset indices
        self.buckets: Dict[int, List[int]] = {}
        epoch_indices = getattr(dataset, "_epoch_indices", None)
        protein_metadata = getattr(dataset, "protein_metadata", None)
        if epoch_indices is None or protein_metadata is None:
            raise AttributeError("Dataset must expose _epoch_indices and protein_metadata.")

        for idx, (pid, _) in enumerate(epoch_indices):
            traj_name = protein_metadata[pid].get("traj_name")
            if traj_name is None:
                continue
            if traj_name not in cluster_map:
                continue
            cid = int(cluster_map[traj_name])
            self.buckets.setdefault(cid, []).append(idx)

        if not self.buckets:
            raise RuntimeError("No dataset indices were assigned to clusters; check the mapping and dataset.")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle cluster order each epoch
        cluster_ids = list(self.buckets.keys())
        self.rng.shuffle(cluster_ids)
        for cid in cluster_ids:
            indices = self.buckets[cid][:]
            self.rng.shuffle(indices)
            # Yield batches within this cluster
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total

