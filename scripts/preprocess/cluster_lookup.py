#!/usr/bin/env python3
"""Lightweight centroid lookup used during preprocessing/evaluation.

Relocated from emprot.data to scripts/preprocess because it is not used by the
training dataloader in classification-only mode.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class ClusterCentroidLookup:
    def __init__(self, num_clusters: int, embedding_dim: int, device: str = 'cuda'):
        self.num_clusters = int(num_clusters)
        self.embedding_dim = int(embedding_dim)
        self.device = device
        self.centroids: Optional[torch.Tensor] = None  # (C, D)

    def load_centroids_from_sklearn(self, filepath: Union[str, Path]) -> None:
        import pickle
        p = str(filepath)
        logger.info(f"Loading centroids from: {p}")
        with open(p, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
        elif hasattr(model, 'centroids'):
            centers = model.centroids
        else:
            raise ValueError("Provided model does not contain centroids")
        centers = np.asarray(centers, dtype=np.float32)
        if centers.shape[1] != self.embedding_dim:
            self.embedding_dim = centers.shape[1]
        if centers.shape[0] != self.num_clusters:
            self.num_clusters = centers.shape[0]
        self.centroids = torch.from_numpy(centers).to(self.device)
        logger.info(f"Loaded centroids: {tuple(self.centroids.shape)}")

    def get_centroid(self, cluster_id: Union[int, torch.Tensor]) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("Centroids not loaded")
        return self.centroids[int(cluster_id)] if isinstance(cluster_id, int) else self.centroids[cluster_id]

    def batch_assign_to_clusters(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("Centroids not loaded")
        B = embeddings.shape[0]
        chunk = 4096
        out = torch.empty(B, dtype=torch.long, device=embeddings.device)
        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            dists = torch.cdist(embeddings[i:j], self.centroids)
            out[i:j] = dists.argmin(dim=1)
        return out


