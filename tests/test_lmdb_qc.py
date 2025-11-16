#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emprot.data.data_loader import LMDBLoader


@pytest.mark.slow
def test_lmdb_frames_decodable_and_embeddings_consistent():
    data_dir = Path("/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings")
    assert data_dir.exists(), f"LMDB data_dir does not exist: {data_dir}"

    traj_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    assert traj_dirs, f"No trajectory directories found under {data_dir}"

    bad = []

    for traj_path in tqdm(traj_dirs, desc="LMDB QC", unit="traj"):
        try:
            with LMDBLoader(str(traj_path), read_only=True) as loader:
                meta = loader.get_metadata()
                num_frames = int(meta.get("num_frames", 0))
                embed_dim = int(meta.get("embedding_dim", 0))

                for i in range(num_frames):
                    frame = loader.load_frame(i)
                    emb = frame.get("embeddings")
                    if emb is None:
                        raise ValueError(f"Frame {i} missing 'embeddings' key")
                    arr = np.asarray(emb)
                    if arr.ndim != 2:
                        raise ValueError(f"Frame {i} embeddings ndim={arr.ndim}, expected 2")
                    if arr.shape[1] != embed_dim:
                        raise ValueError(
                            f"Frame {i} embeddings dim={arr.shape[1]} "
                            f"!= metadata embedding_dim={embed_dim}"
                        )
        except Exception as e:
            bad.append((traj_path.name, str(e)))

    if bad:
        print(f"\nFound {len(bad)} bad LMDB trajectories under {data_dir}:")
        for name, err in bad:
            print(f"- {name}: {err}")
        pytest.fail(f"LMDB QC failed for {len(bad)} trajectories (see above).")


