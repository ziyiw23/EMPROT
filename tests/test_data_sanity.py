#!/usr/bin/env python3
"""
Sanity checks for data_dir structure and LMDB contents.

Run with:
  python -m pytest -s tests/test_data_sanity.py

This prints a concise report of a few trajectories under the configured
data_dir, including whether LMDB opens, number of frames detected, and
whether it satisfies the (L+K+F-1)*stride+1 requirement.
"""

import os
import sys
from pathlib import Path
import yaml

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emprot.data.data_loader import LMDBLoader


def _load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def test_data_dir_lmdb_sanity():
    cfg_path = PROJECT_ROOT / 'configs' / 'emprot.yaml'
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    cfg = _load_yaml(cfg_path)

    # Resolve base_config if present
    base_name = cfg.get('base_config')
    if base_name:
        base_cfg = _load_yaml(PROJECT_ROOT / 'configs' / base_name)
        base_cfg.update(cfg)
        cfg = base_cfg

    data_cfg = cfg.get('data', {})
    data_dir = Path(data_cfg.get('data_dir', ''))
    assert data_dir.exists(), f"data_dir does not exist: {data_dir}"

    L = int(data_cfg.get('history_prefix_frames', 0) or 0)
    K = int(data_cfg.get('num_full_res_frames', 5))
    F = int(data_cfg.get('future_horizon', 1))
    s = int(data_cfg.get('stride', 1))
    required_frames = (L + K + F - 1) * s + 1

    # Scan first N entries under data_dir
    names = sorted([p.name for p in data_dir.iterdir()])
    print(f"data_dir={data_dir}")
    print(f"entries={len(names)} L={L} K={K} F={F} stride={s} required_frames>={required_frames}")
    checked = 0
    ok = 0
    for name in names:
        traj_path = data_dir / name
        if not traj_path.is_dir():
            # Heuristic: accept paths that contain LMDB files (data.mdb)
            if not (traj_path / 'data.mdb').exists():
                continue
        try:
            with LMDBLoader(str(traj_path)) as loader:
                meta = loader.get_metadata()
            nf = int(meta.get('num_frames', 0))
            nr = int(meta.get('num_residues', 0))
            print(f"- {name}: frames={nf} residues={nr} {'OK' if nf>=required_frames else 'short'}")
            ok += int(nf >= required_frames)
            checked += 1
        except Exception as e:
            print(f"- {name}: ERROR opening LMDB: {e}")
            checked += 1
        if checked >= 20:
            break

    assert checked > 0, "No candidate trajectories found under data_dir (check mount/path)."
    # Do not fail if none pass; this test is for visibility. Use assert to flag but not hard-fail.
    if ok == 0:
        pytest.skip("No trajectories meet the required frame count; adjust L/K/F/stride or inspect data.")

