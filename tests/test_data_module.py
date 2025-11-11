import os
import torch
import pytest

from emprot.data import create_dataloaders
import sys

# Usage: python -m pytest -s tests/test_data_module.py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def test_end_to_end_fixed_windows_real_lmdb():
    data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings"
    meta_csv = "/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv"

    K = 3
    F = 5
    stride = 1

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        metadata_path=meta_csv,
        batch_size=10,
        num_full_res_frames=K,
        stride=stride,
        future_horizon=F,
        num_workers=0,
        seed=123,
    )

    print(f"data_dir={data_dir}")
    print(f"meta_csv={meta_csv}")
    print(f"train_size={len(train_loader.dataset)} val_size={len(val_loader.dataset)} test_size={len(test_loader.dataset)}")

    batch = next(iter(train_loader))
    print("batch keys:", sorted(list(batch.keys())))
    assert 'input_cluster_ids' in batch
    assert 'history_mask' in batch
    assert 'residue_mask' in batch
    assert 'future_cluster_ids' in batch
    assert 'future_step_mask' in batch

    x = batch['input_cluster_ids']
    m_hist = batch['history_mask']
    m_res = batch['residue_mask']
    y = batch['future_cluster_ids']
    m_future = batch['future_step_mask']

    print(f"input_cluster_ids shape={tuple(x.shape)}")
    print(f"future_cluster_ids shape={tuple(y.shape)}")
    print(f"history_mask shape={tuple(m_hist.shape)} residue_mask shape={tuple(m_res.shape)} future_step_mask shape={tuple(m_future.shape)}")

    if 'traj_name' in batch:
        print("traj_name:", batch['traj_name'])
    print("protein_indices:", batch['protein_indices'].tolist())
    print("start_frames:", batch['start_frames'].tolist())

    print("uniprot_ids:", batch['uniprot_ids'])
    print("pdb_ids:", batch['pdb_ids'])

    active = getattr(train_loader.dataset, 'active_protein_indices', [])
    for i in range(len(batch['protein_indices'])):
        gpid = int(batch['protein_indices'][i])
        try:
            lpid = active.index(gpid)
            m = train_loader.dataset.protein_metadata[lpid]
            print("csv_meta:", {
                'traj': m.get('traj_name'),
                'uniprot': m.get('uniprot_id'),
                'pdb': m.get('pdb_id'),
                'num_frames': m.get('num_frames'),
                'num_residues': m.get('num_residues'),
            })
        except Exception:
            pass

    # Print cluster ids for the first valid residue in sample 0
    b_idx = 0
    valid_res = batch['residue_mask'][b_idx].nonzero(as_tuple=False).squeeze(-1)
    if valid_res.numel() > 0:
        r_idx = int(valid_res[0].item())
        print("sample0_residue", r_idx, "input_ids", x[b_idx, :, r_idx].tolist())
        print("sample0_residue", r_idx, "future_ids", y[b_idx, :, r_idx].tolist())

    B, T, N = x.shape
    assert T == K
    assert y.shape[0] == B and y.shape[1] == F and y.shape[2] == N
    assert m_hist.shape == (B, T, N)
    assert m_res.shape == (B, N)
    assert m_future.shape == (B, F)

    assert m_hist.dtype == torch.bool
    assert m_res.dtype == torch.bool
    assert m_future.dtype == torch.bool

    assert batch['times'].shape == (B, T)
    assert 'delta_t' in batch and batch['delta_t'].shape == (B, T)


