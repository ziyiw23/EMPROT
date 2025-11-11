import os
import csv
import math
import pickle
import tempfile

import lmdb
import numpy as np
import torch

from emprot.data.dataset import create_dataloaders
from emprot.training.trainer import EMPROTTrainer
from emprot.losses import masked_cross_entropy
from emprot.utils.metrics import compute_classification_metrics


def _make_mock_lmdb(root: str, name: str, num_frames: int, num_residues: int) -> str:
    traj_dir = os.path.join(root, name)
    os.makedirs(traj_dir, exist_ok=True)
    env = lmdb.open(traj_dir, map_size=int(4e6), subdir=True, writemap=True, meminit=False)
    try:
        with env.begin(write=True) as txn:
            rng = np.random.default_rng(0)
            for i in range(num_frames):
                cid = rng.integers(low=1, high=20, size=(num_residues,), dtype=np.int64)
                frame = {'cluster_ids': cid}
                txn.put(str(i).encode(), pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))
            id_to_idx = {i: str(i) for i in range(num_frames)}
            txn.put(b'id_to_idx', pickle.dumps(id_to_idx, protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        env.close()
    return traj_dir


def _write_metadata_csv(path: str, dynamic_id: str = "123", uniprot: str = "P01234", pdb: str = "1ABC") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dynamic id", "Uniprot ID", "PDB ID"])  # header
        w.writerow([dynamic_id, uniprot, pdb])


def _build_small_loaders(tmpdir: str, K: int = 3, F: int = 2, stride: int = 1):
    data_root = os.path.join(tmpdir, "lmdb_root")
    os.makedirs(data_root, exist_ok=True)
    traj_name = "prot_dyn_123_traj"
    # For K=3, F=2, stride=1, 6 frames are sufficient
    _make_mock_lmdb(data_root, traj_name, num_frames=max((K + F) * stride, 6), num_residues=4)
    meta_csv = os.path.join(tmpdir, "traj_metadata.csv")
    _write_metadata_csv(meta_csv, dynamic_id="123")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_root,
        metadata_path=meta_csv,
        batch_size=2,
        num_full_res_frames=K,
        stride=stride,
        future_horizon=F,
        num_workers=0,
        train_split=1.0,
        val_split=0.0,
        seed=7,
    )
    return train_loader, val_loader, K, F


def test_trainer_end_to_end_and_wandb_logging(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        train_loader, val_loader, K, F = _build_small_loaders(tmpdir, K=3, F=2, stride=1)

        # Inspect a batch tensors
        batch = next(iter(train_loader))
        assert set(['input_cluster_ids', 'times', 'sequence_lengths', 'history_mask', 'residue_mask', 'future_cluster_ids', 'future_step_mask']).issubset(batch.keys())
        B, T, N = batch['input_cluster_ids'].shape
        assert T == K
        assert batch['future_cluster_ids'].shape[1] == F

        # Mock wandb.log to capture payload
        logged = {}
        try:
            import wandb  # type: ignore
            def _mock_log(payload, step=None):
                logged.clear()
                logged.update(payload)
                if step is not None:
                    logged['step'] = int(step)
            monkeypatch.setattr(wandb, 'log', _mock_log, raising=False)
        except Exception:
            pass

        # Build a compact config
        max_input = batch['input_cluster_ids'].clamp_min(0).max()
        max_future = batch['future_cluster_ids'].clamp_min(0).max()
        C = int(torch.max(max_input, max_future).item()) + 1
        cfg = {
            'd_embed': 16,
            'num_heads': 4,
            'num_layers': 1,
            'num_clusters': max(C, 20),
            'future_horizon': F,
            'learning_rate': 1e-3,
            'weight_decay': 1e-2,
            'use_scheduler': False,
            'use_amp': False,
            'max_epochs': 1,
            'patience': 1,
            'checkpoint_dir': os.path.join(tmpdir, 'ckpts'),
            'scheduled_sampling_p': 0.0,
        }
        trainer = EMPROTTrainer(cfg)
        best = trainer.train(train_loader, val_loader)

        # Check best loss and checkpoints
        assert math.isfinite(float(best))
        assert os.path.exists(os.path.join(cfg['checkpoint_dir'], 'final.pt'))

        # Verify wandb payload fields
        assert 'epoch' in logged and 'train/loss' in logged and 'val/loss' in logged
        # Expected metric keys
        for k in ['acc_f1', 'top5_f1', 'mtp_f1', 'entropy_f1', 'ece_f1', 'acc_change_f1', 'acc_stay_f1']:
            assert f'val/{k}' in logged or k in logged


def test_masked_ce_and_metrics_small_known():
    # Small, deterministic logits/targets
    # B=1, F=2, N=3, C=3
    logits = torch.tensor([
        [
            # step 0
            [[5.0, 1.0, 0.0],  # r0 → pred 0
             [0.0, 5.0, 0.0],  # r1 → pred 1
             [0.0, 1.0, 5.0]], # r2 → pred 2
            # step 1
            [[1.0, 0.0, 5.0],  # r0 → pred 2
             [5.0, 0.0, 1.0],  # r1 → pred 0
             [0.0, 5.0, 1.0]]  # r2 → pred 1
        ]
    ], dtype=torch.float32)
    targets = torch.tensor([[[0, 1, 2], [2, 0, 1]]], dtype=torch.long)
    future_step_mask = torch.tensor([[True, False]])  # only step0 valid
    residue_mask = torch.tensor([[True, True, True]])

    # Loss: Only step 0 counts; confident correct predictions → small
    loss = masked_cross_entropy(
        logits, targets,
        future_step_mask=future_step_mask,
        residue_mask=residue_mask,
    )
    assert loss.item() < 0.05

    # Metrics: on step0, accuracy 1.0; top5=1.0; mtp close to 1.0
    m = compute_classification_metrics(
        logits, targets,
        input_cluster_ids=None,
        future_step_mask=future_step_mask,
        residue_mask=residue_mask,
        compute_ece=True,
    )
    assert isinstance(m['acc_per_horizon'], torch.Tensor)
    assert abs(m['acc_f1'] - 1.0) < 1e-6
    assert abs(m['top5_f1'] - 1.0) < 1e-6
    assert m['mtp_f1'] > 0.9
    # ece is non-negative
    assert m['ece_f1'] >= 0.0
