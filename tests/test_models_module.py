import os
import torch
import sys
# Usage: python -m pytest -q tests/test_models_module.py

from emprot.models.transformer import ProteinTransformerClassificationOnly
from emprot.data import create_dataloaders

sys.path.append("/oak/stanford/groups/rbaltman/ziyiw23/EMPROT")
def _get_real_batch():
    data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings"
    meta_csv = "/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/traj_metadata.csv"
    K = 3
    F = 5
    stride = 1
    train_loader, _, _ = create_dataloaders(
        data_dir=data_dir,
        metadata_path=meta_csv,
        batch_size=4,
        num_full_res_frames=K,
        stride=stride,
        future_horizon=F,
        num_workers=0,
        seed=123,
    )
    return next(iter(train_loader))


def _make_model_from_batch(batch, d_model=64, num_layers=2):
    B, T, N = batch['input_cluster_ids'].shape
    F = batch['future_cluster_ids'].shape[1]
    max_input = batch['input_cluster_ids'].clamp_min(0).max()
    max_future = batch['future_cluster_ids'].clamp_min(0).max()
    C = int(torch.max(max_input, max_future).item()) + 1
    num_heads = 8 if d_model % 8 == 0 else 4
    model = ProteinTransformerClassificationOnly(
        d_embed=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_clusters=C,
        future_horizon=F,
        use_gradient_checkpointing=False,
    )
    return model, C, F, N


def test_forward_shapes_and_logits_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=64, num_layers=2)
    out = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
    )
    logits = out['cluster_logits']
    assert logits.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    assert torch.isfinite(logits).all()
    ctx = out['context']
    assert ctx.shape[0] == batch['input_cluster_ids'].shape[0] and ctx.shape[1] == N
    # Prints for inspection
    print("[forward] logits shape=", tuple(logits.shape))
    print("[forward] logits stats:", float(logits.min().item()), float(logits.mean().item()), float(logits.max().item()))
    b0 = 0
    valid_res = batch.get('residue_mask', torch.ones(logits.size(0), N, dtype=torch.bool))[b0].nonzero(as_tuple=False).squeeze(-1)
    if valid_res.numel() > 0:
        r0 = int(valid_res[0].item())
        top0 = torch.topk(logits[b0, 0, r0], k=min(5, C)).indices.tolist()
        topL = torch.topk(logits[b0, -1, r0], k=min(5, C)).indices.tolist()
        print(f"[forward] sample0 residue{r0} top-5 @step0:", top0)
        print(f"[forward] sample0 residue{r0} top-5 @step{F-1}:", topL)


def test_padding_and_sequence_lengths_masking_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=32, num_layers=1)
    out = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
    )
    logits = out['cluster_logits']
    assert logits.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    assert torch.isfinite(logits).all()
    print("[masking] logits shape=", tuple(logits.shape))
    print("[masking] logits stats:", float(logits.min().item()), float(logits.mean().item()), float(logits.max().item()))


def test_state_roundtrip_update_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=64, num_layers=2)
    out1 = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
    )
    state = out1.get('new_state')
    out2 = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
        state=state,
    )
    logits2 = out2['cluster_logits']
    assert logits2.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    print("[state] logits2 shape=", tuple(logits2.shape))
    print("[state] logits2 stats:", float(logits2.min().item()), float(logits2.mean().item()), float(logits2.max().item()))


def test_infer_autoregressive_helper_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=32, num_layers=1)
    out = model.infer_autoregressive_step(
        history_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        history_mask=batch['history_mask'],
        sequence_lengths=batch['sequence_lengths'],
    )
    logits = out['cluster_logits']
    assert logits.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    print("[infer_step] logits shape=", tuple(logits.shape))
    print("[infer_step] logits stats:", float(logits.min().item()), float(logits.mean().item()), float(logits.max().item()))


def test_direct_multi_horizon_teacher_forced_parallel_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=48, num_layers=1)
    out = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
        teacher_future_ids=batch['future_cluster_ids'],
        scheduled_sampling_p=0.0,
    )
    logits = out['cluster_logits']
    assert logits.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    print("[parallel TF] logits shape=", tuple(logits.shape))
    print("[parallel TF] logits stats:", float(logits.min().item()), float(logits.mean().item()), float(logits.max().item()))


def test_autoregressive_unroll_with_scheduled_sampling_real_data():
    batch = _get_real_batch()
    model, C, F, N = _make_model_from_batch(batch, d_model=48, num_layers=1)
    out = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=batch['sequence_lengths'],
        history_mask=batch['history_mask'],
        teacher_future_ids=batch['future_cluster_ids'],
        scheduled_sampling_p=0.25,
    )
    logits = out['cluster_logits']
    assert logits.shape == (batch['input_cluster_ids'].shape[0], F, N, C)
    print("[AR SS] logits shape=", tuple(logits.shape))
    print("[AR SS] logits stats:", float(logits.min().item()), float(logits.mean().item()), float(logits.max().item()))

