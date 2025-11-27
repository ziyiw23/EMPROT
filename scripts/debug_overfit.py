
import torch
import os
from torch.utils.data import DataLoader
from emprot.data.dataset import create_dataloaders
from emprot.models.transformer import ProteinTransformerClassificationOnly
from emprot.losses.distributional import residue_centric_loss

def debug_training():
    # 1. Setup Config
    data_dir = "/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings"
    protein_id = "11697_dyn_181_traj_11693"
    
    print(f"DEBUG: Creating loader for {protein_id}...")
    train_loader, _, _ = create_dataloaders(
        data_dir=data_dir,
        metadata_path="./traj_metadata.csv",
        batch_size=2,
        num_full_res_frames=5,
        stride=5,
        future_horizon=3,
        train_only_proteins=[protein_id]
    )
    
    print("DEBUG: Fetching one batch...")
    batch = next(iter(train_loader))
    print(f"DEBUG: Batch keys: {batch.keys()}")
    print(f"DEBUG: Input shape: {batch['input_cluster_ids'].shape}")
    print(f"DEBUG: Future shape: {batch['future_cluster_ids'].shape}")
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device {device}")
    
    model = ProteinTransformerClassificationOnly(
        d_embed=256,
        num_heads=4,
        num_layers=2,
        future_horizon=3,
        recent_full_frames=5,
        num_clusters=50000
    ).to(device)
    
    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 3. Forward Pass
    print("DEBUG: Running Forward Pass...")
    outputs = model(
        input_cluster_ids=batch['input_cluster_ids'],
        times=batch['times'],
        sequence_lengths=None, # Let model infer or handle
        history_mask=None,     # Let model infer
        teacher_future_ids=batch['future_cluster_ids']
    )
    
    logits = outputs['cluster_logits']
    print(f"DEBUG: Logits shape: {logits.shape}")
    print(f"DEBUG: Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

    # 4. Compute Loss
    print("DEBUG: Computing Loss...")
    loss, debug_metrics = residue_centric_loss(
        logits=logits,
        future_ids=batch['future_cluster_ids'],
        num_samples=32,
        ce_weight=2.0,
        js_weight=0.1
    )
    print(f"DEBUG: Loss: {loss.item():.4f}")
    print(f"DEBUG: Metrics: {debug_metrics}")

    # 5. Backward Pass
    print("DEBUG: Running Backward Pass...")
    loss.backward()
    
    # 6. Check Gradients
    emb_grad = model.cluster_embedding.weight.grad
    head_grad = model.classification_head.net[3].weight.grad # Last linear layer
    
    if emb_grad is None:
        print("ERROR: Embedding gradient is None!")
    else:
        print(f"DEBUG: Embedding grad norm: {emb_grad.norm().item():.4f}")
        print(f"DEBUG: Embedding grad mean: {emb_grad.abs().mean().item():.4f}")

    if head_grad is None:
        print("ERROR: Head gradient is None!")
    else:
        print(f"DEBUG: Head grad norm: {head_grad.norm().item():.4f}")

    if emb_grad is not None and emb_grad.norm().item() == 0:
        print("CRITICAL: Embedding gradient is ZERO. Input IDs might be disconnected or masked out.")

if __name__ == "__main__":
    debug_training()

