#!/usr/bin/env python3
"""
Train the lookup table and projector offline to align with COLLAPSE embeddings.
This allows for fast, focused pre-training of the input layer before training the Transformer.
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emprot.data.dataset import create_dataloaders
from scripts.utils.config_parser import merge_config_with_args

log = logging.getLogger("emprot.align")

class AlignmentModel(nn.Module):
    def __init__(self, num_clusters, d_embed, pretrained_input_dim):
        super().__init__()
        self.cluster_embedding = nn.Embedding(num_clusters + 1, d_embed, padding_idx=0)
        self.input_projector = nn.Sequential(
            nn.Linear(pretrained_input_dim, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed)
        )
        
        # Init projector to near-identity if dimensions matched (optional, but good for stability)
        # For now, standard init is fine as we want to adapt to random lookup or vice versa.
        
    def forward(self, cluster_ids, dense_embeddings, mask=None):
        # cluster_ids: (B, T, N)
        # dense_embeddings: (B, T, N, D_in)
        
        # Lookup
        # Clamp ids to be safe
        max_idx = self.cluster_embedding.num_embeddings - 1
        safe_ids = cluster_ids.clamp(0, max_idx)
        lookup = self.cluster_embedding(safe_ids) # (B, T, N, D_model)
        
        # Project
        projected = self.input_projector(dense_embeddings) # (B, T, N, D_model)
        
        # MSE Loss
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            diff = (lookup - projected) * mask_float
            mse = (diff ** 2).sum() / (mask_float.sum() * lookup.shape[-1] + 1e-6)
        else:
            mse = F.mse_loss(lookup, projected)
            
        return mse

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    parser = argparse.ArgumentParser(description='Train Lookup Table Alignment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--config_dir', type=str, default='configs', help='Config directory (for merge_config_with_args)')
    parser.add_argument('--print_config', action='store_true', default=False, help='Print merged config and exit')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (can be large)')
    parser.add_argument('--save_path', type=str, default='output/checkpoints/aligned_lookup.pt', help='Where to save checkpoint')
    parser.add_argument('--save_every_steps', type=int, default=5000, help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    # Use existing config loader to handle dataset paths etc.
    args = merge_config_with_args(args)
    
    # Setup Data
    log.info("Loading data...")
    train_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        history_prefix_frames=0,
        num_full_res_frames=1, # We only need 1 frame to learn the mapping! Or small chunks.
        stride=1,
        future_horizon=0, # No future needed
        num_workers=4,
        train_only_proteins=getattr(args, 'train_only_proteins', None),
        max_train_proteins=getattr(args, 'max_train_proteins', None)
    )
    
    # Setup Model
    d_embed = int(args.d_embed)
    pretrained_dim = int(getattr(args, 'pretrained_input_dim', 512) or 512)
    num_clusters = int(getattr(args, 'num_clusters', 50000))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlignmentModel(num_clusters, d_embed, pretrained_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    log.info(f"Training alignment: {pretrained_dim} -> {d_embed} | Batch={args.batch_size}")
    
    # Training Loop
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0
        count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            ids = batch['input_cluster_ids'].to(device)
            mask = batch['history_mask'].to(device)
            
            if 'input_embeddings' not in batch:
                continue
                
            emb = batch['input_embeddings'].to(device)
            
            loss = model(ids, emb, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            global_step += 1
            
            pbar.set_postfix({'mse': f"{loss.item():.4f}"})
            
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                torch.save(model.state_dict(), args.save_path)
                log.info(f"[Step {global_step}] Saved checkpoint to {args.save_path}")

        avg_loss = total_loss / max(1, count)
        log.info(f"Epoch {epoch+1} done. Avg MSE: {avg_loss:.5f}")

        # Epoch-based checkpointing
        torch.save(model.state_dict(), args.save_path)
        log.info(f"[Epoch {epoch+1}] Saved checkpoint to {args.save_path}")

    # Final save
    torch.save(model.state_dict(), args.save_path)
    log.info("Training complete.")

if __name__ == '__main__':
    main()

