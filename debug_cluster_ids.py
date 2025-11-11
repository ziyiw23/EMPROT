#!/usr/bin/env python3
"""
Debug script to diagnose cluster ID loading issues in the dual-head training system.
This will help identify exactly where the problem is occurring.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from emprot.data.data_loader import LMDBLoader
    from emprot.data.dataset import ProteinTrajectoryDataset
    from emprot.data.cluster_lookup import create_cluster_lookup_from_sklearn
    from emprot.losses.dual_head_loss import DualHeadMultiTaskLoss
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def debug_lmdb_structure(data_dir: str, max_trajectories: int = 3):
    """Debug the LMDB structure to see what's actually stored."""
    print("🔍 DEBUGGING LMDB STRUCTURE")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    trajectories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    trajectories.sort()
    
    print(f"📁 Found {len(trajectories)} trajectory directories")
    
    for i, traj_name in enumerate(trajectories[:max_trajectories]):
        traj_path = os.path.join(data_dir, traj_name)
        print(f"\n{i+1}. {traj_name}")
        
        try:
            with LMDBLoader(traj_path, read_only=True) as loader:
                metadata = loader.get_metadata()
                num_frames = metadata['num_frames']
                print(f"   📊 Total frames: {num_frames}")
                
                # Check first few frames
                for frame_idx in [0, 1, 10, 100]:
                    if frame_idx < num_frames:
                        try:
                            frame_data = loader.load_frame(frame_idx)
                            print(f"   Frame {frame_idx}:")
                            print(f"     Keys: {list(frame_data.keys())}")
                            
                            if 'cluster_ids' in frame_data:
                                cluster_ids = frame_data['cluster_ids']
                                print(f"     ✅ cluster_ids: shape={cluster_ids.shape}, dtype={cluster_ids.dtype}")
                                print(f"        Range: {cluster_ids.min()} to {cluster_ids.max()}")
                                print(f"        Sample: {cluster_ids[:5]}")
                            else:
                                print(f"     ❌ No cluster_ids key found")
                            
                            if 'embeddings' in frame_data:
                                embeddings = frame_data['embeddings']
                                print(f"     📊 embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
                            
                        except Exception as e:
                            print(f"     ⚠️  Error loading frame {frame_idx}: {e}")
                
        except Exception as e:
            print(f"   ❌ Error reading LMDB: {e}")

def debug_dataset_loading(data_dir: str, metadata_path: str, max_samples: int = 5):
    """Debug the dataset loading to see what gets passed to the loss function."""
    print("\n🔍 DEBUGGING DATASET LOADING")
    print("=" * 60)
    
    try:
        # Create dataset
        dataset = ProteinTrajectoryDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            max_sequence_length=5,
            min_sequence_length=3,
            stride=5
        )
        
        print(f"📊 Dataset created successfully: {len(dataset)} samples")
        
        # Test a few samples
        for i in range(min(max_samples, len(dataset))):
            print(f"\n--- Sample {i} ---")
            try:
                sample = dataset[i]
                print(f"   Sample keys: {list(sample.keys())}")
                
                if 'targets' in sample:
                    targets = sample['targets']
                    print(f"   Targets keys: {list(targets.keys())}")
                    
                    if 'cluster_ids' in targets:
                        cluster_ids = targets['cluster_ids']
                        print(f"   ✅ cluster_ids in targets: shape={cluster_ids.shape}, dtype={cluster_ids.dtype}")
                        print(f"      Range: {cluster_ids.min()} to {cluster_ids.max()}")
                    else:
                        print(f"   ❌ No cluster_ids in targets")
                        
                    if 'short_term' in targets:
                        short_term = targets['short_term']
                        print(f"   📊 short_term: shape={short_term.shape}, dtype={short_term.dtype}")
                else:
                    print(f"   ❌ No targets key in sample")
                
                if 'embeddings' in sample:
                    embeddings = sample['embeddings']
                    print(f"   📊 embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
                
            except Exception as e:
                print(f"   ❌ Error loading sample {i}: {e}")
                
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")

def debug_loss_function(cluster_model_path: str, max_samples: int = 3):
    """Debug the loss function to see if it can process cluster IDs correctly."""
    print("\n🔍 DEBUGGING LOSS FUNCTION")
    print("=" * 60)
    
    try:
        # Load cluster lookup
        cluster_lookup = create_cluster_lookup_from_sklearn(
            sklearn_model_path=cluster_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✅ Loaded cluster lookup: {cluster_lookup.num_clusters} clusters")
        
        # Create loss function
        loss_fn = DualHeadMultiTaskLoss(
            regression_weight=1.0,
            classification_weight=1.0,
            cluster_lookup=cluster_lookup
        )
        print(f"✅ Created DualHeadMultiTaskLoss")
        
        # Create dummy data to test loss computation
        B, N, E = 2, 100, 512  # Batch size, residues, embedding dim
        num_clusters = cluster_lookup.num_clusters
        
        # Mock predictions
        predictions = {
            'delta_embedding': torch.randn(B, N, E),
            'cluster_logits': torch.randn(B, N, num_clusters)
        }
        
        # Mock targets with cluster IDs
        targets = {
            'short_term': torch.randn(B, N, E),
            'cluster_ids': torch.randint(0, num_clusters, (B, N), dtype=torch.long)
        }
        
        # Mock batch - include all keys that the loss function expects
        batch = {
            'residue_mask': torch.ones(B, N, dtype=torch.bool),
            # --- ADD THESE MISSING KEYS ---
            'embeddings': torch.randn(B, 5, N, E),  # Mock 5 historical frames
            'sequence_lengths': torch.tensor([5] * B)
            # ----------------------------
        }
        
        print(f"   📊 Mock data shapes:")
        print(f"      predictions['delta_embedding']: {predictions['delta_embedding'].shape}")
        print(f"      predictions['cluster_logits']: {predictions['cluster_logits'].shape}")
        print(f"      targets['short_term']: {targets['short_term'].shape}")
        print(f"      targets['cluster_ids']: {targets['cluster_ids'].shape}")
        print(f"      batch['residue_mask']: {batch['residue_mask'].shape}")
        
        # Test loss computation
        try:
            loss, metrics = loss_fn.compute_loss(predictions, targets, batch)
            print(f"   ✅ Loss computation successful:")
            print(f"      Total loss: {loss.item():.6f}")
            print(f"      Metrics keys: {list(metrics.keys())}")
            
            if 'classification_loss' in metrics:
                print(f"      Classification loss: {metrics['classification_loss']:.6f}")
            if 'regression_loss' in metrics:
                print(f"      Regression loss: {metrics['regression_loss']:.6f}")
                
        except Exception as e:
            print(f"   ❌ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error setting up loss function: {e}")
        import traceback
        traceback.print_exc()

def debug_training_pipeline(data_dir: str, metadata_path: str, cluster_model_path: str):
    """Debug the entire training pipeline to see where cluster IDs get lost."""
    print("\n🔍 DEBUGGING TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        # Create dataset
        dataset = ProteinTrajectoryDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            max_sequence_length=5,
            min_sequence_length=3,
            stride=5
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        print(f"📊 Created dataloader with {len(dataloader)} batches")
        
        # Test first batch
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            print(f"\n--- Batch {batch_idx} ---")
            print(f"   Batch keys: {list(batch.keys())}")
            
            if 'targets' in batch:
                targets = batch['targets']
                print(f"   Targets type: {type(targets)}")
                
                if isinstance(targets, list):
                    print(f"   Targets is a list of length {len(targets)}")
                    for i, target in enumerate(targets):
                        if isinstance(target, dict):
                            print(f"     Target {i} keys: {list(target.keys())}")
                            if 'cluster_ids' in target:
                                cluster_ids = target['cluster_ids']
                                print(f"     ✅ Target {i} has cluster_ids: {cluster_ids.shape}")
                            else:
                                print(f"     ❌ Target {i} missing cluster_ids")
                elif isinstance(targets, dict):
                    print(f"   Targets is a dict with keys: {list(targets.keys())}")
                    if 'cluster_ids' in targets:
                        cluster_ids = targets['cluster_ids']
                        print(f"     ✅ Targets has cluster_ids: {cluster_ids.shape}")
                    else:
                        print(f"     ❌ Targets missing cluster_ids")
                else:
                    print(f"   Targets is unexpected type: {type(targets)}")
            else:
                print(f"   ❌ No targets key in batch")
            
            if 'embeddings' in batch:
                embeddings = batch['embeddings']
                print(f"   📊 embeddings: {embeddings.shape}")
            
            print(f"   Batch size: {len(batch.get('embeddings', []))}")
            
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function."""
    print("🐛 EMPROT Cluster ID Debugging Script")
    print("=" * 60)
    
    # Configuration
    data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings/"
    metadata_path = "./traj_metadata.csv"
    cluster_model_path = "/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl"
    
    # Check if paths exist
    paths_to_check = [
        (data_dir, "Data directory"),
        (metadata_path, "Metadata file"),
        (cluster_model_path, "Cluster model file")
    ]
    
    for path, description in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            print(f"❌ {description}: {path} (NOT FOUND)")
    
    print("\n" + "=" * 60)
    
    # Run debugging steps
    if os.path.exists(data_dir):
        debug_lmdb_structure(data_dir, max_trajectories=3)
        
        if os.path.exists(metadata_path):
            debug_dataset_loading(data_dir, metadata_path, max_samples=5)
            debug_training_pipeline(data_dir, metadata_path, cluster_model_path)
    
    if os.path.exists(cluster_model_path):
        debug_loss_function(cluster_model_path, max_samples=3)
    
    print("\n" + "=" * 60)
    print("🏁 Debugging completed!")
    print("\n📋 SUMMARY OF FINDINGS:")
    print("1. Check if LMDBs actually contain cluster_ids")
    print("2. Check if dataset is loading cluster_ids correctly")
    print("3. Check if loss function can process cluster_ids")
    print("4. Check if training pipeline preserves cluster_ids")

if __name__ == "__main__":
    main()
