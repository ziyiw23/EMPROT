#!/usr/bin/env python3
"""
Quick test of temporal stride fix
"""

import sys
sys.path.append('.')

try:
    from emprot.utils.dataset_temporal_fix import ProteinTrajectoryDatasetFixed
    print("✅ Successfully imported ProteinTrajectoryDatasetFixed")
    
    # Test creating the dataset
    dataset = ProteinTrajectoryDatasetFixed(
        data_dir="/scratch/groups/rbaltman/ziyiw23/traj_embeddings/",
        metadata_path="traj_metadata.csv",
        temporal_stride=5,
        sequence_length=5
    )
    print("✅ Successfully created ProteinTrajectoryDatasetFixed")
    print(f"   Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("✅ Successfully loaded a sample")
        print(f"   Embeddings shape: {sample['embeddings'].shape}")
        print(f"   Times: {sample['times'].numpy()}")
        print(f"   Temporal stride: {sample['temporal_stride']}")
        print(f"   Effective timestep: {sample['effective_timestep']:.1f}ns")
        
        # Check similarity improvement
        embeddings = sample['embeddings']
        if len(embeddings) >= 2:
            import torch
            frame_a = embeddings[0].flatten()
            frame_b = embeddings[1].flatten()
            cos_sim = torch.dot(frame_a, frame_b) / (
                torch.norm(frame_a) * torch.norm(frame_b)
            )
            print(f"   Frame 0-1 similarity: {cos_sim.item():.6f}")
            
            if cos_sim.item() < 0.98:
                print("🎯 SUCCESS: Temporal fix working! Similarity < 0.98")
            else:
                print("⚠️  Still high similarity, but improvement expected")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 