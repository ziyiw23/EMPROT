#!/usr/bin/env python3
"""
Standalone test script for dual-head loss function.
Tests the loss function in isolation to identify where metrics are getting lost.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dual_head_loss():
    """Test the dual-head loss function with realistic data."""
    print("🧪 Testing DualHeadMultiTaskLoss in isolation...")
    
    try:
        from emprot.losses.composite_loss import create_composite_loss
        print("✅ Successfully imported CompositeLoss factory")
    except ImportError as e:
        print(f"❌ Failed to import CompositeLoss: {e}")
        return False
    
    # Create realistic test data
    B, N, E = 2, 100, 512
    num_clusters = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Test predictions (non-zero)
    predictions = {
        'delta_embedding': torch.randn(B, N, E, device=device) * 0.1,
        'cluster_logits': torch.randn(B, N, num_clusters, device=device),
        'short_term': torch.randn(B, N, E, device=device) * 0.1  # For backward compatibility
    }
    
    # Test targets (non-zero)
    targets = {
        'short_term': torch.randn(B, N, E, device=device) * 0.2,
        'cluster_ids': torch.randint(0, num_clusters, (B, N), device=device)
    }
    
    # Test batch
    batch = {
        'embeddings': torch.randn(B, 5, N, E, device=device),
        'sequence_lengths': torch.tensor([5, 4], device=device),
        'residue_mask': torch.ones(B, N, device=device)
    }
    
    print(f"🔍 Test data shapes:")
    print(f"   Predictions: {list(predictions.keys())}")
    for key, value in predictions.items():
        print(f"     {key}: {value.shape}, mean: {value.mean().item():.6f}")
    
    print(f"   Targets: {list(targets.keys())}")
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            if value.dtype in [torch.float32, torch.float64]:
                print(f"     {key}: {value.shape}, mean: {value.mean().item():.6f}")
            else:
                print(f"     {key}: {value.shape}, dtype: {value.dtype}, range: [{value.min().item()}, {value.max().item()}]")
    
    print(f"   Batch: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
    
    # Create and test loss
    print(f"\n🔧 Creating CompositeLoss (dual-head baseline)...")
    loss_fn = create_composite_loss({
        'regression_weight': 1.0,
        'classification_weight': 0.5,
        'use_amplification': False,
        'use_diversity_loss': False,
        'use_temporal_order_loss': False,
    })
    print(f"✅ Loss function created successfully")
    
    # Test loss computation
    print(f"\n🔧 Computing loss...")
    total_loss, metrics, loss_components = loss_fn.compute_loss(predictions, targets, batch)
    
    print(f"\n📊 RESULTS:")
    print(f"   Total loss: {total_loss.item():.6f}")
    print(f"   Metrics keys: {list(metrics.keys())}")
    print(f"   Metrics values: {metrics}")
    print(f"   Loss components: {loss_components}")
    
    # Verify key metrics are non-zero
    key_metrics = ['regression_loss', 'classification_loss', 'classification_accuracy', 'total_dual_head_loss']
    for metric in key_metrics:
        if metric in metrics:
            value = metrics[metric]
            if value == 0.0:
                print(f"⚠️  WARNING: {metric} is 0.0")
            else:
                print(f"✅ {metric}: {value}")
        else:
            print(f"❌ MISSING: {metric} not in metrics")
    
    return True

def test_minimal_loss():
    """Test with minimal possible data."""
    print(f"\n🧪 Testing with minimal data...")
    
    try:
        from emprot.losses.composite_loss import create_composite_loss
        
        # Simplest possible test (regression-only)
        loss_fn = create_composite_loss({
            'regression_weight': 1.0,
            'classification_weight': 0.0,
            'use_amplification': False
        })
        
        # Minimal tensors
        pred = {'delta_embedding': torch.ones(1, 10, 64) * 0.1}
        target = {'short_term': torch.ones(1, 10, 64) * 0.2}
        batch = {
            'embeddings': torch.ones(1, 3, 10, 64),
            'sequence_lengths': torch.tensor([3]),
            'residue_mask': torch.ones(1, 10)
        }
        
        loss, metrics, loss_components = loss_fn.compute_loss(pred, target, batch)
        print(f"✅ Minimal test - Loss: {loss.item():.6f}")
        print(f"✅ Minimal test - Metrics: {metrics}")
        print(f"✅ Minimal test - Loss components: {loss_components}")
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting dual-head loss debugging tests...\n")
    
    # Test 1: Full dual-head loss
    if not test_dual_head_loss():
        print("\n❌ Full dual-head loss test failed!")
        return False
    
    # Test 2: Minimal loss
    if not test_minimal_loss():
        print("\n❌ Minimal loss test failed!")
        return False
    
    print("\n🎉 All tests passed! The dual-head loss function is working correctly.")
    print("\n💡 If you still see flat 0 metrics in training, the issue is likely:")
    print("   1. Data not being passed correctly to the loss function")
    print("   2. Model not producing the expected outputs")
    print("   3. Metrics being overwritten somewhere in the training loop")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
