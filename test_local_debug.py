#!/usr/bin/env python3
"""
Local test version that mocks data loading to verify script structure.
This doesn't require HCP access and can be run locally.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_local_mock_data():
    """Test with locally generated mock data."""
    print("🧪 Testing with local mock data...")
    
    try:
        # Import the trainer
        from scripts.engines.emprot_trainer import EMPROTTrainer
        print("✅ Successfully imported EMPROTTrainer")
    except ImportError as e:
        print(f"❌ Failed to import EMPROTTrainer: {e}")
        return False
    
    # Create minimal config
    config = {
        'd_embed': 512,
        'num_heads': 8,
        'learning_rate': 3e-5,
        'max_epochs': 1,
        'patience': 10,
        'checkpoint_interval': 5,
        'regression_weight': 1.0,
        'classification_weight': 3.0,
        'temporal_order_weight': 2.0,
        'consistency_weight': 0.02,
        'diversity_weight': 0.1,
        'magnitude_threshold': 0.003,
        'max_amplification': 5.0,
        'temporal_amplification_factor': 3.0,
        'enable_attention_amplification': True,
        'enable_gradient_amplification': True,
        'use_gradient_checkpointing': True,
        'run_name': "local_debug_test",
        'wandb_project': "emprot-local-debug"
    }
    
    print(f"🔧 Config created with {len(config)} parameters")
    
    try:
        # Create trainer without cluster lookup (will use single-head mode)
        print("🔧 Creating trainer...")
        trainer = EMPROTTrainer(config, cluster_lookup=None)
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return False
    
    try:
        # Create mock batch data
        print("🔧 Creating mock batch data...")
        B, T, N, E = 2, 5, 100, 512
        
        batch = {
            'embeddings': torch.randn(B, T, N, E),
            'sequence_lengths': torch.tensor([5, 4]),
            'residue_mask': torch.ones(B, N),
            'times': torch.arange(T).float().unsqueeze(0).expand(B, -1) * 0.2,  # 0.2ns per frame
            'history_mask': torch.ones(B, T, N),  # All frames and residues are valid
            'targets': {
                'short_term': torch.randn(B, N, E) * 0.1,
                'long_term': torch.randn(B, N, E) * 0.2
            }
        }
        
        print(f"✅ Mock batch created with keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}, dtype: {value.dtype}")
            elif isinstance(value, dict):
                print(f"   {key}: dict with keys {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"     {subkey}: {subvalue.shape}, dtype: {subvalue.dtype}")
        
        # Move batch to device
        batch = trainer._move_to_device(batch, trainer.device)
        print(f"✅ Batch moved to device: {trainer.device}")
        
    except Exception as e:
        print(f"❌ Failed to create mock batch: {e}")
        return False
    
    try:
        # Run one forward pass
        print("🔧 Running one forward pass...")
        trainer.model.eval()
        
        with torch.no_grad():
            # Forward pass
            predictions = trainer.model(
                embeddings=batch['embeddings'],
                times=batch['times'],
                sequence_lengths=batch['sequence_lengths'],
                history_mask=batch['history_mask']
            )
            
            print(f"✅ Forward pass completed")
            print(f"   Predictions keys: {list(predictions.keys())}")
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape}, mean: {value.mean().item():.6f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    try:
        # Calculate loss (this is where we'll see the debug output)
        print("🔧 Calculating loss...")
        total_loss, metrics = trainer._calculate_loss(predictions, batch, trainer.model, compute_interpretable_metrics=True)
        
        print(f"✅ Loss calculation completed")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   Metrics keys: {list(metrics.keys())}")
        print(f"   Metrics values: {metrics}")
        
        # Check for key metrics (single-head mode)
        key_metrics = ['mse_loss', 'temporal_order_loss', 'diversity_loss']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if value == 0.0:
                    print(f"⚠️  WARNING: {metric} is 0.0")
                else:
                    print(f"✅ {metric}: {value}")
            else:
                print(f"❌ MISSING: {metric} not in metrics")
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎯 LOCAL TEST COMPLETED!")
    print("   This verifies the script structure works locally")
    print("   Run the HCP version to test with actual data")
    
    return True

def main():
    """Run the local test."""
    print("🚀 Starting local mock data test...")
    print("   This verifies the script structure without HCP access\n")
    
    success = test_local_mock_data()
    
    if success:
        print("\n🎉 Local test completed successfully!")
        print("   Script structure is working correctly")
        print("   Next: Run on HCP with actual data")
    else:
        print("\n❌ Local test failed!")
        print("   Check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
