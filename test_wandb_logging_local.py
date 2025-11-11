#!/usr/bin/env python3
"""
Local test script to debug W&B flat 0 metrics issue.
Uses mock data to test W&B logging without requiring HCP access.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_wandb_logging_local():
    """Test W&B logging locally with mock data."""
    print("🧪 Testing W&B logging locally with mock data...")
    
    try:
        # Import the training script components
        from scripts.engines.emprot_trainer import EMPROTTrainer
        print("✅ Successfully imported EMPROTTrainer")
    except ImportError as e:
        print(f"❌ Failed to import EMPROTTrainer: {e}")
        return False
    
    # Create minimal config for local testing
    config = {
        'd_embed': 512,
        'num_heads': 8,
        'batch_size': 2,
        'max_residues': 100,
        'num_workers': 0,
        'min_sequence_length': 5,
        'max_sequence_length': 5,
        'stride': 10,
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
        'run_name': "wandb_debug_local",
        'wandb_project': "emprot-debug-local"
    }
    
    print(f"🔧 Config created with {len(config)} parameters")
    
    try:
        # Create trainer (without cluster lookup for local testing)
        print("🔧 Creating trainer...")
        trainer = EMPROTTrainer(config, cluster_lookup=None)  # No cluster lookup for local test
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return False
    
    try:
        # Create mock batch data
        print("🔧 Creating mock batch data...")
        B, T, N, E = 2, 5, 100, 512
        
        # Create mock batch that matches what the model expects
        batch = {
            'embeddings': torch.randn(B, T, N, E),
            'sequence_lengths': torch.tensor([5, 5]),
            'residue_mask': torch.ones(B, N),
            'times': torch.arange(T).float().unsqueeze(0).expand(B, -1) * 0.2,
            'history_mask': torch.ones(B, T, N),
            'targets': {
                'short_term': torch.randn(B, N, E) * 0.1,
                'long_term': torch.randn(B, N, E) * 0.2
            }
        }
        
        print(f"✅ Mock batch created with keys: {list(batch.keys())}")
        
        # Move batch to device
        batch = trainer._move_to_device(batch, trainer.device)
        print(f"✅ Batch moved to device: {trainer.device}")
        
    except Exception as e:
        print(f"❌ Failed to create mock batch: {e}")
        return False
    
    try:
        # Run multiple training steps to trigger W&B logging
        print("\n🔧 Running multiple training steps to trigger W&B logging...")
        print("   W&B logging happens every 10 steps, so we'll run 15 steps")
        
        # Run 15 steps to trigger W&B logging
        for step in range(15):
            try:
                print(f"\n--- Step {step} ---")
                print(f"   Global step: {trainer.global_step}")
                print(f"   W&B logging in {10 - (trainer.global_step % 10)} more steps")
                
                # Forward pass
                trainer.model.train()  # Set to training mode
                predictions = trainer.model(
                    embeddings=batch['embeddings'],
                    times=batch['times'],
                    sequence_lengths=batch['sequence_lengths'],
                    history_mask=batch['history_mask']
                )
                
                # Calculate loss (this will trigger W&B logging every 10 steps)
                total_loss, metrics = trainer._calculate_loss(predictions, batch, trainer.model, compute_interpretable_metrics=True)
                
                print(f"   Loss: {total_loss.item():.6f}")
                
                # Check for key metrics
                key_metrics = ['regression_loss', 'classification_loss', 'classification_accuracy', 'total_dual_head_loss']
                for metric in key_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        print(f"   {metric}: {value}")
                    else:
                        print(f"   {metric}: NOT FOUND")
                
                # Simulate training step (update global_step)
                trainer.global_step += 1
                
                # Check if this step should trigger W&B logging
                if (trainer.global_step - 1) % 10 == 0:
                    print(f"   🎯 STEP {trainer.global_step - 1}: W&B logging should have happened!")
                    print(f"   Check the W&B logging debug output above")
                
            except Exception as e:
                print(f"   ❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n✅ Completed {trainer.global_step} training steps")
        
    except Exception as e:
        print(f"❌ Failed to run training steps: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎯 LOCAL W&B LOGGING TEST COMPLETED!")
    print("   This should have triggered W&B logging at steps 0, 10, etc.")
    print("   Check the debug output above to see what was sent to W&B")
    print("   Note: This uses mock data, so the actual values may be different")
    
    return True

def main():
    """Run the local W&B logging test."""
    print("🚀 Starting LOCAL W&B logging debug test...")
    print("   This will run multiple training steps with mock data")
    print("   to trigger W&B logging and debug the flat 0 metrics issue\n")
    
    success = test_wandb_logging_local()
    
    if success:
        print("\n🎉 Local test completed successfully!")
        print("   Check the debug output above to see W&B logging in action")
        print("   Next: Run on HCP with real data for the full test")
    else:
        print("\n❌ Local test failed!")
        print("   Check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
