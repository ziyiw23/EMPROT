#!/usr/bin/env python3
"""
Quick diagnostic to check if the classification head was actually trained.
If weights are near zero, the head never learned anything.
"""

import torch
import sys

def diagnose_classification_head(ckpt_path):
    """Check if classification head weights indicate training occurred."""
    print(f"🔍 Diagnosing classification head in: {ckpt_path}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        
        W = None
        b = None
        
        # Find classification head weights
        for key, tensor in state_dict.items():
            if key.endswith("classification_head.3.weight"):
                W = tensor
                print(f"   Found weight tensor: {key} {W.shape}")
            if key.endswith("classification_head.3.bias"):
                b = tensor
                print(f"   Found bias tensor: {key} {b.shape}")
        
        if W is None:
            print("❌ No classification head weight found!")
            return False
            
        # Analyze weight statistics
        w_mean = W.abs().mean().item()
        w_std = W.std().item()
        w_max = W.abs().max().item()
        w_min = W.abs().min().item()
        
        print(f"\n📊 Weight Analysis:")
        print(f"   Mean absolute weight: {w_mean:.6f}")
        print(f"   Weight std dev: {w_std:.6f}")
        print(f"   Max absolute weight: {w_max:.6f}")
        print(f"   Min absolute weight: {w_min:.6f}")
        
        if b is not None:
            b_mean = b.abs().mean().item()
            b_std = b.std().item()
            print(f"   Bias mean absolute: {b_mean:.6f}")
            print(f"   Bias std dev: {b_std:.6f}")
        else:
            print("   No bias found")
            
        # Diagnostic assessment
        print(f"\n🎯 Assessment:")
        if w_std < 1e-3:
            print("   ❌ LIKELY UNTRAINED: Weight std < 1e-3 suggests head never learned")
            print("   💡 Check training logs for classification_loss decreasing")
            trained = False
        elif w_std < 1e-2:
            print("   ⚠️  POSSIBLY UNDERTRAINED: Weight std < 1e-2 suggests minimal learning")
            trained = True
        else:
            print("   ✅ LIKELY TRAINED: Weight std indicates meaningful learning occurred")
            trained = True
            
        print(f"   Rule of thumb: std > 1e-2 = trained, < 1e-3 = untrained")
        
        return trained
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_head_trained.py <checkpoint_path>")
        print("Example: python diagnose_head_trained.py checkpoints/emprot_classification_ONLY/best_model_epoch_36.pt")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    trained = diagnose_classification_head(ckpt_path)
    
    print(f"\n📋 Next Steps:")
    if trained:
        print("1. Head appears trained → proceed to evaluation")
        print("2. Use scripts/evaluate_single_model.py with model_type=classification")
    else:
        print("1. Head appears untrained → check training logs")
        print("2. Verify classification_loss was decreasing during training")
        print("3. If loss was flat, retrain with proper classification setup")
