#!/usr/bin/env python3
"""
Quick analysis of trained EMPROT checkpoint - focuses on key insights.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emprot.models.transformer import ProteinTransformerClassificationOnly


def quick_checkpoint_analysis(checkpoint_path: str, config: dict):
    """Quick analysis of what the model has learned."""
    
    print(f"🔍 Analyzing checkpoint: {checkpoint_path}")
    
    # Load model
    model = ProteinTransformerClassificationOnly(
        d_embed=config['d_embed'],
        num_heads=config['num_heads'],
        dropout=config.get('dropout', 0.1),
        use_gradient_checkpointing=False,
        min_context_frames=config.get('min_context_frames', 2),
        num_clusters=50000,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Analysis results
    results = {}
    
    # 1. Analyze learned temporal decay parameters
    if hasattr(model.cross_attention, 'decay_params'):
        decay_params = model.cross_attention.decay_params.data.cpu().numpy()
        decay_rates = torch.nn.functional.softplus(model.cross_attention.decay_params).data.cpu().numpy()
        
        results['temporal_decay'] = {
            'raw_params': decay_params.tolist(),
            'decay_rates': decay_rates.tolist(),
            'mean_decay': float(np.mean(decay_rates)),
            'std_decay': float(np.std(decay_rates)),
            'range': [float(np.min(decay_rates)), float(np.max(decay_rates))]
        }
        
        print(f"\n🕐 TEMPORAL DECAY ANALYSIS:")
        print(f"   Mean decay rate: {np.mean(decay_rates):.3f}")
        print(f"   Decay range: [{np.min(decay_rates):.3f}, {np.max(decay_rates):.3f}]")
        print(f"   Head diversity: {np.std(decay_rates):.3f}")
        
        # Interpret decay rates
        fast_heads = np.sum(decay_rates > np.mean(decay_rates) + np.std(decay_rates))
        slow_heads = np.sum(decay_rates < np.mean(decay_rates) - np.std(decay_rates))
        print(f"   Fast decay heads: {fast_heads} (focus on recent frames)")
        print(f"   Slow decay heads: {slow_heads} (focus on long-term history)")
    
    # 2. Analyze learned residue importance (if available)
    if hasattr(model.cross_attention, 'residue_importance') and 'cross_attention.residue_importance' in checkpoint['model_state_dict']:
        residue_weights = model.cross_attention.residue_importance.data.cpu().numpy()
        
        # Basic statistics
        results['residue_importance'] = {
            'shape': residue_weights.shape,
            'mean_per_head': np.mean(residue_weights, axis=1).tolist(),
            'std_per_head': np.std(residue_weights, axis=1).tolist(),
            'head_diversity': float(np.mean([np.std(head_weights) for head_weights in residue_weights]))
        }
        
        print(f"\n🧬 RESIDUE IMPORTANCE ANALYSIS:")
        print(f"   Shape: {residue_weights.shape} (heads x max_residues)")
        print(f"   Average head diversity: {np.mean([np.std(head_weights) for head_weights in residue_weights]):.3f}")
        
        # Find most/least important residues per head
        for head_idx in range(min(4, residue_weights.shape[0])):  # Show first 4 heads
            head_weights = residue_weights[head_idx]
            top_residues = np.argsort(head_weights)[-5:][::-1]
            bottom_residues = np.argsort(head_weights)[:5]
            
            print(f"   Head {head_idx}:")
            print(f"     Most important: {top_residues} (weights: {head_weights[top_residues]:.3f})")
            print(f"     Least important: {bottom_residues} (weights: {head_weights[bottom_residues]:.3f})")
        
        # Check if heads have specialized
        head_correlations = np.corrcoef(residue_weights)
        mean_correlation = np.mean(head_correlations[np.triu_indices_from(head_correlations, k=1)])
        print(f"   Head similarity (correlation): {mean_correlation:.3f}")
        if mean_correlation < 0.3:
            print("   ✅ Heads have SPECIALIZED (low correlation = good!)")
        elif mean_correlation > 0.7:
            print("   ⚠️ Heads are SIMILAR (high correlation = redundant)")
        else:
            print("   📊 Heads are MODERATELY specialized")
    else:
        print("\n⚠️ No residue_importance found in checkpoint. Skipping residue importance analysis.")
    
    # 3. Check training progress from checkpoint
    if 'epoch' in checkpoint:
        print(f"\n📊 TRAINING PROGRESS:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Global step: {checkpoint.get('global_step', 'N/A')}")
        print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # 4. Model architecture summary
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"   Embedding dim: {config['d_embed']}")
    print(f"   Attention heads: {config['num_heads']}")
    print(f"   Max residues: {config.get('max_residues', 1000)}")
    
    # Count learnable parameters by component
    attention_params = sum(p.numel() for p in model.cross_attention.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Attention parameters: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
    print(f"   Total parameters: {total_params:,}")
    
    return results


def visualize_quick_analysis(results: dict, output_path: str):
    """Create quick visualization of analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Temporal decay rates
    if 'temporal_decay' in results:
        ax = axes[0, 0]
        decay_rates = results['temporal_decay']['decay_rates']
        ax.bar(range(len(decay_rates)), decay_rates, alpha=0.7)
        ax.set_title('Learned Temporal Decay Rates')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Decay Rate')
        ax.grid(True, alpha=0.3)
    
    # 2. Residue importance heatmap (if available)
    if 'residue_importance' in results and hasattr(results, 'raw_weights'):
        ax = axes[0, 1]
        # This would need the raw weights to be saved
        ax.set_title('Residue Importance Heatmap')
        ax.text(0.5, 0.5, 'Raw weights needed\nfor visualization', 
                ha='center', va='center', transform=ax.transAxes)
    else:
        ax = axes[0, 1]
        ax.set_title('Residue Importance Analysis')
        if 'residue_importance' in results:
            diversity = results['residue_importance']['head_diversity']
            ax.bar(['Head Diversity'], [diversity], alpha=0.7)
            ax.set_ylabel('Standard Deviation')
        else:
            ax.text(0.5, 0.5, 'No residue importance\nweights found', 
                    ha='center', va='center', transform=ax.transAxes)
    
    # 3. Head diversity analysis
    ax = axes[1, 0]
    if 'temporal_decay' in results:
        decay_rates = results['temporal_decay']['decay_rates']
        ax.hist(decay_rates, bins=10, alpha=0.7, edgecolor='black')
        ax.set_title('Distribution of Decay Rates')
        ax.set_xlabel('Decay Rate')
        ax.set_ylabel('Number of Heads')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "ANALYSIS SUMMARY:\n\n"
    
    if 'temporal_decay' in results:
        mean_decay = results['temporal_decay']['mean_decay']
        std_decay = results['temporal_decay']['std_decay']
        summary_text += f"Temporal Memory:\n"
        summary_text += f"  Mean decay: {mean_decay:.3f}\n"
        summary_text += f"  Diversity: {std_decay:.3f}\n\n"
    
    if 'residue_importance' in results:
        diversity = results['residue_importance']['head_diversity']
        summary_text += f"Residue Focus:\n"
        summary_text += f"  Head diversity: {diversity:.3f}\n"
        summary_text += f"  Specialization: {'Good' if diversity > 0.5 else 'Limited'}\n\n"
    
    summary_text += "Interpretation:\n"
    summary_text += "• High decay diversity = heads learn\n  different temporal patterns\n"
    summary_text += "• High residue diversity = heads\n  focus on different regions"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {output_path}")


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Quick EMPROT checkpoint analysis')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='data/results/quick_analysis_output',
                       help='Output directory')
    
    # Model config (adjust based on your trained model)
    parser.add_argument('--d_embed', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--max_residues', type=int, default=1000)
    
    args = parser.parse_args()
    
    config = {
        'd_embed': args.d_embed,
        'num_heads': args.num_heads,
        'max_residues': args.max_residues
    }
    
    # Run analysis
    results = quick_checkpoint_analysis(args.checkpoint_path, config)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'quick_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    viz_path = os.path.join(args.output_dir, 'quick_analysis_viz.png')
    visualize_quick_analysis(results, viz_path)
    
    print(f"\n✅ Quick analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 