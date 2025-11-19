#!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_comparison(json_path: str, output_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Define metrics to compare
    # (Metric Key in JSON, Display Name, Lower is Better?)
    metrics_map = [
        # Distributional
        ('visitation_js', 'Visitation JS', True),
        ('per_residue.per_residue_js_mean', 'Residue JS (Mean)', True),
        ('per_residue.per_residue_l1_mean', 'Residue L1 (Mean)', True),
        
        # Dynamics (New)
        ('dynamics.dwell_mean_diff', 'Dwell Time Error', True),
        ('dynamics.change_rate_error', 'Change Rate Error', True),
        ('dynamics.norm_edit_dist', 'Edit Distance (Norm)', True),
        ('dynamics.transition_overlap', 'Transition Overlap', False), # Higher is better
    ]
    
    transformer_vals = []
    markov_vals = []
    labels = []
    
    markov_data = data.get('markov_baseline', {})
    
    for key_path, label, lower_better in metrics_map:
        # Extract value from nested dict
        def get_val(d, path):
            keys = path.split('.')
            curr = d
            for k in keys:
                if curr is None or not isinstance(curr, dict):
                    return None
                curr = curr.get(k)
            return curr
            
        t_val = get_val(data, key_path)
        m_val = get_val(markov_data, key_path)
        
        if t_val is not None and m_val is not None:
            transformer_vals.append(float(t_val))
            markov_vals.append(float(m_val))
            suffix = " (↓)" if lower_better else " (↑)"
            labels.append(label + suffix)
            
    if not labels:
        print("No matching metrics found to plot.")
        return

    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, transformer_vals, width, label='Transformer', color='#1f77b4', alpha=0.8)
    rects2 = ax.bar(x + width/2, markov_vals, width, label='Markov Baseline', color='#2ca02c', alpha=0.8)
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Model vs Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    out = args.output
    if out is None:
        out = str(Path(args.json_path).parent / 'metrics_comparison.png')
        
    plot_comparison(args.json_path, out)

