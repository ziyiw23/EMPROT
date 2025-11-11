import sys
sys.path.append('.')

print("Testing EXTREME temporal strides for meaningful conformational changes...")

from emprot.utils.dataset import ProteinTrajectoryDataset
import torch
import datetime
import numpy as np
from scipy import stats

def analyze_residue_level_dynamics_robust(dataset, stride, selected_proteins, output_lines=None):
    """
    Analyze dynamics at individual residue level with proper validation and statistical testing.
    """
    
    all_residue_similarities = []
    all_residue_changes = []
    all_l2_distances = []
    random_baseline_similarities = []
    total_residues_analyzed = 0
    
    if output_lines is not None:
        output_lines.append(f"   📊 Analyzing {len(selected_proteins)} proteins with stride={stride} ({stride * 0.2:.1f}ns spacing)")
    
    for sample_idx in selected_proteins:
        sample = dataset[sample_idx]
        embeddings = sample['embeddings']  # (T, N, E)
        
        protein_id = sample.get('uniprot_id', f'Protein_{sample_idx}')
        if output_lines is not None:
            output_lines.append(f"   🧬 Analyzing {protein_id} (sample {sample_idx})")
        
        if 'targets' not in sample or 'short_term' not in sample['targets']:
            continue
            
        last_input = embeddings[-1]  # (N, E) - per residue
        target = sample['targets']['short_term']  # (N, E) - per residue
        
        N, E = last_input.shape
        
        # NO LayerNorm normalization - preserve actual embedding magnitudes
        # This is crucial for meaningful dynamics detection
        
        sample_output = []
        sample_output.append(f"\n🧬 Sample {sample_idx} - Robust Residue-Level Analysis:")
        sample_output.append(f"   Protein: {protein_id}")
        sample_output.append(f"   Residues: {N}, Embedding dim: {E}")
        sample_output.append(f"   Temporal stride: {stride} ({stride * 0.2:.1f}ns spacing)")
        sample_output.append(f"   Using RAW embeddings (no normalization) to preserve dynamics")
        
        print(sample_output[-4])
        print(sample_output[-3])
        print(sample_output[-2])
        print(sample_output[-1])
        
        # Calculate actual changes per residue
        residue_similarities = []
        residue_changes = []
        residue_l2_distances = []
        
        for residue_idx in range(N):
            res_input = last_input[residue_idx]  # (E,)
            res_target = target[residue_idx]     # (E,)
            
            # Raw cosine similarity (no normalization)
            cos_sim = torch.dot(res_input, res_target) / (torch.norm(res_input) * torch.norm(res_target))
            
            # Raw L2 distance
            l2_dist = torch.norm(res_input - res_target)
            
            # Relative change using actual magnitudes
            denom = torch.norm(res_target) + 1e-8
            relative_change = l2_dist / denom
            
            residue_similarities.append(cos_sim.item())
            residue_changes.append(relative_change.item())
            residue_l2_distances.append(l2_dist.item())
        
        # RANDOM BASELINE TEST - critical validation
        # Shuffle the target embeddings randomly to see if our "dynamics" are just noise
        random_indices = torch.randperm(N)
        random_target = target[random_indices]
        
        random_similarities = []
        for residue_idx in range(N):
            res_input = last_input[residue_idx]
            res_random = random_target[residue_idx]
            random_cos_sim = torch.dot(res_input, res_random) / (torch.norm(res_input) * torch.norm(res_random))
            random_similarities.append(random_cos_sim.item())
        
        random_baseline_similarities.extend(random_similarities)
        
        # STATISTICAL SIGNIFICANCE TESTING
        # Compare actual changes to random baseline
        actual_avg_sim = np.mean(residue_similarities)
        random_avg_sim = np.mean(random_similarities)
        
        # T-test to see if actual changes are significantly different from random
        t_stat, p_value = stats.ttest_ind(residue_similarities, random_similarities)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(residue_similarities) - 1) * np.var(residue_similarities, ddof=1) + 
                             (len(random_similarities) - 1) * np.var(random_similarities, ddof=1)) / 
                            (len(residue_similarities) + len(random_similarities) - 2))
        cohens_d = (actual_avg_sim - random_avg_sim) / pooled_std if pooled_std > 0 else 0
        
        # MEANINGFUL DYNAMICS DETECTION
        # Only consider residues as "dynamic" if they're significantly different from random
        # AND if the effect size is meaningful (Cohen's d > 0.5 = medium effect)
        
        meaningful_dynamic_count = 0
        meaningful_dynamic_residues = []
        
        for residue_idx in range(N):
            actual_sim = residue_similarities[residue_idx]
            random_sim = random_similarities[residue_idx]
            
            # A residue is "meaningfully dynamic" if:
            # 1. It's significantly different from random (using 2 std dev threshold)
            # 2. The effect size is meaningful
            # 3. The absolute change is above noise level
            
            # Calculate how many std devs away from random this residue is
            random_std = np.std(random_similarities)
            z_score = (actual_sim - random_sim) / random_std if random_std > 0 else 0
            
            # Threshold: must be at least 2 std devs away from random baseline
            # AND have meaningful effect size
            if (abs(z_score) > 2.0 and 
                abs(actual_sim - random_sim) > 0.02 and  # At least 2% different from random
                residue_l2_distances[residue_idx] > np.mean(residue_l2_distances) + np.std(residue_l2_distances)):
                
                meaningful_dynamic_count += 1
                meaningful_dynamic_residues.append({
                    'sample_idx': sample_idx,
                    'residue_idx': residue_idx,
                    'cosine_sim': actual_sim,
                    'random_sim': random_sim,
                    'z_score': z_score,
                    'l2_distance': residue_l2_distances[residue_idx],
                    'relative_change': residue_changes[residue_idx]
                })
        
        # Summary with statistical validation
        summary_lines = [
            f"   📊 Statistical Validation:",
            f"      Actual avg similarity: {actual_avg_sim:.6f}",
            f"      Random baseline: {random_avg_sim:.6f}",
            f"      Difference: {actual_avg_sim - random_avg_sim:.6f}",
            f"      T-test p-value: {p_value:.6f}",
            f"      Effect size (Cohen's d): {cohens_d:.3f}",
            f"      Meaningful dynamic residues: {meaningful_dynamic_count}/{N} ({meaningful_dynamic_count/N*100:.1f}%)"
        ]
        
        # Assessment based on statistical significance
        if p_value < 0.001 and cohens_d > 0.8:
            assessment = "   🎉 STRONG EVIDENCE: Changes significantly different from random (large effect)"
        elif p_value < 0.01 and cohens_d > 0.5:
            assessment = "   ✅ GOOD EVIDENCE: Changes significantly different from random (medium effect)"
        elif p_value < 0.05:
            assessment = "   ⚡ WEAK EVIDENCE: Changes different from random but small effect"
        else:
            assessment = "   ❌ NO EVIDENCE: Changes not significantly different from random noise"
        
        summary_lines.append(assessment)
        sample_output.extend(summary_lines)
        
        for line in summary_lines:
            print(line)
        
        # Show top dynamic residues if any found
        if meaningful_dynamic_residues:
            sample_output.append(f"   🔥 Top meaningful dynamic residues:")
            # Sort by z-score (most different from random)
            meaningful_dynamic_residues.sort(key=lambda x: abs(x['z_score']), reverse=True)
            
            for i, res in enumerate(meaningful_dynamic_residues[:5]):  # Top 5
                line = (f"      Residue {res['residue_idx']:3d}: "
                       f"actual={res['cosine_sim']:.4f}, random={res['random_sim']:.4f}, "
                       f"z-score={res['z_score']:.2f}")
                sample_output.append(line)
                print(line)
        
        all_residue_similarities.extend(residue_similarities)
        all_residue_changes.extend(residue_changes)
        all_l2_distances.extend(residue_l2_distances)
        total_residues_analyzed += N
    
    # OVERALL ANALYSIS with proper statistical validation
    overall_output = []
    
    if all_residue_similarities and random_baseline_similarities:
        actual_overall_avg = np.mean(all_residue_similarities)
        random_overall_avg = np.mean(random_baseline_similarities)
        overall_diff = actual_overall_avg - random_overall_avg
        
        # Overall statistical test
        overall_t_stat, overall_p_value = stats.ttest_ind(all_residue_similarities, random_baseline_similarities)
        
        # Calculate meaningful dynamic fraction
        meaningful_dynamic_fraction = 0
        if total_residues_analyzed > 0:
            # Count residues that are meaningfully different from random
            meaningful_count = 0
            for i, (actual_sim, random_sim) in enumerate(zip(all_residue_similarities, random_baseline_similarities)):
                random_std = np.std(random_baseline_similarities)
                z_score = (actual_sim - random_sim) / random_std if random_std > 0 else 0
                if abs(z_score) > 2.0 and abs(actual_sim - random_sim) > 0.02:
                    meaningful_count += 1
            
            meaningful_dynamic_fraction = meaningful_count / total_residues_analyzed
        
        overall_lines = [
            f"\n🎯 ROBUST OVERALL ANALYSIS:",
            f"   Total residues analyzed: {total_residues_analyzed}",
            f"   Actual avg similarity: {actual_overall_avg:.6f}",
            f"   Random baseline: {random_overall_avg:.6f}",
            f"   Overall difference: {overall_diff:.6f}",
            f"   Statistical significance: p = {overall_p_value:.6f}",
            f"   Meaningful dynamic fraction: {meaningful_dynamic_fraction*100:.1f}%"
        ]
        
        # Final assessment based on statistical evidence
        if overall_p_value < 0.001 and abs(overall_diff) > 0.01:
            final_assessment = "   🎉 STRONG EVIDENCE: Temporal changes are real, not noise!"
        elif overall_p_value < 0.01 and abs(overall_diff) > 0.005:
            final_assessment = "   ✅ GOOD EVIDENCE: Some real temporal dynamics detected"
        elif overall_p_value < 0.05:
            final_assessment = "   ⚡ WEAK EVIDENCE: Small temporal effects, may be noise"
        else:
            final_assessment = "   ❌ NO EVIDENCE: Changes indistinguishable from random noise"
        
        overall_lines.append(final_assessment)
        overall_output.extend(overall_lines)
        
        for line in overall_lines:
            print(line)
    
    # Add all output to the main output list
    if output_lines is not None:
        output_lines.extend(overall_output)
    
    return {
        'all_similarities': all_residue_similarities,
        'all_changes': all_residue_changes,
        'all_l2_distances': all_l2_distances,
        'random_baseline': random_baseline_similarities,
        'meaningful_dynamic_fraction': meaningful_dynamic_fraction if 'meaningful_dynamic_fraction' in locals() else 0,
        'total_residues_analyzed': total_residues_analyzed,
        'overall_avg': actual_overall_avg if 'actual_overall_avg' in locals() else 1.0,
        'overall_min': min(all_residue_similarities) if all_residue_similarities else 1.0,
        'statistical_p_value': overall_p_value if 'overall_p_value' in locals() else 1.0,
        'effect_size': overall_diff if 'overall_diff' in locals() else 0.0
    }

# Test extreme strides with robust analysis
stride_values = [1, 5, 10, 25, 50, 75, 100, 125]  # 0.2ns, 1ns, 2ns, 5ns, 10ns, 15ns, 20ns, 25ns

results = []
all_output_lines = []

# Add header to output
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
all_output_lines.append(f"ROBUST EXTREME TEMPORAL STRIDES ANALYSIS")
all_output_lines.append(f"Generated: {timestamp}")
all_output_lines.append(f"Key improvements: No normalization, random baseline testing, statistical validation")
all_output_lines.append(f"=" * 80)
all_output_lines.append("")

# Randomly select 3 proteins for analysis across all strides
import random
random.seed(42)  # For reproducible results
selected_proteins = random.sample(range(1000), 3)  # Assuming dataset has at least 1000 samples
all_output_lines.append(f"🎯 Selected proteins for analysis: {selected_proteins}")
all_output_lines.append("")

for stride in stride_values:
    stride_header = f"\n{'='*80}"
    stride_title = f"🔧 Testing EXTREME stride={stride} ({stride * 0.2:.1f}ns spacing)"
    stride_separator = f"{'='*80}"
    
    all_output_lines.extend([stride_header, stride_title, stride_separator])
    print(stride_header)
    print(stride_title)
    print(stride_separator)
    
    try:
        dataset = ProteinTrajectoryDataset(
            data_dir="/scratch/groups/rbaltman/ziyiw23/traj_embeddings/",
            metadata_path="traj_metadata.csv",
            max_sequence_length=5,
            min_sequence_length=1,
            stride=stride
        )
        
        dataset_info = f"   Dataset size: {len(dataset)} samples"
        all_output_lines.append(dataset_info)
        print(dataset_info)
        
        if len(dataset) == 0:
            no_samples_msg = "   ❌ No samples available - stride too large for data"
            all_output_lines.append(no_samples_msg)
            print(no_samples_msg)
            continue
        
        # Run robust residue-level analysis
        residue_results = analyze_residue_level_dynamics_robust(dataset, stride, selected_proteins, output_lines=all_output_lines)
        
        results.append({
            'stride': stride,
            'spacing_ns': stride * 0.2,
            'overall_avg_similarity': residue_results['overall_avg'],
            'overall_min_similarity': residue_results['overall_min'],
            'meaningful_dynamic_fraction': residue_results['meaningful_dynamic_fraction'],
            'statistical_p_value': residue_results['statistical_p_value'],
            'effect_size': residue_results['effect_size'],
            'total_residues': residue_results['total_residues_analyzed'],
            'samples': len(dataset)
        })
        
        # Assessment based on statistical evidence
        p_val = residue_results['statistical_p_value']
        effect = abs(residue_results['effect_size'])
        
        if p_val < 0.001 and effect > 0.01:
            assessment = "   🎯 STRONG EVIDENCE: Real temporal dynamics detected!"
        elif p_val < 0.01 and effect > 0.005:
            assessment = "   ✅ GOOD EVIDENCE: Some real temporal dynamics"
        elif p_val < 0.05:
            assessment = "   ⚡ WEAK EVIDENCE: Small effects, may be noise"
        else:
            assessment = "   ❌ NO EVIDENCE: Changes indistinguishable from random"
        
        all_output_lines.append(assessment)
        print(assessment)
            
    except Exception as e:
        error_msg = f"   ❌ Error: {e}"
        all_output_lines.append(error_msg)
        print(error_msg)

# Results summary with statistical validation
results_header = f"\n📊 ROBUST STRIDE RESULTS WITH STATISTICAL VALIDATION:"
results_separator = "=" * 120
results_table_header = "Stride | Spacing | Samples | Avg Sim | Min Sim | Meaningful% | P-Value | Effect Size | Total Residues"
results_table_separator = "-" * 120

all_output_lines.extend([results_header, results_separator, results_table_header, results_table_separator])
print(results_header)
print(results_separator)
print(results_table_header)
print(results_table_separator)

for result in results:
    p_val_str = f"{result['statistical_p_value']:.6f}" if result['statistical_p_value'] < 0.001 else f"{result['statistical_p_value']:.4f}"
    effect_str = f"{result['effect_size']:.6f}"
    
    result_line = (f"{result['stride']:6d} | {result['spacing_ns']:7.1f}ns | "
                   f"{result['samples']:7d} | {result['overall_avg_similarity']:7.4f} | "
                   f"{result['overall_min_similarity']:7.4f} | {result['meaningful_dynamic_fraction']*100:10.1f}% | "
                   f"{p_val_str:7s} | {effect_str:10s} | {result['total_residues']:13d}")
    all_output_lines.append(result_line)
    print(result_line)

# Final analysis with statistical interpretation
analysis_header = f"\n🎯 STATISTICAL INTERPRETATION:"
all_output_lines.append(analysis_header)
print(analysis_header)

if results:
    # Find stride with strongest statistical evidence
    significant_results = [r for r in results if r['statistical_p_value'] < 0.05]
    
    if significant_results:
        best_result = min(significant_results, key=lambda x: x['statistical_p_value'])
        largest_effect = max(significant_results, key=lambda x: abs(x['effect_size']))
        
        analysis_lines = [
            f"   Most statistically significant: Stride {best_result['stride']} ({best_result['spacing_ns']:.1f}ns)",
            f"   P-value: {best_result['statistical_p_value']:.6f}",
            f"   Largest effect size: Stride {largest_effect['stride']} ({largest_effect['spacing_ns']:.1f}ns)",
            f"   Effect size: {largest_effect['effect_size']:.6f}"
        ]
        
        # Statistical significance interpretation
        if best_result['statistical_p_value'] < 0.001:
            sig_level = "highly significant (p < 0.001)"
        elif best_result['statistical_p_value'] < 0.01:
            sig_level = "very significant (p < 0.01)"
        else:
            sig_level = "significant (p < 0.05)"
        
        analysis_lines.append(f"   Statistical conclusion: {sig_level}")
        
    else:
        analysis_lines = [
            f"   ❌ No statistically significant temporal dynamics detected",
            f"   All p-values > 0.05 - changes may be random noise"
        ]
    
    all_output_lines.extend(analysis_lines)
    for line in analysis_lines:
        print(line)
    
    # Recommendations based on statistical evidence
    rec_header = f"\n💡 EVIDENCE-BASED RECOMMENDATIONS:"
    all_output_lines.append(rec_header)
    print(rec_header)
    
    if significant_results:
        best_stride = min(significant_results, key=lambda x: x['statistical_p_value'])
        
        if best_stride['statistical_p_value'] < 0.001 and abs(best_stride['effect_size']) > 0.01:
            rec_lines = [
                f"   🎯 STRONG RECOMMENDATION: Use stride {best_stride['stride']} for training!",
                f"      High statistical significance (p < 0.001) with meaningful effect size",
                f"      {best_stride['meaningful_dynamic_fraction']*100:.1f}% of residues show real dynamics"
            ]
        elif best_stride['statistical_p_value'] < 0.01:
            rec_lines = [
                f"   ✅ MODERATE RECOMMENDATION: Consider stride {best_stride['stride']}",
                f"      Statistically significant but effect size may be small",
                f"      Monitor training performance closely"
            ]
        else:
            rec_lines = [
                f"   ⚠️  WEAK RECOMMENDATION: Limited evidence for stride {best_stride['stride']}",
                f"      Barely significant - may not provide meaningful improvements"
            ]
    else:
        rec_lines = [
            f"   ❌ NO RECOMMENDATION: No statistical evidence for temporal dynamics",
            f"      Consider alternative approaches:",
            f"      • Enhanced temporal order loss with distant negatives",
            f"      • Attention pattern regularization", 
            f"      • Architecture modifications",
            f"      • Focus on static structural features instead"
        ]
    
    all_output_lines.extend(rec_lines)
    for line in rec_lines:
        print(line)
        
else:
    no_results_msg = "   No successful tests - all strides too large for available data"
    all_output_lines.append(no_results_msg)
    print(no_results_msg)

# Write results to file
output_filename = f"robust_extreme_strides_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_filename, 'w') as f:
    f.write('\n'.join(all_output_lines))

print(f"\n📄 Results saved to: {output_filename}") 