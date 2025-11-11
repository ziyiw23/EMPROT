#!/usr/bin/env python3
"""
Enhanced curriculum learning stride analysis script.
This script analyzes cluster ID changes for curriculum learning stride numbers.
This helps understand how cluster IDs evolve at different temporal resolutions.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from emprot.data.data_loader import LMDBLoader
    from emprot.data.metadata import MetadataManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Curriculum learning stride numbers
CURRICULUM_STRIDES = [125, 50, 10, 5, 1]

def get_protein_info(lmdb_path, metadata_path="traj_metadata.csv"):
    """Extract protein information using the same method as comp_viz_input.py"""
    try:
        dynamic_id = os.path.basename(lmdb_path).split("_")[2]
        metadata = MetadataManager(metadata_path)
        protein_info = metadata.get_protein_info(dynamic_id)
        
        # Convert pandas Series to string if needed
        if hasattr(protein_info, 'to_string'):
            protein_info = protein_info.to_string()
        elif isinstance(protein_info, (list, tuple)):
            protein_info = str(protein_info)
        elif protein_info is None:
            protein_info = None
        else:
            protein_info = str(protein_info)
            
        return protein_info
    except Exception as e:
        print(f"      ⚠️  Warning: Could not extract protein info: {e}")
        return None

def analyze_cluster_id_changes_for_stride(traj_path: str, stride: int):
    """
    Analyze cluster ID changes for a specific stride number.
    
    Args:
        traj_path: Path to trajectory LMDB
        stride: Stride number to analyze
        
    Returns:
        Dictionary with analysis results
    """
    try:
        with LMDBLoader(traj_path, read_only=True) as loader:
            metadata = loader.get_metadata()
            num_frames = metadata['num_frames']  # Use ALL frames
            
            # Calculate frame indices for this stride
            # For stride N, we sample frames: 0, N, 2N, 3N, ...
            frame_indices = list(range(0, num_frames, stride))
            
            if len(frame_indices) < 2:
                return {
                    'stride': stride,
                    'frames_analyzed': len(frame_indices),
                    'cluster_changes': 0,
                    'change_rate': 0.0,
                    'error': 'Not enough frames for this stride'
                }
            
            print(f"      📊 Stride {stride}: Analyzing {len(frame_indices)} frames...")
            
            # Load cluster IDs for stride-sampled frames
            cluster_ids_sequence = []
            valid_frame_indices = []
            
            for frame_idx in frame_indices:
                try:
                    frame_data = loader.load_frame(frame_idx)
                    if 'cluster_ids' in frame_data:
                        cluster_ids = frame_data['cluster_ids']
                        cluster_ids_sequence.append(cluster_ids)
                        valid_frame_indices.append(frame_idx)
                except Exception as e:
                    print(f"         ⚠️  Error loading frame {frame_idx}: {e}")
                    continue
            
            if len(cluster_ids_sequence) < 2:
                return {
                    'stride': stride,
                    'frames_analyzed': len(cluster_ids_sequence),
                    'cluster_changes': 0,
                    'change_rate': 0.0,
                    'error': 'Not enough valid frames'
                }
            
            # Convert to numpy array
            cluster_ids_array = np.array(cluster_ids_sequence)  # (num_frames, num_residues)
            valid_frame_indices = np.array(valid_frame_indices)
            
            # Calculate cluster ID changes between consecutive frames
            cluster_changes = np.diff(cluster_ids_array, axis=0) != 0  # (num_frames-1, num_residues)
            
            # Count total changes
            total_changes = np.sum(cluster_changes)
            total_possible_changes = cluster_changes.size
            change_rate = total_changes / total_possible_changes if total_possible_changes > 0 else 0.0
            
            # Calculate changes per residue
            changes_per_residue = np.sum(cluster_changes, axis=0)
            dynamic_residues = np.sum(changes_per_residue > 0)
            
            # Calculate temporal spacing in nanoseconds (assuming 0.2ns per frame)
            temporal_spacing_ns = stride * 0.2
            
            return {
                'stride': stride,
                'frames_analyzed': len(valid_frame_indices),
                'temporal_spacing_ns': temporal_spacing_ns,
                'cluster_changes': total_changes,
                'total_possible_changes': total_possible_changes,
                'change_rate': change_rate,
                'dynamic_residues': dynamic_residues,
                'total_residues': cluster_ids_array.shape[1],
                'dynamic_fraction': dynamic_residues / cluster_ids_array.shape[1] if cluster_ids_array.shape[1] > 0 else 0.0,
                'cluster_ids_array': cluster_ids_array,
                'frame_indices': valid_frame_indices
            }
            
    except Exception as e:
        return {
            'stride': stride,
            'frames_analyzed': 0,
            'cluster_changes': 0,
            'change_rate': 0.0,
            'error': str(e)
        }

def plot_stride_analysis(traj_name: str, stride_results: list, protein_info: str = None):
    """
    Create comprehensive plots for stride analysis with protein names in titles.
    
    Args:
        traj_name: Name of trajectory
        stride_results: List of analysis results for each stride
        protein_info: Protein information to include in plot titles
    """
    # Filter out results with errors
    valid_results = [r for r in stride_results if 'error' not in r]
    
    if not valid_results:
        print(f"      ❌ No valid results to plot for {traj_name}")
        return
    
    # Create subplots for 5 strides
    num_strides = len(valid_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns = 6 subplots
    axes = axes.flatten()
    
    # Create clean, simple title
    if protein_info is not None and str(protein_info).strip():
        protein_str = str(protein_info)
        
        # Extract just the protein type/name for title
        if '\n' in protein_str:
            first_line = protein_str.split('\n')[0].strip()
            if 'receptor' in first_line.lower():
                protein_title = 'Receptor'
            elif 'protein' in first_line.lower():
                protein_title = 'Protein'
            elif 'enzyme' in first_line.lower():
                protein_title = 'Enzyme'
            else:
                protein_title = first_line.split()[0] if first_line.split() else 'Unknown'
        else:
            protein_title = protein_str.split()[0] if protein_str.split() else 'Unknown'
        
        main_title = f'Cluster ID Evolution - {traj_name}\n{protein_title} - Curriculum Learning Strides'
    else:
        main_title = f'Cluster ID Evolution - {traj_name}\nCurriculum Learning Strides'
    
    for i, result in enumerate(valid_results):
        if i >= 5:  # Only plot first 5 strides
            break
            
        ax = axes[i]
        stride = result['stride']
        
        # Plot cluster ID evolution for first 10 residues
        cluster_ids_array = result['cluster_ids_array']
        frame_indices = result['frame_indices']
        
        num_residues_to_plot = min(10, cluster_ids_array.shape[1])
        for residue_idx in range(num_residues_to_plot):
            ax.plot(frame_indices, cluster_ids_array[:, residue_idx], 
                   alpha=0.7, linewidth=1, marker='o', markersize=3)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Cluster ID')
        
        # Clean, simple title
        ax.set_title(f'Stride {stride} ({result["temporal_spacing_ns"]:.1f}ns)\n'
                    f'Change Rate: {result["change_rate"]:.1%}\n'
                    f'Dynamic: {result["dynamic_residues"]}/{result["total_residues"]}')
        
        ax.grid(True, alpha=0.3)
        
        # Add legend for first few residues
        if i == 0:  # Only add legend to first plot to avoid clutter
            ax.legend([f'Residue {j}' for j in range(num_residues_to_plot)], 
                     loc='upper right', fontsize=8)
    
    # Hide the 6th subplot (unused)
    axes[5].set_visible(False)
    
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    
    # Create results directory
    results_dir = Path(project_root) / "data" / "results" / "curriculum_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot with simple, clean filename
    if protein_info is not None and str(protein_info).strip():
        protein_str = str(protein_info)
        
        # Extract just the main protein name for filename
        if '\n' in protein_str:
            # Take first line and extract protein name
            first_line = protein_str.split('\n')[0].strip()
            # Look for key protein identifiers
            if 'receptor' in first_line.lower():
                protein_name = 'Receptor'
            elif 'protein' in first_line.lower():
                protein_name = 'Protein'
            elif 'enzyme' in first_line.lower():
                protein_name = 'Enzyme'
            else:
                # Take first meaningful word
                words = first_line.split()
                protein_name = words[0] if words else 'Unknown'
        else:
            protein_name = protein_str.split()[0] if protein_str.split() else 'Unknown'
        
        # Clean and limit filename
        clean_name = protein_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        plot_filename = f"stride_analysis_{traj_name}_{clean_name}.png"
    else:
        plot_filename = f"stride_analysis_{traj_name}.png"
    
    plot_path = results_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"      📊 Stride analysis plot saved as: {plot_path}")
    
    # Save detailed metadata to separate text file
    metadata_filename = f"protein_metadata_{traj_name}.txt"
    metadata_path = results_dir / metadata_filename
    
    with open(metadata_path, 'w') as f:
        f.write(f"PROTEIN METADATA: {traj_name}\n")
        f.write("=" * 60 + "\n\n")
        
        if protein_info is not None and str(protein_info).strip():
            f.write("FULL PROTEIN INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(str(protein_info))
            f.write("\n\n")
        else:
            f.write("Protein information not available\n\n")
        
        f.write("STRIDE ANALYSIS RESULTS:\n")
        f.write("-" * 40 + "\n")
        for result in stride_results:
            if 'error' not in result:
                f.write(f"Stride {result['stride']:3d} ({result['temporal_spacing_ns']:5.1f}ns):\n")
                f.write(f"  Frames analyzed: {result['frames_analyzed']}\n")
                f.write(f"  Change rate: {result['change_rate']:.1%}\n")
                f.write(f"  Dynamic residues: {result['dynamic_residues']}/{result['total_residues']} ({result['dynamic_fraction']:.1%})\n")
                f.write(f"  Cluster changes: {result['cluster_changes']}\n\n")
            else:
                f.write(f"Stride {result['stride']}: ERROR - {result['error']}\n\n")
    
    print(f"      📄 Detailed metadata saved to: {metadata_path}")
    
    return str(plot_path)

def test_curriculum_stride_analysis(data_dir: str):
    """
    Test cluster ID changes for curriculum learning stride numbers.
    
    Args:
        data_dir: Path to trajectory embeddings directory
    """
    print(f"🔍 Curriculum Learning Stride Analysis")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Analyzing strides: {CURRICULUM_STRIDES}")
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Get list of trajectory directories
    try:
        trajectories = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]
        trajectories.sort()
        print(f"📁 Found {len(trajectories)} trajectory directories")
    except Exception as e:
        print(f"❌ Error listing directory: {e}")
        return
    
    # Randomly select 2 trajectories instead of analyzing all
    print(f"\n🎲 Randomly selecting 2 trajectories from {len(trajectories)} available:")
    test_trajectories = random.sample(trajectories, min(2, len(trajectories)))
    
    all_results = []
    
    for i, traj_name in enumerate(test_trajectories, 1):
        traj_path = os.path.join(data_dir, traj_name)
        print(f"\n{i}. {traj_name}")
        print("   " + "-" * 60)
        
        # Extract protein information
        protein_info = get_protein_info(traj_path)
        if protein_info is not None and str(protein_info).strip():
            protein_str = str(protein_info)
            
            # Display protein info in a readable format
            if '\n' in protein_str:
                # Multi-line info - show key details
                lines = protein_str.split('\n')
                print(f"   🧬 Protein: {lines[0].strip()}")
                
                # Show additional key information
                key_info = []
                for line in lines[1:]:
                    line = line.strip()
                    if line and any(keyword in line.lower() for keyword in ['pdb', 'uniprot', 'family', 'species']):
                        key_info.append(line)
                
                if key_info:
                    print(f"      📋 Key details:")
                    for info in key_info[:3]:  # Show first 3 key details
                        print(f"         {info}")
            else:
                # Single line info
                print(f"   🧬 Protein: {protein_str}")
        else:
            print(f"   ⚠️  Protein info not available")
        
        # Analyze each stride
        stride_results = []
        for stride in CURRICULUM_STRIDES:
            print(f"   🔍 Analyzing stride {stride}...")
            result = analyze_cluster_id_changes_for_stride(traj_path, stride)
            stride_results.append(result)
            
            if 'error' in result:
                print(f"      ❌ Error: {result['error']}")
            else:
                print(f"      ✅ Frames: {result['frames_analyzed']}, "
                      f"Temporal spacing: {result['temporal_spacing_ns']:.1f}ns, "
                      f"Change rate: {result['change_rate']:.1%}, "
                      f"Dynamic residues: {result['dynamic_residues']}/{result['total_residues']} "
                      f"({result['dynamic_fraction']:.1%})")
        
        # Plot stride analysis with protein info
        plot_path = plot_stride_analysis(traj_name, stride_results, protein_info)
        
        # Store results
        all_results.append({
            'trajectory': traj_name,
            'protein_info': protein_info,
            'stride_results': stride_results,
            'plot_path': plot_path
        })
    
    # Summary analysis
    print("\n" + "=" * 80)
    print("📋 CURRICULUM STRIDE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for traj_result in all_results:
        traj_name = traj_result['trajectory']
        protein_info = traj_result['protein_info']
        
        if protein_info is not None and str(protein_info).strip():
            print(f"\n🎯 {traj_name} - {protein_info}:")
        else:
            print(f"\n🎯 {traj_name}:")
        
        for result in traj_result['stride_results']:
            if 'error' in result:
                print(f"   Stride {result['stride']}: ❌ {result['error']}")
            else:
                print(f"   Stride {result['stride']:3d} ({result['temporal_spacing_ns']:5.1f}ns): "
                      f"Change rate: {result['change_rate']:6.1%}, "
                      f"Dynamic: {result['dynamic_residues']:3d}/{result['total_residues']:3d} "
                      f"({result['dynamic_fraction']:5.1%})")
    
    # Cross-trajectory analysis
    print("\n📊 CROSS-TRAJECTORY STRIDE ANALYSIS:")
    print("-" * 60)
    
    for stride in CURRICULUM_STRIDES:
        print(f"\n🔍 Stride {stride}:")
        
        # Collect results for this stride across all trajectories
        stride_data = []
        for traj_result in all_results:
            for result in traj_result['stride_results']:
                if result['stride'] == stride and 'error' not in result:
                    stride_data.append({
                        'trajectory': traj_result['trajectory'],
                        'protein_info': traj_result['protein_info'],
                        'change_rate': result['change_rate'],
                        'dynamic_fraction': result['dynamic_fraction'],
                        'temporal_spacing_ns': result['temporal_spacing_ns']
                    })
        
        if stride_data:
            change_rates = [d['change_rate'] for d in stride_data]
            dynamic_fractions = [d['dynamic_fraction'] for d in stride_data]
            
            print(f"   📈 Change Rate: {np.mean(change_rates):.1%} ± {np.std(change_rates):.1%}")
            print(f"   🎯 Dynamic Fraction: {np.mean(dynamic_fractions):.1%} ± {np.std(dynamic_fractions):.1%}")
            print(f"   ⏱️  Temporal Spacing: {stride_data[0]['temporal_spacing_ns']:.1f}ns")
            
            # Show trajectory-specific results with clean protein names
            for data in stride_data:
                if data['protein_info'] is not None and str(data['protein_info']).strip():
                    protein_str = str(data['protein_info'])
                    if '\n' in protein_str:
                        first_line = protein_str.split('\n')[0].strip()
                        if 'receptor' in first_line.lower():
                            protein_name = 'Receptor'
                        elif 'protein' in first_line.lower():
                            protein_name = 'Protein'
                        elif 'enzyme' in first_line.lower():
                            protein_name = 'Enzyme'
                        else:
                            protein_name = first_line.split()[0] if first_line.split() else 'Unknown'
                    else:
                        protein_name = protein_str.split()[0] if protein_str.split() else 'Unknown'
                    
                    print(f"      {data['trajectory']} ({protein_name}): {data['change_rate']:.1%} changes, {data['dynamic_fraction']:.1%} dynamic")
                else:
                    print(f"      {data['trajectory']}: {data['change_rate']:.1%} changes, {data['dynamic_fraction']:.1%} dynamic")
        else:
            print(f"   ❌ No valid data for stride {stride}")
    
    # Save summary to results directory
    results_dir = Path(project_root) / "data" / "results" / "curriculum_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = results_dir / "curriculum_stride_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CURRICULUM STRIDE ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for traj_result in all_results:
            traj_name = traj_result['trajectory']
            protein_info = traj_result['protein_info']
            
            # Extract clean protein name for summary
            if protein_info is not None and str(protein_info).strip():
                protein_str = str(protein_info)
                if '\n' in protein_str:
                    first_line = protein_str.split('\n')[0].strip()
                    if 'receptor' in first_line.lower():
                        protein_name = 'Receptor'
                    elif 'protein' in first_line.lower():
                        protein_name = 'Protein'
                    elif 'enzyme' in first_line.lower():
                        protein_name = 'Enzyme'
                    else:
                        protein_name = first_line.split()[0] if first_line.split() else 'Unknown'
                else:
                    protein_name = protein_str.split()[0] if protein_str.split() else 'Unknown'
                
                f.write(f"🎯 {traj_name} - {protein_name}\n")
            else:
                f.write(f"🎯 {traj_name}:\n")
            
            for result in traj_result['stride_results']:
                if 'error' in result:
                    f.write(f"   Stride {result['stride']}: ❌ {result['error']}\n")
                else:
                    f.write(f"   Stride {result['stride']:3d} ({result['temporal_spacing_ns']:5.1f}ns): "
                           f"Change rate: {result['change_rate']:6.1%}, "
                           f"Dynamic: {result['dynamic_residues']:3d}/{result['total_residues']:3d} "
                           f"({result['dynamic_fraction']:5.1%})\n")
            f.write("\n")
    
    print(f"\n📄 Summary saved to: {summary_path}")
    
    return all_results

def main():
    """Main function with default data directory."""
    
    # Default data directory (adjust if needed)
    default_data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings/"
    
    print("🧪 Curriculum Learning Stride Analysis")
    print("=" * 80)
    print("🎯 This script analyzes cluster ID changes for curriculum learning strides:")
    print(f"   Strides: {CURRICULUM_STRIDES}")
    print(f"   Temporal spacing: {[s*0.2 for s in CURRICULUM_STRIDES]} ns")
    print("   🎲 Randomly selecting 2 trajectories for analysis")
    print("=" * 80)
    
    # Check if data directory exists
    if not os.path.exists(default_data_dir):
        print(f"❌ Default data directory not found: {default_data_dir}")
        print("\nPlease provide the correct path to your trajectory embeddings directory:")
        data_dir = input("Data directory path: ").strip()
        if not data_dir or not os.path.exists(data_dir):
            print("❌ Invalid path. Exiting.")
            return
    else:
        data_dir = default_data_dir
        print(f"📁 Using default data directory: {data_dir}")
    
    # Run the curriculum stride analysis
    results = test_curriculum_stride_analysis(data_dir)
    
    print("\n" + "=" * 80)
    print("🏁 Curriculum stride analysis completed!")
    print("\n💡 Insights:")
    print("   • Higher strides (125, 50) show long-term conformational changes")
    print("   • Lower strides (10, 1) show fine-grained dynamics")
    print("   • Change rates help tune curriculum learning parameters")
    print(f"   • Results saved to: {Path(project_root) / 'data' / 'results' / 'curriculum_analysis'}")

if __name__ == "__main__":
    main()
