#!/usr/bin/env python3
"""
Extract and display comprehensive metadata information about proteins.
This script provides detailed information about the proteins analyzed in the curriculum stride analysis.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from emprot.data.data_loader import LMDBLoader
    from emprot.data.metadata import MetadataManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def get_protein_info(lmdb_path, metadata_path="traj_metadata.csv"):
    """Extract protein information using the same method as comp_viz_input.py"""
    try:
        dynamic_id = os.path.basename(lmdb_path).split("_")[2]
        metadata = MetadataManager(metadata_path)
        protein_info = metadata.get_protein_info(dynamic_id)
        return protein_info
    except Exception as e:
        print(f"      ⚠️  Warning: Could not extract protein info: {e}")
        return None

def extract_protein_metadata(data_dir: str, trajectory_names: list):
    """
    Extract comprehensive metadata for specific protein trajectories.
    
    Args:
        data_dir: Path to trajectory embeddings directory
        trajectory_names: List of trajectory names to analyze
    """
    print("🔍 EXTRACTING PROTEIN METADATA")
    print("=" * 80)
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Analyzing {len(trajectory_names)} trajectories")
    print("=" * 80)
    
    metadata_results = []
    
    for i, traj_name in enumerate(trajectory_names, 1):
        print(f"\n{i}. {traj_name}")
        print("   " + "-" * 60)
        
        traj_path = os.path.join(data_dir, traj_name)
        
        # Extract protein information using the same method as comp_viz_input.py
        protein_info = get_protein_info(traj_path)
        if protein_info:
            print(f"   🧬 Protein: {protein_info}")
        else:
            print(f"   ⚠️  Protein info not available")
        
        try:
            with LMDBLoader(traj_path, read_only=True) as loader:
                # Get metadata
                metadata = loader.get_metadata()
                
                # Extract key information
                protein_data = {
                    'trajectory_name': traj_name,
                    'protein_info': protein_info,
                    'num_frames': metadata.get('num_frames', 'Unknown'),
                    'embedding_dim': metadata.get('embedding_dim', 'Unknown'),
                    'max_residues': metadata.get('max_residues', 'Unknown'),
                    'has_cluster_ids': False,
                    'cluster_id_range': None,
                    'protein_sequence': None,
                    'protein_name': None,
                    'pdb_id': None,
                    'chain_id': None,
                    'resolution': None,
                    'experimental_method': None,
                    'organism': None,
                    'additional_info': {}
                }
                
                # Check for cluster IDs
                try:
                    frame_data = loader.load_frame(0)
                    if 'cluster_ids' in frame_data:
                        cluster_ids = frame_data['cluster_ids']
                        protein_data['has_cluster_ids'] = True
                        
                        # Try to import torch for cluster ID analysis
                        try:
                            import torch
                            protein_data['cluster_id_range'] = f"{cluster_ids.min().item()} - {cluster_ids.max().item()}"
                            # Count unique cluster IDs
                            unique_clusters = len(torch.unique(cluster_ids))
                            protein_data['unique_clusters'] = unique_clusters
                        except ImportError:
                            # Fallback if torch not available
                            if hasattr(cluster_ids, 'min') and hasattr(cluster_ids, 'max'):
                                protein_data['cluster_id_range'] = f"{cluster_ids.min()} - {cluster_ids.max()}"
                            else:
                                protein_data['cluster_id_range'] = "Available (torch not available for analysis)"
                            protein_data['unique_clusters'] = "Unknown (torch not available)"
                except Exception as e:
                    print(f"      ⚠️  Error checking cluster IDs: {e}")
                
                # Try to extract protein sequence if available
                try:
                    if 'protein_sequence' in metadata:
                        protein_data['protein_sequence'] = metadata['protein_sequence']
                    elif 'sequence' in metadata:
                        protein_data['protein_sequence'] = metadata['sequence']
                    
                    # Extract protein name from trajectory name or metadata
                    if 'protein_name' in metadata:
                        protein_data['protein_name'] = metadata['protein_name']
                    elif 'name' in metadata:
                        protein_data['protein_name'] = metadata['name']
                    else:
                        # Try to parse from trajectory name
                        # Format: 10189_dyn_11_traj_10187
                        parts = traj_name.split('_')
                        if len(parts) >= 4:
                            protein_data['protein_name'] = f"Protein_{parts[0]}_{parts[2]}"
                    
                    # Extract PDB ID if available
                    if 'pdb_id' in metadata:
                        protein_data['pdb_id'] = metadata['pdb_id']
                    elif 'pdb' in metadata:
                        protein_data['pdb_id'] = metadata['pdb']
                    
                    # Extract chain ID
                    if 'chain_id' in metadata:
                        protein_data['chain_id'] = metadata['chain_id']
                    elif 'chain' in metadata:
                        protein_data['chain_id'] = metadata['chain']
                    
                    # Extract resolution
                    if 'resolution' in metadata:
                        protein_data['resolution'] = metadata['resolution']
                    
                    # Extract experimental method
                    if 'experimental_method' in metadata:
                        protein_data['experimental_method'] = metadata['experimental_method']
                    elif 'method' in metadata:
                        protein_data['experimental_method'] = metadata['method']
                    
                    # Extract organism
                    if 'organism' in metadata:
                        protein_data['organism'] = metadata['organism']
                    elif 'taxonomy' in metadata:
                        protein_data['organism'] = metadata['taxonomy']
                    
                    # Store any additional metadata
                    for key, value in metadata.items():
                        if key not in ['num_frames', 'embedding_dim', 'max_residues', 
                                     'protein_sequence', 'protein_name', 'pdb_id', 
                                     'chain_id', 'resolution', 'experimental_method', 'organism']:
                            protein_data['additional_info'][key] = value
                            
                except Exception as e:
                    print(f"      ⚠️  Error extracting protein metadata: {e}")
                
                # Display extracted information
                print(f"   📊 Basic Info:")
                print(f"      Frames: {protein_data['num_frames']}")
                print(f"      Embedding dim: {protein_data['embedding_dim']}")
                print(f"      Max residues: {protein_data['max_residues']}")
                
                if protein_data['has_cluster_ids']:
                    print(f"      Cluster IDs: {protein_data['cluster_id_range']}")
                    if 'unique_clusters' in protein_data:
                        print(f"      Unique clusters: {protein_data['unique_clusters']}")
                
                if protein_data['protein_name']:
                    print(f"   🧬 Protein Name: {protein_data['protein_name']}")
                
                if protein_data['pdb_id']:
                    print(f"   🏗️  PDB ID: {protein_data['pdb_id']}")
                    if protein_data['chain_id']:
                        print(f"      Chain: {protein_data['chain_id']}")
                
                if protein_data['resolution']:
                    print(f"   🔬 Resolution: {protein_data['resolution']} Å")
                
                if protein_data['experimental_method']:
                    print(f"   🧪 Method: {protein_data['experimental_method']}")
                
                if protein_data['organism']:
                    print(f"   🌱 Organism: {protein_data['organism']}")
                
                if protein_data['protein_sequence']:
                    seq = protein_data['protein_sequence']
                    if len(seq) > 100:
                        seq_display = seq[:50] + "..." + seq[-50:]
                    else:
                        seq_display = seq
                    print(f"   📝 Sequence: {seq_display}")
                    print(f"      Length: {len(seq)} residues")
                
                if protein_data['additional_info']:
                    print(f"   📋 Additional Info:")
                    for key, value in protein_data['additional_info'].items():
                        if isinstance(value, (str, int, float)) and len(str(value)) < 100:
                            print(f"      {key}: {value}")
                
                metadata_results.append(protein_data)
                
        except Exception as e:
            print(f"   ❌ Error reading trajectory: {e}")
            metadata_results.append({
                'trajectory_name': traj_name,
                'protein_info': protein_info,
                'error': str(e)
            })
    
    return metadata_results

def generate_metadata_report(metadata_results: list, output_file: str = "protein_metadata_report.txt"):
    """
    Generate a comprehensive metadata report file.
    
    Args:
        metadata_results: List of metadata dictionaries
        output_file: Output file path
    """
    print(f"\n📝 GENERATING METADATA REPORT")
    print("=" * 80)
    
    with open(output_file, 'w') as f:
        f.write("EMPROT PROTEIN METADATA REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report contains detailed metadata information for proteins analyzed\n")
        f.write("in the curriculum learning stride analysis.\n\n")
        
        f.write("CURRICULUM LEARNING STRIDE ANALYSIS PROTEINS\n")
        f.write("-" * 60 + "\n\n")
        
        for i, protein in enumerate(metadata_results, 1):
            f.write(f"{i}. {protein['trajectory_name']}\n")
            f.write("   " + "-" * 60 + "\n")
            
            if 'error' in protein:
                f.write(f"   ❌ Error: {protein['error']}\n\n")
                continue
            
            # Protein identification
            if protein['protein_info']:
                f.write(f"   🧬 PROTEIN IDENTIFICATION\n")
                f.write(f"      {protein['protein_info']}\n\n")
            
            # Basic information
            f.write(f"   📊 BASIC INFORMATION\n")
            f.write(f"      Trajectory frames: {protein['num_frames']}\n")
            f.write(f"      Embedding dimension: {protein['embedding_dim']}\n")
            f.write(f"      Maximum residues: {protein['max_residues']}\n")
            
            if protein['has_cluster_ids']:
                f.write(f"      Cluster ID range: {protein['cluster_id_range']}\n")
                if 'unique_clusters' in protein:
                    f.write(f"      Unique clusters: {protein['unique_clusters']}\n")
            
            f.write(f"\n")
            
            # Additional protein details
            if protein['protein_name']:
                f.write(f"   🧬 PROTEIN DETAILS\n")
                f.write(f"      Name: {protein['protein_name']}\n")
                
                if protein['pdb_id']:
                    f.write(f"      PDB ID: {protein['pdb_id']}")
                    if protein['chain_id']:
                        f.write(f" (Chain {protein['chain_id']})")
                    f.write(f"\n")
                
                if protein['resolution']:
                    f.write(f"      Resolution: {protein['resolution']} Å\n")
                
                if protein['experimental_method']:
                    f.write(f"      Experimental method: {protein['experimental_method']}\n")
                
                if protein['organism']:
                    f.write(f"      Organism: {protein['organism']}\n")
                
                f.write(f"\n")
            
            # Sequence information
            if protein['protein_sequence']:
                f.write(f"   📝 PROTEIN SEQUENCE\n")
                seq = protein['protein_sequence']
                f.write(f"      Length: {len(seq)} residues\n")
                
                # Show sequence in chunks
                chunk_size = 80
                for j in range(0, len(seq), chunk_size):
                    chunk = seq[j:j+chunk_size]
                    f.write(f"      {j+1:4d}-{j+len(chunk):4d}: {chunk}\n")
                
                f.write(f"\n")
            
            # Additional metadata
            if protein['additional_info']:
                f.write(f"   📋 ADDITIONAL METADATA\n")
                for key, value in protein['additional_info'].items():
                    if isinstance(value, (str, int, float)) and len(str(value)) < 200:
                        f.write(f"      {key}: {value}\n")
                f.write(f"\n")
            
            f.write(f"\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        successful = sum(1 for p in metadata_results if 'error' not in p)
        total = len(metadata_results)
        
        f.write(f"Total trajectories analyzed: {total}\n")
        f.write(f"Successful metadata extraction: {successful}\n")
        f.write(f"Failed extractions: {total - successful}\n\n")
        
        # Count proteins with protein info
        proteins_with_info = sum(1 for p in metadata_results if p.get('protein_info'))
        f.write(f"Proteins with identification: {proteins_with_info}/{total}\n")
        
        if successful > 0:
            # Calculate average sequence length
            seq_lengths = [len(p['protein_sequence']) for p in metadata_results 
                          if 'protein_sequence' in p and p['protein_sequence']]
            if seq_lengths:
                f.write(f"Average protein sequence length: {sum(seq_lengths)/len(seq_lengths):.1f} residues\n")
                f.write(f"Sequence length range: {min(seq_lengths)} - {max(seq_lengths)} residues\n\n")
            
            # Count proteins with PDB IDs
            pdb_proteins = sum(1 for p in metadata_results if 'pdb_id' in p and p['pdb_id'])
            f.write(f"Proteins with PDB IDs: {pdb_proteins}/{successful}\n")
            
            # Count proteins with cluster IDs
            cluster_proteins = sum(1 for p in metadata_results if p.get('has_cluster_ids', False))
            f.write(f"Proteins with cluster IDs: {cluster_proteins}/{successful}\n")
    
    print(f"✅ Metadata report saved to: {output_file}")
    return output_file

def main():
    """Main function to extract protein metadata."""
    
    # Default data directory
    default_data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings/"
    
    print("🧪 PROTEIN METADATA EXTRACTION")
    print("=" * 80)
    print("🎯 This script extracts comprehensive metadata for proteins analyzed")
    print("   in the curriculum learning stride analysis.")
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
    
    # Trajectories from the curriculum stride analysis
    trajectory_names = [
        "10189_dyn_11_traj_10187",
        "10205_dyn_13_traj_10202", 
        "10213_dyn_14_traj_10211",
        "10224_dyn_15_traj_10221",
        "10231_dyn_16_traj_10227"
    ]
    
    print(f"\n🎯 Analyzing {len(trajectory_names)} trajectories from curriculum stride analysis")
    
    # Extract metadata
    metadata_results = extract_protein_metadata(data_dir, trajectory_names)
    
    # Generate report
    output_file = generate_metadata_report(metadata_results)
    
    print(f"\n" + "=" * 80)
    print("🏁 Protein metadata extraction completed!")
    print(f"📄 Detailed report saved to: {output_file}")
    print("\n💡 The report contains:")
    print("   • Protein identification and PDB information")
    print("   • Sequence details and structural metadata")
    print("   • Cluster ID analysis results")
    print("   • Summary statistics across all proteins")

if __name__ == "__main__":
    main()
