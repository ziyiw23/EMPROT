import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- Add project root to Python path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from emprot.data.data_loader import LMDBLoader

def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two flattened numpy vectors."""
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    # Ensure vectors are not zero to avoid division by zero
    if np.all(vec1_flat == 0) or np.all(vec2_flat == 0):
        return 0.0
        
    dot_product = np.dot(vec1_flat, vec2_flat)
    norm_prod = np.linalg.norm(vec1_flat) * np.linalg.norm(vec2_flat)
    
    return dot_product / norm_prod

def analyze_trajectory(traj_path, num_pairs):
    """
    Analyzes a single trajectory to calculate correlations between consecutive frames.
    """
    embedding_correlations = []
    coord_correlations = []

    try:
        with LMDBLoader(traj_path) as loader:
            metadata = loader.get_metadata()
            num_frames = metadata['num_frames']
            
            if num_frames < 2:
                return [], []

            # Select random start frames to sample from the trajectory
            max_start_frame = num_frames - 2
            if max_start_frame < 1:
                return [],[]
            start_frames = [random.randint(0, max_start_frame) for _ in range(num_pairs)]

            for start_frame in start_frames:
                frame1_data = loader.load_frame(start_frame)
                frame2_data = loader.load_frame(start_frame + 1)

                # 1. Calculate correlation for embeddings
                emb1 = frame1_data['embeddings']
                emb2 = frame2_data['embeddings']
                emb_sim = calculate_cosine_similarity(emb1, emb2)
                embedding_correlations.append(emb_sim)

                # 2. Calculate correlation for 3D coordinates
                coords1 = frame1_data['atoms'][['x', 'y', 'z']].values
                coords2 = frame2_data['atoms'][['x', 'y', 'z']].values
                coord_sim = calculate_cosine_similarity(coords1, coords2)
                coord_correlations.append(coord_sim)
                
    except Exception as e:
        print(f"\nCould not process trajectory {os.path.basename(traj_path)}: {e}")
        return [], []
        
    return embedding_correlations, coord_correlations

def visualize_results(embedding_correlations, coord_correlations, output_file):
    """
    Visualizes the correlation distributions and saves the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot histograms
    sns.histplot(embedding_correlations, color="skyblue", label='Embedding Similarity', kde=True, ax=ax, stat='density', element='step')
    sns.histplot(coord_correlations, color="red", label='3D Coordinate Similarity', kde=True, ax=ax, stat='density', element='step')

    ax.set_title('Distribution of Consecutive Frame Similarities', fontsize=18, fontweight='bold')
    ax.set_xlabel('Cosine Similarity', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim(0.95, 1.0) # Zoom in on the high-similarity region

    # Add statistics to the plot
    emb_mean, emb_std = np.mean(embedding_correlations), np.std(embedding_correlations)
    coord_mean, coord_std = np.mean(coord_correlations), np.std(coord_correlations)

    stats_text = (
        f"Embeddings:\n"
        f"  Mean: {emb_mean:.6f}\n"
        f"  Std:  {emb_std:.6f}\n\n"
        f"Coordinates:\n"
        f"  Mean: {coord_mean:.6f}\n"
        f"  Std:  {coord_std:.6f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n📊 Visualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze frame-to-frame correlation in EmProt trajectories.')
    parser.add_argument('--data_dir', type=str, default='/scratch/groups/rbaltman/ziyiw23/traj_embeddings/',
                        help='Path to the directory containing trajectory LMDB databases.')
    parser.add_argument('--num_proteins', type=int, default=10,
                        help='Number of random proteins to sample for analysis.')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='Number of consecutive frame pairs to sample per protein.')
    parser.add_argument('--output_file', type=str, default='frame_similarity_analysis.png',
                        help='Path to save the output visualization plot.')
    args = parser.parse_args()

    print("--- Starting Frame Similarity Analysis ---")
    if not os.path.isdir(args.data_dir):
        print(f"❌ Error: Data directory not found at {args.data_dir}")
        return

    all_traj_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d))]

    if len(all_traj_dirs) < args.num_proteins:
        print(f"⚠️ Warning: Requested {args.num_proteins} proteins, but only found {len(all_traj_dirs)}. Using all.")
        args.num_proteins = len(all_traj_dirs)

    sampled_traj_paths = random.sample(all_traj_dirs, args.num_proteins)

    all_embedding_correlations = []
    all_coord_correlations = []

    print(f"Analyzing {args.num_pairs} frame pairs from {args.num_proteins} proteins...")
    for traj_path in tqdm(sampled_traj_paths, desc="Processing Proteins"):
        emb_corr, coord_corr = analyze_trajectory(traj_path, args.num_pairs)
        all_embedding_correlations.extend(emb_corr)
        all_coord_correlations.extend(coord_corr)

    if not all_embedding_correlations:
        print("\n❌ Analysis failed: No data was processed. Check trajectory paths and content.")
        return

    print("\n--- Analysis Summary ---")
    print(f"Total pairs analyzed: {len(all_embedding_correlations)}")
    print(f"Embedding Similarity -> Mean: {np.mean(all_embedding_correlations):.6f}, Std: {np.std(all_embedding_correlations):.6f}")
    print(f"Coordinate Similarity -> Mean: {np.mean(all_coord_correlations):.6f}, Std: {np.std(all_coord_correlations):.6f}")

    visualize_results(all_embedding_correlations, all_coord_correlations, args.output_file)

if __name__ == "__main__":
    main() 