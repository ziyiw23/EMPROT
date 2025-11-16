#!/usr/bin/env python3
"""
Pre-process cluster IDs for EMPROT training.

This script:
1. Loads the fitted sklearn k-means model
2. Iterates through every protein trajectory in the LMDB database
3. Assigns cluster IDs to every residue in every frame
4. Saves the cluster IDs back to the LMDB database

This eliminates the need for on-the-fly cluster assignment during training,
which was causing the classification head to receive zero gradients.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm
import pickle

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.preprocess.cluster_lookup import ClusterCentroidLookup
from emprot.data.data_loader import LMDBLoader
# from emprot.utils.logging import setup_logging  # This module doesn't exist

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Pre-process cluster IDs for EMPROT training dataset"
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to the LMDB dataset directory'
    )
    
    parser.add_argument(
        '--cluster_model_path',
        type=str,
        required=True,
        help='Path to the pickled sklearn k-means model (.pkl file)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Chunk size for processing residues during cluster assignment (default: 1000)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for cluster assignment (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run: analyze dataset without writing changes'
    )
    
    parser.add_argument(
        '--max_trajectories',
        type=int,
        default=None,
        help='Maximum number of trajectories to process (for testing)'
    )
    
    return parser

def load_cluster_model(cluster_model_path: str, device: str) -> ClusterCentroidLookup:
    """Load the fitted sklearn k-means model and create cluster lookup."""
    logger = logging.getLogger(__name__)
    
    logger.info(f" Loading cluster model from: {cluster_model_path}")
    
    try:
        with open(cluster_model_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        
        if not hasattr(kmeans_model, 'cluster_centers_'):
            raise AttributeError("Loaded object does not have 'cluster_centers_' attribute")
        
        # Create cluster lookup
        num_clusters = kmeans_model.n_clusters
        embedding_dim = kmeans_model.n_features_in_
        
        logger.info(f"SUCCESS: Loaded k-means model:")
        logger.info(f"   Number of clusters: {num_clusters:,}")
        logger.info(f"   Embedding dimension: {embedding_dim}")
        
        cluster_lookup = ClusterCentroidLookup(
            num_clusters=num_clusters,
            embedding_dim=embedding_dim,
            device=device
        )
        
        # Load centroids directly from the model
        cluster_lookup.load_centroids_from_sklearn(cluster_model_path)
        
        return cluster_lookup
        
    except Exception as e:
        logger.error(f"Failed to load cluster model: {e}")
        raise

def analyze_dataset(data_dir: str) -> Dict[str, Any]:
    """Analyze the nested LMDB dataset structure."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing nested dataset structure in: {data_dir}")
    
    # Find all trajectory subdirectories
    try:
        trajectory_paths = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
    except FileNotFoundError:
        logger.error(f"Data directory not found at: {data_dir}")
        return {}
    
    logger.info(f"   Found {len(trajectory_paths):,} trajectory directories")
    
    # Analyze first few trajectories to understand structure
    sample_trajectories = trajectory_paths[:3]  # Sample first 3
    total_frames = 0
    sample_structure = {}
    
    for traj_path in sample_trajectories:
        logger.info(f"   Analyzing trajectory: {traj_path.name}")
        
        try:
            # Use LMDBLoader to properly handle gzipped data
            with LMDBLoader(str(traj_path), read_only=True) as loader:
                metadata = loader.get_metadata()
                num_frames = metadata['num_frames']
                total_frames += num_frames
                
                logger.info(f"     Frames in {traj_path.name}: {num_frames:,}")
                
                # Sample first frame structure
                frame_data = loader.load_frame(0)  # Load first frame
                
                logger.info(f"     Frame 0 structure:")
                logger.info(f"       Keys: {list(frame_data.keys())}")
                
                if 'embeddings' in frame_data:
                    embeddings = frame_data['embeddings']
                    logger.info(f"       Embeddings shape: {embeddings.shape}")
                    logger.info(f"       Embeddings dtype: {embeddings.dtype}")
                
                if 'targets' in frame_data:
                    targets = frame_data['targets']
                    logger.info(f"       Targets keys: {list(targets.keys())}")
                    if 'short_term' in targets:
                        logger.info(f"       Short-term targets shape: {targets['short_term'].shape}")
                
                sample_structure[traj_path.name] = frame_data
            
        except Exception as e:
            logger.error(f"Error analyzing {traj_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"   Total frames across sample trajectories: {total_frames:,}")
    logger.info(f"   Estimated total frames (179 trajectories × 2500 frames): {179 * 2500:,}")
    
    return {
        'num_trajectories': len(trajectory_paths),
        'sample_trajectories': [p.name for p in sample_trajectories],
        'total_frames_sample': total_frames,
        'estimated_total_frames': 179 * 2500,
        'sample_structure': sample_structure
    }

def process_trajectory_embeddings(
    trajectory_embeddings: torch.Tensor,
    cluster_lookup: ClusterCentroidLookup,
    batch_size: int,
    device: str
) -> np.ndarray:
    """Process a whole trajectory's embeddings to assign cluster IDs."""
    logger = logging.getLogger(__name__)

    # Ensure the input is a 3D tensor (frames, residues, embed_dim)
    if trajectory_embeddings.dim() != 3:
        raise ValueError(f"Expected a 3D tensor for trajectory embeddings, got shape: {trajectory_embeddings.shape}")

    num_frames, num_residues, embed_dim = trajectory_embeddings.shape
    
    # Flatten all residues from all frames into a single large batch
    embeddings_flat = trajectory_embeddings.view(-1, embed_dim)
    logger.info(f"Flattened {num_frames * num_residues:,} total residue embeddings for processing.")
    
    # Use the memory-efficient batch_assign_to_clusters.
    # The 'batch_size' argument here is the chunk size for this internal processing.
    cluster_ids_flat = cluster_lookup.batch_assign_to_clusters(embeddings_flat)
    
    # Reshape the flat cluster IDs back to the original trajectory shape
    cluster_ids = cluster_ids_flat.view(num_frames, num_residues)
    
    # Convert to numpy for storage
    cluster_ids_np = cluster_ids.cpu().numpy()
    
    logger.info(f"Generated cluster IDs: {cluster_ids_np.shape}")
    logger.info(f"Cluster ID range: {cluster_ids_np.min()} to {cluster_ids_np.max()}")
    
    return cluster_ids_np

def preprocess_dataset(
    data_dir: str,
    cluster_lookup: ClusterCentroidLookup,
    batch_size: int, # This is the residue processing chunk size (kept for API, not used directly here)
    device: str,
    dry_run: bool = False,
    max_trajectories: Optional[int] = None
) -> None:
    """Main preprocessing function that iterates through trajectory subdirectories."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"   Starting cluster ID preprocessing...")
    logger.info(f"   Data directory: {data_dir}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Dry run: {dry_run}")
    
    try:
        trajectory_paths = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
    except FileNotFoundError:
        logger.error(f" Data directory not found at: {data_dir}")
        return

    if max_trajectories:
        trajectory_paths = trajectory_paths[:max_trajectories]
        logger.info(f"   Processing first {len(trajectory_paths)} trajectories (testing mode)")

    logger.info(f"   Found {len(trajectory_paths):,} total trajectory directories to process.")

    overall_processed = 0
    overall_errors = 0

    for traj_path in tqdm(trajectory_paths, desc="Processing Trajectories"):
        logger.info(f"--- Processing trajectory: {traj_path.name} ---")
        try:
            # First pass: inspect metadata and collect embeddings for all frames
            with LMDBLoader(str(traj_path), read_only=True) as loader:
                metadata = loader.get_metadata()
                num_frames = metadata['num_frames']
                
                logger.info(f"     Processing {num_frames:,} frames...")
                
                # Collect embeddings for all frames, allowing variable residue counts
                all_embeddings_list = []
                lengths = []
                for i in range(num_frames):
                    frame_data = loader.load_frame(i)
                    emb = np.asarray(frame_data['embeddings'], dtype=np.float32)
                    all_embeddings_list.append(emb)
                    lengths.append(emb.shape[0])
            
            # Flatten over frames: (total_residues, embed_dim)
            try:
                embeddings_flat = np.concatenate(all_embeddings_list, axis=0)
            except Exception as e_concat:
                logger.error(f"     Failed to concatenate embeddings for {traj_path.name}: {e_concat}")
                overall_errors += 1
                continue

            logger.info(f"     Total residue embeddings to cluster: {embeddings_flat.shape[0]:,}")

            # Assign cluster IDs in a single batched call
            emb_t = torch.from_numpy(embeddings_flat).float().to(device)
            cluster_ids_flat = cluster_lookup.batch_assign_to_clusters(emb_t)
            cluster_ids_flat_np = cluster_ids_flat.cpu().numpy()

            # Split flat IDs back into per-frame arrays
            cluster_ids_per_frame = []
            offset = 0
            for L in lengths:
                cluster_ids_per_frame.append(cluster_ids_flat_np[offset:offset + L])
                offset += L

            logger.info(f"     Generated cluster IDs for {len(cluster_ids_per_frame)} frames")
            
            # Write the new data back to the LMDB
            if not dry_run:
                with LMDBLoader(str(traj_path), read_only=False) as writer:
                    for i in range(num_frames):
                        frame_data = writer.load_frame(i)
                        frame_data['cluster_ids'] = cluster_ids_per_frame[i]
                        writer.add_frame(i, frame_data)
                
                logger.info(f"SUCCESS: Updated {num_frames:,} frames with cluster IDs")
            
            overall_processed += 1
            
        except Exception as e:
            logger.error(f" Failed to process entire trajectory {traj_path.name}: {e}")
            import traceback
            traceback.print_exc()
            overall_errors += 1
            continue

    logger.info(f" Preprocessing completed!")
    logger.info(f"   Successfully processed trajectories: {overall_processed:,}")
    logger.info(f"   Total errors: {overall_errors}")
    if dry_run:
        logger.info(" This was a dry run - no changes were written.")

def main():
    """Main function."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(" EMPROT Cluster ID Preprocessing")
    logger.info("=" * 50)
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        logger.error(f" Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.cluster_model_path):
        logger.error(f" Cluster model file does not exist: {args.cluster_model_path}")
        sys.exit(1)
    
    # Load cluster model
    try:
        cluster_lookup = load_cluster_model(args.cluster_model_path, args.device)
    except Exception as e:
        logger.error(f" Failed to load cluster model: {e}")
        sys.exit(1)
    
    # Analyze dataset first
    try:
        dataset_info = analyze_dataset(args.data_dir)
        logger.info("Dataset analysis completed successfully")
    except Exception as e:
        logger.error(f" Dataset analysis failed: {e}")
        sys.exit(1)
    
    # Preprocess dataset
    try:
        preprocess_dataset(
            data_dir=args.data_dir,
            cluster_lookup=cluster_lookup,
            batch_size=args.batch_size,
            device=args.device,
            dry_run=args.dry_run,
            max_trajectories=args.max_trajectories
        )
        logger.info(" Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f" Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
