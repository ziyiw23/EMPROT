#!/usr/bin/env python3
"""
Script to inspect an input LMDB file for EMPROT.
Visualizes COLLAPSE embeddings and cluster IDs.

Usage:
    python scripts/inspect_lmdb.py --path /path/to/traj_folder
    python scripts/inspect_lmdb.py --name protein_name --data_dir /path/to/data
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emprot.data.data_loader import LMDBLoader

def inspect_lmdb(path: str, num_frames: int = 3):
    print(f"Inspecting LMDB at: {path}")
    
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return

    try:
        with LMDBLoader(path) as loader:
            meta = loader.get_metadata()
            print("\n=== Metadata ===")
            for k, v in meta.items():
                print(f"  {k}: {v}")
            
            total_frames = meta.get('num_frames', 0)
            print(f"\nTotal Frames: {total_frames}")
            
            indices = list(range(min(num_frames, total_frames)))
            print(f"\n=== Inspecting first {len(indices)} frames ===")
            
            for idx in indices:
                print(f"\n--- Frame {idx} ---")
                frame = loader.load_frame(idx)
                
                # Print available keys
                print(f"Keys: {list(frame.keys())}")
                
                # Embeddings (COLLAPSE)
                if 'embeddings' in frame:
                    emb = frame['embeddings']
                    if isinstance(emb, (np.ndarray, torch.Tensor)):
                        print(f"embeddings shape: {emb.shape}")
                        print(f"embeddings stats: mean={emb.mean():.4f}, std={emb.std():.4f}, min={emb.min():.4f}, max={emb.max():.4f}")
                        
                        # Detail Table
                        print(f"\n  {'Res':<4} | {'Cluster':<8} | {'Embedding (First 8 dims)'}")
                        print("  " + "-"*75)
                        
                        cids_val = frame.get('cluster_ids', [])
                        
                        for r in range(min(10, len(cids_val))):
                            # Handle Embeddings
                            e_val = emb[r, :8]
                            if hasattr(e_val, 'tolist'):
                                e_val = e_val.tolist()
                            e_str = ", ".join([f"{x:.3f}" for x in e_val])
                            
                            # Handle Cluster ID
                            cid_print = cids_val[r]
                            if hasattr(cid_print, 'item'):
                                cid_print = cid_print.item()
                                
                            print(f"  {r:<4} | {cid_print:<8} | [{e_str}]")
                    else:
                        print(f"embeddings: {type(emb)}")
                else:
                    print("embeddings: NOT FOUND")
                    
    except Exception as e:
        print(f"Error reading LMDB: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Inspect EMPROT LMDB files")
    parser.add_argument('--path', type=str, help='Full path to the trajectory folder (containing data.mdb)')
    parser.add_argument('--name', type=str, help='Name of the trajectory/protein folder (searches in data_dir)')
    parser.add_argument('--data_dir', type=str, default='/oak/stanford/groups/rbaltman/ziyiw23/traj_embeddings/',
                        help='Base data directory')
    parser.add_argument('--frames', type=int, default=3, help='Number of frames to inspect')

    args = parser.parse_args()

    target_path = args.path
    if not target_path and args.name:
        target_path = os.path.join(args.data_dir, args.name)
    
    if not target_path:
        print("Error: Must provide either --path or --name")
        parser.print_help()
        return

    inspect_lmdb(target_path, args.frames)

if __name__ == "__main__":
    main()
