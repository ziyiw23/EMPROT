import os
import re
import random
# Set threading environment variables BEFORE importing numpy or mdtraj
# This is a critical step to prevent deadlocks in multiprocessing on some systems.

# Note that the sampled pdb files that represent the trajectories won't contain bfactor:
# For Experimental Structures (e.g., from X-ray crystallography): 
    # The B-factor, or temperature factor, indicates the mobility or thermal fluctuation of an atom. 
    # A lower B-factor means an atom is more stable and its position is well-defined, 
    # while a higher B-factor indicates it's more mobile or disordered.

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import mdtraj as md
import numpy as np
import argparse
import time
from collections import defaultdict
from multiprocessing import cpu_count, Pool

random.seed(42)
NUM_FRAMES = 2500

def analyze_xtc_group(group_id, pdb_file, xtc_files):
    """Analyze all XTC files for a group (quality control)"""
    print(f"  Analyzing all XTC files for group {group_id}:")
    
    valid_xtcs = []
    frame_counts = []
    
    for i, xtc_file in enumerate(xtc_files):
        print(f"    XTC {i+1}: {os.path.basename(xtc_file)}")
        
        try:
            # Get frame count
            with md.open(xtc_file) as f:
                frame_count = len(f)
            
            print(f"      Frames: {frame_count}")
            
            # Check if frame count is exactly 2500 (strict validation)
            if frame_count != NUM_FRAMES:
                print(f"      WARNING: Frame count ({frame_count}) does not match expected ({NUM_FRAMES}). Skipping.")
                continue
            
            # Get basic trajectory info (similar to GROMACS gmx check)
            try:
                test_traj = md.load([xtc_file], top=pdb_file, stride=1)
                if len(test_traj) >= 2 and hasattr(test_traj, 'time') and test_traj.time is not None:
                    first_time = test_traj.time[0]
                    last_time = test_traj.time[-1]
                    duration = last_time - first_time
                    print(f"      Time range: {first_time:.1f} - {last_time:.1f} ps (duration: {duration:.1f} ps)")
            except Exception as e:
                print(f"      WARNING: Topology mismatch - {e}")
                print(f"      STATUS: INVALID (topology mismatch)")
                continue
            
            # This XTC passed all checks
            valid_xtcs.append(xtc_file)
            frame_counts.append(frame_count)
            print(f"      STATUS: VALID")
            
        except Exception as e:
            print(f"      ERROR: Could not determine frame count. {e}")
            continue
    
    print(f"  Summary for group {group_id}:")
    print(f"    Total XTC files found: {len(xtc_files)}")
    print(f"    Valid XTC files: {len(valid_xtcs)}")
    
    if len(valid_xtcs) == 0:
        print(f"    ERROR: No valid XTC files found for this group!")
        return None
    
    # Show frame count statistics
    if len(frame_counts) > 1:
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
        
        # All valid XTC files should have exactly 2500 frames
        if min_frames == max_frames and min_frames == NUM_FRAMES:
            print(f"    All valid XTC files have expected frame count: {NUM_FRAMES} ✓")
        else:
            print(f"    ERROR: Inconsistent frame counts among valid XTC files!")
    else:
        print(f"    Frame count: {frame_counts[0]}")
        if frame_counts[0] == NUM_FRAMES:
            print(f"    Frame count matches expectation ✓")
        else:
            print(f"    ERROR: Frame count does not match expected {NUM_FRAMES}")
    
    return valid_xtcs

def count_frames_in_group(complete_groups):
    print("=" * 60)
    
    total_frames_all = 0
    frame_counts = []
    groups_to_remove = []  # Track groups to remove instead of modifying during iteration
    
    for group_id, files in sorted(complete_groups.items()):
        pdb_file = files['pdb']
        xtc_files = files['xtc']
        
        print(f"Group {group_id}:")
        print(f"  PDB: {os.path.basename(pdb_file)}")
        
        # Use the new analysis function to get valid XTCs
        valid_xtcs = analyze_xtc_group(group_id, pdb_file, xtc_files)
        
        if valid_xtcs is None:
            print(f"  SKIPPING: No valid XTC files found")
            # Mark this group for removal instead of deleting during iteration
            groups_to_remove.append(group_id)
            continue
        
        # Update the group with only valid XTCs
        files['xtc'] = valid_xtcs
        
        # Calculate total frames from valid XTCs (each should have NUM_FRAMES)
        group_total = len(valid_xtcs) * NUM_FRAMES
        
        print(f"  Total valid XTC frames: {group_total:,}")
        print(f"  Expected samples: {group_total:,}")
        print()
        
        total_frames_all += group_total
        frame_counts.append(group_total)
    
    # Remove invalid groups after iteration is complete
    for group_id in groups_to_remove:
        del complete_groups[group_id]
    
    if len(frame_counts) == 0:
        print("=" * 60)
        print("ERROR: No valid trajectory groups found!")
        return
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total valid groups: {len(complete_groups)}")
    print(f"Total XTC frames across all groups: {total_frames_all:,}")
    print(f"Average frames per group: {sum(frame_counts)/len(frame_counts):,.0f}")
    print(f"Min frames in a group: {min(frame_counts):,}")
    print(f"Max frames in a group: {max(frame_counts):,}")
    print(f"Expected total samples: {total_frames_all:,}")

def group_filter_files(directory):
    """Groups PDB and XTC files based on a common identifier."""
    file_groups = defaultdict(lambda: {'pdb': None, 'xtc': []})
    
    # Regex to capture dynamics ID from filenames:
    # - PDB: '<fileid>_dyn_<dyn>.pdb'  → capture <dyn>
    # - XTC: 'd<dyn>_trj_<fileid>.xtc' OR 'd<dyn>_traj_<fileid>.xtc' → capture <dyn>
    pdb_re = re.compile(r'.*_dyn_(\d+)\.pdb$')
    xtc_re = re.compile(r'^d(\d+)_(?:trj|traj)_(\d+)\.xtc$')

    print(f"Scanning directory: {directory}")
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return {}
    
    print(f"Found {len(filenames)} total files and directories.")

    tmp_files = []
    pdb_files_found = 0
    xtc_files_found = 0
    for filename in filenames:
        if 'tmp' in filename:
            tmp_files.append(filename)
            continue

        pdb_match = pdb_re.match(filename)

        if pdb_match:
            group_id = pdb_match.group(1)
            file_groups[group_id]['pdb'] = os.path.join(directory, filename)
            pdb_files_found += 1
            continue

        xtc_match = xtc_re.match(filename)
        if xtc_match:
            group_id = xtc_match.group(1)  # dynamics id
            file_groups[group_id]['xtc'].append(os.path.join(directory, filename))
            xtc_files_found += 1
            
    print(f"Finished scanning. Found {pdb_files_found} PDB files and {xtc_files_found} XTC files.")

    # Filter for complete groups that have a PDB and at least one XTC file.
    complete_groups = {
        gid: g for gid, g in file_groups.items() 
        if g['pdb'] and g['xtc']
    }
    
    print("\n--- Grouping Analysis ---")
    print(f"Found {len(complete_groups)} complete groups (have both PDB and XTC).")
    print(f"Skipped {len(tmp_files)} tmp PDB file(s).")

    incomplete_groups = {
        gid: data for gid, data in file_groups.items() if not data['pdb'] or not data['xtc']
    }
    if incomplete_groups:
        print(f"Found {len(incomplete_groups)} incomplete groups (missing PDB or XTC):")
        for gid, data in incomplete_groups.items():
            missing = "PDB" if not data['pdb'] else "XTC"
            print(f"  Group {gid}: Missing {missing} file.")
    
    if len(complete_groups) <= 0:
        print("\nERROR: No complete trajectory groups found to process.")
        return {} # Return empty dict instead of raising error

    return complete_groups

def get_total_frames(xtc_file):
    """Calculate the total number of frames across multiple XTC files without loading them into memory."""
    total_frames = 0
    try:
        # md.open opens the file and reads metadata, but not all the coordinate data.
        # This is much more memory-efficient.
        with md.open(xtc_file) as f:
            total_frames += len(f)
    except Exception as e:
        print(f"  Warning: Could not read metadata from {os.path.basename(xtc_file)}: {e}")
    return total_frames

# --- Globals for Multiprocessing ---
# These are global to be inherited by child processes on fork, avoiding pickling
g_traj_to_save = None
g_output_subdir = None

def init_worker(traj, out_dir):
    """Initializer for the multiprocessing pool."""
    global g_traj_to_save, g_output_subdir
    g_traj_to_save = traj
    g_output_subdir = out_dir

def save_frame_worker(frame_idx):
    """Worker function to save a single PDB frame."""
    try:
        output_filename = os.path.join(g_output_subdir, f"frame_{frame_idx}.pdb")
        g_traj_to_save[frame_idx].save_pdb(output_filename)
        return True
    except Exception as e:
        print(f"Error in worker processing frame {frame_idx}: {e}")
        return False

def load_trajectory(pdb_file, xtc_file, validate_coordinates=True):
    """
    Load ALL frames from XTC file (no stride) - matches GROMACS script logic.
    Fixed to handle unit cell mismatches and API compatibility.
    
    Note: Minor time irregularities (~0.1% of frames) are acceptable for transformer training.
    If needed, can use MDGEN for interpolation post-training.
    """
    try:
        # Load ALL frames from the XTC file (stride=1, which is default)
        sampled_traj = md.load([xtc_file], top=pdb_file)
        print(f"  Loaded {len(sampled_traj)} frames from {os.path.basename(xtc_file)}")

        # Optional: Validate that coordinates are reasonable
        if validate_coordinates and len(sampled_traj) > 1:
            try:
                # Calculate RMSD between first and second XTC frame
                # Use only protein atoms and add least-squares fitting (like GROMACS)
                protein_atoms = sampled_traj.topology.select('protein')
                if len(protein_atoms) > 0:
                    # Select protein atoms only
                    frame0_protein = sampled_traj[0:1].atom_slice(protein_atoms)
                    frame1_protein = sampled_traj[1:2].atom_slice(protein_atoms)
                    
                    # Calculate RMSD with least-squares fitting (like GROMACS)
                    rmsd = md.rmsd(frame1_protein, frame0_protein, 0)[0]
                    print(f"  RMSD between first and second XTC frame (protein only): {rmsd:.3f} nm")
                    
                    # Warn if coordinates seem very different (might indicate a problem)
                    if rmsd > 1.0:  # 1.0 nm is quite large for consecutive frames
                        print(f"  WARNING: Large RMSD ({rmsd:.3f} nm) between consecutive frames. Check trajectory quality.")
                else:
                    print(f"  Warning: No protein atoms found for RMSD calculation")
            except Exception as e:
                print(f"  Warning: Could not calculate RMSD: {e}")

        # Handle unit cell mismatch by removing unit cell information
        if sampled_traj.unitcell_vectors is not None:
            print(f"  Handling unit cell mismatch...")
            # Remove unit cell information to avoid mixing error
            sampled_traj.unitcell_vectors = None
            sampled_traj.unitcell_lengths = None
            sampled_traj.unitcell_angles = None
            
            print(f"  Unit cell information removed from trajectory")

        # Check for time irregularities but don't fail on them
        if hasattr(sampled_traj, 'time') and sampled_traj.time is not None and len(sampled_traj.time) > 1:
            time_diffs = np.diff(sampled_traj.time)
            median_interval = np.median(time_diffs)
            irregular_frames = np.sum(np.abs(time_diffs - median_interval) > median_interval * 0.1)
            if irregular_frames > 0:
                print(f"  Note: {irregular_frames}/{len(time_diffs)} frames have irregular time intervals")
                print(f"  This is {irregular_frames/len(time_diffs)*100:.2f}% - acceptable for transformer training")

        # Return the XTC trajectory directly
        if hasattr(sampled_traj, 'time') and sampled_traj.time is not None and len(sampled_traj.time) > 0:
            print(f"  Time range: {sampled_traj.time[0]:.1f} to {sampled_traj.time[-1]:.1f} ps")

        return sampled_traj

    except Exception as e:
        print(f"  Error loading trajectory: {e}")
        print(f"  Attempting alternative loading strategy...")
        
        # Alternative strategy: Load only XTC trajectory without PDB frame 0
        try:
            sampled_traj = md.load([xtc_file], top=pdb_file)
            print(f"  Alternative: Loaded {len(sampled_traj)} frames from XTC only")
            
            # Remove unit cell information if present
            if sampled_traj.unitcell_vectors is not None:
                sampled_traj.unitcell_vectors = None
                sampled_traj.unitcell_lengths = None
                sampled_traj.unitcell_angles = None
                print(f"  Unit cell information removed from XTC trajectory")
                
            return sampled_traj
            
        except Exception as e2:
            print(f"  Alternative loading also failed: {e2}")
            return None

def get_timestep_info(xtc_file, pdb_file):
    """Get timestep information for reporting (matches GROMACS script logic)"""
    try:
        print(f"  Getting timestep info...")
        
        # Load first few frames to get timestep
        test_traj = md.load([xtc_file], top=pdb_file, stride=1)
        test_frames = test_traj[:min(10, len(test_traj))]
        
        if len(test_frames) >= 2 and hasattr(test_frames, 'time') and test_frames.time is not None:
            # Calculate timestep
            time_diffs = np.diff(test_frames.time)
            timestep = np.median(time_diffs)
            print(f"  Timestep: {timestep:.1f} ps")
            return timestep
        else:
            print(f"  Timestep: Unknown")
            return 200.0  # Default
            
    except Exception as e:
        print(f"  Error getting timestep: {e}")
        return 200.0  # Default

def process_trajectory(pdb_file, xtc_files, output_dir, num_workers):
    """Process a single trajectory group - matches GROMACS script logic."""
    pdb_basename = os.path.splitext(os.path.basename(pdb_file))[0]
    group_id = pdb_basename.split('_')[-1]  # Extract group ID from PDB name
    
    print(f"Processing group {group_id}...")
    print(f"  PDB: {os.path.basename(pdb_file)}")
    
    # Analyze all XTC files and get valid ones (matches GROMACS quality control)
    print(f"  Quality control check...")
    valid_xtcs = analyze_xtc_group(group_id, pdb_file, xtc_files)
    
    if valid_xtcs is None or len(valid_xtcs) == 0:
        print(f"  SKIPPING: No valid XTC files found")
        return False
    
    # Select XTC file to process (matches GROMACS random selection)
    if len(valid_xtcs) > 1:
        chosen_xtc = random.choice(valid_xtcs)
        print(f"  SELECTION: Randomly chose {os.path.basename(chosen_xtc)} from {len(valid_xtcs)} valid XTC files")
    else:
        chosen_xtc = valid_xtcs[0]
        print(f"  SELECTION: Using only valid XTC file: {os.path.basename(chosen_xtc)}")
    
    # Extract unique trajectory ID from chosen XTC filename (matches GROMACS naming)
    xtc_basename = os.path.basename(chosen_xtc)
    m_xtc = re.match(r'^d(\d+)_(?:trj|traj)_(\d+)\.xtc$', xtc_basename)
    if m_xtc:
        # Use the per-file unique id (second capture) for subdir naming
        unique_traj_id = m_xtc.group(2)
    else:
        # Fallback: use the full basename
        unique_traj_id = os.path.splitext(xtc_basename)[0]
    
    # Create output directory with unique trajectory ID for traceability
    output_subdir = os.path.join(output_dir, f"{pdb_basename}_traj_{unique_traj_id}")
    
    # Get final frame count for chosen XTC
    final_frame_count = get_total_frames(chosen_xtc)
    print(f"  PROCESSING: Will extract all {final_frame_count} frames from {os.path.basename(chosen_xtc)}")
    print(f"  OUTPUT DIR: {os.path.basename(output_subdir)} (trajectory ID: {unique_traj_id})")
    
    # If outputs already exist and appear complete, skip re-processing to enable incremental runs
    if os.path.isdir(output_subdir):
        existing_frames = len([fn for fn in os.listdir(output_subdir) if fn.startswith("frame_") and fn.endswith(".pdb")])
        if existing_frames >= max(1, final_frame_count):
            print(f"  SKIP: Found existing sampled frames ({existing_frames} >= {final_frame_count}) in '{output_subdir}'.")
            return True
    
    # Get timestep info (for information only, matches GROMACS)
    timestep_ps = get_timestep_info(chosen_xtc, pdb_file)
    
    try:
        start_time_total = time.time()

        print(f"  Will extract ALL {final_frame_count} frames")
        
        # Create output directory
        os.makedirs(output_subdir, exist_ok=True)

        # Load ALL frames from the trajectory (no stride)
        start_time = time.time()
        traj_to_save = load_trajectory(pdb_file, chosen_xtc)
        time_to_load = time.time() - start_time

        if traj_to_save is None or len(traj_to_save) == 0:
            print("  Trajectory loading failed or resulted in an empty trajectory. Skipping.")
            return False

        print(f"  Successfully loaded trajectory with {len(traj_to_save)} frames. (Took {time_to_load:.2f}s)")
        
        # Verify we got the expected number of frames (matches GROMACS verification)
        actual_samples = len(traj_to_save)
        if actual_samples != final_frame_count:
            print(f"  WARNING: Frame count mismatch! Got {actual_samples}, expected {final_frame_count}")
        else:
            print(f"  VERIFICATION: Frame count matches expectation ✓")

        # --- Parallel Saving ---
        start_save_time = time.time()
        
        # Initialize the pool with the trajectory data and output path
        # This makes them available to all worker processes without pickling
        # which is a common source of deadlocks with complex objects.
        initializer = init_worker
        initargs = (traj_to_save, output_subdir)
        
        with Pool(processes=num_workers, initializer=initializer, initargs=initargs) as pool:
            frame_indices = range(len(traj_to_save))
            
            # Use imap_unordered for efficiency, process results as they complete
            results = list(pool.imap_unordered(save_frame_worker, frame_indices))
            
            saved_count = sum(1 for r in results if r)

        time_to_save = time.time() - start_save_time
        print(f"Finished saving {saved_count} frames in parallel. (Took {time_to_save:.2f}s)")
        
        print(f"  SUCCESS: Saved {saved_count} frames (expected: {final_frame_count})")
        print(f"  Finished saving to '{output_subdir}'. (Took {time.time() - start_time_total:.2f}s total)")
        
        return True
        
    except Exception as e:
        print(f"  [CRITICAL] An unexpected error occurred in process_trajectory: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    """Main function to process trajectories - matches GROMACS script logic."""
    print("Starting main function.")
    directory = args.input_dir
    output_dir = args.output_dir
    num_workers = args.num_workers

    print("Grouping files...")
    file_groups = group_filter_files(directory)
    
    if not file_groups:
        print("No file groups found to process. Exiting.")
        return

    # Sort groups by ID for deterministic processing
    all_groups = sorted(file_groups.items())

    groups_to_process = all_groups
    print(f"\nFound {len(all_groups)} groups to process.")
    if args.debug:
        print("\n[DEBUG] Debug mode is ON. Processing will be limited to 1 group (29).")
        groups_to_process = [g for g in all_groups if g[0] == '29']
        print(f"Will process {len(groups_to_process)} group in debug mode.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputting sampled PDBs to: {output_dir}\n")

    total_to_process = len(groups_to_process)
    processed_count = 0
    
    for i, (group_id, files) in enumerate(groups_to_process):
        pdb_file = files['pdb']
        xtc_files = sorted(files['xtc'])  # Sort for consistent order

        print(f"--- Processing Group {i+1}/{total_to_process} (ID: {group_id}) ---")

        # Process the trajectory (quality control and random selection happens inside)
        # process_trajectory returns a boolean indicating success
        success = process_trajectory(pdb_file, xtc_files, output_dir, num_workers)
        if success:
            processed_count += 1
        
        print("----------------------------------------")
    
    print("Main function finished.")
    print(f"Successfully processed {processed_count}/{total_to_process} groups.")

if __name__ == '__main__':
    # This script requires the mdtraj library.
    # You can install it with: pip install mdtraj
    parser = argparse.ArgumentParser(
        description="Extract ALL frames from GROMACS XTC trajectories and save them as PDB files. "
                    "This script automatically groups one PDB file with its corresponding XTC files, then randomly selects one "
                    "of the valid XTC replicates for processing. Only XTC files with exactly NUM_FRAMES frames are processed.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='/oak/stanford/groups/rbaltman/ziyiw23/GPCR_trajectories/xtc',
        help='Directory containing your PDB and XTC files.'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='/oak/stanford/groups/rbaltman/ziyiw23/traj_sampled_pdbs',
        help='Directory where the sampled PDB files will be saved.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=max(1, cpu_count() // 2),
        help='Number of CPU cores to use for saving PDB files in parallel.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug information and process only first group.'
    )
    
    args = parser.parse_args()
    main(args)