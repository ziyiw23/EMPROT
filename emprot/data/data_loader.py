import lmdb
import pickle
import os
import io
import numpy as np
from typing import Dict, Tuple, Optional, Dict as TypingDict, Any

class LMDBLoader:
    """
    Utility class for loading and writing data from/to LMDB databases 
    containing protein trajectory embeddings.
    """
    
    def __init__(self, traj_path: str, read_only: bool = True):
        """
        Initialize LMDB loader for a specific trajectory path.
        
        Args:
            traj_path: Path to the LMDB database directory
            read_only: Whether to open the database in read-only mode.
        """
        self.traj_path = traj_path
        self.read_only = read_only
        self.env = None
        self._metadata = None
        self._sorted_lmdb_keys = None
    
    def __enter__(self):
        """Context manager entry - open LMDB environment."""
        if not self.read_only:
            os.makedirs(self.traj_path, exist_ok=True)
            
        # Map size: allow override via env, default larger for write mode
        try:
            env_map_size = int(os.environ.get('EMPROT_LMDB_MAP_SIZE', '0') or 0)
        except Exception:
            env_map_size = 0
        map_size = env_map_size if env_map_size > 0 else (int(1e9) if self.read_only else int(64e9))

        self.env = lmdb.open(
            self.traj_path,
            readonly=self.read_only,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=map_size,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close LMDB environment."""
        if self.env:
            self.env.close()
            self.env = None
    
    def get_metadata(self) -> Dict:
        """
        Get metadata for the protein trajectory.
        
        Returns:
            Dictionary containing metadata about the trajectory
        """
        if self._metadata is not None:
            return self._metadata
        
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        
        with self.env.begin(buffers=True) as txn:
            id_to_idx_data = txn.get(b'id_to_idx')
            sorted_lmdb_keys = None
            if id_to_idx_data is not None:
                try:
                    id_to_idx = pickle.loads(id_to_idx_data)
                    try:
                        items = sorted(id_to_idx.items(), key=lambda kv: int(kv[1]))
                    except Exception:
                        items = sorted(id_to_idx.items(), key=lambda kv: kv[1])
                    sorted_lmdb_keys = [str(v) for _, v in items]
                except Exception:
                    sorted_lmdb_keys = None
            if not sorted_lmdb_keys:
                keys = []
                cur = txn.cursor()
                for k, _ in cur:
                    try:
                        ks = (k.decode() if isinstance(k, (bytes, bytearray, memoryview)) else str(k))
                    except Exception:
                        try:
                            ks = bytes(k).decode()
                        except Exception:
                            continue
                    if ks == 'id_to_idx':
                        continue
                    keys.append(ks)
                if not keys:
                    raise ValueError(f"Could not infer LMDB frame keys in {self.traj_path}")
                try:
                    sorted_lmdb_keys = sorted(keys, key=lambda s: int(s))
                except Exception:
                    import re as _re
                    def _num_token(s):
                        m = _re.findall(r"(\d+)$", s)
                        return int(m[0]) if m else 0
                    try:
                        sorted_lmdb_keys = sorted(keys, key=_num_token)
                    except Exception:
                        sorted_lmdb_keys = sorted(keys)

            num_examples = len(sorted_lmdb_keys)
            if num_examples == 0:
                raise ValueError(f"No valid frames found in {self.traj_path}")

            first_frame_key = sorted_lmdb_keys[0].encode()
            first_frame_data = txn.get(first_frame_key)
            if first_frame_data is None:
                raise ValueError(f"No data found for first frame key '{sorted_lmdb_keys[0]}' in {self.traj_path}")

            frame_dict = self._load_frame_data(first_frame_data)
            if 'cluster_ids' in frame_dict:
                num_residues = int(np.array(frame_dict['cluster_ids']).shape[0])
                embedding_dim = int(np.array(frame_dict.get('embeddings')).shape[1]) if 'embeddings' in frame_dict else 0
            else:
                num_residues = int(np.array(frame_dict['embeddings']).shape[0])
                embedding_dim = int(np.array(frame_dict['embeddings']).shape[1])
        
        traj_name = os.path.basename(os.path.normpath(self.traj_path))

        self._metadata = {
            'num_frames': num_examples,
            'num_residues': num_residues,
            'embedding_dim': embedding_dim,
            'path': self.traj_path,
            'sorted_lmdb_keys': sorted_lmdb_keys,
            'traj_name': traj_name,
        }
        
        return self._metadata
    
    def load_frame(self, frame_idx: int) -> Dict:
        """
        Load a single frame from the trajectory.
        
        Args:
            frame_idx: Index of the frame to load
            
        Returns:
            Dictionary containing frame data
        """
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        
        metadata = self.get_metadata()
        sorted_lmdb_keys = metadata['sorted_lmdb_keys']
        
        if frame_idx >= len(sorted_lmdb_keys):
            raise ValueError(f"Frame index {frame_idx} is out of bounds for protein with {len(sorted_lmdb_keys)} frames.")
        
        with self.env.begin() as txn:
            frame_key = sorted_lmdb_keys[frame_idx].encode()
            frame_data = txn.get(frame_key)
            
            if frame_data is None:
                raise ValueError(f"No data for frame_key '{frame_key.decode()}' (index {frame_idx}) in {self.traj_path}")
            
            return self._load_frame_data(frame_data)

    def get_raw_frame(self, frame_idx: int) -> bytes:
        """
        Return raw bytes for a given frame index without decoding.
        Useful for out-of-band metadata extraction.
        """
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        metadata = self.get_metadata()
        sorted_lmdb_keys = metadata['sorted_lmdb_keys']
        if frame_idx >= len(sorted_lmdb_keys):
            raise ValueError(f"Frame index {frame_idx} is out of bounds for protein with {len(sorted_lmdb_keys)} frames.")
        with self.env.begin() as txn:
            frame_key = sorted_lmdb_keys[frame_idx].encode()
            frame_data = txn.get(frame_key)
            if frame_data is None:
                raise ValueError(f"No data for frame_key '{frame_key.decode()}' (index {frame_idx}) in {self.traj_path}")
            return frame_data
    
    def load_sequence(self, start_frame: int, seq_len: int, time_step: float = 0.2) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Load a sequence window and return only times (classification-only)."""
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        
        metadata = self.get_metadata()
        sorted_lmdb_keys = metadata['sorted_lmdb_keys']
        times_list = []
        
        with self.env.begin() as txn:
            for i in range(seq_len):
                frame_idx = start_frame + i
                
                if frame_idx >= len(sorted_lmdb_keys):
                    raise ValueError(f"Frame index {frame_idx} is out of bounds for protein with {len(sorted_lmdb_keys)} frames.")
                
                frame_key = sorted_lmdb_keys[frame_idx].encode()
                frame_data = txn.get(frame_key)
                
                if frame_data is None:
                    raise ValueError(f"No data for frame_key '{frame_key.decode()}' (index {frame_idx}) in {self.traj_path}")
                
                _ = self._load_frame_data(frame_data)
                times_list.append(frame_idx * time_step)
        times = np.array(times_list)
        return None, times
    
    def _load_frame_data(self, frame_data: bytes) -> Dict:
        """
        Load frame data from bytes, handling gzip, lz4, or raw pickle based on header.

        Includes a compatibility unpickler that maps numpy._core.* modules
        (NumPy 2.x pickles) to numpy.core.* so they can be loaded under
        NumPy 1.x environments.
        """
        if isinstance(frame_data, memoryview):
            frame_data = frame_data.tobytes()

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Map NumPy 2.x module path to 1.x
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core', 1)
                return super().find_class(module, name)

        def _loads_compat(data: bytes) -> Dict:
            try:
                return _CompatUnpickler(io.BytesIO(data)).load()
            except Exception:
                # Fall back to default loader
                return pickle.loads(data)

        if len(frame_data) >= 2 and frame_data[:2] == b"\x1f\x8b":
            import gzip as _gzip
            return _loads_compat(_gzip.decompress(frame_data))
        # lz4 frame magic: 0x04 0x22 0x4D 0x18
        if len(frame_data) >= 4 and frame_data[:4] == b"\x04\x22\x4d\x18":
            try:
                import lz4.frame as _lz4f
                return _loads_compat(_lz4f.decompress(frame_data))
            except Exception:
                # If lz4 is unavailable, fall through to raw pickle
                pass
        return _loads_compat(frame_data)

    def load_frames(self, frame_indices: list) -> list:
        """Load multiple frames within a single read-only transaction.

        Args:
            frame_indices: List of integer indices to load

        Returns:
            List[Dict]: Decoded frame dictionaries in the same order
        """
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        metadata = self.get_metadata()
        sorted_lmdb_keys = metadata['sorted_lmdb_keys']
        results = []
        keys = []
        for frame_idx in frame_indices:
            if frame_idx >= len(sorted_lmdb_keys):
                raise ValueError(f"Frame index {frame_idx} is out of bounds for protein with {len(sorted_lmdb_keys)} frames.")
            keys.append(sorted_lmdb_keys[frame_idx].encode())
        with self.env.begin(write=False, buffers=True) as txn:
            blobs = [txn.get(k) for k in keys]
        for blob in blobs:
            if blob is None:
                raise ValueError(f"Missing frame blob in {self.traj_path}")
            results.append(self._load_frame_data(blob))
        return results

    def add_frame(self, frame_idx: int, frame_data: TypingDict[str, Any]) -> None:
        """
        Overwrite an existing frame in the LMDB with updated data (e.g., adding cluster_ids).
        """
        if not self.env:
            raise RuntimeError("LMDB environment not open. Use as context manager.")
        if self.read_only:
            raise RuntimeError("LMDBLoader is in read-only mode; cannot add_frame.")

        metadata = self.get_metadata()
        sorted_lmdb_keys = metadata['sorted_lmdb_keys']
        if frame_idx >= len(sorted_lmdb_keys):
            raise ValueError(
                f"Frame index {frame_idx} is out of bounds for protein with {len(sorted_lmdb_keys)} frames."
            )

        key = sorted_lmdb_keys[frame_idx].encode()
        blob = pickle.dumps(frame_data, protocol=pickle.HIGHEST_PROTOCOL)
        with self.env.begin(write=True) as txn:
            ok = txn.put(key, blob)
            if not ok:
                raise RuntimeError(f"Failed to write frame {frame_idx} to {self.traj_path}")

if __name__ == "__main__":
    pass
