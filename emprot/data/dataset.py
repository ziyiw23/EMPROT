import os
import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from emprot.data.data_loader import LMDBLoader
from emprot.data.metadata import MetadataManager
from emprot.data.sampling import collate_variable_length

class ProteinTrajectoryDataset(Dataset):
    """Classification-only dataset emitting fixed-length windows of cluster IDs."""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_path: str,
                 num_full_res_frames: int = 5,
                 history_prefix_frames: int = 0,
                 time_step: float = 0.2,
                 stride: int = 5,
                 future_horizon: int = 1,
                 window_start_stride: int = 1,
                 change_target_fraction: float = 0.0,
                 change_probe_interval: int = 20,
                 dynamic_score_mode: str = 'f0_any',
                 dynamic_weight_gamma: float = 0.0,
                 dynamic_epsilon: float = 1e-3,
                 score_f_max: int = 5,
                 sample_with_replacement: bool = False,

                 all_protein_metadata: Optional[List[Dict]] = None,
                 active_protein_indices: Optional[List[int]] = None):
        
        self.data_dir = data_dir
        self.num_full_res_frames = int(num_full_res_frames)  # K (recent full-resolution)
        self.history_prefix_frames = int(max(0, history_prefix_frames))  # L (older history for summarizer)
        assert self.num_full_res_frames >= 1, "num_full_res_frames must be >= 1"
        
        self.time_step = time_step
        self.stride = int(stride)
        self.future_horizon = max(1, int(future_horizon))
        self.window_start_stride = max(1, int(window_start_stride))
        self.change_target_fraction = float(max(0.0, min(1.0, change_target_fraction)))
        self.change_probe_interval = max(1, int(change_probe_interval))
        self.dynamic_score_mode = (dynamic_score_mode or 'f0_any').lower()
        self.dynamic_weight_gamma = float(max(0.0, dynamic_weight_gamma))
        self.dynamic_epsilon = float(max(0.0, dynamic_epsilon))
        self.score_f_max = int(max(1, score_f_max))
        self.sample_with_replacement = bool(sample_with_replacement)
        self.metadata_manager = MetadataManager(metadata_path)
        self._env_cache = {}
        self._change_flag_cache: Dict[Tuple[int, int], bool] = {}
        self._dyn_score_cache: Dict[Tuple[int, int], float] = {}
        self.last_change_fraction_realized: Optional[float] = None
        self.last_mean_window_score: Optional[float] = None

        self.all_protein_metadata = all_protein_metadata or self._load_all_metadata()
        self.active_protein_indices = active_protein_indices or list(range(len(self.all_protein_metadata)))
        self.protein_metadata = [self.all_protein_metadata[i] for i in self.active_protein_indices]
        self._build_all_windows()
        self._build_epoch_indices(rng=np.random.default_rng(self.metadata_manager.metadata_df.shape[0]))

    def reset_env_cache(self):
        self._env_cache = {}

    def _get_loader(self, traj_path: str) -> LMDBLoader:
        loader = self._env_cache.get(traj_path)
        if loader is None:
            loader = LMDBLoader(traj_path)
            loader.__enter__()
            self._env_cache[traj_path] = loader
        return loader

    
    def _load_all_metadata(self) -> List[Dict]:
        """Preload LMDB metadata and join with global metadata CSV."""
        traj_names = sorted([name for name in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, name))])
        
        loaded_metadata = []
        for traj_name in traj_names:
            base_path = os.path.join(self.data_dir, traj_name)
            try:
                parts = traj_name.split('_')
                dynamic_id = parts[2] if len(parts) > 2 and parts[2].isdigit() else (re.findall(r"\d+", traj_name)[0] if re.findall(r"\d+", traj_name) else None)

                traj_path = base_path
                if not os.path.exists(os.path.join(traj_path, 'data.mdb')):
                    try:
                        children = [p for p in os.listdir(base_path)]
                    except Exception:
                        children = []
                    for child in children:
                        cpath = os.path.join(base_path, child)
                        if os.path.isdir(cpath) and os.path.exists(os.path.join(cpath, 'data.mdb')):
                            traj_path = cpath
                            break
                with LMDBLoader(traj_path) as loader:
                    metadata = loader.get_metadata()

                uniprot_id = 'unknown'
                pdb_id = 'unknown'
                if dynamic_id is not None:
                    try:
                        protein_info = self.metadata_manager.get_protein_info(dynamic_id)
                        uniprot_id = protein_info.get('Uniprot ID', uniprot_id)
                        pdb_id = protein_info.get('PDB ID', pdb_id)
                    except Exception:
                        pass
                metadata['uniprot_id'] = uniprot_id
                metadata['pdb_id'] = pdb_id
                metadata['traj_name'] = traj_name
                metadata['path'] = traj_path

                required_frames = (self.history_prefix_frames + self.num_full_res_frames + self.future_horizon - 1) * self.stride + 1
                if int(metadata.get('num_frames', 0)) >= required_frames:
                    loaded_metadata.append(metadata)
            except Exception:
                continue
        
        if not loaded_metadata:
            raise ValueError("No valid proteins found. Check data_dir path, LMDB contents, and L/K/F/stride settings.")
            
        return loaded_metadata

    def _build_all_windows(self):
        """Precompute all valid windows per protein for fixed-length sampling."""
        self._all_windows_by_protein: Dict[int, List[Tuple[int, int]]] = {}
        for pid, metadata in enumerate(self.protein_metadata):
            num_frames = metadata['num_frames']
            required_frames = (self.history_prefix_frames + self.num_full_res_frames + self.future_horizon - 1) * self.stride + 1
            wins: List[Tuple[int, int]] = []
            if num_frames >= required_frames:
                max_start_frame = num_frames - required_frames
                start = 0
                step = self.stride * self.window_start_stride
                while start <= max_start_frame:
                    target = start + (self.history_prefix_frames + self.num_full_res_frames) * self.stride
                    wins.append((start, target))
                    start += step
            self._all_windows_by_protein[pid] = wins

    def _build_epoch_indices(self, rng: np.random.Generator):
        """Build epoch indices with optional change-aware/dynamic-weighted sampling.

        Modes:
        - Uniform: shuffle all windows.
        - Change fraction (legacy): aim for change_target_fraction of windows with immediate change.
        - Dynamic weighting (advanced): sample windows with weights w=(eps + score)^gamma,
          where score depends on dynamic_score_mode: 'f0_any', 'any_future', or 'future_count'.
        """
        base_indices: List[Tuple[int, int]] = []
        self._epoch_windows_by_protein = {}
        for pid, wins in self._all_windows_by_protein.items():
            if not wins:
                continue
            self._epoch_windows_by_protein[pid] = wins
            base_indices.extend([(pid, i) for i in range(len(wins))])

        if not base_indices:
            self._epoch_indices = []
            return

        if self.change_target_fraction <= 0.0 and self.dynamic_weight_gamma <= 0.0:
            rng.shuffle(base_indices)
            self._epoch_indices = base_indices
            return

        desired = int(self.change_target_fraction * len(base_indices)) if self.change_target_fraction > 0.0 else 0
        change_list: List[Tuple[int, int]] = []
        other_list: List[Tuple[int, int]] = []

        def _has_change(pid: int, widx: int) -> bool:
            key = (pid, widx)
            if key in self._change_flag_cache:
                return self._change_flag_cache[key]
            wins = self._epoch_windows_by_protein[pid]
            start, target = wins[widx]
            last_input_idx = target - self.stride
            first_future_idx = target
            try:
                loader = self._get_loader(self.protein_metadata[pid]['path'])
                frames = loader.load_frames([last_input_idx, first_future_idx])
                a = torch.as_tensor(frames[0]['cluster_ids']).long()
                b = torch.as_tensor(frames[1]['cluster_ids']).long()
                flag = bool((a != b).any().item())
            except Exception:
                flag = False
            self._change_flag_cache[key] = flag
            return flag

        if self.dynamic_weight_gamma > 0.0:
            def _score_window(pid: int, widx: int) -> float:
                key = (pid, widx)
                if key in self._dyn_score_cache:
                    return self._dyn_score_cache[key]
                wins = self._epoch_windows_by_protein[pid]
                start, target = wins[widx]
                last_input_idx = target - self.stride
                indices = [last_input_idx]
                fL = min(self.future_horizon, self.score_f_max)
                for i in range(fL):
                    indices.append(target + i * self.stride)
                try:
                    loader = self._get_loader(self.protein_metadata[pid]['path'])
                    frames = loader.load_frames(indices)
                    ref = torch.as_tensor(frames[0]['cluster_ids'])
                    if self.dynamic_score_mode == 'f0_any':
                        b = torch.as_tensor(frames[1]['cluster_ids'])
                        score = float((ref != b).any().item())
                    else:
                        diffs = []
                        for k in range(1, len(frames)):
                            fut = torch.as_tensor(frames[k]['cluster_ids'])
                            diffs.append((ref != fut).float())
                        if not diffs:
                            score = 0.0
                        else:
                            D = torch.stack(diffs, dim=0)  # (fL, N)
                            if self.dynamic_score_mode == 'any_future':
                                score = float(D.any().item())
                            else:  # 'future_count'
                                score = float(D.mean().item())
                except Exception:
                    score = 0.0
                self._dyn_score_cache[key] = score
                return score

            scores: List[float] = []
            weights: List[float] = []
            for idx, (pid, widx) in enumerate(base_indices):
                s = _score_window(pid, widx) if (idx % self.change_probe_interval) == 0 else self._dyn_score_cache.get((pid, widx), 0.0)
                scores.append(float(s))
                w = (self.dynamic_epsilon + float(s)) ** self.dynamic_weight_gamma
                if self.change_target_fraction > 0.0 and s > 0.0:
                    w *= (1.0 + self.change_target_fraction)
                weights.append(w)

            total = max(1, len(scores))
            change_cnt = sum(1 for s in scores if s > 0.0)
            self.last_change_fraction_realized = float(change_cnt / total)
            self.last_mean_window_score = float(sum(scores) / total)

            import numpy as _np
            w_arr = _np.asarray(weights, dtype=_np.float64)
            if w_arr.sum() > 0:
                p = w_arr / w_arr.sum()
            else:
                p = None
            K = len(base_indices)
            rng_np = _np.random.default_rng()
            try:
                if self.sample_with_replacement:
                    choices = rng_np.choice(K, size=K, replace=True, p=p)
                    final_list = [base_indices[i] for i in choices.tolist()]
                else:
                    choices = rng_np.choice(K, size=K, replace=False, p=p)
                    final_list = [base_indices[i] for i in choices.tolist()]
            except Exception:
                rng.shuffle(base_indices)
                final_list = base_indices
            self._epoch_indices = final_list
            return

        for idx, (pid, widx) in enumerate(base_indices):
            if len(change_list) >= desired:
                other_list.append((pid, widx))
                continue
            if (idx % self.change_probe_interval) == 0:
                if _has_change(pid, widx):
                    change_list.append((pid, widx))
                else:
                    other_list.append((pid, widx))
            else:
                other_list.append((pid, widx))

        if len(change_list) < desired:
            for (pid, widx) in other_list:
                if len(change_list) >= desired:
                    break
                if _has_change(pid, widx):
                    change_list.append((pid, widx))

        rng.shuffle(change_list)
        rng.shuffle(other_list)
        final_list = change_list[:desired] + other_list
        rng.shuffle(final_list)
        self._epoch_indices = final_list

    def on_epoch_start(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed if seed is not None else self.metadata_manager.metadata_df.shape[0])
        self._build_epoch_indices(rng=rng)

    def __len__(self) -> int:
        return len(getattr(self, '_epoch_indices', []))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a training sequence with large-stride sampling (cluster IDs only)."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        pid, widx = self._epoch_indices[idx]
        metadata = self.protein_metadata[pid]
        start_frame, target_frame = self._epoch_windows_by_protein[pid][widx]
        
        available_frames = metadata['num_frames'] - start_frame
        max_steps_total = (available_frames - 1) // self.stride
        need_total = self.history_prefix_frames + self.num_full_res_frames
        max_possible_total = max_steps_total - self.future_horizon + 1
        if need_total > max_possible_total:
            raise ValueError(f"Cannot sample valid sequence with L={self.history_prefix_frames}, K={self.num_full_res_frames} at index {idx}")

        current_sequence_length = self.history_prefix_frames + self.num_full_res_frames
        
        input_frames = [start_frame + i * self.stride for i in range(current_sequence_length)]
        target_frame = start_frame + current_sequence_length * self.stride

        if target_frame >= metadata['num_frames']:
            raise ValueError(f"Target frame {target_frame} >= available frames {metadata['num_frames']}")
        assert target_frame >= 0, f"Invalid target_frame {target_frame} (must be >= 0)"

        loader = self._get_loader(metadata['path'])
        loader_meta = loader.get_metadata()
        input_cluster_ids = []
        input_times = []

        future_frames_idx: List[int] = [target_frame + i * self.stride for i in range(self.future_horizon)]
        if future_frames_idx and future_frames_idx[-1] >= metadata['num_frames']:
            raise ValueError("Insufficient frames to satisfy future horizon")

        frames = loader.load_frames(input_frames + future_frames_idx)
        for i, frame_idx in enumerate(input_frames):
            frame_data = frames[i]
            input_times.append(frame_idx * self.time_step)
            if 'cluster_ids' in frame_data:
                input_cluster_ids.append(torch.from_numpy(frame_data['cluster_ids']).long())
            else:
                raise RuntimeError("cluster_ids missing in frame data")

        input_times = torch.tensor(input_times, dtype=torch.float32)
        change_mask_seq: Optional[torch.Tensor] = None
        run_length_seq: Optional[torch.Tensor] = None
        if len(input_cluster_ids) > 0:
            input_cluster_ids = torch.stack(input_cluster_ids, dim=0)
            change_mask_seq = torch.zeros_like(input_cluster_ids, dtype=torch.bool)
            if input_cluster_ids.size(0) > 1:
                change_mask_seq[1:] = input_cluster_ids[1:] != input_cluster_ids[:-1]
            run_length_seq = torch.ones_like(input_cluster_ids, dtype=torch.long)
            for t in range(1, input_cluster_ids.size(0)):
                unchanged = change_mask_seq[t].logical_not()
                prev_len = run_length_seq[t - 1] + 1
                run_length_seq[t] = torch.where(unchanged, prev_len, torch.ones_like(prev_len))
        else:
            input_cluster_ids = torch.tensor([])

        delta_t = torch.zeros_like(input_times)
        if input_times.numel() > 1:
            delta_t[1:] = input_times[1:] - input_times[:-1]

        t_scalar = self.stride * self.time_step
        short_dt = t_scalar

        future_cluster_ids = None
        future_times = None
        future_cluster_ids_list = []
        future_times_list = []
        for offset, frame_idx in enumerate(future_frames_idx):
            frame_data = frames[len(input_frames) + offset]
            if 'cluster_ids' not in frame_data:
                raise RuntimeError("Future frame missing cluster_ids")
            future_cluster_ids_list.append(torch.from_numpy(frame_data['cluster_ids']).long())
            future_times_list.append(frame_idx * self.time_step)
        if future_cluster_ids_list:
            future_cluster_ids = torch.stack(future_cluster_ids_list, dim=0)
            future_times = torch.tensor(future_times_list, dtype=torch.float32)
        
        traj_name = loader_meta.get('traj_name', metadata.get('traj_name', 'unknown'))

        batch_dict = {
            'times': input_times,
            'delta_t': delta_t,
            'protein_idx': self.active_protein_indices[pid],
            'uniprot_id': metadata['uniprot_id'],
            'pdb_id': metadata['pdb_id'],
            'traj_name': traj_name,
            'start_frame': start_frame,
            'num_residues': int(input_cluster_ids.shape[1]) if isinstance(input_cluster_ids, torch.Tensor) else metadata['num_residues'],
            'sequence_length': current_sequence_length,
            'temporal_info': {
                'stride': self.stride,
                'short_dt': short_dt,
                't_scalar': t_scalar, 
                'time_step': self.time_step,
                'target_frame': int(target_frame),
                'input_frames': input_frames,
                'frame_id': f'frame_{int(target_frame)}',
                'history_prefix_frames': int(self.history_prefix_frames),
                'recent_full_frames': int(self.num_full_res_frames),
            },
        }
        
        if len(input_cluster_ids) > 0:
            batch_dict['input_cluster_ids'] = input_cluster_ids
            batch_dict['ids'] = input_cluster_ids
            if change_mask_seq is not None:
                batch_dict['change_mask'] = change_mask_seq
            if run_length_seq is not None:
                batch_dict['run_length'] = run_length_seq
        else:
            raise RuntimeError("No input cluster IDs available; run preprocessing first")

        if future_cluster_ids is not None and future_times is not None:
            batch_dict['future_cluster_ids'] = future_cluster_ids
            batch_dict['future_times'] = future_times

        return batch_dict


def create_dataloaders(
        data_dir: str,
        metadata_path: str,
        batch_size: int = 4,
        num_full_res_frames: int = 5,
        history_prefix_frames: int = 0,
        stride: int = 1,
        future_horizon: int = 1,
        window_start_stride: int = 1,
        change_target_fraction: float = 0.0,
        change_probe_interval: int = 20,
        dynamic_score_mode: str = 'f0_any',
        dynamic_weight_gamma: float = 0.0,
        dynamic_epsilon: float = 1e-3,
        score_f_max: int = 5,
        sample_with_replacement: bool = False,
        train_split: float = 0.9,
        val_split: float = 0.05,
        num_workers: int = 4,
        seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders using standard batching and collation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    master_dataset = ProteinTrajectoryDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        num_full_res_frames=num_full_res_frames,
        history_prefix_frames=history_prefix_frames,
        stride=stride,
        future_horizon=future_horizon,
        window_start_stride=window_start_stride,
        change_target_fraction=change_target_fraction,
        change_probe_interval=change_probe_interval,
        dynamic_score_mode=dynamic_score_mode,
        dynamic_weight_gamma=dynamic_weight_gamma,
        dynamic_epsilon=dynamic_epsilon,
        score_f_max=score_f_max,
        sample_with_replacement=sample_with_replacement,
    )

    num_proteins = len(master_dataset.all_protein_metadata)
    all_indices = list(range(num_proteins))
    random.shuffle(all_indices)

    train_end = int(num_proteins * train_split)
    val_end = int(num_proteins * (train_split + val_split))
    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]

    common_params = dict(
        data_dir=data_dir,
        metadata_path=metadata_path,
        num_full_res_frames=num_full_res_frames,
        history_prefix_frames=history_prefix_frames,
        stride=stride,
        future_horizon=future_horizon,
        window_start_stride=window_start_stride,
        change_target_fraction=change_target_fraction,
        change_probe_interval=change_probe_interval,
        dynamic_score_mode=dynamic_score_mode,
        dynamic_weight_gamma=dynamic_weight_gamma,
        dynamic_epsilon=dynamic_epsilon,
        score_f_max=score_f_max,
        sample_with_replacement=sample_with_replacement,
        all_protein_metadata=master_dataset.all_protein_metadata,
    )

    train_dataset = ProteinTrajectoryDataset(**common_params, active_protein_indices=train_indices)
    val_dataset = ProteinTrajectoryDataset(**common_params, active_protein_indices=val_indices)
    test_dataset = ProteinTrajectoryDataset(**common_params, active_protein_indices=test_indices)

    def _worker_init_fn(_):
        for ds in (train_dataset, val_dataset, test_dataset):
            if getattr(ds, 'reset_env_cache', None):
                ds.reset_env_cache()

    persistent = bool(num_workers > 0)
    prefetch = 4 if persistent else None

    dl_common = dict(
        collate_fn=collate_variable_length,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=_worker_init_fn,
    )
    if prefetch is not None:
        dl_common['prefetch_factor'] = prefetch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dl_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_common,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    pass