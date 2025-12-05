import torch
from typing import List, Dict


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to the max length in the batch."""

    batch_size = len(batch)
    max_timesteps = max(item['sequence_length'] for item in batch)
    max_residues = max(item['num_residues'] for item in batch)
    padded_times = torch.zeros(batch_size, max_timesteps, dtype=torch.float32)
    padded_delta_t = None
    if 'delta_t' in batch[0]:
        padded_delta_t = torch.zeros(batch_size, max_timesteps, dtype=torch.float32)

    padded_input_cluster_ids = None
    if 'input_cluster_ids' in batch[0]:
        padded_input_cluster_ids = torch.full(
            (batch_size, max_timesteps, max_residues), -1, dtype=torch.long
        )

    padded_input_embeddings = None
    # Check if any item in the batch has input_embeddings
    first_emb_item = next((item for item in batch if 'input_embeddings' in item), None)
    if first_emb_item is not None:
        # input_embeddings shape: (T, N, D)
        # We need to find D
        D = first_emb_item['input_embeddings'].shape[-1]
        padded_input_embeddings = torch.zeros(
            (batch_size, max_timesteps, max_residues, D), dtype=torch.float32
        )

    # Create masks
    residue_mask = torch.zeros(batch_size, max_residues, dtype=torch.bool)
    history_mask = torch.zeros(batch_size, max_timesteps, max_residues, dtype=torch.bool)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)

    max_future = 0
    if 'future_cluster_ids' in batch[0]:
        max_future = max(item['future_cluster_ids'].shape[0] for item in batch)
    padded_future_cluster_ids = None
    padded_future_times = None
    future_step_mask = None
    if max_future > 0:
        padded_future_cluster_ids = torch.full((batch_size, max_future, max_residues), -1, dtype=torch.long)
        padded_future_times = torch.zeros(batch_size, max_future, dtype=torch.float32)
        future_step_mask = torch.zeros(batch_size, max_future, dtype=torch.bool)

    padded_change_mask = None
    if 'change_mask' in batch[0]:
        padded_change_mask = torch.zeros(batch_size, max_timesteps, max_residues, dtype=torch.bool)
    padded_run_length = None
    if 'run_length' in batch[0]:
        padded_run_length = torch.zeros(batch_size, max_timesteps, max_residues, dtype=torch.long)

    protein_indices = []
    num_residues_list = []
    uniprot_ids = []
    pdb_ids = []
    start_frames = []

    for i, item in enumerate(batch):
        seq_len = item['sequence_length']
        num_res = item['num_residues']

        padded_times[i, :seq_len] = item['times']
        if padded_delta_t is not None and 'delta_t' in item:
            padded_delta_t[i, :seq_len] = item['delta_t']
        if padded_input_cluster_ids is not None and 'input_cluster_ids' in item:
            padded_input_cluster_ids[i, :seq_len, :num_res] = item['input_cluster_ids']
        
        if padded_input_embeddings is not None and 'input_embeddings' in item:
            padded_input_embeddings[i, :seq_len, :num_res, :] = item['input_embeddings']

        residue_mask[i, :num_res] = True
        history_mask[i, :seq_len, :num_res] = True
        sequence_lengths[i] = seq_len

        if padded_future_cluster_ids is not None and 'future_cluster_ids' in item:
            future_steps = item['future_cluster_ids'].shape[0]
            if future_steps > 0:
                padded_future_cluster_ids[i, :future_steps, :num_res] = item['future_cluster_ids']
                padded_future_times[i, :future_steps] = item.get('future_times', torch.zeros_like(padded_future_times[i, :future_steps]))
                future_step_mask[i, :future_steps] = True

        if padded_change_mask is not None and 'change_mask' in item:
            cm = item['change_mask']
            if torch.is_tensor(cm) and cm.dim() == 2:
                padded_change_mask[i, :cm.size(0), :num_res] = cm
            elif torch.is_tensor(cm) and cm.dim() == 1:
                padded_change_mask[i, seq_len - 1, :num_res] = cm
        if padded_run_length is not None and 'run_length' in item:
            rl = item['run_length']
            if torch.is_tensor(rl) and rl.dim() == 2:
                padded_run_length[i, :rl.size(0), :num_res] = rl
            elif torch.is_tensor(rl) and rl.dim() == 1:
                padded_run_length[i, seq_len - 1, :num_res] = rl

        protein_indices.append(item['protein_idx'])
        num_residues_list.append(num_res)
        uniprot_ids.append(item['uniprot_id'])
        pdb_ids.append(item['pdb_id'])
        start_frames.append(item['start_frame'])

    result = {
        'times': padded_times,
        'residue_mask': residue_mask,
        'history_mask': history_mask,
        'sequence_lengths': sequence_lengths,
        'num_residues': torch.tensor(num_residues_list, dtype=torch.long),
        'protein_indices': torch.tensor(protein_indices, dtype=torch.long),
        'start_frames': torch.tensor(start_frames, dtype=torch.long),
        'uniprot_ids': uniprot_ids,
        'pdb_ids': pdb_ids,
    }
    if padded_delta_t is not None:
        result['delta_t'] = padded_delta_t

    if 'traj_name' in batch[0]:
        result['traj_name'] = [item.get('traj_name', 'unknown') for item in batch]
    if 'temporal_info' in batch[0]:
        result['temporal_info'] = [item.get('temporal_info', {}) for item in batch]
    
    # Add cluster IDs
    if padded_input_cluster_ids is not None:
        result['input_cluster_ids'] = padded_input_cluster_ids
        result['ids'] = padded_input_cluster_ids

    if padded_input_embeddings is not None:
        result['input_embeddings'] = padded_input_embeddings

    if padded_future_cluster_ids is not None:
        result['future_cluster_ids'] = padded_future_cluster_ids
        result['future_times'] = padded_future_times
        result['future_step_mask'] = future_step_mask
    if padded_change_mask is not None:
        result['change_mask'] = padded_change_mask
    if padded_run_length is not None:
        result['run_length'] = padded_run_length

    return result
