#!/usr/bin/env python3
"""
Autoregressive evaluation for EMPROT classification-only models (cluster-ID prediction).

This script rolls out future cluster-ID frames autoregressively for a single trajectory,
computes per-time accuracy, and produces diagnostic plots and metadata.

Assumptions
- Dataset LMDBs store per-frame 'cluster_ids' arrays shaped [N_residues].
- Model runs in classification-only mode and accepts previous cluster IDs as inputs.
- The model/backbone internally handles hybrid context (recent K full frames and latent summaries).

Run from project root. Example:
  python scripts/autoregressive_eval.py \
    --ckpt checkpoints/emprot_classification_ONLY/best_model_epoch_120.pt \
    --data_root /path/to/lmdb_root \
    --split val \
    --time_start 200 --time_steps 50 \
    --output_dir outputs/auto_reg_eval
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import logging
import numpy as np
import torch

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from emprot.models.transformer import (
    ProteinTransformerClassificationOnly,
)
from emprot.data.dataset import create_dataloaders
from emprot.data.data_loader import LMDBLoader


# -------------------------
# Utilities and dataclasses
# -------------------------

log = logging.getLogger("emprot.autoreg")


def _maybe_set_plot_style() -> None:
    """Try to apply a clean plotting style (seaborn if available)."""
    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(context="talk", style="whitegrid")
    except Exception:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            pass
    # Tweak rcParams for readability
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.25,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

@dataclass
class EvalOutputs:
    gt: np.ndarray         # [T_pred, N]
    pred: np.ndarray       # [T_pred, N]
    acc_t: np.ndarray      # [T_pred]
    # Optional distributional metrics (filled later)
    dist_metrics: Optional[Dict] = None


# -------------------------
# Attention capture utilities
# -------------------------

def _set_store_attention(model: torch.nn.Module, enabled: bool = True) -> None:
    """Toggle storing attention weights on compatible attention modules."""
    try:
        for m in model.modules():
            if hasattr(m, 'store_attention_weights'):
                try:
                    setattr(m, 'store_attention_weights', bool(enabled))
                except Exception:
                    pass
    except Exception:
        pass


@torch.no_grad()
def capture_temporal_attention_per_frame(model: torch.nn.Module,
                                         hist_ids: torch.Tensor,
                                         times: torch.Tensor,
                                         history_mask: torch.Tensor,
                                         seq_lens: torch.Tensor) -> Optional[np.ndarray]:
    """Run a single forward pass with attention recording enabled and return per-frame attention.

    Returns array of shape (N_residues, T_frames) with attention mass per query residue over frames,
    averaged over heads. Each row approximately sums to 1 over valid frames.
    """
    device = hist_ids.device
    B, T, N = hist_ids.shape
    _set_store_attention(model, True)
    try:
        out = model(
            input_cluster_ids=hist_ids,
            times=times,
            sequence_lengths=seq_lens,
            history_mask=history_mask,
        )
        attn = getattr(getattr(model, 'backbone', None), 'last_attention_weights', None)
        if attn is None:
            return None
        # attn: (H, N, S) averaged over batch in attention module; otherwise (B,H,N,S)
        if attn.dim() == 4:
            attn = attn.mean(dim=0)  # (H,N,S)
        # Average over heads → (N,S)
        if attn.dim() == 3:
            attn = attn.mean(dim=0)
        if attn.dim() != 2 or attn.shape[0] != N:
            return None
        S = attn.shape[1]
        # Map S tokens back to (T,N) layout; extras (state/latents) appear as leading/trailing tokens
        # In our eval path we use only history tokens, so S == T*N.
        if S < (T * N):
            return None
        attn_reshaped = attn[:, : T * N].view(N, T, N)  # (N_query, T_frames, N_src)
        per_frame = attn_reshaped.sum(dim=-1)           # (N_query, T_frames)
        # Normalize rows for readability
        denom = per_frame.sum(dim=1, keepdim=True).clamp_min(1e-12)
        per_frame = (per_frame / denom).detach().cpu().numpy()
        return per_frame
    finally:
        _set_store_attention(model, False)


def plot_temporal_attention_over_frames(times_ns: np.ndarray,
                                        attn_per_frame: np.ndarray,
                                        residue_indices: List[int],
                                        out_path: Path,
                                        k_recent: Optional[int] = None) -> None:
    """Plot attention mass per frame for selected residues as a heatmap plus mean curve."""
    _maybe_set_plot_style()
    if attn_per_frame is None or len(residue_indices) == 0:
        return
    ridxs = [int(r) for r in residue_indices if 0 <= int(r) < attn_per_frame.shape[0]]
    if not ridxs:
        return
    sub = attn_per_frame[ridxs, :]  # (k, T)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(sub, aspect='auto', interpolation='nearest', cmap='magma',
                    extent=[times_ns[0], times_ns[-1] if times_ns.size>0 else 1.0, 0, sub.shape[0]])
    ax0.set_yticks(np.arange(len(ridxs)) + 0.5)
    ax0.set_yticklabels([str(r) for r in ridxs])
    ax0.set_title('Temporal Attention Mass per Frame (rows: selected residues)')
    ax0.set_ylabel('Residue idx')
    try:
        cbar = fig.colorbar(im, ax=ax0, orientation='vertical', fraction=0.025, pad=0.02)
        cbar.set_label('Attention mass', rotation=90)
    except Exception:
        pass
    if isinstance(k_recent, int) and k_recent > 0 and times_ns.size > k_recent:
        # Draw a vertical line at the start of the last K frames for context
        t_k = times_ns[-k_recent] if k_recent < times_ns.size else times_ns[0]
        ax0.axvline(t_k, color='white', linestyle='--', linewidth=1.0, alpha=0.8)
        ax0.text(t_k, -0.5, f'K={k_recent} start', color='white', ha='center', va='bottom', fontsize=9)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    mean_curve = sub.mean(axis=0)
    ax1.plot(times_ns, mean_curve, color='#2ca02c', linewidth=2)
    ax1.set_xlabel('Time (ns; history prior to rollout start)')
    ax1.set_ylabel('Mean attention')
    ax1.set_title('Mean attention across selected residues')
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
        return out.decode().strip()
    except Exception:
        return None


# -------------------------
# Model loading
# -------------------------

def _as_list(x) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    s = str(x).strip()
    return [t.strip() for t in s.split(',') if t.strip()] if s else None


def load_model(ckpt_path: str, device: torch.device, use_sparse_logits: bool):
    """Load EMPROT model from checkpoint.

    Returns (model, config, id2col, col2id, col2id_array)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {}) or {}
    cfg_model = config.get('model', {}) if isinstance(config.get('model', {}), dict) else {}

    d_embed = int(config.get('d_embed', cfg_model.get('d_embed', 512)))
    num_heads = int(config.get('num_heads', cfg_model.get('num_heads', 8)))
    num_layers = int(config.get('num_layers', cfg_model.get('num_layers', 1)))
    dropout = float(config.get('dropout', cfg_model.get('dropout', 0.1)))
    use_grad_ckpt = bool(config.get('use_gradient_checkpointing', cfg_model.get('use_gradient_checkpointing', True)))
    min_context_frames = int(config.get('min_context_frames', cfg_model.get('min_context_frames', 2)))
    attention_type = str(config.get('attention_type', cfg_model.get('attention_type', 'cross_temporal')))
    recent_full_frames = int(config.get('recent_full_frames', cfg_model.get('recent_full_frames', 5)))

    # Robust latent config: tolerate explicit nulls in YAML by falling back to sane defaults
    def _coalesce(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    latent_enabled = bool(_coalesce(config.get('latent_summary_enabled'), cfg_model.get('latent_summary_enabled'), False))
    latent_num_lat = int(_coalesce(config.get('latent_summary_num_latents'), cfg_model.get('latent_summary_num_latents'), 0) or 0)
    latent_layers = int(_coalesce(config.get('latent_summary_layers'), cfg_model.get('latent_summary_layers'), 1) or 1)
    latent_d_model = int(_coalesce(config.get('latent_summary_d_model'), cfg_model.get('latent_summary_d_model'), d_embed) or d_embed)
    latent_heads = int(_coalesce(config.get('latent_summary_heads'), cfg_model.get('latent_summary_heads'), num_heads) or num_heads)
    # Dropout may be explicitly null; fall back to main dropout
    _ld = _coalesce(config.get('latent_summary_dropout'), cfg_model.get('latent_summary_dropout'), dropout)
    try:
        latent_drop = float(_ld)
    except Exception:
        latent_drop = float(dropout)

    latent_cfg = {
        'enabled': latent_enabled,
        'num_latents': latent_num_lat,
        'layers': latent_layers,
        'd_model': latent_d_model,
        'heads': latent_heads,
        'dropout': latent_drop,
        'context_mode': config.get('context_mode', cfg_model.get('context_mode', 'latent_cap')),
        'train_context': config.get('train_context', cfg_model.get('train_context', {})),
        'summarizer': config.get('summarizer', cfg_model.get('summarizer', {})),
        'hier_pool': config.get('hier_pool', cfg_model.get('hier_pool', {})),
        'memory': config.get('memory', cfg_model.get('memory', {})),
    }

    # Resolve model state dict from checkpoint
    sd = ckpt.get('model', None)
    if sd is None:
        sd = ckpt.get('model_state_dict', None)
    if sd is None and all(isinstance(k, str) for k in ckpt.keys()):
        # Might be a raw state dict saved directly
        sd = ckpt

    num_clusters = int(config.get('num_clusters', cfg_model.get('num_clusters', 50000)))
    future_horizon = int(config.get('future_horizon', cfg_model.get('future_horizon', 1)))

    # Optional per-source KV flags (backward compatible)
    per_source_kv = bool(config.get('per_source_kv', cfg_model.get('per_source_kv', False)))
    per_source_kv_max_buckets = config.get('per_source_kv_max_buckets', cfg_model.get('per_source_kv_max_buckets', None))

    ModelCls = ProteinTransformerClassificationOnly
    model = ModelCls(
        d_embed=d_embed,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_gradient_checkpointing=use_grad_ckpt,
        min_context_frames=min_context_frames,
        attention_type=attention_type,
        num_clusters=num_clusters,
        future_horizon=future_horizon,
        recent_full_frames=recent_full_frames,
        latent_summary_enabled=bool(latent_cfg.get('enabled', False)),
        latent_summary_num_latents=int(latent_cfg.get('num_latents', 0)),
        latent_summary_heads=int(latent_cfg.get('heads', num_heads)),
        latent_summary_dropout=float(latent_cfg.get('dropout', dropout)),
        latent_summary_max_prefix=int(config.get('latent_summary_max_prefix', cfg_model.get('latent_summary_max_prefix', 0)) or 0) or None,
        per_source_kv=per_source_kv,
        per_source_kv_max_buckets=int(per_source_kv_max_buckets) if per_source_kv_max_buckets is not None else None,
    ).to(device)

    # Handle DataParallel 'module.' prefix
    if isinstance(sd, dict):
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k[len('module.'):]: v for k, v in sd.items() if k.startswith('module.')}
        model_keys = set(model.state_dict().keys())
        sd_filtered = {k: v for k, v in sd.items() if k in model_keys}
        missing = [k for k in model_keys if k not in sd_filtered]
        if sd_filtered:
            model.load_state_dict(sd_filtered, strict=False)
        if missing:
            print(f"[WARN] Missing keys in checkpoint (loaded partial): {len(missing)} keys")
    else:
        print("[WARN] No valid state dict found in checkpoint; using random init.")

    # Backward-compat hint flag (ignored by current model; tolerate silently)
    try:
        setattr(model, 'sparse_classification_logits', bool(use_sparse_logits))
    except Exception:
        pass
    model.eval()

    # Label mapping (identity if absent)
    id2col = None
    for key in ['id2col', 'label_map', 'cluster_mapping', 'class_to_idx']:
        mapping = ckpt.get(key)
        if isinstance(mapping, dict) and mapping:
            try:
                id2col = {int(k): int(v) for k, v in mapping.items()}
                break
            except Exception:
                continue
    col2id = None
    if id2col is None:
        m = ckpt.get('col2id')
        if isinstance(m, dict) and m:
            try:
                col2id = {int(k): int(v) for k, v in m.items()}
                id2col = {rid: col for col, rid in col2id.items()}
            except Exception:
                col2id = None
    if col2id is None and id2col is not None:
        col2id = {col: rid for rid, col in id2col.items()}

    if id2col is None:
        num_eff = getattr(model, 'classification_head', None).num_clusters
        id2col = {i: i for i in range(num_eff)}
        col2id = {i: i for i in range(num_eff)}
        print("[WARN] No label mapping in checkpoint; assuming identity (col == raw ID).")

    num_eff = getattr(model, 'classification_head', None).num_clusters
    col2id_array = np.full(num_eff, -1, dtype=np.int32)
    for col, rid in (col2id or {}).items():
        if 0 <= int(col) < num_eff:
            col2id_array[int(col)] = int(rid)

    return model, config, id2col, col2id, col2id_array


def _remap_neighbors_to_col_space(nei: Dict[str, np.ndarray], id2col: Optional[Dict[int, int]],
                                  num_clusters: int) -> Dict[str, np.ndarray]:
    """Map neighbors from raw ID space to model column space using id2col.

    If id2col is None, return as-is. Any unmapped entries become -1 and should
    be ignored downstream.
    """
    if nei is None or id2col is None:
        return nei
    out = dict(nei)
    neigh = out.get('neighbors', None)
    if isinstance(neigh, np.ndarray):
        # Build fast lookup from raw id to col id
        max_raw = int(neigh.max()) if neigh.size > 0 else -1
        lut_size = max(max_raw + 1, max(id2col.keys()) + 1 if id2col else 0)
        lut = np.full((lut_size,), -1, dtype=np.int32)
        for k, v in id2col.items():
            if 0 <= int(k) < lut_size:
                lut[int(k)] = int(v)
        mapped = np.full_like(neigh, -1, dtype=np.int32)
        valid_mask = (neigh >= 0) & (neigh < lut_size)
        mapped[valid_mask] = lut[neigh[valid_mask]]
        out['neighbors'] = mapped
    return out


def maybe_set_context_knobs(model, recent_full_frames: int, context_latents: int) -> Dict[str, Optional[int]]:
    """Try to adjust model's context parameters at eval time.

    Returns dict of the values that were actually set.
    """
    applied = {
        'recent_full_frames': None,
        'context_latents': None,
    }
    try:
        if hasattr(model, 'recent_full_frames'):
            model.recent_full_frames = int(max(0, recent_full_frames))
            applied['recent_full_frames'] = int(recent_full_frames)
    except Exception:
        pass

    # Adjust number of latents if a ContextBuilder exists
    try:
        cb = getattr(model.backbone, 'context_builder', None)
        if cb is not None and hasattr(cb, 'summarizer') and hasattr(cb.summarizer, 'latents'):
            want = int(max(0, context_latents))
            have = cb.summarizer.latents.shape[1]
            if want != have and want > 0:
                with torch.no_grad():
                    base = cb.summarizer.latents
                    if want < have:
                        cb.summarizer.latents = torch.nn.Parameter(base[:, :want, :].clone())
                    else:
                        reps = int(np.ceil(want / have))
                        tiled = base.repeat(1, reps, 1)[:, :want, :].clone()
                        cb.summarizer.latents = torch.nn.Parameter(tiled)
                applied['context_latents'] = want
            else:
                applied['context_latents'] = have
    except Exception:
        pass

    return applied


# -------------------------
# Data loading helpers
# -------------------------

def find_metadata_csv(data_root: str) -> Optional[str]:
    candidates = [
        str(ROOT / 'traj_metadata.csv'),
        os.path.join(data_root, 'traj_metadata.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def list_split_proteins(data_root: str, split: str, seed: int, batch_size: int = 1) -> List[Dict]:
    """Use the dataset splitter to get the protein metadata for a split."""
    meta_csv = find_metadata_csv(data_root)
    if meta_csv is None:
        # Fallback: treat each directory under data_root as a protein
        traj_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        metas = []
        for name in traj_names:
            p = os.path.join(data_root, name)
            try:
                with LMDBLoader(p) as ld:
                    m = ld.get_metadata()
                metas.append(m)
            except Exception:
                continue
        return metas

    # Create loaders so we can access split datasets' metadata (match current API)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_root,
            metadata_path=meta_csv,
            batch_size=int(batch_size),
            history_prefix_frames=5,
            num_full_res_frames=8,
            stride=1,
            future_horizon=1,
            train_split=0.8,
            val_split=0.1,
            num_workers=0,
            seed=int(seed),
        )
    except Exception as e:
        print(f"[WARN] Failed to create split loaders with metadata ({e}). Falling back to directory scan of {data_root}.")
        traj_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        metas = []
        for name in traj_names:
            p = os.path.join(data_root, name)
            try:
                with LMDBLoader(p) as ld:
                    m = ld.get_metadata()
                metas.append(m)
            except Exception:
                continue
        return metas
    split = str(split).lower()
    if split == 'train':
        ds = train_loader.dataset
    elif split == 'val' or split == 'valid' or split == 'validation':
        ds = val_loader.dataset
    else:
        ds = test_loader.dataset
    return list(ds.protein_metadata)


def select_protein(metas: List[Dict], protein_id: Optional[str]) -> Dict:
    if not metas:
        raise RuntimeError("No proteins available in the requested split.")
    if protein_id is None:
        return metas[0]
    # Match by directory name or dynamic id segment
    pid = str(protein_id)
    for m in metas:
        name = m.get('traj_name') or os.path.basename(m.get('path', ''))
        if name == pid:
            return m
    # Try dynamic id (third token by convention)
    for m in metas:
        name = m.get('traj_name') or os.path.basename(m.get('path', ''))
        parts = name.split('_')
        if len(parts) >= 3 and parts[2] == pid:
            return m
    raise ValueError(f"protein_id '{protein_id}' not found in split.")


def load_sequence(data_root: str, split: str, protein_id: Optional[str], seed: int) -> Tuple[str, np.ndarray, np.ndarray]:
    """Return (traj_name, Y_all[t,n], times_all[t]) for the selected protein."""
    metas = list_split_proteins(data_root, split, seed)
    m = select_protein(metas, protein_id)
    traj_path = m['path']
    traj_name = m.get('traj_name') or os.path.basename(traj_path)
    with LMDBLoader(traj_path) as loader:
        meta = loader.get_metadata()
        T = int(meta['num_frames'])
        Ys = []
        times = []
        for t in range(T):
            fr = loader.load_frame(t)
            if 'cluster_ids' not in fr:
                raise KeyError(f"cluster_ids missing in frame {t} for trajectory {traj_name}")
            Ys.append(fr['cluster_ids'].astype(np.int32))
            times.append(float(t))
    Y_all = np.stack(Ys, axis=0)
    times_all = np.asarray(times, dtype=np.float32)
    return traj_name, Y_all, times_all


def protein_display_name(data_root: str, traj_name: str) -> str:
    """Try to derive a human-readable protein name using traj_metadata.csv.

    Fallbacks: Uniprot ID → PDB ID → trajectory folder name.
    """
    meta_csv = find_metadata_csv(data_root)
    if meta_csv is None:
        return traj_name
    try:
        from emprot.data.metadata import MetadataManager
        mm = MetadataManager(meta_csv)
        parts = traj_name.split('_')
        dynamic_id = parts[2] if len(parts) >= 3 else parts[-1]
        row = mm.get_protein_info(dynamic_id)
        # Try common columns for human-readable names
        for key in ('Protein name', 'Protein names', 'Entry name', 'Entry', 'Gene Names', 'Gene names'):
            if key in row and isinstance(row[key], str) and row[key].strip():
                return row[key].strip()
        for key in ('Uniprot ID', 'PDB ID'):
            if key in row and isinstance(row[key], str) and row[key].strip():
                return row[key].strip()
    except Exception:
        pass
    return traj_name


# -------------------------
# Core evaluation
# -------------------------

def columns_to_raw(columns: np.ndarray, col2id_array: np.ndarray, col2id_dict: Dict[int, int]) -> np.ndarray:
    raw = np.full_like(columns, -1, dtype=np.int32)
    if col2id_array is not None and col2id_array.size > 0:
        valid = (columns >= 0) & (columns < col2id_array.size)
        raw[valid] = col2id_array[columns[valid]]
    if (raw < 0).any() and col2id_dict:
        unk_cols = np.unique(columns[raw < 0])
        for c in unk_cols:
            mapped = int(col2id_dict.get(int(c), -1))
            raw[columns == c] = mapped
    return raw


def rollout_autoregressive(model, Y_all: np.ndarray, time_start: int, time_steps: int, device: torch.device,
                           recent_full_frames: int, col2id_array: np.ndarray, col2id: Dict[int, int],
                           decode_mode: str = 'argmax', temperature: float = 1.0, sample_topk: int = 0,
                           temp_anneal_gamma: float = 1.0, min_temperature: float = 0.1,
                           top_p: float = 0.0, copy_bias: float = 0.0, min_dwell: int = 1,
                           neighbors: Optional[Dict[str, np.ndarray]] = None, neighbor_k: int = 256,
                           neighbor_fallback_top_p: float = 0.1,
                           use_context_prior: bool = False, context_prior_weight: float = 1.0,
                           restrict_to_history_support: bool = False, history_support_k: int = 0,
                           simple_nucleus: bool = False) -> EvalOutputs:
    """Roll out predictions autoregressively using previous cluster IDs as context.

    Args:
        decode_mode: 'argmax' or 'sample'. If 'sample', draw from softmax with temperature and optional top-k.
        temperature: Initial sampling temperature (ignored for argmax).
        sample_topk: If > 0, restrict sampling to the top-k classes per residue during sampling.
        temp_anneal_gamma: Multiply temperature by this factor each step (<=1.0 to anneal).
        min_temperature: Lower bound applied after annealing each step.
        top_p: If > 0, nucleus sampling keeps the smallest set of classes whose cumulative prob ≥ top_p.
        copy_bias: Probability per residue of copying the previous raw ID instead of sampling.
        min_dwell: If > 1, enforce a minimum number of consecutive steps before allowing a change.
    """
    T_total, N = Y_all.shape
    assert 0 <= time_start < T_total, f"time_start {time_start} out of range [0, {T_total-1}]"
    assert time_start + time_steps <= T_total, "Requested rollout exceeds sequence length"

    # Seed history with all frames < time_start
    hist_ids = torch.from_numpy(Y_all[:time_start]).long().unsqueeze(0).to(device)  # [1, Th, N]
    hist_ids = torch.where(hist_ids < 0, torch.zeros_like(hist_ids), hist_ids)
    times = torch.arange(time_start, dtype=torch.float32, device=device).view(1, -1) * 0.2  # ns
    hist_mask = torch.ones(1, time_start, N, dtype=torch.bool, device=device)
    seq_lens = torch.tensor([time_start], dtype=torch.long, device=device)

    gt_window = Y_all[time_start:time_start + time_steps].copy()
    pred_window = np.full_like(gt_window, -1, dtype=np.int32)
    acc_t = np.zeros((time_steps,), dtype=np.float32)

    t_curr = float(max(1e-6, temperature))
    # Online dwell tracking per residue (raw IDs)
    if time_start > 0:
        curr_id_np = Y_all[time_start - 1].copy()
        curr_id_np = np.where(curr_id_np < 0, 0, curr_id_np).astype(np.int32)
    else:
        curr_id_np = np.zeros((N,), dtype=np.int32)
    run_len_np = np.ones((N,), dtype=np.int32)
    for step in range(time_steps):
        with torch.no_grad():
            out = model(
                input_cluster_ids=hist_ids,
                times=times,
                sequence_lengths=seq_lens,
                history_mask=hist_mask,
            )
            if 'cluster_logits' in out:
                logits = out['cluster_logits']
                # Newer model returns (B,F,N,C) even when no futures are provided; take the first horizon
                if logits.dim() == 4:
                    logits = logits[:, 0, :, :]
            else:
                # Fallback: compute full logits from context
                ctx = out['context']  # [1, N, D]
                logits = model.classification_head(ctx)

            # Optional context prior: per-residue visitation prior from observed history
            if use_context_prior and hist_ids.size(1) > 0:
                C = int(logits.size(-1))
                _N = int(logits.size(1))
                # Build per-residue priors from history: counts over classes seen at that residue
                try:
                    hist_np = hist_ids.detach().cpu().numpy()[0]  # (T_hist, N)
                    priors = np.zeros(( _N, C ), dtype=np.float64)
                    for i in range(_N):
                        h = hist_np[:, i]
                        h = h[h >= 0]
                        if h.size > 0:
                            cnt = np.bincount(h, minlength=C).astype(np.float64)
                            s = cnt.sum()
                            if s > 0:
                                priors[i, :] = cnt / s
                    # Convert to log prior with small smoothing to avoid -inf on zeros
                    alpha = 1e-12
                    log_prior = np.log(priors + alpha) * float(context_prior_weight)
                    prior_t = torch.from_numpy(log_prior).to(device=logits.device, dtype=logits.dtype)
                    logits = logits + prior_t.view(1, _N, C)
                except Exception:
                    pass
            if decode_mode == 'argmax':
                cols = torch.argmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()  # [N]
                pred_ids = columns_to_raw(cols, col2id_array, col2id)
            else:
                # Sampling (softmax with temperature)
                logits_scaled = logits / max(t_curr, 1e-6)
                _B1, _N, C = logits_scaled.shape
                probs_full = torch.softmax(logits_scaled, dim=-1)  # [1,N,C]

                if bool(simple_nucleus):
                    # Pure nucleus sampling on logits: apply only top-p (no priors, no neighbors, no top-k, no dwell)
                    if isinstance(top_p, (float, int)) and float(top_p) > 0.0 and float(top_p) <= 1.0:
                        probs_sort, idx_sort = torch.sort(probs_full, dim=-1, descending=True)
                        cum = torch.cumsum(probs_sort, dim=-1)
                        cum2 = cum.squeeze(0)
                        pvals = torch.full((cum2.size(0), 1), float(top_p), device=cum2.device)
                        cutoff = torch.searchsorted(cum2, pvals, right=False).squeeze(-1)
                        cutoff = torch.clamp(cutoff, min=0, max=C - 1)
                        sorted_mask = (torch.arange(C, device=cum2.device).view(1, C) <= cutoff.view(-1, 1)).unsqueeze(0)
                        mask = torch.zeros_like(probs_full, dtype=torch.bool)
                        mask.scatter_(dim=-1, index=idx_sort, src=sorted_mask)
                        masked_probs = probs_full * mask.float()
                        denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        probs = masked_probs / denom
                    else:
                        probs = probs_full
                else:
                    mask = None
                # Neighbor constraint mask
                if not simple_nucleus and neighbors is not None:
                    neigh_idx = neighbors.get('neighbors', None)
                    if isinstance(neigh_idx, np.ndarray) and neigh_idx.ndim == 2:
                        C = probs_full.size(-1)
                        _cur = hist_ids[:, -1, :].squeeze(0).detach().cpu().numpy() if hist_ids.size(1) > 0 else np.zeros((_N,), dtype=np.int32)
                        _cur = np.clip(_cur, 0, C - 1)
                        L = min(int(neighbor_k), neigh_idx.shape[1])
                        mask = torch.zeros_like(probs_full, dtype=torch.bool)  # [1,N,C]
                        for i in range(_N):
                            src = int(_cur[i])
                            neigh = neigh_idx[src, :L]
                            neigh_t = torch.from_numpy(neigh).to(mask.device, dtype=torch.long).view(1, -1)
                            mask[0, i, :].scatter_(dim=-1, index=neigh_t, src=torch.ones_like(neigh_t, dtype=torch.bool))
                        # Optional small fallback mass via top-p
                        fp = float(neighbor_fallback_top_p)
                        if fp > 0.0 and fp <= 1.0:
                            probs_sort, idx_sort = torch.sort(probs_full, dim=-1, descending=True)
                            cum = torch.cumsum(probs_sort, dim=-1)
                            cum2 = cum.squeeze(0)
                            pvals = torch.full((cum2.size(0), 1), fp, device=cum2.device)
                            cutoff = torch.searchsorted(cum2, pvals, right=False).squeeze(-1)
                            cutoff = torch.clamp(cutoff, min=0, max=C - 1)
                            sorted_mask = (torch.arange(C, device=cum2.device).view(1, C) <= cutoff.view(-1, 1))
                            sorted_mask = sorted_mask.unsqueeze(0)
                            fallback_mask = torch.zeros_like(probs_full, dtype=torch.bool)
                            fallback_mask.scatter_(dim=-1, index=idx_sort, src=sorted_mask)
                            mask = mask | fallback_mask
                # History-support mask per residue (clusters seen in each residue's history)
                if (not simple_nucleus) and bool(restrict_to_history_support) and hist_ids.size(1) > 0:
                    C = probs_full.size(-1)
                    _N = probs_full.size(1)
                    hist_np = hist_ids.detach().cpu().numpy()[0]  # (T_hist, N)
                    mask_h = torch.zeros_like(probs_full, dtype=torch.bool)
                    for i in range(_N):
                        h = hist_np[:, i]
                        h = h[h >= 0]
                        if h.size > 0:
                            if int(history_support_k) and int(history_support_k) > 0:
                                cnt = np.bincount(h, minlength=C)
                                order = np.argsort(-cnt)[: int(history_support_k)]
                                idx = torch.from_numpy(order.astype(np.int64)).to(mask_h.device)
                            else:
                                uniq = np.unique(h)
                                idx = torch.from_numpy(uniq.astype(np.int64)).to(mask_h.device)
                            if idx.numel() > 0:
                                mask_h[0, i, idx] = True
                    mask = mask_h if mask is None else (mask | mask_h)

                # Nucleus (top-p) mask
                if (not simple_nucleus) and isinstance(top_p, (float, int)) and float(top_p) > 0.0 and float(top_p) <= 1.0:
                    probs_sort, idx_sort = torch.sort(probs_full, dim=-1, descending=True)  # [1,N,C]
                    cum = torch.cumsum(probs_sort, dim=-1)  # [1,N,C]
                    pval = float(top_p)
                    # Row-wise cutoff per residue
                    cum2 = cum.squeeze(0)  # [N,C]
                    # values must be broadcastable to boundaries' shape except last dim → use [N,1]
                    pvals = torch.full((cum2.size(0), 1), pval, device=cum2.device)  # [N,1]
                    cutoff = torch.searchsorted(cum2, pvals, right=False).squeeze(-1)  # [N]
                    cutoff = torch.clamp(cutoff, min=0, max=C - 1)
                    # Build boolean mask in sorted index space, then scatter back to original class indices
                    sorted_mask = (torch.arange(C, device=cum2.device).view(1, C) <= cutoff.view(-1, 1))  # [N,C]
                    sorted_mask = sorted_mask.unsqueeze(0)  # [1,N,C]
                    mask = torch.zeros_like(probs_full, dtype=torch.bool)  # [1,N,C]
                    mask.scatter_(dim=-1, index=idx_sort, src=sorted_mask)
                # Top-k mask
                if (not simple_nucleus) and isinstance(sample_topk, int) and sample_topk > 0:
                    k = int(min(sample_topk, int(C)))
                    topv_k, topi_k = torch.topk(probs_full, k=k, dim=-1)
                    mask_k = torch.zeros_like(probs_full, dtype=torch.bool)
                    mask_k.scatter_(dim=-1, index=topi_k, src=torch.ones_like(topi_k, dtype=torch.bool))
                    mask = mask_k if mask is None else (mask & mask_k)
                if not simple_nucleus and mask is not None:
                    masked_probs = probs_full * mask.float()
                    denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    probs = masked_probs / denom
                elif not simple_nucleus:
                    probs = probs_full
                cols = torch.multinomial(probs.view(-1, C), num_samples=1).view(1, _N).squeeze(0).detach().cpu().numpy()
                pred_ids = columns_to_raw(cols, col2id_array, col2id)

            # Copy-bias: with probability copy_bias per residue, copy previous raw id
            if (not simple_nucleus) and isinstance(copy_bias, (float, int)) and float(copy_bias) > 0.0:
                prev_raw = hist_ids[:, -1, :].squeeze(0).detach().cpu().numpy() if hist_ids.size(1) > 0 else curr_id_np
                m = np.random.rand(pred_ids.shape[0]) < float(copy_bias)
                pred_ids = np.where(m, prev_raw, pred_ids)

            # Minimum dwell constraint (online)
            if (not simple_nucleus) and isinstance(min_dwell, int) and int(min_dwell) > 1:
                md = int(min_dwell)
                same = (pred_ids == curr_id_np)
                need_hold = (~same) & (run_len_np < md)
                pred_ids = np.where(need_hold, curr_id_np, pred_ids)
                same_after = (pred_ids == curr_id_np)
                run_len_np = np.where(same_after, run_len_np + 1, np.ones_like(run_len_np))
                curr_id_np = np.where(same_after, curr_id_np, pred_ids)
            else:
                same_after = (pred_ids == curr_id_np)
                run_len_np = np.where(same_after, run_len_np + 1, np.ones_like(run_len_np))
                curr_id_np = np.where(same_after, curr_id_np, pred_ids)

            pred_window[step] = pred_ids

        gt = gt_window[step]
        valid = (gt >= 0)
        if valid.any():
            acc_t[step] = float((pred_ids[valid] == gt[valid]).sum()) / float(valid.sum())
        else:
            acc_t[step] = 0.0

        # Append prediction to history for next step
        next_ids = torch.from_numpy(np.where(pred_ids < 0, 0, pred_ids)).long().to(device)[None, None, :]  # [1,1,N]
        hist_ids = torch.cat([hist_ids, next_ids], dim=1)
        next_t = torch.tensor([[0.2 * (time_start + step)]], dtype=torch.float32, device=device)
        times = torch.cat([times, next_t], dim=1)
        next_mask = torch.ones(1, 1, N, dtype=torch.bool, device=device)
        hist_mask = torch.cat([hist_mask, next_mask], dim=1)
        seq_lens = torch.tensor([hist_ids.shape[1]], dtype=torch.long, device=device)

        # Anneal temperature for next step
        if decode_mode != 'argmax':
            try:
                t_curr = max(float(min_temperature), float(t_curr) * float(temp_anneal_gamma))
            except Exception:
                pass

    return EvalOutputs(gt=gt_window, pred=pred_window, acc_t=acc_t)


# -------------------------
# Residue selection & plots
# -------------------------

def valid_residues_across_window(Y_window: np.ndarray) -> np.ndarray:
    return np.where((Y_window >= 0).all(axis=0))[0]


def select_residues(Y_window: np.ndarray, k: int, mode: str, seed: int) -> List[int]:
    valid = valid_residues_across_window(Y_window)
    if valid.size == 0:
        return []
    k = int(min(k, valid.size))
    if mode == 'random':
        rng = np.random.default_rng(seed)
        sel = rng.choice(valid, size=k, replace=False)
        return sorted(sel.tolist())
    if mode == 'most_change':
        changes = (Y_window[1:] != Y_window[:-1]).sum(axis=0)
        changes_valid = changes[valid]
        order = np.argsort(-changes_valid)
        sel = valid[order[:k]]
        return sorted(sel.tolist())
    if mode == 'uniform':
        if k == 1:
            return [int(valid[valid.size // 2])]
        positions = np.linspace(0, valid.size - 1, num=k).round().astype(int)
        return sorted(valid[positions].tolist())
    return sorted(valid[:k].tolist())


def plot_residue_subplots(times_ns: np.ndarray, Y_gt: np.ndarray, Y_pred: np.ndarray, residue_indices: List[int], protein_name: str, out_path: Path) -> None:
    _maybe_set_plot_style()
    k = len(residue_indices)
    if k == 0:
        return
    ncols = 1 if k <= 3 else 2
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), sharex=True, constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, ridx in enumerate(residue_indices):
        ax = axes[i]
        gt = Y_gt[:, ridx]
        pr = Y_pred[:, ridx]
        ax.step(times_ns, gt, where='mid', color='#4a4a4a', linewidth=1.4, label='GT', alpha=0.95)
        ax.step(times_ns, pr, where='mid', color='#1f77b4', linewidth=1.6, label='Pred', alpha=0.95)
        ax.set_title(f'Residue {ridx}')
        ax.set_ylabel('Cluster ID')
        if i == 0:
            ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f'Cluster IDs vs Time — {protein_name} (0.2 ns/frame)')
    axes[max(0, i)].set_xlabel('Time (ns)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_composite_heatmaps_and_accuracy(times_ns: np.ndarray, Y_gt: np.ndarray, Y_pred: np.ndarray, acc_t: np.ndarray, residue_indices: List[int], protein_name: str, out_path: Path) -> None:
    _maybe_set_plot_style()
    sub_gt = Y_gt[:, residue_indices].T
    sub_pr = Y_pred[:, residue_indices].T

    # Discrete colormap with many distinct colors
    from matplotlib.colors import ListedColormap
    base = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    cmap = ListedColormap(base)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), gridspec_kw={'height_ratios': [1.1, 1.1, 0.9]}, sharex=True, constrained_layout=True)

    im0 = axes[0].imshow(sub_gt, aspect='auto', interpolation='nearest', cmap=cmap)
    axes[0].set_ylabel('Residue')
    axes[0].set_yticks(np.arange(len(residue_indices)))
    axes[0].set_yticklabels([str(i) for i in residue_indices])
    axes[0].set_title('Ground Truth Clusters')
    try:
        uniq0 = np.unique(sub_gt)
        if uniq0.size <= 20:
            cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.025, pad=0.02)
            cbar0.set_ticks(list(uniq0))
            cbar0.set_label('Cluster ID', rotation=90)
    except Exception:
        pass

    im1 = axes[1].imshow(sub_pr, aspect='auto', interpolation='nearest', cmap=cmap)
    axes[1].set_ylabel('Residue')
    axes[1].set_yticks(np.arange(len(residue_indices)))
    axes[1].set_yticklabels([str(i) for i in residue_indices])
    axes[1].set_title('Predicted Clusters')
    try:
        uniq1 = np.unique(sub_pr)
        if uniq1.size <= 20:
            cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.025, pad=0.02)
            cbar1.set_ticks(list(uniq1))
            cbar1.set_label('Cluster ID', rotation=90)
    except Exception:
        pass

    acc_pct = 100.0 * acc_t
    axes[2].plot(times_ns, acc_pct, marker='o', linestyle='-', color='#2ca02c', linewidth=2)
    axes[2].fill_between(times_ns, acc_pct, step='mid', alpha=0.12, color='#2ca02c')
    last_y = float(acc_pct[-1]) if acc_pct.size > 0 else 0.0
    mean_y = float(np.mean(acc_pct)) if acc_pct.size > 0 else 0.0
    for x, y in zip(times_ns, acc_pct):
        axes[2].annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 6), textcoords='offset points', fontsize=9, ha='center')
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel('Time (ns; 0.2 ns/frame)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Accuracy vs Time')
    axes[2].text(1.0, 1.02, f'Last={last_y:.1f}%  |  Mean={mean_y:.1f}%', transform=axes[2].transAxes, ha='right', va='bottom', fontsize=11)

    fig.suptitle(f'Autoregressive Evaluation — {protein_name}')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_accuracy(times_ns: np.ndarray, acc_t: np.ndarray, out_path: Path) -> None:
    _maybe_set_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(times_ns, 100.0 * acc_t, marker='o', color='#2ca02c', linewidth=2)
    ax.fill_between(times_ns, 100.0 * acc_t, step='mid', alpha=0.12, color='#2ca02c')
    for x, y in zip(times_ns, 100.0 * acc_t):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 6), textcoords='offset points', fontsize=9, ha='center')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_xlabel('Time (ns; 0.2 ns/frame)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Time', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_residue_trajectories_pretty(traj_name: str, times_ns: np.ndarray, Y_gt: np.ndarray, Y_pred: np.ndarray,
                                     residue_indices: List[int], out_path: Path) -> None:
    _maybe_set_plot_style()
    if len(residue_indices) == 0:
        return
    ridxs = list(residue_indices)
    # Scale figure height with number of residues to keep readability
    _height = max(2.2 * len(ridxs), 6.0)
    fig, axes = plt.subplots(len(ridxs), 1, figsize=(12, _height), sharex=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.suptitle(f"{traj_name} – Residue Cluster Trajectories")
    # Use time relative to rollout start to avoid large empty margins
    t_rel = np.arange(Y_gt.shape[0], dtype=np.float32) * 0.2
    for i, (ax, ridx) in enumerate(zip(axes, ridxs)):
        gt = Y_gt[:, ridx]
        pr = Y_pred[:, ridx]
        ax.plot(t_rel, gt, color='#1f77b4', linewidth=1.8, label='Ground Truth')
        ax.plot(t_rel, pr, color='#ff7f0e', linewidth=1.8, label='Autoregressive Prediction')
        # Per-residue similarity metrics on the plotted window
        try:
            mask = (gt >= 0) & (pr >= 0)
            denom = float(np.maximum(1, mask.sum()))
            acc = float(np.mean((gt[mask] == pr[mask])) if mask.any() else np.nan)
            # Order-free visitation similarity (JS)
            C = int(max(int(gt[mask].max(initial=0)), int(pr[mask].max(initial=0))) + 1) if mask.any() else 1
            if mask.any() and C > 0:
                h_gt = _safe_hist(gt[mask].astype(np.int64), C)
                h_pr = _safe_hist(pr[mask].astype(np.int64), C)
                js = float(_js_divergence(h_gt, h_pr))
            else:
                js = float('nan')
            title_suffix = f" — acc={acc:.2f}" + (f", JS={js:.3f}" if np.isfinite(js) else "")
        except Exception:
            title_suffix = ""
        # Optional amino acid letter if available in metadata later; for now just index + metrics
        ax.set_title(f"Residue {int(ridx)}{title_suffix}")
        ax.set_ylabel('Cluster ID')
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc='upper right', frameon=True)
    axes[-1].set_xlabel('Time (ns; since rollout start)')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)


def plot_residue_visitation_bars(traj_name: str,
                                 Y_gt: np.ndarray,
                                 Y_pred: np.ndarray,
                                 residue_indices: List[int],
                                 num_classes: int,
                                 out_path: Path,
                                 topk: int = 20) -> None:
    _maybe_set_plot_style()
    if not residue_indices:
        return
    ridxs = [int(r) for r in residue_indices]
    k = len(ridxs)
    ncols = 1 if k <= 3 else 2
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, ridx in enumerate(ridxs):
        ax = axes[i]
        gt = Y_gt[:, ridx]
        pr = Y_pred[:, ridx]
        h_gt = _safe_hist(gt, num_classes)
        h_pr = _safe_hist(pr, num_classes)
        # choose top-K by GT visitation (fallback to combined if GT empty)
        if h_gt.sum() <= 0 and h_pr.sum() > 0:
            scores = h_pr
        else:
            scores = h_gt
        order = np.argsort(-scores)[:max(1, int(topk))]
        x = np.arange(order.size)
        ax.bar(x - 0.2, h_gt[order], width=0.4, label='GT')
        ax.bar(x + 0.2, h_pr[order], width=0.4, label='Pred')
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in order], rotation=45, ha='right')
        ax.set_ylabel('Freq')
        ax.set_title(f'Residue {int(ridx)} — visitation (top-{order.size})')
        if i == 0:
            ax.legend(frameon=True)
        ax.grid(alpha=0.2)
    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f'{traj_name} — Per-residue visitation')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_residue_visitation_single(traj_name: str,
                                   Y_gt: np.ndarray,
                                   Y_pred: np.ndarray,
                                   ridx: int,
                                   num_classes: int,
                                   out_path: Path,
                                   topk: int = 20) -> None:
    _maybe_set_plot_style()
    gt = Y_gt[:, int(ridx)]
    pr = Y_pred[:, int(ridx)]
    h_gt = _safe_hist(gt, num_classes)
    h_pr = _safe_hist(pr, num_classes)
    if h_gt.sum() <= 0 and h_pr.sum() > 0:
        scores = h_pr
    else:
        scores = h_gt
    order = np.argsort(-scores)[:max(1, int(topk))]
    x = np.arange(order.size)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, h_gt[order], width=0.4, label='GT')
    ax.bar(x + 0.2, h_pr[order], width=0.4, label='Pred')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in order], rotation=45, ha='right')
    ax.set_ylabel('Freq')
    ax.set_title(f'{traj_name} — Residue {int(ridx)} visitation (top-{order.size})')
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_rollout_panels_pretty(traj_name: str, times_ns: np.ndarray, Y_gt: np.ndarray, Y_pred: np.ndarray,
                               acc_t: np.ndarray, residue_indices: List[int], out_path: Path,
                               decode_annotation: Optional[str] = None) -> None:
    _maybe_set_plot_style()
    ridxs = list(residue_indices)
    if len(ridxs) == 0:
        return
    sub_gt = Y_gt[:, ridxs].T
    sub_pr = Y_pred[:, ridxs].T

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), gridspec_kw={'height_ratios': [1.1, 1.1, 0.9]}, sharex=True, constrained_layout=True)
    if decode_annotation:
        fig.suptitle(f"{traj_name} — Autoregressive Rollout [{decode_annotation}]")

    # Heatmaps with viridis
    # Use relative time for x so plots fill the axis
    t_rel = np.arange(Y_gt.shape[0], dtype=np.float32) * 0.2
    t0, t1 = (float(t_rel[0]), float(t_rel[-1] if t_rel.size > 0 else 1.0))
    im0 = axes[0].imshow(sub_gt, aspect='auto', interpolation='nearest', cmap='viridis',
                         extent=[t0, t1, 0, sub_gt.shape[0]])
    axes[0].set_title(f"{traj_name} – Ground Truth Clusters")
    axes[0].set_ylabel('Residue idx')
    try:
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.025, pad=0.02)
        cbar0.set_label('Cluster ID', rotation=90)
    except Exception:
        pass

    im1 = axes[1].imshow(sub_pr, aspect='auto', interpolation='nearest', cmap='viridis',
                         extent=[t0, t1, 0, sub_pr.shape[0]])
    pred_title = 'Predicted Clusters'
    if decode_annotation:
        pred_title += f" ({decode_annotation})"
    axes[1].set_title(pred_title)
    axes[1].set_ylabel('Residue idx')
    try:
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.025, pad=0.02)
        cbar1.set_label('Cluster ID', rotation=90)
    except Exception:
        pass

    # Accuracy (overall)
    acc_pct = 100.0 * acc_t
    axes[2].plot(t_rel, acc_pct, 'o-', color='#1f77b4', linewidth=2, markersize=4)
    axes[2].set_ylim(0, 100)
    axes[2].set_title('Per-step Accuracy')
    axes[2].set_xlabel('Time (ns; since rollout start)')
    axes[2].set_ylabel('Step accuracy')
    axes[2].grid(alpha=0.25)

    try:
        # Per-residue accuracies across the window
        gt = Y_gt
        pr = Y_pred
        valid = (gt >= 0)
        per_res = (gt == pr) & valid
        per_res_acc = per_res.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        mean_acc = float(np.mean(per_res_acc)) * 100.0
        axes[2].text(0.01, 0.94, f"Mean per-residue acc: {mean_acc:.1f}%", transform=axes[2].transAxes,
                     ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        # Highlight shown residues with their individual accuracies
        if len(residue_indices) > 0:
            labels = ', '.join([f"{int(r)}:{per_res_acc[int(r)]*100:.1f}%" for r in residue_indices[:5]])
            axes[2].text(0.99, 0.94, f"Shown residues acc: {labels}", transform=axes[2].transAxes,
                         ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    except Exception:
        pass

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)


# -------------------------
# Distributional metrics
# -------------------------

def _safe_hist(vec: np.ndarray, num_classes: int) -> np.ndarray:
    h = np.bincount(vec[vec >= 0], minlength=num_classes).astype(np.float64)
    s = float(h.sum())
    return h / s if s > 0 else h


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def _empirical_dwell_lengths(seq: np.ndarray) -> List[int]:
    out = []
    if seq.size == 0:
        return out
    cur = seq[0]
    run = 1
    for x in seq[1:]:
        if x == cur:
            run += 1
        else:
            if cur >= 0:
                out.append(run)
            cur = x
            run = 1
    if cur >= 0:
        out.append(run)
    return out


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    # Two-sample KS statistic for 1D non-negative integer arrays (dwell lengths)
    if a.size == 0 or b.size == 0:
        return 1.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    va = np.unique(a_sorted)
    vb = np.unique(b_sorted)
    grid = np.unique(np.concatenate([va, vb]))
    cdf_a = np.searchsorted(a_sorted, grid, side='right') / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, grid, side='right') / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def pi_hat_from_preds(pred_ids: np.ndarray, num_classes: int) -> np.ndarray:
    mask = pred_ids >= 0
    counts = np.bincount(pred_ids[mask].ravel(), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full((num_classes,), 1.0 / max(1, num_classes))
    return counts / total


def P_from_seq(ids: np.ndarray, num_classes: int) -> np.ndarray:
    # Fallback to dense for small class counts; otherwise caller should prefer compact path
    T, N = ids.shape
    if num_classes <= 4096:
        P = np.zeros((num_classes, num_classes), dtype=np.float64)
        for t in range(T - 1):
            src = ids[t]
            dst = ids[t + 1]
            mask = (src >= 0) & (dst >= 0)
            for i, j in zip(src[mask], dst[mask]):
                P[int(i), int(j)] += 1.0
        P = P + 1e-8
        P = P / P.sum(axis=1, keepdims=True)
        return P
    # For large class counts, build compact transition matrix over classes present in ids
    flat = ids.reshape(-1)
    flat = flat[flat >= 0]
    uniq = np.unique(flat)
    K = int(uniq.size)
    idx = {int(c): k for k, c in enumerate(uniq)}
    P = np.zeros((K, K), dtype=np.float64)
    for t in range(T - 1):
        src = ids[t]
        dst = ids[t + 1]
        mask = (src >= 0) & (dst >= 0)
        for i, j in zip(src[mask], dst[mask]):
            ii = idx.get(int(i), None)
            jj = idx.get(int(j), None)
            if ii is not None and jj is not None:
                P[ii, jj] += 1.0
    P = P + 1e-8
    P = P / P.sum(axis=1, keepdims=True)
    return P


def rowwise_kl_np(Pt: np.ndarray, Ph: np.ndarray, eps: float = 1e-12) -> float:
    Pt = (Pt + eps) / (Pt.sum(axis=1, keepdims=True) + eps)
    Ph = (Ph + eps) / (Ph.sum(axis=1, keepdims=True) + eps)
    kl = (Pt * (np.log(Pt + eps) - np.log(Ph + eps))).sum(axis=1)
    return float(kl.mean())


def compute_distributional_metrics(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> Dict:
    # Visitation histograms over window
    gt_vis = _safe_hist(gt.reshape(-1), num_classes)
    pr_vis = _safe_hist(pred.reshape(-1), num_classes)
    js_vis = _js_divergence(gt_vis, pr_vis)

    # Transition distributions over compact class sets to avoid OOM for large C
    P_gt = P_from_seq(gt, num_classes)
    P_pr = P_from_seq(pred, num_classes)
    # Align shapes if built in compact mode
    if P_gt.shape != P_pr.shape:
        # Expand both to the union of classes; rebuild compactly using union mapping
        def _compact_P(ids: np.ndarray) -> np.ndarray:
            T, N = ids.shape
            flat = ids.reshape(-1)
            flat = flat[flat >= 0]
            return flat
        u_all = np.unique(np.concatenate([_compact_P(gt), _compact_P(pred)]))
        K = int(u_all.size)
        index = {int(c): k for k, c in enumerate(u_all)}
        def _P(ids: np.ndarray) -> np.ndarray:
            T, N = ids.shape
            P = np.zeros((K, K), dtype=np.float64)
            for t in range(T - 1):
                src = ids[t]
                dst = ids[t + 1]
                mask = (src >= 0) & (dst >= 0)
                for i, j in zip(src[mask], dst[mask]):
                    ii = index.get(int(i), None)
                    jj = index.get(int(j), None)
                    if ii is not None and jj is not None:
                        P[ii, jj] += 1.0
            P = P + 1e-8
            P = P / P.sum(axis=1, keepdims=True)
            return P
        P_gt = _P(gt)
        P_pr = _P(pred)
    row_kl_gt_pr = rowwise_kl_np(P_gt, P_pr)
    row_kl_pr_gt = rowwise_kl_np(P_pr, P_gt)
    fro_norm = float(np.linalg.norm(P_gt - P_pr))

    # Dwell lengths pooled across residues
    gt_dw = []
    pr_dw = []
    for r in range(gt.shape[1]):
        gt_dw.extend(_empirical_dwell_lengths(gt[:, r]))
        pr_dw.extend(_empirical_dwell_lengths(pred[:, r]))
    gt_dw = np.asarray(gt_dw, dtype=np.int32)
    pr_dw = np.asarray(pr_dw, dtype=np.int32)
    ks_dwell = _ks_stat(gt_dw, pr_dw)
    dwell_stats = {
        'gt_mean': float(gt_dw.mean()) if gt_dw.size else 0.0,
        'pr_mean': float(pr_dw.mean()) if pr_dw.size else 0.0,
        'gt_median': float(np.median(gt_dw)) if gt_dw.size else 0.0,
        'pr_median': float(np.median(pr_dw)) if pr_dw.size else 0.0,
        'ks_stat': ks_dwell,
        'gt_count': int(gt_dw.size),
        'pr_count': int(pr_dw.size),
    }
    if gt_dw.size > 0:
        mean_gt = float(gt_dw.mean())
        if mean_gt > 0:
            p_geometric = min(0.999, max(1e-6, 1.0 / mean_gt))
            dwell_stats['gt_geometric_p'] = p_geometric
            dwell_stats['gt_geometric_nll'] = float(-np.mean(np.log(p_geometric) + (gt_dw - 1) * np.log(1.0 - p_geometric)))
            if pr_dw.size > 0:
                dwell_stats['pr_geometric_nll'] = float(-np.mean(np.log(p_geometric) + (pr_dw - 1) * np.log(1.0 - p_geometric)))

    # Transition distributions: estimate P(i→j) on the fly for classes seen in GT
    # For robustness, compare per-source JS where source has ≥K events
    K = 5
    js_by_source = []
    for r in range(gt.shape[1]):
        s_gt = gt[:-1, r]
        t_gt = gt[1:, r]
        s_pr = pred[:-1, r]
        t_pr = pred[1:, r]
        mask_gt = (s_gt >= 0) & (t_gt >= 0)
        mask_pr = (s_pr >= 0) & (t_pr >= 0)
        sgt = s_gt[mask_gt]
        tgt = t_gt[mask_gt]
        spr = s_pr[mask_pr]
        tpr = t_pr[mask_pr]
        if sgt.size == 0 or spr.size == 0:
            continue
        # Group by source cluster in GT
        sources = np.unique(sgt)
        for src in sources:
            tgt_src = tgt[sgt == src]
            if tgt_src.size < K:
                continue
            pr_tgt_src = tpr[spr == src]
            if pr_tgt_src.size < K:
                continue
            p = _safe_hist(tgt_src, num_classes)
            q = _safe_hist(pr_tgt_src, num_classes)
            js = _js_divergence(p, q)
            js_by_source.append(js)
    js_trans_mean = float(np.mean(js_by_source)) if js_by_source else None
    js_trans_count = int(len(js_by_source))

    return {
        'visitation_js': js_vis,
        'js_pi': js_vis,
        'visitation_kl_gt_pr': float(np.sum(gt_vis * (np.log(gt_vis + 1e-12) - np.log(pr_vis + 1e-12)))) if gt_vis.sum() > 0 else 0.0,
        'visitation_kl_pr_gt': float(np.sum(pr_vis * (np.log(pr_vis + 1e-12) - np.log(gt_vis + 1e-12)))) if pr_vis.sum() > 0 else 0.0,
        'dwell_stats': dwell_stats,
        'transition_js_mean_over_sources': js_trans_mean,
        'transition_js_sources_evaluated': js_trans_count,
        'transition_row_kl_gt_pr': row_kl_gt_pr,
        'transition_row_kl_pr_gt': row_kl_pr_gt,
        'transition_frobenius': fro_norm,
    }


def compute_per_residue_metrics(gt: np.ndarray, pred: np.ndarray, num_classes: int, top_k: int = 10) -> Dict:
    R = int(gt.shape[1])
    js_list = []
    l1_list = []
    cov_gt_list = []
    cov_pr_list = []
    gt_only_counts = []
    pr_only_counts = []
    for r in range(R):
        g = gt[:, r]
        p = pred[:, r]
        h_g = _safe_hist(g, num_classes)
        h_p = _safe_hist(p, num_classes)
        js = _js_divergence(h_g, h_p)
        l1 = 0.5 * float(np.abs(h_g - h_p).sum())
        js_list.append(js)
        l1_list.append(l1)
        set_g = set(np.where(h_g > 0)[0].tolist())
        set_p = set(np.where(h_p > 0)[0].tolist())
        inter = set_g & set_p
        cov_gt = (len(inter) / max(1, len(set_g))) if len(set_g) > 0 else 1.0
        cov_pr = (len(inter) / max(1, len(set_p))) if len(set_p) > 0 else 1.0
        cov_gt_list.append(cov_gt)
        cov_pr_list.append(cov_pr)
        gt_only_counts.append(len(set_g - set_p))
        pr_only_counts.append(len(set_p - set_g))

    js_arr = np.asarray(js_list, dtype=np.float64)
    l1_arr = np.asarray(l1_list, dtype=np.float64)
    cov_gt_arr = np.asarray(cov_gt_list, dtype=np.float64)
    cov_pr_arr = np.asarray(cov_pr_list, dtype=np.float64)
    gt_only_arr = np.asarray(gt_only_counts, dtype=np.int32)
    pr_only_arr = np.asarray(pr_only_counts, dtype=np.int32)

    order = np.argsort(-js_arr)[: max(1, int(top_k))]
    worst = [
        {
            'residue': int(r),
            'js': float(js_arr[r]),
            'l1': float(l1_arr[r]),
            'coverage_gt': float(cov_gt_arr[r]),
            'coverage_pred': float(cov_pr_arr[r]),
            'gt_only_count': int(gt_only_arr[r]),
            'pred_only_count': int(pr_only_arr[r]),
        }
        for r in order
    ]

    return {
        'per_residue_js_mean': float(np.mean(js_arr)),
        'per_residue_js_median': float(np.median(js_arr)),
        'per_residue_js_p90': float(np.percentile(js_arr, 90.0)),
        'per_residue_l1_mean': float(np.mean(l1_arr)),
        'per_residue_coverage_gt_mean': float(np.mean(cov_gt_arr)),
        'per_residue_coverage_pred_mean': float(np.mean(cov_pr_arr)),
        'worst_residues_by_js': worst,
    }


def plot_distributions(traj_name: str, gt: np.ndarray, pred: np.ndarray, ridxs: List[int], out_dir: Path,
                       num_classes: int, topk_clusters: int = 20, transitions_top_states: int = 15) -> None:
    _maybe_set_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Global visitation (top-K by GT frequency)
    gt_vis = _safe_hist(gt.reshape(-1), num_classes)
    pr_vis = _safe_hist(pred.reshape(-1), num_classes)
    order = np.argsort(-gt_vis)[:max(1, topk_clusters)]
    x = np.arange(order.size)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, gt_vis[order], width=0.4, label='GT')
    ax.bar(x + 0.2, pr_vis[order], width=0.4, label='Pred')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in order], rotation=45, ha='right')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{traj_name} — Global visitation (top-{order.size} GT clusters)')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / 'visitation_topk.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2) Global dwell-time distributions (histogram + CDF overlay)
    def _dwell_array(Y: np.ndarray) -> np.ndarray:
        all_dw = []
        for r in range(Y.shape[1]):
            all_dw.extend(_empirical_dwell_lengths(Y[:, r]))
        return np.asarray(all_dw, dtype=np.int32)
    gt_dw = _dwell_array(gt)
    pr_dw = _dwell_array(pred)
    # Histogram (cap long tail for readability)
    max_bin = int(np.percentile(np.concatenate([gt_dw, pr_dw]) if gt_dw.size and pr_dw.size else np.array([1]), 99))
    max_bin = max(5, max_bin)
    bins = np.arange(1, max_bin + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(gt_dw, bins=bins, alpha=0.6, label='GT')
    axes[0].hist(pr_dw, bins=bins, alpha=0.6, label='Pred')
    axes[0].set_title('Dwell length histogram')
    axes[0].set_xlabel('Run length')
    axes[0].set_ylabel('Count')
    axes[0].legend(frameon=True)
    # CDF
    def _cdf(a):
        if a.size == 0:
            return np.array([0.0]), np.array([0.0])
        a = np.sort(a)
        grid = np.arange(1, min(max_bin, a.max()) + 1)
        cdf = np.searchsorted(a, grid, side='right') / a.size
        return grid, cdf
    gx, gc = _cdf(gt_dw)
    px, pc = _cdf(pr_dw)
    axes[1].plot(gx, gc, label='GT')
    axes[1].plot(px, pc, label='Pred')
    axes[1].set_title('Dwell length CDF')
    axes[1].set_xlabel('Run length')
    axes[1].set_ylabel('CDF')
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / 'dwell_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3) Per-residue transition heatmaps for selected residues
    # Show compact top-M state space per residue (by GT visitation)
    M = max(3, int(transitions_top_states))
    if len(ridxs) > 0:
        for ridx in ridxs:
            s_gt = gt[:-1, ridx]
            t_gt = gt[1:, ridx]
            s_pr = pred[:-1, ridx]
            t_pr = pred[1:, ridx]
            mask_gt = (s_gt >= 0) & (t_gt >= 0)
            mask_pr = (s_pr >= 0) & (t_pr >= 0)
            if not mask_gt.any() or not mask_pr.any():
                continue
            vis_r = _safe_hist(gt[:, ridx], num_classes)
            top_states = np.argsort(-vis_r)[:M]
            idx_map = {int(s): i for i, s in enumerate(top_states)}
            def _count_mat(s, t):
                mat = np.zeros((M, M), dtype=np.float64)
                for si, ti in zip(s, t):
                    a = idx_map.get(int(si), None)
                    b = idx_map.get(int(ti), None)
                    if a is not None and b is not None:
                        mat[a, b] += 1.0
                # Row-normalize
                rs = mat.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                return mat / rs
            Pgt = _count_mat(s_gt[mask_gt], t_gt[mask_gt])
            Ppr = _count_mat(s_pr[mask_pr], t_pr[mask_pr])
            labels = [str(int(s)) for s in top_states]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            im0 = axes[0].imshow(Pgt, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0.0, vmax=max(1e-8, Pgt.max()))
            axes[0].set_title(f'Residue {int(ridx)} — GT P(i→j)')
            axes[0].set_xlabel('j')
            axes[0].set_ylabel('i')
            axes[0].set_xticks(np.arange(M))
            axes[0].set_yticks(np.arange(M))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].set_yticklabels(labels)
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(Ppr, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0.0, vmax=max(1e-8, Pgt.max()))
            axes[1].set_title('Pred P(i→j) (same state subset)')
            axes[1].set_xlabel('j')
            axes[1].set_ylabel('i')
            axes[1].set_xticks(np.arange(M))
            axes[1].set_yticks(np.arange(M))
            axes[1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1].set_yticklabels(labels)
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            fig.suptitle(f'{traj_name} — Residue {int(ridx)} transitions (top-{M} states by GT freq)')
            fig.savefig(out_dir / f'residue_{int(ridx)}_transitions.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
# -------------------------
# Metadata
# -------------------------

def save_metadata_json(out_dir: Path, payload: Dict) -> None:
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(payload, f, indent=2)


def save_readme(out_dir: Path, summary: str) -> None:
    with open(out_dir / 'README.txt', 'w') as f:
        f.write(summary)


# -------------------------
# Main
# -------------------------

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    _maybe_set_plot_style()
    parser = argparse.ArgumentParser(description='Autoregressive eval for EMPROT classification-only')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Dataset/LMDB root directory')
    parser.add_argument('--split', type=str, default='val', help='Split to evaluate: train/val/test')
    parser.add_argument('--recent_full_frames', type=int, default=5, help='Recent frames at full resolution')
    parser.add_argument('--context_latents', type=int, default=60, help='Number of latent summary tokens to use')
    parser.add_argument('--protein_id', type=str, default=None, help='Specific protein/trajectory to evaluate')
    parser.add_argument('--time_start', type=int, required=True, help='First predicted time index (last observed is time_start-1)')
    parser.add_argument('--time_steps', type=int, required=True, help='Number of future steps to roll out')
    parser.add_argument('--k_residues', type=int, default=5, help='Number of residues to visualize')
    parser.add_argument('--residue_select', type=str, default='most_change', choices=['random', 'most_change', 'uniform'], help='Residue selection mode')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_sparse_logits', action='store_true', default=True, help="Keep model's sparse-logit path on eval")
    # Decoding controls
    parser.add_argument('--decode_mode', type=str, default='config', choices=['config', 'argmax', 'sample'], help='Decoding strategy (use config defaults or override)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (ignored for argmax)')
    parser.add_argument('--sample_topk', type=int, default=0, help='If >0, sample only from top-k per residue')
    parser.add_argument('--temp_anneal_gamma', type=float, default=1.0, help='Multiply temperature by this each step (<=1 to anneal)')
    parser.add_argument('--min_temperature', type=float, default=0.1, help='Lower bound for annealed temperature')
    parser.add_argument('--top_p', type=float, default=0.0, help='Nucleus sampling top-p (0 disables)')
    parser.add_argument('--copy_bias', type=float, default=0.0, help='Probability to copy previous ID per residue')
    parser.add_argument('--min_dwell', type=int, default=1, help='Minimum steps a residue must stay before changing')
    # Distribution plotting controls
    parser.add_argument('--plot_distributions', action='store_true', help='Plot visitation/dwell and per-residue transition heatmaps')
    parser.add_argument('--plot_topk_clusters', type=int, default=20, help='Top-K clusters (by GT frequency) to show in visitation bars')
    parser.add_argument('--plot_transitions_top_states', type=int, default=15, help='Max states per residue for transition heatmaps')
    # Neighbor-constrained decoding and context prior
    parser.add_argument('--transition_neighbors', type=str, default=None, help='Path to npz with arrays: neighbors [C,L], probs [C,L]')
    parser.add_argument('--neighbor_k', type=int, default=256, help='Max neighbors to allow per current cluster')
    parser.add_argument('--neighbor_fallback_top_p', type=float, default=0.1, help='Portion of sampling mass drawn from global top-p fallback (0 disables)')
    parser.add_argument('--use_context_prior', action='store_true', help='Apply log-prior from observed history visitation to logits before sampling')
    parser.add_argument('--context_prior_weight', type=float, default=1.0, help='Strength of context prior (multiplies log prior)')
    # History-support sampling
    parser.add_argument('--restrict_to_history_support', action='store_true', help='Restrict sampling to clusters seen in each residue\'s history (with fallback top-p)')
    parser.add_argument('--history_support_k', type=int, default=0, help='If >0, keep only top-K history classes per residue')

    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Load model
    model, cfg, id2col, col2id, col2id_array = load_model(args.ckpt, device, bool(args.use_sparse_logits))
    knobs = maybe_set_context_knobs(model, args.recent_full_frames, args.context_latents)

    if args.decode_mode == 'config':
        cfg_mode = str(cfg.get('prediction_sampling_mode', 'argmax')).lower()
        cfg_temp = float(cfg.get('prediction_sampling_temperature', 1.0) or 1.0)
        cfg_top_p = float(cfg.get('prediction_sampling_top_p', 0.0) or 0.0)
        cfg_top_k = int(cfg.get('prediction_sampling_top_k', 0) or 0)
        if cfg_mode == 'argmax':
            args.decode_mode = 'argmax'
            args.top_p = 0.0
            args.sample_topk = 0
        else:
            args.decode_mode = 'sample'
            args.temperature = cfg_temp
            if cfg_mode == 'top_k':
                args.top_p = 0.0
                args.sample_topk = cfg_top_k
            else:
                args.top_p = cfg_top_p
                args.sample_topk = cfg_top_k

    # Load sequence
    traj_name, Y_all, times_all = load_sequence(args.data_root, args.split, args.protein_id, args.seed)

    # Bounds check
    T_total = Y_all.shape[0]
    if args.time_start + args.time_steps > T_total:
        raise ValueError(f"Requested window [{args.time_start}, {args.time_start + args.time_steps - 1}] exceeds sequence length {T_total}")

    # Attention visualization on the history up to time_start
    try:
        B_hist = int(args.time_start)
        if B_hist > 1:
            hist_ids = torch.from_numpy(Y_all[:args.time_start]).long().unsqueeze(0).to(device)
            hist_ids = torch.where(hist_ids < 0, torch.zeros_like(hist_ids), hist_ids)
            times = torch.arange(args.time_start, dtype=torch.float32, device=device).view(1, -1) * 0.2
            hist_mask = torch.ones(1, args.time_start, hist_ids.size(-1), dtype=torch.bool, device=device)
            seq_lens = torch.tensor([args.time_start], dtype=torch.long, device=device)
            attn_pf = capture_temporal_attention_per_frame(model, hist_ids, times, hist_mask, seq_lens)
            # Select residues (same policy as rollout plots) and plot
            ridxs_attn = select_residues(Y_all[max(0, args.time_start-args.recent_full_frames):args.time_start], int(args.k_residues), args.residue_select, args.seed)
            times_hist_ns = np.arange(args.time_start, dtype=np.float32) * 0.2
            plot_temporal_attention_over_frames(times_hist_ns, attn_pf, ridxs_attn, out_dir / f'{traj_name}_attention_over_frames.png', k_recent=int(args.recent_full_frames))
    except Exception as e:
        log.warning("Attention visualization failed: %s", e)

    # Rollout
    # Prepare neighbors (remap to model's column space if provided in raw IDs)
    nei_dict = dict(np.load(args.transition_neighbors)) if args.transition_neighbors else None
    if nei_dict is not None:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', None)
        try:
            nei_dict = _remap_neighbors_to_col_space(nei_dict, id2col=id2col, num_clusters=int(num_classes) if num_classes is not None else 0)
        except Exception:
            pass

    eval_out = rollout_autoregressive(
        model=model,
        Y_all=Y_all,
        time_start=int(args.time_start),
        time_steps=int(args.time_steps),
        device=device,
        recent_full_frames=int(args.recent_full_frames),
        col2id_array=col2id_array,
        col2id=col2id or {},
        decode_mode=str(args.decode_mode),
        temperature=float(args.temperature),
        sample_topk=int(args.sample_topk or 0),
        temp_anneal_gamma=float(args.temp_anneal_gamma),
        min_temperature=float(args.min_temperature),
        top_p=float(args.top_p or 0.0),
        copy_bias=float(args.copy_bias or 0.0),
        min_dwell=int(args.min_dwell or 1),
        neighbors=nei_dict,
        neighbor_k=int(args.neighbor_k),
        neighbor_fallback_top_p=float(args.neighbor_fallback_top_p),
        use_context_prior=bool(args.use_context_prior),
        context_prior_weight=float(args.context_prior_weight),
        restrict_to_history_support=bool(args.restrict_to_history_support),
        history_support_k=int(args.history_support_k),
    )

    # Residue selection for plots
    ridxs = select_residues(eval_out.gt, int(args.k_residues), args.residue_select, args.seed)

    # Plots
    times_abs = np.arange(args.time_start, args.time_start + args.time_steps, dtype=np.float32)
    times_ns = times_abs * 0.2  # 0.2 ns per frame
    prot_name = protein_display_name(args.data_root, traj_name)
    # 1) Pretty roll-out panels (GT/Pred heatmaps + accuracy)
    decode_annot = None
    if str(args.decode_mode) == 'sample':
        decode_annot = f"sample T={args.temperature:.2f}, top_p={args.top_p:.2f}, top_k={int(args.sample_topk)}"
    plot_rollout_panels_pretty(traj_name, times_ns, eval_out.gt, eval_out.pred, eval_out.acc_t, ridxs,
                               out_dir / f'{traj_name}_rollout.png', decode_annotation=decode_annot)
    # 2) Pretty residue trajectories (3 panels)
    plot_residue_trajectories_pretty(traj_name, times_ns, eval_out.gt, eval_out.pred, ridxs, out_dir / f'{traj_name}_residue_panel.png')
    # 2b) Per-residue visitation histograms for the same residues
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        plot_residue_visitation_bars(
            traj_name=traj_name,
            Y_gt=eval_out.gt,
            Y_pred=eval_out.pred,
            residue_indices=ridxs,
            num_classes=int(num_classes),
            out_path=out_dir / f'{traj_name}_residue_visitation.png',
            topk=int(getattr(args, 'plot_topk_clusters', 20)),
        )
        for ridx in ridxs:
            plot_residue_visitation_single(
                traj_name=traj_name,
                Y_gt=eval_out.gt,
                Y_pred=eval_out.pred,
                ridx=int(ridx),
                num_classes=int(num_classes),
                out_path=out_dir / f'{traj_name}_residue_{int(ridx)}_visitation.png',
                topk=int(getattr(args, 'plot_topk_clusters', 20)),
            )
    except Exception:
        pass

    # Metadata
    commit = git_commit_hash()
    logits_mode = 'sampled' if bool(args.use_sparse_logits) else 'full'
    model_flags = {
        'use_sparse_logits': bool(args.use_sparse_logits),
        'classifier_type': getattr(getattr(model, 'classification_head', None), 'classifier_type', 'linear'),
        'classifier_scale': float(getattr(getattr(model, 'classification_head', None), 'scale', 0.0)) if hasattr(getattr(model, 'classification_head', None), 'scale') else None,
        'recent_full_frames_applied': knobs.get('recent_full_frames'),
        'context_latents_applied': knobs.get('context_latents'),
        'logits_mode': logits_mode,
        'decode_mode': str(args.decode_mode),
        'temperature': float(args.temperature),
        'sample_topk': int(args.sample_topk or 0),
        'temp_anneal_gamma': float(args.temp_anneal_gamma),
        'min_temperature': float(args.min_temperature),
        'top_p': float(args.top_p or 0.0),
        'copy_bias': float(args.copy_bias or 0.0),
        'min_dwell': int(args.min_dwell or 1),
        'transition_neighbors': str(args.transition_neighbors) if args.transition_neighbors else None,
        'neighbor_k': int(args.neighbor_k),
        'neighbor_fallback_top_p': float(args.neighbor_fallback_top_p),
        'use_context_prior': bool(args.use_context_prior),
        'context_prior_weight': float(args.context_prior_weight),
    }

    metadata = {
        'protein_id': traj_name,
        'protein_display_name': prot_name,
        'time_start': int(args.time_start),
        'time_steps': int(args.time_steps),
        'timestep_ns': 0.2,
        'recent_full_frames': int(args.recent_full_frames),
        'context_latents': int(args.context_latents),
        'k_residues': int(args.k_residues),
        'residue_indices': ridxs,
        'ckpt_path': str(Path(args.ckpt).resolve()),
        'commit_hash': commit,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime()),
        'model_flags': model_flags,
        'notes': 'If sparse logits were active, top-1 is within the sampled set; otherwise full top-1.',
    }
    save_metadata_json(out_dir, metadata)

    # README
    readme = (
        f"Autoregressive eval for {traj_name}\n"
        f"Checkpoint: {args.ckpt}\n"
        f"Window: [{args.time_start}, {args.time_start + args.time_steps - 1}]\n"
        f"recent_full_frames={args.recent_full_frames}, context_latents={args.context_latents}\n"
        f"Residues plotted (k={args.k_residues}, mode={args.residue_select}): {ridxs}\n"
        f"Outputs:\n"
        f"  - {out_dir / 'summary.png'} (GT heatmap, Pred heatmap, Acc vs time)\n"
        f"  - {out_dir / 'cluster_vs_time.png'} (GT vs Pred for selected residues)\n"
        f"  - {out_dir / 'metadata.json'}\n"
    )
    save_readme(out_dir, readme)

    # Distributional metrics
    try:
        num_classes = getattr(getattr(model, 'classification_head', None), 'num_clusters', int(np.max(eval_out.gt) + 1))
        dist = compute_distributional_metrics(eval_out.gt, eval_out.pred, num_classes)
        # Per-residue metrics summary
        dist_per = compute_per_residue_metrics(eval_out.gt, eval_out.pred, int(num_classes), top_k=10)
        dist['per_residue'] = dist_per
        eval_out.dist_metrics = dist
        with open(out_dir / 'distribution_metrics.json', 'w') as f:
            json.dump(dist, f, indent=2)
        log.info("Distributional metrics: JS(vis)=%.4f, KS(dwell)=%.4f, rowKL=%.4f, frob=%.4f, JS_trans(mean)=%s",
                 dist.get('visitation_js', float('nan')),
                 dist.get('dwell_stats', {}).get('ks_stat', float('nan')),
                 dist.get('transition_row_kl_gt_pr', float('nan')),
                 dist.get('transition_frobenius', float('nan')),
                 str(dist.get('transition_js_mean_over_sources')))
        log.info("Per-residue: meanJS=%.4f, medianJS=%.4f, p90JS=%.4f, covGT=%.3f, covPred=%.3f",
                 dist_per.get('per_residue_js_mean', float('nan')),
                 dist_per.get('per_residue_js_median', float('nan')),
                 dist_per.get('per_residue_js_p90', float('nan')),
                 dist_per.get('per_residue_coverage_gt_mean', float('nan')),
                 dist_per.get('per_residue_coverage_pred_mean', float('nan')))
        if 'dwell_stats' in dist and 'pr_geometric_nll' in dist['dwell_stats']:
            ds = dist['dwell_stats']
            log.info("Dwell geometric NLL: gt=%.4f, pred=%.4f (p=%.4f)",
                     ds.get('gt_geometric_nll', float('nan')),
                     ds.get('pr_geometric_nll', float('nan')),
                     ds.get('gt_geometric_p', float('nan')))
        if bool(getattr(args, 'plot_distributions', False)):
            plot_distributions(
                traj_name=traj_name,
                gt=eval_out.gt,
                pred=eval_out.pred,
                ridxs=ridxs,
                out_dir=out_dir,
                num_classes=int(num_classes),
                topk_clusters=int(getattr(args, 'plot_topk_clusters', 20)),
                transitions_top_states=int(getattr(args, 'plot_transitions_top_states', 15)),
            )
    except Exception as e:
        log.warning("Failed to compute distributional metrics: %s", e)

    log.info("Done. Results in: %s", out_dir)


if __name__ == '__main__':
    main()
