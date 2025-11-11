#!/usr/bin/env python3
"""
Visualize attention for every decoder block in EMPROT (hybrid context), plus latent pool.

CLI:
  python viz_attention_all_blocks.py \
    --ckpt <optional_path> \
    --mode {latent_cap,hier_pool} \
    --layout cta,spatial,cta,cta,spatial,cta \
    --B 2 --T 8 --N 64 --C 50000 --d 512 --heads 8 \
    --recent_full 5

Saves figures:
  attn_latentpool.png
  attn_block_01_cta.png, attn_block_02_spatial.png, ...
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Make local package importable if not installed
sys.path.insert(0, os.path.abspath('.'))

from emprot.models.transformer import ProteinTransformerClassificationOnly
from emprot.models.cta import CrossTemporalAttention, SpatialAxialAttention


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--mode', type=str, default='latent_cap', choices=['latent_cap', 'hier_pool'])
    ap.add_argument('--layout', type=str, default='cta,spatial,cta,cta,spatial,cta')
    ap.add_argument('--B', type=int, default=2)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--C', type=int, default=50000)
    ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--recent_full', type=int, default=5)
    return ap.parse_args()


def build_model(args) -> ProteinTransformerClassificationOnly:
    latent_cfg = {
        'enabled': True,
        'num_latents': 32,
        'layers': 1,
        'heads': args.heads,
        'dropout': 0.0,
        'context_mode': args.mode,
        'summarizer': {'num_latents': 32, 'layers': 1, 'heads': args.heads, 'dropout': 0.0},
        'train_context': {'kmin': 3, 'kmax': 8, 'variable_k': True, 'stopgrad_tail': True},
        'hier_pool': {'R': 3, 'scorer_hidden': 256},
        'memory': {'store_attn_eval_only': True},
    }
    layout = [tok.strip() for tok in args.layout.split(',') if tok.strip()]
    model = ProteinTransformerClassificationOnly(
        d_embed=args.d,
        num_heads=args.heads,
        dropout=0.0,
        use_gradient_checkpointing=False,
        min_context_frames=2,
        num_layers=1,
        num_clusters=args.C,
        hybrid_context=True,
        recent_full_frames=args.recent_full,
        decoder_layout=layout,
        latent_summary_config=latent_cfg,
        spatial_attn_temperature=1.0,
        spatial_attn_dropout=0.0,
        latent_skip_gate=False,
    )
    model.eval()
    return model


def enable_attn_capture(model: ProteinTransformerClassificationOnly) -> None:
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'decoder_blocks'):
        for blk in model.backbone.decoder_blocks:
            if hasattr(blk, 'attn') and isinstance(blk.attn, CrossTemporalAttention):
                blk.attn.store_attention_weights = True
            if hasattr(blk, 'attn') and isinstance(blk.attn, SpatialAxialAttention):
                blk.attn.store_attention_weights = True


def synthetic_batch(args, device) -> dict:
    B, T, N, C = args.B, args.T, args.N, args.C
    # Cluster ids with some padding zeros
    ids = torch.randint(low=1, high=max(2, C), size=(B, T, N), device=device)
    pad = torch.rand(B, T, N, device=device) < 0.05
    ids[pad] = 0
    # Times increasing
    times = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    # Sequence lengths (ensure >= recent_full)
    seq_lens = torch.randint(low=max(3, args.recent_full + 1), high=T + 1, size=(B,), device=device)
    # History mask True=valid
    hist_mask = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1) < seq_lens.view(B, 1, 1)
    hist_mask = hist_mask.expand(-1, -1, N)
    return {
        'input_cluster_ids': ids,
        'times': times,
        'sequence_lengths': seq_lens,
        'history_mask': hist_mask,
    }


def plot_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, vline: int | None, out: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(mat, aspect='auto', interpolation='nearest')
    if vline is not None:
        plt.axvline(x=vline - 0.5, color='w', linestyle='--', linewidth=1)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args).to(device)
    enable_attn_capture(model)

    out_dir = os.path.dirname(args.ckpt).split('/')[-1]
    out_dir = f"output/evaluation_results/{out_dir}/attention_viz"
    # Optional checkpoint load
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            state = torch.load(args.ckpt, map_location=device)
            sd = state.get('model_state_dict', state)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")

    batch = synthetic_batch(args, device)

    # Forward (eval, teacher-forced path)
    with torch.no_grad():
        _ = model(
            input_cluster_ids=batch['input_cluster_ids'],
            times=batch['times'],
            sequence_lengths=batch['sequence_lengths'],
            history_mask=batch['history_mask'],
            t_scalar=None,
        )

    # LatentPool attention from ContextBuilder.summarizer
    cb = getattr(model.backbone, 'context_builder', None)
    lat_w = None
    if cb is not None and getattr(cb, 'summarizer', None) is not None:
        lat_w = getattr(cb.summarizer, 'last_attention_weights', None)
    if lat_w is None:
        print("[LatentPool] No attention captured. Ensure eval and memory.store_attn_eval_only=True.")
    else:
        w = lat_w.numpy() if hasattr(lat_w, 'numpy') else np.asarray(lat_w)
        if w.ndim == 3:  # (H, L, S)
            print(f"LatentPool attn shape: {w.shape}")
            plot_heatmap(w.mean(axis=0), f"LatentPool ({args.mode}) avg heads", xlabel="Tail tokens (S)", ylabel="Latents (L)", vline=None, out=f"{out_dir}/attn_latentpool.png")
        else:
            print(f"LatentPool unexpected shape: {w.shape}")

    # Determine KV partition sizes (latents | recent)
    try:
        emb = model.cluster_embedding(batch['input_cluster_ids'])
        key_all, key_mask, delta_t, latent_tokens = model.backbone._build_hybrid_context(emb, batch['history_mask'], batch['sequence_lengths'])
        kv_lat = int(latent_tokens.shape[1]) if (latent_tokens is not None and latent_tokens.numel() > 0) else 0
        kv_recent = int(key_all.shape[1] - kv_lat)
    except Exception as e:
        print(f"Partition size probe failed: {e}")
        kv_lat = 0
        kv_recent = 0

    # Iterate blocks and save attention
    for idx, blk in enumerate(getattr(model.backbone, 'decoder_blocks', []), start=1):
        # CTA
        if hasattr(blk, 'attn') and isinstance(blk.attn, CrossTemporalAttention):
            w = getattr(blk, 'last_attention_weights', None)
            if w is None:
                print(f"Block {idx} CTA: attention not captured. Run in eval and set store_attention_weights=True.")
                continue
            arr = w.numpy() if hasattr(w, 'numpy') else np.asarray(w)
            # Expected shapes: (H, Lq, S) or (B, H, Lq, S). Use batch 0 if present.
            if arr.ndim == 4:
                arr = arr[0]
            print(f"Block {idx} CTA attn shape: {arr.shape}")
            avg = arr.mean(axis=0)  # (Lq, S)
            vline = kv_lat if kv_lat > 0 else None
            plot_heatmap(avg, f"CTA block {idx} (avg heads)", xlabel=f"Keys (latents|recent={kv_lat}|{kv_recent})", ylabel="Queries (Lq)", vline=vline, out=f"{out_dir}/attn_block_{idx:02d}_cta.png")
        # Spatial
        elif hasattr(blk, 'attn') and isinstance(blk.attn, SpatialAxialAttention):
            w = getattr(blk.attn, 'last_attention_weights', None)
            if w is None:
                print(f"Block {idx} Spatial: attention not captured. Run in eval and set store_attention_weights=True.")
                continue
            arr = w.numpy() if hasattr(w, 'numpy') else np.asarray(w)
            if arr.ndim == 4:
                arr = arr[0]
            print(f"Block {idx} Spatial attn shape: {arr.shape}")
            avg = arr.mean(axis=0)  # (N, N)
            plot_heatmap(avg, f"Spatial block {idx} (avg heads)", xlabel="Residue (key)", ylabel="Residue (query)", vline=None, out=f"{out_dir}/attn_block_{idx:02d}_spatial.png")
        else:
            print(f"Block {idx} unknown type; skipping")

    print("Done. Saved attention figures.")


if __name__ == '__main__':
    main()


