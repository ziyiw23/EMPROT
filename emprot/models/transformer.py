import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from emprot.models.cta import build_attention
from emprot.models.context import TemporalFeatureProjector, prefix_ema_sequence, build_parallel_windows, build_prefix_states_from_ema, StreamingLatentPool

class TemporalEncoder(nn.Module):
    """
    Simplified temporal encoder using sinusoidal positional encodings only.
    Adds sinusoidal encodings across the full embedding dimension and sums with inputs.
    """
    def __init__(self, d_embed: int, max_time_steps: int = 10000, max_time_gap_ns: float = 20.0):
        super().__init__()
        self.d_embed = d_embed
        self.max_time_gap_ns = max_time_gap_ns
        self.register_buffer('sinusoidal_table', self._get_sinusoidal_table(max_time_steps, d_embed))
        
    def _get_sinusoidal_table(self, max_time_steps: int, d_model: int) -> torch.Tensor:
        """Vectorized sinusoidal table (float32); buffer will follow module device."""
        pos = torch.arange(max_time_steps, dtype=torch.float32).unsqueeze(1)  # (T,1)
        # Base 10000.0 frequency schedule (standard Transformer)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_time_steps, d_model, dtype=torch.float32)
        table[:, 0::2] = torch.sin(pos * div_term)
        table[:, 1::2] = torch.cos(pos * div_term)
        return table
        
    def forward(self, 
                embeddings: torch.Tensor, 
                times: torch.Tensor,
                sequence_lengths: torch.Tensor,
                t_scalar: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, N, E = embeddings.shape
        device = embeddings.device
        valid_mask_bt = (torch.arange(T, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1))
        time_indices = torch.round(times * 10).long().clamp(0, self.sinusoidal_table.size(0) - 1)
        sinusoidal_embed = self.sinusoidal_table[time_indices]  # (B, T, E)
        sinusoidal_embed = sinusoidal_embed.unsqueeze(2).expand(-1, -1, N, -1)
        sinusoidal_embed = sinusoidal_embed * valid_mask_bt.unsqueeze(2).unsqueeze(-1).float()
        
        # Scale embeddings to match PE magnitude (standard Transformer practice)
        embeddings = embeddings * math.sqrt(self.d_embed)
        
        return embeddings + sinusoidal_embed

class TemporalBackbone(nn.Module):
    """
    Shared temporal backbone with optional hybrid context decoder that mixes recent frames
    at full resolution with latent summaries of distant history.
    """

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        min_context_frames: int = 1,
        num_layers: int = 1,
        attention_type: str = 'cross_temporal',
        per_source_kv: bool = False,
        per_source_kv_max_buckets: int = 0,
        recent_full_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.min_context_frames = min_context_frames
        self.num_layers = int(max(1, num_layers))
        self.attention_type = attention_type
        self.hybrid_context_enabled = False
        self.per_source_kv = bool(per_source_kv)
        self.per_source_kv_max_buckets = int(max(0, per_source_kv_max_buckets))
        self.recent_full_frames = int(recent_full_frames) if (recent_full_frames is not None) else None

        # Learnable Residue Positional Embedding (Spatial)
        # Max residues 2048
        self.residue_pos_emb = nn.Embedding(2048, d_embed)
        # Initialize with small variance
        nn.init.normal_(self.residue_pos_emb.weight, mean=0.0, std=0.02)

        self.input_norm = nn.LayerNorm(d_embed, eps=1e-6)
        self.temporal_encoder = TemporalEncoder(
            d_embed=d_embed,
            max_time_steps=1000,
            max_time_gap_ns=20.0,
        )
        self.feature_projector = None
        self.last_attention_weights: Optional[torch.Tensor] = None

        self.latent_cfg = {}
        self.context_builder = None
        self.latent_summarizer = None
        self.decoder_blocks = None
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            attn = build_attention(
                attention_type=self.attention_type,
                d_model=d_embed,
                num_heads=num_heads,
                dropout=dropout,
                per_source_kv=self.per_source_kv,
                max_source_buckets=self.per_source_kv_max_buckets,
            )
            attn_norm = nn.LayerNorm(d_embed)
            ffn = nn.Sequential(
                nn.Linear(d_embed, 4 * d_embed),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_embed, d_embed),
                nn.Dropout(dropout),
            )
            ffn_norm = nn.LayerNorm(d_embed)
            self.layers.append(
                nn.ModuleDict({
                    'attn': attn,
                    'attn_norm': attn_norm,
                    'ffn': ffn,
                    'ffn_norm': ffn_norm,
                })
            )
        # Rolling state EMA coefficient
        self.state_ema_alpha = 0.9
    

    def encode(
        self,
        embeddings: torch.Tensor,
        times: torch.Tensor,
        sequence_lengths: torch.Tensor,
        history_mask: torch.Tensor,
        t_scalar: Optional[torch.Tensor] = None,
        change_mask: Optional[torch.Tensor] = None,
        run_length: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        extra_kv: Optional[torch.Tensor] = None,
        extra_kv_mask: Optional[torch.Tensor] = None,
        extra_kv_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Temporal feature projector disabled by default
        embeddings = self.input_norm(embeddings)
        B, T_max, N_max, _ = embeddings.shape
        
        # Add Spatial (Residue) Positional Embeddings
        # (1, 1, N, D) broadcast to (B, T, N, D)
        # Clamp indices to [0, 2047] to handle extremely large proteins gracefully
        res_indices = torch.arange(N_max, device=embeddings.device).clamp(max=2047)
        res_emb = self.residue_pos_emb(res_indices).view(1, 1, N_max, -1)
        embeddings = embeddings + res_emb

        embeddings = self.temporal_encoder(embeddings, times, sequence_lengths, t_scalar)
        if T_max < self.min_context_frames:
            raise ValueError(
                f"The longest sequence in this batch ({T_max}) is less than the minimum required context of {self.min_context_frames} frames."
            )

        last_indices = (sequence_lengths - 1).clamp(min=0)
        gather_idx = last_indices.view(B, 1, 1, 1).expand(-1, 1, N_max, embeddings.size(-1))
        query0 = torch.gather(embeddings, 1, gather_idx).squeeze(1)
        out = self._encode_standard(
            embeddings, times, sequence_lengths, history_mask, query0, state,
            extra_kv=extra_kv, extra_kv_mask=extra_kv_mask, extra_kv_time=extra_kv_time,
        )
        h = out
        new_state: Optional[torch.Tensor] = None
        if state is not None:
            alpha = float(self.state_ema_alpha)
            if alpha < 0.0:
                alpha = 0.0
            if alpha > 1.0:
                alpha = 1.0
            new_state = alpha * state + (1.0 - alpha) * h.detach()
        return query0, h, new_state

    def _encode_standard(
        self,
        embeddings: torch.Tensor,
        times: torch.Tensor,
        sequence_lengths: torch.Tensor,
        history_mask: torch.Tensor,
        query0: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        extra_kv: Optional[torch.Tensor] = None,
        extra_kv_mask: Optional[torch.Tensor] = None,
        extra_kv_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_max, N_max, _ = embeddings.shape
        query_time_indices = (sequence_lengths - 1).clamp(min=0).view(B, 1)
        query_times = torch.gather(times, 1, query_time_indices).expand(-1, N_max)
        history = embeddings.view(B, T_max * N_max, embeddings.size(-1))
        history_times = times.repeat_interleave(N_max, dim=1)
        history_padding_mask = history_mask.view(B, T_max * N_max)
        if state is not None and state.numel() > 0:
            st = state.view(B, N_max, embeddings.size(-1))
            history = torch.cat([st, history], dim=1)
            # approximate an earlier time index for state tokens
            min_t = times.min(dim=1, keepdim=True).values - 1.0
            state_times = min_t.expand(B, 1).repeat_interleave(N_max, dim=1)
            history_times = torch.cat([state_times, history_times], dim=1)
            # use last-frame residue validity as state mask
            last_mask = history_mask[:, -1, :]
            history_padding_mask = torch.cat([last_mask, history_padding_mask], dim=1)
        # Append extra_kv tokens (latents) along token axis at the same time as the last frame
        if extra_kv is not None and extra_kv.numel() > 0:
            # extra_kv: (B, M, D)
            M = extra_kv.size(1)
            history = torch.cat([history, extra_kv], dim=1)
            if extra_kv_time is None:
                # default to last frame time
                t_last = torch.gather(times, 1, (sequence_lengths - 1).clamp(min=0).view(B, 1)).expand(B, M)
                extra_t = t_last
            else:
                # support (B,) or (B,1) or (B,M)
                if extra_kv_time.dim() == 1:
                    extra_t = extra_kv_time.view(B, 1).expand(B, M)
                elif extra_kv_time.dim() == 2 and extra_kv_time.size(1) == 1:
                    extra_t = extra_kv_time.expand(B, M)
                else:
                    extra_t = extra_kv_time
            history_times = torch.cat([history_times, extra_t], dim=1)
            if extra_kv_mask is None:
                extra_kv_mask = torch.ones(B, M, dtype=history_mask.dtype, device=history_mask.device)
            history_padding_mask = torch.cat([history_padding_mask, extra_kv_mask], dim=1)
        if history_padding_mask.dtype == torch.bool:
            history_padding_mask_kpm = ~history_padding_mask
        else:
            history_padding_mask_kpm = (history_padding_mask <= 0)

        x = query0
        self.last_attention_weights = None
        for layer in self.layers:
            attn_fn = layer['attn']
            # Build optional per-source bucket ids for K/V modulation
            source_bucket_ids = None
            if getattr(attn_fn, 'per_source_kv', False) and getattr(attn_fn, 'max_source_buckets', 0) > 0:
                K = int(self.recent_full_frames) if (self.recent_full_frames is not None and self.recent_full_frames > 0) else min(5, T_max)
                # Buckets: 0..K-1 for offsets 0..K-1 (F-1 .. F-K), K for older frames, K+1 for latents
                max_buckets = K + 2
                sb = torch.full((B, history.shape[1]), K, dtype=torch.long, device=history.device)
                # If state was prepended, mark as older
                prefix = 0
                if state is not None and state.numel() > 0:
                    prefix = N_max
                    # already set to older for first N_max
                # Frame tokens
                base_len = T_max * N_max
                frame_indices = torch.arange(T_max, device=times.device).repeat_interleave(N_max).view(1, -1).expand(B, -1)
                last_idx = (sequence_lengths - 1).view(B, 1)
                offsets = (last_idx - frame_indices).clamp(min=0)
                offsets = torch.where(offsets < K, offsets, torch.full_like(offsets, K))
                sb[:, prefix:prefix + base_len] = offsets
                # Latents bucket at the end if present
                total_S = history.shape[1]
                tail = total_S - (prefix + base_len)
                if tail > 0:
                    sb[:, -tail:] = K + 1
                source_bucket_ids = sb.clamp(max=max_buckets - 1)
            attn_out = attn_fn(
                query=x,
                key=history,
                value=history,
                query_times=query_times,
                key_times=history_times,
                key_padding_mask=history_padding_mask_kpm,
                source_bucket_ids=source_bucket_ids,
            )
            self.last_attention_weights = getattr(attn_fn, 'last_attention_weights', None)
            x = layer['attn_norm'](x + attn_out)
            x = layer['ffn_norm'](x + layer['ffn'](x))
        return x

    # Legacy hybrid context removed

class ClassificationHead(nn.Module):
    def __init__(self, d_embed: int, num_clusters: int = 50000,
                 classifier_type: str = 'linear', classifier_scale: float = 20.0, dropout: float = 0.1):
        super().__init__()
        self.classifier_type = (classifier_type or 'linear').lower()
        self.num_clusters = num_clusters
        if self.classifier_type == 'cosine':
            # Weight only; use cosine similarity with normalization
            self.weight = nn.Parameter(torch.empty(num_clusters, d_embed))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
            self.scale = float(classifier_scale)
        else:
            # Default MLP-based linear head (baseline-compatible)
            self.net = nn.Sequential(
                nn.Linear(d_embed, d_embed // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_embed // 2, num_clusters)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier_type == 'cosine':
            # x: (B, N, D)
            x_flat = x.reshape(-1, x.shape[-1])
            x_n = F.normalize(x_flat, dim=-1)
            w_n = F.normalize(self.weight, dim=-1)
            logits_flat = self.scale * (x_n @ w_n.t())  # (B*N, C)
            return logits_flat.view(x.shape[0], x.shape[1], self.num_clusters)
        else:
            return self.net(x)

class ProteinTransformerClassificationOnly(nn.Module):
    def __init__(self, d_embed: int, num_heads: int, dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True, min_context_frames: int = 1,
                 num_clusters: int = 50000, num_layers: int = 1, attention_type: str = 'cross_temporal',
                 future_horizon: int = 1,
                 recent_full_frames: Optional[int] = None,
                 latent_summary_enabled: bool = False,
                 latent_summary_num_latents: int = 0,
                 latent_summary_heads: Optional[int] = None,
                 latent_summary_dropout: Optional[float] = None,
                 latent_summary_max_prefix: Optional[int] = None,
                 per_source_kv: bool = False,
                 per_source_kv_max_buckets: Optional[int] = None,
                 pretrained_input_dim: Optional[int] = None,
                 **backbone_kwargs):
        super().__init__()
        self.future_horizon = int(max(1, future_horizon))
        self.backbone = TemporalBackbone(
            d_embed=d_embed,
            num_heads=num_heads,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
            min_context_frames=min_context_frames,
            num_layers=num_layers,
            attention_type=attention_type,
            per_source_kv=bool(per_source_kv),
            per_source_kv_max_buckets=int(per_source_kv_max_buckets) if per_source_kv_max_buckets is not None else (int(recent_full_frames) + 2 if recent_full_frames else 0),
            recent_full_frames=recent_full_frames,
            **backbone_kwargs,
        )
        # If set, use only the last K_recent frames of provided history as full-resolution context
        self.recent_full_frames: Optional[int] = int(recent_full_frames) if recent_full_frames is not None else None
        self.classification_head = ClassificationHead(d_embed, num_clusters, classifier_type='linear', dropout=dropout)
        self.cluster_embedding = nn.Embedding(num_embeddings=num_clusters + 1, embedding_dim=d_embed, padding_idx=0)
        # Optional global latent pool summarizer
        self.latent_pool: Optional[StreamingLatentPool] = None
        if bool(latent_summary_enabled) and int(latent_summary_num_latents) > 0:
            pool_heads = int(latent_summary_heads) if latent_summary_heads is not None else num_heads
            pool_dropout = float(latent_summary_dropout) if latent_summary_dropout is not None else dropout
            self.latent_pool = StreamingLatentPool(d_model=d_embed, num_latents=int(latent_summary_num_latents), heads=pool_heads, dropout=pool_dropout)
        self.latent_max_prefix = int(latent_summary_max_prefix) if latent_summary_max_prefix is not None else None
        
        # Input Projector for dense embeddings
        self.pretrained_input_dim = pretrained_input_dim
        if pretrained_input_dim is not None:
            self.input_projector = nn.Linear(int(pretrained_input_dim), d_embed)
        else:
            self.input_projector = None

    def _normalize_inference_inputs(
        self,
        cluster_ids: torch.Tensor,
        times: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
        history_mask: Optional[torch.Tensor],
        change_mask: Optional[torch.Tensor] = None,
        run_length: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure inputs satisfy minimum context and augment with temporal features."""

        if cluster_ids.dim() != 3:
            raise ValueError(f"Expected cluster_ids to have shape (B, T, N); got {cluster_ids.shape}")

        device = cluster_ids.device
        B, T, N = cluster_ids.shape

        if times is None:
            times = torch.zeros(B, T, dtype=torch.float32, device=device)
        else:
            times = times.to(device=device, dtype=torch.float32)

        if history_mask is None:
            history_mask = torch.ones(B, T, N, dtype=torch.bool, device=device)
        else:
            history_mask = history_mask.to(device=device, dtype=torch.bool)

        if sequence_lengths is None:
            sequence_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)

        def _align_feature(tensor: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            tensor = tensor.to(device=device)
            if tensor.dim() == 3:
                if tensor.shape[:2] != (B, T):
                    raise ValueError(f"{name} must have shape (B,T,N); got {tensor.shape}")
                return tensor
            if tensor.dim() == 2:
                if tensor.shape[0] not in (B, T):
                    raise ValueError(f"{name} must have shape (B,T) or (T,N)")
                if tensor.shape[0] == B:
                    return tensor.unsqueeze(1).expand(-1, T, -1)
                return tensor.unsqueeze(0).expand(B, -1, -1)
            if tensor.dim() == 1:
                return tensor.view(1, T, 1).expand(B, T, N)
            raise ValueError(f"Unsupported {name} shape: {tensor.shape}")

        aligned_change = _align_feature(change_mask, "change_mask")
        aligned_run = _align_feature(run_length, "run_length")

        if delta_t is not None:
            delta_t = delta_t.to(device=device, dtype=torch.float32)
            if delta_t.dim() == 1:
                delta_t = delta_t.view(1, -1).expand(B, -1)
            elif delta_t.dim() != 2:
                raise ValueError(f"delta_t must have shape (B,T) or (T,); got {delta_t.shape}")
            if delta_t.size(1) != T:
                raise ValueError(f"delta_t has mismatched temporal dimension: expected {T}, got {delta_t.size(1)}")

        min_ctx = int(max(1, getattr(self.backbone, 'min_context_frames', 1)))
        hybrid_recent = int(max(1, getattr(self.backbone, 'recent_full_frames', 1))) if getattr(self.backbone, 'hybrid_context_enabled', False) else 1
        required = max(min_ctx, hybrid_recent)
        if T < required:
            pad = required - T
            pad_src = cluster_ids[:, :1, :]
            pad_ids = pad_src.expand(-1, pad, -1).clone()
            cluster_ids = torch.cat([pad_ids, cluster_ids], dim=1)

            times_dtype = times.dtype
            if T >= 2:
                delta = (times[:, 1:2] - times[:, :1]).abs().clamp(min=1e-6)
            else:
                delta = torch.ones(B, 1, dtype=times_dtype, device=device)
            offsets = torch.arange(pad, 0, -1, device=device, dtype=times_dtype).view(1, pad)
            pad_times = times[:, :1] - offsets * delta
            times = torch.cat([pad_times, times], dim=1)

            pad_mask = history_mask[:, :1, :].expand(-1, pad, -1).clone()
            history_mask = torch.cat([pad_mask, history_mask], dim=1)

            if aligned_change is not None:
                pad_change = torch.zeros(B, pad, N, dtype=aligned_change.dtype, device=device)
                aligned_change = torch.cat([pad_change, aligned_change], dim=1)
            if aligned_run is not None:
                pad_run = torch.ones(B, pad, N, dtype=torch.long, device=device)
                aligned_run = torch.cat([pad_run, aligned_run], dim=1)
            if delta_t is not None:
                pad_delta = torch.zeros(B, pad, dtype=delta_t.dtype, device=device)
                delta_t = torch.cat([pad_delta, delta_t], dim=1)

            sequence_lengths = sequence_lengths + pad

        if times.size(1) != cluster_ids.size(1):
            raise ValueError("Times tensor is not aligned with cluster history after padding")
        if history_mask.size(1) != cluster_ids.size(1):
            raise ValueError("History mask is not aligned with cluster history after padding")

        B, T_new, N_new = cluster_ids.shape

        if aligned_change is None:
            aligned_change = torch.zeros(B, T_new, N_new, dtype=torch.bool, device=device)
            if T_new > 1:
                aligned_change[:, 1:] = cluster_ids[:, 1:] != cluster_ids[:, :-1]
        else:
            if aligned_change.size(1) != T_new:
                if aligned_change.size(1) < T_new:
                    pad_change = torch.zeros(B, T_new - aligned_change.size(1), N_new, dtype=aligned_change.dtype, device=device)
                    aligned_change = torch.cat([pad_change, aligned_change], dim=1)
                else:
                    aligned_change = aligned_change[:, -T_new:, :]

        if aligned_run is None:
            aligned_run = torch.ones(B, T_new, N_new, dtype=torch.long, device=device)
            for t in range(1, T_new):
                same = aligned_change[:, t].logical_not()
                prev = aligned_run[:, t - 1] + 1
                aligned_run[:, t] = torch.where(same, prev, torch.ones_like(prev))
        else:
            if aligned_run.size(1) != T_new:
                if aligned_run.size(1) < T_new:
                    pad_run = torch.ones(B, T_new - aligned_run.size(1), N_new, dtype=aligned_run.dtype, device=device)
                    aligned_run = torch.cat([pad_run, aligned_run], dim=1)
                else:
                    aligned_run = aligned_run[:, -T_new:, :]

        if delta_t is None:
            delta_t = torch.zeros(B, T_new, dtype=torch.float32, device=device)
            if T_new > 1:
                delta_t[:, 1:] = times[:, 1:] - times[:, :-1]
        else:
            if delta_t.size(1) != T_new:
                if delta_t.size(1) < T_new:
                    pad_delta = torch.zeros(B, T_new - delta_t.size(1), dtype=delta_t.dtype, device=device)
                    delta_t = torch.cat([pad_delta, delta_t], dim=1)
                else:
                    delta_t = delta_t[:, -T_new:]

        mask_bool = history_mask.to(device=device, dtype=torch.bool)
        aligned_change = aligned_change & mask_bool
        aligned_run = aligned_run * mask_bool.long()

        return cluster_ids, times, sequence_lengths, history_mask, aligned_change, aligned_run, delta_t

    def forward(self,
                input_cluster_ids: torch.Tensor,
                times: Optional[torch.Tensor],
                sequence_lengths: Optional[torch.Tensor],
                history_mask: Optional[torch.Tensor],
                change_mask: Optional[torch.Tensor] = None,
                run_length: Optional[torch.Tensor] = None,
                delta_t: Optional[torch.Tensor] = None,
                t_scalar: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None,
                teacher_future_ids: Optional[torch.Tensor] = None,
                future_step_mask: Optional[torch.Tensor] = None,
                scheduled_sampling_p: float = 0.0,
                input_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        input_cluster_ids, times, sequence_lengths, history_mask, change_mask, run_length, delta_t = self._normalize_inference_inputs(
            input_cluster_ids, times, sequence_lengths, history_mask,
            change_mask=change_mask, run_length=run_length, delta_t=delta_t
        )

        max_emb_idx = self.cluster_embedding.num_embeddings - 1
        def _safe_clamp(t: torch.Tensor) -> torch.Tensor:
            return torch.clamp(torch.where(t < 0, 0, t), min=0, max=max_emb_idx)

        valid_cluster_ids = _safe_clamp(input_cluster_ids)
        
        align_loss = 0.0
        if input_embeddings is not None and self.input_projector is not None:
            # Normalize inputs if needed, assume already correct shape (B, T, N, D_pre) from loader
            # Project dense embeddings to model dim
            embeddings = self.input_projector(input_embeddings.to(valid_cluster_ids.device)) # (B, T, N, D_model)
            
            # Compute alignment loss: force lookup table to match projected dense input
            # Targets for alignment are the cluster IDs (lookup table entries)
            # We want E[id] ~= Project(dense)
            # Detach the dense path so we only update the embedding table to match the dense input?
            # Actually, standard practice is usually to train both or detach the target.
            # Here, the dense input is the "ground truth" semantic signal. 
            # We want the embedding table (which is random init) to move towards the dense input.
            # We definitely want gradients to flow into input_projector to adapt dense input to model space.
            # But we also want gradients to flow into cluster_embedding to align it.
            # So we don't detach either side, or we treat 'embeddings' as target?
            # Let's treat embeddings (dense) as target for the lookup table?
            # Actually, simplest is MSE(lookup, projected_dense). Both are trainable. 
            # They will meet in the middle, but projected_dense is anchored by the fixed input data.
            lookup_emb = self.cluster_embedding(valid_cluster_ids)
            
            # Mask out padding/invalid IDs for loss calculation
            # alignment_mask = (input_cluster_ids >= 0) & history_mask
            # Just use history_mask since valid_cluster_ids are clamped
            if history_mask is not None:
                mask_float = history_mask.float().unsqueeze(-1)
                diff = (lookup_emb - embeddings) * mask_float
                align_loss = (diff ** 2).sum() / (mask_float.sum() * lookup_emb.size(-1) + 1e-6)
            else:
                align_loss = F.mse_loss(lookup_emb, embeddings)
        else:
            embeddings = self.cluster_embedding(valid_cluster_ids)

        cluster_valid_mask = (input_cluster_ids >= 0).bool()
        effective_mask = history_mask & cluster_valid_mask
        change_mask = change_mask & effective_mask
        run_length = run_length * effective_mask.long()

        # If teacher_future_ids provided: run multi-horizon with teacher forcing / scheduled sampling
        if torch.is_tensor(teacher_future_ids) and float(scheduled_sampling_p) == 0.0:
            # Parallel direct multi-horizon (teacher forcing, no scheduled sampling)
            B, T, N = input_cluster_ids.shape
            Fh = int(min(self.future_horizon, teacher_future_ids.shape[1]))
            # Determine K_recent (<= T) and base offset
            K_recent = int(self.recent_full_frames) if (self.recent_full_frames is not None and self.recent_full_frames > 0) else T
            K_recent = int(max(1, min(K_recent, T)))
            base = int(T - K_recent)
            safe_teacher = _safe_clamp(teacher_future_ids[:, :Fh, :])
            all_ids = torch.cat([valid_cluster_ids, safe_teacher], dim=1)  # (B, T+Fh, N)
            all_emb = self.cluster_embedding(all_ids)  # (B, T+Fh, N, D)
            alpha = float(getattr(self.backbone, 'state_ema_alpha', 0.9))
            ema_seq = prefix_ema_sequence(all_emb, alpha)  # (B, L, N, D)
            # Build K_recent windows starting from base
            ids_wins = []
            for f in range(Fh):
                s = base + f
                ids_wins.append(all_ids[:, s:s + K_recent, :])
            ids_windows = torch.stack(ids_wins, dim=1)  # (B, Fh, K_recent, N)
            emb_windows = self.cluster_embedding(ids_windows.view(B * Fh, K_recent, N))  # (B*Fh, K_recent, N, D)
            # Base times: last K_recent history times; approximate futures using constant dt
            if times is None:
                times = torch.zeros(B, T, dtype=torch.float32, device=all_emb.device)
            base_times = times[:, max(0, T - K_recent):]
            if K_recent > 1:
                dt = (base_times[:, 1] - base_times[:, 0]).to(dtype=torch.float32)
            else:
                dt = torch.ones(B, dtype=torch.float32, device=all_emb.device)
            times_list = [base_times + dt.view(B, 1) * float(f) for f in range(Fh)]
            times_windows = torch.stack(times_list, dim=1).view(B * Fh, K_recent)
            lens_windows = torch.full((B * Fh,), K_recent, dtype=torch.long, device=all_emb.device)
            mask_windows = torch.ones(B * Fh, K_recent, N, dtype=torch.bool, device=all_emb.device)
            # Prefix EMA state before each window start if no latent pool
            if self.latent_pool is None:
                states_list = []
                for f in range(Fh):
                    s = base + f
                    if s - 1 >= 0:
                        states_list.append(ema_seq[:, s - 1, :, :])
                    else:
                        states_list.append(all_emb.new_zeros(B, N, all_emb.size(-1)))
                states_windows = torch.stack(states_list, dim=1).view(B * Fh, N, all_emb.size(-1))
            else:
                states_windows = None
            # Optional global latent pool over older prefix frames
            if self.latent_pool is not None:
                # For each horizon f: window start s = base + f; older prefix is [:s)
                z_list = []
                for f in range(Fh):
                    s = base + f
                    L_cap = self.latent_max_prefix if (self.latent_max_prefix is not None and self.latent_max_prefix > 0) else s
                    ctx_start = max(0, s - int(L_cap))
                    seg_len = max(0, s - ctx_start)
                    if seg_len <= 0:
                        zeros_feat = all_emb.new_zeros(B, N, all_emb.size(-1))
                        z = self.latent_pool(None, zeros_feat)
                    else:
                        older = all_emb[:, ctx_start:s, :, :].reshape(B, seg_len * N, all_emb.size(-1))
                        z = self.latent_pool(None, older)
                    z_list.append(z)
                z_windows = torch.stack(z_list, dim=1)  # (B, Fh, Lz, D)
                # Prepare extra_kv per window batch item
                extra_kv = z_windows.view(B * Fh, -1, all_emb.size(-1))  # (B*Fh, Lz, D)
                extra_kv_mask = torch.ones(B * Fh, extra_kv.size(1), dtype=torch.bool, device=mask_windows.device)
                # Set extra kv time equal to the last frame time in each window
                last_t = times_windows[:, -1]
                extra_kv_time = last_t  # (B*Fh,)
            # Encode all windows in one pass
            # Encode all windows in one pass
            _, h_last, new_state_unused = self.backbone.encode(
                emb_windows,
                times_windows,
                lens_windows,
                mask_windows,
                t_scalar,
                change_mask=None,
                run_length=None,
                delta_t=None,
                state=states_windows,
                extra_kv=(extra_kv if (self.latent_pool is not None) else None),
                extra_kv_mask=(extra_kv_mask if (self.latent_pool is not None) else None),
                extra_kv_time=(extra_kv_time if (self.latent_pool is not None) else None),
            )
            logits = self.classification_head(h_last).view(B, Fh, N, -1)
            output = h_last.view(B, Fh, N, -1)[:, -1, :, :]  # last horizon context (arbitrary for return)
            new_state = None
        elif torch.is_tensor(teacher_future_ids):
            B, T, N = input_cluster_ids.shape
            Fh = int(min(self.future_horizon, teacher_future_ids.shape[1]))
            # Build base dt per sample
            if times is None:
                times = torch.zeros(B, T, dtype=torch.float32, device=embeddings.device)
            # Determine K_recent and base offset
            K_recent = int(self.recent_full_frames) if (self.recent_full_frames is not None and self.recent_full_frames > 0) else T
            K_recent = int(max(1, min(K_recent, T)))
            base = int(T - K_recent)
            base_times = times[:, max(0, T - K_recent):]
            if K_recent > 1:
                dt = (base_times[:, 1] - base_times[:, 0]).to(dtype=torch.float32)
            else:
                dt = torch.ones(B, dtype=torch.float32, device=embeddings.device)
            # Working buffers
            buffer_ids = valid_cluster_ids.clone()
            work_state = None if (self.latent_pool is not None) else state
            # Initialize latent summary Z from older-than-K prefix of initial history
            if self.latent_pool is not None and base > 0:
                older_init = self.cluster_embedding(valid_cluster_ids[:, :base, :]).reshape(B, base * N, -1)
                Z = self.latent_pool(None, older_init)
            else:
                Z = None
            step_logits = []
            for f in range(Fh):
                step_ids = buffer_ids[:, -K_recent:, :]
                # Shift times by f*dt for temporal encoding alignment (use last K_recent history times)
                step_times = base_times + dt.view(B, 1) * float(f)
                # Build encode inputs for the K recent frames
                Xk = self.cluster_embedding(step_ids)  # (B, K_recent, N, D)
                enc_inp = Xk
                enc_times = step_times
                if history_mask is not None:
                    enc_mask = history_mask[:, -K_recent:, :]
                else:
                    enc_mask = torch.ones(B, K_recent, N, dtype=torch.bool, device=Xk.device)
                enc_lens = torch.full((B,), K_recent, dtype=torch.long, device=Xk.device)
                # Prepare extra_kv from Z if present
                if Z is not None and Z.numel() > 0:
                    extra_kv = Z  # (B, Lz, D)
                    extra_kv_mask = torch.ones(B, extra_kv.size(1), dtype=torch.bool, device=Xk.device)
                    extra_kv_time = enc_times[:, -1]
                else:
                    extra_kv = None
                    extra_kv_mask = None
                    extra_kv_time = None
                # Use recurrent state only when no latent summarizer is active
                state_arg = None if (self.latent_pool is not None) else work_state
                _, h_last, state_arg = self.backbone.encode(
                    enc_inp,
                    enc_times,
                    enc_lens,
                    enc_mask,
                    t_scalar,
                    change_mask=None,
                    run_length=None,
                    delta_t=None,
                    state=state_arg,
                    extra_kv=extra_kv,
                    extra_kv_mask=extra_kv_mask,
                    extra_kv_time=extra_kv_time,
                )
                # Update recurrent work_state when applicable
                if self.latent_pool is None:
                    work_state = state_arg
                logits_f = self.classification_head(h_last)  # (B, N, C)
                step_logits.append(logits_f.unsqueeze(1))
                # Decide next frame tokens (teacher vs predicted) for SS
                teacher_next = _safe_clamp(teacher_future_ids[:, f, :])
                pred_next = logits_f.argmax(dim=-1)
                if scheduled_sampling_p > 0.0:
                    prob = float(max(0.0, min(1.0, scheduled_sampling_p)))
                    mask = (torch.rand(B, N, device=embeddings.device) < prob)
                    next_ids = torch.where(mask, pred_next, teacher_next)
                else:
                    next_ids = teacher_next
                buffer_ids = torch.cat([buffer_ids, next_ids.unsqueeze(1)], dim=1)
                # Update latent summary Z with the frame that leaves the K window
                if self.latent_pool is not None:
                    leaving = step_ids[:, 0, :]
                    X_leave = self.cluster_embedding(leaving.clamp_min(0))
                    Z = self.latent_pool(Z, X_leave)
            logits = torch.cat(step_logits, dim=1)
            output = h_last
            new_state = work_state
        else:
            # Autoregressive rollout (greedy) when no teacher futures are provided
            B, T, N = input_cluster_ids.shape
            device = input_cluster_ids.device
            K_recent = int(self.recent_full_frames) if (self.recent_full_frames is not None and self.recent_full_frames > 0) else T
            K_recent = int(max(1, min(K_recent, T)))

            # Rolling buffers
            buffer_ids = input_cluster_ids.clone()
            buffer_mask = effective_mask.clone()
            if times is None:
                times = torch.zeros(B, T, dtype=torch.float32, device=device)
            buffer_times = times.clone()
            if T >= 2:
                dt = (buffer_times[:, -1] - buffer_times[:, -2]).abs().clamp_min(1e-6)
            else:
                dt = torch.ones(B, dtype=torch.float32, device=device)

            work_state = None if (self.latent_pool is not None) else state
            Z = None
            base = max(0, T - K_recent)
            if self.latent_pool is not None and base > 0:
                older_init = self.cluster_embedding(buffer_ids[:, :base, :].clamp_min(0)).reshape(B, base * N, -1)
                Z = self.latent_pool(None, older_init)

            logits_steps = []
            for f in range(self.future_horizon):
                step_ids = buffer_ids[:, -K_recent:, :]
                step_mask = buffer_mask[:, -K_recent:, :]
                step_times = buffer_times[:, -K_recent:]
                
                # Correctly set lens to full window to avoid alignment issues with sparse masks in TemporalEncoder
                step_lens = torch.full((B,), K_recent, dtype=torch.long, device=device)
                
                step_emb = self.cluster_embedding(step_ids)

                extra_kv = None
                extra_kv_mask = None
                extra_kv_time = None
                encode_state = None if (self.latent_pool is not None) else work_state
                if Z is not None and Z.numel() > 0:
                    extra_kv = Z
                    extra_kv_mask = torch.ones(B, Z.size(1), dtype=torch.bool, device=device)
                    extra_kv_time = step_times[:, -1]

                _, h_step, work_state = self.backbone.encode(
                    step_emb,
                    step_times,
                    step_lens,
                    step_mask,
                    t_scalar,
                    change_mask=None,
                    run_length=None,
                    delta_t=None,
                    state=encode_state,
                    extra_kv=extra_kv,
                    extra_kv_mask=extra_kv_mask,
                    extra_kv_time=extra_kv_time,
                )
                logits_f = self.classification_head(h_step)  # (B, N, C)
                logits_steps.append(logits_f.unsqueeze(1))

                # Greedy decode next ids
                next_ids = logits_f.argmax(dim=-1)  # (B, N) safe by construction
                buffer_ids = torch.cat([buffer_ids, next_ids.unsqueeze(1)], dim=1)

                # Append time and mask for new frame
                next_time = (buffer_times[:, -1] + dt).unsqueeze(1)
                buffer_times = torch.cat([buffer_times, next_time], dim=1)
                next_mask = torch.ones(B, 1, N, dtype=torch.bool, device=device)
                buffer_mask = torch.cat([buffer_mask, next_mask], dim=1)

                # Update latent summary with frame leaving the window
                if self.latent_pool is not None:
                    leaving = step_ids[:, 0, :]
                    X_leave = self.cluster_embedding(leaving.clamp_min(0))
                    Z = self.latent_pool(Z, X_leave)

            logits = torch.cat(logits_steps, dim=1)
            output = h_step
            new_state = work_state
        out = {
            'cluster_logits': logits,
            'context': output,
            'align_loss': align_loss,
        }
        if new_state is not None:
            out['new_state'] = new_state
        if hasattr(self.backbone, 'aux_losses') and isinstance(self.backbone.aux_losses, dict):
            out['aux_losses'] = self.backbone.aux_losses
        return out

    def infer_autoregressive_step(
        self,
        history_cluster_ids: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        change_mask: Optional[torch.Tensor] = None,
        run_length: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
        t_scalar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced inference helper that tolerates short histories."""

        if sequence_lengths is None:
            B = history_cluster_ids.shape[0]
            T = history_cluster_ids.shape[1]
            device = history_cluster_ids.device
            sequence_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        return self.forward(
            input_cluster_ids=history_cluster_ids,
            times=times if times is not None else None,
            sequence_lengths=sequence_lengths,
            history_mask=history_mask,
            change_mask=change_mask,
            run_length=run_length,
            delta_t=delta_t,
            t_scalar=t_scalar,
        )
