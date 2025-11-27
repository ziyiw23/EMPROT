#!/usr/bin/env python3
from typing import Dict, Optional, Tuple
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional
try:
    from transformers import get_cosine_schedule_with_warmup
    HAS_TRANSFORMERS = True
except Exception:
    get_cosine_schedule_with_warmup = None  # type: ignore
    HAS_TRANSFORMERS = False

from emprot.models.transformer import ProteinTransformerClassificationOnly
from emprot.losses import (
    masked_cross_entropy,
    histogram_ce_loss,
    aggregated_probability_kl_loss,
    st_gumbel_hist_kl_loss,
    transition_row_js_loss_from_logits,
    coverage_hinge_loss,
    per_residue_histogram_from_ids,
    straight_through_gumbel_softmax,
    residue_centric_loss,
)
from emprot.losses.distributional import _st_histogram_partial_teacher
from emprot.utils.metrics import (
    compute_classification_metrics,
    compute_histogram_metrics,
    compute_brier_score,
    compute_aggregated_kl_metric,
)
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    wandb = None  # type: ignore
    _HAS_WANDB = False


DEFAULT_NUM_CLASSES = 50000


class EMPROTTrainer:
    def __init__(self, config: Dict):
        self.cfg = dict(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ProteinTransformerClassificationOnly(
            d_embed=int(self.cfg['d_embed']),
            num_heads=int(self.cfg['num_heads']),
            num_layers=int(self.cfg.get('num_layers', 1)),
            num_clusters=int(self.cfg.get('num_clusters', DEFAULT_NUM_CLASSES)),
            future_horizon=int(self.cfg.get('future_horizon', 1)),
            recent_full_frames=self.cfg.get('num_full_res_frames', None),
            dropout=float(self.cfg.get('dropout', 0.1)),
            use_gradient_checkpointing=bool(self.cfg.get('use_gradient_checkpointing', False)),
            latent_summary_enabled=bool(self.cfg.get('latent_summary_enabled', False)),
            latent_summary_num_latents=int(self.cfg.get('latent_summary_num_latents', 0)),
            latent_summary_heads=self.cfg.get('latent_summary_heads', None),
            latent_summary_dropout=self.cfg.get('latent_summary_dropout', None),
            latent_summary_max_prefix=self.cfg.get('latent_summary_max_prefix', None),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg['learning_rate']),
            weight_decay=float(self.cfg.get('weight_decay', 0.01)),
            betas=tuple(self.cfg.get('betas', (0.9, 0.999))),
            eps=1e-8,
        )

        self.scheduler = None
        if bool(self.cfg.get('use_scheduler', False)) and HAS_TRANSFORMERS:
            steps_per_epoch = int(self.cfg.get('estimated_steps_per_epoch', 0)) or 1000
            total_steps = steps_per_epoch * int(self.cfg.get('max_epochs', 40))
            
            # Support explicit warmup steps if provided, else derived from proportion
            warmup_steps_cfg = self.cfg.get('warmup_steps', None)
            if warmup_steps_cfg is not None:
                warmup = int(warmup_steps_cfg)
            else:
                warmup = int(total_steps * float(self.cfg.get('warmup_proportion', 0.1)))

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
            )

        self.use_amp = bool(self.cfg.get('use_amp', True)) and torch.cuda.is_available()
        self.amp_dtype = (
            torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_val_loss = float('inf')
        self.patience = int(self.cfg.get('patience', 10))
        self.min_delta = float(self.cfg.get('early_stopping_min_delta', 0.0))
        self._no_improve = 0
        self.epoch = 0
        self.global_step = 0
        # Logging
        self.log_interval = int(self.cfg.get('log_interval', 1000))
        # Mid-epoch eval
        self.eval_every_n_steps = int(self.cfg.get('eval_every_n_steps', 0) or 0)
        self.max_val_batches = int(self.cfg.get('max_val_batches', 0) or 0)

        # W&B setup (guarded): initialize if requested and not already initialized
        self.use_wandb = bool(self.cfg.get('use_wandb', False)) and _HAS_WANDB
        if self.use_wandb and getattr(wandb, 'run', None) is None:  # type: ignore
            try:
                wandb.init(  # type: ignore
                    project=self.cfg.get('wandb_project', 'emprot'),
                    entity=self.cfg.get('entity', None),
                    name=self.cfg.get('run_name', None),
                    tags=self.cfg.get('tags', None),
                    notes=self.cfg.get('notes', None),
                    config=self.cfg,
                )
            except Exception:
                # If initialization fails, disable wandb to avoid crashes
                self.use_wandb = False

    def train(self, train_loader, val_loader) -> float:
        max_epochs = int(self.cfg.get('max_epochs', 40))
        for self.epoch in range(max_epochs):
            # Online sampler: rebuild epoch indices and log realized mix if provided
            ds = getattr(train_loader, 'dataset', None)
            if hasattr(ds, 'on_epoch_start'):
                try:
                    ds.on_epoch_start(seed=int(self.epoch))
                    cf = getattr(ds, 'last_change_fraction_realized', None)
                    ms = getattr(ds, 'last_mean_window_score', None)
                    if self.use_wandb and getattr(wandb, 'run', None) is not None:
                        payload = {}
                        if cf is not None:
                            payload['data/change_fraction_realized'] = float(cf)
                        if ms is not None:
                            payload['data/mean_window_score'] = float(ms)
                        if payload:
                            wandb.log(payload, step=int(self.global_step))  # type: ignore
                except Exception:
                    pass
            train_loss = self.train_epoch(train_loader)
            val_report = self.validate(val_loader)
            val_loss = float(val_report.get('val/loss', val_report.get('loss', float('inf'))))

            # Always save an epoch-specific checkpoint
            self._save_checkpoint(f'epoch_{int(self.epoch)}.pt')

            payload = {'epoch': int(self.epoch), 'train/loss': float(train_loss), 'val/loss': float(val_loss)}
            # Merge validation metrics (prefixed val/ if not already)
            for k, v in val_report.items():
                if k in {'val/loss', 'loss'}:
                    continue
                key = k if k.startswith('val/') else f"val/{k}"
                payload[key] = float(v) if isinstance(v, (int, float)) else (float(v.item()) if hasattr(v, 'item') else None)
            # Drop Nones
            payload = {k: v for k, v in payload.items() if v is not None}
            if self.use_wandb and getattr(wandb, 'run', None) is not None:  # type: ignore
                try:
                    wandb.log(payload, step=int(self.global_step))  # type: ignore
                except Exception:
                    pass

            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self._no_improve = 0
                self._save_checkpoint('best.pt')
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    break
        self._save_checkpoint('final.pt')
        return self.best_val_loss

    def train_epoch(self, loader, val_loader=None) -> float:
        self.model.train()
        accum = max(1, int(self.cfg.get('grad_accum_steps', 1)))
        max_grad_norm = float(self.cfg.get('max_grad_norm', 0.5))
        running = 0.0
        count = 0
        import time as _time
        t0 = _time.time()

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            loss, train_metrics = self._compute_loss(batch, training=True, return_metrics=True)
            loss_to_backprop = loss / accum
            if self.use_amp:
                self.scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if ((step + 1) % accum) == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                curr_lr = float(self.optimizer.param_groups[0]['lr'])
                if self.use_wandb and getattr(wandb, 'run', None) is not None:  # type: ignore
                    try:
                        wandb.log({'train/lr': curr_lr}, step=int(self.global_step))  # type: ignore
                    except Exception:
                        pass
                self.optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach().item())
            count += 1
            self.global_step += 1

            # Periodic progress logging (no tqdm to keep logs clean on Slurm)
            if self.log_interval > 0 and ((step + 1) % self.log_interval == 0 or (step + 1) == len(loader)):
                elapsed = _time.time() - t0
                avg_loss = running / max(count, 1)
                steps_done = step + 1
                total_steps = len(loader)
                spb = elapsed / max(count, 1)
                eta = (total_steps - steps_done) * spb
                msg = (
                    f"[train] epoch={self.epoch} step={steps_done}/{total_steps} "
                    f"avg_loss={avg_loss:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                )
                print(msg, flush=True)
                if self.use_wandb and getattr(wandb, 'run', None) is not None:  # type: ignore
                    try:
                        payload = {'train/avg_loss': avg_loss, 'train/steps_done': steps_done, 'train/eta_min': eta/60.0}
                        # Include current-batch training metrics if available
                        if isinstance(train_metrics, dict):
                            for k, v in train_metrics.items():
                                if isinstance(v, (float, int)):
                                    payload[f'train/{k}'] = float(v)
                        wandb.log(payload, step=int(self.global_step))  # type: ignore
                    except Exception:
                        pass

            # Mid-epoch validation
            if val_loader is not None and self.eval_every_n_steps > 0 and (self.global_step % self.eval_every_n_steps == 0):
                report = self.validate(val_loader, max_batches=(self.max_val_batches or None))
                if self.use_wandb and getattr(wandb, 'run', None) is not None:  # type: ignore
                    try:
                        payload = {f'mid/{k}': (float(v) if isinstance(v, (int, float)) else v) for k, v in report.items()}
                        wandb.log(payload, step=int(self.global_step))  # type: ignore
                    except Exception:
                        pass
        return running / max(count, 1)

    @torch.no_grad()
    def validate(self, loader, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.eval()
        total = 0.0
        count = 0
        objective = str(self.cfg.get('objective', 'token_ce')).lower()
        st_metric_keys = {'dwell/loss', 'change_event/loss', 'change_event/acc', 'mean_dwell_gt', 'mean_dwell_pred'}
        st_metric_totals = {k: 0.0 for k in st_metric_keys}
        st_metric_count = 0
        # aggregate simple means across batches
        agg = {
            'acc_f1': 0.0,
            'top5_f1': 0.0,
            'mtp_f1': 0.0,
            'entropy_f1': 0.0,
            'ece_f1': 0.0,
            'acc_change_f1': 0.0,
            'acc_stay_f1': 0.0,
        }
        mcount = 0
        acc_h_sum = None  # sum of per-horizon accuracies across batches
        last_hmetrics = None
        extra_sum = {'dist_kl_agg': 0.0, 'dist_kl_st': 0.0, 'brier': 0.0}
        extra_count = {'dist_kl_agg': 0, 'dist_kl_st': 0, 'brier': 0}
        dist_dbg_last: Dict[str, float] = {}
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            batch = self._to_device(batch)
            loss_out = self._compute_loss(batch, training=False, return_metrics=True)
            if isinstance(loss_out, tuple):
                loss, extra_metrics = loss_out
            else:
                loss = loss_out
                extra_metrics = None
            total += float(loss.detach().item())
            count += 1
            if extra_metrics and objective == 'st_gumbel_hist':
                st_metric_count += 1
                for key in st_metric_keys:
                    val = extra_metrics.get(key)
                    if isinstance(val, (int, float)):
                        st_metric_totals[key] += float(val)
            try:
                outputs = self.model(
                    input_cluster_ids=batch['input_cluster_ids'],
                    times=batch.get('times'),
                    sequence_lengths=batch.get('sequence_lengths'),
                    history_mask=batch.get('history_mask'),
                    teacher_future_ids=batch.get('future_cluster_ids'),
                    scheduled_sampling_p=0.0,
                )
                metrics = compute_classification_metrics(
                    outputs['cluster_logits'],
                    batch['future_cluster_ids'],
                    input_cluster_ids=batch.get('input_cluster_ids'),
                    future_step_mask=batch.get('future_step_mask'),
                    residue_mask=batch.get('residue_mask'),
                    compute_ece=True,
                )
                # Distributional metrics (order-free)
                last_hmetrics = compute_histogram_metrics(
                    outputs['cluster_logits'],
                    batch['future_cluster_ids'],
                    future_step_mask=batch.get('future_step_mask'),
                    residue_mask=batch.get('residue_mask'),
                    input_cluster_ids=batch.get('input_cluster_ids'),
                )
                try:
                    dist_kl_val = compute_aggregated_kl_metric(
                        outputs['cluster_logits'],
                        batch['future_cluster_ids'],
                        future_step_mask=batch.get('future_step_mask'),
                        residue_mask=batch.get('residue_mask'),
                        label_smoothing=float(self.cfg.get('dist_kl_label_smoothing', 0.0) or 0.0),
                        eps=1e-8,
                    )
                    extra_sum['dist_kl_agg'] += float(dist_kl_val)
                    extra_count['dist_kl_agg'] += 1
                except Exception:
                    pass
                try:
                    tau_eval = float(self.cfg.get('gumbel_tau_end', 0.7))
                    st_loss, st_dbg = st_gumbel_hist_kl_loss(
                        self.model,
                        batch,
                        tau=tau_eval,
                        M=2,
                        eps=1e-8,
                        label_smoothing=float(self.cfg.get('dist_kl_label_smoothing', 0.0) or 0.0),
                        logits=outputs['cluster_logits'],
                        future_step_mask=batch.get('future_step_mask'),
                        residue_mask=batch.get('residue_mask'),
                        partial_tf=True,
                        use_scheduled_sampling=False,
                        scheduled_sampling_p=0.0,
                    )
                    extra_sum['dist_kl_st'] += float(st_loss.detach().item())
                    extra_count['dist_kl_st'] += 1
                    if isinstance(st_dbg, dict):
                        dist_dbg_last = {k: float(v) for k, v in st_dbg.items() if isinstance(v, (int, float))}
                except Exception:
                    pass
                try:
                    brier_val = compute_brier_score(
                        outputs['cluster_logits'],
                        batch['future_cluster_ids'],
                        future_step_mask=batch.get('future_step_mask'),
                        residue_mask=batch.get('residue_mask'),
                    )
                    extra_sum['brier'] += float(brier_val)
                    extra_count['brier'] += 1
                except Exception:
                    pass
                # average floats only; skip tensors here (acc_per_horizon)
                for key in agg.keys():
                    val = metrics.get(key, None)
                    if isinstance(val, (int, float)):
                        agg[key] += float(val)
                # Accumulate per-horizon accuracy to derive interpretable means
                aph = metrics.get('acc_per_horizon', None)
                if torch.is_tensor(aph):
                    aph = aph.to(dtype=torch.float32).cpu()
                    if acc_h_sum is None:
                        acc_h_sum = torch.zeros_like(aph)
                    # handle possible horizon length mismatch defensively
                    if acc_h_sum.numel() == aph.numel():
                        acc_h_sum += aph
                mcount += 1
            except Exception:
                pass
        val_loss = total / max(count, 1)
        report: Dict[str, float] = {'val/loss': float(val_loss)}
        if mcount > 0:
            for key, val in agg.items():
                report[key] = float(val / mcount)
            # Add interpretable multi-horizon summaries
            if acc_h_sum is not None and acc_h_sum.numel() > 0:
                acc_mean = float((acc_h_sum / mcount).mean().item())
                acc_last = float((acc_h_sum / mcount)[-1].item())
                report['acc_mean'] = acc_mean
                report['acc_last'] = acc_last
                # Optional: weighted mean using horizon_weights if present in cfg
                hw = self.cfg.get('horizon_weights', None)
                if isinstance(hw, (list, tuple)) and len(hw) == acc_h_sum.numel():
                    import numpy as _np  # local, avoid global dep
                    w = torch.as_tensor(hw, dtype=torch.float32)
                    w = w / (w.sum().clamp_min(1e-6))
                    report['acc_weighted'] = float(((acc_h_sum / mcount) * w).sum().item())
            # Attach histogram metrics from the last evaluated batch as a proxy
            if isinstance(last_hmetrics, dict):
                for k, v in last_hmetrics.items():
                    if isinstance(v, (int, float)):
                        report[k] = float(v)
            for key, total in extra_sum.items():
                if extra_count[key] > 0:
                    report[key] = float(total / extra_count[key])
            for k, v in dist_dbg_last.items():
                report[f'dist_dbg/{k}'] = float(v)
        if objective == 'st_gumbel_hist' and st_metric_count > 0:
            for key, total_val in st_metric_totals.items():
                report[key] = float(total_val / st_metric_count)
        return report

    def load_checkpoint(self, path: str) -> bool:
        try:
            state = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state['model'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])
            if self.scheduler is not None and state.get('scheduler') is not None:
                self.scheduler.load_state_dict(state['scheduler'])
            self.epoch = int(state.get('epoch', 0))
            self.best_val_loss = float(state.get('best_val_loss', self.best_val_loss))
            return True
        except Exception:
            return False

    def _compute_loss(self, batch: Dict, training: bool, return_metrics: bool = False):
        objective = str(self.cfg.get('objective', 'token_ce')).lower()
        use_f_horizon_ce = bool(self.cfg.get('use_f_horizon_ce', False))
        ce_w = float(self.cfg.get('token_ce_weight', 1.0) or 0.0)
        ss_cfg = float(self.cfg.get('scheduled_sampling_p', 0.0))
        st_use_ss = bool(self.cfg.get('st_use_scheduled_sampling', False))
        ss_p = ss_cfg if training else 0.0
        forward_ss_p = ss_p
        if objective == 'st_gumbel_hist' and not st_use_ss:
            forward_ss_p = 0.0
        outputs = self.model(
            input_cluster_ids=batch['input_cluster_ids'],
            times=batch.get('times'),
            sequence_lengths=batch.get('sequence_lengths'),
            history_mask=batch.get('history_mask'),
            teacher_future_ids=batch.get('future_cluster_ids'),
            scheduled_sampling_p=forward_ss_p,
        )
        logits = outputs['cluster_logits']
        targets = batch['future_cluster_ids']
        step_mask = batch.get('future_step_mask')
        res_mask = batch.get('residue_mask')

        label_smoothing = float(self.cfg.get('label_smoothing', 0.0) or 0.0)
        change_upweight = float(self.cfg.get('change_upweight', 1.0) or 1.0)
        class_weights = self.cfg.get('class_weights', None)
        horizon_weights = self.cfg.get('horizon_weights', None)

        aux_hist_val: Optional[float] = None
        ent_val: Optional[float] = None
        st_debug: Dict[str, float] = {}
        res_dbg: Dict[str, float] = {}
        entropy_bits_mean: Optional[float] = None
        entropy_floor_penalty: Optional[float] = None

        if objective == 'token_ce':
            if logits.dim() != 4 or targets.dim() != 3:
                ce_logits = logits
                ce_targets = targets
                ce_step_mask = step_mask
                ce_horizon_weights = horizon_weights
            elif use_f_horizon_ce:
                ce_logits = logits
                ce_targets = targets
                ce_step_mask = step_mask
                ce_horizon_weights = horizon_weights
            else:
                ce_logits = logits[:, 0:1, :, :]
                ce_targets = targets[:, 0:1, :]
                ce_step_mask = step_mask[:, 0].unsqueeze(1) if (step_mask is not None and step_mask.dim() == 2) else None
                ce_horizon_weights = None
            ce_kwargs = {
                'future_step_mask': ce_step_mask,
                'residue_mask': res_mask,
                'label_smoothing': label_smoothing,
                'horizon_weights': ce_horizon_weights,
                'class_weights': class_weights,
                'input_cluster_ids': batch.get('input_cluster_ids'),
                'change_upweight': change_upweight,
            }
            if self.use_amp and training:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    ce_loss = masked_cross_entropy(ce_logits, ce_targets, **ce_kwargs)
            else:
                ce_loss = masked_cross_entropy(ce_logits, ce_targets, **ce_kwargs)
            loss_core = ce_loss * ce_w
            aux_w = float(self.cfg.get('aux_hist_ce_weight', 0.0) or 0.0)
            if aux_w != 0.0:
                use_rand = bool(self.cfg.get('histogram_random_horizon', False))
                if use_rand and logits.size(1) > 1:
                    f_sel = int(torch.randint(low=0, high=int(logits.size(1)), size=(1,), device=logits.device).item())
                    logits_hist = logits[:, f_sel:f_sel + 1, :, :]
                    targets_hist = targets[:, f_sel:f_sel + 1, :]
                    step_mask_hist = step_mask[:, f_sel].unsqueeze(1) if (step_mask is not None and step_mask.dim() == 2) else None
                else:
                    logits_hist = ce_logits
                    targets_hist = ce_targets
                    step_mask_hist = ce_step_mask
                if self.use_amp and training:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        h_ce = histogram_ce_loss(
                            logits_hist,
                            targets_hist,
                            future_step_mask=step_mask_hist,
                            residue_mask=res_mask,
                        )
                else:
                    h_ce = histogram_ce_loss(
                        logits_hist,
                        targets_hist,
                        future_step_mask=step_mask_hist,
                        residue_mask=res_mask,
                    )
                loss_core = loss_core + aux_w * h_ce
                try:
                    aux_hist_val = float(h_ce.detach().item())
                except Exception:
                    aux_hist_val = None
            ent_w = float(self.cfg.get('entropy_bonus_weight', 0.0) or 0.0)
            if ent_w != 0.0:
                probs_ent = torch.softmax(logits, dim=-1)
                if step_mask is None:
                    fmask = torch.ones(logits.size(0), logits.size(1), dtype=torch.bool, device=logits.device)
                else:
                    fmask = step_mask.to(dtype=torch.bool, device=logits.device)
                if res_mask is None:
                    rmask = torch.ones(logits.size(0), logits.size(2), dtype=torch.bool, device=logits.device)
                else:
                    rmask = res_mask.to(dtype=torch.bool, device=logits.device)
                valid = (targets >= 0) & fmask[:, :, None] & rmask[:, None, :]
                if valid.any():
                    p_sel = probs_ent[valid]
                    entropy = -(p_sel * p_sel.clamp_min(1e-12).log()).sum(dim=-1).mean()
                    loss_core = loss_core - float(ent_w) * entropy
                    try:
                        ent_val = float(entropy.detach().item())
                    except Exception:
                        ent_val = None
        elif objective == 'agg_kl':
            loss_core = aggregated_probability_kl_loss(
                logits,
                targets,
                future_step_mask=step_mask,
                residue_mask=res_mask,
                label_smoothing=float(self.cfg.get('dist_kl_label_smoothing', 0.0) or 0.0),
                eps=float(self.cfg.get('agg_kl_eps', 1e-8) or 1e-8),
                reduce=str(self.cfg.get('agg_kl_reduce', 'mean')),
            )
        elif objective == 'residue_centric':
            num_samples = int(self.cfg.get('res_num_samples', 32))
            res_ce_w = float(self.cfg.get('res_ce_weight', 1.0))
            res_js_w = float(self.cfg.get('res_js_weight', 1.0))
            eps_val = float(self.cfg.get('agg_kl_eps', 1e-8) or 1e-8)
            loss_core, res_dbg = residue_centric_loss(
                logits,
                targets,
                future_step_mask=step_mask,
                residue_mask=res_mask,
                num_samples=num_samples,
                ce_weight=res_ce_w,
                js_weight=res_js_w,
                eps=eps_val,
                label_smoothing=label_smoothing,
            )
        elif objective == 'st_gumbel_hist':
            tau = self._anneal_tau(
                step=int(self.global_step),
                t0=float(self.cfg.get('gumbel_tau_start', 1.5)),
                t1=float(self.cfg.get('gumbel_tau_end', 0.7)),
                T=int(self.cfg.get('gumbel_tau_steps', 20000)),
            )
            partial_tf = bool(self.cfg.get('st_rollout_partial_tf', True))
            M = int(self.cfg.get('gumbel_samples', 3) or 3)
            dist_ls = float(self.cfg.get('dist_kl_label_smoothing', 0.0) or 0.0)
            st_loss, st_debug = st_gumbel_hist_kl_loss(
                self.model,
                batch,
                tau=tau,
                M=M,
                eps=1e-8,
                label_smoothing=dist_ls,
                logits=(logits if partial_tf else None),
                future_step_mask=step_mask,
                residue_mask=res_mask,
                partial_tf=partial_tf,
                use_scheduled_sampling=(st_use_ss and training),
                scheduled_sampling_p=(ss_p if (st_use_ss and training) else 0.0),
            )
            lam_dwell = float(self.cfg.get('lambda_dwell', 0.2))
            lam_trans = float(self.cfg.get('lambda_trans', 0.2))
            lam_cov = float(self.cfg.get('lambda_cov', 0.05))
            lam_change = float(self.cfg.get('lambda_change', 0.3))
            trans_min_count = int(self.cfg.get('trans_row_min_count', 5))
            cov_thr = float(self.cfg.get('coverage_threshold', 1e-4))

            probs_full = torch.softmax(logits, dim=-1)
            gumbel_sample = straight_through_gumbel_softmax(logits, tau=tau)
            pred_ids = torch.argmax(gumbel_sample, dim=-1)
            step_mask_bool = None
            if step_mask is not None:
                step_mask_bool = step_mask.to(dtype=torch.bool, device=logits.device)
                pred_ids = pred_ids.masked_fill(~step_mask_bool[:, :, None], -1)
            res_mask_bool = None
            if res_mask is not None:
                res_mask_bool = res_mask.to(dtype=torch.bool, device=logits.device)
                pred_ids = pred_ids.masked_fill(~res_mask_bool[:, None, :], -1)

            mean_dwell_gt, dwell_valid_gt = self._compute_mean_dwell_from_ids(
                targets, future_step_mask=step_mask_bool, residue_mask=res_mask_bool
            )
            mean_dwell_pred, dwell_valid_pred = self._compute_mean_dwell_from_ids(
                pred_ids, future_step_mask=step_mask_bool, residue_mask=res_mask_bool
            )
            dwell_mask = dwell_valid_gt & dwell_valid_pred
            if dwell_mask.any():
                dwell_loss = F.l1_loss(
                    mean_dwell_pred[dwell_mask], mean_dwell_gt[dwell_mask]
                )
                mean_dwell_gt_mean = mean_dwell_gt[dwell_mask].mean()
                mean_dwell_pred_mean = mean_dwell_pred[dwell_mask].mean()
            else:
                dwell_loss = logits.new_tensor(0.0)
                mean_dwell_gt_mean = logits.new_tensor(0.0)
                mean_dwell_pred_mean = logits.new_tensor(0.0)
            mean_dwell_gt_scalar = float(mean_dwell_gt_mean.detach().item())
            mean_dwell_pred_scalar = float(mean_dwell_pred_mean.detach().item())
            dwell_dbg = {
                'mean_gt': mean_dwell_gt_scalar,
                'mean_pred': mean_dwell_pred_scalar,
            }

            change_event_loss, change_event_acc = self._compute_change_event_terms(
                probs_full, targets, future_step_mask=step_mask_bool, residue_mask=res_mask_bool
            )
            row_js_loss, row_dbg = transition_row_js_loss_from_logits(
                logits=logits,
                future_ids=targets,
                min_count=trans_min_count,
                future_step_mask=step_mask,
                residue_mask=res_mask,
                eps=1e-8,
            )
            with torch.no_grad():
                q_hist = per_residue_histogram_from_ids(
                    targets,
                    logits.size(-1),
                    future_step_mask=step_mask,
                    residue_mask=res_mask,
                    label_smoothing=dist_ls,
                    eps=1e-8,
                )
            p_hat, cov_mask, _ = _st_histogram_partial_teacher(
                logits,
                targets,
                future_step_mask=step_mask,
                residue_mask=res_mask,
                tau=tau,
                M=M,
                eps=1e-8,
            )
            cov_loss, cov_dbg = coverage_hinge_loss(
                p_hat,
                q_hist,
                mask=cov_mask,
                eps=1e-8,
                thr=cov_thr,
            )
            loss_core = (
                st_loss
                + lam_dwell * dwell_loss
                + lam_trans * row_js_loss
                + lam_cov * cov_loss
                + lam_change * change_event_loss
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        loss = loss_core
        entropy_floor_bits = float(self.cfg.get('entropy_floor_bits', 0.0) or 0.0)
        if entropy_floor_bits > 0.0:
            probs_floor = torch.softmax(logits, dim=-1)
            if step_mask is None:
                fmask = torch.ones(logits.size(0), logits.size(1), dtype=torch.bool, device=logits.device)
            else:
                fmask = step_mask.to(dtype=torch.bool, device=logits.device)
            if res_mask is None:
                rmask = torch.ones(logits.size(0), logits.size(2), dtype=torch.bool, device=logits.device)
            else:
                rmask = res_mask.to(dtype=torch.bool, device=logits.device)
            valid = (targets >= 0) & fmask[:, :, None] & rmask[:, None, :]
            if valid.any():
                ent = -(probs_floor * probs_floor.clamp_min(1e-12).log()).sum(dim=-1) / math.log(2.0)
                ent_valid = ent[valid]
                entropy_bits_mean = float(ent_valid.mean().detach().item())
                penalty = (entropy_floor_bits - ent_valid).clamp_min(0.0).mean()
                loss = loss + penalty
                try:
                    entropy_floor_penalty = float(penalty.detach().item())
                except Exception:
                    entropy_floor_penalty = None
            else:
                entropy_bits_mean = 0.0

        if return_metrics:
            m = compute_classification_metrics(
                logits,
                targets,
                input_cluster_ids=batch.get('input_cluster_ids'),
                future_step_mask=step_mask,
                residue_mask=res_mask,
                compute_ece=False,
            )
            js_smoothed: Dict[str, float] = {}
            try:
                with torch.no_grad():
                    B, F, N, C = logits.shape
                    device = logits.device
                    ls = float(self.cfg.get('label_smoothing', 0.0) or 0.0)
                    P = torch.softmax(logits, dim=-1)
                    fmask = step_mask if step_mask is not None else torch.ones(B, F, dtype=torch.bool, device=device)
                    rmask = res_mask if res_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
                    valid = (targets >= 0) & fmask[:, :, None] & rmask[:, None, :]
                    eps = 1e-12
                    for f in range(F):
                        v = valid[:, f]
                        if not v.any():
                            continue
                        t = targets[:, f].clamp_min(0)
                        Q = torch.full((B, N, C), fill_value=ls / max(1, C - 1), dtype=P.dtype, device=device)
                        idx = t.unsqueeze(-1)
                        Q.scatter_(dim=-1, index=idx, src=torch.full_like(idx, fill_value=1.0 - ls, dtype=P.dtype))
                        P_sel = P[:, f][v]
                        Q_sel = Q[v]
                        M = 0.5 * (P_sel + Q_sel)
                        js = 0.5 * (P_sel * (P_sel.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(dim=-1)
                        js = js + 0.5 * (Q_sel * (Q_sel.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(dim=-1)
                        js_smoothed[f"js_smoothed_f{f}"] = float(js.mean().item())
                    if js_smoothed:
                        js_vals = list(js_smoothed.values())
                        js_smoothed['js_smoothed_mean'] = float(sum(js_vals) / max(1, len(js_vals)))
            except Exception:
                pass
            loss_share_change = None
            weight_share_change = None
            try:
                with torch.no_grad():
                    B, F, N, C = logits.shape
                    device = logits.device
                    fmask = step_mask if step_mask is not None else torch.ones(B, F, dtype=torch.bool, device=device)
                    rmask = res_mask if res_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
                    valid = (targets >= 0) & fmask[:, :, None] & rmask[:, None, :]
                    if self.cfg.get('horizon_weights', None) is not None:
                        hw = self.cfg.get('horizon_weights')
                        if isinstance(hw, (list, tuple)):
                            hw_t = torch.as_tensor(hw, dtype=logits.dtype, device=device)
                        else:
                            hw_t = torch.as_tensor(hw, dtype=logits.dtype, device=device)
                        if hw_t.dim() == 1 and hw_t.numel() == F:
                            token_weights = (valid.to(logits.dtype) * hw_t.view(1, F, 1))
                        else:
                            token_weights = valid.to(logits.dtype)
                    else:
                        token_weights = valid.to(logits.dtype)
                    change_up = float(self.cfg.get('change_upweight', 1.0) or 1.0)
                    change_mask = torch.zeros(B, F, N, dtype=torch.bool, device=device)
                    in_hist = batch.get('input_cluster_ids')
                    if in_hist is not None and torch.is_tensor(in_hist) and in_hist.dim() == 3:
                        last_hist = in_hist[:, -1, :].to(device=device)
                        f0 = targets[:, 0, :].to(device=device)
                        change_mask[:, 0, :] = (f0 != last_hist) & (f0 >= 0)
                    if F > 1:
                        t_prev = targets[:, :-1, :]
                        t_curr = targets[:, 1:, :]
                        vpair = (t_prev >= 0) & (t_curr >= 0)
                        change_mask[:, 1:, :] = (t_curr != t_prev) & vpair
                    if change_up > 1.0:
                        token_weights = torch.where(change_mask & valid, token_weights * change_up, token_weights)
                    logits_sel = logits[valid]
                    targets_sel = targets[valid].long()
                    cw = self.cfg.get('class_weights', None)
                    if cw is not None:
                        if isinstance(cw, (list, tuple)):
                            cw_t = torch.as_tensor(cw, dtype=logits.dtype, device=device)
                        else:
                            cw_t = torch.as_tensor(cw, dtype=logits.dtype, device=device)
                    else:
                        cw_t = None
                    ce_per = torch.nn.functional.cross_entropy(
                        logits_sel,
                        targets_sel,
                        reduction='none',
                        label_smoothing=label_smoothing,
                        weight=cw_t,
                    )
                    weights_sel = token_weights[valid]
                    change_sel = change_mask[valid]
                    total_loss_weighted = (ce_per * weights_sel).sum().clamp_min(1e-6)
                    change_loss_weighted = (ce_per[change_sel] * weights_sel[change_sel]).sum() if change_sel.any() else ce_per.new_zeros(())
                    loss_share_change = float((change_loss_weighted / total_loss_weighted).item())
                    total_weight = weights_sel.sum().clamp_min(1e-6)
                    change_weight = weights_sel[change_sel].sum() if change_sel.any() else weights_sel.new_zeros(())
                    weight_share_change = float((change_weight / total_weight).item())
            except Exception:
                pass
            acc_mean = float(m['acc_per_horizon'].mean().item()) if hasattr(m.get('acc_per_horizon'), 'mean') else None
            acc_last = float(m['acc_per_horizon'][-1].item()) if hasattr(m.get('acc_per_horizon'), '__getitem__') else None
            train_metrics = {
                'acc_f1': float(m.get('acc_f1', 0.0)),
                'top5_f1': float(m.get('top5_f1', 0.0)),
                'acc_change_f1': float(m.get('acc_change_f1', 0.0)),
                'acc_stay_f1': float(m.get('acc_stay_f1', 0.0)),
                'loss_core': float(loss_core.detach().item()),
            }
            if aux_hist_val is not None:
                train_metrics['aux_hist_ce'] = aux_hist_val
            if ent_val is not None:
                train_metrics['entropy'] = ent_val
            for k, v in js_smoothed.items():
                train_metrics[k] = float(v)
            try:
                hmetrics = compute_histogram_metrics(
                    logits,
                    targets,
                    future_step_mask=step_mask,
                    residue_mask=res_mask,
                    input_cluster_ids=batch.get('input_cluster_ids'),
                )
                if isinstance(hmetrics, dict):
                    val = hmetrics.get('js_hist', None)
                    if isinstance(val, (int, float)):
                        train_metrics['hist_js'] = float(val)
                    val = hmetrics.get('l1_hist', None)
                    if isinstance(val, (int, float)):
                        train_metrics['hist_l1'] = float(val)
                    val = hmetrics.get('pred_entropy', None)
                    if isinstance(val, (int, float)):
                        train_metrics['hist_pred_entropy'] = float(val)
            except Exception:
                pass
            if loss_share_change is not None:
                train_metrics['loss_share_change'] = float(loss_share_change)
            if weight_share_change is not None:
                train_metrics['weight_share_change'] = float(weight_share_change)
            if acc_mean is not None:
                train_metrics['acc_mean'] = acc_mean
            if acc_last is not None:
                train_metrics['acc_last'] = acc_last
            if entropy_bits_mean is not None:
                train_metrics['entropy_bits_mean'] = float(entropy_bits_mean)
            if entropy_floor_penalty is not None:
                train_metrics['entropy_floor_penalty'] = float(entropy_floor_penalty)
            if objective == 'agg_kl':
                train_metrics['dist_kl_agg'] = float(loss_core.detach().item())
            if objective == 'residue_centric':
                train_metrics['res_ce_mean'] = float(res_dbg.get('res_ce_mean', 0.0))
                train_metrics['res_js_mean'] = float(res_dbg.get('res_js_mean', 0.0))
                train_metrics['res_num_used'] = float(res_dbg.get('res_num_used', 0.0))
            if objective == 'st_gumbel_hist':
                train_metrics['dist_kl_st'] = float(st_loss.detach().item())
                train_metrics['dwell/loss'] = float(dwell_loss.detach().item())
                train_metrics['mean_dwell_gt'] = mean_dwell_gt_scalar
                train_metrics['mean_dwell_pred'] = mean_dwell_pred_scalar
                train_metrics['change_event/loss'] = float(change_event_loss.detach().item())
                train_metrics['change_event/acc'] = float(change_event_acc.detach().item())
                train_metrics['trans/loss'] = float(row_js_loss.detach().item())
                train_metrics['cov/loss'] = float(cov_loss.detach().item())
                for k, v in st_debug.items():
                    if isinstance(v, (int, float)):
                        train_metrics[f'st_dbg/{k}'] = float(v)
                for k, v in dwell_dbg.items():
                    train_metrics[f'dwell/{k}'] = float(v)
                for k, v in row_dbg.items():
                    train_metrics[f'trans/{k}'] = float(v)
                for k, v in cov_dbg.items():
                    train_metrics[f'cov/{k}'] = float(v)
            return loss, train_metrics
        return loss

    def _anneal_tau(self, step: int, t0: float, t1: float, T: int) -> float:
        if step >= T:
            return float(t1)
        return float(t0 + (t1 - t0) * (step / max(1, T)))

    def _compute_mean_dwell_from_ids(
        self,
        ids: torch.Tensor,
        future_step_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ids: [B, F, N] long tensor. Returns (mean_dwell [B,N], valid_mask [B,N]).
        """
        B, F, N = ids.shape
        device = ids.device
        valid = (ids >= 0)
        if future_step_mask is not None:
            step_mask = future_step_mask.to(dtype=torch.bool, device=device)
            valid = valid & step_mask[:, :, None]
        if residue_mask is not None:
            res_mask = residue_mask.to(dtype=torch.bool, device=device)
            valid = valid & res_mask[:, None, :]
        counts = valid.sum(dim=1).clamp_min(1)
        pair_valid = valid[:, 1:, :] & valid[:, :-1, :]
        changes = ((ids[:, 1:, :] != ids[:, :-1, :]) & pair_valid).to(torch.float32)
        num_changes = changes.sum(dim=1)
        mean_dwell = counts.to(torch.float32) / (1.0 + num_changes)
        valid_residues = counts > 0
        return mean_dwell, valid_residues

    def _compute_change_event_terms(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        future_step_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        probs: [B, F, N, C] soft distributions over rollout window.
        targets: [B, F, N] ground-truth ids.
        Returns (loss, accuracy) scalars.
        """
        B, F, N, _ = probs.shape
        if F <= 1:
            zero = probs.new_tensor(0.0)
            return zero, zero
        device = probs.device
        gt_change = (targets[:, 1:, :] != targets[:, :-1, :]).to(probs.dtype)
        mask = torch.ones(B, F - 1, N, dtype=torch.bool, device=device)
        if future_step_mask is not None:
            step_mask = future_step_mask.to(dtype=torch.bool, device=device)
            mask = mask & step_mask[:, 1:, None] & step_mask[:, :-1, None]
        if residue_mask is not None:
            res_mask = residue_mask.to(dtype=torch.bool, device=device)
            mask = mask & res_mask[:, None, :]
        mask = mask & (targets[:, 1:, :] >= 0) & (targets[:, :-1, :] >= 0)
        probs_prev = probs[:, :-1, :, :]
        probs_next = probs[:, 1:, :, :]
        p_stay = (probs_prev * probs_next).sum(dim=-1).clamp(0.0, 1.0)
        p_change = 1.0 - p_stay
        eps = 1e-6
        if mask.any():
            p_sel = p_change.clamp(eps, 1.0 - eps)[mask]
            gt_sel = gt_change[mask]
            loss = F.binary_cross_entropy(p_sel, gt_sel)
            acc = ((p_sel > 0.5).float() == gt_sel).float().mean()
        else:
            loss = probs.new_tensor(0.0)
            acc = probs.new_tensor(0.0)
        return loss, acc

    # losses are implemented in emprot.losses

    def _to_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [self._to_device(v) for v in x]
            return type(x)(t) if isinstance(x, tuple) else t
        return x

    def _save_checkpoint(self, name: str):
        ckpt_dir = self.cfg.get('checkpoint_dir', '.')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, name)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': (self.scheduler.state_dict() if self.scheduler is not None else None),
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.cfg,
        }, path)
