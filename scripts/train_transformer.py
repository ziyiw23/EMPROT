#!/usr/bin/env python3
"""
Train EMPROT Transformer (classification-only) with a clean, config-first script.

Highlights
- YAML-first config with CLI overrides (merge_config_with_args)
- Reproducible seeding and basic logging
- Simple DataLoader creation (no custom curriculum wrapper)
- Optional Weights & Biases logging
"""

import os
import sys
import argparse
import time
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import numpy as np

try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    wandb = None  # type: ignore
    _HAS_WANDB = False

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emprot.data.dataset import create_dataloaders
from emprot.training.trainer import EMPROTTrainer
from scripts.utils.config_parser import merge_config_with_args


log = logging.getLogger("emprot.train")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    def _extract_epoch(p: Path) -> int:
        m = re.search(r'epoch_(\d+).pt', p.name)
        return int(m.group(1)) if m else -1

    if not checkpoint_dir.exists():
        return None
    
    # Look for epoch_N.pt files
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if checkpoints:
        # Return the path to the one with the highest epoch number
        latest_ckpt = max(checkpoints, key=_extract_epoch)
        return str(latest_ckpt)

    # 2. Fuzzy match in parent directory
    # If specific directory is empty/missing, check siblings with matching prefix
    parent_dir = checkpoint_dir.parent
    if parent_dir.exists():
        # Match prefix: strip the last _\d+ suffix (e.g. run_name_123 -> run_name)
        name = checkpoint_dir.name
        match = re.match(r'^(.*)_\d+$', name)
        if match:
            prefix = match.group(1)
            # Find all sibling directories starting with this prefix
            siblings = [
                p for p in parent_dir.iterdir() 
                if p.is_dir() and p.name.startswith(prefix) and p.name != name
            ]
            # Sort by modification time (newest first)
            siblings.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            log.info(f"[AutoResume] Found {len(siblings)} siblings for prefix '{prefix}'")
            for sib in siblings:
                log.info(f"[AutoResume] Checking sibling: {sib.name}")
                # Inline check for sibling checkpoints
                s_ckpts = list(sib.glob("epoch_*.pt"))
                if s_ckpts:
                    latest_sib = max(s_ckpts, key=_extract_epoch)
                    log.info(f"[AutoResume] Found valid checkpoint in sibling: {sib}")
                    return str(latest_sib)
    
    return None


def _coerce_arg_types(ns):
    numeric_casts = {
        'learning_rate': float,
        'weight_decay': float,
        'warmup_proportion': float,
        'max_grad_norm': float,
        'batch_size': int,
        'max_epochs': int,
        'estimated_steps_per_epoch': int,
        'num_full_res_frames': int,
        'stride': int,
        'num_workers': int,
        'future_horizon': int,
        'seed': int,
        'lambda_change': float,
    }
    for key, caster in numeric_casts.items():
        if hasattr(ns, key):
            val = getattr(ns, key)
            try:
                setattr(ns, key, caster(val))
            except Exception:
                pass
    return ns


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Train EMPROT Transformer (config-first)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (relative to configs/ or absolute)')
    parser.add_argument('--config_dir', type=str, default='configs',
                        help='Directory containing configuration files')
    parser.add_argument('--print_config', action='store_true', default=False,
                        help='Print configuration summary and exit')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides default output/checkpoints/<run_name>)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of the run (for WandB and logging)')
    # no auto-resume; pass --resume_from_checkpoint explicitly if desired

    # Parse and merge with YAML
    args = parser.parse_args()
    args = merge_config_with_args(args)
    args = _coerce_arg_types(args)

    # no legacy key migrations

    # Fallbacks for missing keys (when not present in YAML)
    # no large fallback set; rely on YAML config for values actually used

    # Early exit if only printing
    if getattr(args, 'print_config', False):
        log.info("Config: %s", vars(args))
        return

    # Seeding
    log.info("Setting random seed: %s", args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    import random as _random
    _random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Run name fallback
    if getattr(args, 'run_name', None) is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.run_name = f"emprot_run_{ts}"

    log.info("Model: %sd embed, %s heads | batch=%s lr=%s", args.d_embed, args.num_heads, args.batch_size, args.learning_rate)

    # Checkpoint directory (default to output/checkpoints/<run_name>)
    default_ckpt_dir = os.path.join('output', 'checkpoints', str(args.run_name))
    ckpt_dir = getattr(args, 'checkpoint_dir', None) or default_ckpt_dir
    # Create Path object for the function
    ckpt_path = Path(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    log.info("Checkpoints will be saved to: %s", ckpt_dir)

    # Auto-resume logic
    log.info(f"Resume check: args.resume_from_checkpoint={args.resume_from_checkpoint}")
    if not args.resume_from_checkpoint:
        latest = find_latest_checkpoint(ckpt_path)
        if latest:
            log.info(f"Found existing checkpoint, auto-resuming from: {latest}")
            args.resume_from_checkpoint = latest

    # Future horizon
    future_horizon = int(getattr(args, 'future_horizon', 0) or 0)

    # Flattened config (handed off to trainer)
    print(f"DEBUG: args.warmup_steps = {getattr(args, 'warmup_steps', 'MISSING')}")
    config = {
        # Model
        'd_embed': args.d_embed,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_gradient_checkpointing': args.use_gradient_checkpointing,
        'latent_summary_enabled': getattr(args, 'latent_summary_enabled', False),
        'latent_summary_num_latents': getattr(args, 'latent_summary_num_latents', 0),
        'latent_summary_heads': getattr(args, 'latent_summary_heads', None),
        'latent_summary_dropout': getattr(args, 'latent_summary_dropout', None),
        'latent_summary_max_prefix': getattr(args, 'latent_summary_max_prefix', None),

        # Training
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'patience': getattr(args, 'patience', 15),
        'early_stopping_min_delta': getattr(args, 'early_stopping_min_delta', 0.0),
        'weight_decay': args.weight_decay,
        'use_scheduler': args.use_scheduler,
        'warmup_proportion': args.warmup_proportion,
        'warmup_steps': getattr(args, 'warmup_steps', None),
        'estimated_steps_per_epoch': args.estimated_steps_per_epoch,
        'use_amp': args.use_amp,
        'max_grad_norm': args.max_grad_norm,
        'grad_accum_steps': getattr(args, 'grad_accum_steps', 1),
        'per_epoch_lrs': getattr(args, 'per_epoch_lrs', None),

        # Objective & Loss
        'objective': getattr(args, 'objective', 'token_ce'),
        'res_num_samples': getattr(args, 'res_num_samples', 32),
        'res_ce_weight': getattr(args, 'res_ce_weight', 1.0),
        'res_js_weight': getattr(args, 'res_js_weight', 1.0),

        # Loss knobs
        'label_smoothing': getattr(args, 'label_smoothing', 0.0),
        'loss_type': getattr(args, 'loss_type', 'token_ce'),
        'aux_hist_ce_weight': getattr(args, 'aux_hist_ce_weight', 0.0),
        'entropy_bonus_weight': getattr(args, 'entropy_bonus_weight', 0.0),
        'histogram_random_horizon': getattr(args, 'histogram_random_horizon', False),
        'change_upweight': getattr(args, 'change_upweight', 1.0),
        'horizon_weights': getattr(args, 'horizon_weights', None),
        'scheduled_sampling_p': getattr(args, 'scheduled_sampling_p', 0.0),
        'lambda_dwell': getattr(args, 'lambda_dwell', 0.2),
        'lambda_trans': getattr(args, 'lambda_trans', 0.2),
        'lambda_cov': getattr(args, 'lambda_cov', 0.05),
        'lambda_change': getattr(args, 'lambda_change', 0.3),
        'trans_row_min_count': getattr(args, 'trans_row_min_count', 5),
        'coverage_threshold': getattr(args, 'coverage_threshold', 1e-4),

        # Data/system
        'data_dir': args.data_dir,
        'metadata_path': args.metadata_path,
        'history_prefix_frames': getattr(args, 'history_prefix_frames', 0),
        'num_full_res_frames': getattr(args, 'num_full_res_frames', 5),
        'stride': args.stride,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'future_horizon': future_horizon,
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'checkpoint_dir': ckpt_dir,

        # W&B
        'wandb_project': args.wandb_project,
        'run_name': args.run_name,
        'entity': args.entity,
        'tags': args.tags,
        'notes': args.notes,
        'use_wandb': args.use_wandb,
    }

    # W&B
    if config['use_wandb'] and _HAS_WANDB:
        log.info("W&B: project=%s run=%s entity=%s", config['wandb_project'], config['run_name'], config['entity'])
        try:
            wandb.init(project=config['wandb_project'], entity=config['entity'], name=config['run_name'], config=config)  # type: ignore
        except Exception:
            log.warning("W&B init failed; continuing without logging")
            config['use_wandb'] = False

    # Dataloaders (simple split)
    log.info("Building dataloaders…")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=int(args.batch_size),
        history_prefix_frames=int(getattr(args, 'history_prefix_frames', 0) or 0),
        num_full_res_frames=int(getattr(args, 'num_full_res_frames', 5)),
        stride=int(args.stride),
        future_horizon=future_horizon,
        # Online sampling + dynamic weighting knobs (if present in config)
        window_start_stride=int(getattr(args, 'window_start_stride', 1) or 1),
        change_target_fraction=float(getattr(args, 'change_target_fraction', 0.0) or 0.0),
        change_probe_interval=int(getattr(args, 'change_probe_interval', 20) or 20),
        dynamic_score_mode=str(getattr(args, 'dynamic_score_mode', 'f0_any') or 'f0_any'),
        dynamic_weight_gamma=float(getattr(args, 'dynamic_weight_gamma', 0.0) or 0.0),
        dynamic_epsilon=float(getattr(args, 'dynamic_epsilon', 1e-3) or 1e-3),
        score_f_max=int(getattr(args, 'score_f_max', 5) or 5),
        sample_with_replacement=bool(getattr(args, 'sample_with_replacement', False)),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        train_only_proteins=getattr(args, 'train_only_proteins', None),
        max_train_proteins=int(getattr(args, 'max_train_proteins', 0) or 0) or None,
    )
    log.info("Dataset sizes | train=%d val=%d test=%d batches", len(train_loader), len(val_loader), len(test_loader))

    # Create trainer
    trainer = EMPROTTrainer(config)

    if args.resume_from_checkpoint:
        if trainer.load_checkpoint(args.resume_from_checkpoint):
            log.info("Resumed from epoch %d", trainer.epoch + 1)
        else:
            log.warning("Failed to load checkpoint, starting fresh")

    # Train
    log.info("Starting training…")
    trainer.train(train_loader, val_loader)
    log.info("Training completed")

    if config['use_wandb'] and _HAS_WANDB:
        try:
            wandb.finish()  # type: ignore
        except Exception:
            pass


if __name__ == '__main__':
    main()
