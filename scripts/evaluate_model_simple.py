#!/usr/bin/env python3
"""
Minimal evaluation entrypoint for classification-only EMPROT models.

Loads a checkpoint, builds test loader via the same dataset API, and reports
loss and simple classification metrics on the test split.

Usage:
  python scripts/evaluate_model_simple.py \
    --config emprot.yaml \
    --ckpt checkpoints/best.pt
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emprot.data.dataset import create_dataloaders
from emprot.training.trainer import EMPROTTrainer
from scripts.utils.config_parser import merge_config_with_args


log = logging.getLogger("emprot.eval_simple")


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help='YAML config (relative to configs/ or absolute)')
    p.add_argument('--config_dir', type=str, default='configs')
    p.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pt file')
    p.add_argument('--print_config', action='store_true', default=False)
    args = p.parse_args(argv)

    # Merge config and coerce
    args = merge_config_with_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Device: %s', device)

    # Build dataloaders
    log.info('Building dataloaders…')
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=int(args.batch_size),
        history_prefix_frames=int(getattr(args, 'history_prefix_frames', 0) or 0),
        num_full_res_frames=int(args.num_full_res_frames),
        stride=int(args.stride),
        future_horizon=int(getattr(args, 'future_horizon', 0) or 0),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    # Create trainer + model and load weights
    log.info('Loading checkpoint: %s', args.ckpt)
    trainer = EMPROTTrainer(vars(args))
    state = torch.load(args.ckpt, map_location='cpu')
    try:
        trainer.model.load_state_dict(state['model'], strict=False)
    except Exception:
        # Some older checkpoints used 'model_state_dict'
        trainer.model.load_state_dict(state.get('model_state_dict', {}), strict=False)
    trainer.model.to(device)

    # Evaluate on test split
    report = trainer.validate(test_loader)
    print('Evaluation (test split)')
    for k, v in report.items():
        print(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}')


if __name__ == '__main__':
    main()

