#!/usr/bin/env python3
import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return lr_lambda

def plot_schedule(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    train_cfg = cfg.get('training', {})
    
    lr = float(train_cfg.get('learning_rate', 1e-4))
    steps_per_epoch = int(train_cfg.get('estimated_steps_per_epoch', 1000))
    max_epochs = int(train_cfg.get('max_epochs', 40))
    warmup_prop = float(train_cfg.get('warmup_proportion', 0.1))
    
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * warmup_prop)
    
    print(f"Config: {config_path}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Steps per Epoch (Optimizer): {steps_per_epoch}")
    print(f"Total Optimizer Steps: {total_steps}")
    print(f"Warmup Steps: {warmup_steps}")
    print(f"Peak LR: {lr}")
    
    # Simulate schedule
    steps = np.arange(total_steps)
    lrs = []
    
    # Mock lambda
    scheduler_lambda = get_cosine_schedule_with_warmup(None, warmup_steps, total_steps)
    
    for s in steps:
        lrs.append(lr * scheduler_lambda(s))
        
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.title(f"Learning Rate Schedule for {Path(config_path).name}")
    plt.xlabel("Optimizer Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    
    # Add epoch markers
    for e in range(max_epochs + 1):
        step = e * steps_per_epoch
        plt.axvline(step, color='r', linestyle='--', alpha=0.3)
        if e < max_epochs:
            plt.text(step + steps_per_epoch/2, lr*0.05, f"Ep {e}", ha='center', alpha=0.5)
            
    out_path = Path(config_path).with_suffix('.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    plot_schedule(args.config)

