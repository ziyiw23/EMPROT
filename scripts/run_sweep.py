#!/usr/bin/env python3
"""
Hyperparameter sweep runner for EMPROT.

Usage:
    python scripts/run_sweep.py

This script:
1. Defines a grid of hyperparameters.
2. Generates unique YAML configs for each combination based on a template.
3. Submits SLURM training jobs for each config in parallel.
"""

import argparse
import copy
import itertools
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# SWEEP CONFIGURATION
# -----------------------------------------------------------------------------

# The template config to start from
BASE_CONFIG_PATH = "configs/residue_centric_f3.yaml"

# The parameter grid to explore
# Keys must match the structure in the YAML (use dots for nesting)
PARAM_GRID = {
    "data.num_full_res_frames": [5, 10],
    "model.d_embed": [768],
    "data.future_horizon": [1, 3],
    "model.num_layers": [8],
    "training.res_ce_weight": [1.0, 2.0],
    "training.res_js_weight": [0.0, 1.0],
    "training.scheduled_sampling_p": [0.0, 0.5]
}

# Where to write the generated configs
SWEEP_CONFIG_DIR = Path("configs/sweeps")

# SLURM script to launch
TRAIN_SCRIPT = "scripts/bash_scripts/train_emprot_config.sh"

# -----------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def update_nested_dict(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """Update a nested dictionary using a dot-notation key path."""
    keys = key_path.split('.')
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def generate_run_name(params: Dict[str, Any]) -> str:
    """Create a concise run name from parameter variations."""
    parts = ["sweep"]
    # Abbreviate keys for brevity
    abbrevs = {
        "data.num_full_res_frames": "K",
        "model.d_embed": "D",
        "model.num_layers": "L",
        "training.res_ce_weight": "CE",
        "training.res_js_weight": "JS",
        "training.scheduled_sampling_p": "SS",
    }
    
    for k, v in params.items():
        short_k = abbrevs.get(k, k.split('.')[-1])
        parts.append(f"{short_k}{v}")
    
    # Add timestamp to ensure uniqueness
    timestamp = int(time.time())
    parts.append(str(timestamp)[-4:])
    return "_".join(parts)

def main():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    print(f" • Preparing to launch {len(combinations)} jobs...")
    print(f" • Template: {BASE_CONFIG_PATH}")
    print(f" • Output Dir: {SWEEP_CONFIG_DIR}")
    
    base_config = load_yaml(BASE_CONFIG_PATH)
    
    for combo in combinations:
        params = dict(zip(keys, combo))

        run_config = copy.deepcopy(base_config)

        for k, v in params.items():
            update_nested_dict(run_config, k, v)

        run_name = generate_run_name(params)
        update_nested_dict(run_config, "experiment.run_name", run_name)

        tags = run_config.get("experiment", {}).get("tags", [])
        if "sweep" not in tags:
            tags.append("sweep")
        update_nested_dict(run_config, "experiment.tags", tags)

        config_path = SWEEP_CONFIG_DIR / f"{run_name}.yaml"
        save_yaml(run_config, config_path)

        # We export CHECKPOINT_DIR to control where output goes
        # The shell script respects this env var
        ckpt_dir = Path("ablations") / run_name
        env = os.environ.copy()
        env["CHECKPOINT_DIR"] = str(ckpt_dir)
        
        # The shell script expects a path relative to configs/ OR a filename in configs/
        # Since our configs are in configs/sweeps, we pass 'sweeps/filename.yaml'
        relative_config_path = f"sweeps/{run_name}.yaml"
        cmd = ["sbatch", TRAIN_SCRIPT, relative_config_path]

        import shutil
        if shutil.which("sbatch") is None:
            print(f" • 'sbatch' not found. Dry run: CHECKPOINT_DIR={ckpt_dir} {' '.join(cmd)}")
            continue

        print(f"   • Submitting {run_name} ... ", end="", flush=True)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            job_id = result.stdout.strip().split()[-1]
            print(f" • Job ID: {job_id}")
            print(f"     Config: {config_path}")
        except subprocess.CalledProcessError as e:
            print(f" • Failed!")
            print(f"     Error: {e.stderr}")

if __name__ == "__main__":
    main()

