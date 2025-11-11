#!/usr/bin/env python3
"""
Configuration parser for EMPROT training
Supports hierarchical config loading with base config inheritance
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

class ConfigParser:
    """Parse and merge YAML configuration files with base config inheritance."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a config file, resolving base_config inheritance.
        
        Args:
            config_path: Path to the config file (relative to config_dir or absolute)
            
        Returns:
            Merged configuration dictionary
        """
        config_path = Path(config_path)
        
        # If relative path, make it relative to config_dir
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        # Load the main config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle base config inheritance
        if 'base_config' in config:
            # For base config, just pass the filename since it's relative to config_dir
            base_config = self.load_config(config['base_config'])
            
            # Merge base config with current config (current config takes precedence)
            merged_config = self._deep_merge(base_config, config)
            
            # Remove the base_config key from final config
            merged_config.pop('base_config', None)
            return merged_config
            
        return config
        
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def config_to_args(self, config: Dict[str, Any]) -> argparse.Namespace:
        """Convert config dictionary to argparse Namespace for compatibility."""
        args = argparse.Namespace()
        
        # Flatten nested config structure to match training script argument names
        flat_config = self._flatten_config(config)
        
        # Set attributes on args namespace
        for key, value in flat_config.items():
            setattr(args, key, value)
            
        return args
        
    def _flatten_config(self, config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested config to match training script argument names."""
        flat = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Special handling for known sections
                if key in ['model', 'data', 'training', 'loss', 'experiment', 
                          'checkpoint', 'validation', 'curriculum', 'mode', 'misc', 'scheduler',
                          'optimization', 'stabilizers']:
                    flat.update(self._flatten_config(value, ''))
                else:
                    flat.update(self._flatten_config(value, f"{prefix}{key}_"))
            else:
                flat_key = f"{prefix}{key}" if prefix else key
                flat[flat_key] = value
        
        return flat
        
    def print_config_summary(self, config: Dict[str, Any], config_name: str):
        """Print a summary of the loaded configuration."""
        print(f"\n Configuration: {config_name}")
        print("=" * 60)
        
        # Training mode detection
        reg_weight = config.get('loss', {}).get('regression_weight', 0.0)
        cls_weight = config.get('loss', {}).get('classification_weight', 0.0)
        
        if cls_weight == 0.0 and reg_weight != 0.0:
            mode = "Continuous regression (MSE only)"
        elif reg_weight == 0.0 and cls_weight != 0.0:
            mode = "Classification only (CrossEntropy only)"
        else:
            mode = "Dual-head (MSE + CrossEntropy)"
            
        print(f" Training Mode: {mode}")
        print(f"   • Regression weight: {reg_weight}")
        print(f"   • Classification weight: {cls_weight}")
        
        # Model config
        model_config = config.get('model', {})
        print(f"\n  Model Architecture:")
        print(f"   • Embedding dim: {model_config.get('d_embed', 512)}")
        print(f"   • Attention heads: {model_config.get('num_heads', 8)}")
        print(f"   • Dropout: {model_config.get('dropout', 0.1)}")
        if model_config.get('latent_summary_enabled', False):
            print(f"   • Latent summary: ENABLED")
            print(f"     - num_latents: {model_config.get('latent_summary_num_latents', 0)}")
            print(f"     - heads: {model_config.get('latent_summary_heads', model_config.get('num_heads', 8))}")
            print(f"     - dropout: {model_config.get('latent_summary_dropout', model_config.get('dropout', 0.1))}")
            print(f"     - max_prefix L: {model_config.get('latent_summary_max_prefix', 'ALL')}")
        
        # Data config
        data_config = config.get('data', {})
        print(f"\n📊 Data Configuration:")
        print(f"   • Batch size: {data_config.get('batch_size', 32)}")
        print(f"   • L (prefix frames): {data_config.get('history_prefix_frames', 0)}")
        print(f"   • K (full-res frames): {data_config.get('num_full_res_frames', 5)}")
        print(f"   • F (future horizon): {data_config.get('future_horizon', 1)}")
        print(f"   • Stride: {data_config.get('stride', 10)}")
        
        # Loss components
        loss_config = config.get('loss', {})
        print(f"\n🔧 Loss Components:")
        components = []
        if loss_config.get('use_amplification', False):
            components.append("magnitude amplification")
        if loss_config.get('use_diversity_loss', False):
            components.append("diversity regularization")
        if loss_config.get('use_temporal_order_loss', False):
            components.append("temporal order")
        if loss_config.get('use_transition_smoothness', False):
            components.append("transition smoothness")
            
        if components:
            print(f"   • Enabled: {', '.join(components)}")
        else:
            print(f"   • No optional components enabled")
            
        # Curriculum
        curriculum_config = config.get('curriculum', {})
        print(f"\n Curriculum Learning:")
        data_curr = not curriculum_config.get('disable_data_curriculum', True)
        loss_curr = not curriculum_config.get('disable_loss_curriculum', True)
        print(f"   • Data curriculum: {'ENABLED' if data_curr else 'DISABLED'}")
        print(f"   • Loss curriculum: {'ENABLED' if loss_curr else 'DISABLED'}")
        
        # Experiment info
        exp_config = config.get('experiment', {})
        print(f"\n Experiment:")
        print(f"   • Run name: {exp_config.get('run_name', 'emprot_training')}")
        print(f"   • Tags: {', '.join(exp_config.get('tags', []))}")
        
        print("=" * 60)


def add_config_args(parser: argparse.ArgumentParser):
    """Add config-related arguments to an existing argument parser."""
    # Check if arguments already exist to avoid conflicts
    existing_args = set()
    for action in parser._actions:
        for option_string in action.option_strings:
            existing_args.add(option_string)
    
    if '--config' not in existing_args:
        parser.add_argument('--config', type=str, default=None,
                           help='Path to YAML configuration file (relative to configs/ or absolute)')
    if '--config_dir' not in existing_args:
        parser.add_argument('--config_dir', type=str, default='configs',
                           help='Directory containing configuration files')
    if '--print_config' not in existing_args:
        parser.add_argument('--print_config', action='store_true', default=False,
                           help='Print configuration summary and exit')


def merge_config_with_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file with command line arguments.
    Command line arguments take precedence over config file.
    """
    if not hasattr(args, 'config') or args.config is None:
        return args
        
    # Load config
    config_parser = ConfigParser(args.config_dir)
    config = config_parser.load_config(args.config)
    
    print(f"Config from {args.config}:")
    if 'loss' in config:
        print(f"   • loss.ce_weight: {config['loss'].get('ce_weight', config['loss'].get('classification_weight', 'MISSING'))}")
        print(f"   • loss.KLdiv_weight: {config['loss'].get('KLdiv_weight', config['loss'].get('kldiv_weight', 'MISSING'))}")
    
    # Print config summary if requested
    if args.print_config:
        config_parser.print_config_summary(config, args.config)
        return args
    
    # Convert config to args format
    config_args = config_parser.config_to_args(config)
    
    # DEBUG: Print flattened args
    print(f"🔍 DEBUG: Flattened config args:")
    # 'classification_only' no longer used
    print(f"   • regression_weight: {getattr(config_args, 'regression_weight', 'MISSING')}")
    print(f"   • ce_weight: {getattr(config_args, 'ce_weight', getattr(config_args, 'classification_weight', 'MISSING'))}")
    print(f"   • KLdiv_weight: {getattr(config_args, 'KLdiv_weight', getattr(config_args, 'kldiv_weight', 'MISSING'))}")
    
    # Merge: start with config values, then override ONLY with CLI-specified options
    merged_args = argparse.Namespace()
    for key, value in vars(config_args).items():
        setattr(merged_args, key, value)

    # Determine which options were explicitly provided on the command line
    def _cli_provided_keys(argv: Optional[list] = None) -> set:
        if argv is None:
            argv = sys.argv[1:]
        provided = set()
        for token in argv:
            if not isinstance(token, str):
                continue
            if not token.startswith('--'):
                continue
            name = token.lstrip('-')
            if '=' in name:
                name = name.split('=', 1)[0]
            dest = name.replace('-', '_')
            if dest == 'no_scheduler':
                dest = 'use_scheduler'
            provided.add(dest)
        return provided

    provided_keys = _cli_provided_keys()
    # When a config is provided, restrict overrides to a small allowlist
    allowlist_with_config = {'config', 'config_dir', 'print_config', 'resume_from_checkpoint', 'auto_resume'}
    if hasattr(args, 'config') and args.config:
        provided_keys = provided_keys.intersection(allowlist_with_config)
        # Also ensure allowlisted meta keys are carried over even if not explicitly provided
        provided_keys.update({'config', 'config_dir', 'print_config'})

    # Apply CLI overrides for allowed provided keys, and carry over any args absent from config
    for key, value in vars(args).items():
        apply_override = (key in provided_keys)
        fill_missing = (key not in vars(config_args))
        if apply_override or fill_missing:
            old_val = getattr(merged_args, key, 'MISSING')
            if apply_override and old_val != value:
                print(f"🔧 DEBUG: Overriding from CLI -> {key}: {old_val} -> {value}")
            setattr(merged_args, key, value)
    
    # Print summary
    if hasattr(args, 'config') and args.config:
        config_parser.print_config_summary(config, args.config)
        
    return merged_args


if __name__ == "__main__":
    # Test the config parser
    parser = ConfigParser()
    
    # Test loading different configs
    configs_to_test = [
        "baseline_mse_only.yaml",
        "classification_only.yaml", 
        "dual_head.yaml",
        "full_features.yaml"
    ]
    
    for config_name in configs_to_test:
        try:
            config = parser.load_config(config_name)
            parser.print_config_summary(config, config_name)
            print()
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_name}")
        except Exception as e:
            print(f"❌ Error loading {config_name}: {e}")
