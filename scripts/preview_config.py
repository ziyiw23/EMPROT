#!/usr/bin/env python3
"""
Preview configuration files for EMPROT training
Useful for checking config settings before starting training
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.config_parser import ConfigParser


def main():
    parser = argparse.ArgumentParser(description='Preview EMPROT training configurations')
    parser.add_argument('config', type=str, nargs='?',
                       help='Configuration file to preview (e.g., baseline_mse_only.yaml)')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Directory containing configuration files')
    parser.add_argument('--list', action='store_true',
                       help='List all available configuration files')
    
    args = parser.parse_args()
    
    if args.list and not args.config:
        # If only listing, config is not required
        pass
    elif not args.config and not args.list:
        parser.error("Either provide a config file name or use --list")
    elif not args.config and not args.list:
        parser.error("Configuration file is required unless using --list")
    
    config_parser = ConfigParser(args.config_dir)
    
    if args.list:
        config_dir = Path(args.config_dir)
        if config_dir.exists():
            print("📋 Available Configuration Files:")
            print("=" * 50)
            yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
            for yaml_file in sorted(yaml_files):
                print(f"   • {yaml_file.name}")
            print()
        else:
            print(f"❌ Config directory not found: {config_dir}")
        return
    
    try:
        config = config_parser.load_config(args.config)
        config_parser.print_config_summary(config, args.config)
        
        # Print some key training parameters
        print("\n🚀 Quick Training Preview:")
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        print(f"   • Will train for max {training_config.get('max_epochs', 20)} epochs")
        print(f"   • Batch size: {data_config.get('batch_size', 32)}")
        print(f"   • Learning rate: {training_config.get('learning_rate', 1e-4)}")
        print(f"   • Data directory: {data_config.get('data_dir', 'Not specified')}")
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        print(f"   Looked in: {Path(args.config_dir).absolute()}")
        print("\n💡 Use --list to see available configurations")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")


if __name__ == "__main__":
    main()
