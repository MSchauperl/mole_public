#!/usr/bin/env python3
"""
Convenience script to run Unsupervised Pretraining on the GuacaMol dataset.

This script trains a model on three tasks simultaneously:
1. Cross-Environment MLM (Radius 0 Structural -> Radius 1 Functional)
2. ClogP Prediction
3. Molecular Weight Prediction

You can override any parameter using command-line arguments:
  python run_unsupervised_pretraining_t4.py --batch_size 8 --learning_rate 2e-4
"""

import argparse
import subprocess
import sys
from pathlib import Path

def parse_override_args():
    """Parse command-line arguments to override default parameters."""
    parser = argparse.ArgumentParser(
        description="Run Unsupervised Pretraining with T4 optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # We'll add help manually to avoid conflicts
    )
    
    # Add help argument manually
    parser.add_argument(
        "-h", "--help", 
        action="store_true", 
        help="Show this help message and exit"
    )
    
    # Common override arguments
    parser.add_argument("--batch_size", type=str, help="Override batch size")
    parser.add_argument("--learning_rate", type=str, help="Override learning rate")
    parser.add_argument("--max_epochs", type=str, help="Override maximum epochs")
    parser.add_argument("--hidden_size", type=str, help="Override hidden size")
    parser.add_argument("--num_hidden_layers", type=str, help="Override number of layers")
    parser.add_argument("--mlm_loss_weight", type=str, help="Override MLM loss weight")
    parser.add_argument("--regression_loss_weight", type=str, help="Override regression loss weight")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder layers")
    
    # Allow any other arguments to be passed through
    args, unknown_args = parser.parse_known_args()
    
    return args, unknown_args

def main():
    # Parse command-line overrides
    args, unknown_args = parse_override_args()
    
    # Show help if requested
    if args.help:
        parser = argparse.ArgumentParser(
            description="Run Unsupervised Pretraining with T4 optimizations",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add the same arguments for help display
        parser.add_argument("--batch_size", type=str, help="Override batch size")
        parser.add_argument("--learning_rate", type=str, help="Override learning rate")
        parser.add_argument("--max_epochs", type=str, help="Override maximum epochs")
        parser.add_argument("--hidden_size", type=str, help="Override hidden size")
        parser.add_argument("--num_hidden_layers", type=str, help="Override number of layers")
        parser.add_argument("--mlm_loss_weight", type=str, help="Override MLM loss weight")
        parser.add_argument("--regression_loss_weight", type=str, help="Override regression loss weight")
        parser.add_argument("--output_dir", type=str, help="Override output directory")
        parser.add_argument("--model_name", type=str, help="Override model name")
        parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
        parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder layers")
        
        parser.print_help()
        print("\nThis script runs unsupervised pretraining with T4-optimized defaults.")
        print("You can override any parameter using command-line arguments.")
        print("\nFor a full list of available parameters, run:")
        print("  python mole/cli/train_unsupervised_pretraining.py --help")
        sys.exit(0)
    
    # Inherit most defaults from the T4-optimized cross-environment script
    from run_crossenv_training_t4 import default_params

    # Add/override parameters for unsupervised learning
    unsupervised_params = {
        "--output_dir": "outputs/guacamol_unsupervised_pretraining_t4",
        "--model_name": "guacamol_unsupervised_pretraining_t4",
        "--mlm_loss_weight": "1.0",
        "--regression_loss_weight": "0.1",  # Start with a smaller weight for regression
    }
    default_params.update(unsupervised_params)
    
    # Apply command-line overrides
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'help':
            param_name = f"--{arg_name}"
            if arg_name == 'freeze_encoder' and arg_value:
                default_params[param_name] = ""  # Flag parameter
            elif arg_name != 'freeze_encoder':
                default_params[param_name] = str(arg_value)

    # Build command
    script_path = (
        Path(__file__).parent.parent.parent / "mole" / "cli" / "train_unsupervised_pretraining.py"
    )
    cmd = [sys.executable, str(script_path)]

    # Add parameters
    for key, value in default_params.items():
        cmd.append(key)
        if value:  # Only add value if it's not an empty string (for flags)
            cmd.append(value)

    # Add any additional unknown arguments
    cmd.extend(unknown_args)

    print("🚀 Starting MolE Unsupervised Pretraining on GuacaMol Dataset")
    print("=" * 70)
    print("🎯 GPU: Tesla T4 (16 GB VRAM) - Optimized Configuration")
    print("🎯 Tasks:")
    print("  - Predict radius-1 functional from radius-0 structural (MLM)")
    print("  - Predict ClogP (Regression)")
    print("  - Predict Molecular Weight (Regression)")
    print(
        f"⚖️ Loss Weights: MLM = {default_params['--mlm_loss_weight']}, Regression = {default_params['--regression_loss_weight']}"
    )
    print(f"📁 Output: {default_params['--output_dir']}")
    
    # Show overridden parameters
    overridden_params = []
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'help':
            overridden_params.append(f"--{arg_name}={arg_value}")
    
    if overridden_params:
        print("🔧 Overridden parameters:")
        for param in overridden_params:
            print(f"  - {param}")
    
    if unknown_args:
        print("➕ Additional arguments:")
        print(f"  {' '.join(unknown_args)}")
    
    print("=" * 70)
    print()

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 