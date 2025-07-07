#!/usr/bin/env python3
"""
Convenience script to run Multi-Task Pretraining on the GuacaMol dataset.

This script trains a model on three tasks simultaneously:
1. Cross-Environment MLM (Radius 0 Structural -> Radius 1 Functional)
2. ClogP Prediction
3. Molecular Weight Prediction
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Inherit most defaults from the T4-optimized cross-environment script
    from run_crossenv_training_t4 import default_params

    # Add/override parameters for multi-task learning
    multitask_params = {
        "--output_dir": "outputs/guacamol_multitask_t4",
        "--model_name": "guacamol_multitask_t4_optimized",
        "--mlm_loss_weight": "1.0",
        "--regression_loss_weight": "0.1",  # Start with a smaller weight for regression
    }
    default_params.update(multitask_params)

    # Build command
    script_path = (
        Path(__file__).parent.parent / "mole" / "cli" / "train_multitask.py"
    )
    cmd = [sys.executable, str(script_path)]

    # Add parameters
    for key, value in default_params.items():
        cmd.append(key)
        if value:
            cmd.append(value)

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    print("🚀 Starting MolE Multi-Task Pretraining on GuacaMol Dataset")
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