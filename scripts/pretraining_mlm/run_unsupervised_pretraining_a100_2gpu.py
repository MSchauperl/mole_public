#!/usr/bin/env python3
"""
Convenience script to run Unsupervised Pretraining on the GuacaMol dataset.

This script trains a model on three tasks simultaneously, optimized for two
NVIDIA A100 (40GB) GPUs:
1. Cross-Environment MLM (Radius 0 Structural -> Radius 1 Functional)
2. ClogP Prediction
3. Molecular Weight Prediction
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run Unsupervised Pretraining, optimized for A100 GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of GPUs to use for training (default: 2).",
    )
    # Parse only the --gpus argument, and forward the rest to the training script
    args, remaining_argv = parser.parse_known_args()

    # Default training parameters optimized for two NVIDIA A100s (40 GB memory)
    a100_params = {
        "--train_data": "data/guacamol_v1_all.smiles",  # Full GuacaMol dataset
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        # Model configuration (BERT-base size)
        "--hidden_size": "768",
        "--num_hidden_layers": "12",
        "--num_attention_heads": "12",
        "--intermediate_size": "3072",
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",  # Flag for functional environments
        # Training configuration (A100 2x40GB optimized)
        "--batch_size": "128",  # Increased for A100
        "--learning_rate": "1e-4",
        "--weight_decay": "0.01",
        "--warmup_steps": "5000",
        "--max_epochs": "30",
        "--validation_split": "0.05",
        "--val_check_interval": "0.2",
        # Hardware configuration (Dynamically set)
        "--gpus": str(args.gpus),
        "--num_workers": "16",
        "--precision": "bf16",
        "--accumulate_grad_batches": "48" if args.gpus == 2 else "16",  # Adjust for GPU count
        "--gradient_clip_val": "1.0",
        # Memory and efficiency optimizations
        "--max_length": "256",
        # Misc
        "--seed": "42",
        "--log_predictions": "",
        "--patience": "5",
        #"--use_torch_compile": "",
    }

    # Start with A100-optimized parameters
    params = a100_params.copy()

    # Add/override parameters for unsupervised learning
    unsupervised_params = {
        "--output_dir": "outputs/guacamol_unsupervised_pretraining_a100_2gpu",
        "--model_name": "guacamol_unsupervised_pretraining_a100_2gpu",
        "--mlm_loss_weight": "1.0",
        "--regression_loss_weight": "0.2",  # Adjusted to balance with MLM loss (10^4 scale difference)
    }
    params.update(unsupervised_params)

    # Build command
    script_path = (
        Path(__file__).parent.parent.parent / "mole" / "cli" / "train_unsupervised_pretraining.py"
    )
    cmd = [sys.executable, str(script_path)]

    # Add parameters
    for key, value in params.items():
        cmd.append(key)
        if value:
            cmd.append(value)

    # Add any additional arguments passed to this script
    cmd.extend(remaining_argv)

    print("🚀 Starting MolE Unsupervised Pretraining on GuacaMol Dataset")
    print("=" * 70)
    print(f"🎯 GPU: {args.gpus}x NVIDIA A100 (40 GB VRAM) - Optimized Configuration")
    print("🎯 Tasks:")
    print("  - Predict radius-1 functional from radius-0 structural (MLM)")
    print("  - Predict ClogP (Regression)")
    print("  - Predict Molecular Weight (Regression)")
    print(
        f"⚖️ Loss Weights: MLM = {params['--mlm_loss_weight']}, Regression = {params['--regression_loss_weight']}"
    )
    print(f"📁 Output: {params['--output_dir']}")
    batch_size = params["--batch_size"]
    grad_batches = params["--accumulate_grad_batches"]
    gpus = params["--gpus"]
    effective_batch = int(batch_size) * int(grad_batches) * int(gpus)
    print(f"💾 Batch size: {batch_size} (per GPU) × {grad_batches} (accum) × {gpus} (GPUs) = {effective_batch} effective")
    print(f"⚡ Precision: {params['--precision']}-bit")
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