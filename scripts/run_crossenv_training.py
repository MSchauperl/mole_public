#!/usr/bin/env python3
"""
Convenience script to run cross-environment MLM training on GuacaMol dataset

This script provides an easy way to start training with optimized defaults for
the full GuacaMol dataset (~1.3M molecules). You can modify the parameters
below or pass your own arguments.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Default training parameters optimized for Tesla T4 (15.36 GB memory)
    default_params = {
        "--train_data": "data/guacamol_v1_all.smiles",  # Full GuacaMol dataset
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        "--output_dir": "outputs/guacamol_crossenv_mlm",
        "--model_name": "guacamol_r0_to_r1_functional_tesla_t4_optimized",
        # Model configuration (Tesla T4 optimized - 15.36 GB memory)
        "--hidden_size": "768",  # Increased from 384 for Tesla T4
        "--num_hidden_layers": "12",  # Increased from 6 for Tesla T4
        "--num_attention_heads": "12",  # Increased from 6 for Tesla T4
        "--intermediate_size": "3072",  # Increased from 1536 for Tesla T4
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",  # Flag for functional environments
        # Training configuration (Tesla T4 optimized for ~1.6M molecules)
        "--batch_size": "32",  # Increased from 16 for Tesla T4
        "--learning_rate": "1e-4",  # Keep same learning rate
        "--weight_decay": "0.01",
        "--warmup_steps": "5000",  # Adjusted for dataset size
        "--max_epochs": "30",  # Can reduce epochs due to larger model capacity
        "--validation_split": "0.05",  # Smaller validation split (still ~80k molecules)
        "--val_check_interval": "0.5",  # Validate twice per epoch
        # Hardware configuration (Tesla T4 optimized)
        "--gpus": "1",
        "--num_workers": "4",  # Increased workers for Tesla T4
        "--precision": "16",  # Mixed precision for memory efficiency
        "--accumulate_grad_batches": "16",  # Reduced accumulation since batch_size is higher
        "--gradient_clip_val": "1.0",  # Add gradient clipping for stability
        # Memory and efficiency optimizations
        "--max_length": "256",  # Increased sequence length for Tesla T4
        # Misc
        "--seed": "42",
        "--log_predictions": "",  # Flag to log examples
    }

    # Build command
    script_path = (
        Path(__file__).parent.parent / "mole" / "cli" / "train_crossenv_mlm.py"
    )
    cmd = [sys.executable, str(script_path)]

    # Add default parameters
    for key, value in default_params.items():
        cmd.append(key)
        if value:  # Only add value if it's not an empty string (flag)
            cmd.append(value)

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    print("🚀 Starting MolE Cross-Environment MLM Training on GuacaMol Dataset")
    print("=" * 70)
    print("📊 Dataset: GuacaMol (~1.6M molecules)")
    print("🎯 GPU: Tesla T4 (15.36 GB VRAM) - Optimized Configuration")
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (Tesla T4 optimized)"
    )

    target_radius = default_params["--target_radius"]
    input_radius = default_params["--input_radius"]
    print(
        f"🎯 Task: Predict radius-{target_radius} functional from radius-{input_radius} structural"
    )

    batch_size = default_params["--batch_size"]
    grad_batches = default_params["--accumulate_grad_batches"]
    effective_batch = int(batch_size) * int(grad_batches)
    print(f"💾 Batch size: {batch_size} × {grad_batches} = {effective_batch} effective")
    print(f"⚡ Precision: {default_params['--precision']}-bit")
    print(f"🧠 Max sequence length: {default_params['--max_length']}")
    print(f"👥 Workers: {default_params['--num_workers']}")
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
