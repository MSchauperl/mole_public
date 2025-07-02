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
    # Default training parameters optimized for GuacaMol dataset with strict memory constraints
    default_params = {
        "--train_data": "data/guacamol_v1_all.smiles",  # Full GuacaMol dataset
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        "--output_dir": "outputs/guacamol_crossenv_mlm",
        "--model_name": "guacamol_r0_to_r1_functional_v3",
        # Model configuration (very memory optimized for RTX 3070)
        "--hidden_size": "384",  # Further reduced from 512 to save memory
        "--num_hidden_layers": "6",  # Further reduced from 8 to save memory
        "--num_attention_heads": "6",  # Further reduced from 8 to save memory
        "--intermediate_size": "1536",  # Further reduced from 2048 to save memory
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",  # Flag for functional environments
        # Training configuration (very memory optimized for ~1.6M molecules)
        "--batch_size": "16",  # Further reduced from 32 to save memory
        "--learning_rate": "1e-4",  # Keep same learning rate
        "--weight_decay": "0.01",
        "--warmup_steps": "5000",  # Adjusted for dataset size
        "--max_epochs": "30",  # More epochs to compensate for smaller model
        "--validation_split": "0.05",  # Smaller validation split (still ~80k molecules)
        "--val_check_interval": "0.5",  # Validate twice per epoch
        # Hardware configuration (very memory optimized)
        "--gpus": "1",
        "--num_workers": "2",  # Further reduced workers to save memory
        "--precision": "16",  # Mixed precision for memory efficiency
        "--accumulate_grad_batches": "8",  # Increased accumulation for effective batch size = 128
        "--gradient_clip_val": "1.0",  # Add gradient clipping for stability
        # Memory and efficiency optimizations
        "--max_length": "100",  # Further reduced sequence length for memory
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
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (memory optimized)"
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
