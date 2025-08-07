#!/usr/bin/env python3
"""
Convenience script to run cross-environment MLM training on GuacaMol dataset

This script provides an easy way to start training with optimized defaults for
the full GuacaMol dataset (~1.3M molecules), specifically tailored for
a Tesla T4 GPU (16 GB memory).
"""

import subprocess
import sys
from pathlib import Path

# Default training parameters optimized for Tesla T4 (16 GB memory)
default_params = {
    "--train_data": "data/guacamol_v1_small.smiles",  # Full GuacaMol dataset
    "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
    "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
    "--output_dir": "outputs/guacamol_crossenv_mlm",
    "--model_name": "guacamol_r0_to_r1_functional_t4_optimized",
    # Model configuration (BERT-base size, should fit on T4 with smaller batches)
    "--hidden_size": "768",
    "--num_hidden_layers": "12",
    "--num_attention_heads": "12",
    "--intermediate_size": "3072",
    "--dropout": "0.1",
    # Environment configuration
    "--input_radius": "0",
    "--target_radius": "1",
    "--target_use_features": "",  # Flag for functional environments
    # Training configuration (Tesla T4 optimized)
    "--batch_size": "32",  # Reduced from 256 for T4
    "--learning_rate": "1e-4",
    "--weight_decay": "0.01",
    "--warmup_steps": "5000",
    "--max_epochs": "5",  # Increased for a more complete run on a single machine
    "--validation_split": "0.05",
    "--val_check_interval": "0.2", # Run validation 5 times per epoch
    # Hardware configuration (Tesla T4 optimized)
    "--gpus": "1",
    "--num_workers": "4",  # Reduced for typical T4 setups
    "--precision": "16",  # Mixed precision is crucial for T4 memory
    "--accumulate_grad_batches": "16",  # Increased to maintain effective batch size
    "--gradient_clip_val": "1.0",
    # Memory and efficiency optimizations
    "--max_length": "256",
    # Misc
    "--seed": "42",
    "--log_predictions": "",
    "--patience": "5", # Stop if validation loss does not improve for 5 checks
}


def main():
    """Main function to run the T4-optimized training script."""
    # Build command
    script_path = (
        Path(__file__).parent.parent.parent / "mole" / "cli" / "train_crossenv_mlm.py"
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
    print("🎯 GPU: Tesla T4 (16 GB VRAM) - Optimized Configuration")
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