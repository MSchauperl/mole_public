#!/usr/bin/env python3
"""
Conservative version of cross-environment MLM training for Tesla T4

This script provides a more conservative configuration that uses about 8-10 GB
of the Tesla T4's memory, leaving headroom for other processes.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Conservative training parameters for Tesla T4 (using ~8-10 GB memory)
    default_params = {
        "--train_data": "data/guacamol_v1_all.smiles",  # Full GuacaMol dataset
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        "--output_dir": "outputs/guacamol_crossenv_mlm",
        "--model_name": "guacamol_r0_to_r1_functional_tesla_t4_conservative",
        # Model configuration (Tesla T4 conservative - ~8-10 GB memory)
        "--hidden_size": "512",  # Moderate increase from 384
        "--num_hidden_layers": "8",  # Moderate increase from 6
        "--num_attention_heads": "8",  # Moderate increase from 6
        "--intermediate_size": "2048",  # Moderate increase from 1536
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",  # Flag for functional environments
        # Training configuration (Tesla T4 conservative for ~1.6M molecules)
        "--batch_size": "24",  # Moderate increase from 16
        "--learning_rate": "1e-4",  # Keep same learning rate
        "--weight_decay": "0.01",
        "--warmup_steps": "5000",  # Adjusted for dataset size
        "--max_epochs": "30",  # Same as optimized version
        "--validation_split": "0.05",  # Smaller validation split (still ~80k molecules)
        "--val_check_interval": "0.5",  # Validate twice per epoch
        # Hardware configuration (Tesla T4 conservative)
        "--gpus": "1",
        "--num_workers": "3",  # Moderate increase from 2
        "--precision": "16",  # Mixed precision for memory efficiency
        "--accumulate_grad_batches": "5",  # Moderate accumulation
        "--gradient_clip_val": "1.0",  # Add gradient clipping for stability
        # Memory and efficiency optimizations
        "--max_length": "128",  # Moderate increase from 100
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
    print("🎯 GPU: Tesla T4 (15.36 GB VRAM) - Conservative Configuration")
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (Tesla T4 conservative)"
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