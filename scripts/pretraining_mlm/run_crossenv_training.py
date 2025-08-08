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


# Default training parameters optimized for NVIDIA A100 (40/80 GB memory)
default_params = {
    "--train_data": "data/guacamol_v1_all.smiles",  # Full GuacaMol dataset
    "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
    "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
    "--output_dir": "outputs/guacamol_crossenv_mlm",
    "--model_name": "guacamol_r0_to_r1_functional_a100_optimized",
    # Model configuration (same as Tesla T4 to allow for larger batches)
    "--hidden_size": "768",
    "--num_hidden_layers": "12",
    "--num_attention_heads": "12",
    "--intermediate_size": "3072",
    "--dropout": "0.1",
    # Environment configuration
    "--input_radius": "0",
    "--target_radius": "1",
    "--target_use_features": "",  # Flag for functional environments
    # Training configuration (NVIDIA A100 optimized for ~1.6M molecules)
    "--batch_size": "128",  # Increased from 32 for A100
    "--learning_rate": "5e-4",  # Keep same learning rate
    "--weight_decay": "0.01",
    "--warmup_steps": "5000",  # Adjusted for dataset size
    "--max_epochs": "100",
    "--validation_split": "0.05",
    "--val_check_interval": "0.25",  # Validate twice per epoch
    # Hardware configuration (NVIDIA A100 optimized)
    "--gpus": "1",
    "--num_workers": "16",  # Increased workers for A100 (tune based on CPU cores)
    "--precision": "16",  # A100 is highly optimized for mixed precision
    "--accumulate_grad_batches": "2",  # Adjust accumulation for larger batch size
    "--gradient_clip_val": "1.0",
    # Memory and efficiency optimizations
    "--max_length": "256",  # Keep sequence length same for comparability
    # Misc
    "--seed": "42",
    "--log_predictions": "",  # Flag to log examples
    #"--use_torch_compile": "",  # Flag to enable torch.compile
    "--patience": "6", # Stop if validation loss does not improve for 5 checks

}


def main():
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
    print("🎯 GPU: NVIDIA A100 (40/80 GB VRAM) - Optimized Configuration")
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (NVIDIA A100 optimized)"
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
