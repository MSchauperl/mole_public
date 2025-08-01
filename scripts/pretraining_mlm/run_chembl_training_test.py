#!/usr/bin/env python3
"""
ChemBL MLM training test configuration (1000 samples)

This script provides a minimal configuration for very rapid testing/debugging
with only 1000 ChemBL samples. Perfect for quick iterations and debugging.
"""

import subprocess
import sys
from pathlib import Path

# Minimal test parameters (1000 samples only)
default_params = {
    "--chembl_smiles_path": "data/ChemBl/chembl20Smiles.pckl",
    "--chembl_labels_path": "data/ChemBl/labelsHard.pckl", 
    "--chembl_target_names_path": "data/ChemBl/labelsWeakHard.targetNames",
    "--chembl_compound_names_path": "data/ChemBl/labelsWeakHard.cmpNames",
    "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
    "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
    "--output_dir": "outputs/chembl_test",
    "--model_name": "chembl_test_tiny",
    # ChemBL-specific configuration (minimal)
    "--max_targets": "10",  # Very small for quick testing
    "--min_target_activity": "50",
    # Model configuration (Tiny for speed)
    "--hidden_size": "128",   # Very small
    "--num_hidden_layers": "2",    # Minimal layers
    "--num_attention_heads": "2",  # Minimal heads
    "--intermediate_size": "256", # Small FFN
    "--dropout": "0.1",
    "--classifier_dropout": "0.1",
    # Environment configuration
    "--input_radius": "0",
    "--target_radius": "1",
    "--target_use_features": "",
    # Training configuration (Fast)
    "--batch_size": "4",
    "--learning_rate": "1e-3",  # Higher LR for faster convergence
    "--weight_decay": "0.01",
    "--warmup_steps": "50",   # Very short warmup
    "--max_epochs": "2",      # Just 2 epochs for testing
    "--validation_split": "0.1",
    "--test_split": "0.1",
    "--val_check_interval": "1.0",  # Validate once per epoch
    "--patience": "10",  # Generous patience
    # Loss weighting
    "--mlm_loss_weight": "1.0",
    "--classification_loss_weight": "1.0",
    # Hardware configuration (CPU/GPU agnostic)
    "--gpus": "1",
    "--num_workers": "0",  # No multiprocessing for simplicity
    "--precision": "32",   # Standard precision for compatibility
    "--accumulate_grad_batches": "4",  # Small accumulation
    "--gradient_clip_val": "1.0",
    # Memory optimizations
    "--max_length": "64",   # Very short sequences
    # Dataset limitation (TEST SIZE)
    "--max_samples": "1000",  # Only 1000 samples for rapid testing!
    # Misc
    "--seed": "42",
    "--log_predictions": "",
    "--log_target_metrics": "",
    "--max_targets_to_log": "5",
}


def main():
    """Main function to run the test ChemBL training script."""
    # Build command
    script_path = (
        Path(__file__).parent.parent.parent / "mole" / "cli" / "train_chembl_mlm.py"
    )
    cmd = [sys.executable, str(script_path)]

    # Add default parameters
    for key, value in default_params.items():
        cmd.append(key)
        if value:  # Only add value if it's not an empty string (flag)
            cmd.append(value)

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    print("🧪 Starting MolE ChemBL MLM Training (RAPID TEST)")
    print("=" * 70)
    max_samples = default_params["--max_samples"]
    print(f"📊 Dataset: ChemBL (TEST SIZE: {max_samples} molecules)")
    print("🎯 Target: Rapid testing and debugging")
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (Tiny for Speed)"
    )

    target_radius = default_params["--target_radius"]
    input_radius = default_params["--input_radius"]
    max_targets = default_params["--max_targets"]
    print(
        f"🎯 Task: Predict radius-{target_radius} functional from radius-{input_radius} structural"
    )
    print(f"🧪 ChemBL: Only {max_targets} targets for rapid testing")

    batch_size = default_params["--batch_size"]
    grad_batches = default_params["--accumulate_grad_batches"]
    effective_batch = int(batch_size) * int(grad_batches)
    max_epochs = default_params["--max_epochs"]
    print(f"💾 Batch size: {batch_size} × {grad_batches} = {effective_batch} effective")
    print(f"⚡ Precision: {default_params['--precision']}-bit")
    print(f"🧠 Max sequence length: {default_params['--max_length']}")
    print(f"🏃 Epochs: {max_epochs} (quick test)")
    
    print(f"📁 Output: {default_params['--output_dir']}")
    print("⚠️  WARNING: This is for rapid testing only - tiny model & dataset!")
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