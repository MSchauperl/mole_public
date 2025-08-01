#!/usr/bin/env python3
"""
Convenience script to run ChemBL MLM training on ChemBL dataset

This script provides an easy way to start ChemBL pretraining with optimized defaults
for the ChemBL dataset (~456K molecules), specifically tailored for
a Tesla T4 GPU (16 GB memory).
"""

import subprocess
import sys
from pathlib import Path

# Default training parameters optimized for Tesla T4 (16 GB memory)
default_params = {
    "--chembl_smiles_path": "data/ChemBl/chembl20Smiles.pckl",
    "--chembl_labels_path": "data/ChemBl/labelsHard.pckl", 
    "--chembl_target_names_path": "data/ChemBl/labelsWeakHard.targetNames",
    "--chembl_compound_names_path": "data/ChemBl/labelsWeakHard.cmpNames",
    "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
    "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
    "--output_dir": "outputs/chembl_mlm",
    "--model_name": "chembl_r0_to_r1_functional_t4_optimized",
    # ChemBL-specific configuration
    "--max_targets": "50",  # Further reduced for T4 memory constraints
    "--min_target_activity": "50",  # Ensure sufficient data per target
    # Model configuration (Reduced for T4 memory constraints)
    "--hidden_size": "512",  # Reduced from 768
    "--num_hidden_layers": "8",   # Reduced from 12
    "--num_attention_heads": "8", # Reduced from 12
    "--intermediate_size": "2048", # Reduced from 3072
    "--dropout": "0.1",
    "--classifier_dropout": "0.1",
    # Environment configuration
    "--input_radius": "0",
    "--target_radius": "1",
    "--target_use_features": "",  # Flag for functional environments
    # Training configuration (Tesla T4 optimized)
    "--batch_size": "4",  # Further reduced for T4 memory
    "--learning_rate": "1e-4",
    "--weight_decay": "0.01",
    "--warmup_steps": "3000",
    "--max_epochs": "20",  # Reasonable for ChemBL pretraining
    "--validation_split": "0.1",
    "--test_split": "0.1",
    "--val_check_interval": "0.25",  # Check validation 4 times per epoch
    "--patience": "3",  # Early stopping patience
    # Loss weighting
    "--mlm_loss_weight": "0.1",
    "--classification_loss_weight": "1.0",
    # Hardware configuration (Tesla T4 optimized)
    "--gpus": "1",
    "--num_workers": "4",  # Reduced for typical T4 setups
    "--precision": "16",  # Mixed precision is crucial for T4 memory
    "--accumulate_grad_batches": "16",  # Increased to maintain effective batch size of 64
    "--gradient_clip_val": "1.0",
    # Memory and efficiency optimizations
    "--max_length": "256",
    # Misc
    "--seed": "42",
    "--log_predictions": "",
    "--log_target_metrics": "",  # Log per-target metrics
    "--max_targets_to_log": "20",  # Limit logged targets
    # Dataset size limitation for testing
    "--max_samples": "45633",  # 10% of ~456K molecules for testing
}


def main():
    """Main function to run the T4-optimized ChemBL training script."""
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

    print("🚀 Starting MolE ChemBL MLM Training")
    print("=" * 70)
    max_samples = default_params.get("--max_samples", "456331")
    if max_samples != "456331":
        print(f"📊 Dataset: ChemBL (LIMITED TO {max_samples} molecules for testing)")
    else:
        print("📊 Dataset: ChemBL (~456K molecules)")
    print("🎯 GPU: Tesla T4 (16 GB VRAM) - Memory Optimized Configuration")
    hidden_size = default_params["--hidden_size"]
    num_layers = default_params["--num_hidden_layers"]
    print(
        f"🏗️  Architecture: {hidden_size} hidden, {num_layers} layers (T4 Memory Optimized)"
    )

    target_radius = default_params["--target_radius"]
    input_radius = default_params["--input_radius"]
    max_targets = default_params["--max_targets"]
    min_activity = default_params["--min_target_activity"]
    print(
        f"🎯 Task: Predict radius-{target_radius} functional from radius-{input_radius} structural"
    )
    print(f"🧪 ChemBL: Max {max_targets} targets, min {min_activity} activities per target")

    batch_size = default_params["--batch_size"]
    grad_batches = default_params["--accumulate_grad_batches"]
    effective_batch = int(batch_size) * int(grad_batches)
    print(f"💾 Batch size: {batch_size} × {grad_batches} = {effective_batch} effective")
    print(f"⚡ Precision: {default_params['--precision']}-bit")
    print(f"🧠 Max sequence length: {default_params['--max_length']}")
    print(f"👥 Workers: {default_params['--num_workers']}")
    
    mlm_weight = default_params["--mlm_loss_weight"]
    cls_weight = default_params["--classification_loss_weight"]
    print(f"⚖️  Loss weights: MLM={mlm_weight}, Classification={cls_weight}")
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