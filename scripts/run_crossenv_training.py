#!/usr/bin/env python3
"""
Convenience script to run cross-environment MLM training

This script provides an easy way to start training with sensible defaults.
You can modify the parameters below or pass your own arguments.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Default training parameters
    default_params = {
        "--train_data": "data/sample_molecules.txt",  # Replace with your data
        "--input_vocab": "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab": "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        "--output_dir": "outputs/crossenv_mlm_experiment",
        "--model_name": "r0_to_r1_functional",
        # Model configuration
        "--hidden_size": "768",
        "--num_hidden_layers": "12",
        "--num_attention_heads": "12",
        "--intermediate_size": "3072",
        "--dropout": "0.1",
        # Environment configuration
        "--input_radius": "0",
        "--target_radius": "1",
        "--target_use_features": "",  # Flag for functional environments
        # Training configuration
        "--batch_size": "32",
        "--learning_rate": "5e-5",
        "--weight_decay": "0.01",
        "--warmup_steps": "10000",
        "--max_epochs": "50",
        "--validation_split": "0.1",
        # Hardware configuration
        "--gpus": "1",
        "--num_workers": "4",
        "--precision": "32",
        "--accumulate_grad_batches": "1",
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

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
