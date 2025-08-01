#!/usr/bin/env python3
"""
Multi-GPU optimized script for cross-environment MLM training on GuacaMol dataset

This script automatically scales parameters based on the number of GPUs available.
It can be used with 1, 2, 4, or 8 GPUs.
"""

import subprocess
import sys
from pathlib import Path


def get_multi_gpu_config(num_gpus=2):
    """Get optimized configuration for multi-GPU training"""
    
    # Base configuration for Tesla T4
    base_config = {
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
        # Hardware configuration
        "--gpus": str(num_gpus),
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
    
    # Scale parameters based on number of GPUs
    if num_gpus == 1:
        # Single GPU Tesla T4 optimized
        base_config.update({
            "--batch_size": "32",
            "--num_workers": "8",
            "--accumulate_grad_batches": "4",
            "--max_length": "256",
        })
    elif num_gpus == 2:
        # Dual GPU optimized - using same params as single GPU
        base_config.update({
            "--batch_size": "32",  # 32 per GPU = 64 total
            "--num_workers": "6",   # 6 per GPU = 12 total
            "--accumulate_grad_batches": "4",  # Effective batch = 64 * 4 = 256
            "--max_length": "256",
        })
    elif num_gpus == 4:
        # Quad GPU optimized
        base_config.update({
            "--batch_size": "16",  # 16 per GPU = 64 total
            "--num_workers": "8",   # 8 per GPU = 32 total
            "--accumulate_grad_batches": "2",  # Effective batch = 64 * 2 = 128
            "--max_length": "256",
        })
    elif num_gpus == 8:
        # 8 GPU optimized
        base_config.update({
            "--batch_size": "12",  # 12 per GPU = 96 total
            "--num_workers": "8",   # 8 per GPU = 64 total
            "--accumulate_grad_batches": "1",  # Effective batch = 96 * 1 = 96
            "--max_length": "256",
        })
    else:
        # Default scaling for other GPU counts
        batch_per_gpu = max(8, 32 // num_gpus)
        workers_per_gpu = min(8, max(2, 4 * num_gpus // 2))
        accumulation = max(1, 128 // (batch_per_gpu * num_gpus))
        
        base_config.update({
            "--batch_size": str(batch_per_gpu),
            "--num_workers": str(workers_per_gpu),
            "--accumulate_grad_batches": str(accumulation),
            "--max_length": "256",
        })
    
    return base_config


def main():
    # Parse number of GPUs from command line or use default
    num_gpus = 2  # Default to 2 GPUs
    
    # Check if --gpus is provided in command line arguments
    if "--gpus" in sys.argv:
        gpu_idx = sys.argv.index("--gpus")
        if gpu_idx + 1 < len(sys.argv):
            try:
                num_gpus = int(sys.argv[gpu_idx + 1])
                # Remove the --gpus argument since we'll add it back
                sys.argv.pop(gpu_idx + 1)  # Remove the value
                sys.argv.pop(gpu_idx)      # Remove the --gpus flag
            except ValueError:
                print("❌ Invalid GPU count specified. Using default: 2")
                num_gpus = 2
    
    # Get optimized parameters for the specified number of GPUs
    default_params = get_multi_gpu_config(num_gpus)

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

    # Calculate effective batch size
    batch_size = int(default_params["--batch_size"])
    grad_batches = int(default_params["--accumulate_grad_batches"])
    effective_batch = batch_size * grad_batches * num_gpus

    print("🚀 Starting Multi-GPU MolE Cross-Environment MLM Training")
    print("=" * 80)
    print("📊 Dataset: GuacaMol (~1.6M molecules)")
    print(f"🎯 Multi-GPU Setup: {num_gpus} GPUs")
    print(f"🏗️  Architecture: {default_params['--hidden_size']} hidden, {default_params['--num_hidden_layers']} layers")
    print(f"💾 Batch per GPU: {batch_size} × {grad_batches} accumulation")
    print(f"📈 Total effective batch: {effective_batch}")
    print(f"👥 Workers per GPU: {default_params['--num_workers']}")
    print(f"⚡ Precision: {default_params['--precision']}-bit")
    print(f"🧠 Max sequence length: {default_params['--max_length']}")
    print(f"📁 Output: {default_params['--output_dir']}")
    
    # Training strategy info
    if num_gpus > 1:
        print(f"🔄 Strategy: Distributed Data Parallel (DDP)")
        print(f"⚡ Expected speedup: ~{num_gpus * 0.8:.1f}x (with communication overhead)")
    
    print("=" * 80)
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