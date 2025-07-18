#!/usr/bin/env python3
"""
Example script for running Table 1 MolE training.

This script demonstrates how to use the training pipeline for the Table 1 MolE model
on ADMET datasets with different configurations.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_training_example():
    """Run a training example with default settings."""
    print("=" * 60)
    print("🚀 RUNNING TABLE 1 MOL E TRAINING EXAMPLE")
    print("=" * 60)
    
    # Get the path to the training script
    script_path = Path(__file__).parent.parent / "mole" / "cli" / "train_table1_mole.py"
    
    if not script_path.exists():
        print(f"❌ Training script not found at: {script_path}")
        return False
    
    # Basic training command with default settings
    cmd = [
        sys.executable, str(script_path),
        "--data_path", "data/tdc/tdc_table1_datasets.csv",
        "--vocab_path", "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--batch_size", "16",  # Smaller batch size for testing
        "--max_epochs", "5",   # Few epochs for quick testing
        "--gpus", "0",         # Use CPU or GPU 0
        "--output_dir", "outputs/table1_example",
        "--model_name", "table1_example_model",
        "--patience", "3",     # Early stopping patience
        "--val_check_interval", "0.5",  # Check validation more frequently
        "--num_workers", "2",  # Fewer workers for testing
    ]
    
    print(f"📋 Running command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Could not find the training script")
        return False

def run_training_with_pretrained():
    """Run training with pretrained weights."""
    print(f"\n" + "=" * 60)
    print("🔄 RUNNING TABLE 1 MOL E TRAINING WITH PRETRAINED WEIGHTS")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "mole" / "cli" / "train_table1_mole.py"
    
    # Training command with pretrained weights
    cmd = [
        sys.executable, str(script_path),
        "--data_path", "data/tdc/tdc_table1_datasets.csv",
        "--vocab_path", "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--pretrained_path", "path/to/pretrained/mole/weights",  # Update this path
        "--freeze_encoder",  # Freeze the pretrained encoder
        "--batch_size", "16",
        "--max_epochs", "10",
        "--gpus", "0",
        "--output_dir", "outputs/table1_pretrained",
        "--model_name", "table1_pretrained_model",
        "--learning_rate", "1e-4",  # Lower learning rate for fine-tuning
        "--patience", "5",
    ]
    
    print(f"📋 Running command with pretrained weights:")
    print(f"   {' '.join(cmd)}")
    print()
    
    print("⚠️  Note: Update the --pretrained_path to point to your pretrained weights")
    print("   This example assumes you have pretrained MolE weights available")
    
    # Uncomment to run (after updating the pretrained path)
    # try:
    #     result = subprocess.run(cmd, check=True, capture_output=False)
    #     print("✅ Training with pretrained weights completed successfully!")
    #     return True
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Training failed with error code: {e.returncode}")
    #     return False

def run_training_with_functional_features():
    """Run training with functional atom environments."""
    print(f"\n" + "=" * 60)
    print("🔬 RUNNING TABLE 1 MOL E TRAINING WITH FUNCTIONAL FEATURES")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "mole" / "cli" / "train_table1_mole.py"
    
    # Training command with functional features
    cmd = [
        sys.executable, str(script_path),
        "--data_path", "data/tdc/tdc_table1_datasets.csv",
        "--vocab_path", "mole/data/vocabularies/vocabulary_radius0_functional_guacamol_v1.pkl",
        "--use_features",  # Use functional features
        "--batch_size", "16",
        "--max_epochs", "5",
        "--gpus", "0",
        "--output_dir", "outputs/table1_functional",
        "--model_name", "table1_functional_model",
        "--patience", "3",
    ]
    
    print(f"📋 Running command with functional features:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training with functional features completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
        return False

def show_available_options():
    """Show available command line options."""
    print(f"\n" + "=" * 60)
    print("📖 AVAILABLE TRAINING OPTIONS")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "mole" / "cli" / "train_table1_mole.py"
    
    if script_path.exists():
        cmd = [sys.executable, str(script_path), "--help"]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("Could not display help. Run the script with --help manually.")
    else:
        print(f"Training script not found at: {script_path}")

def main():
    """Main function to run training examples."""
    print("🧪 TABLE 1 MOL E TRAINING EXAMPLES")
    print("=" * 60)
    
    # Check if data file exists
    data_path = Path("data/tdc/tdc_table1_datasets.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure the Table 1 datasets CSV file is available")
        return
    
    # Check if vocabulary file exists
    vocab_path = Path("mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl")
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found: {vocab_path}")
        print("   Please ensure the atom environment vocabulary file is available")
        return
    
    print("✅ Data files found, proceeding with examples")
    print()
    
    # Run basic training example
    success = run_training_example()
    
    if success:
        print(f"\n" + "=" * 60)
        print("🎉 BASIC TRAINING EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show other examples
        run_training_with_pretrained()
        run_training_with_functional_features()
        
        print(f"\n" + "=" * 60)
        print("📚 ADDITIONAL TRAINING OPTIONS")
        print("=" * 60)
        print("You can also run training with different configurations:")
        print()
        print("🔧 Model Configuration:")
        print("   --hidden_size 512          # Smaller model")
        print("   --num_hidden_layers 6      # Fewer layers")
        print("   --dropout 0.2              # Higher dropout")
        print()
        print("🎯 Training Configuration:")
        print("   --learning_rate 1e-4       # Different learning rate")
        print("   --batch_size 64            # Larger batch size")
        print("   --max_epochs 50            # More epochs")
        print()
        print("⚡ Hardware Configuration:")
        print("   --gpus 2                   # Multiple GPUs")
        print("   --precision 16             # Mixed precision")
        print("   --use_torch_compile        # Model compilation")
        print()
        
        show_available_options()
    else:
        print(f"\n" + "=" * 60)
        print("❌ TRAINING EXAMPLE FAILED")
        print("=" * 60)
        print("Please check the error messages above and ensure:")
        print("1. All required data files are available")
        print("2. Dependencies are properly installed")
        print("3. GPU is available (if using --gpus > 0)")
        print("4. Sufficient disk space for outputs")

if __name__ == "__main__":
    main() 