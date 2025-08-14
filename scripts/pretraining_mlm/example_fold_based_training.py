#!/usr/bin/env python3
"""
Example: Fold-based Training for ChemBL Dataset

This script demonstrates how to use the fold-based training configurations
for the ChemBL dataset. It shows different training scenarios and provides
examples for cross-validation.

Usage:
    python example_fold_based_training.py --show_examples  # Show usage examples
    python example_fold_based_training.py --run_test       # Run a quick test
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def show_training_examples():
    """Show examples of how to use fold-based training."""
    print("🚀 ChemBL Fold-based Training Examples")
    print("=" * 60)
    print()

    print("1. Basic T4 Training with Fold-based Splits:")
    print("   python scripts/pretraining_mlm/run_chembl_t4_folds.py")
    print("   # Trains MLM + classification on filtered dataset with fold splits")
    print()

    print("2. ChemBL-only Training with Fold-based Splits:")
    print("   python scripts/pretraining_mlm/run_chembl_t4_folds.py --chembl_only")
    print("   # Trains classification only on filtered dataset with fold splits")
    print()

    print("3. Fine-tuning with Fold-based Splits:")
    print("   python scripts/pretraining_mlm/run_chembl_finetune_t4.py \\")
    print("       --pretrained_path outputs/chembl_mlm/best_model.ckpt \\")
    print("       --use_folds --freeze_encoder")
    print("   # Fine-tunes with frozen encoder using fold-based splits")
    print()

    print("4. Unsupervised Pretraining with Checkpoint Resumption:")
    print(
        "   python scripts/pretraining_mlm/run_unsupervised_pretraining_a100_2gpu.py \\"
    )
    print(
        "       --gpus 2 --resume_from_checkpoint outputs/guacamol_unsupervised/last.ckpt"
    )
    print("   # Resume unsupervised pretraining from checkpoint")
    print()
    print(
        "   python scripts/pretraining_mlm/run_unsupervised_pretraining_a100_2gpu.py \\"
    )
    print(
        "       --gpus 2 --resume_from_checkpoint outputs/guacamol_unsupervised/best_model.ckpt \\"
    )
    print("       --freeze_encoder")
    print("   # Resume with frozen encoder (fine-tuning)")
    print()

    print("5. Cross-validation Training (All Folds):")

    print("5. Cross-validation Training (All Folds):")
    print(
        "   python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --cross_validation"
    )
    print("   # Trains on all 3 folds for complete cross-validation")
    print()

    print("6. Single Fold Training:")
    print(
        "   python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 0"
    )
    print(
        "   python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 1"
    )
    print(
        "   python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 2"
    )
    print("   # Train on specific fold (0, 1, or 2)")
    print()

    print("7. Checkpoint Resumption:")
    print("   python scripts/pretraining_mlm/run_chembl_t4_folds.py \\")
    print("       --resume_from_checkpoint outputs/chembl_mlm_folds/best_model.ckpt")
    print("   # Resume training from checkpoint")
    print()
    print("   python scripts/pretraining_mlm/run_chembl_t4_folds.py \\")
    print("       --resume_from_checkpoint outputs/chembl_mlm_folds/best_model.ckpt \\")
    print("       --freeze_encoder")
    print("   # Resume with frozen encoder (fine-tuning)")
    print()
    print("   python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py \\")
    print(
        "       --fold 0 --resume_from_checkpoint outputs/chembl_filtered_fold0/best_model.ckpt"
    )
    print("   # Resume specific fold training")
    print()

    print("8. Custom Configuration Overrides:")
    print("   python scripts/pretraining_mlm/run_chembl_t4_folds.py \\")
    print("       --batch_size 16 --learning_rate 2e-4 --max_epochs 50")
    print("   # Override default settings")
    print()

    print("📊 Dataset Information:")
    print("   • Filtered dataset: 84,413 compounds with ≥3 target measurements")
    print("   • 10 most active targets from original 1,310")
    print("   • 3-fold cross-validation splits (no data leakage)")
    print("   • Each fold: ~56,275 train, ~28,138 validation")
    print()

    print("🎯 Training Configurations Available:")
    print("   • T4ConfigWithFolds: MLM + classification, T4-optimized")
    print("   • T4OnlyConfigWithFolds: Classification only, T4-optimized")
    print("   • FilteredChemBLConfig: Uses fold splits by default")
    print()


def run_quick_test():
    """Run a quick test to verify fold-based training works."""
    print("🧪 Running Quick Test of Fold-based Training")
    print("=" * 50)

    # Check if filtered dataset exists
    filtered_data_path = Path("data/ChemBl_filtered")
    if not filtered_data_path.exists():
        print("❌ Filtered dataset not found!")
        print(
            "   Please run: python scripts/pretraining_mlm/create_filtered_chembl_dataset_with_folds.py"
        )
        return False

    # Check if split indices exist
    splits_path = Path("data/ChemBl_filtered/splits_top10_min3.pckl")
    if not splits_path.exists():
        print("❌ Split indices not found!")
        print(
            "   Please run: python scripts/pretraining_mlm/create_filtered_chembl_dataset_with_folds.py"
        )
        return False

    print("✅ Filtered dataset and splits found")

    # Test the datamodule with fold-based splits
    try:
        print("🔍 Testing ChemBLDataModule with fold-based splits...")
        result = subprocess.run(
            ["python", "scripts/pretraining_mlm/test_fold_splits.py"],
            capture_output=True,
            text=True,
            check=True,
        )

        print("✅ Fold-based splits test passed!")
        print("Output:")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("❌ Fold-based splits test failed!")
        print("Error output:")
        print(e.stderr)
        return False


def show_dataset_creation():
    """Show how to create the fold-based dataset."""
    print("📦 Creating Fold-based Dataset")
    print("=" * 40)
    print()

    print("1. Create filtered dataset with fold-based splits:")
    print(
        "   python scripts/pretraining_mlm/create_filtered_chembl_dataset_with_folds.py"
    )
    print("   # Creates filtered dataset and saves train/val splits")
    print()

    print("2. Create all fold combinations:")
    print("   python scripts/pretraining_mlm/create_all_fold_combinations.py")
    print("   # Creates datasets for all 3 folds")
    print()

    print("3. Example usage of the created dataset:")
    print("   python scripts/pretraining_mlm/example_usage_fold_splits.py")
    print("   # Demonstrates loading and using the fold-based data")
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Example and test fold-based training for ChemBL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--show_examples", action="store_true", help="Show training examples and usage"
    )

    parser.add_argument(
        "--run_test",
        action="store_true",
        help="Run a quick test to verify fold-based training works",
    )

    parser.add_argument(
        "--show_dataset_creation",
        action="store_true",
        help="Show how to create the fold-based dataset",
    )

    args = parser.parse_args()

    if not any([args.show_examples, args.run_test, args.show_dataset_creation]):
        # Default: show examples
        args.show_examples = True

    if args.show_examples:
        show_training_examples()
        print()

    if args.show_dataset_creation:
        show_dataset_creation()
        print()

    if args.run_test:
        success = run_quick_test()
        if not success:
            sys.exit(1)

    print("🎉 Fold-based training setup complete!")
    print("   Ready to run training with proper cross-validation splits.")


if __name__ == "__main__":
    main()
