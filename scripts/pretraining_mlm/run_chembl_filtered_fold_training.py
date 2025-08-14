#!/usr/bin/env python3
"""
ChemBL Filtered Dataset Training with Fold-based Splits

Train on the filtered ChemBL dataset using the fold-based training/validation splits
we created. This ensures proper cross-validation and no data leakage.

Usage:
    python run_chembl_filtered_fold_training.py --fold 0  # Use fold 0 as validation
    python run_chembl_filtered_fold_training.py --fold 1  # Use fold 1 as validation
    python run_chembl_filtered_fold_training.py --fold 2  # Use fold 2 as validation
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def run_fold_training(fold: int, output_suffix: str = None):
    """Run training for a specific fold."""

    # Determine output directory
    if output_suffix is None:
        output_suffix = f"fold{fold}"

    output_dir = f"outputs/chembl_filtered_{output_suffix}"

    # Build command
    cmd = [
        sys.executable,
        "mole/cli/train_chembl_mlm.py",
        # Data paths (filtered dataset)
        "--chembl_smiles_path",
        "data/ChemBl_filtered/chemblSmiles_top10_min3.pckl",
        "--chembl_labels_path",
        "data/ChemBl_filtered/labelsHard_top10_min3.pckl",
        "--chembl_target_names_path",
        "data/ChemBl_filtered/targetNames_top10_min3.txt",
        "--chembl_compound_names_path",
        "data/ChemBl_filtered/compoundNames_top10_min3.txt",
        # Use fold-based splits
        "--split_indices_path",
        "data/ChemBl_filtered/splits_top10_min3.pckl",
        # Vocabularies
        "--input_vocab",
        "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "--target_vocab",
        "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        # Model configuration
        "--hidden_size",
        "768",
        "--num_hidden_layers",
        "12",
        "--num_attention_heads",
        "12",
        "--intermediate_size",
        "3072",
        "--dropout",
        "0.1",
        "--classifier_dropout",
        "0.1",
        # Training configuration
        "--batch_size",
        "16",
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "0.01",
        "--warmup_steps",
        "2000",
        "--max_epochs",
        "50",
        "--patience",
        "8",
        "--val_check_interval",
        "0.2",
        "--accumulate_grad_batches",
        "4",
        "--gradient_clip_val",
        "1.0",
        # Loss weights
        "--mlm_loss_weight",
        "0.2",
        "--classification_loss_weight",
        "1.0",
        # Hardware
        "--gpus",
        "1",
        "--num_workers",
        "8",
        "--precision",
        "16",
        # Sequence configuration
        "--max_length",
        "512",
        "--input_radius",
        "0",
        "--target_radius",
        "1",
        "--target_use_features",
        # Output
        "--output_dir",
        output_dir,
        "--model_name",
        f"chembl_filtered_fold{fold}",
        # Logging
        "--log_predictions",
        "--log_target_metrics",
        "--max_targets_to_log",
        "10",
        # Seed for reproducibility
        "--seed",
        "42",
    ]

    print(f"🚀 Starting training for fold {fold}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Command: {' '.join(cmd[:10])}...")
    print("=" * 80)

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Training completed successfully for fold {fold}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for fold {fold}")
        print(f"Error code: {e.returncode}")
        return False


def run_cross_validation():
    """Run training for all folds to perform cross-validation."""
    print("🔄 Running cross-validation training for all folds...")
    print("=" * 80)

    results = {}
    for fold in range(3):
        print(f"\n📊 Training fold {fold}...")
        success = run_fold_training(fold)
        results[fold] = success

        if not success:
            print(f"⚠️  Fold {fold} failed, but continuing with other folds...")

    # Summary
    print("\n" + "=" * 80)
    print("📈 Cross-validation training summary:")
    successful_folds = [fold for fold, success in results.items() if success]
    failed_folds = [fold for fold, success in results.items() if not success]

    if successful_folds:
        print(f"✅ Successful folds: {successful_folds}")
    if failed_folds:
        print(f"❌ Failed folds: {failed_folds}")

    print(f"📊 Success rate: {len(successful_folds)}/3 folds")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train on filtered ChemBL dataset with fold-based splits"
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2],
        help="Which fold to use as validation set (0, 1, or 2)",
    )
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        help="Run training for all folds (cross-validation)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Custom suffix for output directory",
    )

    args = parser.parse_args()

    # Check if filtered dataset exists
    filtered_dir = Path("data/ChemBl_filtered")
    if not filtered_dir.exists():
        print("❌ Error: Filtered dataset not found!")
        print("Please run create_filtered_chembl_dataset_with_folds.py first.")
        return 1

    # Check if split indices exist
    splits_path = filtered_dir / "splits_top10_min3.pckl"
    if not splits_path.exists():
        print("❌ Error: Split indices not found!")
        print("Please run create_filtered_chembl_dataset_with_folds.py first.")
        return 1

    if args.cross_validation:
        # Run cross-validation
        results = run_cross_validation()
        return 0 if any(results.values()) else 1
    elif args.fold is not None:
        # Run single fold
        success = run_fold_training(args.fold, args.output_suffix)
        return 0 if success else 1
    else:
        print("❌ Error: Please specify either --fold or --cross_validation")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
