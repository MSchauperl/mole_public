#!/usr/bin/env python3
"""
Unified ChemBL Training Script

A simplified script that allows you to choose:
- Dataset: Full ChemBL, Filtered ChemBL, or Test dataset
- Splits: Random splits or Fold-based splits
- MLM: Enable/disable masked language modeling
- Fine-tuning: Resume from checkpoint with optional encoder freezing

Usage examples:
    # Train on full ChemBL with MLM and random splits
    python run_chembl_unified.py --dataset full --mlm true --splits random

    # Train on filtered ChemBL with classification only and fold-based splits
    python run_chembl_unified.py --dataset filtered --mlm false --splits folds

    # Fine-tune a pretrained model with frozen encoder
    python run_chembl_unified.py --dataset filtered --mlm true --splits folds \
        --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt --freeze_encoder

    # Quick test run (MLM enabled by default)
    python run_chembl_unified.py --dataset test --splits random
"""

import argparse
import sys
import warnings
from chembl_config_base import (
    T4Config,
    T4OnlyConfig,
    T4ConfigWithFolds,
    T4OnlyConfigWithFolds,
    T4FineTuneConfig,
    T4OnlyFineTuneConfig,
    FilteredChemBLConfig,
    FilteredChemBLOnlyConfig,
    TestConfig,
    run_chembl_training,
)

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def get_config_class(dataset: str, mlm: bool, splits: str, is_finetune: bool = False):
    """
    Get the appropriate configuration class based on parameters.
    
    Args:
        dataset: 'full', 'filtered', or 'test'
        mlm: Whether to enable MLM
        splits: 'random' or 'folds'
        is_finetune: Whether this is a fine-tuning run
    
    Returns:
        Configuration class to use
    """
    
    if dataset == "test":
        return TestConfig
    
    if is_finetune:
        if dataset == "full":
            if mlm:
                return T4FineTuneConfig
            else:
                return T4OnlyFineTuneConfig
        else:  # filtered
            if mlm:
                return T4FineTuneConfig  # Will use filtered data via overrides
            else:
                return T4OnlyFineTuneConfig  # Will use filtered data via overrides
    
    # Regular training (not fine-tuning)
    if dataset == "full":
        if splits == "folds":
            if mlm:
                return T4ConfigWithFolds
            else:
                return T4OnlyConfigWithFolds
        else:  # random splits
            if mlm:
                return T4Config
            else:
                return T4OnlyConfig
    else:  # filtered dataset
        if mlm:
            return FilteredChemBLConfig
        else:
            return FilteredChemBLOnlyConfig


def get_title(dataset: str, mlm: bool, splits: str, is_finetune: bool = False, freeze_encoder: bool = False):
    """Generate a descriptive title for the training run."""
    
    # Dataset description
    if dataset == "test":
        dataset_desc = "Test Dataset (1K samples)"
    elif dataset == "filtered":
        dataset_desc = "Filtered ChemBL (Top 10 targets, 84K compounds)"
    else:
        dataset_desc = "Full ChemBL (sampled)"
    
    # Task description
    if mlm:
        task_desc = "MLM + Classification"
    else:
        task_desc = "Classification Only"
    
    # Split description
    if splits == "folds":
        split_desc = "Fold-based Splits"
    else:
        split_desc = "Random Splits"
    
    # Mode description
    if is_finetune:
        mode_desc = "Fine-tuning"
        if freeze_encoder:
            mode_desc += " (Frozen Encoder)"
        else:
            mode_desc += " (Trainable Encoder)"
    else:
        mode_desc = "Training"
    
    return f"MolE ChemBL {mode_desc} - {dataset_desc} - {task_desc} - {split_desc}"


def main():
    """Main function for unified ChemBL training."""
    parser = argparse.ArgumentParser(
        description="Unified ChemBL training script with flexible configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["full", "filtered", "test"],
        default="filtered",
        help="Dataset to use: 'full' (sampled ChemBL), 'filtered' (top 10 targets), or 'test' (1K samples)",
    )

    # MLM configuration
    parser.add_argument(
        "--mlm",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable/disable masked language modeling: 'true' (default) or 'false' (classification only)",
    )

    # Split configuration
    parser.add_argument(
        "--splits",
        type=str,
        choices=["random", "folds"],
        default="folds",
        help="Split type: 'random' or 'folds' (fold-based cross-validation)",
    )

    # Fine-tuning options
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (for fine-tuning)",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder layers when resuming from checkpoint (for fine-tuning)",
    )

    # Parse known args to allow passing through other arguments
    args, unknown_args = parser.parse_known_args()

    # Add unknown args back to sys.argv for the training script
    sys.argv = [sys.argv[0]] + unknown_args

    # Validate arguments
    if args.dataset == "test" and args.splits == "folds":
        print("⚠️  Warning: Test dataset doesn't support fold-based splits. Using random splits.")
        args.splits = "random"
    
    if args.freeze_encoder and not args.resume_from_checkpoint:
        print("⚠️  Warning: --freeze_encoder specified but no checkpoint provided. Ignoring freeze_encoder.")
        args.freeze_encoder = False

    # Convert MLM string to boolean
    mlm_enabled = args.mlm.lower() == "true"

    # Determine if this is a fine-tuning run
    is_finetune = args.resume_from_checkpoint is not None

    # Get configuration class
    config_class = get_config_class(
        dataset=args.dataset,
        mlm=mlm_enabled,
        splits=args.splits,
        is_finetune=is_finetune
    )

    # Generate title
    title = get_title(
        dataset=args.dataset,
        mlm=mlm_enabled,
        splits=args.splits,
        is_finetune=is_finetune,
        freeze_encoder=args.freeze_encoder
    )

    # Print configuration summary
    print("🚀 Unified ChemBL Training Configuration")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 MLM: {'Enabled' if mlm_enabled else 'Disabled'}")
    print(f"✂️  Splits: {args.splits}")
    print(f"🔄 Mode: {'Fine-tuning' if is_finetune else 'Training'}")
    if is_finetune:
        print(f"📁 Checkpoint: {args.resume_from_checkpoint}")
        print(f"🧊 Encoder: {'Frozen' if args.freeze_encoder else 'Trainable'}")
    print(f"⚙️  Config: {config_class.__name__}")
    print("=" * 60)

    # Run training
    run_chembl_training(
        config_class=config_class,
        title=title,
        pretrained_path=args.resume_from_checkpoint,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
