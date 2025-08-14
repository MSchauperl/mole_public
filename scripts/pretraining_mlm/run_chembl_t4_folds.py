#!/usr/bin/env python3
"""
ChemBL T4 Training with Fold-based Splits

Train on the filtered ChemBL dataset using fold-based training/validation splits
with T4-optimized settings. This ensures proper cross-validation and no data leakage.

Usage:
    python run_chembl_t4_folds.py  # Train with MLM + classification
    python run_chembl_t4_folds.py --chembl_only  # Train classification only
"""

import argparse
import warnings
from chembl_config_base import (
    T4ConfigWithFolds,
    T4OnlyConfigWithFolds,
    run_chembl_training,
)

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def main():
    """Main function to run T4 training with fold-based splits."""
    parser = argparse.ArgumentParser(
        description="Train ChemBL model with T4-optimized settings and fold-based splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--chembl_only",
        action="store_true",
        help="Use ChemBL-only configuration (no MLM, classification only)",
    )

    # Parse known args to allow passing through other arguments
    args, unknown_args = parser.parse_known_args()

    # Add unknown args back to sys.argv for the training script
    import sys

    sys.argv = [sys.argv[0]] + unknown_args

    # Choose configuration based on flags
    if args.chembl_only:
        config_class = T4OnlyConfigWithFolds
        title = "ChemBL T4 Training with Fold-based Splits (Classification Only)"
    else:
        config_class = T4ConfigWithFolds
        title = "ChemBL T4 Training with Fold-based Splits (MLM + Classification)"

    # Run training
    run_chembl_training(config_class=config_class, title=title)


if __name__ == "__main__":
    main()
