#!/usr/bin/env python3
"""
Convenience script to run ChemBL MLM training on ChemBL dataset

This script provides an easy way to start ChemBL pretraining with optimized defaults
for the ChemBL dataset (~456K molecules), specifically tailored for
a Tesla T4 GPU (16 GB memory).
"""

from chembl_config_base import T4Config, run_chembl_training


def main():
    """Main function to run the T4-optimized ChemBL training script."""
    run_chembl_training(
        config_class=T4Config,
        title="Starting MolE ChemBL MLM Training"
    )


if __name__ == "__main__":
    main() 