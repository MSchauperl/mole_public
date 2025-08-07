#!/usr/bin/env python3
"""
Convenience script to run ChemBL-only training (no MLM) on ChemBL dataset

This script provides an easy way to start ChemBL classification pretraining 
WITHOUT masked language modeling, specifically tailored for
a Tesla T4 GPU (16 GB memory).
"""

from chembl_config_base import T4OnlyConfig, run_chembl_training


def main():
    """Main function to run the ChemBL-only training script."""
    run_chembl_training(
        config_class=T4OnlyConfig,
        title="Starting MolE ChemBL-Only Training (NO MLM)"
    )


if __name__ == "__main__":
    main() 