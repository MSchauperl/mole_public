#!/usr/bin/env python3
"""
ChemBL Filtered Dataset Training Script

Train on the filtered ChemBL dataset containing:
- Top 10 targets with the most measurements
- Only compounds with ≥3 target measurements (84,413 compounds)

This provides a high-quality, focused dataset for efficient training
and better convergence compared to the full sparse dataset.
"""

from chembl_config_base import FilteredChemBLConfig, run_chembl_training


def main():
    """Main training function for filtered ChemBL dataset."""
    run_chembl_training(
        config_class=FilteredChemBLConfig,
        title="ChemBL Filtered Dataset Training (Top 10 Targets)",
    )


if __name__ == "__main__":
    main() 