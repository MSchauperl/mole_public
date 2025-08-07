#!/usr/bin/env python3
"""
ChemBL Filtered Dataset Training Script (ChemBL-Only Mode)

Train on the filtered ChemBL dataset with MLM disabled:
- Top 10 targets with the most measurements
- Only compounds with ≥3 target measurements (84,413 compounds)
- ChemBL classification only (no masked language modeling)

This provides focused multi-target classification training
without the computational overhead of MLM.
"""

from chembl_config_base import FilteredChemBLOnlyConfig, run_chembl_training


def main():
    """Main training function for filtered ChemBL dataset (ChemBL-only)."""
    run_chembl_training(
        config_class=FilteredChemBLOnlyConfig,
        title="ChemBL Filtered Dataset Training (ChemBL-Only Mode)",
    )


if __name__ == "__main__":
    main() 