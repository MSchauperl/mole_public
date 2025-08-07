#!/usr/bin/env python3
"""
ChemBL MLM training test configuration (1000 samples)

This script provides a minimal configuration for very rapid testing/debugging
with only 1000 ChemBL samples. Perfect for quick iterations and debugging.
"""

from chembl_config_base import TestConfig, run_chembl_training


def main():
    """Main function to run the test ChemBL training script."""
    run_chembl_training(
        config_class=TestConfig,
        title="Starting MolE ChemBL MLM Training (RAPID TEST)"
    )


if __name__ == "__main__":
    main() 