#!/usr/bin/env python3
"""
Create sample SMILES data for testing cross-environment MLM training

This script generates a small dataset of diverse molecules for testing purposes.
"""

import pandas as pd
from pathlib import Path
import random


def main():
    """Create sample SMILES dataset"""

    # Sample molecules with various structural features
    sample_smiles = [
        # Simple molecules
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "C",  # Methane
        "CC",  # Ethane
        "CCC",  # Propane
        "CCCC",  # Butane
        "CCCCC",  # Pentane
        # Aromatics
        "c1ccccc1",  # Benzene
        "Cc1ccccc1",  # Toluene
        "c1ccc(cc1)O",  # Phenol
        "c1ccc(cc1)N",  # Aniline
        "c1ccc(cc1)C(=O)O",  # Benzoic acid
        # Heterocycles
        "c1ccncc1",  # Pyridine
        "c1coccn1",  # Oxazole
        "c1csccn1",  # Thiazole
        "c1cnccn1",  # Pyrimidine
        "c1cncnc1",  # Pyrazine
        # Drug-like molecules
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)NCC(c1ccc(cc1)O)O",  # Isoproterenol
        "CCN(CC)CCNC(=O)c1cc(ccc1OC)N",  # Procainamide
        # Natural products
        "CC1=CC2=C(C=C1)N=C3C=CC=CC3=N2",  # Phenazine
        "COc1cc(ccc1O)CCN",  # Tyramine
        "CC(C)(C)c1ccc(cc1)O",  # 4-tert-butylphenol
        "c1ccc2c(c1)c(cn2)CCN",  # Tryptamine
        # Diverse structures
        "CC(C)CC(=O)O",  # Isobutyric acid
        "CCCCCCCCCC(=O)O",  # Decanoic acid
        "CC(C)C(C)(C)C",  # 2,2-dimethylpentane
        "CC1CCCCC1",  # Methylcyclohexane
        "C1CCCCC1O",  # Cyclohexanol
        "C1=CC=C2C=CC=CC2=C1",  # Naphthalene
        # Functional groups
        "CCOC(=O)C",  # Ethyl acetate
        "CC(=O)NC",  # N-methylacetamide
        "CCS",  # Ethanethiol
        "CCN",  # Ethylamine
        "CC(C)O",  # Isopropanol
        "CC(=O)C",  # Acetone
        "CCO[CH2]",  # Diethyl ether (corrected)
        "CCOCC",  # Diethyl ether
        # Complex molecules
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "CC(=O)OCC(COC(=O)C)OC(=O)C",  # Glycerol triacetate
        "c1ccc(cc1)C(c2ccccc2)N3CCCC3",  # Diphenylpyrrolidine
        "COc1ccc(cc1)c2cc(c(c(c2)OC)O)OC",  # Sinapic acid derivative
        # Additional diversity
        "C#C",  # Acetylene
        "C=C",  # Ethene
        "CC=C",  # Propene
        "C#N",  # Hydrogen cyanide
        "CC#N",  # Acetonitrile
        "C=O",  # Formaldehyde
        "CC=O",  # Acetaldehyde
        "CCF",  # Fluoroethane
        "CCCl",  # Chloropropane
        "CCBr",  # Bromoethane
        # Polycyclics
        "C1CC2CCC1C2",  # Norbornane
        "C1=CC=C2C=CC=CC2=C1",  # Naphthalene (duplicate for frequency)
        "c1ccc2cc3ccccc3cc2c1",  # Anthracene
        "c1ccc2c(c1)ccc3c2cccc3",  # Phenanthrene
        # Bioactive molecules
        "NC(=O)c1ccncc1",  # Nicotinamide
        "c1ccc(cc1)S(=O)(=O)Nc2ncccn2",  # Sulfamethylpyrimidine
        "CC(=O)N",  # Acetamide
        "CNC(=O)C",  # N-methylacetamide (duplicate)
        # Steroids and terpenes
        "CC(C)CCCC(C)C",  # 2,6-dimethylheptane
        "CC1=CCC(CC1)C(C)C",  # Limonene-like
        "CC(C)CC=C",  # 3-methyl-1-butene
        # Pharmaceuticals
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC1COc2c(O1)ccc(c2)CC(C(=O)O)N",  # DOPA derivative
        "Clc1ccc(cc1)C(c2ccccc2)N3CCCC3",  # Chlorinated diphenylpyrrolidine
        # Diverse ring systems
        "C1CC1",  # Cyclopropane
        "C1CCC1",  # Cyclobutane
        "C1CCCC1",  # Cyclopentane
        "C1CCCCC1",  # Cyclohexane (duplicate)
        "C1CCCCCC1",  # Cycloheptane
        "C1CCCCCCC1",  # Cyclooctane
        # Spiro compounds
        "C12CCC(CC1)CC2",  # Spirodecane-like
        "C1CC2(CCC1)CCC2",  # Spiro compound
        # Bridged compounds
        "C1CC2CCC(C1)C2",  # Bicyclic
        "C1CC2CCC1CC2",  # Another bicyclic
        # Macrocycles (simplified)
        "CCCCCCCCCCCC",  # Dodecane (linear)
        "CCCCCCCCCCC",  # Undecane
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_smiles = []
    for smiles in sample_smiles:
        if smiles not in seen:
            seen.add(smiles)
            unique_smiles.append(smiles)

    # Create additional random variations by duplicating some molecules
    # This ensures we have enough data for training
    extended_smiles = unique_smiles.copy()

    # Add more instances of diverse molecules
    for _ in range(200):  # Add 200 more samples
        extended_smiles.append(random.choice(unique_smiles))

    # Shuffle the dataset
    random.shuffle(extended_smiles)

    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Save as different formats
    print(f"Creating sample dataset with {len(extended_smiles)} molecules...")

    # 1. Simple text file (one SMILES per line)
    with open(output_dir / "sample_molecules.txt", "w") as f:
        for smiles in extended_smiles:
            f.write(f"{smiles}\n")

    # 2. CSV file
    df = pd.DataFrame({"smiles": extended_smiles})
    df.to_csv(output_dir / "sample_molecules.csv", index=False)

    # 3. Parquet file
    df.to_parquet(output_dir / "sample_molecules.parquet", index=False)

    # Create train/val/test splits
    total_size = len(extended_smiles)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_smiles = extended_smiles[:train_size]
    val_smiles = extended_smiles[train_size : train_size + val_size]
    test_smiles = extended_smiles[train_size + val_size :]

    # Save splits
    with open(output_dir / "train_molecules.txt", "w") as f:
        for smiles in train_smiles:
            f.write(f"{smiles}\n")

    with open(output_dir / "val_molecules.txt", "w") as f:
        for smiles in val_smiles:
            f.write(f"{smiles}\n")

    with open(output_dir / "test_molecules.txt", "w") as f:
        for smiles in test_smiles:
            f.write(f"{smiles}\n")

    print(f"Dataset created successfully!")
    print(f"  Total molecules: {total_size}")
    print(f"  Unique molecules: {len(unique_smiles)}")
    print(f"  Training: {len(train_smiles)}")
    print(f"  Validation: {len(val_smiles)}")
    print(f"  Test: {len(test_smiles)}")
    print(f"  Files saved in: {output_dir}")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    main()
