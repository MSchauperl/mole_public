#!/usr/bin/env python3
"""
Test script to verify functional hash calculation works correctly.
"""

from rdkit import Chem
from rdkit.Chem import AllChem


def test_functional_hash():
    """Test functional vs structural hash calculation."""

    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol - simple molecule
        "c1ccccc1",  # Benzene - aromatic
        "CCN(C)C",  # Dimethylamine - nitrogen functional group
    ]

    print("Testing Functional vs Structural Hash Calculation")
    print("=" * 55)

    for smiles in test_smiles:
        print(f"\nMolecule: {smiles}")
        print("-" * 40)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  Invalid SMILES")
            continue

        # Get structural hashes (radius 0)
        structural_info = {}
        AllChem.GetMorganFingerprint(mol, 0, bitInfo=structural_info)

        # Get functional hashes (radius 0 with features)
        functional_info = {}
        AllChem.GetMorganFingerprint(mol, 0, useFeatures=True, bitInfo=functional_info)

        print("  Structural environments (radius 0, no features):")
        for env_hash, atom_lists in structural_info.items():
            for atom_idx, _ in atom_lists:
                atom = mol.GetAtomWithIdx(atom_idx)
                element = atom.GetSymbol()
                print(f"    Atom {atom_idx} ({element}): {env_hash}")

        print("  Functional environments (radius 0, with features):")
        functional_dict = {}
        for env_hash, atom_lists in functional_info.items():
            for atom_idx, _ in atom_lists:
                functional_dict[atom_idx] = env_hash

        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            element = atom.GetSymbol()
            func_hash = functional_dict.get(atom_idx, "N/A")
            print(f"    Atom {atom_idx} ({element}): {func_hash}")


if __name__ == "__main__":
    test_functional_hash()
