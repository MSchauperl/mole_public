#!/usr/bin/env python3
"""
Example script demonstrating how to create GuacaMol vocabularies.

This script shows different ways to use the vocabulary creation function:
1. Create all vocabularies (structural + functional, radii 0,1,2)
2. Create only structural vocabularies
3. Test with a small subset of molecules
4. Inspect existing vocabularies
"""

import sys
import os

sys.path.append("..")  # Add parent directory to path

from mole.data.create_vocabularies import (
    create_guacamol_vocabularies,
    analyze_vocabulary_differences,
    load_and_inspect_vocabulary,
)


def example_1_create_all_vocabularies():
    """
    Example 1: Create all vocabulary files (structural + functional, radii 0,1,2)
    This will create 6 vocabulary files total.
    """
    print("🚀 Example 1: Creating all GuacaMol vocabularies")
    print("=" * 50)

    vocab_stats = create_guacamol_vocabularies(
        smiles_file="data/guacamol_v1_all.smiles",
        output_dir="mole/data/vocabularies",
        radii=[0, 1, 2],
        use_features_options=[False, True],  # Both structural and functional
        max_molecules=None,  # Process all molecules
        verbose=True,
    )

    # Analyze differences between structural and functional
    analyze_vocabulary_differences(vocab_stats)

    return vocab_stats


def example_2_structural_only():
    """
    Example 2: Create only structural vocabularies (useFeatures=False)
    """
    print("\n🏗️ Example 2: Creating only structural vocabularies")
    print("=" * 50)

    vocab_stats = create_guacamol_vocabularies(
        smiles_file="data/guacamol_v1_all.smiles",
        output_dir="mole/data/vocabularies",
        radii=[0, 1, 2],
        use_features_options=[False],  # Only structural
        verbose=True,
    )

    return vocab_stats


def example_3_test_with_subset():
    """
    Example 3: Test with a small subset of molecules for quick testing
    """
    print("\n🧪 Example 3: Testing with 1000 molecules")
    print("=" * 50)

    vocab_stats = create_guacamol_vocabularies(
        smiles_file="data/guacamol_v1_all.smiles",
        output_dir="test_vocabularies",  # Different output directory
        radii=[0, 1],  # Only radius 0 and 1
        use_features_options=[False, True],
        max_molecules=1000,  # Only process first 1000 molecules
        verbose=True,
    )

    return vocab_stats


def example_4_inspect_vocabularies():
    """
    Example 4: Inspect existing vocabulary files
    """
    print("\n🔍 Example 4: Inspecting vocabulary files")
    print("=" * 50)

    # List of vocabulary files to inspect
    vocab_files = [
        "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        "mole/data/vocabularies/vocabulary_radius0_functional_guacamol_v1.pkl",
        "mole/data/vocabularies/vocabulary_radius2_structural_guacamol_v1.pkl",
    ]

    for vocab_file in vocab_files:
        if os.path.exists(vocab_file):
            print(f"\n📚 Inspecting: {vocab_file}")
            vocab = load_and_inspect_vocabulary(vocab_file)
        else:
            print(f"❌ File not found: {vocab_file}")


def custom_example():
    """
    Custom example: Create vocabularies with specific settings
    """
    print("\n⚙️ Custom Example: Custom vocabulary settings")
    print("=" * 50)

    # Create only radius 0 vocabularies (like the original MolE)
    vocab_stats = create_guacamol_vocabularies(
        smiles_file="data/guacamol_v1_all.smiles",
        output_dir="mole/data/vocabularies",
        radii=[0],  # Only radius 0
        use_features_options=[False],  # Only structural
        verbose=True,
    )

    # Show comparison with existing MolE vocabulary
    existing_vocab_path = (
        "mole/data/vocabularies/vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
    )
    if os.path.exists(existing_vocab_path):
        print(f"\n🔄 Comparing with existing MolE vocabulary:")
        load_and_inspect_vocabulary(existing_vocab_path)

    return vocab_stats


def main():
    """
    Run different examples based on command line argument
    """
    if len(sys.argv) < 2:
        print("Usage: python create_guacamol_vocabularies_example.py [example_number]")
        print("Examples:")
        print("  1 - Create all vocabularies (structural + functional, radii 0,1,2)")
        print("  2 - Create only structural vocabularies")
        print("  3 - Test with 1000 molecules subset")
        print("  4 - Inspect existing vocabularies")
        print("  5 - Custom example (radius 0 only)")
        return

    example_num = sys.argv[1]

    try:
        if example_num == "1":
            example_1_create_all_vocabularies()
        elif example_num == "2":
            example_2_structural_only()
        elif example_num == "3":
            example_3_test_with_subset()
        elif example_num == "4":
            example_4_inspect_vocabularies()
        elif example_num == "5":
            custom_example()
        else:
            print(f"❌ Unknown example number: {example_num}")
            print("Choose from 1, 2, 3, 4, or 5")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(
            "Make sure the GuacaMol SMILES file exists at: data/guacamol_v1_all.smiles"
        )
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
