#!/usr/bin/env python3
"""
Create all fold combinations for cross-validation

This script creates filtered datasets for all possible fold combinations
(using each fold as validation set) to enable proper cross-validation.
"""

import subprocess
import sys
from pathlib import Path


def create_all_fold_combinations():
    """Create filtered datasets for all fold combinations."""
    script_path = Path(
        "scripts/pretraining_mlm/create_filtered_chembl_dataset_with_folds.py"
    )

    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return

    print("Creating filtered datasets for all fold combinations...")
    print("=" * 60)

    for fold in range(3):
        print(f"\n🔄 Creating dataset with fold {fold} as validation set...")

        # Create output directory for this fold
        output_dir = f"data/ChemBl_filtered_fold{fold}"

        # Run the script
        cmd = [
            sys.executable,
            str(script_path),
            "--test_fold",
            str(fold),
            "--output_dir",
            output_dir,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully created dataset in {output_dir}")

            # Show summary from output
            lines = result.stdout.split("\n")
            for line in lines:
                if "Filtered dataset summary:" in line:
                    print(f"  Summary: {line.strip()}")
                    break

        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating dataset for fold {fold}:")
            print(f"  Error: {e}")
            print(f"  Output: {e.stdout}")
            print(f"  Error: {e.stderr}")

    print("\n🎉 All fold combinations created!")
    print("\nAvailable datasets:")
    for fold in range(3):
        output_dir = f"data/ChemBl_filtered_fold{fold}"
        if Path(output_dir).exists():
            print(f"  - {output_dir}")
        else:
            print(f"  - {output_dir} (failed to create)")


def show_usage_examples():
    """Show examples of how to use the different fold datasets."""
    print("\n📖 Usage Examples:")
    print("=" * 40)

    print("\n1. Load a specific fold dataset:")
    print("   from pathlib import Path")
    print("   import pickle")
    print("   ")
    print("   # Load fold 0 dataset")
    print("   data_dir = Path('data/ChemBl_filtered_fold0')")
    print("   with open(data_dir / 'splits_top10_min3.pckl', 'rb') as f:")
    print("       splits = pickle.load(f)")
    print("   ")
    print("   print(f'Training: {len(splits[\"train\"])} compounds')")
    print("   print(f'Validation: {len(splits[\"val\"])} compounds')")

    print("\n2. Cross-validation training loop:")
    print("   for fold in range(3):")
    print("       data_dir = Path(f'data/ChemBl_filtered_fold{fold}')")
    print("       # Load data for this fold")
    print("       # Train model")
    print("       # Evaluate on validation set")
    print("       # Save results")

    print("\n3. Ensemble training:")
    print("   models = []")
    print("   for fold in range(3):")
    print("       # Train model on fold {fold}")
    print("       # models.append(trained_model)")
    print("   ")
    print("   # Use ensemble for predictions")


if __name__ == "__main__":
    create_all_fold_combinations()
    show_usage_examples()
