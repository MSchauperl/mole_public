#!/usr/bin/env python3
"""
Example usage of fold-based splits for training

This script demonstrates how to load and use the fold-based splits
created by create_filtered_chembl_dataset_with_folds.py
"""

import pickle
import numpy as np
from pathlib import Path


def load_filtered_dataset_with_splits(data_dir: Path):
    """Load the filtered dataset and split information."""
    print("Loading filtered dataset with splits...")

    # Load split information
    splits_path = data_dir / "splits_top10_min3.pckl"
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    # Load filtered labels matrix
    labels_path = data_dir / "labelsHard_top10_min3.pckl"
    with open(labels_path, "rb") as f:
        labels_matrix = pickle.load(f)

    # Load target names
    target_names_path = data_dir / "targetNames_top10_min3.txt"
    with open(target_names_path, "r") as f:
        target_names = f.read().strip().split("\n")

    # Load compound names
    compound_names_path = data_dir / "compoundNames_top10_min3.txt"
    with open(compound_names_path, "r") as f:
        compound_names = f.read().strip().split("\n")

    # Load SMILES
    smiles_path = data_dir / "chemblSmiles_top10_min3.pckl"
    with open(smiles_path, "rb") as f:
        smiles_data = pickle.load(f)

    print("Dataset loaded:")
    print(f"  Total compounds: {len(compound_names)}")
    print(f"  Total targets: {len(target_names)}")
    print(f"  Training compounds: {len(splits['train'])}")
    print(f"  Validation compounds: {len(splits['val'])}")

    return {
        "labels_matrix": labels_matrix,
        "target_names": target_names,
        "compound_names": compound_names,
        "smiles_data": smiles_data,
        "splits": splits,
    }


def create_training_validation_data(data_dict):
    """Create training and validation datasets from the splits."""
    print("\nCreating training and validation datasets...")

    labels_matrix = data_dict["labels_matrix"]
    splits = data_dict["splits"]

    # Extract training data
    train_labels = labels_matrix[splits["train"]]
    train_smiles = [data_dict["smiles_data"][i] for i in splits["train"]]
    train_compounds = [data_dict["compound_names"][i] for i in splits["train"]]

    # Extract validation data
    val_labels = labels_matrix[splits["val"]]
    val_smiles = [data_dict["smiles_data"][i] for i in splits["val"]]
    val_compounds = [data_dict["compound_names"][i] for i in splits["val"]]

    print("Training dataset:")
    print(f"  Compounds: {len(train_compounds)}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Active measurements: {(train_labels == 1).sum()}")
    print(f"  Inactive measurements: {(train_labels == -1).sum()}")

    print("\nValidation dataset:")
    print(f"  Compounds: {len(val_compounds)}")
    print(f"  Labels shape: {val_labels.shape}")
    print(f"  Active measurements: {(val_labels == 1).sum()}")
    print(f"  Inactive measurements: {(val_labels == -1).sum()}")

    return {
        "train": {
            "labels": train_labels,
            "smiles": train_smiles,
            "compounds": train_compounds,
            "indices": splits["train"],
        },
        "val": {
            "labels": val_labels,
            "smiles": val_smiles,
            "compounds": val_compounds,
            "indices": splits["val"],
        },
    }


def example_training_loop(train_data, val_data, target_names):
    """Example of how to use the data in a training loop."""
    print("\nExample training loop structure:")
    print(f"  Training on {len(train_data['smiles'])} compounds")
    print(f"  Validating on {len(val_data['smiles'])} compounds")
    print(f"  Predicting {len(target_names)} targets")

    # Example: iterate through training data
    print("\nExample training iteration:")
    for i in range(min(5, len(train_data["smiles"]))):
        smiles = train_data["smiles"][i]
        labels = train_data["labels"][i]
        compound_name = train_data["compounds"][i]

        # Count active/inactive measurements for this compound
        labels_1d = np.atleast_1d(
            labels.toarray().flatten() if hasattr(labels, "toarray") else labels
        )
        active_targets = np.where(labels_1d == 1)[0]
        inactive_targets = np.where(labels_1d == -1)[0]

        print(f"  Compound {i}: {compound_name}")
        print(f"    SMILES: {smiles[:50]}...")
        print(f"    Active targets: {len(active_targets)}")
        print(f"    Inactive targets: {len(inactive_targets)}")
        if len(active_targets) > 0:
            print(f"    Example active target: {target_names[active_targets[0]]}")
        print()


def main():
    """Main function demonstrating usage."""
    data_dir = Path("data/ChemBl_filtered")

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        print("Please run create_filtered_chembl_dataset_with_folds.py first.")
        return

    # Load the dataset
    data_dict = load_filtered_dataset_with_splits(data_dir)

    # Create training/validation splits
    train_val_data = create_training_validation_data(data_dict)

    # Example training loop
    example_training_loop(
        train_val_data["train"], train_val_data["val"], data_dict["target_names"]
    )

    print("✅ Example completed successfully!")


if __name__ == "__main__":
    main()
