#!/usr/bin/env python3
"""
Create Filtered ChemBL Dataset with Fold-based Splits (CSV Output)

This script creates a filtered version of the ChemBL dataset that:
1. Selects the 10 targets with the most measurements
2. Only includes compounds that have measurements for at least 3 of these targets
3. Uses existing fold information to create training/validation splits
4. Saves the filtered data as CSV files for easy analysis

The dataset uses three-state classification:
- 1: Active compounds
- -1: Inactive compounds  
- 0: Missing/not measured compounds

Output files:
- main_dataset.csv: Complete filtered dataset with all compounds and targets
- train_dataset.csv: Training set compounds  
- val_dataset.csv: Validation set compounds
- metadata.csv: Dataset statistics and filtering information

Note: The 'fold' column in the CSV files contains the original fold assignment
from the ChemBL dataset (fold_0, fold_1, fold_2), not the train/val split.
The train/val split is determined by the --test_fold parameter.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import argparse
from typing import List, Tuple, Dict


def load_chembl_data(
    data_dir: Path,
) -> Tuple[sparse.csr_matrix, List[str], List[str], List[str]]:
    """Load ChemBL data files."""
    print("Loading ChemBL data...")

    # Load labels matrix
    labels_path = data_dir / "labelsHard.pckl"
    with open(labels_path, "rb") as f:
        labels_matrix = pickle.load(f)

    # Load target names
    target_names_path = data_dir / "labelsWeakHard.targetNames"
    with open(target_names_path, "r") as f:
        target_names = f.read().strip().split("\n")

    # Load compound names
    compound_names_path = data_dir / "labelsWeakHard.cmpNames"
    if compound_names_path.exists():
        with open(compound_names_path, "r") as f:
            compound_names = f.read().strip().split("\n")
    else:
        compound_names = [f"compound_{i}" for i in range(labels_matrix.shape[0])]

    # Load SMILES
    smiles_path = data_dir / "chembl20Smiles.pckl"
    with open(smiles_path, "rb") as f:
        smiles_data = pickle.load(f)

    print(
        f"Loaded: {labels_matrix.shape[0]} compounds, {labels_matrix.shape[1]} targets"
    )
    return labels_matrix, target_names, compound_names, smiles_data


def load_fold_information(data_dir: Path) -> List[np.ndarray]:
    """Load fold information from folds0.pckl."""
    print("Loading fold information...")

    folds_path = data_dir / "folds0.pckl"
    with open(folds_path, "rb") as f:
        folds_data = pickle.load(f)

    print(f"Loaded {len(folds_data)} folds:")
    for i, fold in enumerate(folds_data):
        print(f"  Fold {i}: {len(fold)} indices")

    return folds_data


def find_top_targets(
    labels_matrix: sparse.csr_matrix, target_names: List[str], top_k: int = 10
) -> Tuple[List[int], List[str]]:
    """Find the top K targets with the most measurements."""
    print(f"Finding top {top_k} targets by measurement count...")

    # Convert to dense if needed for counting
    if sparse.issparse(labels_matrix):
        matrix = labels_matrix.toarray()
    else:
        matrix = labels_matrix

    # Count total measurements per target (active + inactive, ignore 0s)
    active_counts = (matrix == 1).sum(axis=0)
    inactive_counts = (matrix == -1).sum(axis=0)
    total_activity = active_counts + inactive_counts

    # Get top K targets
    top_target_indices = np.argsort(total_activity)[::-1][:top_k]
    top_target_names = [target_names[i] for i in top_target_indices]

    print(f"Top {top_k} targets:")
    for i, (idx, name) in enumerate(zip(top_target_indices, top_target_names)):
        active = active_counts[idx]
        inactive = inactive_counts[idx]
        total = total_activity[idx]
        print(f"  {i+1:2d}. {name[:50]:50s} - {total:6d} measurements ({active:5d} active, {inactive:5d} inactive)")

    return top_target_indices.tolist(), top_target_names


def filter_compounds_by_coverage(
    labels_matrix: sparse.csr_matrix, target_indices: List[int], min_targets: int = 3
) -> List[int]:
    """Filter compounds to only include those with measurements for at least min_targets."""
    print(
        f"Filtering compounds with measurements for at least {min_targets} targets..."
    )

    # Extract submatrix for selected targets
    target_matrix = labels_matrix[:, target_indices]

    if sparse.issparse(target_matrix):
        target_matrix = target_matrix.toarray()

    # Count measurements per compound (active or inactive, ignore 0s)
    measurements_per_compound = ((target_matrix == 1) | (target_matrix == -1)).sum(axis=1)

    # Find compounds with sufficient coverage
    valid_compounds = np.where(measurements_per_compound >= min_targets)[0]

    print(
        f"Compounds with ≥{min_targets} target measurements: {len(valid_compounds):6d}"
    )
    print(f"Retention rate: {len(valid_compounds)/labels_matrix.shape[0]*100:.1f}%")

    return valid_compounds.tolist()


def create_fold_based_splits(
    folds_data: List[np.ndarray], valid_compound_indices: List[int], test_fold: int = 0
) -> Dict[str, List[int]]:
    """Create training/validation splits based on fold information."""
    print(f"Creating fold-based splits (using fold {test_fold} as validation)...")

    # Create a mapping from original indices to filtered indices
    original_to_filtered = {
        orig_idx: filtered_idx
        for filtered_idx, orig_idx in enumerate(valid_compound_indices)
    }

    # Get validation indices (from the specified fold)
    validation_original_indices = folds_data[test_fold]

    # Convert to filtered indices and filter out compounds not in our filtered dataset
    validation_filtered_indices = []
    for orig_idx in validation_original_indices:
        if orig_idx in original_to_filtered:
            validation_filtered_indices.append(original_to_filtered[orig_idx])

    # Get training indices (from all other folds)
    training_original_indices = []
    for i, fold in enumerate(folds_data):
        if i != test_fold:
            training_original_indices.extend(fold)

    # Convert to filtered indices and filter out compounds not in our filtered dataset
    training_filtered_indices = []
    for orig_idx in training_original_indices:
        if orig_idx in original_to_filtered:
            training_filtered_indices.append(original_to_filtered[orig_idx])

    # Remove duplicates from training indices
    training_filtered_indices = list(set(training_filtered_indices))

    print("Split sizes:")
    print(f"  Training: {len(training_filtered_indices)} compounds")
    print(f"  Validation: {len(validation_filtered_indices)} compounds")
    print(
        f"  Total: {len(training_filtered_indices) + len(validation_filtered_indices)} compounds"
    )

    return {"train": training_filtered_indices, "val": validation_filtered_indices}


def create_csv_dataset(
    labels_matrix: sparse.csr_matrix,
    target_names: List[str],
    compound_names: List[str],
    smiles_data: List[str],
    target_indices: List[int],
    compound_indices: List[int],
    splits: Dict[str, List[int]],
    folds_data: List[np.ndarray],
    output_dir: Path,
):
    """Create filtered dataset as CSV files."""
    print("Creating CSV dataset files...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter labels matrix
    filtered_labels = labels_matrix[compound_indices][:, target_indices]

    # Convert to dense array if sparse
    if sparse.issparse(filtered_labels):
        filtered_labels = filtered_labels.toarray()

    # Filter other data
    filtered_target_names = [target_names[i] for i in target_indices]
    filtered_compound_names = [compound_names[i] for i in compound_indices]
    filtered_smiles = [smiles_data[i] for i in compound_indices]

    # Create main dataset DataFrame
    print("Creating main dataset CSV...")
    
    # Create fold assignments based on the original fold information
    fold_assignments = []
    for compound_idx in compound_indices:
        # Find which fold this compound belongs to in the original dataset
        fold_found = False
        for fold_idx, fold_indices in enumerate(folds_data):
            if compound_idx in fold_indices:
                fold_assignments.append(f'fold_{fold_idx}')
                fold_found = True
                break
        if not fold_found:
            # If compound not found in any fold, assign to fold_0
            fold_assignments.append('fold_0')
    
    main_data = {
        'compound_id': compound_indices,
        'compound_name': filtered_compound_names,
        'smiles': filtered_smiles,
        'fold': fold_assignments
    }
    
    # Add target columns
    for i, target_name in enumerate(filtered_target_names):
        main_data[f'activity_{target_name}'] = filtered_labels[:, i]

    main_df = pd.DataFrame(main_data)
    
    # Save main dataset
    main_output_path = output_dir / "chembl_filtered_main_dataset.csv"
    main_df.to_csv(main_output_path, index=False)
    print(f"Saved main dataset: {main_output_path}")

    # Create training dataset
    print("Creating training dataset CSV...")
    train_df = main_df.iloc[splits["train"]].copy()
    train_output_path = output_dir / "chembl_filtered_train_dataset.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"Saved training dataset: {train_output_path}")

    # Create validation dataset
    print("Creating validation dataset CSV...")
    val_df = main_df.iloc[splits["val"]].copy()
    val_output_path = output_dir / "chembl_filtered_val_dataset.csv"
    val_df.to_csv(val_output_path, index=False)
    print(f"Saved validation dataset: {val_output_path}")

    # Create metadata CSV
    print("Creating metadata CSV...")
    metadata_data = []
    
    for i, (idx, name) in enumerate(zip(target_indices, filtered_target_names)):
        # Original dataset statistics
        if sparse.issparse(labels_matrix):
            original_active = (labels_matrix[:, idx].toarray().flatten() == 1).sum()
            original_inactive = (labels_matrix[:, idx].toarray().flatten() == -1).sum()
        else:
            original_active = (labels_matrix[:, idx] == 1).sum()
            original_inactive = (labels_matrix[:, idx] == -1).sum()
        original_total = original_active + original_inactive

        # Filtered dataset statistics
        filtered_active = (filtered_labels[:, i] == 1).sum()
        filtered_inactive = (filtered_labels[:, i] == -1).sum()
        filtered_total = filtered_active + filtered_inactive

        metadata_data.append({
            'target_index': i,
            'target_name': name,
            'original_total_measurements': original_total,
            'original_active_count': original_active,
            'original_inactive_count': original_inactive,
            'original_active_ratio': original_active / original_total if original_total > 0 else 0,
            'filtered_total_measurements': filtered_total,
            'filtered_active_count': filtered_active,
            'filtered_inactive_count': filtered_inactive,
            'filtered_active_ratio': filtered_active / filtered_total if filtered_total > 0 else 0,
            'retention_rate': filtered_total / original_total if original_total > 0 else 0
        })

    metadata_df = pd.DataFrame(metadata_data)
    metadata_output_path = output_dir / "chembl_filtered_metadata.csv"
    metadata_df.to_csv(metadata_output_path, index=False)
    print(f"Saved metadata: {metadata_output_path}")

    # Create summary statistics
    print("Creating summary statistics...")
    summary_data = {
        'metric': [
            'original_compounds',
            'original_targets', 
            'filtered_compounds',
            'filtered_targets',
            'compound_retention_rate',
            'training_compounds',
            'validation_compounds',
            'total_split_compounds'
        ],
        'value': [
            labels_matrix.shape[0],
            labels_matrix.shape[1],
            len(compound_indices),
            len(target_indices),
            len(compound_indices) / labels_matrix.shape[0],
            len(splits['train']),
            len(splits['val']),
            len(splits['train']) + len(splits['val'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = output_dir / "chembl_filtered_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Saved summary: {summary_output_path}")

    # Create split indices files (for compatibility)
    train_indices_path = output_dir / "train_indices.txt"
    with open(train_indices_path, "w") as f:
        f.write("\n".join(map(str, splits["train"])))
    print(f"Saved training indices: {train_indices_path}")

    val_indices_path = output_dir / "val_indices.txt"
    with open(val_indices_path, "w") as f:
        f.write("\n".join(map(str, splits["val"])))
    print(f"Saved validation indices: {val_indices_path}")

    # Print summary
    print("\n📊 Dataset Summary:")
    print("=" * 50)
    print(f"Original dataset: {labels_matrix.shape[0]:,} compounds × {labels_matrix.shape[1]:,} targets")
    print(f"Filtered dataset: {len(compound_indices):,} compounds × {len(target_indices):,} targets")
    print(f"Compound retention: {len(compound_indices)/labels_matrix.shape[0]*100:.1f}%")
    print(f"Training set: {len(splits['train']):,} compounds")
    print(f"Validation set: {len(splits['val']):,} compounds")
    print(f"Total split compounds: {len(splits['train']) + len(splits['val']):,}")
    
    # Show fold distribution
    fold_counts = {}
    for fold_assignment in fold_assignments:
        fold_counts[fold_assignment] = fold_counts.get(fold_assignment, 0) + 1
    
    print(f"\n📁 Fold Distribution:")
    for fold_name in sorted(fold_counts.keys()):
        print(f"  {fold_name}: {fold_counts[fold_name]:,} compounds")
    
    print(f"\n📁 Output files created in: {output_dir}")
    print("  • chembl_filtered_main_dataset.csv - Complete filtered dataset")
    print("  • chembl_filtered_train_dataset.csv - Training set")
    print("  • chembl_filtered_val_dataset.csv - Validation set")
    print("  • chembl_filtered_metadata.csv - Target-level statistics")
    print("  • chembl_filtered_summary.csv - Dataset summary")
    print("  • train_indices.txt - Training compound indices")
    print("  • val_indices.txt - Validation compound indices")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create filtered ChemBL dataset with fold-based splits (CSV output)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ChemBl",
        help="Directory containing ChemBL data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ChemBl_filtered_csv",
        help="Output directory for filtered dataset CSV files",
    )
    parser.add_argument(
        "--top_targets", type=int, default=10, help="Number of top targets to select"
    )
    parser.add_argument(
        "--min_measurements",
        type=int,
        default=3,
        help="Minimum number of target measurements per compound",
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=0,
        help="Which fold to use as validation set (0, 1, or 2)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load data
    labels_matrix, target_names, compound_names, smiles_data = load_chembl_data(
        data_dir
    )

    # Load fold information
    folds_data = load_fold_information(data_dir)

    # Find top targets
    top_target_indices, top_target_names = find_top_targets(
        labels_matrix, target_names, top_k=args.top_targets
    )

    # Filter compounds
    valid_compound_indices = filter_compounds_by_coverage(
        labels_matrix, top_target_indices, min_targets=args.min_measurements
    )

    # Create fold-based splits
    splits = create_fold_based_splits(
        folds_data, valid_compound_indices, test_fold=args.test_fold
    )

    # Create CSV dataset
    create_csv_dataset(
        labels_matrix=labels_matrix,
        target_names=target_names,
        compound_names=compound_names,
        smiles_data=smiles_data,
        target_indices=top_target_indices,
        compound_indices=valid_compound_indices,
        splits=splits,
        folds_data=folds_data,
        output_dir=output_dir,
    )

    print(
        f"\n✅ Filtered ChemBL dataset with CSV output created successfully in: {output_dir}"
    )


if __name__ == "__main__":
    main()
