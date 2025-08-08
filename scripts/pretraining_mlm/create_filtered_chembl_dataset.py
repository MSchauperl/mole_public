#!/usr/bin/env python3
"""
Create Filtered ChemBL Dataset

This script creates a filtered version of the ChemBL dataset that:
1. Selects the 10 targets with the most measurements
2. Only includes compounds that have measurements for at least 3 of these targets
3. Saves the filtered data to new files
"""

import pickle
import numpy as np
from pathlib import Path
from scipy import sparse
import argparse
from typing import List, Tuple


def load_chembl_data(data_dir: Path) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """Load ChemBL data files."""
    print("Loading ChemBL data...")
    
    # Load labels matrix
    labels_path = data_dir / "labelsHard.pckl"
    with open(labels_path, 'rb') as f:
        labels_matrix = pickle.load(f)
    
    # Load target names
    target_names_path = data_dir / "labelsWeakHard.targetNames"
    with open(target_names_path, 'r') as f:
        target_names = f.read().strip().split('\n')
    
    # Load compound names
    compound_names_path = data_dir / "labelsWeakHard.cmpNames"
    if compound_names_path.exists():
        with open(compound_names_path, 'r') as f:
            compound_names = f.read().strip().split('\n')
    else:
        compound_names = [f"compound_{i}" for i in range(labels_matrix.shape[0])]
    
    # Load SMILES
    smiles_path = data_dir / "chembl20Smiles.pckl"
    with open(smiles_path, 'rb') as f:
        smiles_data = pickle.load(f)
    
    print(f"Loaded: {labels_matrix.shape[0]} compounds, {labels_matrix.shape[1]} targets")
    return labels_matrix, target_names, compound_names, smiles_data


def find_top_targets(labels_matrix: sparse.csr_matrix, target_names: List[str], top_k: int = 10) -> Tuple[List[int], List[str]]:
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
        print(f"  {i+1:2d}. {name[:50]:50s} - {total_activity[idx]:6d} measurements")
    
    return top_target_indices.tolist(), top_target_names


def filter_compounds_by_coverage(labels_matrix: sparse.csr_matrix, target_indices: List[int], min_targets: int = 3) -> List[int]:
    """Filter compounds to only include those with measurements for at least min_targets."""
    print(f"Filtering compounds with measurements for at least {min_targets} targets...")
    
    # Extract submatrix for selected targets
    target_matrix = labels_matrix[:, target_indices]
    
    if sparse.issparse(target_matrix):
        target_matrix = target_matrix.toarray()
    
    # Count measurements per compound (non-zero entries)
    measurements_per_compound = (target_matrix != 0).sum(axis=1)
    
    # Find compounds with sufficient coverage
    valid_compounds = np.where(measurements_per_compound >= min_targets)[0]
    
    print(f"Compounds with ≥{min_targets} target measurements: {len(valid_compounds):6d}")
    print(f"Retention rate: {len(valid_compounds)/labels_matrix.shape[0]*100:.1f}%")
    
    return valid_compounds.tolist()


def create_filtered_dataset(
    labels_matrix: sparse.csr_matrix,
    target_names: List[str],
    compound_names: List[str],
    smiles_data: List[str],
    target_indices: List[int],
    compound_indices: List[int],
    output_dir: Path
):
    """Create filtered dataset files."""
    print("Creating filtered dataset...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter labels matrix
    filtered_labels = labels_matrix[compound_indices][:, target_indices]
    
    # Filter other data
    filtered_target_names = [target_names[i] for i in target_indices]
    filtered_compound_names = [compound_names[i] for i in compound_indices]
    filtered_smiles = [smiles_data[i] for i in compound_indices]
    
    # Save filtered labels matrix
    labels_output_path = output_dir / "labelsHard_top10_min3.pckl"
    with open(labels_output_path, 'wb') as f:
        pickle.dump(filtered_labels, f)
    print(f"Saved filtered labels matrix: {labels_output_path}")
    
    # Save filtered target names
    target_names_output_path = output_dir / "targetNames_top10_min3.txt"
    with open(target_names_output_path, 'w') as f:
        f.write('\n'.join(filtered_target_names))
    print(f"Saved filtered target names: {target_names_output_path}")
    
    # Save filtered compound names
    compound_names_output_path = output_dir / "compoundNames_top10_min3.txt"
    with open(compound_names_output_path, 'w') as f:
        f.write('\n'.join(filtered_compound_names))
    print(f"Saved filtered compound names: {compound_names_output_path}")
    
    # Save filtered SMILES
    smiles_output_path = output_dir / "chemblSmiles_top10_min3.pckl"
    with open(smiles_output_path, 'wb') as f:
        pickle.dump(filtered_smiles, f)
    print(f"Saved filtered SMILES: {smiles_output_path}")
    
    # Save metadata
    metadata_path = output_dir / "filtering_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("ChemBL Filtered Dataset Metadata\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Original dataset: {labels_matrix.shape[0]} compounds, {labels_matrix.shape[1]} targets\n")
        f.write(f"Filtered dataset: {len(compound_indices)} compounds, {len(target_indices)} targets\n")
        f.write(f"Retention rate: {len(compound_indices)/labels_matrix.shape[0]*100:.1f}% compounds\n\n")
        f.write("Filtering criteria:\n")
        f.write(f"- Top {len(target_indices)} targets by measurement count\n")
        f.write(f"- Compounds with measurements for ≥3 targets\n\n")
        f.write("Selected targets:\n")
        for i, (idx, name) in enumerate(zip(target_indices, filtered_target_names)):
            if sparse.issparse(labels_matrix):
                original_counts = (labels_matrix[:, idx].toarray().flatten() != 0).sum()
            else:
                original_counts = (labels_matrix[:, idx] != 0).sum()
            
            if sparse.issparse(filtered_labels):
                filtered_counts = (filtered_labels[:, i].toarray().flatten() != 0).sum()
            else:
                filtered_counts = (filtered_labels[:, i] != 0).sum()
            f.write(f"  {i+1:2d}. {name} - Original: {original_counts:6d}, Filtered: {filtered_counts:6d}\n")
    print(f"Saved metadata: {metadata_path}")
    
    print(f"\nFiltered dataset summary:")
    print(f"  Original: {labels_matrix.shape[0]:6d} compounds × {labels_matrix.shape[1]:4d} targets")
    print(f"  Filtered: {len(compound_indices):6d} compounds × {len(target_indices):4d} targets")
    print(f"  Retention: {len(compound_indices)/labels_matrix.shape[0]*100:5.1f}% compounds")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create filtered ChemBL dataset")
    parser.add_argument("--data_dir", type=str, default="data/ChemBl",
                        help="Directory containing ChemBL data files")
    parser.add_argument("--output_dir", type=str, default="data/ChemBl_filtered",
                        help="Output directory for filtered dataset")
    parser.add_argument("--top_targets", type=int, default=10,
                        help="Number of top targets to select")
    parser.add_argument("--min_measurements", type=int, default=3,
                        help="Minimum number of target measurements per compound")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Load data
    labels_matrix, target_names, compound_names, smiles_data = load_chembl_data(data_dir)
    
    # Find top targets
    top_target_indices, top_target_names = find_top_targets(
        labels_matrix, target_names, top_k=args.top_targets
    )
    
    # Filter compounds
    valid_compound_indices = filter_compounds_by_coverage(
        labels_matrix, top_target_indices, min_targets=args.min_measurements
    )
    
    # Create filtered dataset
    create_filtered_dataset(
        labels_matrix=labels_matrix,
        target_names=target_names,
        compound_names=compound_names,
        smiles_data=smiles_data,
        target_indices=top_target_indices,
        compound_indices=valid_compound_indices,
        output_dir=output_dir
    )
    
    print(f"\n✅ Filtered ChemBL dataset created successfully in: {output_dir}")


if __name__ == "__main__":
    main() 