#!/usr/bin/env python3
"""
Extract Multi-Target Dataset from ChEMBL Metadata Analysis

This script extracts the assays identified in the metadata analysis and creates a multi-target dataset.
It uses the assays that have sufficient active and inactive measurements.

Usage:
    python extract_assays_from_metadata.py --csv chembl_complete_dataset.csv --metadata chembl_target_metadata.csv --output multi_target_dataset.csv
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def get_assays_from_metadata(metadata_path, min_active=10000, min_inactive=10000):
    """
    Extract assay IDs from metadata that meet the minimum criteria.
    
    Args:
        metadata_path (str): Path to chembl_target_metadata.csv
        min_active (int): Minimum number of active compounds required
        min_inactive (int): Minimum number of inactive compounds required
    
    Returns:
        list: List of assay IDs that meet the criteria
    """
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Filter assays that meet the criteria
    filtered_df = df[(df.active_count > min_active) & (df.inactive_count > min_inactive)]
    
    print(f"Found {len(filtered_df)} assays with >{min_active} active and >{min_inactive} inactive compounds")
    
    # Extract assay IDs (remove 'CHEMBL' prefix)
    assay_ids = []
    for target_name in filtered_df.target_name.values:
        if target_name.startswith('CHEMBL'):
            assay_id = target_name
            assay_ids.append(assay_id)
    
    print(f"Assay IDs: {assay_ids}")
    
    # Print details for each assay
    print(f"\n📊 Assay Details:")
    print("=" * 80)
    for _, row in filtered_df.iterrows():
        print(f"{row['target_name']}:")
        print(f"  • Active compounds: {row['active_count']:,}")
        print(f"  • Inactive compounds: {row['inactive_count']:,}")
        print(f"  • Total measurements: {row['total_measurements']:,}")
        print(f"  • Active ratio: {row['active_ratio']:.4f} ({row['active_ratio']*100:.2f}%)")
        print()
    
    return assay_ids


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract multi-target dataset from ChEMBL metadata analysis."
    )
    parser.add_argument(
        "--csv", 
        type=str, 
        required=True,
        help="Path to the input chembl_complete_dataset.csv."
    )
    parser.add_argument(
        "--metadata", 
        type=str, 
        required=True,
        help="Path to the chembl_target_metadata.csv file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save the multi-target dataset CSV."
    )
    parser.add_argument(
        "--min_active", 
        type=int, 
        default=10000,
        help="Minimum number of active compounds required (default: 10000)."
    )
    parser.add_argument(
        "--min_inactive", 
        type=int, 
        default=10000,
        help="Minimum number of inactive compounds required (default: 10000)."
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Number of rows to process at a time (default: 1000)."
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.csv).exists():
        print(f"Error: Input CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not Path(args.metadata).exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Get assays from metadata
    assay_ids = get_assays_from_metadata(args.metadata, args.min_active, args.min_inactive)
    
    if not assay_ids:
        print("No assays found that meet the criteria. Try lowering the minimum thresholds.")
        sys.exit(1)
    
    # Convert to comma-separated string for the extract script
    assay_ids_str = ','.join(assay_ids)
    
    # Call the multi-target extraction function
    from extract_single_target import extract_multi_target
    extract_multi_target(args.csv, assay_ids, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
