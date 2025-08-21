#!/usr/bin/env python3
"""
Extract Multi-Target Dataset from ChEMBL Complete Dataset

This script extracts compounds measured in multiple target assays from the complete ChEMBL dataset.
It properly handles the three-state classification: 1 (active), -1 (inactive), 0 (missing/not measured).

Usage:
    # Single target (backward compatible)
    python extract_single_target.py --csv chembl_complete_dataset.csv --target_assay_id CHEMBL1794580 --output target_dataset.csv
    
    # Multiple targets
    python extract_single_target.py --csv chembl_complete_dataset.csv --target_assay_ids CHEMBL1794580,CHEMBL1614161,CHEMBL1614441 --output multi_target_dataset.csv
    
    # Using a list of assays from metadata analysis
    python extract_single_target.py --csv chembl_complete_dataset.csv --target_assay_ids "['CHEMBL1614161', 'CHEMBL1614441', 'CHEMBL1614458', 'CHEMBL1614530', 'CHEMBL1738442', 'CHEMBL1794345', 'CHEMBL1794580']" --output multi_target_dataset.csv
"""

import pandas as pd
import argparse
import sys
import ast
from pathlib import Path


def extract_multi_target(input_csv_path, target_assay_ids, output_csv_path, chunk_size=1000):
    """
    Extracts compounds measured in multiple target assays with proper three-state classification.
    Values: 1 (active), -1 (inactive), 0 (missing/not measured)

    Args:
        input_csv_path (str): Path to the input chembl_complete_dataset.csv.
        target_assay_ids (list): List of CHEMBL IDs of the target assays to filter by.
        output_csv_path (str): Path to save the filtered dataset.
        chunk_size (int): Number of rows to process at a time.
    """
    print(f"Loading data from {input_csv_path} in chunks...")
    print(f"Extracting measurements for {len(target_assay_ids)} assays: {target_assay_ids}")

    # Construct the column names for the target assays
    target_col_names = [f'assay_{assay_id}' for assay_id in target_assay_ids]

    # First, let's check what columns exist in the CSV
    print("Reading CSV header to identify columns...")
    try:
        # Read just the header to get column names
        header_df = pd.read_csv(input_csv_path, nrows=0)
        print(f"Found {len(header_df.columns)} columns in the CSV")
        
        # Check if our target columns exist
        missing_cols = [col for col in target_col_names if col not in header_df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Available assay columns: {[col for col in header_df.columns if col.startswith('assay_')][:10]}...")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error reading CSV header: {e}")
        sys.exit(1)

    # Define the columns we need
    required_cols = ['compound_id', 'smiles', 'compound_name', 'fold'] + target_col_names
    
    # Verify all required columns exist
    missing_cols = [col for col in required_cols if col not in header_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"Extracting compounds for {len(target_assay_ids)} assays with three-state classification...")
    print(f"Processing in chunks of {chunk_size} rows...")

    # Process the CSV in chunks
    total_rows = 0
    filtered_rows = 0
    assay_stats = {assay_id: {'active': 0, 'inactive': 0, 'missing': 0} for assay_id in target_assay_ids}
    compounds_with_any_measurement = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(input_csv_path, usecols=required_cols, chunksize=chunk_size):
            total_rows += len(chunk)
            
            # Create a mask for compounds that have ANY measurement in the target assays
            # (at least one assay has a value != 0)
            any_measurement_mask = chunk[target_col_names].ne(0.0).any(axis=1)
            compounds_with_measurements = chunk[any_measurement_mask].copy()
            
            if len(compounds_with_measurements) > 0:
                compounds_with_any_measurement += len(compounds_with_measurements)
                
                # Count statistics for each assay
                for i, assay_id in enumerate(target_assay_ids):
                    col_name = target_col_names[i]
                    values = compounds_with_measurements[col_name]
                    
                    assay_stats[assay_id]['active'] += (values == 1.0).sum()
                    assay_stats[assay_id]['inactive'] += (values == -1.0).sum()
                    assay_stats[assay_id]['missing'] += (values == 0.0).sum()
                
                # Rename assay columns to be more descriptive
                column_mapping = {}
                for col_name, assay_id in zip(target_col_names, target_assay_ids):
                    column_mapping[col_name] = f'activity_{assay_id}'
                
                compounds_with_measurements = compounds_with_measurements.rename(columns=column_mapping)
                filtered_rows += len(compounds_with_measurements)
                
                # Write to CSV (append mode after first chunk)
                if first_chunk:
                    compounds_with_measurements.to_csv(output_csv_path, index=False, mode='w')
                    first_chunk = False
                else:
                    compounds_with_measurements.to_csv(output_csv_path, index=False, mode='a', header=False)
            
            # Progress update
            if total_rows % (chunk_size * 10) == 0:
                print(f"Processed {total_rows:,} rows, found {compounds_with_any_measurement:,} compounds with measurements...")

    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

    print(f"\nMulti-target dataset created successfully!")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Compounds with any measurement: {compounds_with_any_measurement:,}")
    print(f"Output saved to: {output_csv_path}")
    
    print(f"\n📊 Assay Statistics:")
    print("=" * 80)
    for assay_id in target_assay_ids:
        stats = assay_stats[assay_id]
        total_measured = stats['active'] + stats['inactive']
        print(f"{assay_id}:")
        print(f"  • Active compounds: {stats['active']:,}")
        print(f"  • Inactive compounds: {stats['inactive']:,}")
        print(f"  • Missing measurements: {stats['missing']:,}")
        print(f"  • Total measured: {total_measured:,}")
        if total_measured > 0:
            active_pct = (stats['active'] / total_measured) * 100
            print(f"  • Active percentage: {active_pct:.1f}%")
        print()


def extract_single_target(input_csv_path, target_assay_id, output_csv_path, chunk_size=1000):
    """
    Extracts compounds measured in a specific target assay with proper three-state classification.
    Values: 1 (active), -1 (inactive), 0 (missing/not measured)

    Args:
        input_csv_path (str): Path to the input chembl_complete_dataset.csv.
        target_assay_id (str): The CHEMBL ID of the target assay to filter by.
        output_csv_path (str): Path to save the filtered dataset.
        chunk_size (int): Number of rows to process at a time.
    """
    print(f"Loading data from {input_csv_path} in chunks...")

    # Construct the column name for the target assay
    target_col_name = f'assay_{target_assay_id}'

    # First, let's check what columns exist in the CSV
    print("Reading CSV header to identify columns...")
    try:
        # Read just the header to get column names
        header_df = pd.read_csv(input_csv_path, nrows=0)
        print(f"Found {len(header_df.columns)} columns in the CSV")
        
        # Check if our target column exists
        if target_col_name not in header_df.columns:
            print(f"Error: Column '{target_col_name}' not found in the CSV.")
            print(f"Available assay columns: {[col for col in header_df.columns if col.startswith('assay_')][:10]}...")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error reading CSV header: {e}")
        sys.exit(1)

    # Define the columns we need
    required_cols = ['compound_id', 'smiles', 'compound_name', 'fold', target_col_name]
    
    # Verify all required columns exist
    missing_cols = [col for col in required_cols if col not in header_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"Extracting compounds for {target_assay_id} with three-state classification...")
    print(f"Processing in chunks of {chunk_size} rows...")

    # Process the CSV in chunks
    total_rows = 0
    filtered_rows = 0
    active_count = 0
    inactive_count = 0
    missing_count = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(input_csv_path, usecols=required_cols, chunksize=chunk_size):
            total_rows += len(chunk)
            
            # Filter rows where the target assay column has a measurement (not 0/missing)
            # We want compounds that are either active (1) or inactive (-1)
            measured_mask = chunk[target_col_name] != 0.0
            measured_compounds = chunk[measured_mask].copy()
            
            if len(measured_compounds) > 0:
                # Create activity column based on the target assay value
                # 1 = active, -1 = inactive, 0 = missing (but we filtered these out)
                measured_compounds.loc[:, 'activity'] = measured_compounds[target_col_name].astype(int)
                
                # Count active vs inactive vs missing
                chunk_active = (measured_compounds[target_col_name] == 1.0).sum()
                chunk_inactive = (measured_compounds[target_col_name] == -1.0).sum()
                chunk_missing = (measured_compounds[target_col_name] == 0.0).sum()
                active_count += chunk_active
                inactive_count += chunk_inactive
                missing_count += chunk_missing
                
                # Drop the target assay column as we now have the activity column
                filtered_chunk = measured_compounds.drop(columns=[target_col_name])
                
                filtered_rows += len(filtered_chunk)
                
                # Write to CSV (append mode after first chunk)
                if first_chunk:
                    filtered_chunk.to_csv(output_csv_path, index=False, mode='w')
                    first_chunk = False
                else:
                    filtered_chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
            
            # Progress update
            if total_rows % (chunk_size * 10) == 0:
                print(f"Processed {total_rows:,} rows, found {filtered_rows:,} measured compounds...")

    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

    print(f"Dataset for {target_assay_id} created successfully!")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Filtered dataset size: {filtered_rows:,} entries")
    print(f"Active compounds (1): {active_count:,}")
    print(f"Inactive compounds (-1): {inactive_count:,}")
    print(f"Missing compounds (0): {missing_count:,}")
    print(f"Output saved to: {output_csv_path}")


def parse_assay_ids(assay_ids_input):
    """
    Parse assay IDs from various input formats.
    
    Args:
        assay_ids_input (str): Can be:
            - Single assay ID: "CHEMBL1794580"
            - Comma-separated: "CHEMBL1794580,CHEMBL1614161"
            - List string: "['CHEMBL1794580', 'CHEMBL1614161']"
    
    Returns:
        list: List of assay IDs
    """
    if not assay_ids_input:
        return []
    
    # Try to parse as a Python list first
    try:
        if assay_ids_input.startswith('[') and assay_ids_input.endswith(']'):
            return ast.literal_eval(assay_ids_input)
    except:
        pass
    
    # Try comma-separated format
    if ',' in assay_ids_input:
        return [aid.strip() for aid in assay_ids_input.split(',')]
    
    # Single assay ID
    return [assay_ids_input.strip()]


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract compounds for specified ChEMBL target assay(s) with three-state classification."
    )
    parser.add_argument(
        "--csv", 
        type=str, 
        required=True,
        help="Path to the input chembl_complete_dataset.csv."
    )
    parser.add_argument(
        "--target_assay_id", 
        type=str, 
        help="The CHEMBL ID of a single target assay (e.g., CHEMBL1794580)."
    )
    parser.add_argument(
        "--target_assay_ids", 
        type=str, 
        help="Multiple CHEMBL IDs. Can be: comma-separated (CHEMBL1794580,CHEMBL1614161) or list string (['CHEMBL1794580', 'CHEMBL1614161'])."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save the filtered dataset CSV."
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Number of rows to process at a time (default: 1000)."
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.csv).exists():
        print(f"Error: Input CSV file not found: {args.csv}")
        sys.exit(1)

    # Determine which assays to extract
    if args.target_assay_ids:
        # Multi-target mode
        target_assay_ids = parse_assay_ids(args.target_assay_ids)
        if not target_assay_ids:
            print("Error: No valid assay IDs provided in --target_assay_ids")
            sys.exit(1)
        print(f"Multi-target mode: Extracting {len(target_assay_ids)} assays")
        extract_multi_target(args.csv, target_assay_ids, args.output, args.chunk_size)
    elif args.target_assay_id:
        # Single target mode (backward compatible)
        print("Single-target mode")
        extract_single_target(args.csv, args.target_assay_id, args.output, args.chunk_size)
    else:
        print("Error: Must provide either --target_assay_id or --target_assay_ids")
        sys.exit(1)


if __name__ == "__main__":
    main()