#!/usr/bin/env python3
"""
Create a complete CSV file of the ChemBL dataset including all assays and fold information.

This script loads the ChemBL dataset and creates a comprehensive CSV file that includes:
- Compound SMILES
- Compound names
- All target assays with their values (1=active, -1=inactive, 0=missing)
- Fold information (train/validation split)
- Target names and metadata

Usage:
    python create_chembl_csv.py --output chembl_complete_dataset.csv
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pickle_file(file_path: Path) -> any:
    """Load a pickle file safely."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def load_text_file(file_path: Path) -> List[str]:
    """Load a text file and return lines."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def load_chembl_data(data_dir: Path, use_filtered: bool = True) -> Dict:
    """Load ChemBL dataset files."""
    data = {}
    
    if use_filtered:
        logger.info("Loading filtered ChemBL dataset...")
        
        # Load SMILES
        smiles_path = data_dir / "chemblSmiles_top10_min3.pckl"
        data['smiles'] = load_pickle_file(smiles_path)
        logger.info(f"Loaded {len(data['smiles'])} SMILES strings")
        
        # Load labels
        labels_path = data_dir / "labelsHard_top10_min3.pckl"
        data['labels'] = load_pickle_file(labels_path)
        logger.info(f"Loaded labels with shape: {data['labels'].shape}")
        
        # Load compound names
        compound_names_path = data_dir / "compoundNames_top10_min3.txt"
        data['compound_names'] = load_text_file(compound_names_path)
        logger.info(f"Loaded {len(data['compound_names'])} compound names")
        
        # Load target names
        target_names_path = data_dir / "targetNames_top10_min3.txt"
        data['target_names'] = load_text_file(target_names_path)
        logger.info(f"Loaded {len(data['target_names'])} target names")
        
        # Load splits
        splits_path = data_dir / "splits_top10_min3.pckl"
        data['splits'] = load_pickle_file(splits_path)
        logger.info(f"Loaded splits data")
        
    else:
        logger.info("Loading full ChemBL dataset...")
        
        # Load SMILES
        smiles_path = data_dir / "chembl20Smiles.pckl"
        data['smiles'] = load_pickle_file(smiles_path)
        logger.info(f"Loaded {len(data['smiles'])} SMILES strings")
        
        # Load labels
        labels_path = data_dir / "labelsHard.pckl"
        data['labels'] = load_pickle_file(labels_path)
        logger.info(f"Loaded labels with shape: {data['labels'].shape}")
        
        # Load compound names
        compound_names_path = data_dir / "labelsWeakHard.cmpNames"
        data['compound_names'] = load_text_file(compound_names_path)
        logger.info(f"Loaded {len(data['compound_names'])} compound names")
        
        # Load target names
        target_names_path = data_dir / "labelsWeakHard.targetNames"
        data['target_names'] = load_text_file(target_names_path)
        logger.info(f"Loaded {len(data['target_names'])} target names")
        
        # Load folds (if available)
        folds_path = data_dir / "folds0.pckl"
        if folds_path.exists():
            data['folds'] = load_pickle_file(folds_path)
            logger.info(f"Loaded folds data")
        else:
            data['folds'] = None
            logger.warning("No folds data found")
    
    return data


def create_fold_assignments(splits_data: any, num_compounds: int) -> List[str]:
    """Create fold assignments for compounds."""
    fold_assignments = ['unknown'] * num_compounds
    
    if splits_data is None:
        return fold_assignments
    
    try:
        # Handle different split data formats
        if isinstance(splits_data, dict):
            # Format: {'train': train_indices, 'val': val_indices, 'test': test_indices}
            for fold_name, indices in splits_data.items():
                for idx in indices:
                    if idx < num_compounds:
                        fold_assignments[idx] = fold_name
        elif isinstance(splits_data, list):
            # Format: [fold_0_indices, fold_1_indices, fold_2_indices, ...]
            for fold_idx, indices in enumerate(splits_data):
                for idx in indices:
                    if idx < num_compounds:
                        fold_assignments[idx] = f'fold_{fold_idx}'
        else:
            logger.warning(f"Unknown splits data format: {type(splits_data)}")
            
    except Exception as e:
        logger.error(f"Error processing splits data: {e}")
    
    return fold_assignments


def create_dataframe(data: Dict) -> pd.DataFrame:
    """Create a pandas DataFrame from the loaded data."""
    logger.info("Creating DataFrame...")
    
    # Basic compound information
    df_data = {
        'compound_id': list(range(len(data['smiles']))),
        'smiles': data['smiles'],
        'compound_name': data['compound_names'][:len(data['smiles'])] if data['compound_names'] else [''] * len(data['smiles'])
    }
    
    # Add fold assignments
    splits_data = data.get('splits') or data.get('folds')
    fold_assignments = create_fold_assignments(splits_data, len(data['smiles']))
    df_data['fold'] = fold_assignments
    
    # Add target assay values
    if data['labels'] is not None:
        # Handle sparse matrix
        if hasattr(data['labels'], 'toarray'):
            labels_array = data['labels'].toarray()
        else:
            labels_array = np.array(data['labels'])
        
        target_names = data['target_names']
        
        logger.info(f"Adding {len(target_names)} target assays...")
        
        for i, target_name in enumerate(target_names):
            if i < labels_array.shape[1]:
                # Keep original values: 1 (active), -1 (inactive), 0 (missing/not measured)
                values = labels_array[:, i].astype(float)
                df_data[f'assay_{target_name}'] = values
            else:
                logger.warning(f"Target {target_name} index {i} out of bounds for labels shape {labels_array.shape}")
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Add metadata columns
    df['dataset'] = 'filtered' if len(data['target_names']) == 10 else 'full'
    df['num_targets'] = len(data['target_names'])
    
    logger.info(f"Created DataFrame with shape: {df.shape}")
    return df


def add_target_metadata(df: pd.DataFrame, data: Dict) -> pd.DataFrame:
    """Add target metadata information."""
    logger.info("Adding target metadata...")
    
    # Create target metadata DataFrame
    target_metadata = []
    for i, target_name in enumerate(data['target_names']):
        if data['labels'] is not None:
            # Handle sparse matrix
            if hasattr(data['labels'], 'toarray'):
                labels_array = data['labels'].toarray()
            else:
                labels_array = np.array(data['labels'])
            
            if i < labels_array.shape[1]:
                values = labels_array[:, i]
                active_count = np.sum(values == 1)
                inactive_count = np.sum(values == -1)
                missing_count = np.sum(values == 0)
                total_count = len(values)
                
                target_metadata.append({
                    'target_id': i,
                    'target_name': target_name,
                    'total_measurements': total_count,
                    'active_count': active_count,
                    'inactive_count': inactive_count,
                    'missing_count': missing_count,
                    'active_ratio': active_count / total_count if total_count > 0 else 0
                })
    
    # Save target metadata separately
    target_df = pd.DataFrame(target_metadata)
    
    return df, target_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create complete ChemBL dataset CSV")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/ChemBl",
        help="Directory containing ChemBL data files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chembl_complete_dataset.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--target_metadata",
        type=str,
        default="chembl_target_metadata.csv",
        help="Output target metadata CSV file path"
    )
    parser.add_argument(
        "--use_filtered_dataset",
        action="store_true",
        help="Use filtered ChemBL dataset instead of full"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for testing (use only first N compounds)"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    # Load data
    data = load_chembl_data(data_dir, use_filtered=args.use_filtered_dataset)
    
    if not data['smiles'] or data['labels'] is None:
        logger.error("Failed to load essential data")
        return 1
    
    # Create DataFrame
    df = create_dataframe(data)
    
    # Add target metadata
    df, target_df = add_target_metadata(df, data)
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} compounds for testing...")
        df = df.head(args.sample_size)
    
    # Save main dataset
    logger.info(f"Saving main dataset to {args.output}...")
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} compounds with {len(df.columns)} columns")
    
    # Save target metadata
    logger.info(f"Saving target metadata to {args.target_metadata}...")
    target_df.to_csv(args.target_metadata, index=False)
    logger.info(f"Saved metadata for {len(target_df)} targets")
    
    # Print summary
    print("\n" + "="*60)
    print("CHEMBL DATASET SUMMARY")
    print("="*60)
    print(f"Total compounds: {len(df)}")
    print(f"Total targets: {len(data['target_names'])}")
    print(f"Dataset type: {'Filtered' if args.use_filtered_dataset else 'Full'}")
    
    if 'fold' in df.columns:
        fold_counts = df['fold'].value_counts()
        print(f"\nFold distribution:")
        for fold, count in fold_counts.items():
            print(f"  {fold}: {count} compounds")
    
    print(f"\nColumns in dataset:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nFiles created:")
    print(f"  - {args.output} (main dataset)")
    print(f"  - {args.target_metadata} (target metadata)")
    
    return 0


if __name__ == "__main__":
    exit(main())
