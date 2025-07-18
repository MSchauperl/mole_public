"""
Dataset loader for Table 1 ADMET datasets.

This module provides data loading and preprocessing functionality for the ADMET datasets
mentioned in Table 1 of the MolE paper. It handles both regression and classification tasks
with proper train/validation/test splitting and batching. Uses the same atom environment
tokenization approach as the multitask model.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .table1_prediction_heads import Table1PredictionHeads
from mole.data.vocabulary import open_dictionary
from configs.table1_task_config import (
    TASK_CONFIG, get_regression_tasks, get_classification_tasks,
    get_metric_mapping, get_task_type_mapping, get_tdc_column_mapping,
    get_all_tasks
)

logger = logging.getLogger(__name__)


class ADMETDataset(Dataset):
    """Dataset class for ADMET data with atom environment tokenization."""
    
    def __init__(
        self,
        smiles_list: List[str],
        targets: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        task_config: Dict[str, Any],
        vocab_path: str = "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        radius: int = 0,
        use_features: bool = False,
        max_length: Optional[int] = None,
        cls_token: bool = True,
        transform=None
    ):
        """
        Initialize ADMET dataset with atom environment tokenization.
        
        Args:
            smiles_list: List of SMILES strings
            targets: Dictionary mapping task names to target arrays
            masks: Dictionary mapping task names to mask arrays
            task_config: Configuration for tasks (from table1_prediction_heads)
            vocab_path: Path to vocabulary file for atom environments
            radius: Morgan fingerprint radius for atom environments
            use_features: Whether to use functional features
            max_length: Maximum sequence length
            cls_token: Whether to add CLS token
            transform: Optional transform to apply to data
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.masks = masks
        self.task_config = task_config
        self.transform = transform
        self.radius = radius
        self.use_features = use_features
        self.max_length = max_length
        self.cls_token = cls_token
        
        # Load vocabulary
        self.vocab = open_dictionary(vocab_path)
        self.pad_id = self.vocab.get("PAD", 0)
        self.mask_id = self.vocab.get("MASK", 1)
        self.unk_id = self.vocab.get("UNK", 2)
        self.cls_id = self.vocab.get("CLS", 3)
        
        # Validate that all tasks have the same number of samples
        sample_counts = [len(targets[task]) for task in targets.keys()]
        if len(set(sample_counts)) > 1:
            raise ValueError(f"All tasks must have the same number of samples. Got: {sample_counts}")
        
        # Validate that masks have the same number of samples
        mask_counts = [len(masks[task]) for task in masks.keys()]
        if len(set(mask_counts)) > 1:
            raise ValueError(f"All masks must have the same number of samples. Got: {mask_counts}")
        
        self.n_samples = len(smiles_list)
        logger.info(f"Created ADMET dataset with {self.n_samples} samples and {len(targets)} tasks")
        logger.info(f"Using vocabulary with {len(self.vocab)} tokens (radius={radius}, features={use_features})")
    
    def get_atom_environments(self, mol: Chem.Mol) -> List[int]:
        """Extract atom environments for a molecule using the same approach as multitask model."""
        if mol is None:
            return []

        info = {}
        _ = AllChem.GetMorganFingerprint(
            mol,
            radius=self.radius,
            bitInfo=info,
            useFeatures=self.use_features,
            includeRedundantEnvironments=True,
        )

        atom_envs = [None] * mol.GetNumAtoms()
        for atom_idx in range(mol.GetNumAtoms()):
            for bit, atom_list in info.items():
                for atom_info_tuple in atom_list:
                    if atom_info_tuple[0] == atom_idx and atom_info_tuple[1] == self.radius:
                        atom_envs[atom_idx] = bit
                        break
                if atom_envs[atom_idx] is not None:
                    break

        return atom_envs
    
    def encode_environments(self, environments: List[int]) -> List[int]:
        """Encode atom environments using vocabulary."""
        tokens = []
        for env in environments:
            if env is None:
                tokens.append(self.unk_id)
            else:
                token = self.vocab.get(env, self.unk_id)
                tokens.append(token)
        return tokens
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # Get targets for this sample
        sample_targets = {}
        sample_masks = {}
        for task_name, target_array in self.targets.items():
            sample_targets[task_name] = target_array[idx]
            sample_masks[task_name] = self.masks[task_name][idx]
        
        # Tokenize SMILES into atom environments
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Extract atom environments
            atom_envs = self.get_atom_environments(mol)
            if len(atom_envs) == 0:
                raise ValueError(f"No atom environments found for: {smiles}")
            
            # Encode with vocabulary
            tokens = self.encode_environments(atom_envs)
            
            # Add CLS token if required
            if self.cls_token:
                tokens = [self.cls_id] + tokens
            
            # Apply length limit if specified
            if self.max_length is not None:
                tokens = tokens[:self.max_length]
            
            # Create distance matrix for molecular graph
            dist_mat = Chem.GetDistanceMatrix(mol)
            dist_mat[dist_mat == 1.0e08] = -1
            
            # Convert to sparse format
            from scipy import sparse
            dist_mat_sparse = sparse.coo_matrix(dist_mat + 1)
            
            # Adjust for CLS token
            if self.cls_token:
                # Add row and column for CLS token
                dist_mat_sparse = sparse.vstack(
                    [np.zeros(dist_mat_sparse.shape[0])[None, :], dist_mat_sparse]
                )
                dist_mat_sparse = sparse.hstack(
                    [np.zeros(dist_mat_sparse.shape[0])[:, None], dist_mat_sparse]
                )
            
            # Create PyTorch Geometric Data object with individual task attributes
            data_dict = {
                'x': torch.tensor(tokens, dtype=torch.long),
                'edge_index': torch.tensor(
                    np.array([dist_mat_sparse.row, dist_mat_sparse.col]),
                    dtype=torch.long,
                ),
                'edge_attr': torch.tensor(dist_mat_sparse.data, dtype=torch.long),
                'smiles': smiles,
            }
            
            # Add individual task targets and masks as attributes
            for task_name, target_value in sample_targets.items():
                # Determine dtype based on task type and value type
                if self.task_config[task_name]['task_type'] == 'regression':
                    data_dict[task_name] = torch.tensor(target_value, dtype=torch.float32)
                else:  # classification
                    data_dict[task_name] = torch.tensor(target_value, dtype=torch.long)
                data_dict[f'{task_name}_mask'] = torch.tensor(sample_masks[task_name], dtype=torch.bool)
            
            data = Data(**data_dict)
            
        except Exception as e:
            # Return a dummy sample for failed molecules
            logger.warning(f"Failed to process molecule {idx}: {e}")
            dummy_tokens = [self.cls_id, self.unk_id] if self.cls_token else [self.unk_id]
            
            # Create dummy PyTorch Geometric Data object with individual task attributes
            data_dict = {
                'x': torch.tensor(dummy_tokens, dtype=torch.long),
                'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                'edge_attr': torch.tensor([1, 1], dtype=torch.long),
                'smiles': smiles,
                'dummy': torch.tensor(True, dtype=torch.bool),
            }
            
            # Add individual task targets and masks as attributes
            for task_name, target_value in sample_targets.items():
                # Determine dtype based on task type and value type
                if self.task_config[task_name]['task_type'] == 'regression':
                    data_dict[task_name] = torch.tensor(target_value, dtype=torch.float32)
                else:  # classification
                    data_dict[task_name] = torch.tensor(target_value, dtype=torch.long)
                data_dict[f'{task_name}_mask'] = torch.tensor(sample_masks[task_name], dtype=torch.bool)
            
            data = Data(**data_dict)
        
        if self.transform:
            data = self.transform(data)
        
        return data


class ADMETDataLoader:
    """Main data loader class for Table 1 ADMET datasets."""
    
    def __init__(
        self,
        data_path: str = "data/tdc/tdc_table1_datasets.csv",
        task_config: Dict[str, Any] = None,
        vocab_path: str = "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        radius: int = 0,
        use_features: bool = False,
        max_length: Optional[int] = None,
        cls_token: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize the ADMET data loader.
        
        Args:
            data_path: Path to the CSV file with Table 1 datasets
            task_config: Task configuration (defaults to flattened TASK_CONFIG)
            vocab_path: Path to vocabulary file for atom environments
            radius: Morgan fingerprint radius for atom environments
            use_features: Whether to use functional features
            max_length: Maximum sequence length
            cls_token: Whether to add CLS token
            test_size: Fraction of data to use for testing
            val_size: Fraction of remaining data to use for validation
            random_state: Random seed for reproducibility
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_path = Path(data_path)
        
        # Create flattened task configuration
        if task_config is None:
            self.task_config = {}
            task_type_mapping = get_task_type_mapping()
            metric_mapping = get_metric_mapping()
            tdc_column_mapping = get_tdc_column_mapping()
            
            for task_name in get_all_tasks():
                self.task_config[task_name] = {
                    'task_type': task_type_mapping[task_name],
                    'metric': metric_mapping[task_name],
                    'tdc_column': tdc_column_mapping[task_name]
                }
        else:
            self.task_config = task_config
            
        self.vocab_path = vocab_path
        self.radius = radius
        self.use_features = use_features
        self.max_length = max_length
        self.cls_token = cls_token
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.scalers = {}
        self.label_encoders = {}
        self.datasets = {}
        self.dataloaders = {}
        
        # Load and process data
        self._load_data()
        self._preprocess_data()
        self._create_datasets()
        self._create_dataloaders()
    
    def _load_data(self):
        """Load the raw data from CSV file."""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.raw_data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_data)} samples with {len(self.raw_data.columns)} columns")
        
        # Validate required columns
        required_cols = ['SMILES'] + [config['tdc_column'] for config in self.task_config.values()]
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _preprocess_data(self):
        """Preprocess the data for each task."""
        logger.info("Preprocessing data for each task")
        
        self.processed_data = {
            'smiles': self.raw_data['SMILES'].tolist(),
            'targets': {},
            'masks': {}
        }
        
        for task_name, config in self.task_config.items():
            logger.info(f"Processing task: {task_name}")
            
            # Get target values from TDC column
            tdc_column = config['tdc_column']
            target_values = self.raw_data[tdc_column].values
            
            # Handle missing values
            if config['task_type'] == 'regression':
                # For regression, keep NaN values and create masks
                valid_mask = ~np.isnan(target_values)
                # Fill NaN with 0 for storage (will be masked during training)
                target_values = np.where(np.isnan(target_values), 0.0, target_values)
                self.processed_data['targets'][task_name] = target_values.astype(np.float32)
                self.processed_data['masks'][task_name] = valid_mask.astype(np.bool_)
                
                valid_count = valid_mask.sum()
                logger.info(f"  - {task_name}: {valid_count} valid samples out of {len(target_values)}")
                
            elif config['task_type'] == 'classification':
                # For classification, remove samples with NaN values
                valid_mask = ~np.isnan(target_values)
                if task_name == 'smiles':  # Special handling for SMILES column
                    continue
                
                # Encode labels
                label_encoder = LabelEncoder()
                valid_values = target_values[valid_mask]
                encoded_values = label_encoder.fit_transform(valid_values)
                
                # Create full array with -1 for invalid samples (will be masked during training)
                full_encoded = np.full(len(target_values), -1, dtype=np.int64)
                full_encoded[valid_mask] = encoded_values
                
                self.processed_data['targets'][task_name] = full_encoded
                self.processed_data['masks'][task_name] = valid_mask.astype(np.bool_)
                self.label_encoders[task_name] = label_encoder
                
                logger.info(f"  - {task_name}: {len(valid_values)} valid samples, "
                          f"{len(label_encoder.classes_)} classes")
        
        # Remove samples that have no valid targets for any task
        valid_samples = []
        for i in range(len(self.processed_data['smiles'])):
            has_valid_target = False
            for task_name, targets in self.processed_data['targets'].items():
                if self.task_config[task_name]['task_type'] == 'regression':
                    # For regression, check if mask is True
                    if self.processed_data['masks'][task_name][i]:
                        has_valid_target = True
                        break
                elif targets[i] != -1:  # Classification with valid label
                    has_valid_target = True
                    break
            
            if has_valid_target:
                valid_samples.append(i)
        
        # Filter data to only valid samples
        self.processed_data['smiles'] = [self.processed_data['smiles'][i] for i in valid_samples]
        for task_name in self.processed_data['targets']:
            self.processed_data['targets'][task_name] = self.processed_data['targets'][task_name][valid_samples]
            self.processed_data['masks'][task_name] = self.processed_data['masks'][task_name][valid_samples]
        
        logger.info(f"After filtering: {len(self.processed_data['smiles'])} valid samples")
    
    def _create_datasets(self):
        """Create train/validation/test datasets."""
        logger.info("Creating train/validation/test datasets")
        
        # Calculate split sizes
        total_size = len(self.processed_data['smiles'])
        test_size = int(total_size * self.test_size)
        val_size = int((total_size - test_size) * self.val_size)
        train_size = total_size - test_size - val_size
        
        logger.info(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
        
        # Create full dataset
        full_dataset = ADMETDataset(
            smiles_list=self.processed_data['smiles'],
            targets=self.processed_data['targets'],
            masks=self.processed_data['masks'],
            task_config=self.task_config,
            vocab_path=self.vocab_path,
            radius=self.radius,
            use_features=self.use_features,
            max_length=self.max_length,
            cls_token=self.cls_token
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        self.datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        logger.info("Datasets created successfully")
    
    def _create_dataloaders(self):
        """Create DataLoader objects for each split."""
        logger.info("Creating DataLoader objects")
        
        from torch_geometric.data import DataLoader as PyGDataLoader
        
        for split_name, dataset in self.datasets.items():
            self.dataloaders[split_name] = PyGDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split_name == 'train'),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=(split_name == 'train')
            )
        
        logger.info("DataLoaders created successfully")
    
    def get_dataloader(self, split: str) -> DataLoader:
        """Get DataLoader for specified split."""
        if split not in self.dataloaders:
            raise ValueError(f"Invalid split: {split}. Available splits: {list(self.dataloaders.keys())}")
        return self.dataloaders[split]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets."""
        info = {
            'total_samples': len(self.processed_data['smiles']),
            'tasks': {},
            'splits': {}
        }
        
        # Task information
        for task_name, config in self.task_config.items():
            if task_name in self.processed_data['targets']:
                targets = self.processed_data['targets'][task_name]
                task_info = {
                    'task_type': config['task_type'],
                    'metric': config['metric'],
                    'n_samples': len(targets)
                }
                
                if config['task_type'] == 'classification':
                    valid_targets = targets[targets != -1]
                    task_info['n_classes'] = len(self.label_encoders[task_name].classes_)
                    task_info['n_valid_samples'] = len(valid_targets)
                elif config['task_type'] == 'regression':
                    # Only compute statistics for valid samples
                    valid_mask = self.processed_data['masks'][task_name]
                    valid_targets = targets[valid_mask]
                    task_info['mean'] = float(np.mean(valid_targets))
                    task_info['std'] = float(np.std(valid_targets))
                    task_info['n_valid_samples'] = len(valid_targets)
                
                info['tasks'][task_name] = task_info
        
        # Split information
        for split_name, dataset in self.datasets.items():
            info['splits'][split_name] = len(dataset)
        
        return info
    
    def get_task_config(self) -> Dict[str, Any]:
        """Get the task configuration."""
        return self.task_config
    
    def get_label_encoders(self) -> Dict[str, LabelEncoder]:
        """Get label encoders for classification tasks."""
        return self.label_encoders


def create_admet_dataloader(
    data_path: str = "data/tdc/tdc_table1_datasets.csv",
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> ADMETDataLoader:
    """
    Convenience function to create an ADMET data loader.
    
    Args:
        data_path: Path to the CSV file with Table 1 datasets
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments passed to ADMETDataLoader
    
    Returns:
        Configured ADMETDataLoader instance
    """
    return ADMETDataLoader(
        data_path=data_path,
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create data loader
    dataloader = create_admet_dataloader(batch_size=16)
    
    # Get dataset information
    info = dataloader.get_dataset_info()
    print("\nDataset Information:")
    print(f"Total samples: {info['total_samples']}")
    print(f"Splits: {info['splits']}")
    
    print("\nTask Information:")
    for task_name, task_info in info['tasks'].items():
        print(f"  {task_name}:")
        print(f"    Type: {task_info['task_type']}")
        print(f"    Metric: {task_info['metric']}")
        print(f"    Samples: {task_info['n_samples']}")
        if task_info['task_type'] == 'classification':
            print(f"    Classes: {task_info['n_classes']}")
            print(f"    Valid samples: {task_info['n_valid_samples']}")
        elif task_info['task_type'] == 'regression':
            print(f"    Mean: {task_info['mean']:.4f}")
            print(f"    Std: {task_info['std']:.4f}")
    
    # Test data loading
    print("\nTesting data loading...")
    train_loader = dataloader.get_dataloader('train')
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  SMILES: {len(batch['smiles'])} samples")
        print(f"  Targets: {list(batch['targets'].keys())}")
        
        # Print first sample details
        first_smiles = batch['smiles'][0]
        first_targets = batch['targets']
        print(f"  First SMILES: {first_smiles}")
        print(f"  First targets: {first_targets}")
        
        if batch_idx >= 2:  # Only test first few batches
            break
    
    print("\nData loader test completed successfully!") 