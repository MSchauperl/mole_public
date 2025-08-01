"""
ChemBL Data Module for MolE Pretraining

Handles data loading, preprocessing, and batching for ChemBL-based training,
combining cross-environment MLM with ChemBL binary classification tasks.
"""

import pickle
from typing import Optional, Union, List, Dict
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from scipy import sparse

from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.data.chembl_dataset import ChemBLDataset


class ChemBLDataModule(CrossEnvDataModule):
    """Lightning data module for ChemBL-based pretraining"""

    def __init__(
        self,
        chembl_smiles_path: str,
        chembl_labels_path: str,
        chembl_target_names_path: str,
        chembl_compound_names_path: Optional[str] = None,
        input_vocab_path: str = None,
        target_vocab_path: str = None,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        max_targets: Optional[int] = None,
        min_target_activity: int = 10,
        max_samples: Optional[int] = None,
        # Inherited MLM parameters
        input_radius: int = 0,
        target_radius: int = 1,
        input_use_features: bool = False,
        target_use_features: bool = True,
        mask_prob: float = 0.15,
        replace_prob: float = 0.8,
        random_prob: float = 0.1,
        max_length: Optional[int] = None,
        cls_token: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs,
    ):
        """
        Initialize ChemBL data module.
        
        Args:
            chembl_smiles_path: Path to ChemBL SMILES pickle file
            chembl_labels_path: Path to ChemBL labels sparse matrix
            chembl_target_names_path: Path to target names file
            chembl_compound_names_path: Path to compound names file (optional)
            input_vocab_path: Path to input vocabulary file
            target_vocab_path: Path to target vocabulary file
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            max_targets: Maximum number of targets to use
            min_target_activity: Minimum activity count per target
            max_samples: Maximum number of samples to use (for testing)
            **kwargs: Additional arguments for parent class
        """
        # Initialize parent without train_data (we'll load ChemBL data differently)
        super().__init__(
            train_data="",  # Placeholder, we'll override setup
            input_vocab_path=input_vocab_path,
            target_vocab_path=target_vocab_path,
            validation_data=None,
            test_data=None,
            input_radius=input_radius,
            target_radius=target_radius,
            input_use_features=input_use_features,
            target_use_features=target_use_features,
            mask_prob=mask_prob,
            replace_prob=replace_prob,
            random_prob=random_prob,
            max_length=max_length,
            cls_token=cls_token,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            validation_split=validation_split,
            **kwargs
        )
        
        # ChemBL specific parameters
        self.chembl_smiles_path = chembl_smiles_path
        self.chembl_labels_path = chembl_labels_path
        self.chembl_target_names_path = chembl_target_names_path
        self.chembl_compound_names_path = chembl_compound_names_path
        self.test_split = test_split
        self.max_targets = max_targets
        self.min_target_activity = min_target_activity
        self.max_samples = max_samples
        
    def _load_chembl_smiles(self) -> List[str]:
        """Load SMILES from ChemBL pickle file."""
        print(f"Loading ChemBL SMILES from {self.chembl_smiles_path}")
        with open(self.chembl_smiles_path, 'rb') as f:
            smiles_data = pickle.load(f)
        
        if isinstance(smiles_data, list):
            # Limit dataset size if specified
            if self.max_samples is not None and len(smiles_data) > self.max_samples:
                print(f"Limiting dataset from {len(smiles_data)} to {self.max_samples} samples")
                smiles_data = smiles_data[:self.max_samples]
            return smiles_data
        else:
            raise ValueError(f"Expected list of SMILES, got {type(smiles_data)}")
    
    def setup(self, stage: str):
        """Setup datasets for train/val/test using ChemBL data."""
        
        if stage == "fit" or stage is None:
            # Load ChemBL SMILES
            all_smiles = self._load_chembl_smiles()
            
            # Split data
            total_size = len(all_smiles)
            test_size = int(total_size * self.test_split)
            val_size = int(total_size * self.validation_split)
            train_size = total_size - test_size - val_size
            
            train_smiles = all_smiles[:train_size]
            val_smiles = all_smiles[train_size:train_size + val_size]
            test_smiles = all_smiles[train_size + val_size:]
            
            print("Setting up ChemBL datasets:")
            print(f"  Training samples: {len(train_smiles)}")
            print(f"  Validation samples: {len(val_smiles)}")
            print(f"  Test samples: {len(test_smiles)}")
            
            # Create dataset kwargs
            dataset_kwargs = {
                "input_vocab_path": self.input_vocab_path,
                "target_vocab_path": self.target_vocab_path,
                "chembl_labels_path": self.chembl_labels_path,
                "chembl_target_names_path": self.chembl_target_names_path,
                "chembl_compound_names_path": self.chembl_compound_names_path,
                "max_targets": self.max_targets,
                "min_target_activity": self.min_target_activity,
                "input_radius": self.input_radius,
                "target_radius": self.target_radius,
                "input_use_features": self.input_use_features,
                "target_use_features": self.target_use_features,
                "mask_prob": self.mask_prob,
                "replace_prob": self.replace_prob,
                "random_prob": self.random_prob,
                "max_length": self.max_length,
                "cls_token": self.cls_token,
            }
            
            # Create datasets with correct indices for ChemBL data
            self.train_dataset = ChemBLDataset(
                smiles=train_smiles, 
                **dataset_kwargs
            )
            
            # Create validation dataset - adjust indices
            val_dataset_kwargs = dataset_kwargs.copy()
            val_start_idx = train_size
            self.val_dataset = ChemBLDatasetWithOffset(
                smiles=val_smiles,
                offset=val_start_idx,
                **val_dataset_kwargs
            )
            
            # Store test smiles for later
            self._test_smiles = test_smiles
            self._test_start_idx = train_size + val_size
            
        if stage == "test" and hasattr(self, '_test_smiles'):
            print(f"Setting up ChemBL test dataset: {len(self._test_smiles)} samples")
            
            dataset_kwargs = {
                "input_vocab_path": self.input_vocab_path,
                "target_vocab_path": self.target_vocab_path,
                "chembl_labels_path": self.chembl_labels_path,
                "chembl_target_names_path": self.chembl_target_names_path,
                "chembl_compound_names_path": self.chembl_compound_names_path,
                "max_targets": self.max_targets,
                "min_target_activity": self.min_target_activity,
                "input_radius": self.input_radius,
                "target_radius": self.target_radius,
                "input_use_features": self.input_use_features,
                "target_use_features": self.target_use_features,
                "mask_prob": self.mask_prob,
                "replace_prob": self.replace_prob,
                "random_prob": self.random_prob,
                "max_length": self.max_length,
                "cls_token": self.cls_token,
            }
            
            self.test_dataset = ChemBLDatasetWithOffset(
                smiles=self._test_smiles,
                offset=self._test_start_idx,
                **dataset_kwargs
            )
            
    def get_target_info(self) -> Dict:
        """Get information about ChemBL targets."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_target_info()
        else:
            return {}


class ChemBLDatasetWithOffset(ChemBLDataset):
    """
    ChemBL dataset that handles offset indices for validation/test splits.
    
    This ensures that we access the correct rows in the ChemBL labels matrix
    when using train/val/test splits.
    """
    
    def __init__(self, smiles: List[str], offset: int = 0, **kwargs):
        """
        Initialize with offset for accessing correct label matrix rows.
        
        Args:
            smiles: List of SMILES strings
            offset: Starting index in the original ChemBL data
            **kwargs: Other arguments passed to parent
        """
        self.offset = offset
        # Convert list to pandas Series if needed for compatibility with parent class
        if isinstance(smiles, list):
            import pandas as pd
            smiles = pd.Series(smiles)
        super().__init__(smiles=smiles, **kwargs)
        
    def __getitem__(self, idx: int):
        """Get item with offset applied to label matrix access."""
        # Get base MLM data using local idx
        data = super(ChemBLDataset, self).__getitem__(idx)  # Skip ChemBLDataset.__getitem__
        
        # If dummy sample, handle appropriately
        if hasattr(data, "dummy") and data.dummy:
            data.chembl_targets = torch.zeros(len(self.target_names), dtype=torch.long)
            data.chembl_mask = torch.zeros(len(self.target_names), dtype=torch.bool)
            data.num_targets = len(self.target_names)
            # Remove dummy attribute to avoid batching issues
            delattr(data, 'dummy')
            return data
            
        try:
            # Use offset to get correct ChemBL labels
            chembl_idx = idx + self.offset
            
            if sparse.issparse(self.labels_matrix):
                molecule_labels = self.labels_matrix[chembl_idx].toarray().flatten()
            else:
                molecule_labels = self.labels_matrix[chembl_idx]
                
            # Convert to PyTorch tensors
            chembl_targets = torch.tensor(molecule_labels, dtype=torch.long)
            chembl_mask = torch.tensor(molecule_labels != 0, dtype=torch.bool)
            
            # Add to data object
            data.chembl_targets = chembl_targets
            data.chembl_mask = chembl_mask
            data.num_targets = len(self.target_names)
            
        except Exception as e:
            print(f"Warning: Failed to add ChemBL targets for molecule {idx} (offset {self.offset}): {e}")
            return self.__handle_fail_case__(idx, data.smiles)
            
        return data 