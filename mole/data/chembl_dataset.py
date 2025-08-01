"""
ChemBL Dataset for MolE Pretraining

This dataset extends the cross-environment setup to include multi-task learning
with ChemBL binary classification tasks instead of molecular properties.

- MLM Input: Radius 0 structural atom environments  
- MLM Target: Radius 1 functional atom environments
- Classification Targets: ChemBL assay results (1=active, -1=inactive, 0=not_measured)
"""

import pickle
from typing import Dict, Optional, List
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from scipy import sparse

from mole.data.crossenv_dataset import CrossEnvMolDataset


class ChemBLDataset(CrossEnvMolDataset):
    """
    Dataset for ChemBL-based pretraining combining MLM with binary classification tasks.
    
    Instead of using molecular properties like clogP and MW, this dataset uses
    ChemBL assay results for multi-target binary classification.
    """

    def __init__(
        self,
        smiles: List[str],
        chembl_labels_path: str,
        chembl_target_names_path: str,
        chembl_compound_names_path: Optional[str] = None,
        max_targets: Optional[int] = None,
        min_target_activity: int = 10,
        **kwargs
    ):
        """
        Initialize ChemBL dataset.
        
        Args:
            smiles: List of SMILES strings
            chembl_labels_path: Path to ChemBL labels sparse matrix (.pckl)
            chembl_target_names_path: Path to target names file  
            chembl_compound_names_path: Path to compound names file (optional)
            max_targets: Maximum number of targets to use (for memory efficiency)
            min_target_activity: Minimum number of active/inactive labels per target
            **kwargs: Additional arguments passed to parent class
        """
        # Convert list to pandas Series if needed for compatibility with parent class
        if isinstance(smiles, list):
            smiles = pd.Series(smiles)
        super().__init__(smiles=smiles, **kwargs)
        
        self.chembl_labels_path = chembl_labels_path
        self.chembl_target_names_path = chembl_target_names_path
        self.chembl_compound_names_path = chembl_compound_names_path
        self.max_targets = max_targets
        self.min_target_activity = min_target_activity
        
        # Load ChemBL data
        self._load_chembl_data()
        
    def _load_chembl_data(self):
        """Load and preprocess ChemBL data."""
        print("Loading ChemBL data...")
        
        # Load labels matrix
        with open(self.chembl_labels_path, 'rb') as f:
            self.labels_matrix = pickle.load(f)
        
        # Load target names
        with open(self.chembl_target_names_path, 'r') as f:
            self.target_names = f.read().strip().split('\n')
            
        # Load compound names if provided
        if self.chembl_compound_names_path:
            with open(self.chembl_compound_names_path, 'r') as f:
                self.compound_names = f.read().strip().split('\n')
        else:
            self.compound_names = None
            
        print(f"Loaded ChemBL data: {self.labels_matrix.shape[0]} compounds, {self.labels_matrix.shape[1]} targets")
        
        # Filter targets by activity if specified
        if self.min_target_activity > 0:
            self._filter_targets_by_activity()
            
        # Limit number of targets if specified
        if self.max_targets is not None and self.max_targets < len(self.target_names):
            self._limit_targets()
            
        print(f"Using {len(self.target_names)} targets after filtering")
        
    def _filter_targets_by_activity(self):
        """Filter targets to only include those with sufficient activity data."""
        print(f"Filtering targets by minimum activity ({self.min_target_activity})...")
        
        # Count non-zero (active/inactive) labels per target
        active_counts = np.array((self.labels_matrix == 1).sum(axis=0)).flatten()
        inactive_counts = np.array((self.labels_matrix == -1).sum(axis=0)).flatten()
        total_activity = active_counts + inactive_counts
        
        # Keep targets with sufficient activity
        valid_targets = total_activity >= self.min_target_activity
        
        print(f"Targets before filtering: {len(self.target_names)}")
        print(f"Targets after filtering: {valid_targets.sum()}")
        
        # Filter data
        self.labels_matrix = self.labels_matrix[:, valid_targets]
        self.target_names = [name for i, name in enumerate(self.target_names) if valid_targets[i]]
        
    def _limit_targets(self):
        """Limit the number of targets for memory efficiency."""
        print(f"Limiting targets to {self.max_targets}...")
        
        # Take first max_targets targets (could be made smarter with target selection)
        self.labels_matrix = self.labels_matrix[:, :self.max_targets]
        self.target_names = self.target_names[:self.max_targets]
        
    def __getitem__(self, idx: int) -> Data:
        """Get a single training sample, including MLM and ChemBL classification targets."""
        # Get the base data object from the parent class (handles MLM)
        data = super().__getitem__(idx)
        
        # If the base class returned a dummy sample, add placeholder targets and pass through
        if hasattr(data, "dummy") and data.dummy:
            data.chembl_targets = torch.zeros(len(self.target_names), dtype=torch.long)
            data.chembl_mask = torch.zeros(len(self.target_names), dtype=torch.bool)
            data.num_targets = len(self.target_names)
            # Remove dummy attribute to avoid batching issues
            delattr(data, 'dummy')
            return data
            
        try:
            # Get ChemBL labels for this molecule
            if sparse.issparse(self.labels_matrix):
                molecule_labels = self.labels_matrix[idx].toarray().flatten()
            else:
                molecule_labels = self.labels_matrix[idx]
                
            # Convert to PyTorch tensors
            # Labels: 1=active, -1=inactive, 0=not_measured
            chembl_targets = torch.tensor(molecule_labels, dtype=torch.long)
            
            # Create mask: True for measured (active/inactive), False for not measured
            chembl_mask = torch.tensor(molecule_labels != 0, dtype=torch.bool)
            
            # Add to data object
            data.chembl_targets = chembl_targets
            data.chembl_mask = chembl_mask
            data.num_targets = len(self.target_names)
            
        except Exception as e:
            print(f"Warning: Failed to add ChemBL targets for molecule {idx}: {e}")
            return self.__handle_fail_case__(idx, data.smiles)
            
        return data
        
    def __handle_fail_case__(self, idx: int, smiles_str: str) -> Data:
        """Creates a dummy sample for a molecule that failed to process."""
        # Get a dummy sample from the parent and add dummy ChemBL targets
        data = super().__handle_fail_case__(idx, smiles_str)
        data.chembl_targets = torch.zeros(len(self.target_names), dtype=torch.long)
        data.chembl_mask = torch.zeros(len(self.target_names), dtype=torch.bool)
        data.num_targets = len(self.target_names)
        # Remove dummy attribute to avoid batching issues
        if hasattr(data, 'dummy'):
            delattr(data, 'dummy')
        return data
        
    def get_target_info(self) -> Dict:
        """Get information about ChemBL targets."""
        info = {
            'num_targets': len(self.target_names),
            'target_names': self.target_names,
        }
        
        if hasattr(self, 'labels_matrix'):
            # Calculate target statistics
            active_counts = np.array((self.labels_matrix == 1).sum(axis=0)).flatten()
            inactive_counts = np.array((self.labels_matrix == -1).sum(axis=0)).flatten()
            
            info.update({
                'active_counts': active_counts.tolist(),
                'inactive_counts': inactive_counts.tolist(),
                'total_activity_counts': (active_counts + inactive_counts).tolist(),
            })
            
        return info 