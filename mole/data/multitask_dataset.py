"""
Multi-Task Dataset for MolE Pretraining

This dataset extends the cross-environment setup to include multi-task learning.
In addition to the MLM task, it adds targets for predicting molecular properties.

- MLM Input: Radius 0 structural atom environments
- MLM Target: Radius 1 functional atom environments
- Regression Target 1: ClogP (logarithm of partition coefficient)
- Regression Target 2: Molecular Weight
"""

from typing import Dict
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from torch_geometric.data import Data

from mole.data.crossenv_dataset import CrossEnvMolDataset


class MultiTaskMolDataset(CrossEnvMolDataset):
    """Dataset for multi-task pretraining (MLM + property prediction)"""

    def __getitem__(self, idx: int) -> Data:
        """Get a single training sample, including MLM and regression targets"""
        # Get the base data object from the parent class (handles MLM)
        data = super().__getitem__(idx)

        # If the base class returned a dummy sample, add placeholder properties and pass through
        if hasattr(data, "dummy") and data.dummy:
            data.clogp = torch.tensor(0.0, dtype=torch.float)
            data.mw = torch.tensor(0.0, dtype=torch.float)
            return data

        try:
            # The base class already parsed the molecule, so we can reuse it if available.
            # However, to be safe and decoupled, we'll re-parse from SMILES.
            smiles_str = data.smiles
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                raise ValueError(f"Invalid SMILES in base data: {smiles_str}")

            # Calculate ClogP and Molecular Weight
            clogp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)

            # Add properties to the data object
            data.clogp = torch.tensor(clogp, dtype=torch.float)
            data.mw = torch.tensor(mw, dtype=torch.float)

        except Exception as e:
            # In case of any error during property calculation, call the fail-case handler
            print(f"Warning: Failed to add properties for molecule {idx}: {e}")
            return self.__handle_fail_case__(idx, data.smiles)

        return data

    def __handle_fail_case__(self, idx: int, smiles_str: str) -> Data:
        """Creates a dummy sample for a molecule that failed to process"""
        # Get a dummy sample from the parent and add dummy properties
        data = super().__handle_fail_case__(idx, smiles_str)
        data.clogp = torch.tensor(0.0, dtype=torch.float)
        data.mw = torch.tensor(0.0, dtype=torch.float)
        return data 