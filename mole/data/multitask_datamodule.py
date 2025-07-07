"""
Multi-Task Data Module for MolE Pretraining

Handles data loading, preprocessing, and batching for multi-task training,
combining cross-environment MLM with molecular property prediction.
"""

from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.data.multitask_dataset import MultiTaskMolDataset


class MultiTaskDataModule(CrossEnvDataModule):
    """Lightning data module for multi-task training"""

    def setup(self, stage: str):
        """Setup datasets for train/val/test using the MultiTaskMolDataset"""

        # This method overrides the parent setup to instantiate MultiTaskMolDataset
        # instead of CrossEnvMolDataset. The rest of the logic is inherited.

        if stage == "fit" or stage is None:
            train_smiles, val_smiles = self._get_train_val_smiles()

            print("Setting up multi-task datasets:")
            print(f"  Training samples: {len(train_smiles)}")
            print(f"  Validation samples: {len(val_smiles)}")

            # Create datasets using the new MultiTaskMolDataset
            dataset_kwargs = self._get_dataset_kwargs()
            self.train_dataset = MultiTaskMolDataset(
                smiles=train_smiles, **dataset_kwargs
            )
            self.val_dataset = MultiTaskMolDataset(smiles=val_smiles, **dataset_kwargs)

        if stage == "test" and self.test_data is not None:
            test_smiles = self._load_smiles_data(self.test_data)
            print(f"Setting up multi-task test dataset: {len(test_smiles)} samples")
            dataset_kwargs = self._get_dataset_kwargs()
            self.test_dataset = MultiTaskMolDataset(
                smiles=test_smiles, **dataset_kwargs
            )

    def _get_train_val_smiles(self):
        """Helper to load and split train/validation smiles data"""
        train_smiles = self._load_smiles_data(self.train_data)
        if self.validation_data is not None:
            val_smiles = self._load_smiles_data(self.validation_data)
        else:
            split_idx = int(len(train_smiles) * (1 - self.validation_split))
            val_smiles = train_smiles.iloc[split_idx:].reset_index(drop=True)
            train_smiles = train_smiles.iloc[:split_idx].reset_index(drop=True)
        return train_smiles, val_smiles

    def _get_dataset_kwargs(self):
        """Helper to gather dataset constructor arguments"""
        return {
            "input_vocab_path": self.input_vocab_path,
            "target_vocab_path": self.target_vocab_path,
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