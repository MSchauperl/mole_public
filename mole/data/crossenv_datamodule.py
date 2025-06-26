"""
Cross-Environment Data Module for MolE Pretraining

Handles data loading, preprocessing, and batching for cross-environment
masked language modeling where input and target use different atom environments.
"""

from typing import Optional, Union, List
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from mole.data.crossenv_dataset import CrossEnvMolDataset


class CrossEnvDataModule(pl.LightningDataModule):
    """Lightning data module for cross-environment MLM training"""

    def __init__(
        self,
        train_data: Union[str, pd.Series, List[str]],
        input_vocab_path: str,
        target_vocab_path: str,
        validation_data: Optional[Union[str, pd.Series, List[str]]] = None,
        test_data: Optional[Union[str, pd.Series, List[str]]] = None,
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
        validation_split: float = 0.1,
        **kwargs,
    ):
        """
        Initialize cross-environment data module

        Args:
            train_data: Training SMILES data (file path, Series, or list)
            input_vocab_path: Path to input vocabulary file
            target_vocab_path: Path to target vocabulary file
            validation_data: Validation SMILES data (optional)
            test_data: Test SMILES data (optional)
            input_radius: Radius for input atom environments
            target_radius: Radius for target atom environments
            input_use_features: Whether to use features for input environments
            target_use_features: Whether to use features for target environments
            mask_prob: Probability of masking a token
            replace_prob: Probability of replacing with [MASK]
            random_prob: Probability of replacing with random token
            max_length: Maximum sequence length
            cls_token: Whether to add CLS token
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            validation_split: Fraction of train data to use for validation (if no val data provided)
        """
        super().__init__()

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.input_vocab_path = input_vocab_path
        self.target_vocab_path = target_vocab_path

        # Environment configuration
        self.input_radius = input_radius
        self.target_radius = target_radius
        self.input_use_features = input_use_features
        self.target_use_features = target_use_features

        # MLM configuration
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.max_length = max_length
        self.cls_token = cls_token

        # DataLoader configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.validation_split = validation_split

        # Store datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.prepare_data_per_node = False

    def _load_smiles_data(self, data: Union[str, pd.Series, List[str]]) -> pd.Series:
        """Load SMILES data from various formats"""
        if isinstance(data, str):
            # Assume it's a file path
            if data.endswith(".parquet"):
                df = pd.read_parquet(data)
            elif data.endswith(".csv"):
                df = pd.read_csv(data)
            else:
                # Try to read as text file with one SMILES per line
                with open(data, "r") as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                return pd.Series(smiles_list, dtype="string[pyarrow]")

            # Look for SMILES column
            smiles_columns = ["smiles", "SMILES", "Smiles", "canonical_smiles"]
            smiles_col = None
            for col in smiles_columns:
                if col in df.columns:
                    smiles_col = col
                    break

            if smiles_col is None:
                # Use first column as SMILES
                smiles_col = df.columns[0]
                print(f"Warning: No standard SMILES column found, using '{smiles_col}'")

            return df[smiles_col].astype("string[pyarrow]")

        elif isinstance(data, pd.Series):
            return data.astype("string[pyarrow]")

        elif isinstance(data, list):
            return pd.Series(data, dtype="string[pyarrow]")

        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

    def setup(self, stage: str):
        """Setup datasets for train/val/test"""

        if stage == "fit" or stage is None:
            # Load training data
            train_smiles = self._load_smiles_data(self.train_data)

            # Handle validation data
            if self.validation_data is not None:
                val_smiles = self._load_smiles_data(self.validation_data)
            else:
                # Split training data
                split_idx = int(len(train_smiles) * (1 - self.validation_split))
                val_smiles = train_smiles.iloc[split_idx:].reset_index(drop=True)
                train_smiles = train_smiles.iloc[:split_idx].reset_index(drop=True)

            print("Setting up datasets:")
            print(f"  Training samples: {len(train_smiles)}")
            print(f"  Validation samples: {len(val_smiles)}")

            # Create datasets
            dataset_kwargs = {
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

            self.train_dataset = CrossEnvMolDataset(
                smiles=train_smiles, **dataset_kwargs
            )

            self.val_dataset = CrossEnvMolDataset(smiles=val_smiles, **dataset_kwargs)

        if stage == "test" and self.test_data is not None:
            test_smiles = self._load_smiles_data(self.test_data)

            print(f"Setting up test dataset: {len(test_smiles)} samples")

            dataset_kwargs = {
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

            self.test_dataset = CrossEnvMolDataset(smiles=test_smiles, **dataset_kwargs)

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Return test dataloader"""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def get_vocab_sizes(self):
        """Get vocabulary sizes from datasets"""
        if self.train_dataset is not None:
            return {
                "input_vocab_size": self.train_dataset.input_vocab_size,
                "target_vocab_size": self.train_dataset.target_vocab_size,
            }
        else:
            # Load vocabularies to get sizes
            from mole.data.vocabulary import open_dictionary

            input_vocab = open_dictionary(self.input_vocab_path)
            target_vocab = open_dictionary(self.target_vocab_path)

            return {
                "input_vocab_size": len(input_vocab),
                "target_vocab_size": len(target_vocab),
            }

    def get_sample_batch(self):
        """Get a sample batch for model initialization"""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before get_sample_batch()")

        # Create a temporary dataloader for one batch
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=min(2, len(self.train_dataset)),
            shuffle=False,
            num_workers=0,
        )

        return next(iter(temp_loader))
