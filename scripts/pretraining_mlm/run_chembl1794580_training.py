#!/usr/bin/env python3
"""
ChemBL1794580 Single-Target Training Script

Train on the CHEMBL1794580 single-target dataset using the corrected 3-state classification.
This script is specifically designed for single-target binary classification using CSV input.

The dataset contains:
- 63,916 compounds measured in CHEMBL1794580
- 17,962 active compounds (1)
- 45,954 inactive compounds (-1)
- Proper 3-state classification (0 values ignored)

Usage:
    python run_chembl1794580_training.py  # Train with MLM + classification
    python run_chembl1794580_training.py --chembl_only  # Train classification only
    python run_chembl1794580_training.py --resume_from_checkpoint path/to/checkpoint.ckpt  # Fine-tune
"""

import argparse
import warnings
import sys
import logging
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mole.data.crossenv_dataset import CrossEnvMolDataset
from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.models.chembl_model import ChemBLModel
from mole.models.chembl_lightning import ChemBLLightningModule


class ChemBL1794580Dataset(CrossEnvMolDataset):
    """Dataset for CHEMBL1794580 single-target classification from CSV."""
    
    def __init__(self, smiles: pd.Series, activity_values: pd.Series, **kwargs):
        """
        Initialize dataset with SMILES and activity values.
        
        Args:
            smiles: Series of SMILES strings
            activity_values: Series of activity values (1=active, -1=inactive, 0=not measured)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(smiles=smiles, **kwargs)
        
        # Convert activity values to tensor
        self.activity_values = torch.tensor(activity_values.values, dtype=torch.long)
        
        # Create mask for measured compounds (active or inactive)
        self.activity_mask = torch.tensor((activity_values != 0).values, dtype=torch.bool)
        
        print(f"Loaded CHEMBL1794580 dataset: {len(smiles)} compounds")
        print(f"Active compounds: {(activity_values == 1).sum()}")
        print(f"Inactive compounds: {(activity_values == -1).sum()}")
        print(f"Not measured: {(activity_values == 0).sum()}")
    
    def __getitem__(self, idx: int):
        """Get a single training sample."""
        # Get base MLM data from parent
        data = super().__getitem__(idx)
        
        # Add CHEMBL1794580 target data
        data.chembl_targets = torch.tensor([self.activity_values[idx]], dtype=torch.long)
        data.chembl_mask = torch.tensor([self.activity_mask[idx]], dtype=torch.bool)
        data.num_targets = 1
        
        return data


class ChemBL1794580DataModule(CrossEnvDataModule):
    """Data module for CHEMBL1794580 single-target training from CSV."""
    
    def __init__(
        self,
        csv_path: str,
        input_vocab_path: str,
        target_vocab_path: str,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        enable_masking: bool = True,
        **kwargs
    ):
        """
        Initialize data module.
        
        Args:
            csv_path: Path to CSV file with SMILES and activity columns
            input_vocab_path: Path to input vocabulary
            target_vocab_path: Path to target vocabulary
            validation_split: Fraction for validation
            test_split: Fraction for test
            batch_size: Batch size
            num_workers: Number of data loading workers
            enable_masking: Whether to enable MLM masking
            **kwargs: Additional arguments
        """
        # Initialize parent with placeholder data (we'll load CSV in setup)
        super().__init__(
            train_data="",  # Placeholder
            input_vocab_path=input_vocab_path,
            target_vocab_path=target_vocab_path,
            validation_split=validation_split,
            batch_size=batch_size,
            num_workers=num_workers,
            enable_masking=enable_masking,
            **kwargs
        )
        
        self.csv_path = csv_path
        self.test_split = test_split
        self.enable_masking = enable_masking
    
    def setup(self, stage: str):
        """Setup datasets from CSV file."""
        print(f"Loading data from {self.csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} compounds from CSV")
        
        # Validate required columns
        required_cols = ['smiles', 'activity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out compounds with 0 activity (not measured)
        measured_mask = df['activity'] != 0
        df_measured = df[measured_mask].copy()
        print(f"Using {len(df_measured)} compounds with measured activity")
        
        # Split data
        total_measured = len(df_measured)
        test_size = int(total_measured * self.test_split)
        val_size = int(total_measured * self.validation_split)
        train_size = total_measured - test_size - val_size
        
        # Create splits
        train_df = df_measured.iloc[:train_size]
        val_df = df_measured.iloc[train_size:train_size + val_size]
        test_df = df_measured.iloc[train_size + val_size:]
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ChemBL1794580Dataset(
                smiles=train_df['smiles'],
                activity_values=train_df['activity'],
                input_vocab_path=self.input_vocab_path,
                target_vocab_path=self.target_vocab_path,
                input_radius=self.input_radius,
                target_radius=self.target_radius,
                input_use_features=self.input_use_features,
                target_use_features=self.target_use_features,
                mask_prob=self.mask_prob,
                replace_prob=self.replace_prob,
                random_prob=self.random_prob,
                max_length=self.max_length,
                cls_token=self.cls_token,
                enable_masking=self.enable_masking,
            )
            
            self.val_dataset = ChemBL1794580Dataset(
                smiles=val_df['smiles'],
                activity_values=val_df['activity'],
                input_vocab_path=self.input_vocab_path,
                target_vocab_path=self.target_vocab_path,
                input_radius=self.input_radius,
                target_radius=self.target_radius,
                input_use_features=self.input_use_features,
                target_use_features=self.target_use_features,
                mask_prob=self.mask_prob,
                replace_prob=self.replace_prob,
                random_prob=self.random_prob,
                max_length=self.max_length,
                cls_token=self.cls_token,
                enable_masking=False,  # No masking for validation
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ChemBL1794580Dataset(
                smiles=test_df['smiles'],
                activity_values=test_df['activity'],
                input_vocab_path=self.input_vocab_path,
                target_vocab_path=self.target_vocab_path,
                input_radius=self.input_radius,
                target_radius=self.target_radius,
                input_use_features=self.input_use_features,
                target_use_features=self.target_use_features,
                mask_prob=self.mask_prob,
                replace_prob=self.replace_prob,
                random_prob=self.random_prob,
                max_length=self.max_length,
                cls_token=self.cls_token,
                enable_masking=False,  # No masking for test
            )
    
    def get_vocab_sizes(self):
        """Get vocabulary sizes."""
        return {
            'input_vocab_size': self.train_dataset.input_vocab_size,
            'target_vocab_size': self.train_dataset.target_vocab_size,
        }
    
    def get_target_info(self):
        """Get target information."""
        return {
            'num_targets': 1,
            'target_names': ['CHEMBL1794580'],
        }


class ChemBL1794580Config:
    """Configuration for CHEMBL1794580 single-target training."""

    def __init__(self, chembl_only: bool = False):
        self.chembl_only = chembl_only
        self.config_name = "chembl1794580_only" if chembl_only else "chembl1794580"

    def get_config(self, overrides=None):
        """Get configuration for CHEMBL1794580 training."""
        
        # Base configuration for single-target training
        config = {
                    # Data paths - using CSV file
        "--csv_path": "../../examples_data_creation/chembl_single_target/chembl_CHEMBL1794580_full_dataset.csv",
            "--input_vocab": "../../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
            "--target_vocab": "../../mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
            
            # Environment configuration
            "--input_radius": "0",
            "--target_radius": "1",
            "--target_use_features": "",  # Flag for functional environments
            
            # Training settings
            "--validation_split": "0.2",
            "--test_split": "0.1",
            "--dropout": "0.1",
            "--classifier_dropout": "0.1",
            "--weight_decay": "0.01",
            "--gradient_clip_val": "1.0",
            "--seed": "42",
            "--log_predictions": "",
            "--log_target_metrics": "",
            
            # Model architecture (optimized for single target)
            "--hidden_size": "512",
            "--num_hidden_layers": "8",
            "--num_attention_heads": "8",
            "--intermediate_size": "2048",
            "--max_length": "256",
            
            # Training parameters
            "--batch_size": "32",
            "--learning_rate": "1e-4",
            "--warmup_steps": "1000",
            "--max_epochs": "50",
            "--val_check_interval": "0.25",
            "--patience": "10",
            "--gpus": "1",
            "--num_workers": "4",
            "--precision": "16",
            "--accumulate_grad_batches": "4",
            
            # Output settings
            "--output_dir": "outputs/chembl1794580_single_target",
            "--model_name": "chembl1794580_binary_classifier",
            "--max_targets_to_log": "1",
        }
        
        # MLM vs Classification-only settings
        if self.chembl_only:
            config.update({
                "--mlm_loss_weight": "0.0",
                "--classification_loss_weight": "1.0",
                "--output_dir": "outputs/chembl1794580_single_target_only",
                "--model_name": "chembl1794580_classification_only",
                "--batch_size": "64",  # Can use larger batch without MLM
                "--learning_rate": "2e-4",
                "--warmup_steps": "500",
                "--max_epochs": "30",
                "--patience": "8",
                "--accumulate_grad_batches": "2",
            })
        else:
            config.update({
                "--mlm_loss_weight": "0.1",
                "--classification_loss_weight": "1.0",
            })
        
        # Apply any overrides
        if overrides:
            config.update(overrides)
            
        return config

    def print_config_summary(self, config, title):
        """Print a formatted summary of the configuration."""
        print(f"\n🔬 {title}")
        print("=" * 60)
        print("📊 Dataset: CHEMBL1794580 Single-Target")
        print("   • 63,916 compounds measured in CHEMBL1794580")
        print("   • 17,962 active compounds (1)")
        print("   • 45,954 inactive compounds (-1)")
        print("   • 3-state classification: 1 (active), -1 (inactive), 0 (not measured)")
        print("   • Only active/inactive compounds used for training (0 values ignored)")
        print(f"   • Data path: {config['--csv_path']}")
        print()
        print("🎯 Model Configuration:")
        print(f"   • Hidden size: {config['--hidden_size']}")
        print(f"   • Layers: {config['--num_hidden_layers']}")
        print(f"   • Attention heads: {config['--num_attention_heads']}")
        print(f"   • Batch size: {config['--batch_size']} × {config['--accumulate_grad_batches']} = {int(config['--batch_size']) * int(config['--accumulate_grad_batches'])} effective")
        print()
        print("⚙️ Training Configuration:")
        print(f"   • Learning rate: {config['--learning_rate']}")
        print(f"   • Warmup steps: {config['--warmup_steps']}")
        print(f"   • Max epochs: {config['--max_epochs']}")
        if not self.chembl_only:
            print(f"   • MLM weight: {config['--mlm_loss_weight']}")
        print(f"   • Classification weight: {config['--classification_loss_weight']}")
        print(f"   • Output: {config['--output_dir']}")
        print()


def run_chembl1794580_training(config_class, title, pretrained_path=None, freeze_encoder=False):
    """Run CHEMBL1794580 training with custom data module."""
    
    # Get configuration
    config = config_class.get_config()
    
    # Print configuration summary
    config_class.print_config_summary(config, title)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set seed
    pl.seed_everything(int(config['--seed']), workers=True)
    
    # Create output directory
    output_dir = Path(config['--output_dir']) / config['--model_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting CHEMBL1794580 training")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CSV path: {config['--csv_path']}")
    
    # Create data module
    logger.info("Setting up data module...")
    data_module = ChemBL1794580DataModule(
        csv_path=config['--csv_path'],
        input_vocab_path=config['--input_vocab'],
        target_vocab_path=config['--target_vocab'],
        validation_split=float(config['--validation_split']),
        test_split=float(config['--test_split']),
        batch_size=int(config['--batch_size']),
        num_workers=int(config['--num_workers']),
        enable_masking=not config_class.chembl_only,
        input_radius=int(config['--input_radius']),
        target_radius=int(config['--target_radius']),
        target_use_features=bool(config['--target_use_features']),
        max_length=int(config['--max_length']) if config['--max_length'] else None,
    )
    
    # Setup data to get vocabulary sizes
    data_module.setup("fit")
    vocab_sizes = data_module.get_vocab_sizes()
    target_info = data_module.get_target_info()
    
    logger.info(f"Input vocabulary size: {vocab_sizes['input_vocab_size']}")
    logger.info(f"Target vocabulary size: {vocab_sizes['target_vocab_size']}")
    logger.info(f"Number of targets: {target_info['num_targets']}")
    
    # Create model configuration
    model_config = {
        "hidden_size": int(config['--hidden_size']),
        "num_hidden_layers": int(config['--num_hidden_layers']),
        "num_attention_heads": int(config['--num_attention_heads']),
        "intermediate_size": int(config['--intermediate_size']),
        "hidden_dropout_prob": float(config['--dropout']),
        "attention_probs_dropout_prob": float(config['--dropout']),
        "max_position_embeddings": 512,
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-7,
        "relative_attention": True,
        "max_relative_positions": 64,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "position_biased_input": False,
        "pos_att_type": "p2c|c2p",
        "norm_rel_ebd": "layer_norm",
        "hidden_act": "gelu",
        "attention_head_type": "disentangled",
        "classifier_dropout": float(config['--classifier_dropout']),
    }
    
    # Create model
    logger.info("Creating model...")
    model = ChemBLModel(
        deberta_config=model_config,
        input_vocab_size=vocab_sizes['input_vocab_size'],
        target_vocab_size=vocab_sizes['target_vocab_size'],
        num_targets=target_info['num_targets'],
        hidden_dropout_prob=float(config['--dropout']),
        classifier_dropout_prob=float(config['--classifier_dropout']),
    )
    
    # Load pretrained weights if specified
    if pretrained_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if freeze_encoder:
            logger.info("Freezing encoder layers")
            for param in model.deberta.parameters():
                param.requires_grad = False
    

    
    # Create Lightning module
    lightning_module = ChemBLLightningModule(
        model=model,
        optimizer_cfg={
            'optimizer': 'AdamW',
            'lr': float(config['--learning_rate']),
            'weight_decay': float(config['--weight_decay']),
            'lr_scheduler': 'LambdaLR',
            'warmup_steps': int(config['--warmup_steps']),
        },
        mlm_loss_weight=float(config['--mlm_loss_weight']),
        classification_loss_weight=float(config['--classification_loss_weight']),
        log_predictions=bool(config['--log_predictions']),
        log_target_metrics=bool(config['--log_target_metrics']),
        max_targets_to_log=int(config['--max_targets_to_log']),
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='best_model',
            monitor='val/classification/auroc',
            mode='max',
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val/classification/auroc',
            mode='max',
            patience=int(config['--patience']),
        ),
    ]
    
    # Create logger
    logger_tb = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
        version=None,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=int(config['--max_epochs']),
        accelerator='gpu' if int(config['--gpus']) > 0 else 'cpu',
        devices=int(config['--gpus']) if int(config['--gpus']) > 0 else None,
        precision=int(config['--precision']),
        accumulate_grad_batches=int(config['--accumulate_grad_batches']),
        val_check_interval=float(config['--val_check_interval']),
        gradient_clip_val=float(config['--gradient_clip_val']),
        callbacks=callbacks,
        logger=logger_tb,
        log_every_n_steps=10,
        deterministic=False,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(lightning_module, data_module)
    
    # Test
    logger.info("Running test...")
    data_module.setup("test")
    trainer.test(lightning_module, data_module)
    
    logger.info(f"Training completed. Model saved to {output_dir}")


def main():
    """Main function to run CHEMBL1794580 single-target training."""
    parser = argparse.ArgumentParser(
        description="Train single-target model on CHEMBL1794580 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--chembl_only",
        action="store_true",
        help="Use ChemBL-only configuration (no MLM, classification only)",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder layers when resuming from checkpoint (for fine-tuning)",
    )

    # Parse known args to allow passing through other arguments
    args, unknown_args = parser.parse_known_args()

    # Add unknown args back to sys.argv for the training script
    sys.argv = [sys.argv[0]] + unknown_args

    # Validate that the dataset file exists
    dataset_path = Path("../../examples_data_creation/chembl_single_target/chembl_CHEMBL1794580_full_dataset.csv")
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        print("Please ensure the CHEMBL1794580 dataset has been created using the extract_single_target.py script.")
        sys.exit(1)

    # Choose configuration based on flags
    config_class = ChemBL1794580Config(chembl_only=args.chembl_only)
    
    if args.chembl_only:
        title = "CHEMBL1794580 Single-Target Training (Classification Only)"
    else:
        title = "CHEMBL1794580 Single-Target Training (MLM + Classification)"

    # Run training
    run_chembl1794580_training(
        config_class=config_class,
        title=title,
        pretrained_path=args.resume_from_checkpoint,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
