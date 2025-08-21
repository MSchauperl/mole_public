#!/usr/bin/env python3
"""
Unified ChemBL Training Script (CSV-based)

A simplified script that allows you to choose:
- Dataset: Full ChemBL CSV, Filtered ChemBL CSV, or Test dataset
- Splits: Random splits or Fold-based splits
- MLM: Enable/disable masked language modeling
- Fine-tuning: Resume from checkpoint with optional encoder freezing

The script now uses CSV files instead of pickle files for better compatibility.

Usage examples:
    # Train on full ChemBL CSV with MLM and random splits
    python run_chembl_unified.py --dataset full --mlm true --splits random

    # Train on filtered ChemBL CSV with classification only and fold-based splits
    python run_chembl_unified.py --dataset filtered --mlm false --splits folds

    # Train on corrected filtered ChemBL CSV with proper 3-state classification
    python run_chembl_unified.py --dataset filtered_corrected --mlm true --splits folds

    # Fine-tune a pretrained model with frozen encoder
    python run_chembl_unified.py --dataset filtered_corrected --mlm true --splits folds \
        --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt --freeze_encoder

    # Quick test run (MLM enabled by default)
    python run_chembl_unified.py --dataset test --splits random
"""

import argparse
import sys
import warnings
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


class ChemBLCSVDataset(CrossEnvMolDataset):
    """Dataset for ChemBL multi-target classification from CSV."""
    
    def __init__(self, smiles: pd.Series, activity_columns: pd.DataFrame, fold_assignments: pd.Series = None, **kwargs):
        """
        Initialize dataset with SMILES and activity values.
        
        Args:
            smiles: Series of SMILES strings
            activity_columns: DataFrame with activity columns (1=active, -1=inactive, 0=not measured)
            fold_assignments: Series of fold assignments (optional)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(smiles=smiles, **kwargs)
        
        # Convert activity values to tensor
        self.activity_values = torch.tensor(activity_columns.values, dtype=torch.long)
        
        # Create mask for measured compounds (active or inactive)
        self.activity_mask = torch.tensor((activity_columns != 0).values, dtype=torch.bool)
        
        # Store fold assignments if provided
        self.fold_assignments = fold_assignments.values if fold_assignments is not None else None
        
        # Store target names
        self.target_names = list(activity_columns.columns)
        self.num_targets = len(self.target_names)
        
        print(f"Loaded ChemBL CSV dataset: {len(smiles)} compounds, {self.num_targets} targets")
        
        # Print activity statistics
        for i, target_name in enumerate(self.target_names):
            target_values = activity_columns.iloc[:, i]
            active_count = (target_values == 1).sum()
            inactive_count = (target_values == -1).sum()
            missing_count = (target_values == 0).sum()
            print(f"  {target_name}: {active_count} active, {inactive_count} inactive, {missing_count} missing")
    
    def __getitem__(self, idx: int):
        """Get a single training sample."""
        # Get base MLM data from parent
        data = super().__getitem__(idx)
        
        # Add ChemBL target data
        data.chembl_targets = self.activity_values[idx]
        data.chembl_mask = self.activity_mask[idx]
        data.num_targets = self.num_targets
        
        return data


class ChemBLCSVDataModule(CrossEnvDataModule):
    """Data module for ChemBL multi-target training from CSV."""
    
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
        use_fold_splits: bool = False,
        test_fold: int = 0,
        **kwargs
    ):
        """
        Initialize data module.
        
        Args:
            csv_path: Path to CSV file with SMILES and activity columns
            input_vocab_path: Path to input vocabulary
            target_vocab_path: Path to target vocabulary
            validation_split: Fraction for validation (used only if use_fold_splits=False)
            test_split: Fraction for test (used only if use_fold_splits=False)
            batch_size: Batch size
            num_workers: Number of data loading workers
            enable_masking: Whether to enable MLM masking
            use_fold_splits: Whether to use fold-based splits
            test_fold: Which fold to use as validation (0, 1, or 2)
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
        self.use_fold_splits = use_fold_splits
        self.test_fold = test_fold
    
    def setup(self, stage: str):
        """Setup datasets from CSV file."""
        print(f"Loading data from {self.csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} compounds from CSV")
        
        # Validate required columns
        required_cols = ['smiles']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get activity columns (columns starting with 'activity_')
        activity_cols = [col for col in df.columns if col.startswith('activity_')]
        if not activity_cols:
            # Fallback: support full CSVs that use 'assay_' prefix
            assay_cols = [col for col in df.columns if col.startswith('assay_')]
            if assay_cols:
                activity_cols = assay_cols
                print(f"No 'activity_' columns found. Falling back to 'assay_' columns ({len(activity_cols)} found)")
            else:
                raise ValueError("No activity columns found (should start with 'activity_' or 'assay_')")
        
        print(f"Found {len(activity_cols)} activity columns: {activity_cols}")
        
        # Get fold column if available
        fold_col = None
        if 'fold' in df.columns:
            fold_col = df['fold']
            print(f"Using fold column: {fold_col.unique()}")
        
        # Split data
        if self.use_fold_splits and fold_col is not None:
            # Use fold-based splits
            train_df, val_df, test_df = self._create_fold_splits(df, fold_col)
        else:
            # Use random splits
            train_df, val_df, test_df = self._create_random_splits(df)
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ChemBLCSVDataset(
                smiles=train_df['smiles'],
                activity_columns=train_df[activity_cols],
                fold_assignments=train_df.get('fold', None),
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
            
            self.val_dataset = ChemBLCSVDataset(
                smiles=val_df['smiles'],
                activity_columns=val_df[activity_cols],
                fold_assignments=val_df.get('fold', None),
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
            self.test_dataset = ChemBLCSVDataset(
                smiles=test_df['smiles'],
                activity_columns=test_df[activity_cols],
                fold_assignments=test_df.get('fold', None),
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
    
    def _create_fold_splits(self, df, fold_col):
        """Create fold-based splits."""
        print(f"Creating fold-based splits (using fold_{self.test_fold} as validation)")
        
        # Get validation fold
        val_fold = f'fold_{self.test_fold}'
        val_df = df[fold_col == val_fold].copy()
        
        # Get training folds (all other folds)
        train_df = df[fold_col != val_fold].copy()
        
        # For test, use a portion of the validation fold
        test_size = int(len(val_df) * 0.5)  # Use 50% of validation for test
        test_df = val_df.iloc[:test_size].copy()
        val_df = val_df.iloc[test_size:].copy()
        
        return train_df, val_df, test_df
    
    def _create_random_splits(self, df):
        """Create random splits."""
        print("Creating random splits")
        
        total_size = len(df)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - test_size - val_size
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        
        return train_df, val_df, test_df
    
    def get_vocab_sizes(self):
        """Get vocabulary sizes."""
        return {
            'input_vocab_size': self.train_dataset.input_vocab_size,
            'target_vocab_size': self.train_dataset.target_vocab_size,
        }
    
    def get_target_info(self):
        """Get target information."""
        return {
            'num_targets': self.train_dataset.num_targets,
            'target_names': self.train_dataset.target_names,
        }


def create_csv_config(dataset_type: str, mlm: bool, splits: str):
    """Create configuration for CSV-based training."""
    
    if dataset_type == "full":
        csv_path = "../../examples_data_creation/chembl_complete_dataset_full.csv"
        title_suffix = "Full ChemBL CSV"
    elif dataset_type == "filtered":
        csv_path = "../../examples_data_creation/chembl_single_target/chembl_multi_target_dataset.csv"
        title_suffix = "Filtered ChemBL CSV (7 assays)"
    elif dataset_type == "filtered_corrected":
        csv_path = "../../examples_data_creation/chembl_single_target/chembl_multi_target_dataset.csv"
        title_suffix = "Filtered ChemBL CSV (Corrected 3-state)"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    config = {
        # Data paths
        "--csv_path": csv_path,
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
        
        # Model architecture
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
        "--output_dir": f"outputs/chembl_csv_{dataset_type}",
        "--model_name": f"chembl_csv_{dataset_type}_model",
        "--max_targets_to_log": "10",
        
        # Split configuration
        "--use_fold_splits": "true" if splits == "folds" else "false",
        "--test_fold": "0",
    }
    
    # MLM vs Classification-only settings
    if mlm:
        config.update({
            "--mlm_loss_weight": "0.1",
            "--classification_loss_weight": "1.0",
        })
    else:
        config.update({
            "--mlm_loss_weight": "0.0",
            "--classification_loss_weight": "1.0",
            "--batch_size": "16",  # Can use larger batch without MLM
            "--learning_rate": "2e-4",
            "--warmup_steps": "500",
            "--max_epochs": "30",
            "--patience": "8",
            "--accumulate_grad_batches": "2",
        })
    
    return config, title_suffix


def run_csv_training(config, title, pretrained_path=None, freeze_encoder=False):
    """Run CSV-based ChemBL training."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set seed
    pl.seed_everything(int(config['--seed']), workers=True)
    
    # Create output directory
    output_dir = Path(config['--output_dir']) / config['--model_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting CSV-based ChemBL training")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CSV path: {config['--csv_path']}")
    logger.info(f"Batch size: {config['--batch_size']} x accumulate {config['--accumulate_grad_batches']} (effective {int(config['--batch_size']) * int(config['--accumulate_grad_batches'])})")
    logger.info(f"Max epochs: {config['--max_epochs']}")
    
    # Create data module
    logger.info("Setting up data module...")
    data_module = ChemBLCSVDataModule(
        csv_path=config['--csv_path'],
        input_vocab_path=config['--input_vocab'],
        target_vocab_path=config['--target_vocab'],
        validation_split=float(config['--validation_split']),
        test_split=float(config['--test_split']),
        batch_size=int(config['--batch_size']),
        num_workers=int(config['--num_workers']),
        enable_masking=config['--mlm_loss_weight'] != "0.0",
        use_fold_splits=config['--use_fold_splits'] == "true",
        test_fold=int(config['--test_fold']),
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
    logger.info(f"Target names: {target_info['target_names']}")
    
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
    """Main function for unified ChemBL training."""
    parser = argparse.ArgumentParser(
        description="Unified ChemBL training script with CSV input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["full", "filtered", "filtered_corrected", "test"],
        default="filtered_corrected",
        help="Dataset to use: 'full' (ChemBL CSV), 'filtered' (multi-target CSV), 'filtered_corrected' (corrected 3-state CSV), or 'test' (1K samples)",
    )

    # MLM configuration
    parser.add_argument(
        "--mlm",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable/disable masked language modeling: 'true' (default) or 'false' (classification only)",
    )

    # Split configuration
    parser.add_argument(
        "--splits",
        type=str,
        choices=["random", "folds"],
        default="folds",
        help="Split type: 'random' or 'folds' (fold-based cross-validation)",
    )

    # Fine-tuning options
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (for fine-tuning)",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder layers when resuming from checkpoint (for fine-tuning)",
    )

    # Common training overrides
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )

    # Parse known args to allow passing through other arguments
    args, unknown_args = parser.parse_known_args()

    # Add unknown args back to sys.argv for the training script (not used here but kept for compatibility)
    sys.argv = [sys.argv[0]] + unknown_args

    # Validate arguments
    if args.dataset == "test":
        print("⚠️  Warning: Test dataset not implemented for CSV version. Use 'filtered_corrected' instead.")
        args.dataset = "filtered_corrected"
    
    if args.dataset == "test" and args.splits == "folds":
        print("⚠️  Warning: Test dataset doesn't support fold-based splits. Using random splits.")
        args.splits = "random"
    
    if args.freeze_encoder and not args.resume_from_checkpoint:
        print("⚠️  Warning: --freeze_encoder specified but no checkpoint provided. Ignoring freeze_encoder.")
        args.freeze_encoder = False

    # Convert MLM string to boolean
    mlm_enabled = args.mlm.lower() == "true"

    # Determine if this is a fine-tuning run
    is_finetune = args.resume_from_checkpoint is not None

    # Create configuration
    config, title_suffix = create_csv_config(
        dataset_type=args.dataset,
        mlm=mlm_enabled,
        splits=args.splits
    )

    # Apply overrides
    if args.batch_size is not None:
        config["--batch_size"] = str(args.batch_size)
    if args.accumulate_grad_batches is not None:
        config["--accumulate_grad_batches"] = str(args.accumulate_grad_batches)
    if args.max_epochs is not None:
        config["--max_epochs"] = str(args.max_epochs)

    # Generate title
    task_desc = "MLM + Classification" if mlm_enabled else "Classification Only"
    split_desc = "Fold-based Splits" if args.splits == "folds" else "Random Splits"
    mode_desc = "Fine-tuning" if is_finetune else "Training"
    
    title = f"MolE ChemBL {mode_desc} - {title_suffix} - {task_desc} - {split_desc}"

    # Print configuration summary
    print("🚀 Unified ChemBL Training Configuration (CSV-based)")
    print("=" * 70)
    print(f"📊 Dataset: {args.dataset}")
    if args.dataset == "filtered_corrected":
        print("   • Using corrected 3-state classification (1=active, -1=inactive, 0=not measured)")
        print("   • Multi-target classification with 7 assays")
    elif args.dataset == "full":
        print("   • Using full ChemBL dataset from CSV")
        print("   • Multi-target classification")
    print(f"🎯 MLM: {'Enabled' if mlm_enabled else 'Disabled'}")
    print(f"✂️  Splits: {args.splits}")
    print(f"🔄 Mode: {'Fine-tuning' if is_finetune else 'Training'}")
    if is_finetune:
        print(f"📁 Checkpoint: {args.resume_from_checkpoint}")
        print(f"🧊 Encoder: {'Frozen' if args.freeze_encoder else 'Trainable'}")
    print(f"📁 CSV Path: {config['--csv_path']}")
    print(f"🧮 Batch size: {config['--batch_size']} x accumulate {config['--accumulate_grad_batches']} (effective {int(config['--batch_size']) * int(config['--accumulate_grad_batches'])})")
    print(f"⏱  Max epochs: {config['--max_epochs']}")
    print("=" * 70)

    # Run training
    run_csv_training(
        config=config,
        title=title,
        pretrained_path=args.resume_from_checkpoint,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
