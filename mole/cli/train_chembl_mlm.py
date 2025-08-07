"""
Training Script for ChemBL MLM Model

This script trains a MolE model using:
- Input: Radius 0 structural atom environments
- Target: Radius 1 functional atom environments
- Classification: ChemBL binary assay results (active/inactive)

The model learns to predict both richer functional representations from simpler structural ones
and binary activity across multiple ChemBL assays.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mole.data.chembl_datamodule import ChemBLDataModule
from mole.models.chembl_model import ChemBLModel
from mole.models.chembl_lightning import ChemBLLightningModule


def get_arg_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Train ChemBL MLM Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--chembl_smiles_path",
        type=str,
        required=True,
        help="Path to ChemBL SMILES pickle file",
    )
    parser.add_argument(
        "--chembl_labels_path", 
        type=str,
        required=True,
        help="Path to ChemBL labels sparse matrix pickle file",
    )
    parser.add_argument(
        "--chembl_target_names_path",
        type=str,
        required=True,
        help="Path to ChemBL target names file",
    )
    parser.add_argument(
        "--chembl_compound_names_path",
        type=str,
        default=None,
        help="Path to ChemBL compound names file (optional)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from ChemBL dataset (for testing)",
    )
    parser.add_argument(
        "--input_vocab",
        type=str,
        required=True,
        help="Path to input vocabulary (radius 0 structural)",
    )
    parser.add_argument(
        "--target_vocab",
        type=str,
        required=True,
        help="Path to target vocabulary (radius 1 functional)",
    )

    # ChemBL-specific arguments
    parser.add_argument(
        "--max_targets",
        type=int,
        default=None,
        help="Maximum number of ChemBL targets to use (for memory efficiency)",
    )
    parser.add_argument(
        "--min_target_activity",
        type=int,
        default=10,
        help="Minimum number of active/inactive labels per target",
    )

    # Model arguments
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size for transformer"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="Intermediate size in feed-forward network",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--classifier_dropout", type=float, default=0.1, help="Classifier dropout probability"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing factor"
    )

    # Environment arguments
    parser.add_argument(
        "--input_radius", type=int, default=0, help="Radius for input environments"
    )
    parser.add_argument(
        "--target_radius", type=int, default=1, help="Radius for target environments"
    )
    parser.add_argument(
        "--input_use_features",
        action="store_true",
        help="Use features for input environments",
    )
    parser.add_argument(
        "--target_use_features",
        action="store_true",
        help="Use features for target environments",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=30, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--val_check_interval", type=float, default=0.5, help="Validation check interval"
    )

    # Loss weighting arguments
    parser.add_argument(
        "--mlm_loss_weight", type=float, default=1.0, help="Weight for MLM loss"
    )
    parser.add_argument(
        "--classification_loss_weight", type=float, default=1.0, help="Weight for classification loss"
    )
    parser.add_argument(
        "--chembl_only",
        action="store_true",
        help="Train only on ChemBL classification tasks (disables MLM, sets mlm_loss_weight=0)",
    )

    # MLM arguments
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking a token"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.8, help="Probability of replacing with [MASK]"
    )
    parser.add_argument(
        "--random_prob", type=float, default=0.1, help="Probability of replacing with random token"
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )
    parser.add_argument(
        "--no_cls_token", action="store_true", help="Don't add CLS token"
    )

    # Hardware arguments
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--precision", type=str, default="32", help="Training precision"
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name"
    )

    # Miscellaneous arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--log_predictions", action="store_true", help="Log predictions during validation"
    )
    parser.add_argument(
        "--log_target_metrics", action="store_true", help="Log per-target metrics"
    )
    parser.add_argument(
        "--max_targets_to_log", type=int, default=50, help="Maximum targets to log individually"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    return parser


def parse_args():
    """Parse command line arguments."""
    parser = get_arg_parser()
    args = parser.parse_args()

    # Validation
    if not (0 <= args.validation_split <= 1):
        raise ValueError("validation_split must be between 0 and 1")
    
    if not (0 <= args.test_split <= 1):
        raise ValueError("test_split must be between 0 and 1")
        
    if args.validation_split + args.test_split >= 1:
        raise ValueError("validation_split + test_split must be < 1")

    return args


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration from arguments."""
    return {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
        "hidden_dropout_prob": args.dropout,
        "attention_probs_dropout_prob": args.dropout,
        "layer_norm_eps": 1e-7,
        "initializer_range": 0.02,
        "relative_attention": True,
        "position_biased_input": False,
        "pos_att_type": "p2c|c2p",
        "max_relative_positions": 512,
        "hidden_act": "gelu",
        "pooler_dropout": args.dropout,
        "pooler_hidden_act": "gelu",
        "summary_use_proj": True,
        "summary_activation": "gelu",
        "summary_last_dropout": args.dropout,
    }


def save_chembl_config_files(args, model_config: Dict[str, Any], vocab_sizes: Dict[str, int], target_info: Dict[str, Any], output_dir: Path):
    """Save configuration files for ChemBL training."""
    import json
    
    # Save model configuration
    model_config_with_vocab = model_config.copy()
    model_config_with_vocab.update({
        "input_vocab_size": vocab_sizes["input_vocab_size"],
        "target_vocab_size": vocab_sizes["target_vocab_size"],
        "num_targets": target_info["num_targets"],
    })
    
    model_config_path = output_dir / "model_config.json"
    with open(model_config_path, 'w') as f:
        json.dump(model_config_with_vocab, f, indent=2)
    
    # Save training arguments
    args_dict = vars(args)
    # Convert Path objects to strings for JSON serialization
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    
    args_config_path = output_dir / "training_args.json"
    with open(args_config_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    # Save a human-readable summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ChemBL MLM Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATA CONFIGURATION:\n")
        f.write(f"  ChemBL SMILES: {args.chembl_smiles_path}\n")
        f.write(f"  ChemBL labels: {args.chembl_labels_path}\n")
        f.write(f"  ChemBL target names: {args.chembl_target_names_path}\n")
        f.write(f"  ChemBL compound names: {args.chembl_compound_names_path}\n")
        f.write(f"  Input vocabulary: {args.input_vocab}\n")
        f.write(f"  Target vocabulary: {args.target_vocab}\n")
        f.write(f"  Input vocab size: {vocab_sizes['input_vocab_size']}\n")
        f.write(f"  Target vocab size: {vocab_sizes['target_vocab_size']}\n")
        f.write(f"  Number of ChemBL targets: {target_info['num_targets']}\n")
        f.write(f"  Max targets: {args.max_targets}\n")
        f.write(f"  Min target activity: {args.min_target_activity}\n")
        f.write(f"  Max samples: {args.max_samples or 'All available'}\n")
        f.write(f"  Input radius: {args.input_radius}\n")
        f.write(f"  Target radius: {args.target_radius}\n")
        f.write(f"  Input use features: {args.input_use_features}\n")
        f.write(f"  Target use features: {args.target_use_features}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Hidden size: {args.hidden_size}\n")
        f.write(f"  Number of layers: {args.num_hidden_layers}\n")
        f.write(f"  Number of attention heads: {args.num_attention_heads}\n")
        f.write(f"  Intermediate size: {args.intermediate_size}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Classifier dropout: {args.classifier_dropout}\n\n")
        
        f.write("MLM CONFIGURATION:\n")
        f.write(f"  Mask probability: {args.mask_prob}\n")
        f.write(f"  Replace probability: {args.replace_prob}\n")
        f.write(f"  Random probability: {args.random_prob}\n")
        f.write(f"  Max length: {args.max_length or 'No limit'}\n")
        f.write(f"  Use CLS token: {not args.no_cls_token}\n")
        f.write(f"  ChemBL only mode: {args.chembl_only}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Weight decay: {args.weight_decay}\n")
        f.write(f"  Warmup steps: {args.warmup_steps}\n")
        f.write(f"  Max epochs: {args.max_epochs}\n")
        f.write(f"  Validation split: {args.validation_split}\n")
        f.write(f"  Test split: {args.test_split}\n")
        f.write(f"  Patience: {args.patience}\n")
        
        if hasattr(args, 'mlm_loss_weight'):
            f.write(f"  MLM loss weight: {args.mlm_loss_weight}\n")
        if hasattr(args, 'classification_loss_weight'):
            f.write(f"  Classification loss weight: {args.classification_loss_weight}\n")
        f.write("\n")
        
        f.write("HARDWARE CONFIGURATION:\n")
        f.write(f"  GPUs: {args.gpus}\n")
        f.write(f"  Precision: {args.precision}\n")
        f.write(f"  Number of workers: {args.num_workers}\n")
        f.write(f"  Gradient accumulation: {args.accumulate_grad_batches}\n")
        f.write(f"  Gradient clipping: {args.gradient_clip_val}\n\n")
        
        f.write("OUTPUT CONFIGURATION:\n")
        f.write(f"  Output directory: {output_dir}\n")
        f.write(f"  Model name: {args.model_name}\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  Debug mode: {args.debug}\n")
        f.write(f"  Log predictions: {args.log_predictions}\n")
        f.write(f"  Log target metrics: {args.log_target_metrics}\n")
        f.write(f"  Max targets to log: {args.max_targets_to_log}\n")


def main():
    """Main training function"""
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Create output directory
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ChemBL MLM training")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Input vocab: {args.input_vocab}")
    logger.info(f"Target vocab: {args.target_vocab}")
    logger.info(f"ChemBL SMILES: {args.chembl_smiles_path}")
    logger.info(f"ChemBL labels: {args.chembl_labels_path}")

    # Create data module
    logger.info("Setting up ChemBL data module...")
    data_module = ChemBLDataModule(
        chembl_smiles_path=args.chembl_smiles_path,
        chembl_labels_path=args.chembl_labels_path,
        chembl_target_names_path=args.chembl_target_names_path,
        chembl_compound_names_path=args.chembl_compound_names_path,
        input_vocab_path=args.input_vocab,
        target_vocab_path=args.target_vocab,
        validation_split=args.validation_split,
        test_split=args.test_split,
        max_targets=args.max_targets,
        min_target_activity=args.min_target_activity,
        max_samples=args.max_samples,
        input_radius=args.input_radius,
        target_radius=args.target_radius,
        input_use_features=args.input_use_features,
        target_use_features=args.target_use_features,
        mask_prob=args.mask_prob,
        replace_prob=args.replace_prob,
        random_prob=args.random_prob,
        max_length=args.max_length,
        cls_token=not args.no_cls_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enable_masking=not args.chembl_only,  # Disable masking in ChemBL-only mode
    )

    # Setup data to get vocabulary sizes and target info
    data_module.setup("fit")
    vocab_sizes = data_module.get_vocab_sizes()
    target_info = data_module.get_target_info()

    logger.info(f"Input vocabulary size: {vocab_sizes['input_vocab_size']}")
    logger.info(f"Target vocabulary size: {vocab_sizes['target_vocab_size']}")
    logger.info(f"Number of ChemBL targets: {target_info['num_targets']}")

    # Create model configuration
    model_config = create_model_config(args)
    
    # Save configuration files
    logger.info("Saving configuration files...")
    save_chembl_config_files(args, model_config, vocab_sizes, target_info, output_dir)

    # Create model
    logger.info("Creating ChemBL model...")
    core_model = ChemBLModel(
        deberta_config=model_config,
        input_vocab_size=vocab_sizes["input_vocab_size"],
        target_vocab_size=vocab_sizes["target_vocab_size"],
        num_targets=target_info["num_targets"],
        hidden_dropout_prob=args.dropout,
        classifier_dropout_prob=args.classifier_dropout,
    )

    # Create Lightning module
    logger.info("Creating Lightning module...")
    
    # Calculate total training steps for scheduler
    total_samples = len(data_module._load_chembl_smiles()) * (1 - args.validation_split - args.test_split)
    steps_per_epoch = max(1, int(total_samples // (args.batch_size * args.accumulate_grad_batches)))
    total_steps = steps_per_epoch * args.max_epochs

    optimizer_cfg = {
        "type": "AdamW",
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    scheduler_cfg = {
        "type": "linear_with_warmup",
        "warmup_steps": args.warmup_steps,
        "total_steps": total_steps,
    }

    # Handle ChemBL-only mode
    if args.chembl_only:
        logger.info("ChemBL-only mode enabled: Disabling MLM training")
        mlm_loss_weight = 0.0
    else:
        mlm_loss_weight = args.mlm_loss_weight

    lightning_model = ChemBLLightningModule(
        model=core_model,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        log_predictions=args.log_predictions,
        mlm_loss_weight=mlm_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        log_target_metrics=args.log_target_metrics,
        max_targets_to_log=args.max_targets_to_log,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:02d}-{step:005d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=5,  # Save the 5 best checkpoints
        save_last=False,  # Don't save the last checkpoint
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    if args.patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="val/total_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="chembl_mlm",
    )

    # Create trainer
    logger.info("Creating trainer...")
    
    # Handle devices configuration
    if args.gpus > 0:
        devices = args.gpus
        accelerator = "gpu"
        precision = args.precision
    else:
        devices = "auto"  # Let Lightning handle CPU device selection
        accelerator = "cpu"
        precision = "32"  # Force 32-bit precision for CPU
        if args.precision != "32":
            logger.warning(f"Overriding precision from {args.precision} to 32 for CPU training")
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit(lightning_model, data_module)

    # Test if test data is available
    if hasattr(data_module, '_test_smiles'):
        logger.info("Running final test...")
        trainer.test(lightning_model, data_module)

    logger.info("Training completed!")
    logger.info(f"Best model saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main() 