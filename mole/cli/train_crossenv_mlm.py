"""
Training Script for Cross-Environment MLM Model

This script trains a MolE model using:
- Input: Radius 0 structural atom environments
- Target: Radius 1 functional atom environments

The model learns to predict richer functional representations from simpler structural ones.
"""

import argparse
import logging
import os
import sys
import json
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

from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.models.crossenv_mlm import CrossEnvMLMModel, CrossEnvMLM


def get_arg_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Cross-Environment MLM Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training SMILES data (parquet, csv, or txt file)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation SMILES data (optional)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test SMILES data (optional)",
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
        "--label_smoothing", type=float, default=0.0, help="Label smoothing factor"
    )

    # Environment arguments
    parser.add_argument(
        "--input_radius", type=int, default=0, help="Radius for input atom environments"
    )
    parser.add_argument(
        "--target_radius",
        type=int,
        default=1,
        help="Radius for target atom environments",
    )
    parser.add_argument(
        "--input_use_features",
        action="store_true",
        help="Use features for input environments",
    )
    parser.add_argument(
        "--target_use_features",
        action="store_true",
        default=True,
        help="Use features for target environments",
    )

    # MLM arguments
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking a token"
    )
    parser.add_argument(
        "--replace_prob",
        type=float,
        default=0.8,
        help="Probability of replacing with [MASK]",
    )
    parser.add_argument(
        "--random_prob",
        type=float,
        default=0.1,
        help="Probability of replacing with random token",
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )
    parser.add_argument(
        "--no_cls_token", action="store_true", help="Don't use CLS token"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=10000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps (overrides max_epochs if set)",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Validation split fraction (if no val data provided)",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval",
    )

    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "32", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/crossenv_mlm",
        help="Output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="crossenv_mlm_model",
        help="Model name for logging and checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder layers during training (for fine-tuning)",
    )

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_predictions",
        action="store_true",
        help="Log prediction examples during validation",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Enable torch.compile for performance improvements (requires PyTorch 2.0+)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping. Set to 0 to disable.",
    )

    return parser


def parse_args():
    """Parse command line arguments"""
    parser = get_arg_parser()
    return parser.parse_args()


def create_model_config(args) -> Dict[str, Any]:
    """Create DeBERTa model configuration"""
    return {
        "attention_head": args.num_attention_heads,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "max_position_embeddings": 512,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "type_vocab_size": 0,
        "vocab_size": 1000,  # Will be overridden by actual input vocab size
        "norm_rel_ebd": "layer_norm",
        "position_biased_input": False,
        "pos_att_type": "p2c|c2p",
        "relative_attention": True,
        "max_relative_positions": 128,
        "layer_norm_eps": 1e-7,
        "dropout": args.dropout,
        "attention_dropout": args.dropout,
        "hidden_dropout_prob": args.dropout,
        "initializer_range": 0.02,
        "summary_type": "first",
        "summary_use_proj": True,
        "summary_activation": "gelu",
        "summary_last_dropout": args.dropout,
    }


def save_config_files(args, model_config: Dict[str, Any], vocab_sizes: Dict[str, int], output_dir: Path):
    """Save configuration files to output directory"""
    
    # Save model configuration
    model_config_with_vocab = model_config.copy()
    model_config_with_vocab.update({
        "input_vocab_size": vocab_sizes["input_vocab_size"],
        "target_vocab_size": vocab_sizes["target_vocab_size"],
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
        f.write("Cross-Environment MLM Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATA CONFIGURATION:\n")
        f.write(f"  Training data: {args.train_data}\n")
        f.write(f"  Validation data: {args.val_data or 'Split from training'}\n")
        f.write(f"  Test data: {args.test_data or 'Not provided'}\n")
        f.write(f"  Input vocabulary: {args.input_vocab}\n")
        f.write(f"  Target vocabulary: {args.target_vocab}\n")
        f.write(f"  Input vocab size: {vocab_sizes['input_vocab_size']}\n")
        f.write(f"  Target vocab size: {vocab_sizes['target_vocab_size']}\n")
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
        f.write(f"  Label smoothing: {args.label_smoothing}\n\n")
        
        f.write("MLM CONFIGURATION:\n")
        f.write(f"  Mask probability: {args.mask_prob}\n")
        f.write(f"  Replace probability: {args.replace_prob}\n")
        f.write(f"  Random probability: {args.random_prob}\n")
        f.write(f"  Max length: {args.max_length or 'No limit'}\n")
        f.write(f"  Use CLS token: {not args.no_cls_token}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Weight decay: {args.weight_decay}\n")
        f.write(f"  Warmup steps: {args.warmup_steps}\n")
        f.write(f"  Max epochs: {args.max_epochs}\n")
        f.write(f"  Max steps: {args.max_steps or 'No limit'}\n")
        f.write(f"  Validation split: {args.validation_split}\n")
        f.write(f"  Patience: {args.patience}\n\n")
        
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
        f.write(f"  Use torch compile: {args.use_torch_compile}\n")


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

    logger.info(f"Starting cross-environment MLM training")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Input vocab: {args.input_vocab}")
    logger.info(f"Target vocab: {args.target_vocab}")
    logger.info(f"Training data: {args.train_data}")

    # Create data module
    logger.info("Setting up data module...")
    data_module = CrossEnvDataModule(
        train_data=args.train_data,
        input_vocab_path=args.input_vocab,
        target_vocab_path=args.target_vocab,
        validation_data=args.val_data,
        test_data=args.test_data,
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
        validation_split=args.validation_split,
    )

    # Setup data to get vocabulary sizes
    data_module.setup("fit")
    vocab_sizes = data_module.get_vocab_sizes()

    logger.info(f"Input vocabulary size: {vocab_sizes['input_vocab_size']}")
    logger.info(f"Target vocabulary size: {vocab_sizes['target_vocab_size']}")

    # Create model configuration
    model_config = create_model_config(args)
    
    # Save configuration files
    logger.info("Saving configuration files...")
    save_config_files(args, model_config, vocab_sizes, output_dir)

    # Create model
    logger.info("Creating model...")
    core_model = CrossEnvMLMModel(
        deberta_config=model_config,
        input_vocab_size=vocab_sizes["input_vocab_size"],
        target_vocab_size=vocab_sizes["target_vocab_size"],
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    # Create optimizer configuration
    optimizer_config = {
        "optimizer": "AdamW",
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    # Create scheduler configuration
    scheduler_config = {
        "scheduler": "linear_with_warmup",
        "warmup_steps": args.warmup_steps,
        "interval": "step",
        "frequency": 1,
    }

    # Create Lightning module
    lightning_model = CrossEnvMLM(
        model=core_model,
        optimizer_cfg=optimizer_config,
        scheduler_cfg=scheduler_config,
        log_predictions=args.log_predictions,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
        save_top_k=5,  # Save the 5 best checkpoints
        monitor="val_loss",
        mode="min",
        save_last=False,  # Don't save the last checkpoint
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    if args.patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)

    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
        version=None,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": args.gradient_clip_val,
        "val_check_interval": args.val_check_interval,
        "callbacks": callbacks,
        "logger": tb_logger,
        "deterministic": True,
        "enable_progress_bar": True,
        "log_every_n_steps": 50,
        "check_val_every_n_epoch": 1,
    }

    # Only add max_steps if it's not None
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps

    if args.gpus > 0:
        trainer_kwargs["devices"] = args.gpus
        trainer_kwargs["accelerator"] = "gpu"
    else:
        trainer_kwargs["accelerator"] = "cpu"

    trainer = pl.Trainer(**trainer_kwargs)

    # Log model summary
    logger.info("Model summary:")
    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(
        p.numel() for p in lightning_model.parameters() if p.requires_grad
    )
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{total_params * 4 / 1024**2:.1f}MB (float32)")

    # Compile model if requested (PyTorch 2.0+)
    # Handle encoder freezing if requested
    if args.freeze_encoder:
        logger.info("Freezing encoder layers for fine-tuning...")
        if hasattr(lightning_model.model, 'encoder'):
            for param in lightning_model.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder layers frozen.")
        else:
            logger.warning("No encoder found in model - cannot freeze encoder")

    if args.use_torch_compile:
        lightning_model.compile()

    # Start training
    logger.info("Starting training...")
    trainer.fit(
        model=lightning_model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )

    # Test if test data provided
    if args.test_data is not None:
        logger.info("Running test evaluation...")
        trainer.test(model=lightning_model, datamodule=data_module)

    # Log checkpoint information
    logger.info("Checkpoint summary:")
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        logger.info(f"Total checkpoints saved: {len(checkpoint_files)}")
        
        # Get the best checkpoint info
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score
        
        if best_checkpoint and Path(best_checkpoint).exists():
            logger.info(f"Best checkpoint: {Path(best_checkpoint).name}")
            logger.info(f"Best validation loss: {best_score:.4f}")
            
            # List all saved checkpoints with their scores
            for i, checkpoint_file in enumerate(sorted(checkpoint_files, key=lambda x: x.name)):
                logger.info(f"  {i+1}. {checkpoint_file.name}")
        else:
            logger.warning("No best checkpoint found")

    logger.info("Training completed!")
    logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
