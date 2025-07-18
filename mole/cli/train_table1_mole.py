"""
Training Script for Table 1 MolE Model

This script trains a MolE model on Table 1 ADMET datasets from the MolE paper.
The model uses atom environment tokenization and predicts multiple ADMET properties
simultaneously for regression and classification tasks.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mole.models.table1_mole_model import create_table1_mole_model
from mole.dataset_loader import create_admet_dataloader
from mole.table1_prediction_heads import Table1PredictionHeads
from configs.table1_task_config import get_all_tasks, get_task_type_mapping


def get_arg_parser():
    """Get argument parser for Table 1 MolE training."""
    parser = argparse.ArgumentParser(
        description="Train Table 1 MolE Model on ADMET Datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/tdc/tdc_table1_datasets.csv",
        help="Path to Table 1 datasets CSV file",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to atom environment vocabulary file",
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
        "--freeze_encoder", action="store_true", help="Freeze MolE encoder weights"
    )
    parser.add_argument(
        "--pretrained_path", type=str, default=None, help="Path to pretrained MolE weights"
    )

    # Atom environment arguments
    parser.add_argument(
        "--radius", type=int, default=0, help="Morgan fingerprint radius for atom environments"
    )
    parser.add_argument(
        "--use_features", action="store_true", help="Use functional features for atom environments"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
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
        "--test_size", type=float, default=0.2, help="Test split fraction"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Validation split fraction"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
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
        default="outputs/table1_mole_training",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="table1_mole_model",
        help="Model name for logging and checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    # Training control arguments
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile() for model optimization",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    return parser


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration from arguments."""
    return {
        'hidden_size': args.hidden_size,
        'num_attention_heads': args.num_attention_heads,
        'num_hidden_layers': args.num_hidden_layers,
        'intermediate_size': args.intermediate_size,
        'vocab_size': 1000,  # Will be overridden by vocabulary size
        'max_position_embeddings': 512,
        'layer_norm_eps': 1e-12,
        'hidden_dropout_prob': args.dropout,
        'attention_probs_dropout_prob': args.dropout,
        'initializer_range': 0.02,
        'type_vocab_size': 0
    }


class Table1MolELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Table 1 MolE training.
    
    This module handles the training loop, validation, and evaluation
    for the integrated MolE model with Table 1 prediction heads.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model: The Table1MolEModel to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        logging.info(f"Initialized Table1MolELightningModule with lr={learning_rate}")
    
    def forward(self, batch):
        """Forward pass through the model."""
        # Move batch to device
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        # Get input data from PyTorch Geometric batch
        input_ids = batch.x
        
        # Ensure input_ids has the correct shape [batch_size, seq_len]
        # PyTorch Geometric batches might have different shapes
        if input_ids.dim() == 1:
            # Single sample case
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        elif input_ids.dim() == 2:
            # Already in correct format [batch_size, seq_len]
            pass
        else:
            raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")
        
        input_mask = (input_ids != 0).long()
        
        # Create simple relative position matrix (identity matrix for now)
        # This avoids the problematic to_dense_adj conversion
        batch_size, seq_len = input_ids.shape
        relative_pos = torch.zeros(batch_size, seq_len, seq_len, device=input_ids.device)
        
        # Create identity-like relative positions (diagonal = 0, others = 1)
        for i in range(batch_size):
            relative_pos[i] = torch.ones(seq_len, seq_len, device=input_ids.device)
            relative_pos[i].fill_diagonal_(0)
        
        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            relative_pos=relative_pos
        )
        
        return outputs, batch
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs, batch = self.forward(batch)
        
        # Get targets and masks from batch
        targets, masks = self._extract_targets_and_masks(batch)
        
        # Compute loss
        device = next(self.parameters()).device
        loss_dict = self.model.compute_loss(
            regression_output=outputs['regression_output'],
            classification_output=outputs['classification_output'],
            targets=targets,
            masks=masks,
            device=device
        )
        
        total_loss = loss_dict['total_loss']
        
        # Log metrics
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs, batch = self.forward(batch)
        
        # Get targets and masks from batch
        targets, masks = self._extract_targets_and_masks(batch)
        
        # Compute loss
        device = next(self.parameters()).device
        loss_dict = self.model.compute_loss(
            regression_output=outputs['regression_output'],
            classification_output=outputs['classification_output'],
            targets=targets,
            masks=masks,
            device=device
        )
        
        total_loss = loss_dict['total_loss']
        
        # Compute metrics
        metrics = self.model.compute_metrics(
            regression_output=outputs['regression_output'],
            classification_output=outputs['classification_output'],
            targets=targets,
            masks=masks,
            device=device
        )
        
        # Log metrics
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'val/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def _extract_targets_and_masks(self, batch):
        """Extract targets and masks from PyTorch Geometric batch."""
        targets = {}
        masks = {}
        
        # Get all task names from the task configuration
        from configs.table1_task_config import get_all_tasks
        all_tasks = get_all_tasks()
        
        # Extract targets and masks for each task
        for task_name in all_tasks:
            if hasattr(batch, task_name) and hasattr(batch, f'{task_name}_mask'):
                targets[task_name] = getattr(batch, task_name)
                masks[task_name] = getattr(batch, f'{task_name}_mask')
        
        # Debug logging
        if not targets:
            logging.warning(f"No targets found in batch. Available attributes: {[attr for attr in dir(batch) if not attr.startswith('_') and not callable(getattr(batch, attr))]}")
        else:
            logging.debug(f"Extracted targets for tasks: {list(targets.keys())}")
        
        return targets, masks
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler with warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_pct = min(self.warmup_steps / total_steps, 1.0)  # Cap at 1.0
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    """Main function to run Table 1 MolE training."""
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s"
    )
    logging.info("Starting Table 1 MolE training")
    logging.info(f"Output directory: {args.output_dir}/{args.model_name}")
    
    # Create data loader
    logging.info("Setting up data loader...")
    data_loader = create_admet_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        vocab_path=args.vocab_path,
        radius=args.radius,
        use_features=args.use_features,
        max_length=args.max_length,
        cls_token=not args.no_cls_token,
        num_workers=args.num_workers
    )
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    logging.info(f"Dataset info: {dataset_info}")
    
    # Create model
    logging.info("Creating model...")
    model_config = create_model_config(args)
    model = create_table1_mole_model(
        deberta_config=model_config,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        pretrained_path=args.pretrained_path
    )
    
    # Create Lightning module
    lightning_module = Table1MolELightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
    
    # Model compilation (if enabled)
    if args.use_torch_compile:
        logging.info("Compiling model with torch.compile()...")
        try:
            lightning_module = torch.compile(lightning_module)
            logging.info("Model compiled successfully.")
        except Exception as e:
            logging.error(f"Failed to compile model: {e}")
            logging.warning("Continuing without torch.compile().")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.output_dir}/{args.model_name}/checkpoints",
        filename="{epoch}-{val/total_loss:.2f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    
    if args.patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir, name=args.model_name, version="lightning_logs"
    )
    
    # Trainer setup
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        deterministic=True,
    )
    
    # Run training
    logging.info("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=data_loader.get_dataloader('train'),
        val_dataloaders=data_loader.get_dataloader('val'),
        ckpt_path=args.resume_from_checkpoint,
    )
    
    # Test on test set
    logging.info("Evaluating on test set...")
    test_results = trainer.test(
        lightning_module,
        dataloaders=data_loader.get_dataloader('test')
    )
    
    logging.info("Training completed successfully!")
    logging.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main() 