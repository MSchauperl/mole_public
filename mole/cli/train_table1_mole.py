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
from typing import Dict, Any, Optional, List

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

    # Dataset arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Table 1 datasets CSV"
    )
    parser.add_argument(
        "--vocab_path", type=str, 
        default="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to atom environment vocabulary file"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None, 
        help="List of specific tasks to train on (e.g., --tasks Caco2 Lipophilicity Solubility). If not specified, all tasks are used."
    )
    parser.add_argument(
        "--task_type", type=str, choices=["regression", "classification", "all"], default="all",
        help="Filter tasks by type: 'regression', 'classification', or 'all' (default)"
    )
    parser.add_argument(
        "--use_features", action="store_true", help="Use functional features"
    )

    # Model arguments
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size for transformer"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=None, help="Embedding size (defaults to hidden_size if not specified)"
    )
    parser.add_argument(
        "--max_position_embeddings", type=int, default=2048, help="Maximum sequence length for position embeddings"
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
    
    # Loss balancing arguments
    parser.add_argument(
        "--regression_loss_weight", type=float, default=0.004, 
        help="Weight for regression loss (default: 0.004 to balance with classification)"
    )
    parser.add_argument(
        "--classification_loss_weight", type=float, default=1.0, 
        help="Weight for classification loss (default: 1.0)"
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


def filter_tasks(args) -> List[str]:
    """Filter tasks based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of task names to train on
    """
    from configs.table1_task_config import get_all_tasks, get_task_type_mapping
    
    all_tasks = get_all_tasks()
    task_type_mapping = get_task_type_mapping()
    
    # Start with all tasks
    filtered_tasks = all_tasks.copy()
    
    # Filter by task type if specified
    if args.task_type != "all":
        filtered_tasks = [task for task in filtered_tasks if task_type_mapping[task] == args.task_type]
    
    # Filter by specific task names if provided
    if args.tasks is not None:
        # Validate that all specified tasks exist
        invalid_tasks = [task for task in args.tasks if task not in all_tasks]
        if invalid_tasks:
            raise ValueError(f"Invalid task names: {invalid_tasks}. Available tasks: {all_tasks}")
        
        # Only keep tasks that are in both the filtered list and the specified list
        filtered_tasks = [task for task in filtered_tasks if task in args.tasks]
    
    if not filtered_tasks:
        raise ValueError("No tasks selected. Please check your --tasks and --task_type arguments.")
    
    logging.info(f"Selected tasks ({len(filtered_tasks)}): {filtered_tasks}")
    
    # Show task types for clarity
    regression_tasks = [task for task in filtered_tasks if task_type_mapping[task] == 'regression']
    classification_tasks = [task for task in filtered_tasks if task_type_mapping[task] == 'classification']
    
    if regression_tasks:
        logging.info(f"Regression tasks ({len(regression_tasks)}): {regression_tasks}")
    if classification_tasks:
        logging.info(f"Classification tasks ({len(classification_tasks)}): {classification_tasks}")
    
    return filtered_tasks


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration from arguments."""
    return {
        'hidden_size': args.hidden_size,
        'embedding_size': args.embedding_size if args.embedding_size is not None else args.hidden_size,
        'num_attention_heads': args.num_attention_heads,
        'num_hidden_layers': args.num_hidden_layers,
        'intermediate_size': args.intermediate_size,
        'vocab_size': 1000,  # Will be overridden by vocabulary size
        'max_position_embeddings': args.max_position_embeddings,
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
        
        # Epoch-level predictions for proper metric computation
        self.val_predictions = {'classification': [], 'regression': []}
        self.val_targets = {'classification': [], 'regression': []}
        self.val_masks = {'classification': [], 'regression': []}
        
        # Training predictions for periodic logging (every 5th epoch)
        self.train_predictions = {'classification': [], 'regression': []}
        self.train_targets = {'classification': [], 'regression': []}
        self.train_masks = {'classification': [], 'regression': []}
        
        # Test predictions for final logging
        self.test_predictions = {'classification': [], 'regression': []}
        self.test_targets = {'classification': [], 'regression': []}
        self.test_masks = {'classification': [], 'regression': []}
        
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
        
        # Accumulate predictions for periodic training metrics (every 5th epoch)
        if (self.current_epoch + 1) % 5 == 0:  # Store predictions for 5th, 10th, 15th, ... epochs
            # Store classification predictions and targets
            classification_targets, classification_masks = self._process_classification_targets_masks(targets, masks)
            if classification_targets is not None and outputs.get('classification_output') is not None:
                self.train_predictions['classification'].append(outputs['classification_output'].detach().cpu())
                self.train_targets['classification'].append(classification_targets.detach().cpu())
                self.train_masks['classification'].append(classification_masks.detach().cpu())
            
            # Store regression predictions and targets
            regression_targets, regression_masks = self._process_regression_targets_masks(targets, masks)
            if regression_targets is not None and outputs.get('regression_output') is not None:
                self.train_predictions['regression'].append(outputs['regression_output'].detach().cpu())
                self.train_targets['regression'].append(regression_targets.detach().cpu())
                self.train_masks['regression'].append(regression_masks.detach().cpu())
        
        # Compute loss and metrics
        loss_dict = self.model.compute_loss_and_metrics(
            regression_output=outputs.get('regression_output'),
            classification_output=outputs.get('classification_output'),
            targets=targets,
            masks=masks,
            device=self.device
        )
        
        total_loss = loss_dict['total_loss']
        
        # Log metrics
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def _process_classification_targets_masks(self, targets, masks):
        """Process classification targets and masks for storage."""
        from configs.table1_task_config import get_all_tasks, get_task_type_mapping
        task_type_mapping = get_task_type_mapping()
        all_tasks = get_all_tasks()
        classification_tasks = [task for task in all_tasks if task_type_mapping[task] == 'classification']
        
        classification_targets = []
        classification_masks = []
        for task in classification_tasks:
            if task in targets and task in masks:
                # Convert classification targets to binary format
                task_targets = targets[task].detach().cpu()
                task_masks = masks[task].detach().cpu()
                
                # Handle -1 values (missing) by setting mask to False
                valid_targets = (task_targets != -1).float()
                task_masks = task_masks & valid_targets.bool()
                
                # Convert to binary (0/1) for BCE loss
                binary_targets = (task_targets == 1).float()
                
                classification_targets.append(binary_targets)
                classification_masks.append(task_masks)
            else:
                # Add dummy data for missing tasks - we need to infer batch size
                batch_size = 1  # Will be updated when we know the actual batch size
                classification_targets.append(torch.zeros(batch_size, device='cpu'))
                classification_masks.append(torch.zeros(batch_size, dtype=torch.bool, device='cpu'))
        
        if classification_targets:
            # Update batch size for dummy tensors
            actual_batch_size = classification_targets[0].shape[0] if any(t.numel() > 0 for t in classification_targets) else 1
            for i, (target, mask) in enumerate(zip(classification_targets, classification_masks)):
                if target.shape[0] == 1 and actual_batch_size > 1:
                    classification_targets[i] = torch.zeros(actual_batch_size, device='cpu')
                    classification_masks[i] = torch.zeros(actual_batch_size, dtype=torch.bool, device='cpu')
            
            return torch.stack(classification_targets, dim=1), torch.stack(classification_masks, dim=1)
        return None, None
    
    def _process_regression_targets_masks(self, targets, masks):
        """Process regression targets and masks for storage."""
        from configs.table1_task_config import get_all_tasks, get_task_type_mapping
        task_type_mapping = get_task_type_mapping()
        all_tasks = get_all_tasks()
        regression_tasks = [task for task in all_tasks if task_type_mapping[task] == 'regression']
        
        regression_targets = []
        regression_masks = []
        for task in regression_tasks:
            if task in targets and task in masks:
                regression_targets.append(targets[task].detach().cpu())
                regression_masks.append(masks[task].detach().cpu())
            else:
                # Add dummy data for missing tasks
                batch_size = 1  # Will be updated when we know the actual batch size
                regression_targets.append(torch.zeros(batch_size, device='cpu'))
                regression_masks.append(torch.zeros(batch_size, dtype=torch.bool, device='cpu'))
        
        if regression_targets:
            # Update batch size for dummy tensors
            actual_batch_size = regression_targets[0].shape[0] if any(t.numel() > 0 for t in regression_targets) else 1
            for i, (target, mask) in enumerate(zip(regression_targets, regression_masks)):
                if target.shape[0] == 1 and actual_batch_size > 1:
                    regression_targets[i] = torch.zeros(actual_batch_size, device='cpu')
                    regression_masks[i] = torch.zeros(actual_batch_size, dtype=torch.bool, device='cpu')
            
            return torch.stack(regression_targets, dim=1), torch.stack(regression_masks, dim=1)
        return None, None
    
    def on_train_epoch_end(self):
        """Compute and log training metrics every 5th epoch."""
        if (self.current_epoch + 1) % 5 == 0 and (self.train_predictions['classification'] or self.train_predictions['regression']):
            logging.info(f"Computing training metrics for epoch {self.current_epoch + 1}")
            
            # Concatenate predictions if available
            if self.train_predictions['classification']:
                classification_preds = torch.cat(self.train_predictions['classification'], dim=0)
                classification_targets = torch.cat(self.train_targets['classification'], dim=0)
                classification_masks = torch.cat(self.train_masks['classification'], dim=0)
            else:
                # Create dummy tensors
                classification_preds = torch.zeros(1, 10)
                classification_targets = torch.zeros(1, 10)
                classification_masks = torch.zeros(1, 10, dtype=torch.bool)
            
            if self.train_predictions['regression']:
                regression_preds = torch.cat(self.train_predictions['regression'], dim=0)
                regression_targets = torch.cat(self.train_targets['regression'], dim=0)
                regression_masks = torch.cat(self.train_masks['regression'], dim=0)
            else:
                # Create dummy tensors
                regression_preds = torch.zeros(1, 8)
                regression_targets = torch.zeros(1, 8)
                regression_masks = torch.zeros(1, 8, dtype=torch.bool)
            
            # Compute training metrics
            try:
                train_metrics = self._compute_epoch_metrics(
                    classification_preds, classification_targets, classification_masks,
                    regression_preds, regression_targets, regression_masks
                )
                
                logging.info(f"Computed {len(train_metrics)} training metrics for epoch {self.current_epoch + 1}")
                
                # Log training metrics
                for metric_name, metric_value in train_metrics.items():
                    self.log(f'train/{metric_name}', metric_value, on_epoch=True)
                    logging.info(f"Logged training metric: train/{metric_name} = {metric_value:.4f}")
                    
            except Exception as e:
                logging.error(f"Error computing training metrics: {e}")
                import traceback
                traceback.print_exc()
            
            # Clear accumulated predictions
            self.train_predictions['classification'].clear()
            self.train_predictions['regression'].clear()
            self.train_targets['classification'].clear()
            self.train_targets['regression'].clear()
            self.train_masks['classification'].clear()
            self.train_masks['regression'].clear()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs, batch = self.forward(batch)
        
        # Get targets and masks from batch
        targets, masks = self._extract_targets_and_masks(batch)
        
        # Compute loss and metrics
        loss_dict = self.model.compute_loss_and_metrics(
            regression_output=outputs.get('regression_output'),
            classification_output=outputs.get('classification_output'),
            targets=targets,
            masks=masks,
            device=self.device
        )
        
        total_loss = loss_dict['total_loss']
        
        # Store predictions for epoch-level metric computation (only if they exist)
        if outputs.get('classification_output') is not None:
            self.val_predictions['classification'].append(outputs['classification_output'].detach().cpu())
        if outputs.get('regression_output') is not None:
            self.val_predictions['regression'].append(outputs['regression_output'].detach().cpu())
        
        # Prepare targets and masks for storage
        from configs.table1_task_config import get_all_tasks, get_task_type_mapping
        task_type_mapping = get_task_type_mapping()
        all_tasks = get_all_tasks()
        
        # Separate regression and classification tasks
        regression_tasks = [task for task in all_tasks if task_type_mapping[task] == 'regression']
        classification_tasks = [task for task in all_tasks if task_type_mapping[task] == 'classification']
        
        # Store classification targets and masks
        classification_targets = []
        classification_masks = []
        for task in classification_tasks:
            if task in targets and task in masks:
                classification_targets.append(targets[task].detach().cpu())
                classification_masks.append(masks[task].detach().cpu())
            else:
                # Add dummy data for missing tasks (get batch_size from input batch)
                batch_size = batch['x'].shape[0]
                classification_targets.append(torch.zeros(batch_size, device='cpu'))
                classification_masks.append(torch.zeros(batch_size, dtype=torch.bool, device='cpu'))
        
        if classification_targets:
            self.val_targets['classification'].append(torch.stack(classification_targets, dim=1))
            self.val_masks['classification'].append(torch.stack(classification_masks, dim=1))
        
        # Store regression targets and masks
        regression_targets = []
        regression_masks = []
        for task in regression_tasks:
            if task in targets and task in masks:
                regression_targets.append(targets[task].detach().cpu())
                regression_masks.append(masks[task].detach().cpu())
            else:
                # Add dummy data for missing tasks (get batch_size from input batch)
                batch_size = batch['x'].shape[0]
                regression_targets.append(torch.zeros(batch_size, device='cpu'))
                regression_masks.append(torch.zeros(batch_size, dtype=torch.bool, device='cpu'))
        
        if regression_targets:
            self.val_targets['regression'].append(torch.stack(regression_targets, dim=1))
            self.val_masks['regression'].append(torch.stack(regression_masks, dim=1))
        
        # Log basic losses
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'val/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics using accumulated predictions."""
        logging.info(f"on_validation_epoch_end called - Classification batches: {len(self.val_predictions['classification'])}, Regression batches: {len(self.val_predictions['regression'])}")
        
        # Check if we have any predictions at all
        if not self.val_predictions['classification'] and not self.val_predictions['regression']:
            logging.warning("No validation predictions accumulated - skipping metrics computation")
            return
        
        # Handle missing classification or regression data
        if self.val_predictions['classification']:
            classification_preds = torch.cat(self.val_predictions['classification'], dim=0)
            classification_targets = torch.cat(self.val_targets['classification'], dim=0)
            classification_masks = torch.cat(self.val_masks['classification'], dim=0)
            logging.info(f"Classification predictions shape: {classification_preds.shape}")
        else:
            # Create dummy tensors if no classification data
            batch_size = len(self.val_predictions['regression'][0]) if self.val_predictions['regression'] else 1
            num_classification_tasks = 10  # From config
            classification_preds = torch.zeros(batch_size, num_classification_tasks)
            classification_targets = torch.zeros(batch_size, num_classification_tasks)
            classification_masks = torch.zeros(batch_size, num_classification_tasks, dtype=torch.bool)
        
        if self.val_predictions['regression']:
            regression_preds = torch.cat(self.val_predictions['regression'], dim=0)
            regression_targets = torch.cat(self.val_targets['regression'], dim=0)
            regression_masks = torch.cat(self.val_masks['regression'], dim=0)
            logging.info(f"Regression predictions shape: {regression_preds.shape}")
        else:
            # Create dummy tensors if no regression data
            batch_size = len(self.val_predictions['classification'][0]) if self.val_predictions['classification'] else 1
            num_regression_tasks = 8  # From config
            regression_preds = torch.zeros(batch_size, num_regression_tasks)
            regression_targets = torch.zeros(batch_size, num_regression_tasks)
            regression_masks = torch.zeros(batch_size, num_regression_tasks, dtype=torch.bool)
        
        # Compute epoch-level metrics
        try:
            epoch_metrics = self._compute_epoch_metrics(
                classification_preds, classification_targets, classification_masks,
                regression_preds, regression_targets, regression_masks
            )
            
            logging.info(f"Computed {len(epoch_metrics)} epoch-level metrics: {list(epoch_metrics.keys())}")
            
            # Log epoch-level metrics
            for metric_name, metric_value in epoch_metrics.items():
                self.log(f'val/{metric_name}', metric_value, on_epoch=True)
                logging.info(f"Logged metric: val/{metric_name} = {metric_value:.4f}")
                
        except Exception as e:
            logging.error(f"Error computing epoch metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear accumulated predictions for next epoch
        self.val_predictions['classification'].clear()
        self.val_predictions['regression'].clear()
        self.val_targets['classification'].clear()
        self.val_targets['regression'].clear()
        self.val_masks['classification'].clear()
        self.val_masks['regression'].clear()
    
    def _compute_epoch_metrics(self, classification_preds, classification_targets, classification_masks,
                              regression_preds, regression_targets, regression_masks):
        """Compute metrics across the entire epoch."""
        from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
        from scipy.stats import spearmanr
        import numpy as np
        
        metrics = {}
        
        # Get task configurations
        from configs.table1_task_config import get_all_tasks, get_task_type_mapping
        task_type_mapping = get_task_type_mapping()
        all_tasks = get_all_tasks()
        
        regression_tasks = [task for task in all_tasks if task_type_mapping[task] == 'regression']
        classification_tasks = [task for task in all_tasks if task_type_mapping[task] == 'classification']
        
        # Regression metrics
        for i, task in enumerate(regression_tasks):
            if i < regression_preds.shape[1]:
                # Ensure mask matches the batch size of predictions
                batch_size = min(regression_preds.shape[0], regression_masks.shape[0])
                mask = regression_masks[:batch_size, i].numpy().astype(bool)
                if mask.sum() > 0:
                    pred = regression_preds[:batch_size, i][mask].numpy()
                    target = regression_targets[:batch_size, i][mask].numpy()
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(pred) | np.isnan(target))
                    if valid_mask.sum() > 0:
                        pred = pred[valid_mask]
                        target = target[valid_mask]
                        
                        # Compute MAE
                        metrics[f'{task}_MAE'] = mean_absolute_error(target, pred)
                        
                        # Compute Spearman correlation if needed
                        from configs.table1_task_config import TASK_CONFIG
                        task_idx = regression_tasks.index(task)
                        metric_type = TASK_CONFIG['regression']['metrics'][task_idx] if task_idx < len(TASK_CONFIG['regression']['metrics']) else 'MAE'
                        if 'Spearman' in str(metric_type) and len(pred) > 1:
                            corr, _ = spearmanr(target, pred)
                            metrics[f'{task}_Spearman'] = corr if not np.isnan(corr) else 0.0
        
        # Classification metrics  
        for i, task in enumerate(classification_tasks):
            if i < classification_preds.shape[1]:
                # Ensure mask matches the batch size of predictions
                batch_size = min(classification_preds.shape[0], classification_masks.shape[0])
                mask = classification_masks[:batch_size, i].numpy().astype(bool)
                if mask.sum() > 0:
                    pred_logits = classification_preds[:batch_size, i][mask].numpy()
                    target = classification_targets[:batch_size, i][mask].numpy()
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(pred_logits) | np.isnan(target))
                    if valid_mask.sum() > 0:
                        pred_logits = pred_logits[valid_mask]
                        target = target[valid_mask]
                        
                        # Convert logits to probabilities
                        pred_proba = 1 / (1 + np.exp(-pred_logits))
                        
                        # Check if we have both classes
                        if len(np.unique(target)) > 1 and len(pred_proba) > 1:
                            try:
                                # Determine metric type
                                from configs.table1_task_config import TASK_CONFIG
                                task_idx = classification_tasks.index(task)
                                metric_type = TASK_CONFIG['classification']['metrics'][task_idx]
                                
                                if metric_type == 'AUROC':
                                    metrics[f'{task}_AUROC'] = roc_auc_score(target, pred_proba)
                                elif metric_type == 'AUPRC':
                                    metrics[f'{task}_AUPRC'] = average_precision_score(target, pred_proba)
                            except (ValueError, IndexError):
                                # Fallback to 0 if computation fails
                                if 'AUROC' in task:
                                    metrics[f'{task}_AUROC'] = 0.0
                                else:
                                    metrics[f'{task}_AUPRC'] = 0.0
        
        return metrics

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs, batch = self.forward(batch)
        
        # Get targets and masks from batch
        targets, masks = self._extract_targets_and_masks(batch)
        
        # Accumulate predictions for test metrics
        # Store classification predictions and targets
        classification_targets, classification_masks = self._process_classification_targets_masks(targets, masks)
        if classification_targets is not None and outputs.get('classification_output') is not None:
            self.test_predictions['classification'].append(outputs['classification_output'].detach().cpu())
            self.test_targets['classification'].append(classification_targets.detach().cpu())
            self.test_masks['classification'].append(classification_masks.detach().cpu())
        
        # Store regression predictions and targets
        regression_targets, regression_masks = self._process_regression_targets_masks(targets, masks)
        if regression_targets is not None and outputs.get('regression_output') is not None:
            self.test_predictions['regression'].append(outputs['regression_output'].detach().cpu())
            self.test_targets['regression'].append(regression_targets.detach().cpu())
            self.test_masks['regression'].append(regression_masks.detach().cpu())
        
        # Compute loss for logging
        device = next(self.parameters()).device
        loss_dict = self.model.compute_loss_and_metrics(
            regression_output=outputs.get('regression_output'),
            classification_output=outputs.get('classification_output'),
            targets=targets,
            masks=masks,
            device=device
        )
        
        total_loss = loss_dict['total_loss']
        
        # Log metrics
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'val/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of testing."""
        if self.test_predictions['classification'] or self.test_predictions['regression']:
            logging.info(f"Computing test metrics")
            
            # Concatenate predictions if available
            if self.test_predictions['classification']:
                classification_preds = torch.cat(self.test_predictions['classification'], dim=0)
                classification_targets = torch.cat(self.test_targets['classification'], dim=0)
                classification_masks = torch.cat(self.test_masks['classification'], dim=0)
            else:
                # Create dummy tensors
                classification_preds = torch.zeros(1, 10)
                classification_targets = torch.zeros(1, 10)
                classification_masks = torch.zeros(1, 10, dtype=torch.bool)
            
            if self.test_predictions['regression']:
                regression_preds = torch.cat(self.test_predictions['regression'], dim=0)
                regression_targets = torch.cat(self.test_targets['regression'], dim=0)
                regression_masks = torch.cat(self.test_masks['regression'], dim=0)
            else:
                # Create dummy tensors
                regression_preds = torch.zeros(1, 8)
                regression_targets = torch.zeros(1, 8)
                regression_masks = torch.zeros(1, 8, dtype=torch.bool)
            
            # Compute test metrics
            try:
                test_metrics = self._compute_epoch_metrics(
                    classification_preds, classification_targets, classification_masks,
                    regression_preds, regression_targets, regression_masks
                )
                
                logging.info(f"Computed {len(test_metrics)} test metrics")
                
                # Log test metrics
                for metric_name, metric_value in test_metrics.items():
                    self.log(f'test/{metric_name}', metric_value, on_epoch=True)
                    logging.info(f"Logged test metric: test/{metric_name} = {metric_value:.4f}")
                    
            except Exception as e:
                logging.error(f"Error computing test metrics: {e}")
                import traceback
                traceback.print_exc()
            
            # Clear accumulated predictions
            self.test_predictions['classification'].clear()
            self.test_predictions['regression'].clear()
            self.test_targets['classification'].clear()
            self.test_targets['regression'].clear()
            self.test_masks['classification'].clear()
            self.test_masks['regression'].clear()
    
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
        
        # Ensure warmup_pct is reasonable to avoid division by zero
        if total_steps > 0:
            warmup_pct = min(self.warmup_steps / total_steps, 0.3)  # Cap at 30%
            warmup_pct = max(warmup_pct, 0.01)  # Minimum 1%
        else:
            warmup_pct = 0.1  # Default 10%
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=max(total_steps, 1),  # Ensure at least 1 step
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
    
    # Filter tasks based on arguments
    selected_tasks = filter_tasks(args)
    
    # Create data loader
    logging.info("Setting up data loader...")
    data_loader = create_admet_dataloader(
        data_path=args.data_path,
        selected_tasks=selected_tasks,
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
        pretrained_path=args.pretrained_path,
        regression_loss_weight=args.regression_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        selected_tasks=selected_tasks
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