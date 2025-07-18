"""
Training script for Table 1 MolE model.

This script demonstrates end-to-end training of the integrated MolE model
with Table 1 prediction heads on ADMET datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path to import mole modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mole.models.table1_mole_model import create_table1_mole_model
from mole.dataset_loader import create_admet_dataloader
from mole.table1_prediction_heads import Table1PredictionHeads

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Table1MolETrainer:
    """
    Trainer class for Table 1 MolE model.
    
    This class handles the training loop, validation, and evaluation
    for the integrated MolE model with Table 1 prediction heads.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        num_epochs: int = 10,
        save_dir: str = "checkpoints",
        log_interval: int = 100,
        eval_interval: int = 500,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The Table1MolEModel to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to run training on
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            log_interval: Interval for logging training progress
            eval_interval: Interval for evaluation
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # Create save directory
        self.save_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    

    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = batch.to(self.device)
            
            # Get input data from PyTorch Geometric batch
            input_ids = batch.x  # [batch_size, seq_len]
            batch_size = input_ids.size(0)
            
            # Create input mask (1 for valid tokens, 0 for padding)
            # Assuming PAD token ID is 0
            input_mask = (input_ids != 0).long()
            
            # Get targets and masks from batch
            # In PyTorch Geometric, the targets and masks are stored as attributes
            targets = {}
            masks = {}
            
            # Get all task names from the first sample in the batch
            if hasattr(batch, 'targets'):
                # If targets is a dictionary attribute
                for task_name in batch.targets.keys():
                    if hasattr(batch, task_name):
                        targets[task_name] = getattr(batch, task_name)
                    if hasattr(batch, f"{task_name}_mask"):
                        masks[task_name] = getattr(batch, f"{task_name}_mask")
            else:
                # Try to get targets and masks directly from batch attributes
                # This assumes the batch has been properly constructed with task data
                for attr_name in dir(batch):
                    if not attr_name.startswith('_') and not callable(getattr(batch, attr_name)):
                        if attr_name.endswith('_mask'):
                            task_name = attr_name[:-5]  # Remove '_mask' suffix
                            if hasattr(batch, task_name):
                                targets[task_name] = getattr(batch, task_name)
                                masks[task_name] = getattr(batch, attr_name)
            
            # Create relative position matrix from edge information
            from torch_geometric.utils import to_dense_batch, to_dense_adj
            relative_pos = to_dense_adj(
                edge_index=batch.edge_index, 
                batch=batch.batch, 
                edge_attr=batch.edge_attr
            )
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids, 
                input_mask=input_mask,
                relative_pos=relative_pos
            )
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                regression_output=outputs['regression_output'],
                classification_output=outputs['classification_output'],
                targets=targets,
                masks=masks,
                device=self.device
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            self.global_step += 1
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log training progress
            if batch_idx % self.log_interval == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")
            
            # Evaluate periodically
            if self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                logger.info(f"Validation at step {self.global_step}: {val_metrics}")
        
        # Compute average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'train_loss': avg_loss.item()}
    
    def evaluate(self, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate the model on validation or test set.
        
        Args:
            split: Which split to evaluate on ('val' or 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        if split == 'val':
            data_loader = self.val_loader
        elif split == 'test':
            data_loader = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")
        
        total_loss = 0.0
        all_metrics = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Get input data from PyTorch Geometric batch
                input_ids = batch.x  # [batch_size, seq_len]
                batch_size = input_ids.size(0)
                
                # Create input mask (1 for valid tokens, 0 for padding)
                input_mask = (input_ids != 0).long()
                
                # Get targets and masks from batch
                # In PyTorch Geometric, the targets and masks are stored as attributes
                targets = {}
                masks = {}
                
                # Get all task names from the first sample in the batch
                if hasattr(batch, 'targets'):
                    # If targets is a dictionary attribute
                    for task_name in batch.targets.keys():
                        if hasattr(batch, task_name):
                            targets[task_name] = getattr(batch, task_name)
                        if hasattr(batch, f"{task_name}_mask"):
                            masks[task_name] = getattr(batch, f"{task_name}_mask")
                else:
                    # Try to get targets and masks directly from batch attributes
                    # This assumes the batch has been properly constructed with task data
                    for attr_name in dir(batch):
                        if not attr_name.startswith('_') and not callable(getattr(batch, attr_name)):
                            if attr_name.endswith('_mask'):
                                task_name = attr_name[:-5]  # Remove '_mask' suffix
                                if hasattr(batch, task_name):
                                    targets[task_name] = getattr(batch, task_name)
                                    masks[task_name] = getattr(batch, attr_name)
                
                # Create relative position matrix from edge information
                from torch_geometric.utils import to_dense_batch, to_dense_adj
                relative_pos = to_dense_adj(
                    edge_index=batch.edge_index, 
                    batch=batch.batch, 
                    edge_attr=batch.edge_attr
                )
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, 
                    input_mask=input_mask,
                    relative_pos=relative_pos
                )
                
                # Compute loss
                loss_dict = self.model.compute_loss(
                    regression_output=outputs['regression_output'],
                    classification_output=outputs['classification_output'],
                    targets=targets,
                    masks=masks,
                    device=self.device
                )
                
                # Compute metrics
                metrics = self.model.compute_metrics(
                    regression_output=outputs['regression_output'],
                    classification_output=outputs['classification_output'],
                    targets=targets,
                    masks=masks,
                    device=self.device
                )
                
                total_loss += loss_dict['total_loss'].item()
                all_metrics.append(metrics)
                num_batches += 1
        
        # Compute average loss and metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Average metrics across batches
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        result = {f'{split}_loss': avg_loss, **avg_metrics}
        
        return result
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['train_loss'])
            
            # Evaluate on validation set
            val_metrics = self.evaluate('val')
            self.val_losses.append(val_metrics['val_loss'])
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Print some key metrics
            for key, value in val_metrics.items():
                if key != 'val_loss':
                    logger.info(f"  {key}: {value:.4f}")
            
            # Check if this is the best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Store metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
        
        # Final evaluation on test set
        logger.info("Training completed. Evaluating on test set...")
        test_metrics = self.evaluate('test')
        
        logger.info("Final Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("Training completed successfully!")


def main():
    """Main training function."""
    logger.info("Setting up Table 1 MolE training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloader = create_admet_dataloader(
        batch_size=16,  # Smaller batch size for demonstration
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    train_loader = dataloader.get_dataloader('train')
    val_loader = dataloader.get_dataloader('val')
    test_loader = dataloader.get_dataloader('test')
    
    # Get dataset information
    info = dataloader.get_dataset_info()
    logger.info(f"Dataset loaded: {info['total_samples']} total samples")
    logger.info(f"Train: {info['splits']['train']}, Val: {info['splits']['val']}, Test: {info['splits']['test']}")
    
    # Create model
    logger.info("Creating model...")
    model = create_table1_mole_model(
        dropout=0.1,
        freeze_encoder=False,  # Set to True if you want to freeze pretrained weights
        pretrained_path=None   # Add path to pretrained weights if available
    )
    
    # Create optimizer and scheduler
    logger.info("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,  # Number of epochs
        eta_min=1e-6
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Table1MolETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=3,  # Small number for demonstration
        save_dir="checkpoints/table1_mole",
        log_interval=10,
        eval_interval=50
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 