"""
PyTorch Lightning Module for ChemBL Pretraining

This module wraps the ChemBL model for training with PyTorch Lightning,
handling both MLM and binary classification tasks with appropriate metrics.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection, 
    Accuracy, 
    AUROC, 
    AveragePrecision,
    F1Score,
    Precision,
    Recall
)
from torchmetrics.text import Perplexity

from mole.models.crossenv_mlm import CrossEnvMLM  
from mole.models.chembl_model import ChemBLModel


class ChemBLLightningModule(CrossEnvMLM):
    """PyTorch Lightning wrapper for ChemBL model training."""

    def __init__(
        self,
        model: ChemBLModel,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        log_predictions: bool = False,
        mlm_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        log_target_metrics: bool = True,
        max_targets_to_log: int = 50,
        **kwargs,
    ):
        """
        Initialize ChemBL Lightning module.
        
        Args:
            model: ChemBL model instance
            optimizer_cfg: Optimizer configuration
            scheduler_cfg: Learning rate scheduler configuration
            log_predictions: Whether to log predictions during validation
            mlm_loss_weight: Weight for MLM loss
            classification_loss_weight: Weight for classification loss
            log_target_metrics: Whether to log per-target metrics
            max_targets_to_log: Maximum number of targets to log individually
        """
        # Initialize parent class (handles MLM components)
        super().__init__(
            model=model,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            log_predictions=log_predictions,
            **kwargs
        )
        
        self.mlm_loss_weight = mlm_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.log_target_metrics = log_target_metrics
        self.max_targets_to_log = max_targets_to_log
        
        # Initialize classification metrics
        self._setup_classification_metrics()
        
    def _setup_classification_metrics(self):
        """Setup metrics for binary classification tasks."""
        num_targets = self.model.num_targets
        
        # Global classification metrics (averaged across all targets)
        classification_metrics = {
            "accuracy": Accuracy(task="binary"),
            "auroc": AUROC(task="binary"),
            "avg_precision": AveragePrecision(task="binary"),
            "f1": F1Score(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
        }
        
        # Training metrics
        self.train_classification_metrics = MetricCollection({
            f"train/classification/{name}": metric.clone()
            for name, metric in classification_metrics.items()
        })
        
        # Validation metrics
        self.val_classification_metrics = MetricCollection({
            f"val/classification/{name}": metric.clone()
            for name, metric in classification_metrics.items()
        })
        
        # Per-target metrics (only for a subset to avoid metric explosion)
        if self.log_target_metrics and num_targets <= self.max_targets_to_log:
            self.train_target_metrics = torch.nn.ModuleDict()
            self.val_target_metrics = torch.nn.ModuleDict()
            
            for i in range(min(num_targets, self.max_targets_to_log)):
                target_name = f"target_{i}"
                
                self.train_target_metrics[target_name] = MetricCollection({
                    f"train/target_{i}/auroc": AUROC(task="binary"),
                    f"train/target_{i}/avg_precision": AveragePrecision(task="binary"),
                })
                
                self.val_target_metrics[target_name] = MetricCollection({
                    f"val/target_{i}/auroc": AUROC(task="binary"),
                    f"val/target_{i}/avg_precision": AveragePrecision(task="binary"),
                })
        
    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        from torch_geometric.utils import to_dense_batch, to_dense_adj

        # Safety check for empty batches
        if batch.num_graphs == 0:
            # Create a dummy loss that requires grad and is connected to model parameters
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return {"loss": dummy_loss}

        # Additional safety checks for edge indices
        if batch.edge_index.numel() > 0 and batch.x.size(0) > 0:
            max_node_idx = batch.edge_index.max().item()
            num_nodes = batch.x.size(0)
            if max_node_idx >= num_nodes:
                print(f"WARNING: Training - Edge index {max_node_idx} >= num_nodes {num_nodes}")
                print(f"Batch info: num_graphs={batch.num_graphs}, edge_index.shape={batch.edge_index.shape}")
                print(f"Batch sizes: {batch.batch.bincount() if hasattr(batch, 'batch') else 'No batch attr'}")
                # Create a dummy loss connected to model parameters to avoid gradient errors
                dummy_param = next(self.parameters())
                dummy_loss = (dummy_param * 0.0).sum()  # This preserves gradient flow
                return {"loss": dummy_loss}

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Create dense relative position matrix from graph connections
        try:
            relative_pos = to_dense_adj(
                edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr
            )
        except (RuntimeError, IndexError) as e:
            print(f"WARNING: Failed to create dense adjacency matrix: {e}")
            # Create a dummy loss connected to model parameters to avoid gradient errors
            dummy_param = next(self.parameters())
            dummy_loss = (dummy_param * 0.0).sum()  # This preserves gradient flow
            return {"loss": dummy_loss}

        # Forward pass
        # Skip MLM computation if MLM weight is 0 (ChemBL-only mode)
        mlm_labels = labels if self.mlm_loss_weight > 0 else None
        outputs = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=mlm_labels,
            chembl_targets=batch.chembl_targets,
            chembl_mask=batch.chembl_mask,
            relative_pos=relative_pos,
        )

        # Get losses
        # Parent model returns MLM loss as "loss", not "mlm_loss"
        mlm_loss = outputs.get("mlm_loss", outputs.get("loss", 0.0))
        classification_loss = outputs.get("classification_loss", 0.0)
        
        # Combine losses
        total_loss = (
            self.mlm_loss_weight * mlm_loss + 
            self.classification_loss_weight * classification_loss
        )

        # Update MLM metrics (inherited from parent)
        if hasattr(self, 'train_mlm_metrics') and 'mlm_logits' in outputs:
            mlm_logits = outputs['mlm_logits']
            if mlm_logits is not None:
                # Perplexity metric
                self.train_mlm_metrics["train/mlm/perplexity"].update(mlm_logits, labels)
                
                # Accuracy metrics (filter out padding tokens)
                mask = labels != -100
                if mask.sum() > 0:
                    flat_logits = mlm_logits.view(-1, mlm_logits.size(-1))
                    flat_labels = labels.view(-1)
                    
                    self.train_mlm_metrics["train/mlm/accuracy"].update(flat_logits, flat_labels)
                    self.train_mlm_metrics["train/mlm/accuracy_top5"].update(flat_logits, flat_labels)

        # Update classification metrics
        if 'classification_logits' in outputs and batch.chembl_mask.sum() > 0:
            self._update_classification_metrics(
                logits=outputs['classification_logits'],
                targets=batch.chembl_targets,
                mask=batch.chembl_mask,
                split='train'
            )

        # Log losses
        self.log_dict({
            "train/total_loss": total_loss,
            "train/mlm_loss": mlm_loss,
            "train/classification_loss": classification_loss,
        }, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        from torch_geometric.utils import to_dense_batch, to_dense_adj

        # Safety check for empty batches
        if batch.num_graphs == 0:
            return {"val_loss": torch.tensor(0.0, device=self.device)}

        # Additional safety checks for edge indices
        if batch.edge_index.numel() > 0 and batch.x.size(0) > 0:
            max_node_idx = batch.edge_index.max().item()
            num_nodes = batch.x.size(0)
            if max_node_idx >= num_nodes:
                print(f"WARNING: Validation - Edge index {max_node_idx} >= num_nodes {num_nodes}")
                # Skip this batch to avoid CUDA error (validation doesn't need gradients)
                return {"val_loss": torch.tensor(0.0, device=self.device)}

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Create dense relative position matrix
        try:
            relative_pos = to_dense_adj(
                edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr
            )
        except (RuntimeError, IndexError) as e:
            print(f"WARNING: Failed to create dense adjacency matrix in validation: {e}")
            # Return a dummy validation loss to continue training (validation doesn't need gradients)
            return {"val_loss": torch.tensor(0.0, device=self.device)}

        # Forward pass
        # Skip MLM computation if MLM weight is 0 (ChemBL-only mode)
        mlm_labels = labels if self.mlm_loss_weight > 0 else None
        outputs = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=mlm_labels,
            chembl_targets=batch.chembl_targets,
            chembl_mask=batch.chembl_mask,
            relative_pos=relative_pos,
        )

        # Get losses
        # Parent model returns MLM loss as "loss", not "mlm_loss"
        mlm_loss = outputs.get("mlm_loss", outputs.get("loss", 0.0))
        classification_loss = outputs.get("classification_loss", 0.0)
        
        total_loss = (
            self.mlm_loss_weight * mlm_loss + 
            self.classification_loss_weight * classification_loss
        )

        # Update MLM metrics
        if hasattr(self, 'val_mlm_metrics') and 'mlm_logits' in outputs:
            mlm_logits = outputs['mlm_logits']
            if mlm_logits is not None:
                self.val_mlm_metrics["val/mlm/perplexity"].update(mlm_logits, labels)
                
                mask = labels != -100
                if mask.sum() > 0:
                    flat_logits = mlm_logits.view(-1, mlm_logits.size(-1))
                    flat_labels = labels.view(-1)
                    
                    self.val_mlm_metrics["val/mlm/accuracy"].update(flat_logits, flat_labels)
                    self.val_mlm_metrics["val/mlm/accuracy_top5"].update(flat_logits, flat_labels)

        # Update classification metrics
        if 'classification_logits' in outputs and batch.chembl_mask.sum() > 0:
            self._update_classification_metrics(
                logits=outputs['classification_logits'],
                targets=batch.chembl_targets,
                mask=batch.chembl_mask,
                split='val'
            )

        # Log validation losses
        self.log_dict({
            "val/total_loss": total_loss,
            "val/mlm_loss": mlm_loss,
            "val/classification_loss": classification_loss,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": total_loss}

    def test_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single test step (same as validation)."""
        return self.validation_step(batch, batch_idx)

    def _update_classification_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor, 
        mask: torch.Tensor,
        split: str
    ):
        """Update classification metrics for train/val."""
        if mask.sum() == 0:
            return
            
        # Ensure mask has the right shape for 2D indexing
        # Note: PyTorch Geometric flattens these tensors during batching, so we need to reshape them
        if mask.dim() == 1:
            # If mask is 1D, reshape it to match the expected 2D format
            batch_size = logits.size(0)
            num_targets = logits.size(1)
            if mask.numel() == batch_size * num_targets:
                mask = mask.view(batch_size, num_targets)
            else:
                print(f"WARNING: Cannot reshape mask from {mask.shape} to [{batch_size}, {num_targets}]")
                return
        
        # Similarly ensure targets has the right shape
        if targets.dim() == 1:
            batch_size = logits.size(0)
            num_targets = logits.size(1)
            if targets.numel() == batch_size * num_targets:
                targets = targets.view(batch_size, num_targets)
            else:
                print(f"WARNING: Cannot reshape targets from {targets.shape} to [{batch_size}, {num_targets}]")
                return
            
        # Get metrics collection for this split
        if split == 'train':
            metrics = self.train_classification_metrics
            target_metrics = getattr(self, 'train_target_metrics', None)
        else:
            metrics = self.val_classification_metrics  
            target_metrics = getattr(self, 'val_target_metrics', None)
            
        # Convert logits to probabilities and extract predictions
        probs = F.softmax(logits, dim=-1)  # [batch, num_targets, 2]
        active_probs = probs[:, :, 1]  # [batch, num_targets] - probability of active
        
        # Convert targets: 1=active (1), -1=inactive (0), 0=unmeasured (ignore)
        binary_targets = (targets == 1).long()
        
        # Flatten and mask
        flat_probs = active_probs.view(-1)  # [batch * num_targets]
        flat_preds = (active_probs > 0.5).long().view(-1)  # [batch * num_targets]
        flat_targets = binary_targets.view(-1)  # [batch * num_targets]
        flat_mask = mask.view(-1)  # [batch * num_targets]
        
        # Update global metrics (only measured targets)
        if flat_mask.sum() > 0:
            masked_probs = flat_probs[flat_mask]
            masked_preds = flat_preds[flat_mask]
            masked_targets = flat_targets[flat_mask]
            
            # Update metrics that need probabilities
            metrics[f"{split}/classification/auroc"].update(masked_probs, masked_targets)
            metrics[f"{split}/classification/avg_precision"].update(masked_probs, masked_targets)
            
            # Update metrics that need predictions
            metrics[f"{split}/classification/accuracy"].update(masked_preds, masked_targets)
            metrics[f"{split}/classification/f1"].update(masked_preds, masked_targets)
            metrics[f"{split}/classification/precision"].update(masked_preds, masked_targets)
            metrics[f"{split}/classification/recall"].update(masked_preds, masked_targets)
        
        # Update per-target metrics if enabled
        if target_metrics is not None and self.log_target_metrics:
            for target_idx in range(min(logits.size(1), self.max_targets_to_log)):
                target_mask = mask[:, target_idx]  # [batch]
                
                if target_mask.sum() > 0:
                    target_probs = active_probs[:, target_idx]  # [batch]
                    target_targets = binary_targets[:, target_idx]  # [batch]
                    
                    # Only measured samples for this target
                    masked_target_probs = target_probs[target_mask]
                    masked_target_targets = target_targets[target_mask]
                    
                    target_name = f"target_{target_idx}"
                    if target_name in target_metrics:
                        target_metrics[target_name][f"{split}/target_{target_idx}/auroc"].update(
                            masked_target_probs, masked_target_targets
                        )
                        target_metrics[target_name][f"{split}/target_{target_idx}/avg_precision"].update(
                            masked_target_probs, masked_target_targets
                        )

    def on_train_epoch_end(self):
        """Log training metrics at end of epoch."""
        # Log MLM metrics
        if hasattr(self, 'train_mlm_metrics'):
            try:
                self.log_dict(self.train_mlm_metrics.compute(), on_epoch=True)
            except (ValueError, RuntimeError) as e:
                self.log("train/mlm/error", 1.0, on_epoch=True)
                print(f"Warning: Failed to compute MLM training metrics: {e}")
            self.train_mlm_metrics.reset()
            
        # Log classification metrics
        try:
            classification_metrics = self.train_classification_metrics.compute()
            self.log_dict(classification_metrics, on_epoch=True)
        except (ValueError, RuntimeError) as e:
            self.log("train/classification/error", 1.0, on_epoch=True)
            print(f"Warning: Failed to compute training classification metrics: {e}")
        self.train_classification_metrics.reset()
        
        # Log per-target metrics
        if hasattr(self, 'train_target_metrics'):
            for target_metrics in self.train_target_metrics.values():
                try:
                    self.log_dict(target_metrics.compute(), on_epoch=True)
                except (ValueError, RuntimeError) as e:
                    self.log("train/target/error", 1.0, on_epoch=True)
                    print(f"Warning: Failed to compute per-target training metrics: {e}")
                target_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at end of epoch.""" 
        # Log MLM metrics
        if hasattr(self, 'val_mlm_metrics'):
            try:
                self.log_dict(self.val_mlm_metrics.compute(), on_epoch=True)
            except (ValueError, RuntimeError) as e:
                self.log("val/mlm/error", 1.0, on_epoch=True)
                print(f"Warning: Failed to compute MLM validation metrics: {e}")
            self.val_mlm_metrics.reset()
            
        # Log classification metrics
        try:
            classification_metrics = self.val_classification_metrics.compute()
            self.log_dict(classification_metrics, on_epoch=True)
        except (ValueError, RuntimeError) as e:
            # This can happen when there are no measured targets in validation set
            self.log("val/classification/error", 1.0, on_epoch=True)
            print(f"Warning: Failed to compute validation classification metrics: {e}")
        self.val_classification_metrics.reset()
        
        # Log per-target metrics
        if hasattr(self, 'val_target_metrics'):
            for target_metrics in self.val_target_metrics.values():
                try:
                    self.log_dict(target_metrics.compute(), on_epoch=True)
                except (ValueError, RuntimeError) as e:
                    self.log("val/target/error", 1.0, on_epoch=True)
                    print(f"Warning: Failed to compute per-target validation metrics: {e}")
                target_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        return super().configure_optimizers() 