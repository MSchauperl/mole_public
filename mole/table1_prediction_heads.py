"""
Prediction heads for Table 1 datasets from the MolE paper.
Separate heads for regression and classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score
from scipy.stats import spearmanr

from configs.table1_task_config import (
    get_regression_tasks, get_classification_tasks, 
    get_metric_mapping, get_task_type_mapping, MOLE_PAPER_RESULTS
)


class RegressionHead(nn.Module):
    """
    Prediction head for regression tasks (MAE, Spearman, AUPRC).
    
    Supports 13 regression tasks from Table 1:
    - Caco2, Lipophilicity, Solubility, PPBR (MAE)
    - VDss, Half_life, Clearance_microsome, Clearance_hepatocyte (Spearman)
    - CYP inhibition and substrate tasks (AUPRC)
    """
    
    def __init__(self, hidden_dim: int, num_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Multi-layer perceptron for regression
        self.regression_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_tasks)
        )
        
        # Task-specific output layers (optional - for more flexibility)
        self.task_outputs = nn.ModuleDict()
        regression_tasks = get_regression_tasks()
        for task in regression_tasks:
            self.task_outputs[task] = nn.Linear(hidden_dim // 4, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression tasks.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_tasks)
        """
        return self.regression_mlp(x)


class ClassificationHead(nn.Module):
    """
    Prediction head for classification tasks (AUROC).
    
    Supports 5 classification tasks from Table 1:
    - HIA, Pgp, Bioavailability, BBB, CYP3A4_substrate (AUROC)
    """
    
    def __init__(self, hidden_dim: int, num_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Multi-layer perceptron for classification
        self.classification_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_tasks)
        )
        
        # Task-specific output layers (optional - for more flexibility)
        self.task_outputs = nn.ModuleDict()
        classification_tasks = get_classification_tasks()
        for task in classification_tasks:
            self.task_outputs[task] = nn.Linear(hidden_dim // 4, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification tasks.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_tasks) with logits
        """
        return self.classification_mlp(x)


class Table1PredictionHeads(nn.Module):
    """
    Combined prediction heads for all Table 1 tasks.
    
    This module combines regression and classification heads
    and provides unified training and evaluation interfaces.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, 
                 regression_loss_weight: float = 1.0, 
                 classification_loss_weight: float = 1.0,
                 selected_tasks: Optional[List[str]] = None):
        super().__init__()
        
        # Get task information
        all_regression_tasks = get_regression_tasks()
        all_classification_tasks = get_classification_tasks()
        
        # Filter tasks based on selected_tasks if provided
        if selected_tasks is not None:
            self.regression_tasks = [task for task in all_regression_tasks if task in selected_tasks]
            self.classification_tasks = [task for task in all_classification_tasks if task in selected_tasks]
        else:
            self.regression_tasks = all_regression_tasks
            self.classification_tasks = all_classification_tasks
        
        self.all_tasks = self.regression_tasks + self.classification_tasks
        
        # Create heads only if there are tasks of each type
        if self.regression_tasks:
            self.regression_head = RegressionHead(
                hidden_dim=hidden_dim,
                num_tasks=len(self.regression_tasks),
                dropout=dropout
            )
        else:
            self.regression_head = None
            
        if self.classification_tasks:
            self.classification_head = ClassificationHead(
                hidden_dim=hidden_dim,
                num_tasks=len(self.classification_tasks),
                dropout=dropout
            )
        else:
            self.classification_head = None
        
        # Task mappings
        self.task_type_mapping = get_task_type_mapping()
        self.metric_mapping = get_metric_mapping()
        
        # Loss weights (configurable for balancing)
        self.regression_loss_weight = regression_loss_weight
        self.classification_loss_weight = classification_loss_weight
        
        # Log the task and loss configuration
        import logging
        logging.info(f"Prediction heads created for {len(self.all_tasks)} tasks:")
        if self.regression_tasks:
            logging.info(f"  Regression tasks ({len(self.regression_tasks)}): {self.regression_tasks}")
        if self.classification_tasks:
            logging.info(f"  Classification tasks ({len(self.classification_tasks)}): {self.classification_tasks}")
        logging.info(f"Loss weights: Regression={self.regression_loss_weight:.4f}, Classification={self.classification_loss_weight:.4f}")
    
    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through available heads.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (regression_output, classification_output)
            Either output can be None if the corresponding head doesn't exist
        """
        regression_output = self.regression_head(x) if self.regression_head is not None else None
        classification_output = self.classification_head(x) if self.classification_head is not None else None
        
        return regression_output, classification_output
    
    def compute_loss(self, 
                    regression_output: torch.Tensor,
                    classification_output: torch.Tensor,
                    regression_targets: torch.Tensor,
                    classification_targets: torch.Tensor,
                    regression_mask: Optional[torch.Tensor] = None,
                    classification_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for regression and classification tasks.
        
        Args:
            regression_output: Model predictions for regression tasks
            classification_output: Model predictions for classification tasks
            regression_targets: Ground truth for regression tasks
            classification_targets: Ground truth for classification tasks
            regression_mask: Mask for valid regression samples
            classification_mask: Mask for valid classification samples
            
        Returns:
            Dictionary containing individual and combined losses
        """
        # Regression loss (MSE)
        if regression_mask is not None and regression_mask.sum() > 0:
            # Apply mask properly - only compute loss for valid samples
            # regression_mask has shape [batch_size, num_regression_tasks]
            # We need to apply it to each task separately
            regression_losses = []
            for i in range(regression_output.shape[1]):
                task_mask = regression_mask[:regression_output.shape[0], i]
                if task_mask.sum() > 0:
                    task_output = regression_output[:, i][task_mask]
                    task_targets = regression_targets[:regression_output.shape[0], i][task_mask]
                    task_loss = F.mse_loss(task_output, task_targets)
                    regression_losses.append(task_loss)
            
            if regression_losses:
                regression_loss = torch.stack(regression_losses).mean()
            else:
                regression_loss = torch.tensor(0.0, device=regression_output.device)
        else:
            # Ensure batch sizes match
            batch_size = min(regression_output.shape[0], regression_targets.shape[0])
            regression_loss = F.mse_loss(regression_output[:batch_size], regression_targets[:batch_size])
        
        # Classification loss (BCE with logits)
        if classification_mask is not None and classification_mask.sum() > 0:
            # Apply mask properly - only compute loss for valid samples
            # classification_mask has shape [batch_size, num_classification_tasks]
            # We need to apply it to each task separately
            classification_losses = []
            for i in range(classification_output.shape[1]):
                task_mask = classification_mask[:classification_output.shape[0], i]
                if task_mask.sum() > 0:
                    task_output = classification_output[:, i][task_mask]
                    task_targets = classification_targets[:classification_output.shape[0], i][task_mask]
                    task_loss = F.binary_cross_entropy_with_logits(task_output, task_targets)
                    classification_losses.append(task_loss)
            
            if classification_losses:
                classification_loss = torch.stack(classification_losses).mean()
            else:
                classification_loss = torch.tensor(0.0, device=classification_output.device)
        else:
            # Ensure batch sizes match
            batch_size = min(classification_output.shape[0], classification_targets.shape[0])
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_output[:batch_size], classification_targets[:batch_size]
            )
        
        # Combined loss
        total_loss = (
            self.regression_loss_weight * regression_loss +
            self.classification_loss_weight * classification_loss
        )
        
        return {
            'regression_loss': regression_loss,
            'classification_loss': classification_loss,
            'total_loss': total_loss
        }
    
    def compute_metrics(self,
                       regression_output: torch.Tensor,
                       classification_output: torch.Tensor,
                       regression_targets: torch.Tensor,
                       classification_targets: torch.Tensor,
                       regression_mask: Optional[torch.Tensor] = None,
                       classification_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics for all tasks.
        
        Args:
            regression_output: Model predictions for regression tasks
            classification_output: Model predictions for classification tasks
            regression_targets: Ground truth for regression tasks
            classification_targets: Ground truth for classification tasks
            regression_mask: Mask for valid regression samples
            classification_mask: Mask for valid classification samples
            
        Returns:
            Dictionary containing metrics for each task
        """
        metrics = {}
        
        # Regression metrics
        regression_output_np = regression_output.detach().cpu().numpy()
        regression_targets_np = regression_targets.detach().cpu().numpy()
        
        for i, task in enumerate(self.regression_tasks):
            metric = self.metric_mapping[task]
            
            if regression_mask is not None:
                mask = regression_mask[:regression_output_np.shape[0], i].detach().cpu().numpy()
                if mask.sum() > 0:
                    pred = regression_output_np[:, i][mask]
                    target = regression_targets_np[:regression_output_np.shape[0], i][mask]
                else:
                    continue  # Skip if no valid samples
            else:
                pred = regression_output_np[:, i]
                target = regression_targets_np[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred) | np.isnan(target))
            if valid_mask.sum() > 0:
                pred = pred[valid_mask]
                target = target[valid_mask]
                
                if metric == 'MAE':
                    metrics[f'{task}_MAE'] = mean_absolute_error(target, pred)
                elif metric == 'Spearman':
                    if len(pred) > 1:  # Need at least 2 samples for correlation
                        corr, _ = spearmanr(target, pred)
                        metrics[f'{task}_Spearman'] = corr if not np.isnan(corr) else 0.0
                    else:
                        metrics[f'{task}_Spearman'] = 0.0
                # AUPRC tasks are now handled in classification metrics
        
        # Classification metrics
        classification_output_np = classification_output.detach().cpu().numpy()
        classification_targets_np = classification_targets.detach().cpu().numpy()
        
        # NOTE: This method computes metrics for a single batch
        # For proper AUROC/AUPRC computation, metrics should be accumulated across all batches
        # and computed at the end of the epoch. This batch-wise computation will be replaced
        # by epoch-level metric computation in the Lightning module.
        
        for i, task in enumerate(self.classification_tasks):
            metric = self.metric_mapping[task]
            
            if classification_mask is not None:
                mask = classification_mask[:classification_output_np.shape[0], i].detach().cpu().numpy()
                if mask.sum() > 0:
                    pred = classification_output_np[:, i][mask]
                    target = classification_targets_np[:classification_output_np.shape[0], i][mask]
                else:
                    continue  # Skip if no valid samples
            else:
                pred = classification_output_np[:, i]
                target = classification_targets_np[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred) | np.isnan(target))
            if valid_mask.sum() > 0:
                pred = pred[valid_mask]
                target = target[valid_mask]
                
                if metric == 'AUROC':
                    # Convert logits to probabilities
                    pred_proba = 1 / (1 + np.exp(-pred))
                    try:
                        # Check if we have both classes (need at least 2 samples with different classes)
                        if len(pred) > 1 and len(np.unique(target)) > 1:
                            auc_score = roc_auc_score(target, pred_proba)
                            metrics[f'{task}_AUROC'] = auc_score
                        else:
                            # Skip batch-wise computation - will be handled at epoch level
                            metrics[f'{task}_AUROC'] = 0.0
                    except ValueError:
                        metrics[f'{task}_AUROC'] = 0.0
                elif metric == 'AUPRC':
                    # Convert logits to probabilities
                    pred_proba = 1 / (1 + np.exp(-pred))
                    try:
                        # Check if we have both classes (need at least 2 samples with different classes)
                        if len(pred) > 1 and len(np.unique(target)) > 1:
                            auprc_score = average_precision_score(target, pred_proba)
                            metrics[f'{task}_AUPRC'] = auprc_score
                        else:
                            # Skip batch-wise computation - will be handled at epoch level
                            metrics[f'{task}_AUPRC'] = 0.0
                    except ValueError:
                        metrics[f'{task}_AUPRC'] = 0.0
        
        return metrics
    
    def compare_with_mole_paper(self, metrics: Dict[str, float]) -> Dict[str, Dict]:
        """
        Compare current results with MolE paper results.
        
        Args:
            metrics: Current evaluation metrics
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {}
        
        for task in self.all_tasks:
            if task in MOLE_PAPER_RESULTS:
                mole_result = MOLE_PAPER_RESULTS[task]
                metric_name = mole_result['metric']
                mole_value = mole_result['result']
                mole_std = mole_result['std']
                
                current_metric_key = f'{task}_{metric_name}'
                if current_metric_key in metrics:
                    current_value = metrics[current_metric_key]
                    
                    comparison[task] = {
                        'metric': metric_name,
                        'mole_result': mole_value,
                        'mole_std': mole_std,
                        'current_result': current_value,
                        'difference': current_value - mole_value,
                        'mole_status': mole_result['status']
                    }
        
        return comparison
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """
        Print a formatted summary of evaluation metrics.
        
        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n=== Table 1 Evaluation Metrics ===")
        
        # Regression tasks
        print("\nREGRESSION TASKS:")
        for task in self.regression_tasks:
            metric = self.metric_mapping[task]
            metric_key = f'{task}_{metric}'
            if metric_key in metrics:
                value = metrics[metric_key]
                mole_result = MOLE_PAPER_RESULTS.get(task, {})
                mole_value = mole_result.get('result', 'N/A')
                mole_status = mole_result.get('status', 'Unknown')
                print(f"  {task:20s} | {metric:8s} | {value:8.4f} | MolE: {mole_value:8.4f} | {mole_status}")
        
        # Classification tasks
        print("\nCLASSIFICATION TASKS:")
        for task in self.classification_tasks:
            metric = self.metric_mapping[task]
            metric_key = f'{task}_{metric}'
            if metric_key in metrics:
                value = metrics[metric_key]
                mole_result = MOLE_PAPER_RESULTS.get(task, {})
                mole_value = mole_result.get('result', 'N/A')
                mole_status = mole_result.get('status', 'Unknown')
                print(f"  {task:20s} | {metric:8s} | {value:8.4f} | MolE: {mole_value:8.4f} | {mole_status}")


def create_table1_prediction_heads(hidden_dim: int, dropout: float = 0.1) -> Table1PredictionHeads:
    """
    Factory function to create Table1PredictionHeads.
    
    Args:
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
        
    Returns:
        Table1PredictionHeads instance
    """
    return Table1PredictionHeads(hidden_dim=hidden_dim, dropout=dropout)


if __name__ == "__main__":
    # Test the prediction heads
    print("Testing Table1PredictionHeads...")
    
    # Create model
    hidden_dim = 512
    model = create_table1_prediction_heads(hidden_dim=hidden_dim)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, hidden_dim)
    regression_output, classification_output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Regression output shape: {regression_output.shape}")
    print(f"Classification output shape: {classification_output.shape}")
    print(f"Number of regression tasks: {len(model.regression_tasks)}")
    print(f"Number of classification tasks: {len(model.classification_tasks)}")
    
    # Test loss computation
    regression_targets = torch.randn(batch_size, len(model.regression_tasks))
    classification_targets = torch.randint(0, 2, (batch_size, len(model.classification_tasks))).float()
    
    losses = model.compute_loss(
        regression_output, classification_output,
        regression_targets, classification_targets
    )
    
    print(f"\nLosses:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Test metrics computation
    metrics = model.compute_metrics(
        regression_output, classification_output,
        regression_targets, classification_targets
    )
    
    print(f"\nMetrics computed: {len(metrics)}")
    for metric_name, metric_value in list(metrics.items())[:5]:  # Show first 5
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n✅ Table1PredictionHeads test completed successfully!")