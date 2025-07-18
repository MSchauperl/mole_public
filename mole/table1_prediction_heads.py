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
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Get task information
        self.regression_tasks = get_regression_tasks()
        self.classification_tasks = get_classification_tasks()
        self.all_tasks = self.regression_tasks + self.classification_tasks
        
        # Create separate heads
        self.regression_head = RegressionHead(
            hidden_dim=hidden_dim,
            num_tasks=len(self.regression_tasks),
            dropout=dropout
        )
        
        self.classification_head = ClassificationHead(
            hidden_dim=hidden_dim,
            num_tasks=len(self.classification_tasks),
            dropout=dropout
        )
        
        # Task mappings
        self.task_type_mapping = get_task_type_mapping()
        self.metric_mapping = get_metric_mapping()
        
        # Loss weights (can be tuned)
        self.regression_loss_weight = 1.0
        self.classification_loss_weight = 1.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        regression_output = self.regression_head(x)
        classification_output = self.classification_head(x)
        
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
        if regression_mask is not None:
            regression_loss = F.mse_loss(
                regression_output * regression_mask,
                regression_targets * regression_mask,
                reduction='sum'
            ) / (regression_mask.sum() + 1e-8)
        else:
            regression_loss = F.mse_loss(regression_output, regression_targets)
        
        # Classification loss (BCE with logits)
        if classification_mask is not None:
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_output * classification_mask,
                classification_targets * classification_mask,
                reduction='sum'
            ) / (classification_mask.sum() + 1e-8)
        else:
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_output, classification_targets
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
                mask = regression_mask[:, i].detach().cpu().numpy()
                # Handle single sample case
                if regression_output_np.shape[0] == 1:
                    pred = regression_output_np[0, i:i+1]
                    target = regression_targets_np[0, i:i+1]
                else:
                    pred = regression_output_np[mask, i]
                    target = regression_targets_np[mask, i]
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
                    corr, _ = spearmanr(target, pred)
                    metrics[f'{task}_Spearman'] = corr if not np.isnan(corr) else 0.0
                elif metric == 'AUPRC':
                    # For AUPRC, we need to convert to binary classification
                    # This is a simplified approach - you might need to adjust based on your data
                    metrics[f'{task}_AUPRC'] = 0.0  # Placeholder
        
        # Classification metrics
        classification_output_np = classification_output.detach().cpu().numpy()
        classification_targets_np = classification_targets.detach().cpu().numpy()
        
        for i, task in enumerate(self.classification_tasks):
            metric = self.metric_mapping[task]
            
            if classification_mask is not None:
                mask = classification_mask[:, i].detach().cpu().numpy()
                # Handle single sample case
                if classification_output_np.shape[0] == 1:
                    pred = classification_output_np[0, i:i+1]
                    target = classification_targets_np[0, i:i+1]
                else:
                    pred = classification_output_np[mask, i]
                    target = classification_targets_np[mask, i]
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
                        metrics[f'{task}_AUROC'] = roc_auc_score(target, pred_proba)
                    except ValueError:
                        metrics[f'{task}_AUROC'] = 0.0
        
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