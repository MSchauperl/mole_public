"""
Example script demonstrating the use of the ADMET dataset loader
with Table 1 prediction heads for training.

This script shows how to:
1. Load and preprocess ADMET data
2. Create prediction heads
3. Set up a simple training loop
4. Handle both regression and classification tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import sys
import os

# Add the parent directory to the path to import mole modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mole.dataset_loader import create_admet_dataloader
from mole.table1_prediction_heads import create_table1_prediction_heads

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMolEModel(nn.Module):
    """
    Simple mock MolE model for demonstration.
    In practice, this would be replaced with the actual MolE model.
    """
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Mock embedding layer (would be replaced with actual MolE embeddings)
        self.embedding = nn.Sequential(
            nn.Linear(100, hidden_dim),  # Mock input dimension
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prediction heads for Table 1 tasks
        self.prediction_heads = create_table1_prediction_heads(
            hidden_dim=hidden_dim,
            dropout=0.1
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (mock molecular representation)
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        # Generate mock embeddings (in practice, this would be MolE embeddings)
        embeddings = self.embedding(x)
        
        # Pass through prediction heads
        regression_output, classification_output = self.prediction_heads(embeddings)
        
        return regression_output, classification_output


def create_mock_batch(batch_size: int = 16, input_dim: int = 100):
    """
    Create a mock batch for demonstration.
    In practice, this would be replaced with actual molecular data.
    """
    # Mock molecular representations
    x = torch.randn(batch_size, input_dim)
    
    # Mock targets (regression and classification)
    regression_targets = torch.randn(batch_size, 13)  # 13 regression tasks
    classification_targets = torch.randint(0, 2, (batch_size, 5)).float()  # 5 classification tasks
    
    # Create masks for valid samples
    regression_mask = torch.ones(batch_size, 13)
    classification_mask = torch.ones(batch_size, 5)
    
    return {
        'x': x,
        'regression_targets': regression_targets,
        'classification_targets': classification_targets,
        'regression_mask': regression_mask,
        'classification_mask': classification_mask
    }


def train_step(model, optimizer, batch, device):
    """
    Single training step.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        batch: Batch of data
        device: Device to run on
        
    Returns:
        Dictionary containing loss and metrics
    """
    model.train()
    
    # Move data to device
    x = batch['x'].to(device)
    regression_targets = batch['regression_targets'].to(device)
    classification_targets = batch['classification_targets'].to(device)
    regression_mask = batch['regression_mask'].to(device)
    classification_mask = batch['classification_mask'].to(device)
    
    # Forward pass
    regression_output, classification_output = model(x)
    
    # Compute loss
    loss_dict = model.prediction_heads.compute_loss(
        regression_output=regression_output,
        classification_output=classification_output,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_mask=regression_mask,
        classification_mask=classification_mask
    )
    
    total_loss = loss_dict['total_loss']
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        metrics = model.prediction_heads.compute_metrics(
            regression_output=regression_output,
            classification_output=classification_output,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            regression_mask=regression_mask,
            classification_mask=classification_mask
        )
    
    return {
        'loss': total_loss.item(),
        'regression_loss': loss_dict['regression_loss'].item(),
        'classification_loss': loss_dict['classification_loss'].item(),
        'metrics': metrics
    }


def validate_step(model, batch, device):
    """
    Single validation step.
    
    Args:
        model: The model to validate
        batch: Batch of data
        device: Device to run on
        
    Returns:
        Dictionary containing loss and metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        x = batch['x'].to(device)
        regression_targets = batch['regression_targets'].to(device)
        classification_targets = batch['classification_targets'].to(device)
        regression_mask = batch['regression_mask'].to(device)
        classification_mask = batch['classification_mask'].to(device)
        
        # Forward pass
        regression_output, classification_output = model(x)
        
        # Compute loss
        loss_dict = model.prediction_heads.compute_loss(
            regression_output=regression_output,
            classification_output=classification_output,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            regression_mask=regression_mask,
            classification_mask=classification_mask
        )
        
        # Compute metrics
        metrics = model.prediction_heads.compute_metrics(
            regression_output=regression_output,
            classification_output=classification_output,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            regression_mask=regression_mask,
            classification_mask=classification_mask
        )
    
    return {
        'loss': loss_dict['total_loss'].item(),
        'regression_loss': loss_dict['regression_loss'].item(),
        'classification_loss': loss_dict['classification_loss'].item(),
        'metrics': metrics
    }


def main():
    """Main training function."""
    logger.info("Starting ADMET training example")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset loader
    logger.info("Creating dataset loader...")
    dataloader = create_admet_dataloader(
        batch_size=32,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Get dataset information
    info = dataloader.get_dataset_info()
    logger.info(f"Dataset loaded: {info['total_samples']} total samples")
    logger.info(f"Train: {info['splits']['train']}, Val: {info['splits']['val']}, Test: {info['splits']['test']}")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleMolEModel(hidden_dim=512).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    logger.info("Starting training loop...")
    num_epochs = 3
    train_steps_per_epoch = 10  # For demonstration
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        train_metrics = []
        
        for step in range(train_steps_per_epoch):
            # Create mock batch (in practice, this would come from the dataloader)
            batch = create_mock_batch(batch_size=32)
            
            # Training step
            result = train_step(model, optimizer, batch, device)
            train_losses.append(result['loss'])
            train_metrics.append(result['metrics'])
            
            if step % 5 == 0:
                logger.info(f"  Step {step}: Loss = {result['loss']:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        val_metrics = []
        
        for step in range(5):  # Fewer validation steps
            batch = create_mock_batch(batch_size=32)
            result = validate_step(model, batch, device)
            val_losses.append(result['loss'])
            val_metrics.append(result['metrics'])
        
        # Log epoch results
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        logger.info(f"  Epoch {epoch + 1} Results:")
        logger.info(f"    Train Loss: {avg_train_loss:.4f}")
        logger.info(f"    Val Loss: {avg_val_loss:.4f}")
        
        # Print some metrics
        if val_metrics:
            avg_metrics = {}
            for key in val_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in val_metrics) / len(val_metrics)
            
            logger.info("    Validation Metrics:")
            for task, metric in avg_metrics.items():
                logger.info(f"      {task}: {metric:.4f}")
    
    logger.info("Training completed!")
    
    # Demonstrate dataset loading with real data
    logger.info("\nDemonstrating real dataset loading...")
    train_loader = dataloader.get_dataloader('train')
    
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Real batch {batch_idx + 1}:")
        logger.info(f"  SMILES count: {len(batch['smiles'])}")
        logger.info(f"  Target tasks: {list(batch['targets'].keys())}")
        
        # Show first SMILES and some target values
        first_smiles = batch['smiles'][0]
        logger.info(f"  First SMILES: {first_smiles}")
        
        # Show some regression targets and masks
        regression_tasks = [task for task, config in dataloader.task_config.items() 
                          if config['task_type'] == 'regression']
        if regression_tasks:
            first_regression = regression_tasks[0]
            first_value = batch['targets'][first_regression][0].item()
            first_mask = batch['masks'][first_regression][0].item()
            logger.info(f"  First {first_regression} value: {first_value:.4f}, mask: {first_mask}")
        
        # Show some classification targets and masks
        classification_tasks = [task for task, config in dataloader.task_config.items() 
                              if config['task_type'] == 'classification']
        if classification_tasks:
            first_classification = classification_tasks[0]
            first_value = batch['targets'][first_classification][0].item()
            first_mask = batch['masks'][first_classification][0].item()
            logger.info(f"  First {first_classification} value: {first_value}, mask: {first_mask}")
        
        # Show mask statistics for the batch
        valid_counts = {}
        for task_name in batch['masks']:
            valid_count = sum(batch['masks'][task_name]).item()
            valid_counts[task_name] = valid_count
        
        logger.info(f"  Valid samples per task: {valid_counts}")
        
        if batch_idx >= 2:  # Only show first few batches
            break
    
    logger.info("Dataset loader demonstration completed!")


if __name__ == "__main__":
    main() 