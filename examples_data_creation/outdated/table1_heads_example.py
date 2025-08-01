"""
Example usage of Table1PredictionHeads.
This script demonstrates how to use the prediction heads for Table 1 tasks.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mole.table1_prediction_heads import create_table1_prediction_heads
from configs.table1_task_config import get_all_tasks, get_task_type_mapping, print_task_summary

def main():
    print("=== Table1PredictionHeads Example ===\n")
    
    # Print task summary
    print_task_summary()
    
    # Create prediction heads
    hidden_dim = 512
    prediction_heads = create_table1_prediction_heads(hidden_dim=hidden_dim)
    
    print(f"\nCreated prediction heads with hidden_dim={hidden_dim}")
    print(f"Regression tasks: {len(prediction_heads.regression_tasks)}")
    print(f"Classification tasks: {len(prediction_heads.classification_tasks)}")
    
    # Simulate MolE embeddings
    batch_size = 16
    mol_embeddings = torch.randn(batch_size, hidden_dim)
    
    print(f"\nInput MolE embeddings shape: {mol_embeddings.shape}")
    
    # Forward pass
    with torch.no_grad():
        regression_output, classification_output = prediction_heads(mol_embeddings)
    
    print(f"Regression output shape: {regression_output.shape}")
    print(f"Classification output shape: {classification_output.shape}")
    
    # Simulate targets
    regression_targets = torch.randn(batch_size, len(prediction_heads.regression_tasks))
    classification_targets = torch.randint(0, 2, (batch_size, len(prediction_heads.classification_tasks))).float()
    
    # Compute loss
    losses = prediction_heads.compute_loss(
        regression_output, classification_output,
        regression_targets, classification_targets
    )
    
    print(f"\nLosses:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Compute metrics
    metrics = prediction_heads.compute_metrics(
        regression_output, classification_output,
        regression_targets, classification_targets
    )
    
    print(f"\nSample metrics:")
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        if i < 5:  # Show first 5 metrics
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Compare with MolE paper results
    comparison = prediction_heads.compare_with_mole_paper(metrics)
    
    print(f"\nComparison with MolE paper (sample):")
    for i, (task, comp) in enumerate(comparison.items()):
        if i < 3:  # Show first 3 comparisons
            print(f"  {task}: Current={comp['current_result']:.4f}, MolE={comp['mole_result']:.4f}, Diff={comp['difference']:.4f}")
    
    print(f"\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()