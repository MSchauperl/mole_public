"""
Integrated Table 1 MolE Model

This module combines the MolE encoder with Table 1 prediction heads
for ADMET property prediction on the datasets mentioned in the MolE paper.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from DeBERTa.deberta.config import ModelConfig
from mole.models.embeddings import AtomEnvEmbeddings
from mole.table1_prediction_heads import Table1PredictionHeads
from mole.dataset_loader import ADMETDataLoader
from configs.table1_task_config import get_all_tasks, get_task_type_mapping

logger = logging.getLogger(__name__)


class Table1MolEModel(nn.Module):
    """
    Integrated MolE model for Table 1 ADMET tasks.
    
    This model combines:
    1. MolE encoder (AtomEnvEmbeddings) for molecular representation
    2. Table 1 prediction heads for regression and classification tasks
    3. Proper handling of missing values via masks
    """
    
    def __init__(
        self,
        deberta_config: Dict[str, Any],
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_path: Optional[str] = None,
        regression_loss_weight: float = 0.004,
        classification_loss_weight: float = 1.0,
        selected_tasks: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the Table 1 MolE model.
        
        Args:
            deberta_config: DeBERTa configuration for MolE encoder
            dropout: Dropout probability for prediction heads
            freeze_encoder: Whether to freeze the MolE encoder weights
            pretrained_path: Path to pretrained MolE weights
            regression_loss_weight: Weight for regression loss (default: 0.004)
            classification_loss_weight: Weight for classification loss (default: 1.0)
            selected_tasks: List of task names to create prediction heads for
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Convert dictionary to config object
        config = ModelConfig.from_dict(deberta_config)
        
        # Initialize MolE encoder
        self.encoder = AtomEnvEmbeddings(
            config=config,
            pre_trained=pretrained_path
        )
        
        # Get encoder output dimension
        self.hidden_dim = deberta_config.get('hidden_size', 768)
        
        # Initialize Table 1 prediction heads with loss weights
        self.prediction_heads = Table1PredictionHeads(
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            regression_loss_weight=regression_loss_weight,
            classification_loss_weight=classification_loss_weight,
            selected_tasks=selected_tasks
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing MolE encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized Table1MolEModel with hidden_dim={self.hidden_dim}")
    
    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            input_mask: Input mask [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            relative_pos: Relative position matrix
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing encoder outputs and predictions
        """
        # Forward pass through MolE encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_mask=input_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            relative_pos=relative_pos,
            output_all_encoded_layers=True
        )
        
        # Get the last layer hidden states
        hidden_states = encoder_outputs['hidden_states']
        last_hidden_state = hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Use CLS token representation for prediction
        cls_representation = last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Forward pass through prediction heads
        regression_output, classification_output = self.prediction_heads(cls_representation)
        
        return {
            'encoder_outputs': encoder_outputs,
            'cls_representation': cls_representation,
            'regression_output': regression_output,
            'classification_output': classification_output,
            'hidden_states': hidden_states
        }
    
    def compute_loss_and_metrics(
        self,
        regression_output: Optional[torch.Tensor],
        classification_output: Optional[torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for all tasks with proper masking.
        
        Args:
            regression_output: Model predictions for regression tasks (can be None)
            classification_output: Model predictions for classification tasks (can be None)
            targets: Dictionary of target values for each task
            masks: Dictionary of masks for each task
            device: Device to run computations on
            
        Returns:
            Dictionary containing individual and combined losses
        """
        # Prepare targets and masks for prediction heads
        regression_targets = []
        regression_masks = []
        classification_targets = []
        classification_masks = []
        
        task_type_mapping = get_task_type_mapping()
        
        # Only get tasks that have corresponding heads
        regression_tasks = self.prediction_heads.regression_tasks if self.prediction_heads.regression_head is not None else []
        classification_tasks = self.prediction_heads.classification_tasks if self.prediction_heads.classification_head is not None else []
        
        # Prepare regression data
        for task in regression_tasks:
            if task in targets and task in masks:
                regression_targets.append(targets[task].to(device))
                regression_masks.append(masks[task].to(device))
        
        # Prepare classification data
        for task in classification_tasks:
            if task in targets and task in masks:
                # Convert classification targets to binary format
                task_targets = targets[task].to(device)
                task_masks = masks[task].to(device)
                
                # Handle -1 values (missing) by setting mask to False
                valid_targets = (task_targets != -1).float()
                task_masks = task_masks & valid_targets.bool()
                
                # Convert to binary (0/1) for BCE loss
                binary_targets = (task_targets == 1).float()
                
                classification_targets.append(binary_targets)
                classification_masks.append(task_masks)
        
        # Stack tensors if we have any tasks
        if regression_targets and regression_output is not None:
            regression_targets = torch.stack(regression_targets, dim=1)  # [batch_size, num_regression_tasks]
            regression_masks = torch.stack(regression_masks, dim=1)  # [batch_size, num_regression_tasks]
        else:
            regression_targets = None
            regression_masks = None
        
        if classification_targets and classification_output is not None:
            classification_targets = torch.stack(classification_targets, dim=1)  # [batch_size, num_classification_tasks]
            classification_masks = torch.stack(classification_masks, dim=1)  # [batch_size, num_classification_tasks]
        else:
            classification_targets = None
            classification_masks = None
        
        # Compute losses only for available heads
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # Regression loss
        if regression_output is not None and regression_targets is not None:
            # Ensure output dimensions match target dimensions
            if regression_targets.shape[1] != regression_output.shape[1]:
                logger.warning(f"Regression output dimension ({regression_output.shape[1]}) doesn't match targets ({regression_targets.shape[1]})")
                # Pad or truncate as needed
                if regression_targets.shape[1] < regression_output.shape[1]:
                    # Pad targets with zeros
                    padding = torch.zeros(regression_output.shape[0], regression_output.shape[1] - regression_targets.shape[1], device=device)
                    regression_targets = torch.cat([regression_targets, padding], dim=1)
                    padding_mask = torch.zeros(regression_output.shape[0], regression_output.shape[1] - regression_masks.shape[1], dtype=torch.bool, device=device)
                    regression_masks = torch.cat([regression_masks, padding_mask], dim=1)
                else:
                    # Truncate targets
                    regression_targets = regression_targets[:, :regression_output.shape[1]]
                    regression_masks = regression_masks[:, :regression_output.shape[1]]
            
            # Compute regression loss
            regression_loss = nn.functional.mse_loss(
                regression_output * regression_masks.float(),
                regression_targets * regression_masks.float(),
                reduction='sum'
            )
            if regression_masks.sum() > 0:
                regression_loss = regression_loss / regression_masks.sum()
            
            loss_dict['regression_loss'] = regression_loss
            total_loss += self.prediction_heads.regression_loss_weight * regression_loss
        
        # Classification loss  
        if classification_output is not None and classification_targets is not None:
            # Ensure output dimensions match target dimensions
            if classification_targets.shape[1] != classification_output.shape[1]:
                logger.warning(f"Classification output dimension ({classification_output.shape[1]}) doesn't match targets ({classification_targets.shape[1]})")
                # Pad or truncate as needed
                if classification_targets.shape[1] < classification_output.shape[1]:
                    # Pad targets with zeros
                    padding = torch.zeros(classification_output.shape[0], classification_output.shape[1] - classification_targets.shape[1], device=device)
                    classification_targets = torch.cat([classification_targets, padding], dim=1)
                    padding_mask = torch.zeros(classification_output.shape[0], classification_output.shape[1] - classification_masks.shape[1], dtype=torch.bool, device=device)
                    classification_masks = torch.cat([classification_masks, padding_mask], dim=1)
                else:
                    # Truncate targets
                    classification_targets = classification_targets[:, :classification_output.shape[1]]
                    classification_masks = classification_masks[:, :classification_output.shape[1]]
            
            # Compute classification loss
            classification_loss = nn.functional.binary_cross_entropy_with_logits(
                classification_output * classification_masks.float(),
                classification_targets * classification_masks.float(),
                reduction='sum'
            )
            if classification_masks.sum() > 0:
                classification_loss = classification_loss / classification_masks.sum()
            
            loss_dict['classification_loss'] = classification_loss
            total_loss += self.prediction_heads.classification_loss_weight * classification_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def compute_metrics(
        self,
        regression_output: torch.Tensor,
        classification_output: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """
        Compute metrics for all tasks with proper masking.
        
        Args:
            regression_output: Model predictions for regression tasks
            classification_output: Model predictions for classification tasks
            targets: Dictionary of target values for each task
            masks: Dictionary of masks for each task
            device: Device to run computations on
            
        Returns:
            Dictionary containing metrics for each task
        """
        # Prepare targets and masks for prediction heads
        regression_targets = []
        regression_masks = []
        classification_targets = []
        classification_masks = []
        
        task_type_mapping = get_task_type_mapping()
        all_tasks = get_all_tasks()
        
        # Separate regression and classification tasks
        regression_tasks = [task for task in all_tasks if task_type_mapping[task] == 'regression']
        classification_tasks = [task for task in all_tasks if task_type_mapping[task] == 'classification']
        
        # Prepare regression data
        for task in regression_tasks:
            if task in targets and task in masks:
                regression_targets.append(targets[task].to(device))
                regression_masks.append(masks[task].to(device))
        
        # Prepare classification data
        for task in classification_tasks:
            if task in targets and task in masks:
                # Convert classification targets to binary format
                task_targets = targets[task].to(device)
                task_masks = masks[task].to(device)
                
                # Handle -1 values (missing) by setting mask to False
                valid_targets = (task_targets != -1).float()
                task_masks = task_masks & valid_targets.bool()
                
                # Convert to binary (0/1) for BCE loss
                binary_targets = (task_targets == 1).float()
                
                classification_targets.append(binary_targets)
                classification_masks.append(task_masks)
        
        # Stack tensors if we have any tasks
        if regression_targets:
            regression_targets = torch.stack(regression_targets, dim=1)
            regression_masks = torch.stack(regression_masks, dim=1)
        else:
            regression_targets = torch.empty(regression_output.shape[0], 0, device=device)
            regression_masks = torch.empty(regression_output.shape[0], 0, dtype=torch.bool, device=device)
        
        if classification_targets:
            classification_targets = torch.stack(classification_targets, dim=1)
            classification_masks = torch.stack(classification_masks, dim=1)
        else:
            classification_targets = torch.empty(classification_output.shape[0], 0, device=device)
            classification_masks = torch.empty(classification_output.shape[0], 0, dtype=torch.bool, device=device)
        
        # Ensure output dimensions match target dimensions
        if regression_targets.shape[1] != regression_output.shape[1]:
            logger.warning(f"Regression output dimension ({regression_output.shape[1]}) doesn't match targets ({regression_targets.shape[1]})")
            # Pad or truncate as needed
            if regression_targets.shape[1] < regression_output.shape[1]:
                # Pad targets with zeros
                padding = torch.zeros(regression_output.shape[0], regression_output.shape[1] - regression_targets.shape[1], device=device)
                regression_targets = torch.cat([regression_targets, padding], dim=1)
                padding_mask = torch.zeros(regression_output.shape[0], regression_output.shape[1] - regression_masks.shape[1], dtype=torch.bool, device=device)
                regression_masks = torch.cat([regression_masks, padding_mask], dim=1)
            else:
                # Truncate targets
                regression_targets = regression_targets[:, :regression_output.shape[1]]
                regression_masks = regression_masks[:, :regression_output.shape[1]]
        
        if classification_targets.shape[1] != classification_output.shape[1]:
            logger.warning(f"Classification output dimension ({classification_output.shape[1]}) doesn't match targets ({classification_targets.shape[1]})")
            # Pad or truncate as needed
            if classification_targets.shape[1] < classification_output.shape[1]:
                # Pad targets with zeros
                padding = torch.zeros(classification_output.shape[0], classification_output.shape[1] - classification_targets.shape[1], device=device)
                classification_targets = torch.cat([classification_targets, padding], dim=1)
                padding_mask = torch.zeros(classification_output.shape[0], classification_output.shape[1] - classification_masks.shape[1], dtype=torch.bool, device=device)
                classification_masks = torch.cat([classification_masks, padding_mask], dim=1)
            else:
                # Truncate targets
                classification_targets = classification_targets[:, :classification_output.shape[1]]
                classification_masks = classification_masks[:, :classification_output.shape[1]]
        
        # Compute metrics using prediction heads
        metrics = self.prediction_heads.compute_metrics(
            regression_output=regression_output,
            classification_output=classification_output,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            regression_mask=regression_masks,
            classification_mask=classification_masks
        )
        
        return metrics


def create_table1_mole_model(
    deberta_config: Optional[Dict[str, Any]] = None,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    pretrained_path: Optional[str] = None,
    regression_loss_weight: float = 0.004,
    classification_loss_weight: float = 1.0,
    selected_tasks: Optional[List[str]] = None,
    **kwargs
) -> Table1MolEModel:
    """
    Convenience function to create a Table 1 MolE model.
    
    Args:
        deberta_config: DeBERTa configuration (defaults to standard MolE config)
        dropout: Dropout probability for prediction heads
        freeze_encoder: Whether to freeze the MolE encoder weights
        pretrained_path: Path to pretrained MolE weights
        regression_loss_weight: Weight for regression loss (default: 0.004)
        classification_loss_weight: Weight for classification loss (default: 1.0)
        selected_tasks: List of task names to create prediction heads for
        **kwargs: Additional arguments
        
    Returns:
        Configured Table1MolEModel instance
    """
    # Default DeBERTa configuration for MolE
    if deberta_config is None:
        deberta_config = {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'intermediate_size': 3072,
            'vocab_size': 8192,  # Adjust based on your vocabulary
            'max_position_embeddings': 512,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'type_vocab_size': 0,
            'pad_token_id': 0,
            'cls_token_id': 1,
            'mask_token_id': 2,
            'sep_token_id': 3,
            'unk_token_id': 4,
        }
    
    return Table1MolEModel(
        deberta_config=deberta_config,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        pretrained_path=pretrained_path,
        regression_loss_weight=regression_loss_weight,
        classification_loss_weight=classification_loss_weight,
        selected_tasks=selected_tasks,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = create_table1_mole_model()
    
    # Create mock data
    batch_size = 4
    seq_len = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    input_mask = torch.ones(batch_size, seq_len)
    
    # Mock targets and masks
    targets = {}
    masks = {}
    
    # Add some regression tasks
    regression_tasks = ['Caco2', 'Lipophilicity', 'Solubility']
    for task in regression_tasks:
        targets[task] = torch.randn(batch_size)
        masks[task] = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
    
    # Add some classification tasks
    classification_tasks = ['HIA', 'Pgp', 'BBB']
    for task in classification_tasks:
        targets[task] = torch.randint(-1, 2, (batch_size,))  # -1 for missing, 0/1 for classes
        masks[task] = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
    
    # Move to device
    model = model.to(device)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids, input_mask=input_mask)
    
    print("Model outputs:")
    print(f"  Regression output shape: {outputs['regression_output'].shape}")
    print(f"  Classification output shape: {outputs['classification_output'].shape}")
    print(f"  CLS representation shape: {outputs['cls_representation'].shape}")
    
    # Compute loss
    loss_dict = model.compute_loss_and_metrics(
        regression_output=outputs['regression_output'],
        classification_output=outputs['classification_output'],
        targets=targets,
        masks=masks,
        device=device
    )
    
    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Compute metrics
    metrics = model.compute_metrics(
        regression_output=outputs['regression_output'],
        classification_output=outputs['classification_output'],
        targets=targets,
        masks=masks,
        device=device
    )
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTable 1 MolE model test completed successfully!") 