"""
ChemBL Model for MolE Pretraining

This model extends the cross-environment MLM model to include ChemBL binary
classification tasks instead of molecular property regression.

- MLM Task: Cross-environment masked language modeling
- Classification Tasks: ChemBL assay binary classification (1=active, -1=inactive, 0=ignore)
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Config

from mole.models.crossenv_mlm import CrossEnvMLMModel


class ChemBLModel(CrossEnvMLMModel):
    """
    ChemBL model that combines cross-environment MLM with multi-target binary classification.
    
    This model extends the base CrossEnvMLMModel by adding binary classification heads
    for ChemBL assay predictions instead of regression heads for molecular properties.
    """

    def __init__(
        self,
        deberta_config: DebertaV2Config,
        input_vocab_size: int,
        target_vocab_size: int,
        num_targets: int,
        hidden_dropout_prob: float = 0.1,
        classifier_dropout_prob: float = 0.1,
        **kwargs
    ):
        """
        Initialize ChemBL model.
        
        Args:
            deberta_config: DeBERTa configuration
            input_vocab_size: Size of input vocabulary
            target_vocab_size: Size of target vocabulary  
            num_targets: Number of ChemBL classification targets
            hidden_dropout_prob: Dropout for hidden layers
            classifier_dropout_prob: Dropout for classification heads
        """
        # Initialize base model (handles MLM)
        # Convert DebertaV2Config to dict if needed
        if hasattr(deberta_config, 'to_dict'):
            config_dict = deberta_config.to_dict()
        else:
            config_dict = deberta_config
            
        super().__init__(
            deberta_config=config_dict,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            **kwargs
        )
        
        self.num_targets = num_targets
        self.classifier_dropout_prob = classifier_dropout_prob
        
        # Remove any regression components from parent if they exist
        if hasattr(self, 'regression_head'):
            delattr(self, 'regression_head')
            
        # Classification components
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
        
        # Multi-target binary classification head
        # Each target gets its own binary classifier (3 classes: active=1, inactive=-1, padded=0)
        # We'll use a single head that outputs [batch, num_targets, 3] and handle masking in loss
        hidden_size = config_dict.get('hidden_size', deberta_config.get('hidden_size', 768))
        self.classification_head = nn.Linear(
            hidden_size, 
            num_targets * 2  # 2 classes per target (active/inactive, ignore unmeasured)
        )
        
        # Initialize classification head with smaller weights
        nn.init.normal_(self.classification_head.weight, std=0.02)
        nn.init.constant_(self.classification_head.bias, 0.0)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        chembl_targets: Optional[torch.Tensor] = None,
        chembl_mask: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ChemBL model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            input_mask: Input attention mask [batch, seq_len]
            labels: MLM target labels [batch, seq_len]
            chembl_targets: ChemBL classification targets [batch, num_targets]
            chembl_mask: Mask for measured ChemBL targets [batch, num_targets]
            relative_pos: Relative position matrix [batch, seq_len, seq_len]
            
        Returns:
            Dictionary containing losses and predictions
        """
        # Get base MLM outputs
        outputs = super().forward(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            relative_pos=relative_pos,
            **kwargs
        )
        
        # Expose MLM loss with the correct key name
        if "loss" in outputs:
            outputs["mlm_loss"] = outputs["loss"]
        
        # Get CLS representation for classification
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None:
            cls_representation = hidden_states[:, 0, :]  # [batch, hidden_size]
            cls_representation = self.classifier_dropout(cls_representation)
            
            # Multi-target binary classification
            classification_logits = self.classification_head(cls_representation)  # [batch, num_targets * 2]
            
            # Reshape to [batch, num_targets, 2] for binary classification per target
            batch_size = classification_logits.size(0)
            classification_logits = classification_logits.view(batch_size, self.num_targets, 2)
            
            outputs["classification_logits"] = classification_logits
            
            # Calculate classification loss if targets provided
            if chembl_targets is not None and chembl_mask is not None:
                classification_loss = self.compute_classification_loss(
                    logits=classification_logits,
                    targets=chembl_targets,
                    mask=chembl_mask
                )
                outputs["classification_loss"] = classification_loss
                
        return outputs
        
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked binary classification loss for ChemBL targets.
        
        Args:
            logits: Classification logits [batch, num_targets, 2]
            targets: Target labels [batch, num_targets] (1=active, -1=inactive, 0=unmeasured)
            mask: Mask for measured targets [batch, num_targets] (True=measured, False=unmeasured)
            
        Returns:
            Masked binary cross-entropy loss
        """
        # Convert ChemBL labels (1=active, -1=inactive) to binary (1=active, 0=inactive)
        # Only consider masked (measured) targets
        binary_targets = (targets == 1).long()  # 1 for active, 0 for inactive
        
        # Flatten for loss computation
        flat_logits = logits.view(-1, 2)  # [batch * num_targets, 2]
        flat_targets = binary_targets.view(-1)  # [batch * num_targets]
        flat_mask = mask.view(-1)  # [batch * num_targets]
        
        # Only compute loss for measured targets
        if flat_mask.sum() == 0:
            # No measured targets in this batch
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        masked_logits = flat_logits[flat_mask]  # [num_measured, 2]
        masked_targets = flat_targets[flat_mask]  # [num_measured]
        
        # Binary cross-entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        
        return loss
        
    def predict_chembl(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        relative_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict ChemBL activities for given molecules.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            input_mask: Input attention mask [batch, seq_len]
            relative_pos: Relative position matrix [batch, seq_len, seq_len]
            
        Returns:
            Predicted probabilities [batch, num_targets] (probability of being active)
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                input_mask=input_mask,
                relative_pos=relative_pos
            )
            
            classification_logits = outputs.get("classification_logits")
            if classification_logits is not None:
                # Apply softmax to get probabilities, take probability of active class
                probs = F.softmax(classification_logits, dim=-1)  # [batch, num_targets, 2]
                active_probs = probs[:, :, 1]  # [batch, num_targets] - probability of active
                return active_probs
            else:
                # Return zeros if no classification logits
                batch_size = input_ids.size(0)
                return torch.zeros(batch_size, self.num_targets, device=input_ids.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        base_info = super().get_model_info() if hasattr(super(), 'get_model_info') else {}
        
        chembl_info = {
            'model_type': 'ChemBL',
            'num_targets': self.num_targets,
            'classifier_dropout': self.classifier_dropout_prob,
            'has_classification_head': True,
            'has_regression_head': False,
        }
        
        return {**base_info, **chembl_info} 