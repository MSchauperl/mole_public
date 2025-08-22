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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Paper: "Focal Loss for Dense Object Detection" by Lin et al. (2017)
    https://arxiv.org/abs/1708.02002
    
    The focal loss is defined as:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - α_t is a weighting factor for class t
    - γ is the focusing parameter that downweights easy examples
    - p_t is the predicted probability for the true class
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0, higher values focus more on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits [batch_size * num_targets, num_classes]
            targets: True labels [batch_size * num_targets]
            mask: Valid sample mask [batch_size * num_targets] (optional)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p = F.softmax(logits, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1.0)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply mask if provided
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


class EnhancedClassificationHead(nn.Module):
    """
    Enhanced multi-layer classification head with residual connections and multi-scale features.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_targets: int,
        dropout_prob: float = 0.1,
        use_multiscale: bool = True,
    ):
        """
        Initialize enhanced classification head.
        
        Args:
            hidden_size: Size of input hidden states
            num_targets: Number of classification targets
            dropout_prob: Dropout probability
            use_multiscale: Whether to use multi-scale feature fusion
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_targets = num_targets
        self.use_multiscale = use_multiscale
        
        # Multi-scale feature fusion increases input size
        input_size = hidden_size * 4 if use_multiscale else hidden_size
        
        # Multi-layer classification head with residual connections
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.norm3 = nn.LayerNorm(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.output = nn.Linear(hidden_size // 4, num_targets * 2)
        
        # Residual connection for layer 1 -> layer 3 (dimension matching)
        self.residual_proj = nn.Linear(hidden_size, hidden_size // 4)
        
        # Attention weights for multi-scale pooling
        if use_multiscale:
            self.attention_proj = nn.Linear(hidden_size, 1, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def extract_multiscale_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from hidden states.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            
        Returns:
            Combined features [batch, hidden_size * 4]
        """
        # CLS token representation
        cls_representation = hidden_states[:, 0, :]  # [batch, hidden_size]
        
        # Mean pooling (excluding CLS token)
        mean_pooled = hidden_states[:, 1:, :].mean(dim=1)  # [batch, hidden_size]
        
        # Max pooling (excluding CLS token)
        max_pooled = hidden_states[:, 1:, :].max(dim=1)[0]  # [batch, hidden_size]
        
        # Attention-weighted pooling
        # Compute attention weights using CLS as query
        attention_scores = self.attention_proj(hidden_states[:, 1:, :])  # [batch, seq_len-1, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len-1, 1]
        attention_pooled = torch.sum(
            hidden_states[:, 1:, :] * attention_weights, dim=1
        )  # [batch, hidden_size]
        
        # Concatenate all representations
        combined_features = torch.cat([
            cls_representation, mean_pooled, max_pooled, attention_pooled
        ], dim=-1)  # [batch, hidden_size * 4]
        
        return combined_features
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced classification head.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            
        Returns:
            Classification logits [batch, num_targets, 2]
        """
        # Extract features
        if self.use_multiscale:
            features = self.extract_multiscale_features(hidden_states)
        else:
            features = hidden_states[:, 0, :]  # Just CLS token
        
        # Layer 1
        x1 = self.layer1(features)
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        x1 = self.dropout1(x1)
        
        # Layer 2
        x2 = self.layer2(x1)
        x2 = self.norm2(x2)
        x2 = F.gelu(x2)
        x2 = self.dropout2(x2)
        
        # Layer 3 with residual connection
        x3 = self.layer3(x2)
        
        # Residual connection from layer 1
        residual = self.residual_proj(x1)
        x3 = x3 + residual
        
        x3 = self.norm3(x3)
        x3 = F.gelu(x3)
        x3 = self.dropout3(x3)
        
        # Output layer
        logits = self.output(x3)  # [batch, num_targets * 2]
        
        # Reshape to [batch, num_targets, 2]
        batch_size = logits.size(0)
        logits = logits.view(batch_size, self.num_targets, 2)
        
        return logits


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
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_enhanced_classifier: bool = False,
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
            use_focal_loss: Whether to use focal loss instead of cross-entropy
            focal_alpha: Alpha parameter for focal loss (class weighting)
            focal_gamma: Gamma parameter for focal loss (focusing parameter)
            use_enhanced_classifier: Whether to use enhanced multi-layer classification head
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
        self.use_focal_loss = use_focal_loss
        self.use_enhanced_classifier = use_enhanced_classifier
        
        # Remove any regression components from parent if they exist
        if hasattr(self, 'regression_head'):
            delattr(self, 'regression_head')
            
        # Get hidden size
        hidden_size = config_dict.get('hidden_size', deberta_config.get('hidden_size', 768))
        
        # Initialize focal loss if requested
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        else:
            self.focal_loss = None
        
        # Classification head - enhanced or simple
        if use_enhanced_classifier:
            self.classification_head = EnhancedClassificationHead(
                hidden_size=hidden_size,
                num_targets=num_targets,
                dropout_prob=classifier_dropout_prob,
                use_multiscale=True,
            )
            self.classifier_dropout = None  # Dropout handled by enhanced head
        else:
            # Simple single linear layer (original implementation)
            self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
            self.classification_head = nn.Linear(
                hidden_size, 
                num_targets * 2  # 2 classes per target (active/inactive, ignore unmeasured)
            )
            # Initialize simple classification head with smaller weights
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
        
        # Get hidden states for classification
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None:
            if self.use_enhanced_classifier:
                # Enhanced classifier handles feature extraction internally
                classification_logits = self.classification_head(hidden_states)  # [batch, num_targets, 2]
            else:
                # Simple classifier uses CLS token
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
            Masked binary cross-entropy or focal loss
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
        
        # Use focal loss if configured, otherwise use cross-entropy
        if self.use_focal_loss and self.focal_loss is not None:
            loss = self.focal_loss(flat_logits, flat_targets, flat_mask)
        else:
            masked_logits = flat_logits[flat_mask]  # [num_measured, 2]
            masked_targets = flat_targets[flat_mask]  # [num_measured]
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
            'use_focal_loss': self.use_focal_loss,
            'focal_alpha': getattr(self.focal_loss, 'alpha', None) if self.focal_loss else None,
            'focal_gamma': getattr(self.focal_loss, 'gamma', None) if self.focal_loss else None,
            'use_enhanced_classifier': self.use_enhanced_classifier,
            'classifier_type': 'Enhanced Multi-layer' if self.use_enhanced_classifier else 'Simple Linear',
        }
        
        return {**base_info, **chembl_info} 