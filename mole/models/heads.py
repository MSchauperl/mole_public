import torch
from torch import nn
from typing import Optional

# Imports likely needed by TaskPredictionHead
from DeBERTa.deberta.ops import ACT2FN
from DeBERTa.deberta.ops import StableDropout


# Removed from mole/nn/layers.py
class TaskPredictionHead(nn.Module):
    def __init__(
        self,
        config,
        loss_fn: Optional[torch.nn.modules.loss._Loss],
        num_tasks: int = 1,
        num_classes: int = 1,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.dense = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
        # Changed name from classifier to out_proj to match original removal diff
        self.out_proj = torch.nn.Linear(
            self.config.hidden_size, num_tasks * num_classes
        )
        dropout = self.config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = StableDropout(dropout)
        self.loss_fn = loss_fn

    def forward(self, context_token, labels=None):
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.hidden_act](
            pooled_output
        )  # Assumes hidden_act in config
        pooled_output = self.dropout(pooled_output)
        logits = self.out_proj(pooled_output)

        loss = torch.tensor(0).to(logits)
        if labels is not None and self.loss_fn is not None:
            if self.num_classes > 1:
                # Classification
                logits_reshaped = logits.view(-1, self.num_classes)
                labels_valid = labels.view(-1)[
                    ~labels.isnan()
                ]  # Filter NaNs if necessary
                if labels_valid.numel() > 0:
                    loss = self.loss_fn(logits_reshaped, labels_valid.long())
                else:
                    loss = torch.tensor(
                        0.0, device=logits.device, requires_grad=True
                    )  # Handle no valid labels
            else:
                # Regression or Binary Classification
                logits_valid = logits.view(-1)[~labels.isnan()]
                labels_valid = labels.view(-1)[~labels.isnan()].to(logits_valid)
                if labels_valid.numel() > 0:
                    loss = self.loss_fn(logits_valid, labels_valid)
                else:
                    loss = torch.tensor(
                        0.0, device=logits.device, requires_grad=True
                    )  # Handle no valid labels

        return logits, loss, labels  # Return original labels for consistency


__all__ = ["TaskPredictionHead"]
