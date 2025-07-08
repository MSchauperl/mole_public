"""
Multi-Task Model for MolE Pretraining

This file defines the components for a multi-task model that combines
cross-environment MLM with molecular property prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from mole.models.crossenv_mlm import CrossEnvMLMModel, CrossEnvMLM


class RegressionHead(nn.Module):
    """Prediction head for molecular property regression tasks."""

    def __init__(self, hidden_size: int, output_size: int = 2):
        """
        Initialize the regression head.

        Args:
            hidden_size: The dimension of the input hidden state (e.g., from CLS token).
            output_size: The number of continuous values to predict (default: 2 for ClogP and MW).
        """
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the regression head.

        Args:
            hidden_state: The pooled hidden state from the encoder (e.g., CLS token).
                          Shape: [batch_size, hidden_size]

        Returns:
            A tensor of predicted property values. Shape: [batch_size, output_size]
        """
        x = self.dense1(hidden_state)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class MultiTaskModel(CrossEnvMLMModel):
    """Multi-task model combining MLM and property prediction."""

    def __init__(
        self,
        deberta_config: Dict[str, Any],
        input_vocab_size: int,
        target_vocab_size: int,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the multi-task model.

        Args:
            deberta_config: DeBERTa configuration dictionary.
            input_vocab_size: Size of the input vocabulary.
            target_vocab_size: Size of the target vocabulary for MLM.
            dropout: Dropout probability for prediction heads.
            label_smoothing: Label smoothing for MLM loss.
        """
        super().__init__(
            deberta_config,
            input_vocab_size,
            target_vocab_size,
            dropout,
            label_smoothing,
        )

        # The MLM head is already created in the parent class (self.prediction_head)
        # Create the regression head for the new tasks
        self.regression_head = RegressionHead(
            hidden_size=self.config.hidden_size, output_size=2
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multi-task model.

        Args:
            input_ids: Input token IDs.
            input_mask: Attention mask.
            labels: Target token IDs for MLM.

        Returns:
            A dictionary containing MLM logits and regression predictions.
        """
        # Get the output from the base encoder
        base_outputs = super().forward(
            input_ids=input_ids, input_mask=input_mask, labels=labels, **kwargs
        )

        # The base_outputs dictionary already contains the MLM logits and loss
        outputs = {"mlm_logits": base_outputs["logits"]}
        if "loss" in base_outputs:
            outputs["mlm_loss"] = base_outputs["loss"]

        # Use the hidden state of the [CLS] token for regression
        # The CLS token is at position 0
        cls_hidden_state = base_outputs["hidden_states"][:, 0, :]

        # Get regression predictions
        regression_preds = self.regression_head(cls_hidden_state)
        outputs["regression_preds"] = regression_preds

        return outputs


class MultiTaskLightningModule(CrossEnvMLM):
    """PyTorch Lightning wrapper for the multi-task model."""

    def __init__(
        self,
        model: MultiTaskModel,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        log_predictions: bool = False,
        mlm_loss_weight: float = 1.0,
        regression_loss_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the Lightning module for multi-task training.

        Args:
            model: The MultiTaskModel to be trained.
            optimizer_cfg: Configuration for the optimizer.
            scheduler_cfg: Configuration for the learning rate scheduler.
            log_predictions: Whether to log MLM prediction examples.
            mlm_loss_weight: The weight for the MLM loss component.
            regression_loss_weight: The weight for the regression loss component.
        """
        # Note: The parent class `CrossEnvMLM` is a `pl.LightningModule`
        super().__init__(
            model=model,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            log_predictions=log_predictions,
            **kwargs,
        )

        self.mlm_loss_weight = mlm_loss_weight
        self.regression_loss_weight = regression_loss_weight
        self.regression_loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        from torch_geometric.utils import to_dense_batch

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Forward pass through the multi-task model
        outputs = self.model(
            input_ids=input_ids, input_mask=input_mask, labels=labels
        )

        # Calculate MLM loss (already done in the model's forward pass)
        mlm_loss = outputs.get("mlm_loss", 0.0)

        # Calculate regression loss
        regression_targets = torch.stack([batch.clogp, batch.log_mw], dim=1)
        regression_preds = outputs["regression_preds"]
        regression_loss = self.regression_loss_fn(
            regression_preds, regression_targets
        )

        # Combine losses
        total_loss = (self.mlm_loss_weight * mlm_loss) + (
            self.regression_loss_weight * regression_loss
        )

        # Log metrics
        self.log_dict(
            {
                "train/total_loss": total_loss,
                "train/mlm_loss": mlm_loss,
                "train/regression_loss": regression_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        from torch_geometric.utils import to_dense_batch

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids, input_mask=input_mask, labels=labels
        )

        # Calculate losses
        mlm_loss = outputs.get("mlm_loss", 0.0)
        regression_targets = torch.stack([batch.clogp, batch.mw], dim=1)
        regression_preds = outputs["regression_preds"]
        regression_loss = self.regression_loss_fn(
            regression_preds, regression_targets
        )
        total_loss = (self.mlm_loss_weight * mlm_loss) + (
            self.regression_loss_weight * regression_loss
        )

        # Log metrics
        self.log_dict(
            {
                "val/total_loss": total_loss,
                "val/mlm_loss": mlm_loss,
                "val/regression_loss": regression_loss,
                "val/accuracy": outputs.get("accuracy", 0.0),
                "val/perplexity": outputs.get("perplexity", 1.0),
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"val_loss": total_loss} 