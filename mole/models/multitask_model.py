"""
Multi-Task Model for MolE Pretraining

This file defines the components for a multi-task model that combines
cross-environment MLM with molecular property prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from mole.models.crossenv_mlm import CrossEnvMLMModel, CrossEnvMLM
from mole.metrics import get_mlm_metrics, get_regression_metrics


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
        if "accuracy" in base_outputs:
            outputs["accuracy"] = base_outputs["accuracy"]
        if "perplexity" in base_outputs:
            outputs["perplexity"] = base_outputs["perplexity"]

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

        # Initialize metrics
        self.train_mlm_metrics = get_mlm_metrics(
            prefix="train/mlm", num_classes=model.target_vocab_size
        )
        self.val_mlm_metrics = get_mlm_metrics(
            prefix="val/mlm", num_classes=model.target_vocab_size
        )

        self.train_clogp_metrics = get_regression_metrics(prefix="train/clogp")
        self.val_clogp_metrics = get_regression_metrics(prefix="val/clogp")

        self.train_mw_metrics = get_regression_metrics(prefix="train/mw")
        self.val_mw_metrics = get_regression_metrics(prefix="val/mw")

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        from torch_geometric.utils import to_dense_batch

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Forward pass through the multi-task model
        outputs = self.model(input_ids=input_ids, input_mask=input_mask, labels=labels)

        # Calculate MLM loss (already done in the model's forward pass)
        mlm_loss = outputs.get("mlm_loss", 0.0)
        mlm_logits = outputs.get("mlm_logits")

        # Calculate regression loss
        regression_targets = torch.stack([batch.clogp, batch.log_mw], dim=1)
        regression_preds = outputs["regression_preds"]

        clogp_loss = self.regression_loss_fn(
            regression_preds[:, 0], regression_targets[:, 0]
        )
        mw_loss = self.regression_loss_fn(
            regression_preds[:, 1], regression_targets[:, 1]
        )
        regression_loss = clogp_loss + mw_loss

        # Combine losses
        total_loss = (self.mlm_loss_weight * mlm_loss) + (
            self.regression_loss_weight * regression_loss
        )

        # Update and log metrics
        # MLM metrics
        if mlm_logits is not None:
            # Accuracy metrics expect 2D inputs, perplexity expects 3D
            self.train_mlm_metrics["train/mlm/perplexity"].update(
                mlm_logits, labels
            )

            # Filter out padding tokens for metrics
            mask = labels != -100
            if mask.sum() > 0:
                flat_logits = mlm_logits.view(-1, mlm_logits.size(-1))
                flat_labels = labels.view(-1)

                self.train_mlm_metrics["train/mlm/accuracy"].update(
                    flat_logits, flat_labels
                )
                self.train_mlm_metrics["train/mlm/accuracy_top5"].update(
                    flat_logits, flat_labels
                )

        # ClogP metrics
        self.train_clogp_metrics["train/clogp/mae"].update(
            regression_preds[:, 0], regression_targets[:, 0]
        )
        self.train_clogp_metrics["train/clogp/rmse"].update(
            regression_preds[:, 0], regression_targets[:, 0]
        )
        self.train_clogp_metrics["train/clogp/r_squared"].update(
            regression_preds[:, 0], regression_targets[:, 0]
        )
        self.train_clogp_metrics["train/clogp/pearson_corr_coef"].update(
            regression_preds[:, 0], regression_targets[:, 0]
        )

        # MW metrics
        self.train_mw_metrics["train/mw/mae"].update(
            regression_preds[:, 1], regression_targets[:, 1]
        )
        self.train_mw_metrics["train/mw/rmse"].update(
            regression_preds[:, 1], regression_targets[:, 1]
        )
        self.train_mw_metrics["train/mw/r_squared"].update(
            regression_preds[:, 1], regression_targets[:, 1]
        )
        self.train_mw_metrics["train/mw/pearson_corr_coef"].update(
            regression_preds[:, 1], regression_targets[:, 1]
        )

        self.log_dict(
            {
                "train/total_loss": total_loss,
                "train/mlm_loss": mlm_loss,
                "train/regression_loss": regression_loss,
                "train/clogp_loss": clogp_loss,
                "train/mw_loss": mw_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss}

    def on_train_epoch_end(self):
        """Log metrics at the end of a training epoch."""
        mlm_metrics = self.train_mlm_metrics.compute()
        clogp_metrics = self.train_clogp_metrics.compute()
        mw_metrics = self.train_mw_metrics.compute()

        all_metrics = {**mlm_metrics, **clogp_metrics, **mw_metrics}
        self.log_dict(all_metrics, on_step=False, on_epoch=True, sync_dist=True)

        self.train_mlm_metrics.reset()
        self.train_clogp_metrics.reset()
        self.train_mw_metrics.reset()

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        from torch_geometric.utils import to_dense_batch

        # Ensure batch is not empty to prevent errors with to_dense_batch
        if batch.num_graphs == 0:
            return {"val_loss": torch.tensor(0.0, device=self.device)}

        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)

        # Forward pass
        outputs = self.model(input_ids=input_ids, input_mask=input_mask, labels=labels)

        # Calculate MLM loss safely
        mlm_loss = outputs.get("mlm_loss", torch.tensor(0.0, device=self.device))
        mlm_logits = outputs.get("mlm_logits")

        # Calculate regression loss safely
        if hasattr(batch, "clogp") and hasattr(batch, "log_mw"):
            regression_targets = torch.stack([batch.clogp, batch.log_mw], dim=1)
            regression_preds = outputs["regression_preds"]
            clogp_loss = self.regression_loss_fn(
                regression_preds[:, 0], regression_targets[:, 0]
            )
            mw_loss = self.regression_loss_fn(
                regression_preds[:, 1], regression_targets[:, 1]
            )
            regression_loss = clogp_loss + mw_loss

            # Update validation metrics
            if mlm_logits is not None:
                # Accuracy metrics expect 2D inputs, perplexity expects 3D
                self.val_mlm_metrics["val/mlm/perplexity"].update(
                    mlm_logits, labels
                )

                # Filter out padding tokens for metrics
                mask = labels != -100
                if mask.sum() > 0:
                    flat_logits = mlm_logits.view(-1, mlm_logits.size(-1))
                    flat_labels = labels.view(-1)

                    self.val_mlm_metrics["val/mlm/accuracy"].update(
                        flat_logits, flat_labels
                    )
                    self.val_mlm_metrics["val/mlm/accuracy_top5"].update(
                        flat_logits, flat_labels
                    )

            # ClogP metrics
            self.val_clogp_metrics["val/clogp/mae"].update(
                regression_preds[:, 0], regression_targets[:, 0]
            )
            self.val_clogp_metrics["val/clogp/rmse"].update(
                regression_preds[:, 0], regression_targets[:, 0]
            )
            self.val_clogp_metrics["val/clogp/r_squared"].update(
                regression_preds[:, 0], regression_targets[:, 0]
            )
            self.val_clogp_metrics["val/clogp/pearson_corr_coef"].update(
                regression_preds[:, 0], regression_targets[:, 0]
            )

            # MW metrics
            self.val_mw_metrics["val/mw/mae"].update(
                regression_preds[:, 1], regression_targets[:, 1]
            )
            self.val_mw_metrics["val/mw/rmse"].update(
                regression_preds[:, 1], regression_targets[:, 1]
            )
            self.val_mw_metrics["val/mw/r_squared"].update(
                regression_preds[:, 1], regression_targets[:, 1]
            )
            self.val_mw_metrics["val/mw/pearson_corr_coef"].update(
                regression_preds[:, 1], regression_targets[:, 1]
            )

        else:
            regression_loss = torch.tensor(0.0, device=self.device)
            clogp_loss = torch.tensor(0.0, device=self.device)
            mw_loss = torch.tensor(0.0, device=self.device)

        # Combine losses
        total_loss = (self.mlm_loss_weight * mlm_loss) + (
            self.regression_loss_weight * regression_loss
        )

        self.log_dict(
            {
                "val/total_loss": total_loss,
                "val/mlm_loss": mlm_loss,
                "val/regression_loss": regression_loss,
                "val/clogp_loss": clogp_loss,
                "val/mw_loss": mw_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"val_loss": total_loss}

    def on_validation_epoch_end(self):
        """Log metrics at the end of a validation epoch."""
        mlm_metrics = self.val_mlm_metrics.compute()
        clogp_metrics = self.val_clogp_metrics.compute()
        mw_metrics = self.val_mw_metrics.compute()

        all_metrics = {**mlm_metrics, **clogp_metrics, **mw_metrics}
        self.log_dict(all_metrics, on_step=False, on_epoch=True, sync_dist=True)

        self.val_mlm_metrics.reset()
        self.val_clogp_metrics.reset()
        self.val_mw_metrics.reset()
