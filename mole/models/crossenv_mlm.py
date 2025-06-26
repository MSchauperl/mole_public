"""
Cross-Environment Masked Language Model

This model implements cross-environment pretraining where:
- Input: Radius 0 structural atom environments
- Output: Radius 1 functional atom environments

The model learns to predict richer functional representations from simpler structural ones.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from DeBERTa.deberta.config import ModelConfig

from mole.models.embeddings import AtomEnvEmbeddings
from mole.models.base import Model, TensorDict, OptimizerConfig
from mole.metrics import MetricsDict
from torchmetrics import MeanMetric


class CrossEnvMLMModel(nn.Module):
    """Cross-Environment Masked Language Model for molecular pretraining"""

    def __init__(
        self,
        deberta_config: Dict[str, Any],
        input_vocab_size: int,
        target_vocab_size: int,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize cross-environment MLM model

        Args:
            deberta_config: DeBERTa configuration dictionary
            input_vocab_size: Size of input vocabulary (radius 0 structural)
            target_vocab_size: Size of target vocabulary (radius 1 functional)
            dropout: Dropout probability for prediction head
            label_smoothing: Label smoothing factor for cross-entropy loss
        """
        super().__init__()

        # Create model configuration
        config = ModelConfig.from_dict(deberta_config)
        config.vocab_size = input_vocab_size
        self.config = config

        # Core molecular encoder (processes radius 0 structural inputs)
        self.encoder = AtomEnvEmbeddings(config)

        # Cross-environment prediction head (predicts radius 1 functional tokens)
        self.prediction_head = CrossEnvMLMHead(
            hidden_size=config.hidden_size,
            target_vocab_size=target_vocab_size,
            dropout=dropout,
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, label_smoothing=label_smoothing
        )

        # Store vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following BERT/DeBERTa conventions"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of cross-environment MLM model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            input_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for masked positions [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            relative_pos: Relative position information
            return_dict: Whether to return dictionary or just logits

        Returns:
            Dictionary with logits, loss, and other outputs
        """

        # Get molecular representations from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            output_all_encoded_layers=True,
            position_ids=position_ids,
            relative_pos=relative_pos,
        )

        # Extract hidden states from last layer
        hidden_states = encoder_outputs["hidden_states"][
            -1
        ]  # [batch, seq_len, hidden_size]

        # Generate predictions for target vocabulary
        logits = self.prediction_head(
            hidden_states
        )  # [batch, seq_len, target_vocab_size]

        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
            "encoder_outputs": encoder_outputs,
        }

        # Calculate loss if labels provided
        if labels is not None:
            # Flatten tensors for loss calculation
            flat_logits = logits.view(-1, self.target_vocab_size)
            flat_labels = labels.view(-1)

            # Calculate cross-entropy loss (ignores -100 labels)
            loss = self.loss_fn(flat_logits, flat_labels)
            outputs["loss"] = loss

            # Calculate accuracy on masked positions only
            mask = flat_labels != -100
            if mask.sum() > 0:
                predictions = flat_logits.argmax(dim=-1)
                correct = (predictions == flat_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                outputs["accuracy"] = accuracy

                # Calculate perplexity
                valid_loss = F.cross_entropy(
                    flat_logits[mask], flat_labels[mask], reduction="mean"
                )
                perplexity = torch.exp(valid_loss)
                outputs["perplexity"] = perplexity

        if return_dict:
            return outputs
        else:
            return logits


class CrossEnvMLMHead(nn.Module):
    """Prediction head for cross-environment MLM"""

    def __init__(self, hidden_size: int, target_vocab_size: int, dropout: float = 0.1):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, target_vocab_size)

        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of prediction head

        Args:
            hidden_states: Hidden states from encoder [batch, seq_len, hidden_size]

        Returns:
            Logits for target vocabulary [batch, seq_len, target_vocab_size]
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.decoder(x)

        return logits


class CrossEnvMLM(Model):
    """PyTorch Lightning wrapper for cross-environment MLM model"""

    def __init__(
        self,
        model: CrossEnvMLMModel,
        optimizer_cfg: OptimizerConfig,
        metrics: Optional[MetricsDict] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        log_predictions: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Lightning module

        Args:
            model: The core CrossEnvMLMModel
            optimizer_cfg: Optimizer configuration
            metrics: Metrics to track
            scheduler_cfg: Learning rate scheduler configuration
            log_predictions: Whether to log prediction examples
        """

        # Initialize default metrics if none provided
        if metrics is None:
            metrics = MetricsDict(
                train_loss=MeanMetric(),
                train_accuracy=MeanMetric(),
                train_perplexity=MeanMetric(),
                val_loss=MeanMetric(),
                val_accuracy=MeanMetric(),
                val_perplexity=MeanMetric(),
            )

        super().__init__(
            optimizer_cfg=optimizer_cfg,
            metrics=metrics,
            scheduler_cfg=scheduler_cfg,
            **kwargs,
        )

        self.model = model
        self.log_predictions = log_predictions

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, float, Any]]:
        """Forward pass through the model"""
        return self.model(
            input_ids=input_ids, input_mask=input_mask, labels=labels, **kwargs
        )

    def training_step(
        self, batch, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Training step"""
        from torch_geometric.utils import to_dense_batch

        # Convert batch to dense format
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)
        # Let the encoder handle relative position creation
        relative_pos = None

        # Forward pass
        outputs = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            relative_pos=relative_pos,
        )

        # Log metrics
        self.log(
            "train_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if "accuracy" in outputs:
            self.log(
                "train_accuracy",
                outputs["accuracy"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        if "perplexity" in outputs:
            self.log(
                "train_perplexity",
                outputs["perplexity"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return outputs

    def validation_step(
        self, batch, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Validation step"""
        from torch_geometric.utils import to_dense_batch

        # Convert batch to dense format
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        labels, _ = to_dense_batch(batch.labels, batch.batch, fill_value=-100)
        # Let the encoder handle relative position creation
        relative_pos = None

        # Forward pass
        outputs = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            relative_pos=relative_pos,
        )

        # Log metrics
        self.log(
            "val_loss", outputs["loss"], on_epoch=True, prog_bar=True, sync_dist=True
        )
        if "accuracy" in outputs:
            self.log(
                "val_accuracy",
                outputs["accuracy"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        if "perplexity" in outputs:
            self.log(
                "val_perplexity", outputs["perplexity"], on_epoch=True, sync_dist=True
            )

        # Log prediction examples
        if self.log_predictions and batch_idx == 0:
            self._log_prediction_examples(
                input_ids, labels, outputs["logits"], input_mask
            )

        return outputs

    def _log_prediction_examples(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        input_mask: torch.Tensor,
        num_examples: int = 3,
    ):
        """Log a few prediction examples for debugging"""
        predictions = logits.argmax(dim=-1)

        for i in range(min(num_examples, input_ids.size(0))):
            # Find masked positions
            masked_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]

            if len(masked_positions) > 0:
                pos = masked_positions[0].item()  # Take first masked position

                input_token = input_ids[i, pos].item()
                true_target = labels[i, pos].item()
                pred_target = predictions[i, pos].item()

                self.logger.experiment.add_text(
                    f"predictions/example_{i}",
                    f"Input: {input_token}, True: {true_target}, Pred: {pred_target}",
                    self.global_step,
                )

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        """Update metrics for train and val steps"""
        # The metrics are already logged in training_step and validation_step
        # So we can just call the parent update_metrics
        super().update_metrics(outputs, batch)
