# Adapted from DeBERTa.DeBERTa.apps.models.masked_language_model.MaskedLanguageModel
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Union

from DeBERTa.deberta.config import ModelConfig
import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric

from mole.models.base import Model, TensorDict, OptimizerConfig, SchedulerConfig

# Comment out the import as the module is missing
# from mole.models.utils.masks import build_attention_span_mask
from mole.models.embeddings import AtomEnvEmbeddings
from mole.models.heads import TaskPredictionHead
from mole.metrics import MetricsDict

__all__ = ["AtomEnvEmbeddings"]


class Supervised(torch.nn.Module):
    """Fine tunning of MolE to predict molecular properties"""

    def __init__(
        self,
        deberta_config: dict,
        loss_fn: torch.nn.modules.loss._Loss,
        num_tasks: int = 1,
        num_classes: int = 1,
        dropout: Optional[float] = None,
        vocab_size_inp: Optional[int] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        config = ModelConfig.from_dict(deberta_config)
        config.vocab_size = (
            vocab_size_inp if vocab_size_inp is not None else config.vocab_size
        )
        self.MolE = AtomEnvEmbeddings(config)
        self.config = self.MolE.config

        self.prediction_head = TaskPredictionHead(
            self.config,
            loss_fn=loss_fn,
            num_tasks=num_tasks,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.apply(self.init_weights)

        if freeze_encoder:
            for name, param in self.MolE.named_parameters():
                print("Freezing layer: ", name)
                param.requires_grad = False

    def init_weights(self, module):
        """Apply Gaussian(mean=0, std=`config.initializer_range`) initialization to the module.
        Args:
        module (:obj:`torch.nn.Module`): The module to apply the initialization.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_mask=None,
        labels=None,
        position_ids=None,
        attention_mask=None,
        relative_pos=None,
        aux_labels=None,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        type_ids = None
        if labels is not None:
            labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = input_mask

        encoder_output = self.MolE(
            input_ids,
            input_mask,
            type_ids,
            output_all_encoded_layers=True,
            position_ids=position_ids,
            relative_pos=relative_pos,
        )
        hidden_states = encoder_output["hidden_states"]
        ctx_layer = hidden_states[-1]  # select last encoder layer
        context_token = ctx_layer[:, 0]  # select embedding of first token ie. CLS token
        logits, loss, labels = self.prediction_head(context_token, labels)

        output = {"logits": logits}
        if labels is not None:
            output.update({"loss": loss, "labels": labels})

        return output


class MolE(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: OptimizerConfig,
        metrics: MetricsDict,
        attention_span_masking: Optional[int] = None,
        aux_loss_lambda: Optional[float] = 0.0,
        scheduler_cfg: Optional[SchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer_cfg=optimizer_cfg,
            metrics=metrics,
            scheduler_cfg=scheduler_cfg,
            **kwargs,
        )
        self.model = model
        self.attention_span_masking = attention_span_masking
        self.aux_loss_lambda = aux_loss_lambda
        self.train_metrics = MetricsDict(
            mean_loss=MeanMetric(),
            mean_main_loss=MeanMetric(),
            mean_aux_loss=MeanMetric(),
        )  # type:ignore[arg-type]
        self.val_metrics = self.train_metrics.clone()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        aux_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, float, Any]]:
        output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
            aux_labels=aux_labels,
        )
        return output

    def training_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        labels = batch.target_labels
        if self.attention_span_masking:
            # Comment out usage as the function is missing
            # attention_mask = build_attention_span_mask(
            #     labels,
            #     relative_pos=relative_pos,
            #     input_mask=input_mask,
            #     span=self.attention_span_masking,
            # )
            # Fallback to using input_mask if attention_span_masking was intended
            attention_mask = input_mask
            print(
                "Warning: build_attention_span_mask is missing, using input_mask instead."
            )  # Optional warning
        else:
            attention_mask = input_mask

        output = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        self.log(
            "train_loss",
            output["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return output

    def validation_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        return self.training_step(batch=batch, batch_idx=batch_idx)

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        """Update metrics for train and val steps."""
        metrics = self.train_metrics if self.training else self.val_metrics
        metrics.mean_loss.update(outputs["loss"])
        metrics.mean_main_loss.update(outputs["main_loss"])
        metrics.mean_aux_loss.update(outputs["aux_loss"])
        super().update_metrics(outputs, batch)
