# Adapted from DeBERTa.DeBERTa.apps.models.masked_language_model.MaskedLanguageModel
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Union, Sequence

from DeBERTa.deberta.config import ModelConfig
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch

from mole.models.base import Model, TensorDict, OptimizerConfig
from mole.models.heads import TaskPredictionHead
from mole.metrics import MetricsDict

from DeBERTa.deberta.da_utils import build_relative_position
from DeBERTa.deberta.ops import ACT2FN, LayerNorm, MaskedLayerNorm, StableDropout

from mole.nn.attention import DisentangledSelfAttention

__all__ = [
    "encoder",
    "Encoder",
    "BertSelfOutput",
    "BertAttention",
    "BertIntermediate",
    "BertOutput",
    "BertLayer",
    "ConvLayer",
    "BertEncoder",
]


class encoder(nn.Module):
    """Fine tunning of MolE to predict molecular properties"""

    def __init__(
        self,
        deberta_config: dict,
        vocab_size_inp: Optional[int] = None,
        freeze_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        from mole.models.embeddings import AtomEnvEmbeddings

        config = ModelConfig.from_dict(deberta_config)
        config.vocab_size = (
            vocab_size_inp if vocab_size_inp is not None else config.vocab_size
        )
        self.MolE = AtomEnvEmbeddings(config)
        self.config = self.MolE.config

        self.prediction_head = TaskPredictionHead(
            self.config,
            loss_fn=None,
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
        position_ids=None,
        attention_mask=None,
        relative_pos=None,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        type_ids = None
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
        return ctx_layer


class Encoder(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: OptimizerConfig,
        metrics: MetricsDict,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer_cfg=optimizer_cfg,
            metrics=metrics,
            scheduler_cfg=scheduler_cfg,
            **kwargs,
        )
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, float, Any]]:
        output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        if isinstance(output, torch.Tensor):
            return {"encoder_output": output}
        return output

    def training_step(self, batch: Data, batch_idx: int):
        raise NotImplementedError("Encoder class does not implement training_step.")

    def validation_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        output_dict = self(
            input_ids=input_ids,
            input_mask=input_mask,
            position_ids=None,
            attention_mask=None,
            relative_pos=relative_pos,
        )

        return output_dict

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        pass


# Helper classes moved from layers.py
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_states, mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_states
        hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        self_output, att_matrix, _ = (
            output["hidden_states"],
            output["attention_probs"],
            output["attention_logits"],
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states, attention_mask)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_states, mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_states
        hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_att=return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(
            intermediate_output, attention_output, attention_mask
        )
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = torch.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = (
            self.conv(hidden_states.permute(0, 2, 1).contiguous())
            .permute(0, 2, 1)
            .contiguous()
        )
        # Ensure rmask is boolean for newer PyTorch versions
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))
        output_states = MaskedLayerNorm(
            self.LayerNorm, residual_states + out, input_mask
        )

        return output_states


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(
                self.max_relative_positions * 2, config.hidden_size
            )
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(
                -2
            ).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        relative_pos_embeddings = None
        if self.relative_attention:
            if relative_pos is None:
                q = (
                    query_states.size(1)
                    if query_states is not None
                    else hidden_states.size(1)
                )
                relative_pos = build_relative_position(
                    q, hidden_states.size(1), self.max_relative_positions
                ).to(hidden_states.device)
            relative_pos_embeddings = self.rel_embeddings(relative_pos)
            relative_pos_embeddings = self.pos_dropout(relative_pos_embeddings)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = relative_pos_embeddings

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_head_mask = None  # Hardcoded to None for now

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                layer_outputs = layer_module(
                    next_kv,
                    attention_mask,
                    return_att=output_attentions,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                )

            hidden_states = layer_outputs[0] if output_attentions else layer_outputs
            next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        # Return a dictionary like the original DeBERTa encoder
        return {
            "hidden_states": (
                all_hidden_states if output_hidden_states else hidden_states
            ),
            "last_hidden_state": hidden_states,
            "attentions": all_attentions if output_attentions else None,
        }
