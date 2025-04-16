import torch
from torch import nn
import copy
import logging

# Imports likely needed by BertEmbeddings
from DeBERTa.deberta.ops import LayerNorm
from DeBERTa.deberta.ops import MaskedLayerNorm
from DeBERTa.deberta.ops import StableDropout

# Imports needed by AtomEnvEmbeddings
from DeBERTa.deberta.cache_utils import load_model_state

# Update import for BertEncoder from nn.layers to models.encoders (assuming it will move there)
# from mole.nn.layers import BertEncoder # OLD
# Temporarily remove import if BertEncoder hasn't moved yet to avoid errors,
# or point to correct future location if confident.
# Let's assume it will be in mole.models.encoders based on plan
from mole.models.encoders import (
    BertEncoder,
)  # NEW - Assuming BertEncoder will be moved here

# Initialize logger for this module
logger = logging.getLogger(__name__)


# Removed from mole/nn/layers.py
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        padding_idx = getattr(config, "padding_idx", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.position_biased_input = getattr(config, "position_biased_input", True)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, self.embedding_size
            )

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, self.embedding_size
            )

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(
                self.embedding_size, config.hidden_size, bias=False
            )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.output_to_half = False
        self.config = config

    def forward(self, input_ids, token_type_ids=None, position_ids=None, mask=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = (
            self.position_embeddings(position_ids.long())
            if self.position_biased_input
            else None
        )
        # position_embeddings = self.position_embeddings(position_ids.long())

        embeddings = words_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.position_biased_input:
            embeddings += position_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
        embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
        embeddings = self.dropout(embeddings)
        return {"embeddings": embeddings, "position_embeddings": position_embeddings}


# Added AtomEnvEmbeddings class definition
class AtomEnvEmbeddings(torch.nn.Module):
    """AtomEnvEmbeddings is a DeBERTa encoder
    This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.
    Parameters:
        config: A model config class instance with the configuration to build a new model.
        pre_trained: Path to a pre-trained model or None.
    """

    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        if pre_trained is not None:
            state, model_config = load_model_state(pre_trained)
            if config is not None and model_config is not None:
                for k in config.__dict__:
                    # Use getattr to avoid error if key doesn't exist in config
                    if k not in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "vocab_size",
                        "max_position_embeddings",
                    ]:
                        setattr(model_config, k, getattr(config, k, None))
            config = copy.copy(model_config)

        # Ensure BertEmbeddings uses the potentially updated config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(
        self,
        input_ids,
        input_mask=None,
        attention_mask=None,
        token_type_ids=None,
        output_all_encoded_layers=True,
        position_ids=None,
        return_att=False,
        relative_pos=None,
    ):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            attention_mask = input_mask
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        ebd_output = self.embeddings(
            input_ids.to(torch.long),
            token_type_ids.to(torch.long),
            position_ids,
            input_mask,
        )
        embedding_output = ebd_output["embeddings"]
        encoder_output = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            return_att=return_att,
            relative_pos=relative_pos,
        )
        encoder_output.update(ebd_output)
        return encoder_output

    def apply_state(self, state=None):
        if self.pre_trained is None and state is None:
            return
        if state is None:
            # Ensure load_model_state handles potential errors
            try:
                state, config = load_model_state(self.pre_trained)
                self.config = config
            except Exception as e:
                # Use logger
                logger.error(
                    f"Error loading pre-trained state for {self.pre_trained}: {e}"
                )
                # Decide how to handle: raise error or continue without pre-trained weights?
                # For now, let's return, effectively skipping pre-trained loading on error
                return

        # Check if state is actually loaded before applying
        if state:
            # Use self.load_state_dict which is the standard PyTorch method
            load_result = self.load_state_dict(
                state, strict=False
            )  # Use strict=False to ignore mismatches
            # Log results
            logger.info(
                f"Loading state dict for AtomEnvEmbeddings. Load result: {load_result}"
            )


# Update __all__
__all__ = ["BertEmbeddings", "AtomEnvEmbeddings"]
