# Modified from DeBERTa.deberta.disentangled_attention
#  Disentangled SelfAttention module

import math

from DeBERTa.deberta.da_utils import build_relative_position
from DeBERTa.deberta.ops import StableDropout
from DeBERTa.utils import get_logger
from packaging import version
import torch
from torch import _softmax_backward_data as _softmax_backward_data
from torch import nn

logger = get_logger()

__all__ = ["DisentangledSelfAttention"]


class DisentangledSelfAttention(nn.Module):
    """
    Implements the Disentangled Self-Attention mechanism from DeBERTa.

    This attention module supports disentangled attention, which means it can separately model content-to-content,
    content-to-position, position-to-content, and position-to-position interactions using relative position embeddings.
    It is highly configurable via the model config.
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration object with attention parameters.
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention_head_size = getattr(
            config, "attention_head_size", _attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Linear projections for query, key, value
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        # Types of positional attention to use (content-to-position, position-to-content, etc.)
        self.pos_att_type = [
            x.strip() for x in getattr(config, "pos_att_type", "c2p").lower().split("|")
        ]  # c2p|p2c
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
                # For backward compatibility

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            # Projections for position-based attention
            if not self.share_att_key:
                if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        config.hidden_size, self.all_head_size, bias=True
                    )
                if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(
                        config.hidden_size, self.all_head_size
                    )

        self.dropout = StableDropout(config.attention_probs_dropout_prob)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def transpose_for_scores(self, x, attention_heads):
        """
        Reshapes and permutes the input tensor for multi-head attention.
        Args:
            x: Input tensor of shape (batch, seq_len, all_head_size)
            attention_heads: Number of attention heads
        Returns:
            Tensor of shape (batch * heads, seq_len, head_size)
        """
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Forward pass for disentangled self-attention.
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Mask tensor (batch, seq_len, seq_len)
            return_att: If True, also return attention weights
            query_states: Optional, use as queries instead of hidden_states
            relative_pos: Optional, relative position indices
            rel_embeddings: Optional, relative position embeddings
        Returns:
            Dictionary with keys: 'hidden_states', 'attention_probs', 'attention_logits'
        """
        if query_states is None:
            query_states = hidden_states
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        ).float()
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        ).float()
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        rel_att = None
        # Compute dot-product attention scores
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        if "p2p" in self.pos_att_type:
            scale_factor += 1
        scale = 1 / math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) * scale)
        # Add disentangled relative position bias if enabled
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        # Normalize scores for numerical stability
        attention_scores = (
            attention_scores
            - attention_scores.max(dim=-1, keepdim=True).values.detach()
        ).to(hidden_states)
        attention_scores = attention_scores.view(
            -1,
            self.num_attention_heads,
            attention_scores.size(-2),
            attention_scores.size(-1),
        )

        # Apply masked softmax (XSoftmax) and dropout
        _attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(_attention_probs)
        # Compute context layer (weighted sum of values)
        context_layer = torch.bmm(
            attention_probs.view(
                -1, attention_probs.size(-2), attention_probs.size(-1)
            ),
            value_layer,
        )
        context_layer = (
            context_layer.view(
                -1,
                self.num_attention_heads,
                context_layer.size(-2),
                context_layer.size(-1),
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {
            "hidden_states": context_layer,  # Output of attention
            "attention_probs": _attention_probs,  # Softmaxed attention weights
            "attention_logits": attention_scores,  # Raw attention logits
        }

    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        """
        Computes the disentangled attention bias for relative positions.
        This includes content-to-position, position-to-content, and position-to-position terms.
        Args:
            query_layer: Projected queries - shape: (batch*heads, seq_len, head_size)
            key_layer: Projected keys - shape: (batch*heads, seq_len, head_size)
            relative_pos: Relative position indices - shape: (seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            rel_embeddings: Relative position embeddings - shape: (seq_len, seq_len, hidden_size)
            scale_factor: Scaling factor for normalization
        Returns:
            Tensor of attention biases to add to attention scores
        """
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        # Ensure relative_pos has the right dimensions
        # Expected: (batch, heads, seq_len, seq_len)
        if relative_pos.dim() == 2:
            # relative_pos shape: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            # relative_pos shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        # rel_embeddings expected shape: (2*pos_ebd_size, hidden_size)
        # But we're getting: (seq_len, seq_len, hidden_size) or (batch, seq_len, seq_len, hidden_size)
        # We need to slice it properly for the attention mechanism

        if rel_embeddings.dim() == 4:
            # rel_embeddings shape: (batch, seq_len, seq_len, hidden_size)
            # Extract a 2D slice for attention computation
            batch_size, seq_len, _, hidden_size = rel_embeddings.shape
            # Take diagonal elements or center slice
            rel_embeddings = rel_embeddings[0, :, 0, :].unsqueeze(
                0
            )  # (1, seq_len, hidden_size)
        elif rel_embeddings.dim() == 3:
            # rel_embeddings shape: (seq_len, seq_len, hidden_size)
            # We need to extract the relevant slice for attention computation
            # Take the center slice that corresponds to our attention span
            seq_len = rel_embeddings.size(0)
            center = seq_len // 2
            start_idx = max(0, center - att_span)
            end_idx = min(seq_len, center + att_span)
            rel_embeddings = rel_embeddings[start_idx:end_idx, 0, :].unsqueeze(0)
            # rel_embeddings shape: (1, 2*att_span, hidden_size)
        else:
            # Original format: (2*pos_ebd_size, hidden_size)
            rel_embeddings = rel_embeddings[
                self.pos_ebd_size - att_span : self.pos_ebd_size + att_span,
                :,  # noqa: E203
            ].unsqueeze(0)

        # rel_embeddings shape: (1, seq_len, hidden_size) or (1, 2*att_span, hidden_size)

        if self.share_att_key:
            # If sharing attention key, use the same projection for Q and K
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            # pos_query_layer shape: (batch*heads, 2*att_span, head_size)
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            # pos_key_layer shape: (batch*heads, 2*att_span, head_size)
        else:
            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
                # pos_key_layer shape: (batch*heads, 2*att_span, head_size)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
                # pos_query_layer shape: (batch*heads, 2*att_span, head_size)

        score = 0
        # --- Content-to-Position (c2p) ---
        if "c2p" in self.pos_att_type:
            scale = 1 / math.sqrt(pos_key_layer.size(-1) * scale_factor)
            c2p_att = torch.bmm(
                query_layer, pos_key_layer.transpose(-1, -2).to(query_layer) * scale
            )
            c2p_pos = (
                torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
                if relative_pos.min() < 0
                else torch.clamp(relative_pos, 0, rel_embeddings.size(1) - 1)
            )
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=torch.repeat_interleave(
                    c2p_pos.squeeze(1),
                    torch.tensor([query_layer.size(0) // c2p_pos.size(0)]).to(
                        c2p_pos.device
                    ),
                    dim=0,
                ),
            )
            score += c2p_att

        # --- Position-to-Content (p2c) ---
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            scale = 1 / math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(
                    key_layer.size(-2),
                    key_layer.size(-2),
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                ).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = (
                torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
                if r_pos.min() < 0
                else torch.clamp(r_pos, 0, rel_embeddings.size(1) - 1)
            )
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = torch.bmm(
                key_layer, pos_query_layer.transpose(-1, -2).to(key_layer) * scale
            )
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=torch.repeat_interleave(
                    p2c_pos.squeeze(1),
                    torch.tensor([query_layer.size(0) // p2c_pos.size(0)]).to(
                        p2c_pos.device
                    ),
                    dim=0,
                ),
            ).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(
                    p2c_att,
                    dim=-2,
                    index=pos_index.expand(
                        p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))
                    ),
                )
            score += p2c_att

        # --- Position-to-Position (p2p) ---
        if "p2p" in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(
                    p2p_att,
                    dim=-2,
                    index=pos_index.expand(
                        query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))
                    ),
                )
            p2p_att = torch.gather(
                p2p_att,
                dim=-1,
                index=c2p_pos.expand(
                    [
                        query_layer.size(0),
                        query_layer.size(1),
                        query_layer.size(2),
                        relative_pos.size(-1),
                    ]
                ),
            )
            score += p2p_att

        return score

    def _pre_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Handles backward compatibility for loading older model checkpoints.
        Converts old projection weights to the new format if needed.
        """
        self_state = self.state_dict()
        if ((prefix + "query_proj.weight") not in state_dict) and (
            (prefix + "in_proj.weight") in state_dict
        ):
            v1_proj = state_dict[prefix + "in_proj.weight"]
            v1_proj = v1_proj.unsqueeze(0).reshape(
                self.num_attention_heads, -1, v1_proj.size(-1)
            )
            q, k, v = v1_proj.chunk(3, dim=1)
            state_dict[prefix + "query_proj.weight"] = q.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + "key_proj.weight"] = k.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + "key_proj.bias"] = self_state["key_proj.bias"]
            state_dict[prefix + "value_proj.weight"] = v.reshape(-1, v1_proj.size(-1))
            v1_query_bias = state_dict[prefix + "q_bias"]
            state_dict[prefix + "query_proj.bias"] = v1_query_bias
            v1_value_bias = state_dict[prefix + "v_bias"]
            state_dict[prefix + "value_proj.bias"] = v1_value_bias

            v1_pos_key_proj = state_dict[prefix + "pos_proj.weight"]
            state_dict[prefix + "pos_key_proj.weight"] = v1_pos_key_proj
            v1_pos_query_proj = state_dict[prefix + "pos_q_proj.weight"]
            state_dict[prefix + "pos_query_proj.weight"] = v1_pos_query_proj
            v1_pos_query_proj_bias = state_dict[prefix + "pos_q_proj.bias"]
            state_dict[prefix + "pos_query_proj.bias"] = v1_pos_query_proj_bias
            state_dict[prefix + "pos_key_proj.bias"] = self_state["pos_key_proj.bias"]

            del state_dict[prefix + "in_proj.weight"]
            del state_dict[prefix + "q_bias"]
            del state_dict[prefix + "v_bias"]
            del state_dict[prefix + "pos_proj.weight"]
            del state_dict[prefix + "pos_q_proj.weight"]
            del state_dict[prefix + "pos_q_proj.bias"]


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax optimized for memory and speed.

    This function applies softmax to the input tensor, but only to the elements where the mask is 1.
    Masked elements (mask == 0) are ignored in the softmax calculation and set to zero in the output.
    This is useful for attention mechanisms where some positions should not be attended to (e.g., padding).
    """

    @staticmethod
    def forward(self, input, mask, dim):
        """
        Forward pass for masked softmax.
        Args:
            input: Input tensor
            mask: Mask tensor (0 = ignore, 1 = include)
            dim: Dimension to apply softmax
        Returns:
            Softmaxed tensor with masked positions set to zero
        """
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        Backward pass for masked softmax.
        Args:
            grad_output: Gradient of the output
        Returns:
            Gradient of the input, None, None
        """
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        """
        Symbolic (ONNX export) implementation for masked softmax.
        """
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill
        from torch.onnx.symbolic_opset9 import softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op(
                "Sub",
                g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)),
                mask_cast_value,
            ),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf")))
        )
        output = softmax(g, output, dim)
        return masked_fill(
            g,
            output,
            r_mask,
            g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)),
        )
