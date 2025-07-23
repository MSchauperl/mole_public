# Disentangled Self-Attention: Summary

## What is Disentangled Self-Attention?
Disentangled Self-Attention is an advanced attention mechanism used in models like DeBERTa. Unlike standard self-attention, which only models content-to-content (token-to-token) relationships, disentangled attention can separately model:

- **Content-to-Content (c2c):** How much one token attends to another based on their content.
- **Content-to-Position (c2p):** How much a token's content attends to the position of another token.
- **Position-to-Content (p2c):** How much a token's position attends to the content of another token.
- **Position-to-Position (p2p):** How much a token's position attends to the position of another token.

This separation allows the model to better capture both semantic and positional relationships in the input sequence.

## Key Components
- **Query, Key, Value Projections:**
  - Standard attention uses linear projections to create queries, keys, and values from the input.
  - Disentangled attention adds additional projections for position-based queries and keys.

- **Relative Position Embeddings:**
  - Instead of absolute positions, the model uses relative positions (distance between tokens) to compute attention biases.
  - These embeddings are used in c2p, p2c, and p2p computations.

- **Attention Score Calculation:**
  - The final attention score for each token pair is a sum of the different components (c2c, c2p, p2c, p2p), each computed using the appropriate projections and relative position embeddings.
  - Each component is scaled for numerical stability.

- **Masked Softmax (XSoftmax):**
  - Used to ensure that attention is only computed over valid (non-masked) positions, e.g., ignoring padding tokens.

## How Relative Position Embeddings Work in Detail

### Molecular Graph-Based Relative Positions

**Important Note**: In this molecular modeling context, relative positions are **not** based on sequential token order like in traditional NLP. Instead, they represent the **molecular graph structure** - the actual chemical bonds between atoms.

The relative position matrix is created from the molecular graph connectivity (lines 281, 333 in `crossenv_mlm.py`):

```python
# Create dense relative position matrix from graph connections
relative_pos = to_dense_adj(
    edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr
)
```

This converts the molecular graph's edge information into a dense adjacency matrix where:
- `edge_index`: Contains pairs of atom indices that are chemically bonded
- `edge_attr`: Contains bond features (single/double bonds, bond types, etc.)
- `relative_pos`: Results in a matrix where connected atoms have non-zero values based on their bond relationships

### Position Index Generation
The relative position embeddings start with creating position indices using the `build_relative_position` function (lines 202-206 in `attention.py`):

```python
relative_pos = build_relative_position(
    q,
    key_layer.size(-2),
    bucket_size=self.position_buckets,
    max_position=self.max_relative_positions,
)
```

**Note**: This sequential fallback is only used when `relative_pos=None` (lines 386-390 in `encoders.py`). In molecular contexts, the graph-based relative positions are passed explicitly.

This function creates a 2D matrix where each element `(i, j)` represents the relative distance between query position `i` and key position `j`. The implementation in `da_utils.py` (lines 28-40) works as follows:

1. **Basic Distance Calculation**: `rel_pos_ids = q_ids.view(-1,1) - k_ids.view(1,-1)` computes the raw relative distances
2. **Logarithmic Bucketing** (optional): When `bucket_size > 0`, positions are mapped to logarithmic buckets to handle long sequences efficiently
3. **Result**: A matrix of shape `(1, query_size, key_size)` containing relative position indices

### Embedding Slicing and Reshaping
The relative position embeddings (`rel_embeddings`) need to be properly sliced for attention computation (lines 222-248 in `attention.py`):

```python
# Extract relevant slice for attention computation
if rel_embeddings.dim() == 3:
    seq_len = rel_embeddings.size(0)
    center = seq_len // 2
    start_idx = max(0, center - att_span)
    end_idx = min(seq_len, center + att_span)
    rel_embeddings = rel_embeddings[start_idx:end_idx, 0, :].unsqueeze(0)
```

This extracts a window of embeddings centered around the current position, with `att_span = pos_ebd_size` determining the window size.

### Position Projections
The relative position embeddings are projected into query and key spaces (lines 250-267 in `attention.py`):

```python
# For content-to-position and position-to-position
if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
    pos_key_layer = self.transpose_for_scores(
        self.pos_key_proj(rel_embeddings), self.num_attention_heads
    ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)

# For position-to-content and position-to-position
if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
    pos_query_layer = self.transpose_for_scores(
        self.pos_query_proj(rel_embeddings), self.num_attention_heads
    ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
```

These projections create position-aware queries and keys that are separate from the content-based ones.

### Content-to-Position (c2p) Computation
The c2p attention computes how content tokens attend to different positions (lines 270-286 in `attention.py`):

```python
if "c2p" in self.pos_att_type:
    scale = 1 / math.sqrt(pos_key_layer.size(-1) * scale_factor)
    c2p_att = torch.bmm(
        query_layer, pos_key_layer.transpose(-1, -2).to(query_layer) * scale
    )
    c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
    c2p_att = torch.gather(
        c2p_att, dim=-1, index=torch.repeat_interleave(c2p_pos.squeeze(1), ...)
    )
    score += c2p_att
```

Key steps:
1. **Scaled Dot-Product**: Content queries multiplied by position keys
2. **Position Indexing**: `c2p_pos` maps relative positions to embedding indices
3. **Gathering**: Select the appropriate attention scores using `torch.gather`

**In molecular context**: This computes how an atom's chemical features attend to the graph positions of other atoms.

### Position-to-Content (p2c) Computation
The p2c attention computes how positions attend to content tokens (lines 304-325 in `attention.py`):

```python
if "p2c" in self.pos_att_type:
    p2c_att = torch.bmm(
        key_layer, pos_query_layer.transpose(-1, -2).to(key_layer) * scale
    )
    p2c_att = torch.gather(
        p2c_att, dim=-1, index=torch.repeat_interleave(p2c_pos.squeeze(1), ...)
    ).transpose(-1, -2)
    score += p2c_att
```

This is similar to c2p but uses content keys and position queries, with the result transposed.

**In molecular context**: This computes how an atom's graph position attends to the chemical features of other atoms.

### Position-to-Position (p2p) Computation
The p2p attention models pure positional relationships (lines 328-345 in `attention.py`):

```python
if "p2p" in self.pos_att_type:
    pos_query = pos_query_layer[:, :, att_span:, :]
    p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
    p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
    score += p2p_att
```

This computes attention between position embeddings directly, without involving content.

**In molecular context**: This models how the graph connectivity patterns themselves interact, independent of atomic features.

### Numerical Stability and Scaling
The attention scores are normalized for numerical stability (lines 135-138 in `attention.py`):

```python
scale_factor = 1
if "c2p" in self.pos_att_type: scale_factor += 1
if "p2c" in self.pos_att_type: scale_factor += 1
if "p2p" in self.pos_att_type: scale_factor += 1
scale = 1 / math.sqrt(query_layer.size(-1) * scale_factor)
```

The `scale_factor` accounts for the additional attention components being summed together.

## Why Use Disentangled Attention?
- **Richer Representations:** By modeling content and position interactions separately, the model can learn more nuanced relationships.
- **Improved Performance:** Especially beneficial for tasks where word order and relative positions are important (e.g., language modeling, question answering).
- **Flexibility:** The mechanism can be configured to use any combination of c2c, c2p, p2c, and p2p, depending on the task and model configuration.

## High-Level Workflow
1. **Input:** Sequence of token embeddings.
2. **Projection:** Compute queries, keys, and values for both content and position.
3. **Relative Position:** Build relative position indices and embeddings.
4. **Attention Scores:**
    - Compute c2c, c2p, p2c, and p2p scores.
    - Sum them to get the final attention logits.
5. **Masking:** Apply XSoftmax to ensure only valid positions are attended to.
6. **Context Layer:** Use attention probabilities to compute a weighted sum of value vectors, producing the output representations.

## References
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention (Microsoft)](https://arxiv.org/abs/2006.03654)
- [DeBERTa Official GitHub](https://github.com/microsoft/DeBERTa)
- [HuggingFace DeBERTa Documentation](https://huggingface.co/docs/transformers/model_doc/deberta)

---

**In summary:** Disentangled Self-Attention enhances the standard attention mechanism by explicitly modeling both content and positional relationships, leading to richer and more flexible sequence representations.