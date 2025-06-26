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