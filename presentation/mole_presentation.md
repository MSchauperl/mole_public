# MolE: Molecular Embeddings with Disentangled Self-Attention
## A Transformer Architecture for Molecular Property Prediction

---

## Slide 1: Title Slide

**Title:** MolE: Molecular Embeddings with Disentangled Self-Attention  
**Subtitle:** A Transformer Architecture for Molecular Property Prediction  

**Visual Elements:**
- Molecular structure (e.g., caffeine or aspirin molecule)
- Transformer architecture icon/diagram
- Arrow showing molecule → prediction

---

## Slide 2: What is MolE?

**MolE (Molecular Embeddings)** is a sophisticated transformer-based model for molecular property prediction that combines:

- **DeBERTa-inspired architecture** with molecular adaptations
- **Disentangled self-attention** for enhanced molecular understanding
- **Graph-aware processing** that respects molecular structure
- **Multi-task prediction** capabilities

**Key Innovation:** Treats molecular graphs as sequences of atom environments while preserving structural information through advanced attention mechanisms.

**Visual:** High-level flow: Molecular Graph → Atom Environments → Transformer → Property Predictions

---

## Slide 3: Core Architectural Innovations

**MolE introduces several key innovations:**

1. **Molecular Graph Integration**
   - Converts molecular graphs to atom environment sequences
   - Preserves structural information through relative position embeddings
   - Handles variable-length molecules efficiently

2. **Disentangled Self-Attention**
   - Separates content (what atoms are) from position (where they are)
   - Four attention types: content-to-content, content-to-position, position-to-content, position-to-position

3. **Pre-training Integration**
   - Compatible with DeBERTa pre-trained weights
   - Transfer learning from language models to molecular domain

**Visual:** Three-column diagram showing Standard Transformer vs. DeBERTa vs. MolE

---

## Slide 4: Input Processing - From Molecules to Sequences

**Challenge:** How do you feed a molecular graph into a transformer?

**MolE's Solution:**
1. **Molecular Graph Representation**
   - Nodes: Atoms with features (element, charge, etc.)
   - Edges: Chemical bonds with attributes (bond type, distance)

2. **Atom Environment Tokenization**
   - Each atom becomes a token representing its local chemical environment
   - Environment includes atom type, bonding patterns, local structure

3. **Sequence Formation**
   - Molecular graph → Sequence of atom environment tokens
   - Variable length sequences (different molecule sizes)
   - Batch processing with padding

**Visual:** Molecular graph (benzene ring) → Tokenized sequence → Padded batch matrix

---

## Slide 5: Data Flow - Input Processing

**Complete Input Processing Pipeline:**

```
Molecular Graph → Atom Environment Tokenization → Sequence Padding → Batch Formation
```

**Technical Details:**
- **PyTorch Geometric integration:** Native graph data handling
- **Dense batch conversion:** `to_dense_batch()` for transformer compatibility
- **Relative position extraction:** `to_dense_adj()` for molecular graph structure
- **Attention masking:** Proper handling of padding tokens

**Code Example:**
```python
# Convert graph to dense representation
input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
```

**Visual:** Step-by-step pipeline diagram showing each transformation

---

## Slide 6: Embedding Layer Architecture

**AtomEnvEmbeddings** transforms molecular tokens into dense representations:

**Components:**
1. **Word Embeddings** (`nn.Embedding`)
   - Maps atom environment tokens to dense vectors
   - Dimension: `vocab_size × embedding_size`

2. **Position Embeddings** (`nn.Embedding`)
   - Encodes sequential position information
   - Dimension: `max_position_embeddings × embedding_size`

3. **Type Embeddings** (`nn.Embedding`)
   - Handles different token types (primarily atoms)
   - Dimension: `type_vocab_size × embedding_size`

**Visual:** Token → [Word Emb] + [Position Emb] + [Type Emb] → LayerNorm → Dropout → Output

---

## Slide 7: Embedding Layer - Technical Details

**BertEmbeddings Implementation:**

```python
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
```

**Key Features:**
- **Configurable dimensions:** Supports different embedding/hidden sizes
- **Projection layer:** When `embedding_size ≠ hidden_size`
- **Masked layer normalization:** Handles padding tokens properly
- **Stable dropout:** Improved training stability

**Visual:** Detailed embedding architecture with dimensions and data flow

---

## Slide 8: Disentangled Self-Attention - Overview

**What makes MolE's attention special?**

**Standard Transformer Attention:**
- Only models content-to-content relationships
- Limited understanding of structural relationships

**MolE's Disentangled Attention:**
- **Content**: Chemical features of atoms (what they are)
- **Position**: Molecular structure/connectivity (where they are)
- **Separation**: Models content and position interactions independently

**Four Attention Types:**
1. **c2c**: Content-to-Content (how atoms attend to other atoms)
2. **c2p**: Content-to-Position (how atoms attend to molecular positions)
3. **p2c**: Position-to-Content (how positions attend to chemical features)
4. **p2p**: Position-to-Position (how molecular structure patterns interact)

**Visual:** 2×2 grid showing the four attention interaction types

---

## Slide 9: Disentangled Attention - Mathematical Formulation

**Attention Score Calculation:**

```
attention_score(i,j) = c2c(i,j) + c2p(i,j) + p2c(i,j) + p2p(i,j)

where:
- c2c(i,j) = query_content(i) · key_content(j)
- c2p(i,j) = query_content(i) · key_position(j)
- p2c(i,j) = query_position(i) · key_content(j)
- p2p(i,j) = query_position(i) · key_position(j)
```

**Scaling Factor:**
```
scale_factor = 1 + |{enabled_attention_types}|
scale = 1 / sqrt(d_k * scale_factor)
```

**Final Attention:**
```
attention_probs = softmax(attention_scores / scale)
output = attention_probs × value_vectors
```

**Visual:** Mathematical formula breakdown with color-coded terms

---

## Slide 10: Molecular Context - What Makes It Special

**In Molecular Modeling Context:**

**Content (Chemical Features):**
- Atom type (C, N, O, etc.)
- Electronic properties
- Chemical environment
- Functional groups

**Position (Molecular Structure):**
- Bond connectivity
- 3D spatial relationships
- Graph topology
- Structural patterns

**Key Insight:** Traditional transformers only see sequence order, but MolE understands chemical structure through graph-aware position embeddings.

**Visual:** Molecular structure with atoms labeled as "content" and bonds/connectivity as "position"

---

## Slide 11: Relative Position Embeddings - Molecular Graph Approach

**Traditional NLP:** Positions = sequential token order (1, 2, 3, 4, ...)

**MolE's Innovation:** Positions = molecular graph connectivity

**Implementation:**
1. **Build Position Matrix**
   ```python
   relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
   ```

2. **Graph-Based Distances**
   - Connected atoms have small relative distances
   - Disconnected atoms have large relative distances
   - Edge attributes encode bond types

3. **Attention Bias**
   - Relative positions create attention bias
   - Encourages attention along chemical bonds
   - Respects molecular structure

**Visual:** Molecular graph → Adjacency matrix → Relative position matrix → Attention bias

---

## Slide 12: Encoder Architecture - BertEncoder

**Multi-Layer Transformer Stack:**

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        self.layer = nn.ModuleList([
            BertLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.rel_embeddings = nn.Embedding(max_relative_positions * 2, hidden_size)
```

**Key Components:**
- **Configurable layers:** Typically 6-12 transformer layers
- **Relative position embeddings:** For molecular graph structure
- **Gradient checkpointing:** Memory-efficient training
- **Attention output:** Hidden states + attention weights

**Visual:** Stack of transformer layers with data flow arrows

---

## Slide 13: BertLayer - Individual Transformer Layer

**Each BertLayer contains:**

1. **BertAttention**
   - Disentangled self-attention mechanism
   - Multi-head attention with molecular awareness
   - Attention dropout for regularization

2. **BertIntermediate**
   - Feed-forward network
   - Configurable activation function (GELU, ReLU, etc.)
   - Dimension expansion (typically 4x hidden_size)

3. **BertOutput**
   - Linear projection back to hidden_size
   - Residual connection
   - Layer normalization

**Visual:** Layer architecture: Input → Attention → Add&Norm → FFN → Add&Norm → Output

---

## Slide 14: Feed-Forward Networks

**BertIntermediate & BertOutput:**

```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

class BertOutput(nn.Module):
    def __init__(self, config):
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
```

**Optional ConvLayer:**
- 1D convolutions for additional processing
- Configurable kernel sizes and groups
- Residual connections with masked operations

**Visual:** FFN architecture showing dimension changes and residual connections

---

## Slide 15: Prediction Heads - Multi-Task Architecture

**TaskPredictionHead** handles diverse molecular property prediction:

**Architecture:**
```python
class TaskPredictionHead(nn.Module):
    def __init__(self, config):
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
```

**Capabilities:**
- **Multi-task support:** Multiple properties simultaneously
- **Classification & Regression:** Flexible output types
- **Context token processing:** Uses CLS token ([CLS] equivalent)
- **NaN handling:** Robust training with missing labels

**Visual:** Context token → Dense → Activation → Dropout → Task-specific outputs

---

## Slide 16: Complete Architecture Flow

**End-to-End MolE Pipeline:**

```
Molecular Graph → Tokenization → Embedding → Encoding → Prediction
```

**Detailed Flow:**
1. **Input:** Molecular graphs (nodes, edges, features)
2. **Processing:** 
   - Atom environment tokenization
   - Embedding layer (word + position + type)
   - Multi-layer transformer encoding
   - Disentangled attention at each layer
3. **Output:** Molecular property predictions

**Technical Integration:**
- **PyTorch Geometric:** Graph data handling
- **PyTorch Lightning:** Training infrastructure
- **Batch processing:** Efficient GPU utilization

**Visual:** Complete pipeline diagram with all components

---

## Slide 17: Data Processing Details

**Batch Processing with Variable-Length Molecules:**

```python
def validation_step(self, batch, batch_idx):
    # Convert graph to dense representation
    input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
    relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
    
    # Forward pass
    output = self.model(
        input_ids=input_ids,
        input_mask=input_mask,
        relative_pos=relative_pos
    )
    return output
```

**Key Features:**
- **Efficient batching:** Handles different molecule sizes
- **Attention masking:** Proper padding token handling
- **Memory optimization:** Gradient checkpointing available

**Visual:** Batch processing diagram showing molecules of different sizes

---

## Slide 18: Training & Optimization Strategy

**Training Infrastructure:**

**PyTorch Lightning Integration:**
- Automatic optimization and gradient clipping
- Distributed training support
- Comprehensive logging and checkpointing

**Optimization Strategy:**
- **Optimizer:** AdamW with weight decay regularization
- **Learning Rate:** Warmup + linear decay scheduling
- **Regularization:** Multiple dropout types, layer normalization

**Performance Optimization:**
- **Gradient checkpointing:** Memory-efficient training
- **Mixed precision:** Faster training on modern GPUs
- **Efficient attention:** Optimized attention computation

**Visual:** Training pipeline diagram with optimization components

---

## Slide 19: Model Scalability & Configuration

**Configuration-Driven Design:**

```python
config = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "vocab_size": 40000,
    "relative_attention": True,
    "pos_att_type": "c2p|p2c"
}
```

**Scalability Features:**
- **Model sizes:** 100M to 1B+ parameters
- **Context length:** Up to 512 atom environments
- **Processing speed:** 1000-5000 molecules/second
- **Memory efficient:** Gradient checkpointing support

**Visual:** Configuration diagram showing different model sizes and capabilities

---

## Slide 20: Applications & Use Cases

**Drug Discovery:**
- Molecular property prediction (LogP, solubility, toxicity)
- ADMET prediction (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- Lead compound optimization

**Materials Science:**
- Catalyst property prediction
- Polymer property modeling
- Novel material design

**Chemical Research:**
- Reaction outcome prediction
- Synthesis route planning
- Chemical space exploration

**Advantages:**
- **Multi-task learning:** Predict multiple properties simultaneously
- **Transfer learning:** Leverage pre-trained molecular representations
- **Interpretability:** Attention weights provide molecular insights

**Visual:** Application areas with example molecules and predicted properties

---

## Slide 21: Performance Characteristics

**Model Performance:**

**Accuracy:**
- State-of-the-art performance on molecular property benchmarks
- Competitive with specialized molecular models
- Benefits from pre-training and transfer learning

**Efficiency:**
- **Training speed:** ~1000-5000 molecules/second
- **Memory usage:** Scalable with gradient checkpointing
- **Inference speed:** Fast batch processing

**Scalability:**
- **Parameter range:** 100M-1B parameters
- **Context length:** Up to 512 atoms per molecule
- **Batch size:** Flexible based on GPU memory

**Visual:** Performance benchmarks and scaling curves

---

## Slide 22: Key Architectural Strengths

**MolE's Unique Advantages:**

1. **Molecular Graph Awareness**
   - Respects chemical structure through graph-aware attention
   - Preserves molecular connectivity information
   - Handles variable-length molecules efficiently

2. **Advanced Attention Mechanisms**
   - Disentangled attention for content/position separation
   - Multi-scale molecular understanding
   - Interpretable attention patterns

3. **Pre-training Integration**
   - Leverages powerful language model pre-training
   - Transfer learning from textual to molecular domains
   - Flexible architecture adaptation

4. **Production-Ready Design**
   - Robust error handling and logging
   - Distributed training support
   - Comprehensive configuration system

**Visual:** Comparison table with other molecular modeling approaches

---

## Slide 23: Future Directions & Extensions

**Ongoing Development:**

**Architectural Improvements:**
- Enhanced attention mechanisms
- Improved molecular graph processing
- Better pre-training strategies

**Application Expansion:**
- Larger molecular datasets
- Additional property types
- Cross-modal learning (text + structure)

**Technical Enhancements:**
- Model compression techniques
- Faster inference methods
- Better interpretability tools

**Research Opportunities:**
- Novel attention mechanisms
- Integration with 3D structure
- Quantum chemical property prediction

**Visual:** Research roadmap with current position and future directions

---

## Slide 24: Summary & Conclusion

**MolE: Key Takeaways**

**Innovation:**
- First transformer architecture with molecular graph-aware disentangled attention
- Separates chemical content from structural position information
- Leverages pre-trained language models for molecular understanding

**Technical Excellence:**
- Robust, production-ready architecture
- Flexible configuration system
- Comprehensive training infrastructure

**Impact:**
- State-of-the-art molecular property prediction
- Enables multi-task learning across diverse chemical properties
- Provides interpretable insights into molecular behavior

**Future:** MolE represents a significant advance in molecular machine learning, combining the power of transformers with domain-specific molecular understanding.

**Visual:** Summary infographic with key achievements and capabilities

---

## Additional Slides for Technical Deep Dive

### Slide 25: XSoftmax - Masked Softmax Implementation

**Challenge:** Standard softmax doesn't handle variable-length sequences properly

**XSoftmax Solution:**
```python
class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, dim):
        # Mask out invalid positions
        rmask = ~(mask.bool())
        output = input.masked_fill(rmask, float('-inf'))
        output = torch.softmax(output, dim)
        output.masked_fill_(rmask, 0)
        return output
```

**Benefits:**
- Proper handling of padding tokens
- Numerical stability
- ONNX export compatibility
- Custom backward pass for efficiency

**Visual:** Attention matrix before and after masking

---

### Slide 26: Attention Visualization

**Understanding Molecular Attention:**

**Content-to-Content Attention:**
- Shows which atoms attend to which other atoms
- Reveals chemical interaction patterns
- Highlights functional groups and reactive sites

**Position-to-Position Attention:**
- Shows structural relationship patterns
- Identifies important molecular scaffolds
- Reveals spatial organization

**Interpretation Benefits:**
- Understand model decision-making
- Identify important molecular features
- Validate chemical intuition

**Visual:** Attention heat maps overlaid on molecular structures

---

## Technical Appendix

### Configuration Examples
### Code Snippets
### Performance Benchmarks
### Mathematical Derivations
### Implementation Details

---

**End of Presentation**

*Total Slides: 26 main slides + technical appendix*
*Estimated Duration: 45-60 minutes*
*Target Audience: Technical researchers, ML engineers, computational chemists*