# MolE Presentation Diagrams

This file contains all the Mermaid diagram code for the MolE presentation. You can use these diagrams to generate visual elements for your slides.

## Diagram 1: High-Level Architecture Flow

```mermaid
graph TD
    A[Molecular Graph] --> B[Atom Environment Tokenization]
    B --> C[Embedding Layer]
    C --> D[Multi-Layer Transformer]
    D --> E[Prediction Head]
    E --> F[Molecular Properties]
    
    subgraph "Input Processing"
        A
        B
    end
    
    subgraph "Core Architecture"
        C
        D
    end
    
    subgraph "Output"
        E
        F
    end
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#e8f5e8
```

**Usage:** Overview slide showing the complete MolE pipeline from molecular input to property predictions.

---

## Diagram 2: Disentangled Attention Types

```mermaid
graph TD
    subgraph "Disentangled Attention Types"
        A[Content Queries] --> E[c2c: Content-to-Content]
        A --> F[c2p: Content-to-Position]
        B[Position Queries] --> G[p2c: Position-to-Content]
        B --> H[p2p: Position-to-Position]
        
        C[Content Keys] --> E
        D[Position Keys] --> F
        C --> G
        D --> H
    end
    
    subgraph "Attention Computation"
        E --> I[Sum All Attention Types]
        F --> I
        G --> I
        H --> I
        I --> J[Scaled Softmax]
        J --> K[Weighted Sum with Values]
        K --> L[Output Representation]
    end
    
    style A fill:#ffecb3
    style B fill:#ffecb3
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#e1f5fe
    style F fill:#e1f5fe
    style G fill:#f3e5f5
    style H fill:#f3e5f5
```

**Usage:** Detailed explanation of the four types of disentangled attention and how they combine.

---

## Diagram 3: Embedding Layer Architecture

```mermaid
graph TD
    A[Atom Environment Tokens] --> B[Word Embeddings]
    C[Position IDs] --> D[Position Embeddings]
    E[Token Type IDs] --> F[Token Type Embeddings]
    
    B --> G[Sum]
    D --> G
    F --> G
    
    G --> H[Projection Layer<br/>if embedding_size ≠ hidden_size]
    H --> I[Masked Layer Normalization]
    I --> J[Stable Dropout]
    J --> K[Final Embeddings]
    
    subgraph "Embedding Components"
        B
        D
        F
    end
    
    subgraph "Post-Processing"
        H
        I
        J
    end
    
    style A fill:#ffecb3
    style C fill:#ffecb3
    style E fill:#ffecb3
    style K fill:#c8e6c9
```

**Usage:** Detailed view of how molecular tokens are converted to dense embeddings.

---

## Diagram 4: Transformer Layer Structure

```mermaid
graph TD
    A[Input Hidden States] --> B[Disentangled Self-Attention]
    A --> C[Residual Connection 1]
    B --> D[Attention Dropout]
    D --> C
    C --> E[Layer Normalization 1]
    
    E --> F[Feed-Forward Network]
    E --> G[Residual Connection 2]
    F --> H[Intermediate Layer<br/>hidden_size → intermediate_size]
    H --> I[Activation Function<br/>GELU/ReLU]
    I --> J[Output Layer<br/>intermediate_size → hidden_size]
    J --> K[FFN Dropout]
    K --> G
    G --> L[Layer Normalization 2]
    L --> M[Output Hidden States]
    
    subgraph "Attention Sub-layer"
        B
        D
        C
        E
    end
    
    subgraph "Feed-Forward Sub-layer"
        F
        H
        I
        J
        K
        G
        L
    end
    
    style A fill:#e1f5fe
    style M fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#fff3e0
```

**Usage:** Internal structure of a single transformer layer showing attention and feed-forward components.

---

## Diagram 5: Molecular Graph to Sequence Conversion

```mermaid
graph TD
    A[Molecular Graph] --> B[Nodes: Atoms with Features]
    A --> C[Edges: Chemical Bonds]
    
    B --> D[Atom Environment Extraction]
    C --> E[Bond Type Information]
    
    D --> F[Tokenization]
    E --> F
    
    F --> G[Sequence of Atom Tokens]
    G --> H[Padding for Batch Processing]
    
    C --> I[Adjacency Matrix Creation]
    I --> J[Relative Position Matrix]
    
    subgraph "Graph Components"
        B
        C
    end
    
    subgraph "Tokenization Process"
        D
        E
        F
        G
        H
    end
    
    subgraph "Position Encoding"
        I
        J
    end
    
    style A fill:#ffecb3
    style G fill:#c8e6c9
    style H fill:#c8e6c9
    style J fill:#e1f5fe
```

**Usage:** How molecular graphs are converted to transformer-compatible sequences.

---

## Diagram 6: Multi-Task Prediction Architecture

```mermaid
graph TD
    A[Final Hidden States] --> B[Context Token Extraction<br/>CLS Token]
    B --> C[Dense Layer<br/>hidden_size → hidden_size]
    C --> D[Activation Function<br/>Tanh]
    D --> E[Stable Dropout]
    
    E --> F[Task Head 1<br/>Solubility]
    E --> G[Task Head 2<br/>Toxicity]
    E --> H[Task Head 3<br/>LogP]
    E --> I[Task Head N<br/>Other Properties]
    
    F --> J[Regression Output]
    G --> K[Classification Output]
    H --> L[Regression Output]
    I --> M[Various Output Types]
    
    subgraph "Shared Processing"
        B
        C
        D
        E
    end
    
    subgraph "Task-Specific Heads"
        F
        G
        H
        I
    end
    
    subgraph "Outputs"
        J
        K
        L
        M
    end
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style J fill:#c8e6c9
    style K fill:#c8e6c9
    style L fill:#c8e6c9
    style M fill:#c8e6c9
```

**Usage:** Multi-task prediction architecture showing how different molecular properties are predicted.

---

## Diagram 7: Attention Mechanism Detail

```mermaid
graph TD
    A[Input Sequence] --> B[Query Projection]
    A --> C[Key Projection]
    A --> D[Value Projection]
    
    B --> E[Content Queries]
    C --> F[Content Keys]
    D --> G[Values]
    
    H[Relative Position<br/>Embeddings] --> I[Position Query Projection]
    H --> J[Position Key Projection]
    
    I --> K[Position Queries]
    J --> L[Position Keys]
    
    E --> M[Content-to-Content<br/>Attention]
    E --> N[Content-to-Position<br/>Attention]
    K --> O[Position-to-Content<br/>Attention]
    K --> P[Position-to-Position<br/>Attention]
    
    F --> M
    L --> N
    F --> O
    L --> P
    
    M --> Q[Sum All Attention Types]
    N --> Q
    O --> Q
    P --> Q
    
    Q --> R[Masked Softmax<br/>XSoftmax]
    R --> S[Attention Weights]
    S --> T[Weighted Sum with Values]
    G --> T
    T --> U[Output Representation]
    
    subgraph "Standard Projections"
        B
        C
        D
    end
    
    subgraph "Position Projections"
        I
        J
    end
    
    subgraph "Attention Types"
        M
        N
        O
        P
    end
    
    style A fill:#ffecb3
    style H fill:#ffecb3
    style U fill:#c8e6c9
```

**Usage:** Detailed breakdown of the disentangled attention mechanism showing all components.

---

## Diagram 8: Training Pipeline

```mermaid
graph TD
    A[Molecular Datasets] --> B[Data Loading<br/>PyTorch Geometric]
    B --> C[Batch Processing<br/>to_dense_batch]
    C --> D[MolE Model<br/>Forward Pass]
    
    D --> E[Task-Specific<br/>Loss Computation]
    E --> F[Multi-Task<br/>Loss Aggregation]
    F --> G[Backpropagation]
    
    G --> H[Gradient Clipping]
    H --> I[Optimizer Step<br/>AdamW]
    I --> J[Learning Rate<br/>Scheduling]
    
    J --> K[Validation<br/>Evaluation]
    K --> L[Metric Computation]
    L --> M[Checkpointing]
    
    subgraph "Data Processing"
        A
        B
        C
    end
    
    subgraph "Model Training"
        D
        E
        F
        G
    end
    
    subgraph "Optimization"
        H
        I
        J
    end
    
    subgraph "Evaluation"
        K
        L
        M
    end
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style M fill:#c8e6c9
```

**Usage:** Complete training pipeline showing data processing, model training, and evaluation.

---

## Diagram 9: Model Scalability

```mermaid
graph TD
    A[Model Configuration] --> B[Small Model<br/>100M Parameters]
    A --> C[Medium Model<br/>300M Parameters]
    A --> D[Large Model<br/>768M Parameters]
    A --> E[XL Model<br/>1.2B Parameters]
    
    B --> F[Speed: 5000 mol/sec<br/>Memory: 8GB<br/>Accuracy: 0.85]
    C --> G[Speed: 3000 mol/sec<br/>Memory: 16GB<br/>Accuracy: 0.88]
    D --> H[Speed: 1500 mol/sec<br/>Memory: 32GB<br/>Accuracy: 0.91]
    E --> I[Speed: 800 mol/sec<br/>Memory: 64GB<br/>Accuracy: 0.93]
    
    subgraph "Model Sizes"
        B
        C
        D
        E
    end
    
    subgraph "Performance Metrics"
        F
        G
        H
        I
    end
    
    style A fill:#ffecb3
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
    style I fill:#c8e6c9
```

**Usage:** Model scalability showing performance trade-offs for different model sizes.

---

## How to Use These Diagrams

### 1. **Online Mermaid Editors**
- [Mermaid Live Editor](https://mermaid.live/)
- [Mermaid Chart](https://www.mermaidchart.com/)
- Copy and paste the code to generate SVG/PNG images

### 2. **Local Generation**
- Install Mermaid CLI: `npm install -g @mermaid-js/mermaid-cli`
- Save diagram code to `.mmd` files
- Generate images: `mmdc -i diagram.mmd -o diagram.png`

### 3. **Integration with Presentation Tools**
- Generate high-resolution PNG or SVG files
- Import into PowerPoint, Google Slides, or Keynote
- Maintain consistent styling with presentation theme

### 4. **Customization**
- Modify colors to match your presentation theme
- Adjust node sizes and text for readability
- Add or remove elements based on audience needs

## Color Scheme Reference

- **Light Blue (#e1f5fe)**: Input/Output components
- **Light Purple (#f3e5f5)**: Processing components
- **Light Green (#c8e6c9)**: Final outputs
- **Light Orange (#ffecb3)**: Configuration/Setup
- **Light Yellow (#fff3e0)**: Intermediate processing

These diagrams provide comprehensive visual support for your MolE presentation, covering all major architectural components and processes.