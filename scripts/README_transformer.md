# Molecular Solubility Prediction with Transformer

This repository contains a Transformer-based model for predicting molecular solubility from SMILES strings. The model replaces the previous DeepChem GCN implementation with a modern Transformer architecture specifically designed for molecular property prediction.

## 🚀 Key Features

### Model Architecture
- **6-layer Transformer Encoder** with 8 attention heads
- **Custom SMILES Tokenizer** with molecular-aware vocabulary
- **Positional Encoding** for sequence awareness
- **Dedicated Solubility Head** with multi-layer perceptron
- **Global Average Pooling** over sequence dimension

### Training Features
- **AdamW Optimizer** with weight decay regularization
- **Learning Rate Scheduling** with ReduceLROnPlateau
- **Early Stopping** to prevent overfitting
- **Gradient Clipping** for training stability
- **RobustScaler** for outlier handling

## 📁 File Structure

```
scripts/
├── solubility_gcn.py              # Main module with Transformer implementation
├── run_solubility_transformer.py  # Execution script
├── requirements_transformer.txt   # Dependencies
└── README_transformer.md         # This file
```

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_transformer.txt
   ```

2. **For GPU acceleration (recommended):**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from solubility_gcn import (
    load_solubility_data,
    preprocess_solubility_data,
    create_transformer_datasets,
    train_transformer_model,
    evaluate_model
)

# Load data
train_df, test_df = load_solubility_data()

# Preprocess
train_subset, test_subset = preprocess_solubility_data(train_df, test_df)

# Create datasets
train_dataset, test_dataset, scaler, tokenizer = create_transformer_datasets(
    train_subset, test_subset
)

# Train model
model = train_transformer_model(
    train_dataset, test_dataset, tokenizer,
    epochs=50, batch_size=32
)

# Evaluate
results = evaluate_model(
    model, train_dataset, test_dataset, scaler, tokenizer,
    model_name="MolecularTransformer"
)
```

### Run Complete Pipeline

```bash
python run_solubility_transformer.py
```

## 🧠 Model Components

### SMILESTokenizer
- **Vocabulary**: 50+ tokens including atoms, bonds, and special characters
- **Tokenization**: Character-level with SMILES-aware rules
- **Padding**: Fixed-length sequences (default: 128 tokens)
- **Special Tokens**: `<START>`, `<END>`, `<PAD>`, `<UNK>`

### MolecularTransformer
- **Embedding Layer**: Token embeddings (vocab_size → 256)
- **Positional Encoding**: Sinusoidal positional encoding
- **Transformer Encoder**: 6 layers with 8 attention heads
- **Pooling**: Masked global average pooling
- **Solubility Head**: MLP (256 → 128 → 64 → 1)

### Training Pipeline
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Patience=10 epochs
- **Gradient Clipping**: max_norm=1.0

## 📊 Performance Metrics

The model outputs comprehensive evaluation metrics:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **R²**: Coefficient of determination
- **Sample Predictions**: First 10 test molecules

## 💾 Output Files

### Model Files
- `best_transformer_model.pth`: Trained model weights

### Prediction Files
- `predictions_moleculartransformer_train_YYYYMMDD_HHMMSS.csv`
- `predictions_moleculartransformer_test_YYYYMMDD_HHMMSS.csv`

Each CSV contains:
- `smiles`: Original SMILES string
- `predicted_solubility`: Model prediction
- `actual_solubility`: Ground truth value
- `absolute_error`: Absolute prediction error
- `squared_error`: Squared prediction error

## 🔧 Configuration

### Model Hyperparameters
```python
# In train_transformer_model()
model = MolecularTransformer(
    vocab_size=tokenizer.vocab_size,  # ~50
    d_model=256,                      # Hidden dimension
    nhead=8,                          # Attention heads
    num_layers=6,                     # Transformer layers
    dim_feedforward=1024,             # Feed-forward dimension
    dropout=0.1,                      # Dropout rate
    max_length=128                    # Sequence length
)
```

### Training Parameters
```python
# Training configuration
epochs=50
batch_size=32
learning_rate=1e-4
weight_decay=1e-5
patience=10  # Early stopping
```

## 🎯 Advantages over GCN

### 1. **Sequence Modeling**
- Direct SMILES processing without graph conversion
- Captures sequential dependencies in molecular representation
- Positional encoding for spatial awareness

### 2. **Attention Mechanism**
- Self-attention captures long-range dependencies
- Multi-head attention for diverse feature learning
- Interpretable attention weights

### 3. **Scalability**
- Parallel processing of sequences
- Efficient training on large datasets
- GPU-optimized implementation

### 4. **Flexibility**
- Easy to extend to other molecular properties
- Configurable architecture
- Transfer learning capabilities

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 16  # or smaller
   ```

2. **Slow Training**
   ```python
   # Use GPU if available
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

3. **Poor Performance**
   ```python
   # Increase model capacity
   d_model = 512
   num_layers = 8
   ```

### Performance Tips

1. **Data Quality**: Ensure clean SMILES strings
2. **Hyperparameter Tuning**: Experiment with learning rates and model sizes
3. **Regularization**: Adjust dropout and weight decay
4. **Early Stopping**: Monitor validation loss to prevent overfitting

## 📈 Future Improvements

### Potential Enhancements
1. **Pre-trained Models**: Use molecular pre-trained transformers
2. **Multi-task Learning**: Predict multiple properties simultaneously
3. **Attention Visualization**: Analyze attention patterns
4. **Ensemble Methods**: Combine multiple transformer models
5. **Advanced Tokenization**: BPE or SentencePiece tokenization

### Research Directions
1. **Molecular Attention**: Domain-specific attention mechanisms
2. **Graph-Transformer Hybrid**: Combine graph and sequence modeling
3. **Contrastive Learning**: Self-supervised pre-training
4. **Interpretability**: Explainable AI for molecular predictions

## 📚 References

- Vaswani, A., et al. "Attention is all you need." NeurIPS 2017
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019
- Chithrananda, S., et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." arXiv 2020

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 