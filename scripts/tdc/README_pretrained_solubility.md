# Solubility Prediction with Pretrained MolE Transformer

This repository contains a fine-tuning approach for molecular solubility prediction using a pretrained MolE (Molecular Environment) Transformer model. The model leverages transfer learning by loading weights from a cross-environment MLM pretraining task and fine-tuning them for solubility regression.

## 🚀 Key Features

### Model Architecture
- **12-layer DeBERTa Encoder** with disentangled attention
- **12 attention heads** with 768 hidden dimensions
- **Pretrained on GuacaMol dataset** (~1.6M molecules)
- **Cross-environment MLM task**: Radius 0 → Radius 1 functional environments
- **Fine-tuned solubility head** with GELU activations
- **Global average pooling** over sequence dimension

### Transfer Learning Benefits
- **Leverages pretrained molecular representations**
- **Faster convergence** compared to training from scratch
- **Better generalization** through pretrained knowledge
- **Reduced data requirements** for fine-tuning
- **Domain-specific molecular understanding**

## 📁 File Structure

```
scripts/
├── solubility_pretrained_transformer.py  # Main module with pretrained model
├── run_solubility_pretrained.py         # Execution script
├── README_pretrained_solubility.md      # This file
└── requirements_transformer.txt         # Dependencies (shared)
```

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_transformer.txt
   ```

2. **Ensure MolE package is available:**
   ```bash
   # The script automatically adds the project root to Python path
   # Make sure you're in the mole_public directory
   ```

3. **Verify checkpoint availability:**
   ```bash
   # Check if the pretrained checkpoint exists
   ls /home/mschauperl/mole_public/outputs/guacamol_crossenv_mlm/guacamol_r0_to_r1_functional_t4_optimized/checkpoints/epoch=06-step=00209.ckpt
   ```

## 🏃‍♂️ Quick Start

### Run Complete Pipeline

```bash
python run_solubility_pretrained.py
```

### Basic Usage

```python
from solubility_pretrained_transformer import (
    load_solubility_data,
    load_pretrained_model,
    create_solubility_datasets,
    train_fine_tuned_model,
    evaluate_fine_tuned_model
)

# Load data
train_df, test_df = load_solubility_data()

# Load pretrained model
fine_tuned_model, input_vocab, target_vocab = load_pretrained_model(
    checkpoint_path="path/to/checkpoint.ckpt",
    input_vocab_path="path/to/input_vocab.pkl",
    target_vocab_path="path/to/target_vocab.pkl"
)

# Create datasets
train_dataset, test_dataset, scaler, data_module = create_solubility_datasets(
    train_subset, test_subset,
    input_vocab_path="path/to/input_vocab.pkl",
    target_vocab_path="path/to/target_vocab.pkl"
)

# Train fine-tuned model
model = train_fine_tuned_model(
    fine_tuned_model, train_dataset, test_dataset, data_module
)

# Evaluate
results = evaluate_fine_tuned_model(
    model, train_dataset, test_dataset, scaler, data_module,
    model_name="PretrainedMolESolubility"
)
```

## 🧠 Model Components

### PretrainedMolESolubilityModel
- **Encoder**: 12-layer DeBERTa with disentangled attention
- **Frozen initially**: Encoder weights are frozen during initial fine-tuning
- **Gradual unfreezing**: Option to unfreeze encoder layers progressively
- **Solubility head**: MLP (768 → 384 → 192 → 1) with GELU activations

### SolubilityPredictionHead
- **Input**: Hidden states from transformer encoder
- **Pooling**: Masked global average pooling
- **Architecture**: 3-layer MLP with dropout
- **Activation**: GELU for smooth gradients
- **Output**: Single solubility value

### Training Strategy
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
- `best_finetuned_solubility_model.pth`: Fine-tuned model weights

### Prediction Files
- `predictions_pretrainedmolesolubility_train_YYYYMMDD_HHMMSS.csv`
- `predictions_pretrainedmolesolubility_test_YYYYMMDD_HHMMSS.csv`

Each CSV contains:
- `smiles`: Original SMILES string
- `predicted_solubility`: Model prediction
- `actual_solubility`: Ground truth value
- `absolute_error`: Absolute prediction error
- `squared_error`: Squared prediction error

## 🔧 Configuration

### Model Hyperparameters
```python
# Pretrained model configuration
deberta_config = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "dropout": 0.1,
    "relative_attention": True,
    "max_relative_positions": 128,
}

# Fine-tuning configuration
fine_tune_config = {
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10,
}
```

### Vocabulary Configuration
- **Input vocabulary**: Radius 0 structural atom environments
- **Target vocabulary**: Radius 1 functional atom environments
- **Max sequence length**: 256 tokens
- **Special tokens**: CLS, SEP, PAD, MASK

## 🎯 Advantages of Transfer Learning

### 1. **Pretrained Knowledge**
- Model understands molecular structure patterns
- Learned representations from 1.6M GuacaMol molecules
- Cross-environment understanding (structural → functional)

### 2. **Faster Training**
- Reduced training time compared to from-scratch training
- Better initialization leads to faster convergence
- Fewer epochs required for good performance

### 3. **Better Generalization**
- Pretrained representations are more robust
- Better handling of molecular diversity
- Improved performance on small datasets

### 4. **Domain Expertise**
- Model understands molecular environments
- Knowledge of atom-atom interactions
- Functional group recognition

## 🔬 Pretraining Details

### Cross-Environment MLM Task
- **Input**: Radius 0 structural atom environments
- **Target**: Radius 1 functional atom environments
- **Task**: Predict richer functional representations from simpler structural ones
- **Dataset**: GuacaMol v1 (~1.6M molecules)
- **Architecture**: DeBERTa with disentangled attention

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning rate**: 1e-4 with warmup
- **Batch size**: 32 (effective 512 with gradient accumulation)
- **Precision**: Mixed precision (16-bit)
- **Hardware**: Tesla T4 GPU optimized

## 🚨 Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   ```bash
   # Verify checkpoint path
   ls /home/mschauperl/mole_public/outputs/guacamol_crossenv_mlm/guacamol_r0_to_r1_functional_t4_optimized/checkpoints/
   ```

2. **Vocabulary Files Missing**
   ```bash
   # Check vocabulary files
   ls mole/data/vocabularies/
   ```

3. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 8  # or smaller
   ```

4. **Poor Fine-tuning Performance**
   ```python
   # Try different learning rates
   learning_rate = 5e-5  # or 2e-4
   
   # Unfreeze encoder layers gradually
   model._unfreeze_encoder()
   ```

### Performance Tips

1. **Gradual Unfreezing**: Start with frozen encoder, then unfreeze gradually
2. **Learning Rate**: Use lower learning rates for fine-tuning (1e-4 to 5e-5)
3. **Batch Size**: Smaller batches for better gradient estimates
4. **Regularization**: Adjust dropout and weight decay
5. **Early Stopping**: Monitor validation loss to prevent overfitting

## 📈 Future Improvements

### Potential Enhancements
1. **Gradual Layer Unfreezing**: Unfreeze encoder layers progressively
2. **Multi-task Learning**: Predict multiple properties simultaneously
3. **Attention Visualization**: Analyze attention patterns for interpretability
4. **Ensemble Methods**: Combine multiple fine-tuned models
5. **Advanced Tokenization**: Use learned tokenization strategies

### Research Directions
1. **Domain Adaptation**: Adapt to specific molecular domains
2. **Continual Learning**: Update model with new data
3. **Interpretability**: Explain model predictions
4. **Uncertainty Quantification**: Provide prediction confidence
5. **Active Learning**: Select most informative samples

## 📚 References

- Vaswani, A., et al. "Attention is all you need." NeurIPS 2017
- He, P., et al. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR 2021
- Brown, N., et al. "GuacaMol: Benchmarking Models for de Novo Molecular Design." J. Chem. Inf. Model. 2019
- Chithrananda, S., et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." arXiv 2020

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 