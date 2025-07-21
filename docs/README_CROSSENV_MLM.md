# Cross-Environment MLM for MolE: Implementation Guide

This implementation provides a complete system for training MolE models using **cross-environment** masked language modeling, where the model learns to predict radius 1 functional atom environments from radius 0 structural atom environments.

## 🎯 **Overview**

### What is Cross-Environment MLM?

Traditional MLM uses the same vocabulary for input and prediction targets. Cross-environment MLM uses **different atom environment representations**:

- **Input**: Radius 0 structural atom environments (simple, local features)
- **Target**: Radius 1 functional atom environments (richer, functional features)

This approach enables the model to learn mappings from simple structural patterns to complex functional representations, potentially improving chemical understanding.

### Key Innovation

```
Radius 0 Structural    →    Model    →    Radius 1 Functional
     Input                              Prediction Target

[Simple local patterns] → [Transformer] → [Rich functional patterns]
```

## 📁 **Implementation Structure**

```
mole/
├── data/
│   ├── crossenv_dataset.py      # Cross-environment dataset
│   └── crossenv_datamodule.py   # Lightning data module
├── models/
│   └── crossenv_mlm.py          # Cross-environment MLM model
└── cli/
    └── train_crossenv_mlm.py    # Training script

scripts/
├── run_crossenv_training.py     # Convenience training script
└── create_sample_data.py        # Sample data generation

examples/
└── crossenv_mlm_training_demo.ipynb  # Demo notebook
```

## 🚀 **Quick Start**

### 1. **Create Sample Data**

```bash
python scripts/create_sample_data.py
```

This creates:
- `data/sample_molecules.txt` - Complete dataset  
- `data/train_molecules.txt` - Training set
- `data/val_molecules.txt` - Validation set
- `data/test_molecules.txt` - Test set

### 2. **Run Training**

```bash
python scripts/run_crossenv_training.py
```

Or use the direct training script:

```bash
python mole/cli/train_crossenv_mlm.py \
    --train_data data/train_molecules.txt \
    --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
    --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
    --output_dir outputs/crossenv_experiment \
    --batch_size 32 \
    --max_epochs 50
```

### 3. **Monitor Training**

```bash
tensorboard --logdir outputs/crossenv_experiment/logs
```

## 📊 **Model Architecture**

### Core Components

1. **CrossEnvMolDataset**: Handles data loading and MLM preprocessing
2. **CrossEnvMLMModel**: Core model with cross-vocabulary prediction
3. **CrossEnvMLM**: PyTorch Lightning wrapper with training logic
4. **CrossEnvDataModule**: Data loading and batching

### Model Flow

```
SMILES → Atom Environments → Tokenization → Masking → Model → Cross-Vocab Prediction
```

1. **Input Processing**: Extract radius 0 structural environments
2. **Target Generation**: Extract radius 1 functional environments  
3. **MLM Masking**: Randomly mask input tokens (15% probability)
4. **Cross Prediction**: Predict functional tokens from structural inputs
5. **Loss Calculation**: Cross-entropy loss on masked positions only

## 🔧 **Configuration Options**

### Environment Configuration

```python
# Input environments (structural, radius 0)
input_radius = 0
input_use_features = False  # Structural only

# Target environments (functional, radius 1)  
target_radius = 1
target_use_features = True  # Include features
```

### Model Configuration

```python
model_config = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "dropout": 0.1,
}
```

### Training Configuration

```python
training_config = {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "max_epochs": 100,
    "warmup_steps": 10000,
    "weight_decay": 0.01,
}
```

### MLM Configuration

```python
mlm_config = {
    "mask_prob": 0.15,      # 15% of tokens masked
    "replace_prob": 0.8,    # 80% replaced with [MASK]
    "random_prob": 0.1,     # 10% replaced with random
}
```

## 📈 **Metrics and Evaluation**

### Training Metrics

- **Loss**: Cross-entropy loss on masked positions
- **Accuracy**: Prediction accuracy on masked tokens
- **Perplexity**: Model uncertainty measure

### Evaluation

```python
# Get embeddings for downstream tasks
embeddings = model.encode(smiles_list)

# Evaluate on molecular property prediction
property_model = train_property_predictor(embeddings, properties)
```

## 🛠 **Advanced Usage**

### Custom Vocabularies

```python
# Create custom vocabularies
python mole/data/create_atomenv_atlas.py \
    --smiles_file your_molecules.txt \
    --output_dir custom_vocabs/ \
    --radius 0 --use_features False  # Input vocab
    
python mole/data/create_atomenv_atlas.py \
    --smiles_file your_molecules.txt \
    --output_dir custom_vocabs/ \
    --radius 1 --use_features True   # Target vocab
```

### Different Radius Combinations

```python
# Experiment with different combinations
configurations = [
    (0, False) → (1, True),   # Structural → Functional
    (0, False) → (2, True),   # Local → Extended functional
    (1, False) → (2, False),  # Medium → Extended structural
]
```

### Multi-Task Learning

```python
# Combine with property prediction
class MultiTaskCrossEnvMLM(CrossEnvMLM):
    def __init__(self, *args, property_heads=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_heads = property_heads
        
    def training_step(self, batch, batch_idx):
        # MLM loss
        mlm_outputs = super().training_step(batch, batch_idx)
        
        # Property prediction loss
        prop_loss = self.compute_property_loss(batch)
        
        # Combined loss
        total_loss = mlm_outputs["loss"] + 0.5 * prop_loss
        return {"loss": total_loss, "mlm_loss": mlm_outputs["loss"], "prop_loss": prop_loss}
```

## 📚 **API Reference**

### CrossEnvMolDataset

```python
dataset = CrossEnvMolDataset(
    smiles=smiles_series,
    input_vocab_path="input_vocab.pkl",
    target_vocab_path="target_vocab.pkl", 
    input_radius=0,
    target_radius=1,
    input_use_features=False,
    target_use_features=True,
    mask_prob=0.15,
    cls_token=True,
)
```

### CrossEnvMLMModel

```python
model = CrossEnvMLMModel(
    deberta_config=config,
    input_vocab_size=207,
    target_vocab_size=156,
    dropout=0.1,
    label_smoothing=0.0,
)
```

### Training Script Arguments

```bash
python mole/cli/train_crossenv_mlm.py \
    --train_data TRAIN_DATA \           # Training SMILES data
    --input_vocab INPUT_VOCAB \         # Input vocabulary path
    --target_vocab TARGET_VOCAB \       # Target vocabulary path
    --hidden_size 768 \                 # Model hidden size
    --num_hidden_layers 12 \            # Number of transformer layers
    --batch_size 32 \                   # Batch size
    --learning_rate 5e-5 \              # Learning rate
    --max_epochs 100 \                  # Maximum epochs
    --output_dir outputs/ \             # Output directory
    --gpus 1                            # Number of GPUs
```

## 🔍 **Troubleshooting**

### Common Issues

1. **Vocabulary Not Found**
   ```
   FileNotFoundError: vocabulary file not found
   ```
   **Solution**: Ensure vocabulary files exist or create them using `create_atomenv_atlas.py`

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or model size:
   ```bash
   --batch_size 16 --hidden_size 512 --num_hidden_layers 6
   ```

3. **No Masked Tokens**
   ```
   Warning: No masked tokens found
   ```
   **Solution**: Check masking probability and sequence lengths:
   ```bash
   --mask_prob 0.15 --max_length 128
   ```

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'mole'
   ```
   **Solution**: Add project root to Python path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/mole_public')
   ```

### Performance Tips

1. **Data Loading**: Use multiple workers but avoid too many in notebooks
   ```bash
   --num_workers 4  # Good for training scripts
   --num_workers 0  # Use in notebooks
   ```

2. **Mixed Precision**: Use for faster training on modern GPUs
   ```bash
   --precision bf16  # On A100/H100
   --precision 16    # On V100/T4
   ```

3. **Gradient Accumulation**: Simulate larger batch sizes
   ```bash
   --batch_size 16 --accumulate_grad_batches 2  # Effective batch size: 32
   ```

## 📖 **Examples**

### Basic Training

```python
from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.models.crossenv_mlm import CrossEnvMLMModel, CrossEnvMLM

# Setup data
data_module = CrossEnvDataModule(
    train_data="molecules.txt",
    input_vocab_path="input_vocab.pkl",
    target_vocab_path="target_vocab.pkl",
)

# Create model
model = CrossEnvMLM(
    model=CrossEnvMLMModel(...),
    optimizer_cfg=OptimizerConfig(...),
)

# Train
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

### Custom Evaluation

```python
# Evaluate trained model
model.eval()
results = trainer.test(model, data_module)

# Extract embeddings
embeddings = []
with torch.no_grad():
    for batch in data_module.test_dataloader():
        outputs = model(batch.x, batch.input_mask)
        embeddings.append(outputs["hidden_states"][:, 0])  # CLS embeddings

embeddings = torch.cat(embeddings, dim=0)
```

## 🔬 **Research Applications**

### Potential Research Directions

1. **Representation Learning**: Study what the model learns about chemical structure
2. **Transfer Learning**: Use pretrained cross-env models for downstream tasks  
3. **Chemical Space Exploration**: Visualize learned molecular embeddings
4. **Property Prediction**: Compare with standard pretraining approaches
5. **Few-Shot Learning**: Evaluate on small property prediction datasets

### Comparison Studies

```python
# Compare different pretraining strategies
strategies = [
    "standard_mlm",           # Same vocab for input/target
    "crossenv_r0_to_r1",      # Radius 0 → 1
    "crossenv_struct_to_func", # Structural → Functional
    "crossenv_r1_to_r2",      # Extended environments
]

for strategy in strategies:
    model = train_model(strategy)
    evaluate_downstream_tasks(model)
```

## 📄 **Citation**

If you use this implementation in your research, please cite:

```bibtex
@misc{crossenv_mlm_mole,
  title={Cross-Environment Masked Language Modeling for Molecular Representation Learning},
  author={Your Name},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/your-repo/mole_public}
}
```

## 🤝 **Contributing**

Contributions are welcome! Areas for improvement:

- [ ] Additional environment combinations
- [ ] Multi-task learning extensions
- [ ] Evaluation metrics and benchmarks
- [ ] Memory optimization
- [ ] Distributed training support

## 📞 **Support**

For questions or issues:

1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Check existing MolE documentation
4. Review the demo notebook for working examples

---

**Happy training! 🧬🤖** 