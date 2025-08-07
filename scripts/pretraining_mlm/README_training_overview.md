# MolE Training Scripts - Complete Overview

This directory contains a comprehensive set of training scripts for the MolE (Molecular Environment) model across different tasks and hardware configurations. All scripts now follow consistent professional standards.

## 🎯 **Script Categories**

### **1. Cross-Environment MLM Training**
- **Purpose**: Train models to predict radius-1 functional environments from radius-0 structural environments
- **Primary Script**: `mole/cli/train_crossenv_mlm.py`

#### Available Configurations:
- `run_crossenv_training_t4.py` - Tesla T4 optimized

### **2. Unsupervised Pretraining**
- **Purpose**: Multi-task training on GuacaMol dataset (MLM + ClogP + Molecular Weight)
- **Primary Script**: `mole/cli/train_unsupervised_pretraining.py`

#### Available Configurations:
- `run_unsupervised_pretraining_t4.py` - Tesla T4 optimized
- `run_unsupervised_pretraining_a100_2gpu.py` - A100 2-GPU optimized

### **3. ChemBL Training**
- **Purpose**: MLM + ChemBL classification or ChemBL classification only
- **Primary Script**: `mole/cli/train_chembl_mlm.py`

#### Available Configurations:
- `run_chembl_training_t4.py` - T4 optimized (MLM + Classification)
- `run_chembl_only_t4.py` - T4 optimized (Classification only)
- `run_chembl_training_test.py` - Rapid testing (1000 samples)
- `run_chembl_finetune_t4.py` - Fine-tuning from pretrained models

### **4. Shared Configuration System**
- **Base Module**: `chembl_config_base.py`
- **Purpose**: Reduces code redundancy and provides consistent configuration management

## 🚀 **Unified Features (All Scripts)**

### **Configuration Management**
All training scripts automatically generate:
- **`model_config.json`**: Complete model architecture and hyperparameters
- **`training_args.json`**: All command-line arguments used
- **`training_summary.txt`**: Human-readable training summary

### **Checkpoint Management**
- **Top 5 checkpoints** saved automatically
- **Consistent naming**: `epoch=XX-step=XXXXX-val_loss=X.XXXX.ckpt`
- **Step tracking** included in filenames
- **No redundant checkpoints** (`save_last=False`)

### **Directory Structure**
```
outputs/
└── model_name/
    ├── checkpoints/
    │   ├── epoch=01-step=00250-val_loss=0.234.ckpt
    │   ├── epoch=02-step=00500-val_loss=0.123.ckpt
    │   ├── epoch=03-step=00750-val_loss=0.098.ckpt  ← Best
    │   ├── epoch=04-step=01000-val_loss=0.145.ckpt
    │   └── epoch=05-step=01250-val_loss=0.112.ckpt
    ├── logs/
    ├── model_config.json
    ├── training_args.json
    └── training_summary.txt
```

## 📊 **Hardware Optimization Matrix**

| Script | GPU | Memory | Batch Size | Precision | Accumulation |
|--------|-----|--------|------------|-----------|--------------|
| Cross-env T4 | Tesla T4 | 16GB | 32 | 16-bit | 16 |
| Unsupervised T4 | Tesla T4 | 16GB | 32 | 16-bit | 16 |
| Unsupervised A100 | A100 x2 | 40GB | 128 | bf16 | 48 |
| ChemBL T4 | Tesla T4 | 16GB | 4 | 16-bit | 16 |
| ChemBL Only T4 | Tesla T4 | 16GB | 8 | 16-bit | 8 |
| ChemBL Test | Any | Any | 4 | 32-bit | 4 |

## 🔧 **Quick Start Guide**

### **Standard Training**
```bash
# Cross-environment MLM
python run_crossenv_training_t4.py

# Unsupervised pretraining (GuacaMol)
python run_unsupervised_pretraining_t4.py

# ChemBL MLM + Classification
python run_chembl_training_t4.py

# ChemBL Classification only
python run_chembl_only_t4.py
```

### **High-Performance Training**
```bash
# A100 2-GPU unsupervised pretraining
python run_unsupervised_pretraining_a100_2gpu.py --gpus 2
```

### **Rapid Testing**
```bash
# Quick ChemBL test (1000 samples)
python run_chembl_training_test.py
```

### **Fine-Tuning**
```bash
# Fine-tune with frozen encoder
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/model/checkpoints/best_model.ckpt \
    --freeze_encoder

# Continue pretraining
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/model/checkpoints/best_model.ckpt
```

## 🎛️ **Configuration Classes**

### **Base Classes**
- **`ChemBLConfig`**: Common parameters for all ChemBL scripts
- **`T4Config`**: Tesla T4 optimized settings
- **`T4OnlyConfig`**: Tesla T4 ChemBL-only settings
- **`TestConfig`**: Minimal rapid testing settings

### **Fine-Tuning Classes**
- **`T4FineTuneConfig`**: T4 optimized for fine-tuning
- **`T4OnlyFineTuneConfig`**: T4 ChemBL-only fine-tuning

### **Usage Example**
```python
from chembl_config_base import T4Config, run_chembl_training

# Custom configuration
overrides = {"--learning_rate": "5e-5", "--batch_size": "8"}

run_chembl_training(
    config_class=T4Config,
    title="Custom ChemBL Training",
    overrides=overrides
)
```

## 📈 **Training Monitoring**

### **Automatic Logging**
- **TensorBoard logs** in `outputs/model_name/logs/`
- **Console logging** with progress bars
- **Validation metrics** logged every N steps
- **Learning rate monitoring** per step

### **Early Stopping**
- **Automatic early stopping** based on validation loss
- **Configurable patience** (default: 3-5 epochs)
- **Best model preservation**

## 🔄 **Pretrained Model Support**

### **Loading Pretrained Models**
```bash
# Fine-tune any model
python run_chembl_finetune_t4.py \
    --pretrained_path /path/to/checkpoint.ckpt \
    --freeze_encoder  # Optional: freeze encoder layers
```

### **Transfer Learning**
- **Cross-task transfer**: Use cross-env models for ChemBL
- **Domain adaptation**: Fine-tune on specific datasets
- **Encoder freezing**: Prevent catastrophic forgetting

## 📁 **File Organization**

```
scripts/pretraining_mlm/
├── chembl_config_base.py              # Shared configuration system
├── run_crossenv_training_t4.py        # Cross-environment MLM
├── run_unsupervised_pretraining_t4.py # Unsupervised (T4)
├── run_unsupervised_pretraining_a100_2gpu.py # Unsupervised (A100)
├── run_chembl_training_t4.py          # ChemBL MLM + Classification
├── run_chembl_only_t4.py              # ChemBL Classification only
├── run_chembl_training_test.py        # ChemBL rapid testing
├── run_chembl_finetune_t4.py          # ChemBL fine-tuning
├── README_pretrained.md               # Pretrained model guide
├── README_training_overview.md        # This file
├── CHANGELOG_unsupervised_updates.md  # Unsupervised updates
└── CHANGELOG_chembl_updates.md        # ChemBL updates
```

## ✅ **Quality Standards**

All scripts follow these professional standards:

### **Reproducibility**
- ✅ Complete configuration saving
- ✅ Deterministic training (seeded)
- ✅ Version tracking capability

### **Monitoring**
- ✅ Detailed logging
- ✅ Progress tracking
- ✅ Validation monitoring
- ✅ Early stopping

### **Efficiency**
- ✅ Hardware-optimized settings
- ✅ Memory-efficient configurations
- ✅ Mixed precision training
- ✅ Gradient accumulation

### **Usability**
- ✅ Clear command-line interfaces
- ✅ Helpful documentation
- ✅ Error handling
- ✅ Progress feedback

## 🎉 **Summary**

This training ecosystem provides:

1. **Complete Coverage**: Cross-environment, unsupervised, and supervised learning
2. **Hardware Optimization**: Optimized for Tesla T4, A100, and other GPUs
3. **Professional Quality**: Consistent standards across all scripts
4. **Easy to Use**: Simple command-line interfaces with sensible defaults
5. **Highly Configurable**: Extensible configuration system
6. **Reproducible**: Complete experiment tracking and configuration saving
7. **Transfer Learning**: Full support for pretrained models and fine-tuning

Whether you're doing initial pretraining, fine-tuning for specific tasks, or rapid prototyping, this ecosystem provides the tools you need with professional-grade quality and consistency! 🚀 