# ADMET Property Prediction with Pretrained MolE Transformer

This document provides comprehensive guidance for using the `run_admet_pretrained.py` script to evaluate ADMET properties using pretrained MolE transformer models.

## 🎯 Overview

The `run_admet_pretrained.py` script enables:
- **Fine-tuning pretrained MolE models** for ADMET property prediction
- **Support for all 22 ADMET properties** from the TDC benchmark group
- **Multiple random seeds** for robust statistical evaluation
- **Automatic task type detection** (regression vs classification)
- **Comprehensive evaluation metrics** with TDC benchmark integration
- **Command-line interface** for easy experimentation

## 🚀 Key Features

### **Pretrained Model Integration**
- Loads pretrained MolE checkpoints from cross-environment training
- Smart checkpoint loading with architecture mismatch handling
- Configurable encoder freezing for transfer learning

### **ADMET Property Support**
- **Absorption**: Caco-2, HIA, Solubility, Lipophilicity
- **Distribution**: BBB, PPBR, VDss
- **Metabolism**: CYP enzymes (2D6, 3A4, 2C9) and substrates
- **Excretion**: Half-life, Clearance (microsome, hepatocyte)
- **Toxicity**: hERG, Ames, DILI, LD50

### **Multiple Run Evaluation**
- Configurable random seeds for statistical robustness
- Automatic averaging of metrics across runs
- TDC benchmark group evaluation for standardized comparison
- Detailed prediction export for analysis

### **Flexible Training**
- Automatic task type detection (regression/classification)
- Configurable training parameters (epochs, batch size, learning rate)
- Early stopping and learning rate scheduling
- GPU/CPU support with automatic device detection

## 📋 Available ADMET Properties

| Property | Type | Description |
|----------|------|-------------|
| **Absorption** |
| `caco2_wang` | Regression | Caco-2 permeability |
| `hia_hou` | Classification | Human intestinal absorption |
| `solubility_aqsoldb` | Regression | Aqueous solubility |
| `lipophilicity_astrazeneca` | Regression | Lipophilicity (LogP) |
| **Distribution** |
| `bbb_martins` | Classification | Blood-brain barrier penetration |
| `ppbr_az` | Regression | Plasma protein binding ratio |
| `vdss_lombardo` | Regression | Volume of distribution |
| **Metabolism** |
| `cyp2d6_veith` | Classification | CYP2D6 inhibition |
| `cyp3a4_veith` | Classification | CYP3A4 inhibition |
| `cyp2c9_veith` | Classification | CYP2C9 inhibition |
| `cyp2d6_substrate_carbonmangels` | Classification | CYP2D6 substrate |
| `cyp3a4_substrate_carbonmangels` | Classification | CYP3A4 substrate |
| `cyp2c9_substrate_carbonmangels` | Classification | CYP2C9 substrate |
| **Excretion** |
| `half_life_obach` | Regression | Half-life |
| `clearance_microsome_az` | Regression | Microsomal clearance |
| `clearance_hepatocyte_az` | Regression | Hepatocyte clearance |
| **Toxicity** |
| `herg` | Classification | hERG inhibition |
| `ames` | Classification | Ames mutagenicity |
| `dili` | Classification | Drug-induced liver injury |
| `ld50_zhu` | Regression | Acute toxicity (LD50) |
| **Additional** |
| `bioavailability_ma` | Classification | Oral bioavailability |
| `pgp_broccatelli` | Classification | P-glycoprotein substrate |

## 🔧 Installation & Setup

### **Dependencies**
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn pandas numpy

# TDC for ADMET datasets
pip install PyTDC

# MolE dependencies (already in project)
# - Custom MolE modules from the project
```

### **Pretrained Model Requirements**
- Pretrained MolE checkpoint (`.ckpt` file)
- Input vocabulary (radius 0 structural)
- Target vocabulary (radius 1 functional)

## 💡 Usage Examples

### **1. Basic Property Evaluation**

#### List available properties:
```bash
python scripts/tdc/run_admet_pretrained.py --list-properties
```

#### Evaluate single property with default settings:
```bash
python scripts/tdc/run_admet_pretrained.py --property solubility_aqsoldb
```

### **2. Customized Evaluation**

#### Multiple properties with custom seeds:
```bash
# Solubility prediction with 3 seeds
python scripts/tdc/run_admet_pretrained.py \
  --property solubility_aqsoldb \
  --seeds 1 2 3

# hERG toxicity with extended training
python scripts/tdc/run_admet_pretrained.py \
  --property herg \
  --epochs 50 \
  --learning_rate 5e-5
```

#### Custom model and training parameters:
```bash
python scripts/tdc/run_admet_pretrained.py \
  --property lipophilicity_astrazeneca \
  --checkpoint_path path/to/your/checkpoint.ckpt \
  --epochs 40 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --seeds 1 2 3 4 5
```

### **3. Comprehensive Evaluation**

#### Multiple properties evaluation:
```bash
# Create a script to evaluate multiple properties
properties=("solubility_aqsoldb" "lipophilicity_astrazeneca" "herg" "ames")

for prop in "${properties[@]}"; do
    echo "Evaluating $prop..."
    python scripts/tdc/run_admet_pretrained.py \
      --property $prop \
      --seeds 1 2 3 \
      --epochs 30
done
```

## 📊 Output and Results

### **Terminal Output**
The script provides detailed progress information:

```
🧬 ADMET PROPERTY PREDICTION WITH PRETRAINED MOLECULAR TRANSFORMER
================================================================================
Using device: cuda
GPU: NVIDIA Tesla V100-SXM2-32GB
✅ Found checkpoint: /path/to/checkpoint.ckpt
🎯 Target property: solubility_aqsoldb
🔄 Random seeds: [1, 2, 3]

1. Initializing TDC benchmark group...
Benchmark: Solubility AqSolDB
Train+Val set: 9982 samples
Test set: 1000 samples

2. Loading pretrained model...
Input vocabulary size: 173
Target vocabulary size: 1472
Model configuration: {'hidden_size': 768, 'num_hidden_layers': 12, ...}
Loaded 249 compatible parameters

============================================================
SEED 1 EVALUATION
============================================================
Train set: 8000 samples
Valid set: 1982 samples
Detected task type: regression
Original target stats - Train mean: -2.456, std: 2.134
Scaled target stats - Train mean: 0.000, std: 1.000

3. Training model for seed 1...
Training for 30 epochs...
Epoch 1/30: Train Loss: 0.8234, Val Loss: 0.6789
...
Training completed. Best validation loss: 0.4123

4. Evaluating on test set...
Test Results:
  MAE: 0.721
  RMSE: 1.043
  R²: 0.798
```

### **Generated Files**

#### **Prediction Files**
For each seed, detailed predictions are saved:
- **Format**: `predictions_{property}_{model}_seed{seed}_{timestamp}.csv`
- **Example**: `predictions_solubility_aqsoldb_mole_pretrained_seed1_20241208_143052.csv`

**Regression CSV columns:**
```csv
smiles,predicted_value,actual_value,absolute_error,squared_error
CC(=O)OC1=CC=CC=C1C(=O)O,-1.23,-1.45,0.22,0.048
CCO,-0.89,-0.92,0.03,0.001
...
```

**Classification CSV columns:**
```csv
smiles,predicted_probability,predicted_class,actual_class,correct_prediction
CC(=O)OC1=CC=CC=C1C(=O)O,0.75,1,1,1
CCO,0.23,0,0,1
...
```

### **Summary Results**

#### **Average Metrics Across Seeds**
```
============================================================
AVERAGE RESULTS ACROSS SEEDS
============================================================
MAE: 0.7234 ± 0.0456
RMSE: 1.0421 ± 0.0678
R²: 0.7981 ± 0.0234
```

#### **TDC Benchmark Results**
```
============================================================
TDC BENCHMARK EVALUATION
============================================================
TDC Benchmark Results:
  MAE: 0.7234 ± 0.0456
  RMSE: 1.0421 ± 0.0678
  R²: 0.7981 ± 0.0234
```

## 🏆 Performance Benchmarks

### **Expected Performance Ranges**

| Property Type | Good Performance | Excellent Performance |
|---------------|------------------|----------------------|
| **Regression** | R² > 0.6 | R² > 0.8 |
| **Classification** | AUC-ROC > 0.7 | AUC-ROC > 0.85 |

### **Property-Specific Benchmarks**

| Property | Task Type | Typical R²/AUC | Notes |
|----------|-----------|----------------|--------|
| Solubility | Regression | 0.7-0.85 | Well-studied property |
| Lipophilicity | Regression | 0.8-0.9 | High accuracy expected |
| hERG | Classification | 0.75-0.88 | Critical safety property |
| Ames | Classification | 0.7-0.85 | Mutagenicity prediction |
| BBB | Classification | 0.65-0.8 | Complex permeability |

## ⚙️ Configuration Options

### **Command-Line Arguments**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--property` | str | `solubility_aqsoldb` | ADMET property to evaluate |
| `--list-properties` | flag | - | List all available properties |
| `--checkpoint_path` | str | Pre-configured | Path to pretrained checkpoint |
| `--input_vocab` | str | Pre-configured | Path to input vocabulary |
| `--target_vocab` | str | Pre-configured | Path to target vocabulary |
| `--seeds` | int[] | `[1,2,3,4,5]` | Random seeds for multiple runs |
| `--epochs` | int | `30` | Training epochs |
| `--batch_size` | int | `16` | Batch size |
| `--learning_rate` | float | `1e-4` | Learning rate |

### **Model Architecture**
- **Encoder**: DeBERTa-style transformer from MolE
- **Hidden Size**: 768 dimensions
- **Layers**: 12 transformer layers
- **Attention Heads**: 12 heads
- **Prediction Head**: 3-layer MLP with GELU activation

### **Training Configuration**
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss Functions**: MSE (regression), CrossEntropy (classification)
- **Early Stopping**: Based on validation loss

## 🔬 Advanced Usage

### **Custom Checkpoint Usage**
```bash
# Use your own trained checkpoint
python scripts/tdc/run_admet_pretrained.py \
  --property your_property \
  --checkpoint_path /path/to/your/checkpoint.ckpt \
  --input_vocab /path/to/your/input_vocab.pkl \
  --target_vocab /path/to/your/target_vocab.pkl
```

### **Hyperparameter Optimization**
```bash
# Example hyperparameter sweep
learning_rates=(1e-4 5e-5 2e-4)
batch_sizes=(16 32)

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    echo "Training with lr=$lr, batch_size=$bs"
    python scripts/tdc/run_admet_pretrained.py \
      --property solubility_aqsoldb \
      --learning_rate $lr \
      --batch_size $bs \
      --seeds 1 2 3
  done
done
```

### **Batch Evaluation Script**
```bash
#!/bin/bash
# evaluate_all_admet.sh

properties=(
    "solubility_aqsoldb"
    "lipophilicity_astrazeneca" 
    "herg"
    "ames"
    "bbb_martins"
    "caco2_wang"
)

for prop in "${properties[@]}"; do
    echo "==============================================="
    echo "Evaluating: $prop"
    echo "==============================================="
    
    python scripts/tdc/run_admet_pretrained.py \
      --property $prop \
      --seeds 1 2 3 4 5 \
      --epochs 30 \
      2>&1 | tee "logs/${prop}_evaluation.log"
done
```

## 📈 Results Analysis

### **Python Analysis Script**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('predictions_solubility_aqsoldb_mole_pretrained_seed1_timestamp.csv')

# Analysis
mae = df['absolute_error'].mean()
rmse = np.sqrt(df['squared_error'].mean())
r2 = 1 - df['squared_error'].sum() / ((df['actual_value'] - df['actual_value'].mean())**2).sum()

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['actual_value'], df['predicted_value'], alpha=0.6)
plt.plot([df['actual_value'].min(), df['actual_value'].max()], 
         [df['actual_value'].min(), df['actual_value'].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.hist(df['absolute_error'], bins=30, alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.tight_layout()
plt.savefig('analysis_results.png')
```

## 🐛 Troubleshooting

### **Common Issues**

#### **Issue: "TDC not available"**
**Solution**: Install TDC package
```bash
pip install PyTDC
```

#### **Issue: "Checkpoint not found"**
**Solution**: Verify checkpoint path
```bash
ls -la /path/to/checkpoint.ckpt
```

#### **Issue: "CUDA out of memory"**
**Solutions**:
- Reduce batch size: `--batch_size 8`
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`
- Use gradient accumulation in code

#### **Issue: "Vocabulary file not found"**
**Solution**: Check vocabulary paths
```bash
ls -la mole/data/vocabularies/vocabulary_*.pkl
```

### **Performance Issues**

#### **Low Performance (R² < 0.5)**
**Potential solutions**:
- Increase training epochs: `--epochs 50`
- Adjust learning rate: `--learning_rate 5e-5`
- Check data quality and preprocessing
- Try different random seeds

#### **Training Too Slow**
**Optimization strategies**:
- Increase batch size: `--batch_size 32`
- Use fewer epochs with early stopping
- Enable mixed precision (modify code)
- Use multiple GPUs (modify code)

### **Data Issues**

#### **Task Type Detection Problems**
- Verify label format (0/1 for classification)
- Check for missing values
- Inspect unique label values

#### **Memory Issues**
- Reduce sequence length in dataset class
- Use smaller vocabularies
- Implement gradient checkpointing

## 🔄 Comparison with Original ADMET Benchmark

### **Key Differences**

| Aspect | Original DeepChem | MolE Pretrained |
|--------|-------------------|-----------------|
| **Model Type** | Graph Neural Networks | Transformer |
| **Features** | Molecular graphs | Sequence tokens |
| **Pretraining** | None | Cross-environment MLM |
| **Transfer Learning** | Task-specific training | Fine-tuning approach |

### **Advantages of MolE Approach**
- **Pretrained representations** capture chemical knowledge
- **Transfer learning** from large molecular datasets
- **Sequence-based** approach handles complex structures
- **Consistent architecture** across properties

### **When to Use Each**
- **MolE Pretrained**: New properties, limited data, transfer learning
- **Graph Models**: Graph-specific properties, interpretability needs

## 📚 Additional Resources

### **Related Scripts**
- `run_solubility_pretrained.py` - Solubility-specific implementation
- `admet_benchmark.py` - Original DeepChem comparison
- MolE training scripts in `scripts/pretraining_mlm/`

### **Documentation**
- TDC Documentation: https://tdcommons.ai/
- PyTorch Lightning: https://lightning.ai/
- MolE Architecture: See project documentation

### **Citations**
```bibtex
@article{mole2024,
  title={MolE: Molecular Environment Transformer for Drug Discovery},
  author={Your Name et al.},
  journal={Your Journal},
  year={2024}
}

@article{tdc2021,
  title={Therapeutics Data Commons: Machine Learning Datasets for Drug Discovery},
  author={Huang et al.},
  journal={arXiv preprint arXiv:2102.09548},
  year={2021}
}
```

---

## 🎯 Quick Start Summary

1. **List properties**: `python scripts/tdc/run_admet_pretrained.py --list-properties`
2. **Basic run**: `python scripts/tdc/run_admet_pretrained.py --property herg`
3. **Custom run**: `python scripts/tdc/run_admet_pretrained.py --property solubility_aqsoldb --seeds 1 2 3 --epochs 40`
4. **Analyze results**: Check generated CSV files and terminal output

The script provides a comprehensive solution for ADMET property evaluation using state-of-the-art pretrained molecular transformers! 