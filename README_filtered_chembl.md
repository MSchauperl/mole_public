# ChemBL Filtered Dataset

This document describes the filtered ChemBL dataset and how to use it for more efficient training.

## Overview

The filtered ChemBL dataset is a curated subset of the original ChemBL dataset designed for:
- **Higher quality training data** with better target coverage
- **Faster training** with fewer targets and more focused compounds  
- **Better convergence** by reducing sparsity in the label matrix

## Dataset Characteristics

### Original vs Filtered Dataset

| Metric | Original ChemBL | Filtered ChemBL | Improvement |
|--------|----------------|-----------------|-------------|
| Compounds | 456,331 | 84,413 | 18.5% retention (high-coverage compounds) |
| Targets | 1,310 | 10 | Top 10 most active targets |
| Sparsity | Very high | Much lower | Better training efficiency |
| Coverage | Variable | ≥3 measurements per compound | Quality guarantee |

### Selection Criteria

1. **Target Selection**: Top 10 targets by total measurement count
2. **Compound Selection**: Only compounds with measurements for ≥3 of these targets

## Filtered Dataset Contents

### Target Information
The 10 selected targets with highest activity:

| Rank | Target ID | Original Measurements | Filtered Measurements |
|------|-----------|----------------------|----------------------|
| 1 | CHEMBL1794483 | 86,690 | 29,799 |
| 2 | CHEMBL1794345 | 83,237 | 38,971 |
| 3 | CHEMBL1794401 | 78,166 | 29,976 |
| 4 | CHEMBL1794580 | 63,916 | 33,921 |
| 5 | CHEMBL2354221 | 63,742 | 37,048 |
| 6 | CHEMBL2114810 | 58,673 | 22,661 |
| 7 | CHEMBL1794553 | 58,100 | 29,902 |
| 8 | CHEMBL1614458 | 50,998 | 23,645 |
| 9 | CHEMBL2354254 | 50,145 | 32,085 |
| 10 | CHEMBL1614530 | 45,751 | 19,058 |

### Files Generated

```
data/ChemBl_filtered/
├── chemblSmiles_top10_min3.pckl          # Filtered SMILES data (84,413 compounds)
├── labelsHard_top10_min3.pckl            # Filtered labels matrix (84,413 × 10)
├── targetNames_top10_min3.txt            # Target names for the 10 selected targets
├── compoundNames_top10_min3.txt          # Compound names for filtered compounds
└── filtering_metadata.txt               # Detailed filtering information
```

## Creating the Filtered Dataset

### Quick Start

```bash
# Create the filtered dataset
conda activate mole-py10
python scripts/pretraining_mlm/create_filtered_chembl_dataset.py

# Custom parameters
python scripts/pretraining_mlm/create_filtered_chembl_dataset.py \
    --data_dir data/ChemBl \
    --output_dir data/ChemBl_filtered \
    --top_targets 10 \
    --min_measurements 3
```

### Script Parameters

- `--data_dir`: Directory containing original ChemBL data files
- `--output_dir`: Output directory for filtered dataset  
- `--top_targets`: Number of top targets to select (default: 10)
- `--min_measurements`: Minimum measurements per compound (default: 3)

## Training with Filtered Dataset

### Available Training Scripts

#### 1. MLM + Classification Training
```bash
# Train with both MLM and ChemBL classification
python scripts/pretraining_mlm/run_chembl_filtered_training.py
```

**Configuration highlights:**
- 768 hidden dimensions, 12 layers
- Batch size: 16 × 4 accumulation = 64 effective
- Learning rate: 1e-4 with 2000 warmup steps
- MLM weight: 0.2, Classification weight: 1.0
- Max epochs: 50, Patience: 8

#### 2. ChemBL-Only Training (No MLM)
```bash
# Train only ChemBL classification (faster)
python scripts/pretraining_mlm/run_chembl_filtered_only.py
```

**Configuration highlights:**
- Same architecture as above
- Batch size: 32 × 2 accumulation = 64 effective  
- Learning rate: 2e-4 with 1000 warmup steps
- Classification only (MLM disabled)
- Max epochs: 30, Patience: 5

### Custom Training

You can also use the configuration classes directly:

```python
from chembl_config_base import FilteredChemBLConfig, FilteredChemBLOnlyConfig, run_chembl_training

# Use filtered dataset with custom parameters
overrides = {
    "--batch_size": "8",
    "--max_epochs": "20", 
    "--learning_rate": "5e-5"
}

run_chembl_training(
    config_class=FilteredChemBLConfig,
    title="Custom Filtered ChemBL Training",
    overrides=overrides
)
```

## Expected Benefits

### Training Efficiency
- **Faster epochs**: Smaller dataset means shorter training time per epoch
- **Better GPU utilization**: Less sparse data allows larger batch sizes
- **Improved convergence**: Higher quality data leads to more stable training

### Model Performance  
- **Better target coverage**: All compounds have measurements for multiple targets
- **Reduced noise**: Focus on high-activity targets with reliable measurements
- **More balanced data**: Better distribution across selected targets

### Resource Usage
- **Memory efficiency**: 10 targets vs 1,310 targets (much smaller model head)
- **Storage savings**: 84K vs 456K compounds (smaller data files)
- **Faster I/O**: Reduced data loading and preprocessing time

## Comparison with Original Dataset

### Training Time Estimates (Tesla T4)

| Dataset | Compounds | Targets | Est. Epoch Time | Est. Total Time (20 epochs) |
|---------|-----------|---------|----------------|------------------------------|
| Original ChemBL | 456,331 | 50-1,310 | ~45 min | ~15 hours |
| Filtered ChemBL | 84,413 | 10 | ~8 min | ~2.7 hours |

### Memory Usage

| Component | Original | Filtered | Reduction |
|-----------|----------|----------|-----------|
| Model head | 768 × 1,310 | 768 × 10 | 99.2% |
| Data batch | Sparse | Dense | Better utilization |
| GPU memory | ~14 GB | ~10 GB | ~30% reduction |

## Advanced Usage

### Custom Filtering Criteria

Modify the filtering script for different criteria:

```python
# Example: Top 5 targets, minimum 5 measurements
python scripts/pretraining_mlm/create_filtered_chembl_dataset.py \
    --top_targets 5 \
    --min_measurements 5

# Example: Top 20 targets, minimum 2 measurements  
python scripts/pretraining_mlm/create_filtered_chembl_dataset.py \
    --top_targets 20 \
    --min_measurements 2
```

### Integration with Existing Workflows

The filtered dataset uses the same format as the original ChemBL data, so it can be used with any existing training script by simply updating the file paths:

```bash
# Use filtered dataset with existing script
python mole/cli/train_chembl_mlm.py \
    --chembl_smiles_path data/ChemBl_filtered/chemblSmiles_top10_min3.pckl \
    --chembl_labels_path data/ChemBl_filtered/labelsHard_top10_min3.pckl \
    --chembl_target_names_path data/ChemBl_filtered/targetNames_top10_min3.txt \
    --chembl_compound_names_path data/ChemBl_filtered/compoundNames_top10_min3.txt \
    # ... other parameters
```

## Best Practices

### For Initial Experiments
1. **Start with filtered dataset** for faster iteration and debugging
2. **Use ChemBL-only mode** for classification-focused research  
3. **Monitor all 10 targets** since the dataset is small enough

### For Production Training
1. **Use MLM + Classification** for better molecular representations
2. **Train longer** (30-50 epochs) since convergence is more reliable
3. **Use larger batch sizes** when possible for better gradient estimates

### For Fine-Tuning
1. **Pre-train on filtered dataset** first for faster convergence
2. **Fine-tune on specific targets** if needed
3. **Use frozen encoder** for task-specific adaptation

## Troubleshooting

### Common Issues

**Dataset not found**: Ensure you've run the filtering script first:
```bash
python scripts/pretraining_mlm/create_filtered_chembl_dataset.py
```

**Memory errors**: Reduce batch size in configuration:
```python
overrides = {"--batch_size": "8", "--accumulate_grad_batches": "8"}
```

**Slow training**: Use ChemBL-only mode for faster training:
```bash
python scripts/pretraining_mlm/run_chembl_filtered_only.py
```

### Performance Tips

1. **Use mixed precision** (`--precision 16`) for faster training
2. **Increase num_workers** (`--num_workers 8`) for faster data loading  
3. **Use larger max_length** (`--max_length 512`) for better model capacity
4. **Monitor validation metrics** to avoid overfitting on the smaller dataset

## References

- Original ChemBL data: `data/ChemBl/`
- Filtering script: `scripts/pretraining_mlm/create_filtered_chembl_dataset.py`
- Training configurations: `scripts/pretraining_mlm/chembl_config_base.py`
- Training scripts: `scripts/pretraining_mlm/run_chembl_filtered_*.py` 