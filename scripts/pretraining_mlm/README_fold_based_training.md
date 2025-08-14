# Fold-based Training for ChemBL Dataset

This document describes the fold-based training system for the ChemBL dataset, which ensures proper cross-validation and eliminates data leakage between training and validation sets.

## Overview

The fold-based training system uses pre-defined cross-validation splits from `folds0.pckl` to create training and validation sets that are completely disjoint. This provides:

- **No data leakage**: Training and validation sets are completely separate
- **Reproducible results**: Same splits every time
- **Proper cross-validation**: 3-fold splits for robust evaluation
- **High-quality dataset**: 84,413 compounds with ≥3 target measurements

## Dataset Information

- **Source**: Filtered ChemBL dataset (top 10 targets, compounds with ≥3 measurements)
- **Size**: 84,413 compounds
- **Targets**: 10 most active targets from original 1,310
- **Splits**: 3-fold cross-validation
  - Fold 0: ~58,815 train, ~25,598 validation
  - Fold 1: ~58,815 train, ~25,598 validation  
  - Fold 2: ~58,815 train, ~25,598 validation

## Training Scripts

### 1. Basic T4 Training with Fold-based Splits

**Script**: `run_chembl_t4_folds.py`

```bash
# MLM + Classification training
python scripts/pretraining_mlm/run_chembl_t4_folds.py

# Classification only training
python scripts/pretraining_mlm/run_chembl_t4_folds.py --chembl_only
```

**Features**:
- T4-optimized settings (512 hidden, 8 layers, 16-bit precision)
- Uses filtered dataset with fold-based splits
- Configurable batch size, learning rate, epochs
- Supports both MLM+classification and classification-only modes

### 2. Fine-tuning with Fold-based Splits

**Script**: `run_chembl_finetune_t4.py`

```bash
# Fine-tune with fold-based splits
python scripts/pretraining_mlm/run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/best_model.ckpt \
    --use_folds --freeze_encoder

# Fine-tune ChemBL-only with fold-based splits
python scripts/pretraining_mlm/run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_only/best_model.ckpt \
    --use_folds --chembl_only
```

**Features**:
- Supports both random and fold-based splits
- Configurable encoder freezing
- Uses same T4-optimized settings

### 3. Cross-validation Training (All Folds)

**Script**: `run_chembl_filtered_fold_training.py`

```bash
# Train on all 3 folds for complete cross-validation
python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --cross_validation

# Train on specific fold
python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 0
python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 1
python scripts/pretraining_mlm/run_chembl_filtered_fold_training.py --fold 2
```

**Features**:
- Automated cross-validation across all folds
- Individual fold training
- Custom output suffixes for each fold

## Configuration Classes

### T4ConfigWithFolds
- **Purpose**: MLM + classification training with fold-based splits
- **Architecture**: 512 hidden, 8 layers, 8 attention heads
- **Dataset**: Filtered ChemBL with fold splits
- **Output**: `outputs/chembl_mlm_folds/`

### T4OnlyConfigWithFolds  
- **Purpose**: Classification-only training with fold-based splits
- **Architecture**: 768 hidden, 8 layers, 12 attention heads
- **Dataset**: Filtered ChemBL with fold splits
- **Output**: `outputs/chembl_only_folds/`

### FilteredChemBLConfig
- **Purpose**: Default filtered dataset training (uses fold splits by default)
- **Architecture**: Configurable
- **Dataset**: Filtered ChemBL with fold splits
- **Output**: `outputs/chembl_filtered/`

## Dataset Creation

### Create Filtered Dataset with Folds

```bash
# Create filtered dataset with fold-based splits
python scripts/pretraining_mlm/create_filtered_chembl_dataset_with_folds.py

# Create all fold combinations
python scripts/pretraining_mlm/create_all_fold_combinations.py
```

### Generated Files

The dataset creation script generates:

- `data/ChemBl_filtered/chemblSmiles_top10_min3.pckl` - Filtered SMILES
- `data/ChemBl_filtered/labelsHard_top10_min3.pckl` - Filtered labels
- `data/ChemBl_filtered/targetNames_top10_min3.txt` - Target names
- `data/ChemBl_filtered/compoundNames_top10_min3.txt` - Compound names
- `data/ChemBl_filtered/splits_top10_min3.pckl` - **Train/validation splits**

## Data Module Integration

The `ChemBLDataModule` has been enhanced to support fold-based splits:

```python
# With fold-based splits
datamodule = ChemBLDataModule(
    split_indices_path="data/ChemBl_filtered/splits_top10_min3.pckl",
    # ... other parameters
)

# With random splits (fallback)
datamodule = ChemBLDataModule(
    # split_indices_path=None (default)
    # ... other parameters
)
```

## Testing and Validation

### Test Fold-based Splits

```bash
# Test the fold-based splits functionality
python scripts/pretraining_mlm/test_fold_splits.py

# Run comprehensive example and test
python scripts/pretraining_mlm/example_fold_based_training.py --run_test
```

### Example Usage

```bash
# Show all training examples
python scripts/pretraining_mlm/example_fold_based_training.py --show_examples

# Show dataset creation examples  
python scripts/pretraining_mlm/example_fold_based_training.py --show_dataset_creation
```

## Key Benefits

1. **No Data Leakage**: Training and validation sets are completely disjoint
2. **Reproducible**: Same splits every time, ensuring consistent results
3. **Cross-validation Ready**: 3-fold splits for robust model evaluation
4. **High Quality**: Uses compounds with ≥3 target measurements
5. **Efficient**: Focused on 10 most active targets
6. **Flexible**: Supports both MLM+classification and classification-only modes

## Migration from Random Splits

To migrate existing training scripts to use fold-based splits:

1. **Update configuration**: Use `T4ConfigWithFolds` or `T4OnlyConfigWithFolds`
2. **Add split_indices_path**: Include `--split_indices_path` parameter
3. **Update output directories**: Use fold-specific output paths
4. **Test thoroughly**: Verify no data leakage and proper splits

## Troubleshooting

### Common Issues

1. **Missing filtered dataset**: Run `create_filtered_chembl_dataset_with_folds.py`
2. **Missing split indices**: Ensure `splits_top10_min3.pckl` exists
3. **Data leakage detected**: Check that fold-based splits are being used
4. **Deprecation warnings**: These are suppressed automatically

### Verification Commands

```bash
# Check if filtered dataset exists
ls -la data/ChemBl_filtered/

# Check if split indices exist
ls -la data/ChemBl_filtered/splits_top10_min3.pckl

# Test fold-based splits
python scripts/pretraining_mlm/test_fold_splits.py
```

## Performance Comparison

| Configuration | Dataset Size | Targets | Splits | Training Time | Memory Usage |
|---------------|--------------|---------|--------|---------------|--------------|
| Full ChemBL | 456,331 | 1,310 | Random | ~8 hours | ~16GB |
| Filtered + Folds | 84,413 | 10 | Fold-based | ~2 hours | ~8GB |
| Filtered + Random | 84,413 | 10 | Random | ~2 hours | ~8GB |

The fold-based approach provides the same efficiency as random splits but with the added benefit of no data leakage and reproducible results.
