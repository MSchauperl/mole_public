# ChemBL Training Scripts

This directory contains scripts for training MolE models on the ChemBL dataset. The scripts have been simplified to reduce redundancy while maintaining flexibility.

## Quick Start

Use the unified script for most training scenarios:

```bash
# Train on filtered ChemBL with MLM and fold-based splits (recommended)
python run_chembl_unified.py --dataset filtered --mlm true --splits folds

# Train on filtered ChemBL with classification only
python run_chembl_unified.py --dataset filtered --mlm false --splits folds

# Quick test run (MLM enabled by default)
python run_chembl_unified.py --dataset test --splits random
```

## Script Overview

### Primary Scripts (Keep These)

1. **`run_chembl_unified.py`** - **Main script for most use cases**
   - Choose dataset: `full`, `filtered`, or `test`
   - Choose splits: `random` or `folds`
   - Enable/disable MLM
   - Fine-tuning support

2. **`run_chembl_filtered_fold_training.py`** - **Advanced fold-based training**
   - Direct CLI parameter control
   - Cross-validation support
   - Individual fold training
   - More granular parameter control

### Legacy Scripts (Can be removed)

The following scripts can be replaced by `run_chembl_unified.py`:

- `run_chembl_only_t4.py` → `run_chembl_unified.py --dataset full --mlm false --splits random`
- `run_chembl_training_t4.py` → `run_chembl_unified.py --dataset full --mlm true --splits random`
- `run_chembl_filtered_only.py` → `run_chembl_unified.py --dataset filtered --mlm false --splits folds`
- `run_chembl_filtered_training.py` → `run_chembl_unified.py --dataset filtered --mlm true --splits folds`
- `run_chembl_training_test.py` → `run_chembl_unified.py --dataset test --mlm true --splits random`
- `run_chembl_t4_folds.py` → `run_chembl_unified.py --dataset filtered --mlm true --splits folds`
- `run_chembl_finetune_t4.py` → `run_chembl_unified.py --resume_from_checkpoint <path>`

## Usage Examples

### Basic Training

```bash
# Filtered dataset with MLM (recommended for most cases)
python run_chembl_unified.py --dataset filtered --mlm true --splits folds

# Full dataset with MLM (larger, sampled dataset)
python run_chembl_unified.py --dataset full --mlm true --splits random

# Classification only (no MLM)
python run_chembl_unified.py --dataset filtered --mlm false --splits folds

# Quick test run (MLM enabled by default)
python run_chembl_unified.py --dataset test --splits random
```

### Fine-tuning

```bash
# Fine-tune with frozen encoder (recommended)
python run_chembl_unified.py --dataset filtered --mlm true --splits folds \
    --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt --freeze_encoder

# Fine-tune with trainable encoder (continued pretraining)
python run_chembl_unified.py --dataset filtered --mlm true --splits folds \
    --resume_from_checkpoint outputs/chembl_mlm/best_model.ckpt

# Fine-tune ChemBL-only model
python run_chembl_unified.py --dataset filtered --mlm false --splits folds \
    --resume_from_checkpoint outputs/chembl_only/best_model.ckpt --freeze_encoder
```

### Advanced Usage

```bash
# Pass additional arguments to the training script
python run_chembl_unified.py --dataset filtered --mlm true --splits folds \
    --batch_size 32 --learning_rate 2e-4

# Use different output directory
python run_chembl_unified.py --dataset filtered --mlm true --splits folds \
    --output_dir outputs/my_experiment
```

## Dataset Options

### `--dataset full`
- **Data**: Full ChemBL dataset (sampled to ~45K compounds)
- **Targets**: Up to 50 targets with ≥500 measurements
- **Use case**: When you need maximum data diversity
- **Splits**: Random splits only

### `--dataset filtered` (Recommended)
- **Data**: Filtered ChemBL dataset (84K compounds)
- **Targets**: Top 10 targets with ≥3 measurements per compound
- **Use case**: High-quality, focused training
- **Splits**: Both random and fold-based splits available

### `--dataset test`
- **Data**: 1,000 samples for rapid testing
- **Targets**: 10 targets
- **Use case**: Quick debugging and testing
- **Splits**: Random splits only

## Split Options

### `--splits random`
- Standard random train/validation/test splits
- Faster to set up
- May have some data leakage between splits

### `--splits folds` (Recommended)
- Fold-based cross-validation splits
- No data leakage between train and validation
- Better for reliable performance estimation
- Only available for filtered dataset

## MLM Options

### `--mlm true` (Default)
- Enable masked language modeling
- Joint training of MLM + classification
- Better representation learning
- Slower training, more memory usage

### `--mlm false`
- Classification only (no MLM)
- Faster training, less memory usage
- Can use larger models/batch sizes
- Good for fine-tuning scenarios

## Fine-tuning Options

### `--resume_from_checkpoint <path>`
- Resume training from a saved checkpoint
- Automatically enables fine-tuning mode
- Path should point to a `.ckpt` file

### `--freeze_encoder`
- Freeze the encoder layers during fine-tuning
- Only train the classification head
- Recommended for most fine-tuning scenarios
- Requires `--resume_from_checkpoint`

## Configuration Summary

The script automatically selects the appropriate configuration based on your choices:

| Dataset | MLM | Splits | Config Class |
|---------|-----|--------|--------------|
| `test` | Any | Any | `TestConfig` |
| `full` | ✅ | `random` | `T4Config` |
| `full` | ❌ | `random` | `T4OnlyConfig` |
| `full` | ✅ | `folds` | `T4ConfigWithFolds` |
| `full` | ❌ | `folds` | `T4OnlyConfigWithFolds` |
| `filtered` | ✅ | Any | `FilteredChemBLConfig` |
| `filtered` | ❌ | Any | `FilteredChemBLOnlyConfig` |

## Migration Guide

If you were using the old scripts, here are the equivalent commands:

### Old → New

```bash
# Old: python run_chembl_training_t4.py
# New: python run_chembl_unified.py --dataset full --mlm true --splits random

# Old: python run_chembl_only_t4.py  
# New: python run_chembl_unified.py --dataset full --mlm false --splits random

# Old: python run_chembl_filtered_training.py
# New: python run_chembl_unified.py --dataset filtered --mlm true --splits folds

# Old: python run_chembl_filtered_only.py
# New: python run_chembl_unified.py --dataset filtered --mlm false --splits folds

# Old: python run_chembl_training_test.py
# New: python run_chembl_unified.py --dataset test --mlm true --splits random

# Old: python run_chembl_t4_folds.py
# New: python run_chembl_unified.py --dataset filtered --mlm true --splits folds

# Old: python run_chembl_finetune_t4.py --pretrained_path <path> --freeze_encoder
# New: python run_chembl_unified.py --resume_from_checkpoint <path> --freeze_encoder
```

## Recommendations

1. **Start with filtered dataset**: Better quality, faster training
2. **Use fold-based splits**: More reliable performance estimation
3. **Enable MLM for pretraining**: Better representation learning
4. **Use frozen encoder for fine-tuning**: More stable training
5. **Use test dataset for debugging**: Quick iteration cycles
