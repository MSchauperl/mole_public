# ChemBL Training Scripts Update

## Changes Made

Updated the ChemBL training scripts to match the functionality of `run_crossenv_training_t4.py`, specifically adding:

1. **Configuration file saving**
2. **Top 5 checkpoint saving** (instead of top 3)
3. **Improved checkpoint naming format**
4. **Better directory structure**
5. **Consistent behavior with cross-environment scripts**

## Files Modified

### 1. `mole/cli/train_chembl_mlm.py`

**Changes:**
- Added configuration file saving using `save_config_files()` from `train_crossenv_mlm.py`
- Updated checkpoint callback to save top 5 checkpoints (was 3)
- Changed checkpoint filename format to `{epoch:02d}-{step:005d}-{val/total_loss:.4f}` (matches cross-env format)
- Added `save_last=False` and `verbose=True` to checkpoint callback
- Restructured callback creation to match cross-environment script pattern
- Improved early stopping handling

**Before:**
```python
callbacks = [
    ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="chembl-mlm-{epoch:02d}-{val/total_loss:.3f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ),
    LearningRateMonitor(logging_interval="step"),
    EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=args.patience,
        verbose=True,
    ),
]
```

**After:**
```python
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir / "checkpoints",
    filename="{epoch:02d}-{step:005d}-{val/total_loss:.4f}",
    monitor="val/total_loss",
    mode="min",
    save_top_k=5,  # Save the 5 best checkpoints
    save_last=False,  # Don't save the last checkpoint
    verbose=True,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [checkpoint_callback, lr_monitor]

if args.patience > 0:
    early_stopping_callback = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=args.patience,
        verbose=True,
    )
    callbacks.append(early_stopping_callback)
```

### 2. ChemBL Configuration Scripts (No Changes Required)

The existing ChemBL configuration scripts already support the updated functionality:
- `run_chembl_training_t4.py`
- `run_chembl_only_t4.py` 
- `run_chembl_training_test.py`
- `run_chembl_finetune_t4.py`

## Benefits

### 1. **Configuration File Saving**
- **`model_config.json`**: Complete model architecture and hyperparameters
- **`training_args.json`**: All command-line arguments used for training
- **`training_summary.txt`**: Human-readable summary of the training configuration

### 2. **Better Checkpoint Management**
- **Top 5 checkpoints** instead of 3 (matches cross-environment script)
- **Consistent naming**: `epoch=XX-step=XXXXX-val_total_loss=X.XXXX.ckpt`
- **Better tracking**: Includes step count and validation loss in filename
- **No redundant checkpoints**: `save_last=False` prevents duplicate storage

### 3. **Consistent Directory Structure**
```
outputs/
└── chembl_model_name/
    ├── checkpoints/
    │   ├── epoch=02-step=00500-val_total_loss=0.1234.ckpt
    │   ├── epoch=03-step=00750-val_total_loss=0.0987.ckpt  ← Best
    │   └── ...
    ├── logs/
    ├── model_config.json
    ├── training_args.json
    └── training_summary.txt
```

### 4. **Improved Reproducibility**
- All training parameters are automatically saved
- Easy to reproduce exact training conditions
- Clear documentation of model architecture and hyperparameters
- Consistent with cross-environment training workflow

## Usage Examples

All ChemBL training scripts now automatically include the enhanced functionality:

### Standard ChemBL Training
```bash
# T4 GPU optimized MLM + Classification
python run_chembl_training_t4.py

# T4 GPU optimized Classification only
python run_chembl_only_t4.py  

# Rapid testing (1000 samples)
python run_chembl_training_test.py
```

### ChemBL Fine-Tuning
```bash
# Fine-tune with frozen encoder
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/epoch=03-step=00750-val_total_loss=0.0987.ckpt \
    --freeze_encoder

# Fine-tune ChemBL-only model
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_only/checkpoints/best_model.ckpt \
    --chembl_only \
    --freeze_encoder
```

## Configuration Files Generated

When running any ChemBL training, you'll now see:
```
INFO - Saving configuration files...
INFO - Configuration files saved to outputs/model_name/
```

### Example `training_summary.txt`:
```
Cross-Environment MLM Training Summary
==================================================

DATA CONFIGURATION:
  ChemBL SMILES: data/ChemBl/chembl20Smiles.pckl
  ChemBL labels: data/ChemBl/labelsHard.pckl
  Input vocabulary: mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl
  Target vocabulary: mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl
  Input vocab size: 1000
  Target vocab size: 1500
  Max targets: 50
  Min target activity: 50

MODEL CONFIGURATION:
  Hidden size: 512
  Number of layers: 8
  Number of attention heads: 8
  Intermediate size: 2048
  Dropout: 0.1
  Classifier dropout: 0.1

TRAINING CONFIGURATION:
  Batch size: 4
  Learning rate: 1e-4
  Weight decay: 0.01
  Warmup steps: 3000
  Max epochs: 20
  MLM loss weight: 0.1
  Classification loss weight: 1.0
```

## Checkpoint Management

The checkpoint directory will now contain:
```
checkpoints/
├── epoch=01-step=00250-val_total_loss=0.234.ckpt
├── epoch=02-step=00500-val_total_loss=0.123.ckpt
├── epoch=03-step=00750-val_total_loss=0.098.ckpt  ← Best model
├── epoch=04-step=01000-val_total_loss=0.145.ckpt
└── epoch=05-step=01250-val_total_loss=0.112.ckpt
```

## Compatibility

- ✅ **Backward compatible**: Existing ChemBL scripts work without changes
- ✅ **Forward compatible**: New features are automatically applied
- ✅ **Consistent**: Now matches cross-environment and unsupervised pretraining behavior
- ✅ **Professional**: Same standard across all training scripts

## Before vs After Comparison

| Feature | Before | After |
|---------|--------|-------|
| Checkpoints saved | 3 | 5 |
| Checkpoint naming | `chembl-mlm-{epoch:02d}-{val/total_loss:.3f}` | `{epoch:02d}-{step:005d}-{val/total_loss:.4f}` |
| Configuration files | ❌ None | ✅ 3 files (JSON + TXT) |
| Step tracking | ❌ No | ✅ Yes |
| Reproducibility | ⚠️ Limited | ✅ Complete |
| Save last checkpoint | ✅ Yes | ❌ No (cleaner) |
| Consistency | ⚠️ Different format | ✅ Matches other scripts |

## Impact

This update brings **all ChemBL training scripts** up to the same professional standard as the cross-environment and unsupervised pretraining scripts, ensuring:

1. **Complete reproducibility** across all model types
2. **Consistent user experience** regardless of training type
3. **Better experiment tracking** with detailed configuration logs
4. **Improved checkpoint management** with more options and better naming
5. **Professional-grade documentation** for every training run

All ChemBL training now follows the same high-quality patterns established in the cross-environment training workflow! 🎉 