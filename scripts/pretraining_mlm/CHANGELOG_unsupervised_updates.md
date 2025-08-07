# Unsupervised Pretraining Scripts Update

## Changes Made

Updated the unsupervised pretraining scripts to match the functionality of `run_crossenv_training_t4.py`, specifically adding:

1. **Configuration file saving**
2. **Top 5 checkpoint saving** (instead of top 3)
3. **Improved checkpoint naming format**
4. **Better directory structure**

## Files Modified

### 1. `mole/cli/train_unsupervised_pretraining.py`

**Changes:**
- Added configuration file saving using `save_config_files()` from `train_crossenv_mlm.py`
- Updated checkpoint callback to save top 5 checkpoints (was 3)
- Changed checkpoint filename format to `{epoch:02d}-{step:005d}-{val/total_loss:.4f}` (matches cross-env format)
- Added `save_last=False` and `verbose=True` to checkpoint callback
- Improved directory structure to match cross-environment script
- Updated TensorBoard logger configuration

**Before:**
```python
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{args.output_dir}/{args.model_name}/checkpoints",
    filename="{epoch}-{val/total_loss:.2f}",
    monitor="val/total_loss",
    mode="min",
    save_top_k=3,
)
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
```

## Benefits

### 1. **Configuration File Saving**
- **`model_config.json`**: Complete model architecture and hyperparameters
- **`training_args.json`**: All command-line arguments used for training
- **`training_summary.txt`**: Human-readable summary of the training configuration

### 2. **Better Checkpoint Management**
- **Top 5 checkpoints** instead of 3 (matches cross-environment script)
- **Consistent naming**: `epoch=XX-step=XXXXX-val_loss=X.XXXX.ckpt`
- **Better tracking**: Includes step count and validation loss in filename
- **No redundant checkpoints**: `save_last=False` prevents duplicate storage

### 3. **Consistent Directory Structure**
```
outputs/
└── unsupervised_pretraining_model/
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

## Usage

The convenience scripts remain unchanged in their interface:

```bash
# T4 GPU optimized
python run_unsupervised_pretraining_t4.py

# A100 2-GPU optimized  
python run_unsupervised_pretraining_a100_2gpu.py --gpus 2
```

But now they will automatically:
- Save complete configuration files
- Keep the 5 best checkpoints with detailed filenames
- Provide better organization and reproducibility

## Compatibility

- ✅ **Backward compatible**: Existing scripts work without changes
- ✅ **Forward compatible**: New features are automatically applied
- ✅ **Consistent**: Matches the cross-environment training script behavior
- ✅ **Documented**: All changes are clearly documented

## Example Output

When running training, you'll now see:
```
INFO - Saving configuration files...
INFO - Configuration files saved to outputs/model_name/
```

And the checkpoint directory will contain:
```
checkpoints/
├── epoch=01-step=00250-val_total_loss=0.234.ckpt
├── epoch=02-step=00500-val_total_loss=0.123.ckpt
├── epoch=03-step=00750-val_total_loss=0.098.ckpt  ← Best model
├── epoch=04-step=01000-val_total_loss=0.145.ckpt
└── epoch=05-step=01250-val_total_loss=0.112.ckpt
```

This update brings the unsupervised pretraining scripts up to the same professional standard as the cross-environment training scripts! 