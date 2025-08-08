# Checkpoint Resumption and Fine-Tuning Guide

This guide explains how to use checkpoint resumption and fine-tuning features across all MolE training scripts.

## 🚀 Available Features

All MolE training scripts now support:
- **Checkpoint Resumption**: Continue training from a previous checkpoint
- **Encoder Freezing**: Freeze encoder layers for fine-tuning scenarios
- **Standardized Checkpoint Management**: Top 5 best checkpoints saved with validation metrics in filenames

## 📋 Supported Scripts

| Script | Type | Resume Support | Freeze Support | Notes |
|--------|------|----------------|----------------|-------|
| `train_crossenv_mlm.py` | Cross-Environment MLM | ✅ | ✅ | Standard pretraining |
| `train_unsupervised_pretraining.py` | Multi-task Unsupervised | ✅ | ✅ | MLM + Regression tasks |
| `train_chembl_mlm.py` | ChemBL MLM + Classification | ✅ | ✅ | Drug discovery focus |
| `run_chembl_finetune_t4.py` | ChemBL Fine-tuning | ✅ | ✅ | Specialized fine-tuning script |

## 🔧 Command Line Arguments

### Resume from Checkpoint
```bash
--resume_from_checkpoint PATH_TO_CHECKPOINT
```
- **Default**: `None` (start training from scratch)
- **Purpose**: Continue training from a saved checkpoint
- **Example**: `--resume_from_checkpoint outputs/crossenv_mlm/checkpoints/05-00179-0.2345.ckpt`

### Freeze Encoder (Fine-tuning)
```bash
--freeze_encoder
```
- **Default**: `False` (encoder is trainable)
- **Purpose**: Freeze encoder layers during training for fine-tuning scenarios
- **Use case**: Transfer learning, domain adaptation

## 💡 Usage Examples

### 1. Cross-Environment Training

#### Start new training:
```bash
python mole/cli/train_crossenv_mlm.py \
  --train_data data/guacamol_r0_to_r1_train.pckl \
  --val_data data/guacamol_r0_to_r1_val.pckl \
  --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
  --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
  --output_dir outputs/my_crossenv_training \
  --model_name my_crossenv_model
```

#### Resume from checkpoint (continued pretraining):
```bash
python mole/cli/train_crossenv_mlm.py \
  --train_data data/guacamol_r0_to_r1_train.pckl \
  --val_data data/guacamol_r0_to_r1_val.pckl \
  --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
  --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
  --output_dir outputs/my_crossenv_training \
  --model_name my_crossenv_model \
  --resume_from_checkpoint outputs/my_crossenv_training/my_crossenv_model/checkpoints/05-00179-0.2345.ckpt
```

#### Resume with frozen encoder (fine-tuning):
```bash
python mole/cli/train_crossenv_mlm.py \
  --train_data data/new_domain_data.pckl \
  --val_data data/new_domain_val.pckl \
  --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
  --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
  --output_dir outputs/my_finetuned_model \
  --model_name finetuned_crossenv \
  --resume_from_checkpoint outputs/my_crossenv_training/my_crossenv_model/checkpoints/05-00179-0.2345.ckpt \
  --freeze_encoder \
  --learning_rate 5e-5 \
  --max_epochs 10
```

### 2. Unsupervised Pretraining

#### Resume training:
```bash
python mole/cli/train_unsupervised_pretraining.py \
  --train_data data/multitask_train.pckl \
  --val_data data/multitask_val.pckl \
  --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
  --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
  --output_dir outputs/unsupervised_continued \
  --resume_from_checkpoint outputs/unsupervised_pretraining/checkpoints/08-01234-0.1856.ckpt
```

### 3. ChemBL Training

#### Resume ChemBL training:
```bash
python mole/cli/train_chembl_mlm.py \
  --chembl_smiles_path data/ChemBl/chembl20Smiles.pckl \
  --chembl_labels_path data/ChemBl/labelsHard.pckl \
  --chembl_target_names_path data/ChemBl/labelsWeakHard.targetNames \
  --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
  --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
  --output_dir outputs/chembl_continued \
  --model_name chembl_resumed \
  --resume_from_checkpoint outputs/chembl_mlm/checkpoints/03-00567-0.3421.ckpt
```

### 4. ChemBL Fine-tuning (Specialized Script)

The `run_chembl_finetune_t4.py` script provides a convenient interface for fine-tuning:

```bash
# Fine-tune with frozen encoder (recommended)
python scripts/pretraining_mlm/run_chembl_finetune_t4.py \
  --pretrained_path outputs/crossenv_mlm/checkpoints/05-00179-0.2345.ckpt \
  --freeze_encoder

# Fine-tune with trainable encoder (continued pretraining)
python scripts/pretraining_mlm/run_chembl_finetune_t4.py \
  --pretrained_path outputs/crossenv_mlm/checkpoints/05-00179-0.2345.ckpt

# Fine-tune ChemBL-only model
python scripts/pretraining_mlm/run_chembl_finetune_t4.py \
  --pretrained_path outputs/crossenv_mlm/checkpoints/05-00179-0.2345.ckpt \
  --chembl_only \
  --freeze_encoder
```

## 📁 Checkpoint File Format

All checkpoints now use a standardized filename format:

```
{epoch:02d}-{step:05d}-{val_loss:.4f}.ckpt
```

**Examples:**
- `05-00179-0.2345.ckpt` - Epoch 5, Step 179, Validation Loss 0.2345
- `12-02847-0.1856.ckpt` - Epoch 12, Step 2847, Validation Loss 0.1856

**Benefits:**
- Easy identification of best models
- Sortable by performance
- Includes both epoch and step information

## 🏆 Checkpoint Management

### Automatic Saving
- **Top 5 best checkpoints** saved based on validation loss
- **No "last" checkpoint** saved (saves disk space)
- **Verbose logging** of checkpoint saves

### Finding Best Checkpoint
```bash
# List checkpoints sorted by validation loss (best first)
ls outputs/*/checkpoints/*.ckpt | sort -t'-' -k3 -n

# Get the best checkpoint path programmatically
find outputs -name "*.ckpt" | sort -t'-' -k3 -n | head -1
```

## 🔄 Fine-tuning Strategies

### 1. Frozen Encoder Fine-tuning
- **Use case**: Adapting to new tasks while preserving learned representations
- **Benefits**: Faster training, less overfitting, stable features
- **Command**: Add `--freeze_encoder` flag

### 2. Full Model Fine-tuning (Continued Pretraining)
- **Use case**: Domain adaptation, extending training on similar data
- **Benefits**: All parameters adapt to new data
- **Command**: Omit `--freeze_encoder` flag

### 3. Architecture Compatibility
When resuming from checkpoints:
- **Compatible**: Same model architecture (hidden size, layers, etc.)
- **Handled gracefully**: Missing classification heads (automatically initialized)
- **Warning given**: Incompatible parameter shapes (skipped with warning)

## ⚠️ Important Notes

### 1. Architecture Matching
Ensure the model architecture matches the checkpoint:
```bash
# Check checkpoint architecture in logs
grep "Architecture" outputs/*/logs/*/events.out.tfevents.*

# Use same architecture parameters when resuming
--hidden_size 768 --num_hidden_layers 12  # Match original training
```

### 2. Learning Rate Adjustment
For fine-tuning, use lower learning rates:
```bash
--learning_rate 5e-5   # Instead of 1e-4 for fine-tuning
--learning_rate 1e-5   # Even lower for frozen encoder fine-tuning
```

### 3. Epoch and Step Counting
When resuming:
- **Epoch counting continues** from checkpoint
- **Step counting continues** from checkpoint
- **Learning rate schedule continues** from checkpoint

### 4. Validation Monitoring
All scripts monitor `val_loss` for:
- Checkpoint saving (best models)
- Early stopping
- Learning rate scheduling

## 🐛 Troubleshooting

### Issue: "Unrecognized arguments: --resume_from_checkpoint"
**Solution**: Ensure you're using the updated training scripts that include checkpoint support.

### Issue: "RuntimeError: size mismatch for..."
**Solution**: Architecture mismatch between checkpoint and current model. Check:
- Hidden size (`--hidden_size`)
- Number of layers (`--num_hidden_layers`)
- Number of attention heads (`--num_attention_heads`)

### Issue: "FileNotFoundError: [Errno 2] No such file or directory"
**Solution**: Verify checkpoint path exists:
```bash
ls -la path/to/checkpoint.ckpt
```

### Issue: Training starts from epoch 0 despite checkpoint
**Solution**: This is normal - PyTorch Lightning handles epoch counting internally.

## 📊 Monitoring Progress

### TensorBoard Logging
```bash
tensorboard --logdir outputs/your_model/logs
```

### Check Checkpoint Quality
```bash
# List all checkpoints with their validation scores
ls -la outputs/*/checkpoints/*.ckpt | grep -o '[0-9]\+\.[0-9]\+\.ckpt'
```

### Best Model Selection
The best model is automatically saved and can be found via:
```python
# In PyTorch Lightning callback
best_model_path = trainer.checkpoint_callback.best_model_path
best_model_score = trainer.checkpoint_callback.best_model_score
```

---

## 🎯 Quick Reference

| Task | Command Flag | Use Case |
|------|--------------|----------|
| Continue training | `--resume_from_checkpoint path.ckpt` | Resume interrupted training |
| Fine-tune (frozen) | `--resume_from_checkpoint path.ckpt --freeze_encoder` | Transfer learning |
| Fine-tune (full) | `--resume_from_checkpoint path.ckpt` | Domain adaptation |
| New training | (no flags) | Start from scratch |

For more examples and advanced usage, see the individual script documentation and configuration files. 