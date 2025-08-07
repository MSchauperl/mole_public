# Pretrained Model Loading and Fine-Tuning

This directory now supports loading pretrained models for fine-tuning. This allows you to:

1. **Continue pretraining** from a previously saved checkpoint
2. **Fine-tune** a pretrained model with frozen encoder layers
3. **Transfer learn** from one task to another

## Quick Start

### 1. Fine-Tuning with Frozen Encoder (Recommended)

```bash
# Fine-tune a pretrained MLM model with frozen encoder
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/best_model.ckpt \
    --freeze_encoder

# Fine-tune a ChemBL-only model with frozen encoder  
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_only/checkpoints/best_model.ckpt \
    --chembl_only \
    --freeze_encoder
```

### 2. Continued Pretraining (Trainable Encoder)

```bash
# Continue pretraining with all layers trainable
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/best_model.ckpt
```

## Available Configurations

### Standard Training (No Pretrained Model)
- `run_chembl_training_t4.py` - MLM + Classification training
- `run_chembl_only_t4.py` - Classification-only training  
- `run_chembl_training_test.py` - Rapid testing (1000 samples)

### Fine-Tuning (With Pretrained Model)
- `run_chembl_finetune_t4.py` - Fine-tuning with optimized settings

## Configuration Classes

### Base Classes
- `ChemBLConfig` - Base configuration with common parameters
- `T4Config` - Tesla T4 optimized for MLM + Classification
- `T4OnlyConfig` - Tesla T4 optimized for Classification only
- `TestConfig` - Minimal configuration for rapid testing

### Fine-Tuning Classes  
- `T4FineTuneConfig` - Tesla T4 optimized for fine-tuning (MLM + Classification)
- `T4OnlyFineTuneConfig` - Tesla T4 optimized for fine-tuning (Classification only)

## Fine-Tuning Optimizations

When using fine-tuning configurations, the following optimizations are automatically applied:

### Learning Rate
- **Standard training**: `1e-4` or `2e-4`
- **Fine-tuning**: `5e-5` or `2e-5` (lower for stability)

### Training Duration
- **Standard training**: 20 epochs, 3000 warmup steps
- **Fine-tuning**: 8-10 epochs, 200-500 warmup steps

### Loss Weighting
- **Standard training**: MLM=0.1, Classification=1.0
- **Fine-tuning**: MLM=0.05, Classification=2.0 (emphasizes classification task)

### Early Stopping
- **Standard training**: Patience=3-5
- **Fine-tuning**: Patience=2-3 (more aggressive)

## Encoder Freezing Options

### Frozen Encoder (`--freeze_encoder`)
- **Use when**: You want to fine-tune only the classification heads
- **Benefits**: Faster training, less memory usage, prevents catastrophic forgetting
- **Recommended for**: Task-specific fine-tuning

### Trainable Encoder (default)
- **Use when**: You want continued pretraining or significant domain adaptation
- **Benefits**: Can adapt to new data distributions
- **Recommended for**: Continued pretraining, domain adaptation

## Examples

### Example 1: Fine-Tune for New ChemBL Targets
```bash
# Start with a pretrained model and fine-tune for new classification targets
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/best_model.ckpt \
    --freeze_encoder \
    --max_targets 100 \
    --learning_rate 3e-5
```

### Example 2: Continue Pretraining with More Data
```bash
# Continue pretraining with additional data
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/best_model.ckpt \
    --max_samples 100000 \
    --max_epochs 5
```

### Example 3: Transfer from MLM to Classification-Only
```bash
# Start with MLM model and fine-tune for classification only
python run_chembl_finetune_t4.py \
    --pretrained_path outputs/chembl_mlm/checkpoints/best_model.ckpt \
    --chembl_only \
    --freeze_encoder
```

## Output Structure

Fine-tuning runs create separate output directories:

```
outputs/
├── chembl_mlm/                    # Original pretraining
│   └── checkpoints/best_model.ckpt
├── chembl_finetune/               # MLM + Classification fine-tuning  
│   ├── checkpoints/
│   ├── model_config.json
│   └── training_args.json
└── chembl_only_finetune/          # Classification-only fine-tuning
    ├── checkpoints/
    ├── model_config.json
    └── training_args.json
```

## Configuration Summary

When running with a pretrained model, the configuration summary will show:

```
🔄 Pretrained model: outputs/chembl_mlm/checkpoints/best_model.ckpt
🧊 Encoder: FROZEN (fine-tuning mode)
```

or 

```
🔄 Pretrained model: outputs/chembl_mlm/checkpoints/best_model.ckpt  
🔥 Encoder: TRAINABLE (continued pretraining)
```

This makes it clear what type of training is being performed. 