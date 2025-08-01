# ChemBL Pretraining Implementation for MolE

## Overview

This implementation extends the MolE model to support ChemBL-based pretraining, replacing the previous logP/MWT regression tasks with multi-target binary classification on ChemBL assay data.

## Key Features

- **Multi-task Learning**: Combines cross-environment MLM with ChemBL binary classification
- **Sparse Data Handling**: Efficiently handles ChemBL's sparse activity matrix (0.78% density)
- **Memory Efficient**: Target filtering and limiting to control memory usage
- **Flexible Configuration**: Easy to switch between old and new pretraining strategies
- **T4 GPU Optimized**: Default configurations optimized for Tesla T4 (16GB VRAM)

## Architecture

### Model Architecture
- **Base**: Cross-environment MLM model (radius 0 → radius 1)
- **Classification Head**: Multi-target binary classifier for ChemBL assays
- **Loss Combination**: Weighted sum of MLM loss + classification loss

### Data Structure
- **Input**: SMILES strings from ChemBL dataset (~456K molecules)
- **Labels**: Sparse binary matrix (456K × 1310 targets)
  - `1`: Active
  - `-1`: Inactive  
  - `0`: Not measured (ignored in loss)
- **Target Filtering**: Configurable minimum activity threshold per target

## Implementation Files

### Core Components

1. **`mole/data/chembl_dataset.py`**
   - `ChemBLDataset`: Extends CrossEnvMolDataset for ChemBL data
   - Handles sparse label loading and target filtering
   - Converts ChemBL labels to PyTorch tensors with proper masking

2. **`mole/data/chembl_datamodule.py`**
   - `ChemBLDataModule`: Lightning data module for ChemBL training
   - `ChemBLDatasetWithOffset`: Handles train/val/test splits with correct indexing
   - Manages data loading and batching for ChemBL experiments

3. **`mole/models/chembl_model.py`**
   - `ChemBLModel`: Extends CrossEnvMLMModel with classification heads
   - Binary classification for each target with masking support
   - Handles both MLM and classification forward passes

4. **`mole/models/chembl_lightning.py`**
   - `ChemBLLightningModule`: PyTorch Lightning wrapper
   - Comprehensive metrics tracking (AUROC, AP, F1, etc.)
   - Optional per-target metrics logging

### Training Scripts

5. **`mole/cli/train_chembl_mlm.py`**
   - Main training script for ChemBL pretraining
   - Comprehensive argument parsing
   - Supports all MLM and ChemBL-specific configurations

6. **`scripts/pretraining_mlm/run_chembl_training_t4.py`**
   - Convenience script with T4-optimized defaults
   - Easy-to-use interface for starting training
   - Predefined hyperparameters for stable training

## Usage Examples

### Quick Start with Convenience Scripts

**Option 1: T4 Optimized (MLM + ChemBL, 10% dataset)**
```bash
# Activate environment
conda activate mole-py10

# Run with T4-optimized defaults (45K molecules - 10% of dataset)
python scripts/pretraining_mlm/run_chembl_training_t4.py

# Override specific parameters
python scripts/pretraining_mlm/run_chembl_training_t4.py \
    --max_targets 25 \
    --batch_size 2 \
    --max_epochs 5
```

**Option 2: ChemBL-Only Training (NO MLM)**
```bash
# Train only on ChemBL classification tasks (no MLM)
python scripts/pretraining_mlm/run_chembl_only_t4.py

# Or enable ChemBL-only mode on any script
python scripts/pretraining_mlm/run_chembl_training_t4.py --chembl_only
```

**Option 3: Ultra-Fast Testing (1K molecules)**
```bash
# Run with minimal dataset for rapid testing/debugging
python scripts/pretraining_mlm/run_chembl_training_test.py

# ChemBL-only testing
python scripts/pretraining_mlm/run_chembl_training_test.py --chembl_only
```

**Option 4: Full Dataset Training**
```bash
# Remove max_samples limitation for full dataset
python scripts/pretraining_mlm/run_chembl_training_t4.py --max_samples 456331
```

### Advanced Usage with Training Script
```bash
python mole/cli/train_chembl_mlm.py \
    --chembl_smiles_path data/ChemBl/chembl20Smiles.pckl \
    --chembl_labels_path data/ChemBl/labelsHard.pckl \
    --chembl_target_names_path data/ChemBl/labelsWeakHard.targetNames \
    --input_vocab mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl \
    --target_vocab mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl \
    --output_dir outputs/chembl_custom \
    --model_name chembl_experiment_1 \
    --max_targets 100 \
    --min_target_activity 50 \
    --batch_size 16 \
    --max_epochs 20 \
    --mlm_loss_weight 1.0 \
    --classification_loss_weight 1.0
```

## Configuration Options

### ChemBL-Specific Parameters
- `--max_targets`: Limit number of targets for memory efficiency (default: 100)
- `--min_target_activity`: Minimum active/inactive labels per target (default: 50)
- `--max_samples`: Limit dataset size for testing (default: None = full dataset)
- `--chembl_only`: Train only on ChemBL classification (disables MLM, sets mlm_loss_weight=0)
- `--mlm_loss_weight`: Weight for MLM loss (default: 1.0, set to 0 for classification-only)
- `--classification_loss_weight`: Weight for classification loss (default: 1.0)
- `--log_target_metrics`: Enable per-target metrics logging
- `--max_targets_to_log`: Limit individually logged targets (default: 50)

### Memory Optimization for T4
- `--batch_size 16`: Reduced batch size for classification heads
- `--accumulate_grad_batches 8`: Maintain effective batch size of 128
- `--precision 16`: Mixed precision training
- `--max_length 256`: Sequence length limit
- `--num_workers 4`: Data loading workers

## Data Statistics

### ChemBL Dataset
- **Molecules**: 456,331 unique compounds
- **Targets**: 1,310 assays (before filtering)
- **Total Labels**: ~598M possible labels
- **Active Labels**: ~992K (0.17%)
- **Inactive Labels**: ~3.65M (0.61%)
- **Sparsity**: 99.22% unmeasured

### After Filtering (min_target_activity=50)
- **Targets**: ~1,149 assays (87% retention)
- **Higher data density per target**
- **More stable training signal**

## Performance Considerations

### Memory Usage
- **Full Dataset**: ~598M labels would require ~2.3GB for dense storage
- **Sparse Format**: ~65MB for ChemBL labels matrix
- **Target Limiting**: Linear memory scaling with `max_targets`

### Training Efficiency
- **Mixed Precision**: Essential for T4 memory constraints
- **Gradient Accumulation**: Maintains large effective batch sizes
- **Target Filtering**: Reduces noise from poorly-represented assays
- **Early Stopping**: Prevents overfitting with patience-based monitoring

## Metrics and Monitoring

### MLM Metrics
- Perplexity
- Accuracy (top-1 and top-5)
- Standard MLM monitoring

### Classification Metrics
- **Global** (averaged across all targets):
  - AUROC, Average Precision
  - Accuracy, F1, Precision, Recall
- **Per-target** (optional, for detailed analysis):
  - AUROC and Average Precision per assay
  - Configurable target limit to prevent metric explosion

### Loss Components
- `total_loss = mlm_loss_weight * mlm_loss + classification_loss_weight * classification_loss`
- Separate logging for analysis and debugging

## Compatibility

### Backward Compatibility
- **Old Training**: Existing GuacaMol training scripts unchanged
- **Vocabularies**: Reuses existing radius 0/1 vocabularies
- **Model Architecture**: Extends existing CrossEnvMLM base

### Forward Compatibility  
- **Modular Design**: Easy to add more classification tasks
- **Configurable Loss**: Simple to adjust task weightings
- **Extensible Metrics**: Framework for additional evaluation metrics

## Dataset Size Options

The system now supports flexible dataset sizing for different use cases:

| Script | Model Size | Targets | Dataset Size | Mode | Est. Time | Use Case |
|--------|------------|---------|--------------|------|-----------|----------|
| `run_chembl_training_test.py` | 128/2L | 10 | 1K (0.2%) | MLM+ChemBL | ~5 min | Rapid debugging |
| `run_chembl_training_test.py --chembl_only` | 128/2L | 10 | 1K (0.2%) | ChemBL only | ~3 min | Classification debugging |
| `run_chembl_training_t4.py` | 512/8L | 50 | 45K (10%) | MLM+ChemBL | ~2 hours | Testing/validation |
| `run_chembl_only_t4.py` | 768/8L | 50 | 45K (10%) | ChemBL only | ~1 hour | Classification only |
| **Full Training** | 512/8L+ | 100+ | 456K (100%) | MLM+ChemBL | ~20+ hours | Production |
| **Full ChemBL-only** | 768/8L+ | 100+ | 456K (100%) | ChemBL only | ~10+ hours | Classification production |

### Usage Recommendations

- **Development/Debugging**: Use `run_chembl_training_test.py` for rapid iteration
- **Validation/Testing**: Use `run_chembl_training_t4.py` with 10% dataset  
- **Production Training**: Remove `--max_samples` limitation for full dataset
- **Memory Issues**: Use `run_chembl_training_small.py` for constrained environments

## Testing and Validation

The implementation has been tested with:
1. **Data Loading**: Verified ChemBL data loading and preprocessing
2. **Model Creation**: Confirmed model instantiation and parameter counts
3. **Forward Pass**: Tested model forward pass with dummy data
4. **Training Step**: Validated complete training step execution
5. **Pipeline Integration**: End-to-end pipeline functionality

## Next Steps

To start ChemBL pretraining:

1. **Ensure Environment**: Activate `mole-py10` conda environment
2. **Check Data**: Verify ChemBL data files in `data/ChemBl/`
3. **Run Training**: Use convenience script or training script directly
4. **Monitor Progress**: Check TensorBoard logs and metrics
5. **Evaluate Results**: Compare with baseline MLM performance

The implementation provides a solid foundation for ChemBL-based pretraining while maintaining compatibility with existing workflows. 