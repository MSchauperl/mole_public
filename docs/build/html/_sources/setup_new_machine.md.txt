# MolE Environment Setup Guide for New Machine

This guide will help you replicate the exact MolE training environment on a new machine.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (tested on RTX 3070, 8GB VRAM)
2. **Linux OS** (tested on Ubuntu)
3. **Anaconda/Miniconda** installed
4. **Git** installed

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd mole_public
```

## Step 2: Create Conda Environment

### Option A: From environment.yml (Recommended)
```bash
# Create environment from the provided environment.yml (now in requirements/)
conda env create -f requirements/environment_mole_py10.yml

# Activate the environment
conda activate mole-py10
```

### Option B: Manual setup (if Option A fails)
```bash
# Create base environment
conda create -n mole-py10 python=3.10 -y
conda activate mole-py10

# Install core packages
conda install -c conda-forge -c pytorch -c pyg \
    pytorch=2.4.1 \
    pytorch-cuda=12.4 \
    torchvision \
    torchaudio \
    pyg=2.6.1 \
    pytorch-lightning=2.5.1 \
    rdkit=2024.09.6 \
    numpy=1.26.4 \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    omegaconf \
    hydra-core \
    click \
    pyarrow \
    h5py \
    jupyter \
    pytest \
    -y

# Install pip packages
pip install git+https://github.com/omendezlucio/DeBERTa.git
```

## Step 3: Install MolE Package

```bash
# Install in development mode
pip install -e .
```

## Step 4: Download GuacaMol Dataset

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download GuacaMol dataset (you'll need to provide the actual download link)
# For now, ensure you have: data/guacamol_v1_all.smiles
```

## Step 5: Create/Download Vocabularies

### Option A: Use existing vocabularies (faster)
Copy the vocabulary files from your original machine:
```bash
# Copy these files to mole/data/vocabularies/
# - vocabulary_radius0_structural_guacamol_v1.pkl
# - vocabulary_radius1_functional_guacamol_v1.pkl
```

### Option B: Generate vocabularies (slower but ensures compatibility)
```bash
# Generate vocabularies from GuacaMol dataset
python mole/data/create_vocabularies.py \
    --smiles_input data/guacamol_v1_all.smiles \
    --output_dir mole/data/vocabularies \
    --radii 0 1 \
    --max_molecules 1000000  # Adjust as needed
```

## Step 6: Test Installation

```bash
# Test the environment setup
python scripts/check_guacamol_dataset.py
```

## Step 7: Start Training

```bash
# Set CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start training
python scripts/run_crossenv_training.py
```

## GPU Memory Optimization

The training script is optimized for RTX 3070 (8GB VRAM). If you have:

### More GPU Memory (>8GB):
You can increase model size in `scripts/run_crossenv_training.py`:
```python
"--hidden_size": "768",        # Instead of 384
"--num_hidden_layers": "12",   # Instead of 6
"--batch_size": "32",          # Instead of 16
```

### Less GPU Memory (<8GB):
You may need to reduce further:
```python
"--hidden_size": "256",        # Even smaller
"--num_hidden_layers": "4",    # Fewer layers
"--batch_size": "8",           # Smaller batch
"--accumulate_grad_batches": "16",  # More accumulation
```

## Troubleshooting

### CUDA Out of Memory
1. Reduce `--batch_size`
2. Increase `--accumulate_grad_batches` to maintain effective batch size
3. Reduce `--hidden_size` and `--num_hidden_layers`
4. Reduce `--max_length`

### Environment Issues
1. Ensure CUDA drivers are properly installed
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify RDKit installation: `python -c "from rdkit import Chem; print('RDKit OK')"`

### Package Conflicts
1. Delete environment and recreate: `conda env remove -n mole-py10`
2. Use the manual setup (Option B)
3. Check for conflicting conda channels

## Expected Training Performance

- **Dataset**: ~1.6M molecules from GuacaMol
- **GPU Memory Usage**: ~800MB on RTX 3070
- **Training Time**: ~2-3 days for 30 epochs
- **Model Size**: ~25M parameters (memory-optimized version)

## Monitoring Training

```bash
# Check GPU usage
nvidia-smi

# Check training process
ps aux | grep train_crossenv_mlm

# Monitor logs (if using TensorBoard)
tensorboard --logdir outputs/guacamol_crossenv_mlm/
```

## Files You Need to Transfer

If copying from an existing setup:

1. **Code**: Entire `mole_public/` directory
2. **Data**: `data/guacamol_v1_all.smiles` (74MB)
3. **Vocabularies**: 
   - `mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl`
   - `mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl`
4. **Environment**: Use the provided `requirements/environment_mole_py10.yml`

## Quick Start Script

See `requirements/setup_machine.sh` for an automated setup script.

### Automated Setup
```bash
# Run the automated setup (from the main mole_public directory)
./requirements/setup_machine.sh
```

This script will:
1. Check prerequisites (conda, NVIDIA drivers, git)
2. Create the mole-py10 environment from `requirements/environment_mole_py10.yml`
3. Install MolE package in development mode
4. Test all installations
5. Check for required data files
6. Provide next steps for training 