#!/bin/bash
# MolE Environment Setup Script for New Machine
# Run this script to automatically set up the MolE training environment

set -e  # Exit on any error

echo "🚀 Starting MolE Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory of this script (should be requirements/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (should be mole_public/)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_status "Script directory: $SCRIPT_DIR"
print_status "Project directory: $PROJECT_DIR"

# Change to the project directory
cd "$PROJECT_DIR" || {
    print_error "Cannot find mole_public directory. Please run this script from mole_public/requirements/"
    exit 1
}

# Verify we're in the correct directory
if [ ! -f "setup.py" ] || [ ! -d "mole" ]; then
    print_error "Not in the correct mole_public directory. Expected to find setup.py and mole/ directory."
    print_error "Current directory: $(pwd)"
    print_error "Please run this script from mole_public/requirements/"
    exit 1
fi

print_status "Working from project directory: $(pwd)"

# Check prerequisites
print_status "Checking prerequisites..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if nvidia-smi works (GPU check)
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. GPU support may not be available."
else
    print_success "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

print_success "All prerequisites met!"

# Step 1: Create conda environment
print_status "Creating conda environment 'mole-py10'..."

# Remove existing environment if it exists
if conda env list | grep -q "mole-py10"; then
    print_warning "Environment 'mole-py10' already exists. Removing it..."
    conda env remove -n mole-py10 -y
fi

# Create environment from yml file (now in requirements/)
if [ -f "$SCRIPT_DIR/environment_mole_py10.yml" ]; then
    print_status "Creating environment from requirements/environment_mole_py10.yml..."
    conda env create -f "$SCRIPT_DIR/environment_mole_py10.yml"
else
    print_warning "requirements/environment_mole_py10.yml not found. Creating environment manually..."
    
    # Create base environment
    conda create -n mole-py10 python=3.10 -y
    
    # Activate environment for package installation
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mole-py10
    
    # Install packages
    print_status "Installing core packages..."
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
    print_status "Installing DeBERTa from source..."
    pip install git+https://github.com/omendezlucio/DeBERTa.git
    pip install tensorboardX
fi

print_success "Conda environment created!"

# Step 2: Activate environment and install MolE package
print_status "Installing MolE package..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mole-py10

# Install MolE in development mode (from the project directory)
pip install -e .

print_success "MolE package installed!"

# Step 3: Test installation
print_status "Testing installation..."

# Test PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test RDKit
python -c "from rdkit import Chem; print('RDKit: OK')"

# Test PyTorch Lightning
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"

# Test MolE imports
python -c "from mole.models.crossenv_mlm import CrossEnvMLM; print('MolE imports: OK')"

print_success "All tests passed!"

# Step 4: Check for data and vocabularies
print_status "Checking for data files..."

if [ ! -f "data/guacamol_v1_all.smiles" ]; then
    print_warning "GuacaMol dataset not found at data/guacamol_v1_all.smiles"
    print_warning "Please download or copy the dataset file."
fi

if [ ! -f "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl" ]; then
    print_warning "Vocabularies not found in mole/data/vocabularies/"
    print_warning "You can either:"
    print_warning "  1. Copy vocabulary files from your original machine"
    print_warning "  2. Generate them using: python mole/data/create_vocabularies.py"
fi

# Step 5: Test dataset check script
if [ -f "data/guacamol_v1_all.smiles" ] && [ -f "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl" ]; then
    print_status "Running dataset check..."
    python scripts/check_guacamol_dataset.py
fi

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "To start training:"
echo "1. Activate environment: conda activate mole-py10"
echo "2. Set CUDA optimization: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "3. Start training: python scripts/run_crossenv_training.py"
echo ""
echo "If you need to copy data files:"
echo "- GuacaMol dataset: data/guacamol_v1_all.smiles"
echo "- Vocabularies: mole/data/vocabularies/*.pkl"
echo ""
echo "For monitoring:"
echo "- GPU usage: nvidia-smi"
echo "- Training process: ps aux | grep train_crossenv_mlm"
echo ""
print_success "Environment setup completed successfully!" 