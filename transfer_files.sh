#!/bin/bash
# File Transfer Script for MolE Setup
# Use this script to copy necessary files to a new machine

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_note() {
    echo -e "${YELLOW}[NOTE]${NC} $1"
}

echo "📦 MolE File Transfer Guide"
echo "==========================="
echo ""

# Check if we're in the right directory
if [ ! -f "setup_machine.sh" ]; then
    echo "❌ Please run this script from the mole_public directory"
    exit 1
fi

NEW_MACHINE_USER="$1"
NEW_MACHINE_IP="$2"
NEW_MACHINE_PATH="$3"

if [ -z "$NEW_MACHINE_USER" ] || [ -z "$NEW_MACHINE_IP" ] || [ -z "$NEW_MACHINE_PATH" ]; then
    echo "Usage: $0 <username> <ip_address> <destination_path>"
    echo ""
    echo "Example: $0 user 192.168.1.100 /home/user/mole_public"
    echo ""
    echo "Or manually copy these files:"
    echo ""
    print_info "Essential files to copy:"
    echo "1. Entire project directory (excluding outputs/)"
    echo "2. GuacaMol dataset: data/guacamol_v1_all.smiles (74MB)"
    echo "3. Vocabularies in mole/data/vocabularies/*.pkl"
    echo ""
    print_info "File sizes:"
    if [ -f "data/guacamol_v1_all.smiles" ]; then
        echo "   GuacaMol dataset: $(du -h data/guacamol_v1_all.smiles | cut -f1)"
    fi
    
    if [ -d "mole/data/vocabularies" ]; then
        echo "   Vocabularies: $(du -sh mole/data/vocabularies | cut -f1)"
    fi
    
    echo "   Total project: $(du -sh . --exclude=outputs | cut -f1)"
    echo ""
    print_note "You can exclude these directories to save space:"
    echo "   - outputs/ (training outputs)"
    echo "   - .git/ (git history)"
    echo "   - __pycache__/ (Python cache)"
    echo "   - *.egg-info/ (package info)"
    exit 1
fi

print_info "Transferring files to $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH"

# Create destination directory
print_info "Creating destination directory..."
ssh $NEW_MACHINE_USER@$NEW_MACHINE_IP "mkdir -p $NEW_MACHINE_PATH"

# Transfer project files (excluding outputs and cache)
print_info "Transferring project files..."
rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='*.pyc' \
    . $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/

print_success "File transfer completed!"

print_info "Next steps on the new machine:"
echo "1. cd $NEW_MACHINE_PATH"
echo "2. chmod +x setup_machine.sh"
echo "3. ./setup_machine.sh"
echo "4. conda activate mole-py10"
echo "5. python scripts/check_guacamol_dataset.py"
echo "6. python scripts/run_crossenv_training.py"

print_note "Don't forget to install NVIDIA drivers and CUDA on the new machine if needed!" 