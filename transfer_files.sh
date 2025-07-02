#!/bin/bash
# File Transfer Script for MolE Setup
# Use this script to copy only the necessary data files (not in git repo) to a new machine

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "📦 MolE Data Files Transfer (Git Repo Already Exists)"
echo "===================================================="
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
    print_note "This script assumes the git repository is already cloned on the destination machine."
    echo ""
    print_info "Files that need to be transferred (not in git repo):"
    echo ""
    
    # Check what files exist and show sizes
    print_info "Data files:"
    if [ -f "data/guacamol_v1_all.smiles" ]; then
        echo "   ✅ data/guacamol_v1_all.smiles ($(du -h data/guacamol_v1_all.smiles | cut -f1))"
    else
        echo "   ❌ data/guacamol_v1_all.smiles (NOT FOUND)"
    fi
    
    print_info "Vocabulary files:"
    if [ -d "mole/data/vocabularies" ]; then
        vocab_count=$(find mole/data/vocabularies -name "*.pkl" | wc -l)
        vocab_size=$(du -sh mole/data/vocabularies 2>/dev/null | cut -f1)
        echo "   ✅ mole/data/vocabularies/ ($vocab_count .pkl files, $vocab_size total)"
        
        # List specific vocabulary files
        if [ -f "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl" ]; then
            echo "     ✅ vocabulary_radius0_structural_guacamol_v1.pkl"
        else
            echo "     ❌ vocabulary_radius0_structural_guacamol_v1.pkl (REQUIRED)"
        fi
        
        if [ -f "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl" ]; then
            echo "     ✅ vocabulary_radius1_functional_guacamol_v1.pkl"
        else
            echo "     ❌ vocabulary_radius1_functional_guacamol_v1.pkl (REQUIRED)"
        fi
    else
        echo "   ❌ mole/data/vocabularies/ (NOT FOUND)"
    fi
    
    print_info "Training outputs (optional):"
    if [ -d "outputs" ]; then
        outputs_size=$(du -sh outputs 2>/dev/null | cut -f1)
        echo "   ✅ outputs/ ($outputs_size) - Contains training checkpoints/logs"
    else
        echo "   ❌ outputs/ (No training outputs yet)"
    fi
    
    echo ""
    print_note "Manual transfer commands:"
    echo "scp data/guacamol_v1_all.smiles $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/data/"
    echo "scp -r mole/data/vocabularies/ $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/mole/data/"
    echo ""
    exit 1
fi

print_info "Transferring data files to $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH"
print_note "Assuming git repository is already cloned at destination"

# Check if required files exist
missing_files=0

if [ ! -f "data/guacamol_v1_all.smiles" ]; then
    print_error "Required file missing: data/guacamol_v1_all.smiles"
    missing_files=$((missing_files + 1))
fi

if [ ! -f "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl" ]; then
    print_error "Required file missing: mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl"
    missing_files=$((missing_files + 1))
fi

if [ ! -f "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl" ]; then
    print_error "Required file missing: mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl"
    missing_files=$((missing_files + 1))
fi

if [ $missing_files -gt 0 ]; then
    print_error "Cannot proceed: $missing_files required files are missing"
    print_note "Generate vocabularies with: python mole/data/create_vocabularies.py"
    exit 1
fi

# Transfer GuacaMol dataset
print_info "Transferring GuacaMol dataset..."
ssh $NEW_MACHINE_USER@$NEW_MACHINE_IP "mkdir -p $NEW_MACHINE_PATH/data"
scp data/guacamol_v1_all.smiles $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/data/

print_success "GuacaMol dataset transferred ($(du -h data/guacamol_v1_all.smiles | cut -f1))"

# Transfer vocabularies
print_info "Transferring vocabulary files..."
ssh $NEW_MACHINE_USER@$NEW_MACHINE_IP "mkdir -p $NEW_MACHINE_PATH/mole/data/vocabularies"
scp mole/data/vocabularies/*.pkl $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/mole/data/vocabularies/

print_success "Vocabularies transferred ($(du -sh mole/data/vocabularies | cut -f1))"

# Optional: Transfer training outputs if they exist
if [ -d "outputs" ]; then
    read -p "Transfer training outputs/checkpoints? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Transferring training outputs..."
        ssh $NEW_MACHINE_USER@$NEW_MACHINE_IP "mkdir -p $NEW_MACHINE_PATH/outputs"
        rsync -avz --progress outputs/ $NEW_MACHINE_USER@$NEW_MACHINE_IP:$NEW_MACHINE_PATH/outputs/
        print_success "Training outputs transferred"
    fi
fi

print_success "Data file transfer completed!"

echo ""
print_info "Files transferred:"
echo "✅ GuacaMol dataset: data/guacamol_v1_all.smiles"
echo "✅ Vocabularies: mole/data/vocabularies/*.pkl"

echo ""
print_info "Next steps on the destination machine:"
echo "1. cd $NEW_MACHINE_PATH"
echo "2. git pull  # Update to latest code if needed"
echo "3. conda activate mole-py10  # Or create environment if needed"
echo "4. python scripts/check_guacamol_dataset.py  # Verify files"
echo "5. python scripts/run_crossenv_training.py  # Start training"

echo ""
print_note "If environment doesn't exist on destination machine:"
echo "1. ./setup_machine.sh  # Will create environment from environment_mole_py10.yml" 