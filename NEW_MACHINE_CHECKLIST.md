# MolE New Machine Setup Checklist

## 📋 Pre-Setup Requirements

### Hardware Requirements
- [ ] NVIDIA GPU with ≥8GB VRAM (tested on RTX 3070)
- [ ] ≥16GB RAM recommended
- [ ] ≥50GB free disk space

### Software Requirements
- [ ] Linux OS (Ubuntu 18.04+ recommended)
- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] CUDA 12.4+ support
- [ ] Anaconda/Miniconda installed
- [ ] Git installed

## 🚀 Quick Setup (Automated)

### Option 1: Use Setup Script
```bash
# 1. Copy files to new machine
./transfer_files.sh username ip_address /path/to/destination

# 2. On new machine, run setup
cd /path/to/destination
chmod +x setup_machine.sh
./setup_machine.sh
```

### Option 2: Manual Setup
Follow the detailed guide in `setup_new_machine.md`

## 📁 Files to Transfer

### Essential Files (Must Have)
- [ ] `data/guacamol_v1_all.smiles` (74MB) - GuacaMol dataset
- [ ] `mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl`
- [ ] `mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl`
- [ ] `environment_mole_py10.yml` - Exact environment specification
- [ ] `setup_machine.sh` - Automated setup script
- [ ] Entire `mole/` directory - Source code
- [ ] Entire `scripts/` directory - Training scripts

### Optional Files (Can Regenerate)
- [ ] Other vocabulary files in `mole/data/vocabularies/`
- [ ] `outputs/` directory (training outputs - can skip)

## ✅ Verification Steps

### Environment Test
```bash
conda activate mole-py10

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test RDKit
python -c "from rdkit import Chem; print('RDKit: OK')"

# Test MolE imports
python -c "from mole.models.crossenv_mlm import CrossEnvMLM; print('MolE: OK')"
```

### Dataset Test
```bash
python scripts/check_guacamol_dataset.py
```

Expected output:
- Dataset: 1,591,378 molecules
- Input vocab: 173 tokens
- Target vocab: 1,472 tokens

### Training Test
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_crossenv_training.py
```

Expected GPU usage: ~800MB on RTX 3070

## 🔧 GPU Memory Configuration

### For RTX 3070 (8GB) - Current Settings
```python
"--hidden_size": "384"
"--num_hidden_layers": "6"
"--batch_size": "16"
"--accumulate_grad_batches": "8"
```

### For RTX 4090 (24GB) - Larger Model
```python
"--hidden_size": "768"
"--num_hidden_layers": "12"
"--batch_size": "64"
"--accumulate_grad_batches": "2"
```

### For RTX 3060 (12GB) - Medium Model
```python
"--hidden_size": "512"
"--num_hidden_layers": "8"
"--batch_size": "32"
"--accumulate_grad_batches": "4"
```

## 🐛 Common Issues & Solutions

### CUDA Out of Memory
1. Reduce `--batch_size` to 8 or 4
2. Increase `--accumulate_grad_batches` accordingly
3. Reduce `--hidden_size` to 256
4. Reduce `--max_length` to 80

### Environment Issues
1. Check CUDA compatibility: `nvcc --version`
2. Reinstall PyTorch: `conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia`
3. Clear conda cache: `conda clean --all`

### Import Errors
1. Reinstall MolE: `pip install -e .`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Reinstall DeBERTa: `pip install git+https://github.com/omendezlucio/DeBERTa.git --force-reinstall`

## 📊 Expected Performance

### Memory Usage (RTX 3070)
- GPU Memory: ~800MB
- System RAM: ~2GB
- Training Speed: ~2.5 it/s

### Training Time Estimates
- **Small Model (384 hidden)**: ~2-3 days for 30 epochs
- **Medium Model (512 hidden)**: ~3-4 days for 25 epochs
- **Large Model (768 hidden)**: ~4-5 days for 20 epochs

## 🎯 Success Criteria

Training is successful when:
- [ ] GPU utilization: 50-95%
- [ ] GPU memory usage: <80% of available
- [ ] Training loss decreasing
- [ ] No CUDA OOM errors
- [ ] Process runs for >30 minutes without crashing

## 📞 Support

If you encounter issues:
1. Check `setup_new_machine.md` for detailed troubleshooting
2. Verify all files transferred correctly
3. Check GPU drivers and CUDA installation
4. Compare `conda list` output with working machine

## 🎉 Final Verification

Once setup is complete:
- [ ] Environment activates: `conda activate mole-py10`
- [ ] All tests pass: Run verification steps above
- [ ] Dataset check passes: `python scripts/check_guacamol_dataset.py`
- [ ] Training starts: `python scripts/run_crossenv_training.py`
- [ ] GPU memory usage looks reasonable: `nvidia-smi`

**Setup Complete!** 🚀 