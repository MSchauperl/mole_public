.. _troubleshooting:

===============
Troubleshooting
===============

Common issues and solutions when working with MolE.

Installation Issues
===================

CUDA Not Available
------------------

If PyTorch cannot detect CUDA:

.. code-block:: bash

   # Check NVIDIA drivers
   nvidia-smi
   
   # Test CUDA in Python
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Reinstall PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RDKit Import Errors
-------------------

If RDKit fails to import:

.. code-block:: bash

   # Install via conda (recommended)
   conda install -c conda-forge rdkit
   
   # Or via pip
   pip install rdkit

Mac M1 Issues
-------------

For Mac users with M1 processors:

.. code-block:: bash

   # Set environment variable
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   
   # Add to bash profile
   echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> ~/.bashrc

Memory Issues
=============

Out of Memory During Training
-----------------------------

- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training
- Use smaller model variants

Out of Memory During Inference
------------------------------

- Process molecules in smaller batches
- Use CPU inference for large datasets
- Clear cache between batches

Data Issues
===========

SMILES Parsing Errors
---------------------

Ensure SMILES are valid:

.. code-block:: python

   from rdkit import Chem
   
   def validate_smiles(smiles):
       mol = Chem.MolFromSmiles(smiles)
       return mol is not None

Vocabulary Mismatch
-------------------

If atom environments are not found in vocabulary:

- Check vocabulary file path
- Ensure vocabulary matches the model
- Use `create_vocabularies.py` to generate new vocabulary

Training Issues
===============

Loss Not Decreasing
-------------------

- Check learning rate (try reducing)
- Verify data format and labels
- Ensure sufficient training data
- Check for data leakage

NaN Values in Loss
------------------

- Reduce learning rate
- Check for invalid SMILES in data
- Enable gradient clipping
- Verify data preprocessing

Model Not Loading
-----------------

- Check checkpoint file path
- Verify model architecture matches
- Ensure all dependencies are installed

Performance Issues
==================

Slow Training
-------------

- Enable multi-GPU training
- Use mixed precision (AMP)
- Optimize data loading (more workers)
- Use faster storage (SSD)

Slow Inference
--------------

- Use GPU acceleration
- Batch multiple predictions
- Use TorchScript compilation
- Consider model quantization

Getting Help
============

If you encounter issues not covered here:

1. Check the `GitHub Issues <https://github.com/recursionpharma/mole_public/issues>`_
2. Review the documentation thoroughly
3. Ensure you're using the latest version
4. Create a minimal reproducible example 