.. _installation:

============
Installation
============

This guide provides comprehensive installation instructions for HotSpot Therapeutics' enhanced MolE implementation on different platforms and setups.

Basic Installation
==================

Requirements
------------

- Python 3.9 or 3.10
- NVIDIA GPU with ≥8GB VRAM (for training)
- ≥16GB RAM recommended
- ≥50GB free disk space

Quick Install
-------------

.. code-block:: bash

   # Create virtual environment
   pyenv virtualenv mole-enhanced
   pyenv activate mole-enhanced

   # Clone repository (update with your internal repository URL)
   git clone <your-internal-repository-url>
   cd mole_enhanced

   # Install dependencies
   pip install -r requirements/main_3.10.txt  # or main_3.10_gpu.txt for CUDA

   # Install MolE Enhanced
   pip install -e .

For detailed setup instructions and troubleshooting, see the sections below.

New Machine Setup
=================

.. include:: NEW_MACHINE_CHECKLIST.md
   :parser: myst_parser.sphinx_

Detailed Setup Guide
====================

.. include:: setup_new_machine.md
   :parser: myst_parser.sphinx_

Environment Configuration
=========================

For specific environment configurations and Conda setup, please refer to the environment files in the ``requirements/`` directory:

- ``environment_mole_py10.yml`` - Complete Conda environment specification
- ``main_3.9.txt`` / ``main_3.10.txt`` - Pip requirements for CPU
- ``main_3.9_gpu.txt`` / ``main_3.10_gpu.txt`` - Pip requirements for GPU

Troubleshooting
===============

Mac Users
---------

If you're using a Mac with M1 processor, set the following environment variable:

.. code-block:: bash

   echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> .bashrc

GPU Setup
---------

Ensure NVIDIA drivers are properly installed and CUDA is available:

.. code-block:: bash

   # Test CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 