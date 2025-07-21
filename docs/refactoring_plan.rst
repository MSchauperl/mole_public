.. _refactoring-plan:

MolE Code Refactoring Plan
==========================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overall Goal
------------
- Refactor the existing MolE codebase into a highly modular structure.
- Use standard PyTorch components (`torch.nn.Module`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, manual training loops) as the primary building blocks.
- **Remove Hydra/OmegaConf dependency.** Manage configuration using Python's `argparse` for command-line arguments, supplemented by simple configuration files (e.g., YAML/JSON) loaded manually if necessary for complex defaults.
- Ensure every component (function, class, module) has clear docstrings.
- Implement unit tests for all components using `pytest`.

Proposed Modular Structure
-------------------------

.. code-block:: text

    mole/
        __init__.py
        config.py           # Configuration loading (parsing args, reading YAML/JSON if used), validation
        data/
            __init__.py
            vocabulary.py     # Vocabulary loading and management (e.g., open_dictionary)
            processing.py     # SMILES to molecule object, feature extraction (e.g., atom environments, distances using RDKit)
            datasets.py       # Core PyTorch Dataset classes (e.g., MolDataset logic)
            dataloaders.py    # Functions to create DataLoaders for train/val/test/predict
        nn/
            __init__.py
            attention.py      # Disentangled attention mechanism
            layers.py         # Basic building blocks (e.g., BertLayer, BertSelfOutput, ConvLayer)
            utils.py          # Low-level NN utilities (e.g., activation functions)
        models/
            __init__.py
            embeddings.py     # Embedding layers (e.g., AtomEnvEmbeddings)
            encoders.py       # Transformer encoder architecture (e.g., BertEncoder)
            heads.py          # Prediction heads (e.g., TaskPredictionHead)
            mole.py           # Main model assembling encoder and head (inheriting torch.nn.Module)
        metrics.py          # Metric definitions and calculation (using torchmetrics or custom)
        engine/             # Training and evaluation logic
            __init__.py
            trainer.py        # Core training loop (epoch/batch iteration, optimizer step, loss calculation, etc.)
            evaluator.py      # Evaluation loop
            utils.py          # Engine utilities (e.g., device placement, checkpointing helpers)
        inference.py        # Functions for prediction and encoding using trained models
        cli.py              # Command-line interface (using argparse, calling engine/inference functions)
        utils.py            # General utility functions
    configs/                # Optional: Directory for default YAML/JSON config files (if not solely using argparse)
        default_train.yaml
        default_finetune.yaml
    tests/
        __init__.py
        data/
            __init__.py
            test_vocabulary.py
            test_processing.py
            test_datasets.py
        nn/
            __init__.py
            test_attention.py
            test_layers.py
        models/
            __init__.py
            test_embeddings.py
            test_encoders.py
            test_heads.py
        engine/
            __init__.py
            test_trainer.py
            test_evaluator.py
        test_config.py
        test_metrics.py
        test_inference.py
        test_cli.py
    scripts/                # Optional: Helper scripts for data download, preprocessing etc.
    README.md
    LICENSE
    requirements.txt

Core Technology
---------------
- **Framework:** PyTorch. All models will be `torch.nn.Module` subclasses. Data handling will use `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`. Training loops will be implemented explicitly.
- **Configuration:** Primarily `argparse` for command-line arguments. Potentially `PyYAML` or `json` library for loading default configurations from files.
- **Dependencies:** RDKit (for cheminformatics), Pandas (data reading), NumPy, TorchMetrics (for metrics), Pytest (for testing), (`PyYAML` if using YAML configs). **Remove `hydra-core`, `omegaconf`**.

.. rubric:: (The rest of the plan continues in the same detailed, structured manner as in the Markdown file. For brevity, only the first sections are shown here. The full plan should be included in the actual file.)
