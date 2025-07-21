# MolE Code Refactoring Plan (Removing Hydra)

**I. Overall Goal:**

*   Refactor the existing MolE codebase into a highly modular structure.
*   Use standard PyTorch components (`torch.nn.Module`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, manual training loops) as the primary building blocks.
*   **Remove Hydra/OmegaConf dependency.** Manage configuration using Python's `argparse` for command-line arguments, supplemented by simple configuration files (e.g., YAML/JSON) loaded manually if necessary for complex defaults.
*   Ensure every component (function, class, module) has clear docstrings.
*   Implement unit tests for all components using `pytest`.

**II. Proposed Modular Structure:**

The codebase will be organized into distinct modules, each with a specific responsibility.

```
mole/
├── __init__.py
├── config.py           # Configuration loading (parsing args, reading YAML/JSON if used), validation
│
├── data/
│   ├── __init__.py
│   ├── vocabulary.py     # Vocabulary loading and management (e.g., open_dictionary)
│   ├── processing.py     # SMILES to molecule object, feature extraction (e.g., atom environments, distances using RDKit)
│   ├── datasets.py       # Core PyTorch Dataset classes (e.g., MolDataset logic)
│   └── dataloaders.py    # Functions to create DataLoaders for train/val/test/predict
│
├── nn/
│   ├── __init__.py
│   ├── attention.py      # Disentangled attention mechanism
│   ├── layers.py         # Basic building blocks (e.g., BertLayer, BertSelfOutput, ConvLayer)
│   └── utils.py          # Low-level NN utilities (e.g., activation functions)
│
├── models/
│   ├── __init__.py
│   ├── embeddings.py     # Embedding layers (e.g., AtomEnvEmbeddings)
│   ├── encoders.py       # Transformer encoder architecture (e.g., BertEncoder)
│   ├── heads.py          # Prediction heads (e.g., TaskPredictionHead)
│   └── mole.py           # Main model assembling encoder and head (inheriting torch.nn.Module)
│
├── metrics.py          # Metric definitions and calculation (using torchmetrics or custom)
│
├── engine/             # Training and evaluation logic
│   ├── __init__.py
│   ├── trainer.py        # Core training loop (epoch/batch iteration, optimizer step, loss calculation, etc.)
│   ├── evaluator.py      # Evaluation loop
│   └── utils.py          # Engine utilities (e.g., device placement, checkpointing helpers)
│
├── inference.py        # Functions for prediction and encoding using trained models
│
├── cli.py              # Command-line interface (using argparse, calling engine/inference functions)
│
└── utils.py            # General utility functions

configs/                # Optional: Directory for default YAML/JSON config files (if not solely using argparse)
    ├── default_train.yaml
    └── default_finetune.yaml

tests/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── test_vocabulary.py
│   ├── test_processing.py
│   └── test_datasets.py
├── nn/
│   ├── __init__.py
│   ├── test_attention.py
│   └── test_layers.py
├── models/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_encoders.py
│   └── test_heads.py
├── engine/
│   ├── __init__.py
│   ├── test_trainer.py
│   └── test_evaluator.py
├── test_config.py
├── test_metrics.py
├── test_inference.py
└── test_cli.py

scripts/                # Optional: Helper scripts for data download, preprocessing etc.

README.md
LICENSE
requirements.txt
```

**III. Core Technology:**

*   **Framework:** PyTorch. All models will be `torch.nn.Module` subclasses. Data handling will use `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`. Training loops will be implemented explicitly.
*   **Configuration:** Primarily `argparse` for command-line arguments. Potentially `PyYAML` or `json` library for loading default configurations from files.
*   **Dependencies:** RDKit (for cheminformatics), Pandas (data reading), NumPy, TorchMetrics (for metrics), Pytest (for testing), (`PyYAML` if using YAML configs). **Remove `hydra-core`, `omegaconf`**.

**IV. Data Handling Plan (`mole/data/`)**

1.  **`vocabulary.py`:** Encapsulate loading (`.pkl` files) and managing the atom environment vocabulary, including special tokens.
2.  **`processing.py`:** Contain functions that take a SMILES string, use RDKit to create a molecule object, calculate distance matrices, and extract atom environment features based on the loaded vocabulary and radius. These functions should be pure and testable.
3.  **`datasets.py`:** Define `torch.utils.data.Dataset` classes. The `__getitem__` method will use functions from `processing.py` to convert a SMILES string (read from a source like a DataFrame) into the required input tensors (token IDs, attention mask, distance representation).
4.  **`dataloaders.py`:** Provide functions that take configuration and dataset objects to instantiate and return configured `torch.utils.data.DataLoader` instances for training, validation, etc.

**V. Model Architecture Plan (`mole/nn/`, `mole/models/`)**

1.  **`mole/nn/`:** Keep low-level, reusable neural network components here (e.g., the attention mechanism, individual BERT-like layers). Ensure they are standard `torch.nn.Module`s.
2.  **`mole/models/`:**
    *   `embeddings.py`: Define the specific embedding layers used by MolE.
    *   `encoders.py`: Define the main transformer encoder stack, assembling layers from `mole/nn/`.
    *   `heads.py`: Define task-specific prediction heads.
    *   `mole.py`: Define the final `MolE` model class (a `torch.nn.Module`) that combines the encoder and potentially a head, defining the `forward` pass.

**VI. Training/Evaluation Engine Plan (`mole/engine/`)**

1.  **`trainer.py`:** Implement a class or functions responsible for the training loop:
    *   Iterating over epochs and batches from the `DataLoader`.
    *   Moving data to the correct device.
    *   Performing the model's forward pass.
    *   Calculating the loss (using the objective defined for pre-training or fine-tuning).
    *   Performing backpropagation and optimizer steps.
    *   Handling learning rate scheduling.
    *   Logging metrics (using `mole/metrics.py`).
    *   Basic checkpointing logic.
2.  **`evaluator.py`:** Implement a similar loop for evaluation, running the model in `eval()` mode and calculating/aggregating validation/test metrics.

**VII. Inference Plan (`mole/inference.py`)**

*   Define functions like `predict_properties` and `generate_embeddings`.
*   These functions will handle:
    *   Loading a trained model checkpoint.
    *   Setting the model to evaluation mode (`model.eval()`).
    *   Processing input SMILES (using `mole/data/processing.py` and creating dataloaders).
    *   Running the model's forward pass without gradient calculation (`torch.no_grad()`).
    *   Formatting and returning the predictions or embeddings.

**VIII. CLI Plan (`mole/cli.py`) (Revised):**

*   **Remove `@hydra.main` decorator.**
*   Implement argument parsing directly using Python's `argparse` module. Define arguments for all necessary configurations (dataset paths, model parameters, training hyperparameters, output directories, optional config file path).
*   Include an optional argument to specify a path to a YAML/JSON config file for loading defaults, which can be overridden by command-line arguments.
*   The main execution block will:
    1.  Parse arguments using `argparse`.
    2.  Optionally load defaults from a config file using `mole/config.py`.
    3.  Merge defaults and command-line arguments (CLI args take precedence) using `mole/config.py`.
    4.  Instantiate necessary components (data loaders, model, optimizer, etc.) based on the final configuration values.
    5.  Call the appropriate functions/methods from `mole/engine/` or `mole/inference.py`, passing the configuration values.

**IX. Configuration Plan (`mole/config.py`, `configs/`) (Revised):**

*   **`mole/config.py`:**
    *   Define functions to set up the `ArgumentParser` with all required arguments and reasonable defaults.
    *   Define functions to optionally load configurations from YAML/JSON files (if the `configs/` directory is used).
    *   Define functions to merge loaded file defaults with parsed command-line arguments.
    *   Potentially define dataclasses or simple objects to hold the validated configuration values, making them easier to pass around.
*   **`configs/` (Optional):**
    *   If used, store simple YAML or JSON files containing *default* parameters. These files will not use any special syntax like Hydra's interpolation. They serve only as a base configuration that can be overridden. If all configuration is manageable via command-line arguments, this directory might be omitted.

**X. Documentation Plan (Revised):**

*   **Docstrings:** Add Google-style or NumPy-style docstrings to *every* class, method, and function. Explain purpose, arguments (`Args:`), return values (`Returns:`), and potentially raised exceptions (`Raises:`). Include simple usage examples where helpful.
*   **Module Docstrings:** Add a docstring at the top of each `.py` file explaining the module's overall purpose.
*   **README.md:** **Crucially update** the setup and usage sections.
    *   Remove all mentions of Hydra.
    *   Clearly document all available command-line arguments provided by `argparse`.
    *   Explain how to use default config files (if implemented).
    *   Provide clear examples of how to run training, inference, etc., using the new command-line structure.
*   **Type Hinting:** Use Python type hints extensively to improve clarity and enable static analysis.

**XI. Testing Plan (`tests/`) (Revised):**

*   **Framework:** Use `pytest`.
*   **Structure:** Mirror the `mole/` source directory structure within `tests/`.
*   **Coverage:** Aim for high unit test coverage for all modules.
    *   Test data processing functions with known SMILES inputs and expected outputs.
    *   Test model components for correct output shapes and device compatibility.
    *   Test metric calculations.
    *   Test training/evaluation engine components (potentially mocking model forward/backward passes).
    *   Test inference functions.
    *   Test config loading. **Add tests for `mole/config.py`** (argument parsing, config file loading, merging logic).
    *   Test the CLI (`mole/cli.py`) by simulating command-line calls and checking if the correct functions are called with the expected parameters.
*   **Fixtures:** Use `pytest` fixtures for reusable setup (e.g., loading small test datasets, initializing models).
*   **Mocking:** Use `unittest.mock` where necessary to isolate components (e.g., mocking RDKit calls or file I/O).

**XII. Dependency Management:**

*   Update `requirements.txt` (or equivalent) to **remove `hydra-core` and `omegaconf`**. Add `PyYAML` if YAML config files are used. 