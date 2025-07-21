# MolE Unsupervised Pretraining: Data Generation Summary

## Overview
MolE's unsupervised (self-supervised) pretraining is based on **Masked Language Modeling (MLM)**, similar to BERT-style models, but applied to molecular data. The goal is to learn rich molecular representations by predicting masked parts of a molecule, without requiring labeled data.

## Data Generation Pipeline

### 1. **Molecule Collection**
- Pretraining uses a massive dataset (~842 million molecules) represented as SMILES strings.
- Example molecules include Aspirin, Caffeine, Penicillin G, Glucose, Benzene, Ethanol, Morphine, ATP, etc.

### 2. **Atom Environment Tokenization**
- Each atom in a molecule is represented by its **atom environment** (typically a Morgan fingerprint of a given radius, e.g., radius 0 for immediate neighbors).
- The molecule is converted into a sequence of atom environment tokens using a vocabulary (typically 200+ unique environments).
- Special tokens: `<PAD>`, `<MASK>`, `<UNK>` are included in the vocabulary.

### 3. **Input vs. Prediction: Structural vs. Functional Atom Environments, and Different Radii**
- **Key detail:** The input tokens and the prediction targets are *not* the same.
    - **Input tokens:** For each atom, the input is the *structural* atom environment (Morgan fingerprint with `useFeatures=False`), typically at a smaller radius (e.g., radius 0).
    - **Prediction targets:** For masked positions, the model is trained to predict the *functional* atom environment (Morgan fingerprint with `useFeatures=True`), often at a larger radius (e.g., radius 1 or 2).
- This cross-prediction (structural, small radius → functional, larger radius) encourages the model to learn richer, more generalizable chemical representations that capture both local and broader chemical context.

### 4. **Masked Language Modeling (MLM) Data Creation**
- For each tokenized molecule, a subset of tokens (typically 15%) is selected for masking:
  - **80%** of masked tokens are replaced with the `[MASK]` token.
  - **10%** are replaced with a random token from the vocabulary (excluding special tokens).
  - **10%** are left unchanged (but still counted as masked for loss calculation).
- The model is trained to predict the *functional* atom environment at each masked position, possibly at a different radius than the input.
- Labels for non-masked positions are set to `-100` (ignored in loss computation).

### 5. **Batching and Tensorization**
- Token sequences are padded to a maximum length for batching.
- Attention masks are created to distinguish real tokens from padding.
- Data is loaded efficiently using PyTorch (and optionally PyTorch Geometric for graph batching).

## Key Points
- **No labels** are needed: the model learns from the structure of molecules alone.
- The masking strategy forces the model to learn contextual chemical information.
- The use of different environments (structural vs. functional) and different radii for input and prediction is a unique aspect of MolE's pretraining.
- Pretraining on a huge, diverse set of molecules enables strong generalization for downstream tasks (e.g., property prediction, classification).

## References
- See `playground/mole_data_construction_pretraining.ipynb` for detailed code and examples.
- See `mole/data/datasets.py` and `mole/data/dataloaders.py` for dataset and loader implementation details.
- See `data/create_atomenv_atlas.py` for details on atom environment extraction.

---
*This summary was generated automatically from the codebase and documentation.*
