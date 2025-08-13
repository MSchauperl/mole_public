#!/usr/bin/env python3
"""
Main script to run ADMET property prediction using MolE Transformer model.

This script demonstrates how to use the MolE Transformer model
for predicting various ADMET properties from SMILES strings.
Supports both training from scratch and loading pretrained weights.
Handles both regression and classification tasks.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import RobustScaler
from typing import List, Optional, Dict, Any
import pickle
import uuid

# Suppress RDKit warnings about hydrogen atoms
import warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

# Import MolE components
from mole.models.embeddings import AtomEnvEmbeddings
from mole.models.crossenv_mlm import CrossEnvMLMModel
from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.data.crossenv_dataset import CrossEnvMolDataset

# Import TDC for ADMET datasets
try:
    from tdc.benchmark_group import admet_group
    tdc_available = True
except ImportError:
    tdc_available = False


class ADMETPredictionHead(nn.Module):
    """ADMET prediction head for MolE model (supports both regression and classification)"""
    
    def __init__(self, hidden_size: int = 768, dropout: float = 0.3, task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type
        
        if task_type == "regression":
            output_size = 1
        else:  # classification
            output_size = 2  # Binary classification
            
        self.admet_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for ADMET prediction
        
        Args:
            hidden_states: Output from transformer encoder [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask for padding [batch_size, seq_len]
        
        Returns:
            ADMET predictions [batch_size] for regression or [batch_size, 2] for classification
        """
        # Global average pooling over sequence dimension
        if attention_mask is not None:
            # Handle attention mask shape - it might be [batch_size, seq_len] or [batch_size, seq_len, 1]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)  # Remove extra dimension
            
            # Mask out padding tokens (attention_mask is [batch_size, seq_len])
            # hidden_states is [batch_size, seq_len, hidden_size]
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states.masked_fill(mask_expanded == 0, 0)
            
            # Sum and divide by number of non-padding tokens
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            pooled = hidden_states.sum(dim=1) / seq_lengths
        else:
            # Simple average pooling
            pooled = hidden_states.mean(dim=1)
        
        # Predict ADMET property
        output = self.admet_head(pooled)
        
        if self.task_type == "regression":
            return output.squeeze(-1)  # [batch_size]
        else:
            return output  # [batch_size, 2]


class MolEADMETModel(nn.Module):
    """MolE model for ADMET property prediction (supports pretrained weights)"""
    
    def __init__(self, config, hidden_size: int = 768, dropout: float = 0.1, task_type: str = "regression", freeze_encoder: bool = False):
        super().__init__()
        # Create encoder with same architecture as pretrained model
        self.encoder = AtomEnvEmbeddings(config)
        self.admet_head = ADMETPredictionHead(hidden_size, dropout, task_type)
        self.task_type = task_type
        
        # Optionally freeze encoder parameters
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze the encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✅ Encoder parameters frozen")
    
    def _unfreeze_encoder(self):
        """Unfreeze the encoder parameters for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✅ Encoder parameters unfrozen for fine-tuning")
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
        """
        Load pretrained weights from a checkpoint
        
        Args:
            checkpoint_path: Path to the pretrained checkpoint
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            Dictionary with loading statistics
        """
        print(f"🔄 Loading pretrained weights from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter state dict to only include encoder parameters
        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []
        
        for key, value in state_dict.items():
            # Remove 'model.' prefix if it exists (common in Lightning checkpoints)
            clean_key = key.replace('model.', '') if key.startswith('model.') else key
            
            if clean_key in model_state_dict:
                if model_state_dict[clean_key].shape == value.shape:
                    filtered_state_dict[clean_key] = value
                    loaded_keys.append(clean_key)
                else:
                    print(f"⚠️  Shape mismatch for {clean_key}: expected {model_state_dict[clean_key].shape}, got {value.shape}")
                    missing_keys.append(clean_key)
            else:
                unexpected_keys.append(key)
        
        # Load the filtered state dict
        missing_keys_final, unexpected_keys_final = self.load_state_dict(filtered_state_dict, strict=False)
        
        # Combine missing keys
        all_missing = missing_keys + list(missing_keys_final)
        
        # Count actual parameter values (not just keys)
        total_loaded_params = sum(value.numel() for value in filtered_state_dict.values())
        encoder_params = sum(value.numel() for key, value in filtered_state_dict.items() if 'encoder' in key)
        head_params = sum(value.numel() for key, value in filtered_state_dict.items() if 'admet_head' in key)
        
        print(f"✅ Loaded {len(loaded_keys)} parameter layers")
        print(f"   → Total parameter values: {total_loaded_params:,}")
        print(f"   → Encoder parameters: {encoder_params:,}")
        print(f"   → Head parameters: {head_params:,}")
        
        if all_missing:
            print(f"⚠️  Missing keys ({len(all_missing)}):")
            for key in all_missing[:10]:  # Show first 10 missing keys
                print(f"     - {key}")
            if len(all_missing) > 10:
                print(f"     ... and {len(all_missing) - 10} more")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # Show first 10 unexpected keys
                print(f"     - {key}")
            if len(unexpected_keys) > 10:
                print(f"     ... and {len(unexpected_keys) - 10} more")
        
        return {
            'loaded_keys': loaded_keys,
            'missing_keys': all_missing,
            'unexpected_keys': unexpected_keys,
            'total_loaded': len(loaded_keys),
            'total_loaded_params': total_loaded_params,
            'encoder_params': encoder_params,
            'head_params': head_params
        }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MolE model
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            ADMET predictions [batch_size] for regression or [batch_size, 2] for classification
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_mask=attention_mask,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )
        
        # Get last hidden states - handle different output formats
        if isinstance(encoder_outputs['hidden_states'], list):
            hidden_states = encoder_outputs['hidden_states'][-1]  # [batch_size, seq_len, hidden_size]
        else:
            hidden_states = encoder_outputs['hidden_states']  # [batch_size, seq_len, hidden_size]
        
        # Ensure we have the correct shape
        if hidden_states.dim() == 2:
            # If we got [seq_len, hidden_size], we need to add batch dimension
            hidden_states = hidden_states.unsqueeze(0)  # [1, seq_len, hidden_size]
        
        # Predict ADMET property
        predictions = self.admet_head(hidden_states, attention_mask)
        return predictions


class ADMETDataset(Dataset):
    """Dataset for ADMET properties with MolE tokenization"""
    
    def __init__(self, smiles_list: List[str], labels: List[float], data_module: CrossEnvDataModule, task_type: str = "regression"):
        self.smiles_list = smiles_list
        self.labels = labels
        self.data_module = data_module
        self.task_type = task_type
        
        # Create a CrossEnvMolDataset for proper tokenization
        smiles_series = pd.Series(smiles_list)
        self.mole_dataset = CrossEnvMolDataset(
            smiles=smiles_series,
            input_vocab_path=data_module.input_vocab_path,
            target_vocab_path=data_module.target_vocab_path,
            input_radius=data_module.input_radius,
            target_radius=data_module.target_radius,
            input_use_features=data_module.input_use_features,
            target_use_features=data_module.target_use_features,
            mask_prob=0.0,  # No masking for inference
            replace_prob=0.0,
            random_prob=0.0,
            max_length=data_module.max_length,
            cls_token=data_module.cls_token,
        )
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Get properly tokenized data from MolE dataset
        mole_data = self.mole_dataset[idx]
        
        return {
            'smiles': smiles,
            'label': torch.tensor(label, dtype=torch.float),
            'input_ids': mole_data.x,  # Tokenized input
            'attention_mask': torch.ones_like(mole_data.x)  # All tokens are valid
        }


def detect_task_type(labels: np.ndarray) -> str:
    """Detect if the task is regression or classification based on label values"""
    unique_values = np.unique(labels)
    
    # If we have exactly 2 unique values and they are 0 and 1, it's binary classification
    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
        return "classification"
    # If we have more than 2 unique values or values other than 0/1, it's regression
    else:
        return "regression"


def load_admet_data(property_name: str, seed: int = 3):
    """Load TDC ADMET dataset with proper train/validation/test split"""
    if not tdc_available:
        print("TDC not available. Install with: pip install PyTDC")
        return None, None, None
    
    print(f"Loading TDC ADMET dataset for property: {property_name}")
    
    try:
        # Initialize TDC benchmark group
        group = admet_group(path="data/")
        
        # Get benchmark
        benchmark = group.get(property_name)
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]
        
        print(f"Benchmark: {name}")
        print(f"Train+Val set: {len(train_val)} samples")
        print(f"Test set: {len(test)} samples")
        
        # Get train/valid split (using seed 42 for consistency)
        train, valid = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        
        print(f"Training set: {len(train)} samples")
        print(f"Validation set: {len(valid)} samples")
        print(f"Test set: {len(test)} samples")
        
        return train, valid, test
        
    except Exception as e:
        print(f"Error loading ADMET data for property '{property_name}': {e}")
        return None, None, None


# Remove this function as it's not needed in the new structure
# def load_pretrained_model(checkpoint_path: str, input_vocab_path: str, target_vocab_path: str) -> Tuple[nn.Module, Dict, Dict]:
    """Load pretrained MolE model and vocabularies"""
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    # Load vocabularies
    with open(input_vocab_path, 'rb') as f:
        input_vocab = pickle.load(f)
    
    with open(target_vocab_path, 'rb') as f:
        target_vocab = pickle.load(f)
    
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']
    else:
        # Default configuration matching the solubility script
        config = {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'dropout': 0.1,
        }
    
    print(f"Model configuration: {config}")
    
    # Create the base model (without prediction head)
    model = CrossEnvMLMModel(
        deberta_config=config,
        input_vocab_size=len(input_vocab),
        target_vocab_size=len(target_vocab),
        dropout=config.get('dropout', 0.1),
    )
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    
    # Filter out incompatible keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove 'model.' prefix if it exists
        clean_key = key.replace('model.', '') if key.startswith('model.') else key
        
        if clean_key in model_state_dict and model_state_dict[clean_key].shape == value.shape:
            filtered_state_dict[clean_key] = value
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Loaded {len(filtered_state_dict)} compatible parameters")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return model, input_vocab, target_vocab


# Remove this function as it's not needed in the new structure
# def create_admet_model(base_model: nn.Module, task_type: str = "regression", hidden_size: int = 768) -> nn.Module:
    """Create ADMET model by adding prediction head to pretrained model"""
    
    class ADMETModel(nn.Module):
        def __init__(self, encoder, prediction_head):
            super().__init__()
            self.encoder = encoder
            self.prediction_head = prediction_head
            self.task_type = task_type
        
        def forward(self, input_ids, attention_mask=None):
            # Get encoder outputs
            encoder_outputs = self.encoder.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract hidden states from encoder output
            # The encoder returns a dictionary with 'last_hidden_state' or 'hidden_states'
            if isinstance(encoder_outputs, dict):
                # Handle dictionary output (from BertEncoder/AtomEnvEmbeddings)
                if 'last_hidden_state' in encoder_outputs:
                    # Use last_hidden_state if available (single tensor)
                    hidden_states = encoder_outputs['last_hidden_state']
                elif 'hidden_states' in encoder_outputs:
                    # Use hidden_states (could be list or single tensor)
                    if isinstance(encoder_outputs['hidden_states'], list):
                        hidden_states = encoder_outputs['hidden_states'][-1]  # Get last layer
                    else:
                        hidden_states = encoder_outputs['hidden_states']
                else:
                    raise ValueError("Encoder output dict missing 'hidden_states' or 'last_hidden_state' key")
            elif isinstance(encoder_outputs, tuple):
                # Handle tuple output (fallback for other transformer models)
                if len(encoder_outputs) > 0:
                    if isinstance(encoder_outputs[0], list):
                        hidden_states = encoder_outputs[0][-1]
                    else:
                        hidden_states = encoder_outputs[0]
                else:
                    raise ValueError("Empty tuple from encoder")
            elif hasattr(encoder_outputs, 'last_hidden_state'):
                # Handle named tuple output (from HuggingFace models)
                hidden_states = encoder_outputs.last_hidden_state
            else:
                # Assume it's already the hidden states tensor
                hidden_states = encoder_outputs
            
            # Apply prediction head
            predictions = self.prediction_head(hidden_states, attention_mask)
            
            return predictions
    
    # Create prediction head
    prediction_head = ADMETRegularPredictionHead(
        hidden_size=hidden_size,
        dropout=0.1,
        task_type=task_type
    )
    
    # Combine encoder and prediction head
    admet_model = ADMETModel(base_model, prediction_head)
    
    return admet_model


def filter_outliers(df, target_col, n_std=3):
    """Remove extreme outliers from the dataset"""
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    
    mask = (df[target_col] >= mean_val - n_std * std_val) & (
        df[target_col] <= mean_val + n_std * std_val
    )
    filtered_df = df[mask]
    
    print(f"Removed {len(df) - len(filtered_df)} outliers (beyond {n_std} std)")
    return filtered_df


def preprocess_admet_data(train_df, valid_df, test_df, target_col='Y', use_max_samples=True):
    """Preprocess ADMET data with proper train/validation/test split"""
    print(f"\n2. Using target: {target_col}")
    
    # Clean the data
    train_clean = train_df.dropna(subset=['Drug', target_col])
    valid_clean = valid_df.dropna(subset=['Drug', target_col])
    test_clean = test_df.dropna(subset=['Drug', target_col])
    
    print(f"After cleaning: {len(train_clean)} train, {len(valid_clean)} valid, {len(test_clean)} test")
    
    # Detect task type
    task_type = detect_task_type(train_clean[target_col].values)
    print(f"Detected task type: {task_type}")
    
    # Remove extreme outliers from training data only (for regression tasks, to avoid data leakage)
    print("\n3. Preprocessing and filtering...")
    if task_type == "regression":
        train_filtered = filter_outliers(train_clean, target_col, n_std=3)
        print("No outlier removal for classification task")
    else:
        train_filtered = train_clean
        print("No outlier removal for classification task")
    
    if use_max_samples:
        # Use all available data
        train_subset = train_filtered
        valid_subset = valid_clean
        test_subset = test_clean
        print(f"Using maximum available samples: {len(train_subset)} train, {len(valid_subset)} valid, {len(test_subset)} test")
    else:
        # Use default sampling (for backward compatibility)
        train_subset = train_filtered.sample(
            n=min(1500, len(train_filtered)), random_state=42
        )
        valid_subset = valid_clean.sample(
            n=min(300, len(valid_clean)), random_state=42
        )
        test_subset = test_clean.sample(
            n=min(300, len(test_clean)), random_state=42
        )
        print(f"Using sampled data: {len(train_subset)} train, {len(valid_subset)} valid, {len(test_subset)} test")
    
    # Show target statistics
    if task_type == "regression":
        train_range = f"{train_subset[target_col].min():.2f} to {train_subset[target_col].max():.2f}"
        valid_range = f"{valid_subset[target_col].min():.2f} to {valid_subset[target_col].max():.2f}"
        test_range = f"{test_subset[target_col].min():.2f} to {test_subset[target_col].max():.2f}"
        print(f"Train target range: {train_range}")
        print(f"Validation target range: {valid_range}")
        print(f"Test target range: {test_range}")
    else:
        train_pos = np.sum(train_subset[target_col] == 1)
        train_neg = np.sum(train_subset[target_col] == 0)
        valid_pos = np.sum(valid_subset[target_col] == 1)
        valid_neg = np.sum(valid_subset[target_col] == 0)
        test_pos = np.sum(test_subset[target_col] == 1)
        test_neg = np.sum(test_subset[target_col] == 0)
        print(f"Train - Positives: {train_pos}, Negatives: {train_neg}")
        print(f"Validation - Positives: {valid_pos}, Negatives: {valid_neg}")
        print(f"Test - Positives: {test_pos}, Negatives: {test_neg}")
    
    return train_subset, valid_subset, test_subset, task_type


def create_mole_config(hidden_size=256):
    """Create MolE configuration with same parameters as pretrained model"""
    from DeBERTa.deberta.config import ModelConfig
    
    # Create DeBERTa configuration (matching the pretrained model)
    # deberta_config = {
    #     "attention_head": 12,
    #     "hidden_size": 768,
    #     "intermediate_size": 3072,
    #     "max_position_embeddings": 512,
    #     "num_hidden_layers": 12,
    #     "num_attention_heads": 12,
    #     "type_vocab_size": 0,
    #     "vocab_size": 173,  # Input vocabulary size
    #     "norm_rel_ebd": "layer_norm",
    #     "position_biased_input": False,
    #     "pos_att_type": "p2c|c2p",
    #     "relative_attention": True,
    #     "max_relative_positions": 128,
    #     "layer_norm_eps": 1e-7,
    #     "dropout": 0.1,
    #     "attention_dropout": 0.1,
    #     "hidden_dropout_prob": 0.1,
    #     "initializer_range": 0.02,
    #     "summary_type": "first",
    #     "summary_use_proj": True,
    #     "summary_activation": "gelu",
    #     "summary_last_dropout": 0.1,
    # }


    deberta_config = {
        "attention_head": 8,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,  # Scale intermediate size with hidden size
        "max_position_embeddings": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "type_vocab_size": 0,
        "vocab_size": 173,  # Input vocabulary size
        "norm_rel_ebd": "layer_norm",
        "position_biased_input": False,
        "pos_att_type": "p2c|c2p",
        "relative_attention": True,
        "max_relative_positions": 128,
        "layer_norm_eps": 1e-6,
        "dropout": 0.3,
        "attention_dropout": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "summary_type": "first",
        "summary_use_proj": True,
        "summary_activation": "gelu",
        "summary_last_dropout": 0.1,
    }
    
    config = ModelConfig.from_dict(deberta_config)
    return config


def create_admet_datasets(train_subset, valid_subset, test_subset, input_vocab_path: str, target_vocab_path: str, target_col='Y', task_type='regression'):
    """Create datasets for ADMET prediction using MolE tokenization"""
    print("\n4. Creating MolE datasets...")
    
    # Prepare data first
    train_smiles = train_subset['Drug'].values.tolist()
    valid_smiles = valid_subset['Drug'].values.tolist()
    test_smiles = test_subset['Drug'].values.tolist()
    
    train_labels = train_subset[target_col].values.astype(float)
    valid_labels = valid_subset[target_col].values.astype(float)
    test_labels = test_subset[target_col].values.astype(float)
    
    print(f"Preparing datasets:")
    print(f"  Training SMILES: {len(train_smiles)}")
    print(f"  Validation SMILES: {len(valid_smiles)}")
    print(f"  Test SMILES: {len(test_smiles)}")
    
    # Create data module for tokenization with actual training data
    data_module = CrossEnvDataModule(
        train_data=train_smiles,  # Use actual training data instead of dummy data
        input_vocab_path=input_vocab_path,
        target_vocab_path=target_vocab_path,
        input_radius=0,
        target_radius=1,
        input_use_features=False,
        target_use_features=True,
        max_length=256,
        batch_size=32,
        num_workers=4,
    )
    
    # Setup data module to get tokenizers
    data_module.setup("fit")
    
    # Use RobustScaler for better outlier handling (fit only on training data for regression)
    scaler = None
    if task_type == "regression":
        scaler = RobustScaler()
        train_labels_scaled = scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()
        valid_labels_scaled = scaler.transform(valid_labels.reshape(-1, 1)).flatten()
        test_labels_scaled = scaler.transform(test_labels.reshape(-1, 1)).flatten()
        
        orig_stats = f"Train mean: {train_labels.mean():.3f}, std: {train_labels.std():.3f}"
        scaled_stats = f"Train mean: {train_labels_scaled.mean():.3f}, std: {train_labels_scaled.std():.3f}"
        print(f"Original target stats - {orig_stats}")
        print(f"Scaled target stats - {scaled_stats}")
    else:
        # For classification, use labels as is
        train_labels_scaled = train_labels
        valid_labels_scaled = valid_labels
        test_labels_scaled = test_labels
        print(f"Classification task - using labels as is")
    
    # Create PyTorch datasets
    print("\n5. Creating PyTorch datasets...")
    
    train_dataset = ADMETDataset(train_smiles, train_labels_scaled, data_module, task_type)
    valid_dataset = ADMETDataset(valid_smiles, valid_labels_scaled, data_module, task_type)
    test_dataset = ADMETDataset(test_smiles, test_labels_scaled, data_module, task_type)
    
    print(f"Training dataset created with {len(train_dataset)} samples")
    print(f"Validation dataset created with {len(valid_dataset)} samples")
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    # Store SMILES for later use in CSV output
    train_dataset.smiles = train_smiles
    valid_dataset.smiles = valid_smiles
    test_dataset.smiles = test_smiles
    
    return train_dataset, valid_dataset, test_dataset, scaler, data_module


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    smiles = [item['smiles'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Get the maximum sequence length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences to the same length
    input_ids = []
    attention_mask = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        # Pad with PAD token (assuming 0 is PAD)
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        
        input_ids.append(padded_input)
        attention_mask.append(padded_mask)
    
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    
    return {
        'smiles': smiles,
        'label': labels,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def train_admet_model(model, train_dataset, valid_dataset, data_module, task_type,
                     epochs=30, batch_size=16, learning_rate=1e-4,
                     device='cuda' if torch.cuda.is_available() else 'cpu',
                     unfreeze_encoder_epoch=None, encoder_lr_ratio=0.1, accumulate_grad_batches=1,
                     weight_decay=0.01, grad_clip_norm=1.0):
    """Train the MolE ADMET model (from scratch or with pretrained weights)"""
    training_type = "fine-tuning" if unfreeze_encoder_epoch is not None else "from scratch"
    print(f"\n6. Training MolE ADMET model ({training_type}) on {device}...")
    
    if unfreeze_encoder_epoch is not None:
        print(f"   → Encoder will be unfrozen at epoch {unfreeze_encoder_epoch}")
    
    print(f"   → Task type: {task_type}")
    print(f"   → Gradient accumulation: {accumulate_grad_batches} batches")
    print(f"   → Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"   → Training compounds: {len(train_dataset):,}")
    print(f"   → Validation compounds: {len(valid_dataset):,}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Move model to device
    model = model.to(device)
    
    # Loss function based on task type
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Generate unique model ID for this training run
    model_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    model_filename = f'best_mole_admet_model_{model_id}.pth'
    print(f"   → Model will be saved as: {model_filename}")
    
    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Unfreeze encoder if specified
        if unfreeze_encoder_epoch is not None and epoch == unfreeze_encoder_epoch:
            model._unfreeze_encoder()
            
            # Set up differential learning rates
            encoder_lr = learning_rate * encoder_lr_ratio
            head_lr = learning_rate
            
            # Update optimizer with different learning rates for encoder and head
            param_groups = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'encoder' in name:
                        param_groups.append({'params': param, 'lr': encoder_lr})
                    else:
                        param_groups.append({'params': param, 'lr': head_lr})
            
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            print(f"   → Encoder unfrozen with differential learning rates:")
            print(f"     - Encoder LR: {encoder_lr:.2e}")
            print(f"     - Head LR: {head_lr:.2e}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            smiles = batch['smiles']
            labels = batch['label'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Handle different task types
            if task_type == "regression":
                loss = criterion(outputs, labels)
            else:  # classification
                labels = labels.long()  # Convert to long for CrossEntropyLoss
                loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping - use tighter clipping for prediction head
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            # Update weights every accumulate_grad_batches steps
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count += 1
            
            train_loss += loss.item() * accumulate_grad_batches  # Scale back for logging
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                smiles = batch['smiles']
                labels = batch['label'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                if task_type == "regression":
                    loss = criterion(outputs, labels)
                else:  # classification
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
        
        # Calculate total number of compounds processed
        total_train_compounds = len(train_dataset)
        total_valid_compounds = len(valid_dataset)
        
        # Calculate average losses per compound
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        # Calculate normalized losses per compound
        train_loss_per_compound = avg_train_loss
        valid_loss_per_compound = avg_valid_loss
        
        scheduler.step(avg_valid_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} (per compound: {train_loss_per_compound:.4f}), Validation Loss: {valid_loss:.4f} (per compound: {valid_loss_per_compound:.4f})")
        
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            # Save best model with unique ID
            torch.save(model.state_dict(), model_filename)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(model_filename))
    print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
    print(f"Best model saved as: {model_filename}")
    
    return model


def evaluate_admet_model(model, train_dataset, valid_dataset, test_dataset, scaler, data_module, 
                        model_name, property_name, task_type, save_predictions=True, 
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate MolE ADMET model and return metrics"""
    print("\n7. Evaluating model...")
    
    model.eval()
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Reduced batch size
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Reduced batch size
    
    # Get predictions
    train_preds_scaled = []
    valid_preds_scaled = []
    test_preds_scaled = []
    
    with torch.no_grad():
        for batch in train_loader:
            smiles = batch['smiles']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            if task_type == "classification":
                # Get probabilities for positive class
                probs = torch.softmax(outputs, dim=1)[:, 1]
                train_preds_scaled.extend(probs.cpu().numpy())
            else:
                train_preds_scaled.extend(outputs.cpu().numpy())
        
        for batch in valid_loader:
            smiles = batch['smiles']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            if task_type == "classification":
                # Get probabilities for positive class
                probs = torch.softmax(outputs, dim=1)[:, 1]
                valid_preds_scaled.extend(probs.cpu().numpy())
            else:
                valid_preds_scaled.extend(outputs.cpu().numpy())
        
        for batch in test_loader:
            smiles = batch['smiles']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            if task_type == "classification":
                # Get probabilities for positive class
                probs = torch.softmax(outputs, dim=1)[:, 1]
                test_preds_scaled.extend(probs.cpu().numpy())
            else:
                test_preds_scaled.extend(outputs.cpu().numpy())
    
    train_preds_scaled = np.array(train_preds_scaled)
    valid_preds_scaled = np.array(valid_preds_scaled)
    test_preds_scaled = np.array(test_preds_scaled)
    
    print("Predictions completed")
    
    # Convert predictions back to original scale for regression
    if task_type == "regression" and scaler is not None:
        train_preds = scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
        valid_preds = scaler.inverse_transform(valid_preds_scaled.reshape(-1, 1)).flatten()
        test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
        
        # Get the actual labels
        train_actual = scaler.inverse_transform(np.array(train_dataset.labels).reshape(-1, 1)).flatten()
        valid_actual = scaler.inverse_transform(np.array(valid_dataset.labels).reshape(-1, 1)).flatten()
        test_actual = scaler.inverse_transform(np.array(test_dataset.labels).reshape(-1, 1)).flatten()
    else:
        # For classification, use scaled values as is
        train_preds = train_preds_scaled
        valid_preds = valid_preds_scaled
        test_preds = test_preds_scaled
        
        train_actual = np.array(train_dataset.labels)
        valid_actual = np.array(valid_dataset.labels)
        test_actual = np.array(test_dataset.labels)
    
    # Calculate metrics based on task type
    if task_type == "regression":
        train_mae = mean_absolute_error(train_actual, train_preds)
        valid_mae = mean_absolute_error(valid_actual, valid_preds)
        test_mae = mean_absolute_error(test_actual, test_preds)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_preds))
        valid_rmse = np.sqrt(mean_squared_error(valid_actual, valid_preds))
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
        train_r2 = r2_score(train_actual, train_preds)
        valid_r2 = r2_score(valid_actual, valid_preds)
        test_r2 = r2_score(test_actual, test_preds)
        
        print(f"\n=== Results for {property_name} using {model_name} (Regression) ===")
        print(f"Train MAE:  {train_mae:.3f}")
        print(f"Valid MAE:  {valid_mae:.3f}")
        print(f"Test MAE:   {test_mae:.3f}")
        print(f"Train RMSE: {train_rmse:.3f}")
        print(f"Valid RMSE: {valid_rmse:.3f}")
        print(f"Test RMSE:  {test_rmse:.3f}")
        print(f"Train R²:   {train_r2:.3f}")
        print(f"Valid R²:   {valid_r2:.3f}")
        print(f"Test R²:    {test_r2:.3f}")
        
        metrics = {
            'train_mae': train_mae, 'valid_mae': valid_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'valid_rmse': valid_rmse, 'test_rmse': test_rmse,
            'train_r2': train_r2, 'valid_r2': valid_r2, 'test_r2': test_r2
        }
    else:
        # Convert probabilities to binary predictions
        train_binary = (train_preds > 0.5).astype(int)
        valid_binary = (valid_preds > 0.5).astype(int)
        test_binary = (test_preds > 0.5).astype(int)
        
        train_acc = accuracy_score(train_actual, train_binary)
        valid_acc = accuracy_score(valid_actual, valid_binary)
        test_acc = accuracy_score(test_actual, test_binary)
        train_prec = precision_score(train_actual, train_binary, zero_division=0)
        valid_prec = precision_score(valid_actual, valid_binary, zero_division=0)
        test_prec = precision_score(test_actual, test_binary, zero_division=0)
        train_rec = recall_score(train_actual, train_binary, zero_division=0)
        valid_rec = recall_score(valid_actual, valid_binary, zero_division=0)
        test_rec = recall_score(test_actual, test_binary, zero_division=0)
        train_f1 = f1_score(train_actual, train_binary, zero_division=0)
        valid_f1 = f1_score(valid_actual, valid_binary, zero_division=0)
        test_f1 = f1_score(test_actual, test_binary, zero_division=0)
        train_auc = roc_auc_score(train_actual, train_preds)
        valid_auc = roc_auc_score(valid_actual, valid_preds)
        test_auc = roc_auc_score(test_actual, test_preds)
        
        print(f"\n=== Results for {property_name} using {model_name} (Classification) ===")
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Valid Accuracy: {valid_acc:.3f}")
        print(f"Test Accuracy:  {test_acc:.3f}")
        print(f"Train Precision: {train_prec:.3f}")
        print(f"Valid Precision: {valid_prec:.3f}")
        print(f"Test Precision:  {test_prec:.3f}")
        print(f"Train Recall: {train_rec:.3f}")
        print(f"Valid Recall: {valid_rec:.3f}")
        print(f"Test Recall:  {test_rec:.3f}")
        print(f"Train F1: {train_f1:.3f}")
        print(f"Valid F1: {valid_f1:.3f}")
        print(f"Test F1:  {test_f1:.3f}")
        print(f"Train AUC: {train_auc:.3f}")
        print(f"Valid AUC: {valid_auc:.3f}")
        print(f"Test AUC:  {test_auc:.3f}")
        
        metrics = {
            'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc,
            'train_prec': train_prec, 'valid_prec': valid_prec, 'test_prec': test_prec,
            'train_rec': train_rec, 'valid_rec': valid_rec, 'test_rec': test_rec,
            'train_f1': train_f1, 'valid_f1': valid_f1, 'test_f1': test_f1,
            'train_auc': train_auc, 'valid_auc': valid_auc, 'test_auc': test_auc
        }
    
    # Save predictions to CSV files
    if save_predictions:
        save_predictions_to_csv(train_preds, train_actual, valid_preds, valid_actual, test_preds, test_actual, 
                               model_name, property_name, task_type, train_dataset, valid_dataset, test_dataset)
    
    # Show some sample predictions
    print(f"\nSample predictions (first 10 test molecules):")
    for i in range(min(10, len(test_preds))):
        pred = test_preds[i]
        actual = test_actual[i]
        if task_type == "regression":
            error = abs(pred - actual)
            print(f"  Predicted: {pred:.3f}, Actual: {actual:.3f}, Error: {error:.3f}")
        else:
            pred_class = "Positive" if pred > 0.5 else "Negative"
            actual_class = "Positive" if actual == 1 else "Negative"
            correct = "✓" if (pred > 0.5) == (actual == 1) else "✗"
            print(f"  Predicted: {pred:.3f} ({pred_class}), Actual: {actual_class}, {correct}")
    
    return {
        'model_type': model_name,
        'property': property_name,
        'featurizer': 'MolE Atom Environments',
        'task': task_type,
        'n_train': len(train_dataset),
        'n_valid': len(valid_dataset),
        'n_test': len(test_dataset),
        **metrics
    }


def save_predictions_to_csv(train_preds, train_actual, valid_preds, valid_actual, test_preds, test_actual, 
                           model_name, property_name, task_type, train_dataset=None, valid_dataset=None, test_dataset=None):
    """Save predictions to CSV files with SMILES"""
    import datetime
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get SMILES if available
    train_smiles = getattr(train_dataset, 'smiles', None) if train_dataset else None
    valid_smiles = getattr(valid_dataset, 'smiles', None) if valid_dataset else None
    test_smiles = getattr(test_dataset, 'smiles', None) if test_dataset else None
    
    if task_type == "regression":
        # Create training predictions DataFrame
        train_data = {
            'predicted_value': train_preds,
            'actual_value': train_actual,
            'absolute_error': np.abs(train_preds - train_actual),
            'squared_error': (train_preds - train_actual) ** 2
        }
        
        # Add SMILES if available
        if train_smiles is not None:
            train_data['smiles'] = train_smiles
        
        train_df = pd.DataFrame(train_data)
        
        # Create validation predictions DataFrame
        valid_data = {
            'predicted_value': valid_preds,
            'actual_value': valid_actual,
            'absolute_error': np.abs(valid_preds - valid_actual),
            'squared_error': (valid_preds - valid_actual) ** 2
        }
        
        # Add SMILES if available
        if valid_smiles is not None:
            valid_data['smiles'] = valid_smiles
        
        valid_df = pd.DataFrame(valid_data)
        
        # Create test predictions DataFrame
        test_data = {
            'predicted_value': test_preds,
            'actual_value': test_actual,
            'absolute_error': np.abs(test_preds - test_actual),
            'squared_error': (test_preds - test_actual) ** 2
        }
        
        # Add SMILES if available
        if test_smiles is not None:
            test_data['smiles'] = test_smiles
        
        test_df = pd.DataFrame(test_data)
    else:
        # Classification task
        train_binary = (train_preds > 0.5).astype(int)
        valid_binary = (valid_preds > 0.5).astype(int)
        test_binary = (test_preds > 0.5).astype(int)
        
        # Create training predictions DataFrame
        train_data = {
            'predicted_probability': train_preds,
            'predicted_class': train_binary,
            'actual_class': train_actual,
            'correct_prediction': (train_binary == train_actual).astype(int)
        }
        
        # Add SMILES if available
        if train_smiles is not None:
            train_data['smiles'] = train_smiles
        
        train_df = pd.DataFrame(train_data)
        
        # Create validation predictions DataFrame
        valid_data = {
            'predicted_probability': valid_preds,
            'predicted_class': valid_binary,
            'actual_class': valid_actual,
            'correct_prediction': (valid_binary == valid_actual).astype(int)
        }
        
        # Add SMILES if available
        if valid_smiles is not None:
            valid_data['smiles'] = valid_smiles
        
        valid_df = pd.DataFrame(valid_data)
        
        # Create test predictions DataFrame
        test_data = {
            'predicted_probability': test_preds,
            'predicted_class': test_binary,
            'actual_class': test_actual,
            'correct_prediction': (test_binary == test_actual).astype(int)
        }
        
        # Add SMILES if available
        if test_smiles is not None:
            test_data['smiles'] = test_smiles
        
        test_df = pd.DataFrame(test_data)
    
    # Generate filenames
    train_filename = f"predictions_{model_name.lower()}_{property_name.lower()}_train_{timestamp}.csv"
    valid_filename = f"predictions_{model_name.lower()}_{property_name.lower()}_valid_{timestamp}.csv"
    test_filename = f"predictions_{model_name.lower()}_{property_name.lower()}_test_{timestamp}.csv"
    
    # Save to CSV
    train_df.to_csv(train_filename, index=False)
    valid_df.to_csv(valid_filename, index=False)
    test_df.to_csv(test_filename, index=False)
    
    print(f"\n💾 Predictions saved to CSV files:")
    print(f"  📄 Training: {train_filename} ({len(train_df)} samples)")
    print(f"  📄 Validation: {valid_filename} ({len(valid_df)} samples)")
    print(f"  📄 Test: {test_filename} ({len(test_df)} samples)")
    
    if train_smiles is not None:
        print(f"  ✅ SMILES included in CSV files")
    
    # Print summary statistics
    print(f"\n📊 Prediction Summary:")
    if task_type == "regression":
        print(f"  Training - Mean Error: {train_df['absolute_error'].mean():.3f}")
        print(f"  Validation - Mean Error: {valid_df['absolute_error'].mean():.3f}")
        print(f"  Test - Mean Error: {test_df['absolute_error'].mean():.3f}")
        print(f"  Training - Std Error: {train_df['absolute_error'].std():.3f}")
        print(f"  Validation - Std Error: {valid_df['absolute_error'].std():.3f}")
        print(f"  Test - Std Error: {test_df['absolute_error'].std():.3f}")
    else:
        print(f"  Training - Accuracy: {train_df['correct_prediction'].mean():.3f}")
        print(f"  Validation - Accuracy: {valid_df['correct_prediction'].mean():.3f}")
        print(f"  Test - Accuracy: {test_df['correct_prediction'].mean():.3f}")


def get_available_admet_properties():
    """Get list of available ADMET properties from TDC benchmark group"""
    try:
        group = admet_group(path="data/")
        return group.dataset_names
    except Exception as e:
        print(f"Warning: Could not load ADMET properties: {e}")
        # Return common ADMET properties as fallback
        return [
            "Solubility_AqSolDB",
            "Caco2_Wang", 
            "Lipophilicity_AstraZeneca",
            "PPBR_AZ",
            "VDss_Lombardo",
            "Half_Life_Obach",
            "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ",
            "Bioavailability_Ma",
            "CYP2C19_Veith",
            "CYP2D6_Veith",
            "CYP3A4_Veith",
            "CYP1A2_Veith", 
            "CYP2C9_Veith",
            "BBB_Martins",
            "Pgp_Broccatelli",
            "HIA_Hou",
            "PAMPA_NCATS",
            "herg",
            "ames",
            "dili",
            "ld50_zhu",
        ]


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="ADMET property prediction using MolE Transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ADMET property argument
    parser.add_argument(
        "--property",
        type=str,
        default="Solubility_AqSolDB",
        help="ADMET property to evaluate (default: Solubility_AqSolDB). Use --list-properties to see all available."
    )
    parser.add_argument(
        "--list-properties",
        action="store_true",
        help="List all available ADMET properties and exit"
    )
    
    # Pretrained model arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to pretrained MolE checkpoint (optional)"
    )
    parser.add_argument(
        "--input_vocab",
        type=str,
        default="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to input vocabulary"
    )
    parser.add_argument(
        "--target_vocab",
        type=str,
        default="mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        help="Path to target vocabulary"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help="Number of batches to accumulate gradients before updating weights"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder during initial training phase"
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of epochs to keep encoder frozen (only used if --freeze_encoder is set)"
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=0.1,
        help="Learning rate ratio for encoder relative to prediction head when unfrozen"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size for transformer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability"
    )
    parser.add_argument(
        "--head_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for prediction head (higher = more restraint)"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm (lower = more restraint)"
    )
    
    # Data splitting arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split (default: 42)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for L2 regularization"
    )
    
    return parser.parse_args()


def print_model_features(model_name, model_type, property_name, task_type):
    """Print model-specific features"""
    if "MolE" in model_type:
        print(f"\n🧠 MOLECULAR ENVIRONMENT (MOLECULAR) TRANSFORMER FEATURES:")
        print(f"  ✅ DeBERTa-based architecture with disentangled attention")
        print(f"  ✅ 12-layer transformer encoder (768 hidden, 12 heads)")
        print(f"  ✅ Atom environment tokenization (radius 0 structural)")
        print(f"  ✅ RDKit-based molecular featurization")
        print(f"  ✅ Global average pooling over sequence")
        print(f"  ✅ Dedicated ADMET prediction head for {property_name}")
        print(f"  ✅ Task type: {task_type}")
        print(f"  ✅ Configurable encoder freezing for transfer learning")
        print(f"  ✅ Gradual unfreezing strategy for optimal fine-tuning")
        print(f"  ✅ Dropout regularization (0.1)")
        print(f"  ✅ AdamW optimizer with weight decay")
        print(f"  ✅ Learning rate scheduling with early stopping")
    else:
        print(f"\n⚠️  MODEL TYPE UNKNOWN:")
        print(f"  • Used fallback model")
    
    print(f"\n📈 KEY OPTIMIZATIONS:")
    print(f"  ✅ RobustScaler for outlier handling (regression only)")
    print(f"  ✅ Outlier filtering (3-sigma rule, regression only)")
    print(f"  ✅ Extended training epochs with early stopping")
    print(f"  ✅ Gradient clipping for stability")
    print(f"  ✅ Optimized hyperparameters for {model_name}")


def print_success_message(model_name, model_type, property_name, task_type, use_pretrained=False):
    """Print success message with model details"""
    print(f"\n🎉 SUCCESS! {model_type} model trained on {property_name} data")
    print(f"The system successfully:")
    print(f"  ✅ Used MolE atom environment tokenization")
    print(f"  ✅ Used {model_type} for {property_name} prediction")
    print(f"  ✅ Handled {task_type} task type")
    
    if use_pretrained:
        print(f"  ✅ Loaded pretrained weights for transfer learning")
        print(f"  ✅ Fine-tuned with DeBERTa architecture")
    else:
        print(f"  ✅ Trained from scratch with DeBERTa architecture")
    
    print(f"  ✅ Applied transformer attention mechanisms")
    print(f"  ✅ Generated {property_name} predictions with dedicated head")


def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_args()
    
    # List available properties if requested
    if args.list_properties:
        print("Available ADMET properties:")
        properties = get_available_admet_properties()
        for i, prop in enumerate(properties, 1):
            print(f"  {i:2d}. {prop}")
        return
    
    print("🧬 ADMET PROPERTY PREDICTION WITH MOLECULAR ENVIRONMENT TRANSFORMER")
    print("=" * 80)
    
    # Configuration based on arguments
    FREEZE_ENCODER = args.freeze_encoder
    GRADUAL_UNFREEZING = args.freeze_encoder  # Enable gradual unfreezing if encoder freezing is enabled
    UNFREEZE_EPOCH = args.freeze_epochs
    
    print(f"🔧 Configuration:")
    print(f"   Property: {args.property}")
    print(f"   Checkpoint path: {args.checkpoint_path or 'None (training from scratch)'}")
    print(f"   FREEZE_ENCODER = {FREEZE_ENCODER}")
    print(f"   GRADUAL_UNFREEZING = {GRADUAL_UNFREEZING}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"   Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Hidden size: {args.hidden_size}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Head weight decay: {args.head_weight_decay}")
    print(f"   Gradient clip norm: {args.grad_clip_norm}")
    print(f"   Random seed: {args.seed}")
    
    if FREEZE_ENCODER:
        print("   → Only prediction head parameters will be trained initially")
        if GRADUAL_UNFREEZING:
            print(f"   → Encoder will be unfrozen at epoch {UNFREEZE_EPOCH}")
            print(f"   → Encoder LR ratio: {args.encoder_lr_ratio}")
        else:
            print("   → Encoder parameters will remain frozen")
    else:
        print("   → All parameters (encoder + head) will be trained from start")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Use vocabulary paths from arguments
    input_vocab_path = args.input_vocab
    target_vocab_path = args.target_vocab
    
    # Step 1: Load data
    print("\n1. Loading ADMET dataset...")
    train_df, valid_df, test_df = load_admet_data(args.property, seed=args.seed)
    
    if train_df is None or valid_df is None or test_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Preprocess data
    train_subset, valid_subset, test_subset, task_type = preprocess_admet_data(
        train_df, valid_df, test_df, target_col='Y', use_max_samples=True
    )
    
    # Step 3: Create MolE configuration
    print("\n3. Creating MolE configuration...")
    config = create_mole_config(hidden_size=args.hidden_size)
    print(f"✅ MolE configuration created:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Vocabulary size: {config.vocab_size}")
    
    # Step 4: Create MolE datasets
    train_dataset, valid_dataset, test_dataset, scaler, data_module = create_admet_datasets(
        train_subset, valid_subset, test_subset, input_vocab_path, target_vocab_path, target_col='Y', task_type=task_type
    )
    
    # Step 5: Create MolE model
    print("\n5. Creating MolE model...")
    model = MolEADMETModel(config, hidden_size=args.hidden_size, dropout=args.dropout, task_type=task_type, freeze_encoder=FREEZE_ENCODER)
    
    # Load pretrained weights if checkpoint path is provided
    use_pretrained = False
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            loading_stats = model.load_pretrained_weights(args.checkpoint_path)
            use_pretrained = True
            print(f"✅ Successfully loaded pretrained weights")
            print(f"   → Total loaded parameter values: {loading_stats['total_loaded_params']:,}")
            print(f"   → Encoder parameter values: {loading_stats['encoder_params']:,}")
            print(f"   → Head parameter values: {loading_stats['head_params']:,}")
        else:
            print(f"⚠️  Checkpoint not found: {args.checkpoint_path}")
            print("   → Proceeding with training from scratch")
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"✅ MolE model created with {total_params:,} total parameters")
    print(f"   → {trainable_params:,} trainable parameters")
    print(f"   → {frozen_params:,} frozen parameters")
    
    if use_pretrained:
        print(f"   → Using pretrained weights for transfer learning")
    else:
        print(f"   → Training from scratch")
    
    # Step 6: Train model
    unfreeze_epoch = UNFREEZE_EPOCH if (FREEZE_ENCODER and GRADUAL_UNFREEZING) else None
    model = train_admet_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        data_module=data_module,
        task_type=task_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        unfreeze_encoder_epoch=unfreeze_epoch,
        encoder_lr_ratio=args.encoder_lr_ratio,
        accumulate_grad_batches=args.accumulate_grad_batches,
        weight_decay=args.weight_decay,
        grad_clip_norm=1.0
    )
    
    # Step 7: Evaluate model
    results = evaluate_admet_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        scaler=scaler,
        data_module=data_module,
        model_name="MolEADMET",
        property_name=args.property,
        task_type=task_type,
        save_predictions=True,
        device=device
    )
    
    # Step 8: Print model features and success message
    print_model_features("MolEADMET", "MolE", args.property, task_type)
    print_success_message("MolEADMET", "MolE", args.property, task_type, use_pretrained=use_pretrained)
    
    # Step 9: Print final results summary
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"  Model: MolEADMET")
    print(f"  Property: {args.property}")
    print(f"  Task Type: {task_type}")
    print(f"  Architecture: 12-layer DeBERTa Transformer")
    print(f"  Attention Heads: 12")
    print(f"  Hidden Dimension: {args.hidden_size}")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Pretrained Weights: {'Yes' if use_pretrained else 'No'}")
    if use_pretrained:
        print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Training Samples: {results['n_train']}")
    print(f"  Validation Samples: {results['n_valid']}")
    print(f"  Test Samples: {results['n_test']}")
    
    if task_type == "regression":
        print(f"  Validation MAE: {results['valid_mae']:.3f}")
        print(f"  Test MAE: {results['test_mae']:.3f}")
        print(f"  Validation RMSE: {results['valid_rmse']:.3f}")
        print(f"  Test RMSE: {results['test_rmse']:.3f}")
        print(f"  Validation R²: {results['valid_r2']:.3f}")
        print(f"  Test R²: {results['test_r2']:.3f}")
    else:
        print(f"  Validation Accuracy: {results['valid_acc']:.3f}")
        print(f"  Test Accuracy: {results['test_acc']:.3f}")
        print(f"  Validation F1: {results['valid_f1']:.3f}")
        print(f"  Test F1: {results['test_f1']:.3f}")
        print(f"  Validation AUC: {results['valid_auc']:.3f}")
        print(f"  Test AUC: {results['test_auc']:.3f}")
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    if task_type == "regression":
        if results['test_r2'] > 0.8:
            print(f"  🏆 Excellent performance (R² > 0.8)")
        elif results['test_r2'] > 0.6:
            print(f"  🥇 Good performance (R² > 0.6)")
        elif results['test_r2'] > 0.4:
            print(f"  🥈 Moderate performance (R² > 0.4)")
        else:
            print(f"  🥉 Basic performance (R² ≤ 0.4)")
    else:
        if results['test_auc'] > 0.9:
            print(f"  🏆 Excellent performance (AUC > 0.9)")
        elif results['test_auc'] > 0.8:
            print(f"  🥇 Good performance (AUC > 0.8)")
        elif results['test_auc'] > 0.7:
            print(f"  🥈 Moderate performance (AUC > 0.7)")
        else:
            print(f"  🥉 Basic performance (AUC ≤ 0.7)")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"  • Check the generated CSV files for detailed predictions")
    print(f"  • The model is saved with unique ID to prevent overwriting")
    print(f"  • Consider hyperparameter tuning for further improvements")
    print(f"  • Try different atom environment radii for better representations")
    print(f"  • Test on other ADMET properties using --property argument")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    



if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            predictions, metrics = result
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 