#!/usr/bin/env python3
"""
ADMET Property Prediction using Pretrained MolE Transformer Model

This script demonstrates how to load a pretrained MolE checkpoint from cross-environment
training and fine-tune it for various ADMET property predictions using the TDC benchmark.

Features:
- Support for all ADMET properties from TDC benchmark group
- Multiple random seeds for robust evaluation
- Automatic task type detection (regression vs classification)
- Pretrained MolE model fine-tuning
- Comprehensive evaluation metrics
- TDC benchmark group evaluation
- Flexible command-line interface

Usage:
    python run_admet_pretrained.py --property herg --seeds 1 2 3
    python run_admet_pretrained.py --list-properties
    python run_admet_pretrained.py --property lipophilicity_astrazeneca --epochs 50
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import datetime
import pickle
from typing import List, Tuple, Dict, Optional, Union

# Suppress RDKit warnings about hydrogen atoms
import warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import RobustScaler

# Import TDC
try:
    from tdc.benchmark_group import admet_group
    from tdc.single_pred import ADME, Tox
    tdc_available = True
except ImportError:
    tdc_available = False
    print("Warning: TDC not available. Please install with: pip install PyTDC")

# Import MolE components
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mole.models.embeddings import AtomEnvEmbeddings
from mole.models.crossenv_mlm import CrossEnvMLMModel
from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.data.crossenv_dataset import CrossEnvMolDataset


class ADMETRegularPredictionHead(nn.Module):
    """ADMET prediction head for fine-tuning pretrained MolE model"""
    
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1, task_type: str = "regression"):
        super().__init__()
        
        self.task_type = task_type
        
        if task_type == "regression":
            output_size = 1
        else:  # classification
            output_size = 2  # Binary classification
            
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # Initialize weights
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
            # Handle attention mask shape
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)
            
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states.masked_fill(mask_expanded == 0, 0)
            
            # Sum and divide by number of non-padding tokens
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            pooled = hidden_states.sum(dim=1) / seq_lengths
        else:
            # Simple average pooling
            pooled = hidden_states.mean(dim=1)
        
        # Make predictions
        output = self.prediction_head(pooled)
        
        if self.task_type == "regression":
            return output.squeeze(-1)  # [batch_size]
        else:
            return output  # [batch_size, 2]


class ADMETDataset(Dataset):
    """Dataset for ADMET properties with MolE tokenization"""
    
    def __init__(self, smiles: List[str], labels: List[float], input_vocab: Dict, target_vocab: Dict,
                 max_length: int = 256, scaler: Optional[RobustScaler] = None, task_type: str = "regression"):
        self.smiles = smiles
        self.labels = np.array(labels)
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length
        self.scaler = scaler
        self.task_type = task_type
        
        # Scale labels for regression
        if task_type == "regression" and scaler is not None:
            self.scaled_labels = scaler.transform(self.labels.reshape(-1, 1)).flatten()
        else:
            self.scaled_labels = self.labels
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.scaled_labels[idx]
        
        # Tokenize SMILES (simplified - you might want to use the actual MolE tokenization)
        # For now, using character-level tokenization as placeholder
        tokens = list(smiles)[:self.max_length-2]  # Reserve space for special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs (simplified)
        token_ids = []
        for token in tokens:
            if token in self.input_vocab:
                token_ids.append(self.input_vocab[token])
            else:
                token_ids.append(self.input_vocab.get('[UNK]', 0))
        
        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(self.input_vocab.get('[PAD]', 0))
        
        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(token_ids[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:self.max_length], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
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


def load_pretrained_model(checkpoint_path: str, input_vocab_path: str, target_vocab_path: str) -> Tuple[nn.Module, Dict, Dict]:
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


def create_admet_model(base_model: nn.Module, task_type: str = "regression", hidden_size: int = 768) -> nn.Module:
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


def preprocess_data(train_df: pd.DataFrame, valid_df: pd.DataFrame, target_col: str = "Y") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Preprocess training and validation data"""
    print(f"Preprocessing data with target: {target_col}")
    
    # Clean the data
    train_clean = train_df.dropna(subset=["Drug", target_col])
    valid_clean = valid_df.dropna(subset=["Drug", target_col])
    
    print(f"After cleaning: {len(train_clean)} train, {len(valid_clean)} valid")
    
    # Detect task type
    task_type = detect_task_type(train_clean[target_col].values)
    print(f"Detected task type: {task_type}")
    
    # Remove extreme outliers from training data only (for regression tasks)
    if task_type == "regression":
        mean_val = train_clean[target_col].mean()
        std_val = train_clean[target_col].std()
        
        mask = (train_clean[target_col] >= mean_val - 3 * std_val) & (
            train_clean[target_col] <= mean_val + 3 * std_val
        )
        train_filtered = train_clean[mask]
        
        print(f"Removed {len(train_clean) - len(train_filtered)} outliers from training")
    else:
        train_filtered = train_clean
        print("No outlier removal for classification task")
    
    return train_filtered, valid_clean, task_type


def create_data_loaders(train_df: pd.DataFrame, valid_df: pd.DataFrame, input_vocab: Dict, target_vocab: Dict,
                       task_type: str = "regression", target_col: str = "Y", batch_size: int = 16) -> Tuple:
    """Create data loaders for training and validation"""
    
    # Prepare data
    train_smiles = train_df["Drug"].tolist()
    valid_smiles = valid_df["Drug"].tolist()
    train_labels = train_df[target_col].values.astype(float)
    valid_labels = valid_df[target_col].values.astype(float)
    
    # Create scaler for regression
    scaler = None
    if task_type == "regression":
        scaler = RobustScaler()
        scaler.fit(train_labels.reshape(-1, 1))
        
        print(f"Original target stats - Train mean: {train_labels.mean():.3f}, std: {train_labels.std():.3f}")
        scaled_train = scaler.transform(train_labels.reshape(-1, 1)).flatten()
        print(f"Scaled target stats - Train mean: {scaled_train.mean():.3f}, std: {scaled_train.std():.3f}")
    else:
        print(f"Classification - Train positives: {np.sum(train_labels == 1)}, negatives: {np.sum(train_labels == 0)}")
        print(f"Classification - Valid positives: {np.sum(valid_labels == 1)}, negatives: {np.sum(valid_labels == 0)}")
    
    # Create datasets
    train_dataset = ADMETDataset(train_smiles, train_labels, input_vocab, target_vocab, 
                                scaler=scaler, task_type=task_type)
    valid_dataset = ADMETDataset(valid_smiles, valid_labels, input_vocab, target_vocab, 
                                scaler=scaler, task_type=task_type)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader, scaler


def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, 
               task_type: str = "regression", epochs: int = 30, learning_rate: float = 1e-4,
               freeze_encoder: bool = False, freeze_epochs: int = 5, encoder_lr_ratio: float = 0.1,
               device: str = "cuda") -> nn.Module:
    """Train the ADMET model with optional encoder freezing"""
    
    model = model.to(device)
    
    if freeze_encoder:
        print(f"🧊 Encoder freezing strategy:")
        print(f"   • Epochs 1-{freeze_epochs}: Encoder FROZEN, train prediction head only")
        print(f"   • Epochs {freeze_epochs+1}-{epochs}: Encoder UNFROZEN with {encoder_lr_ratio}x learning rate")
    else:
        print("🔥 Full fine-tuning: All parameters trainable from start")
    
    # Loss function
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer and scheduler (will be updated each epoch based on freezing strategy)
    optimizer = None
    scheduler = None
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Training for {epochs} epochs...")
    
    def setup_optimizer_for_epoch(epoch):
        """Setup optimizer based on freezing strategy"""
        if freeze_encoder and epoch < freeze_epochs:
            # Phase 1: Freeze encoder, train only prediction head
            encoder_param_count = 0
            head_param_count = 0
            
            for name, param in model.named_parameters():
                if 'encoder' in name:  # Freeze encoder parameters
                    param.requires_grad = False
                    encoder_param_count += param.numel()
                else:  # Keep prediction head trainable
                    param.requires_grad = True
                    head_param_count += param.numel()
            
            # Only optimize prediction head parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
            
            if epoch == 0:
                print(f"🧊 Epoch {epoch+1}: Encoder FROZEN ({encoder_param_count:,} params), training prediction head ({head_param_count:,} params)")
                
                # Debug: Show parameter categorization
                print("📋 Parameter categorization:")
                for name, param in model.named_parameters():
                    category = "ENCODER (frozen)" if 'encoder' in name else "HEAD (trainable)"
                    print(f"   {name}: {category} ({param.numel():,} params)")
                print()
        else:
            # Phase 2: Unfreeze encoder, use different learning rates
            for param in model.parameters():
                param.requires_grad = True
            
            if freeze_encoder:
                # Differential learning rates: lower for encoder, higher for prediction head
                encoder_params = []
                head_params = []
                encoder_param_count = 0
                head_param_count = 0
                
                for name, param in model.named_parameters():
                    if 'encoder' in name:
                        encoder_params.append(param)
                        encoder_param_count += param.numel()
                    else:
                        head_params.append(param)
                        head_param_count += param.numel()
                
                optimizer = optim.AdamW([
                    {'params': encoder_params, 'lr': learning_rate * encoder_lr_ratio},
                    {'params': head_params, 'lr': learning_rate}
                ], weight_decay=0.01)
                
                if epoch == freeze_epochs:
                    print(f"🔥 Epoch {epoch+1}: Encoder UNFROZEN ({encoder_param_count:,} params, LR={learning_rate * encoder_lr_ratio:.2e}), Head ({head_param_count:,} params, LR={learning_rate:.2e})")
            else:
                # Standard fine-tuning: same learning rate for all parameters
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                
                if epoch == 0:
                    print(f"🔥 Epoch {epoch+1}: Full fine-tuning with LR={learning_rate:.2e}")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        return optimizer, scheduler

    for epoch in range(epochs):
        # Setup optimizer based on current epoch and freezing strategy
        optimizer, scheduler = setup_optimizer_for_epoch(epoch)
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            
            if task_type == "regression":
                loss = criterion(outputs, labels)
            else:
                # Convert labels to long for classification
                labels = labels.long()
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
            train_samples += input_ids.size(0)
        
        avg_train_loss = train_loss / train_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                if task_type == "regression":
                    loss = criterion(outputs, labels)
                else:
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * input_ids.size(0)
                val_samples += input_ids.size(0)
        
        avg_val_loss = val_loss / val_samples
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return model


def evaluate_model(model: nn.Module, test_df: pd.DataFrame, input_vocab: Dict, target_vocab: Dict,
                  scaler: Optional[RobustScaler], task_type: str = "regression", 
                  target_col: str = "Y", device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evaluate model on test set"""
    
    # Clean test data
    test_clean = test_df.dropna(subset=["Drug", target_col])
    print(f"Test set after cleaning: {len(test_clean)} samples")
    
    # Create test dataset
    test_smiles = test_clean["Drug"].tolist()
    test_labels = test_clean[target_col].values.astype(float)
    
    test_dataset = ADMETDataset(test_smiles, test_labels, input_vocab, target_vocab, 
                               scaler=scaler, task_type=task_type)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            
            if task_type == "classification":
                # Get probabilities for positive class
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_predictions.extend(probs.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy())
            
            all_labels.extend(labels.numpy())
    
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    # Inverse transform for regression
    if task_type == "regression" and scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        true_labels = scaler.inverse_transform(true_labels.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    if task_type == "regression":
        metrics = {
            "MAE": mean_absolute_error(true_labels, predictions),
            "RMSE": np.sqrt(mean_squared_error(true_labels, predictions)),
            "R²": r2_score(true_labels, predictions)
        }
    else:
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        metrics = {
            "Accuracy": accuracy_score(true_labels, binary_predictions),
            "Precision": precision_score(true_labels, binary_predictions, zero_division=0),
            "Recall": recall_score(true_labels, binary_predictions, zero_division=0),
            "F1": f1_score(true_labels, binary_predictions, zero_division=0),
            "AUC-ROC": roc_auc_score(true_labels, predictions),
            "AUC-PR": average_precision_score(true_labels, predictions)
        }
    
    print("Test Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return predictions, true_labels, metrics


def save_predictions_to_csv(predictions: np.ndarray, true_labels: np.ndarray, test_smiles: List[str],
                           property_name: str, seed: int, timestamp: str, task_type: str, metrics: Dict):
    """Save predictions to CSV file"""
    
    if task_type == "regression":
        test_data = {
            "smiles": test_smiles,
            "predicted_value": predictions,
            "actual_value": true_labels,
            "absolute_error": np.abs(predictions - true_labels),
            "squared_error": (predictions - true_labels) ** 2,
        }
    else:
        binary_predictions = (predictions > 0.5).astype(int)
        test_data = {
            "smiles": test_smiles,
            "predicted_probability": predictions,
            "predicted_class": binary_predictions,
            "actual_class": true_labels,
            "correct_prediction": (binary_predictions == true_labels).astype(int),
        }
    
    test_df = pd.DataFrame(test_data)
    
    filename = f"predictions_{property_name}_mole_pretrained_seed{seed}_{timestamp}.csv"
    test_df.to_csv(filename, index=False)
    
    print(f"Predictions saved to: {filename}")
    
    return test_df


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


def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="ADMET property prediction using pretrained MolE Transformer"
    )
    parser.add_argument(
        "--property",
        type=str,
        default="Solubility_AqSolDB",
        help="ADMET property to evaluate (default: Solubility_AqSolDB). Use --list-properties to see all available.",
    )
    parser.add_argument(
        "--list-properties",
        action="store_true",
        help="List all available ADMET properties and exit",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/mschauperl/mole_bucket/outputs/guacamol_crossenv_mlm/guacamol_r0_to_r1_functional_a100_optimized/checkpoints/epoch=11-step=66442-val_loss=2.6733.ckpt",
        help="Path to pretrained MolE checkpoint",
    )
    parser.add_argument(
        "--input_vocab",
        type=str,
        default="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to input vocabulary",
    )
    parser.add_argument(
        "--target_vocab",
        type=str,
        default="mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl",
        help="Path to target vocabulary",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds to use for multiple runs (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder during initial training phase",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Number of epochs to keep encoder frozen (only used if --freeze_encoder is set) (default: 5)",
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=0.1,
        help="Learning rate ratio for encoder relative to prediction head when unfrozen (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    # Check TDC availability
    if not tdc_available:
        print("❌ TDC not available. Please install with: pip install PyTDC")
        return
    
    # List available properties if requested
    if args.list_properties:
        print("Available ADMET properties:")
        properties = get_available_admet_properties()
        for i, prop in enumerate(properties, 1):
            print(f"  {i:2d}. {prop}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        print("Please ensure the checkpoint file exists and the path is correct.")
        return
    
    print("🧬 ADMET PROPERTY PREDICTION WITH PRETRAINED MOLECULAR TRANSFORMER")
    print("=" * 80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print(f"✅ Found checkpoint: {args.checkpoint_path}")
    print(f"🎯 Target property: {args.property}")
    print(f"🔄 Random seeds: {args.seeds}")
    print(f"🔧 Training epochs: {args.epochs}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"🔧 Learning rate: {args.learning_rate}")
    
    if args.freeze_encoder:
        print(f"🧊 Encoder freezing: ON (first {args.freeze_epochs} epochs)")
        print(f"🧊 Encoder LR ratio: {args.encoder_lr_ratio} (when unfrozen)")
    else:
        print("🔥 Encoder freezing: OFF (full fine-tuning)")
    print()
    
    # Initialize TDC benchmark group
    print("\n1. Initializing TDC benchmark group...")
    group = admet_group(path="data/")
    
    # Get benchmark
    try:
        benchmark = group.get(args.property)
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]
    except Exception as e:
        print(f"Error: Could not load property '{args.property}': {e}")
        print("Use --list-properties to see available properties")
        return
    
    print(f"Benchmark: {name}")
    print(f"Train+Val set: {len(train_val)} samples")
    print(f"Test set: {len(test)} samples")
    
    # Load pretrained model and vocabularies
    print("\n2. Loading pretrained model...")
    base_model, input_vocab, target_vocab = load_pretrained_model(
        args.checkpoint_path, args.input_vocab, args.target_vocab
    )
    
    # Store results for TDC evaluation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_list = []
    all_metrics = []
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed} EVALUATION")
        print(f"{'='*60}")
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Get train/valid split for this seed
        train, valid = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        
        print(f"Train set: {len(train)} samples")
        print(f"Valid set: {len(valid)} samples")
        
        # Preprocess data
        train_processed, valid_processed, task_type = preprocess_data(train, valid)
        
        # Create ADMET model
        admet_model = create_admet_model(base_model, task_type=task_type, hidden_size=768)
        
        # Create data loaders
        train_loader, valid_loader, scaler = create_data_loaders(
            train_processed, valid_processed, input_vocab, target_vocab,
            task_type=task_type, batch_size=args.batch_size
        )
        
        # Train model
        print(f"\n3. Training model for seed {seed}...")
        trained_model = train_model(
            admet_model, train_loader, valid_loader, task_type=task_type,
            epochs=args.epochs, learning_rate=args.learning_rate,
            freeze_encoder=args.freeze_encoder, freeze_epochs=args.freeze_epochs,
            encoder_lr_ratio=args.encoder_lr_ratio, device=device
        )
        
        # Evaluate on test set
        print(f"\n4. Evaluating on test set...")
        test_predictions, test_actual, metrics = evaluate_model(
            trained_model, test, input_vocab, target_vocab, scaler,
            task_type=task_type, device=device
        )
        
        # Save predictions
        test_smiles = test.dropna(subset=["Drug", "Y"])["Drug"].tolist()
        save_predictions_to_csv(
            test_predictions, test_actual, test_smiles,
            args.property, seed, timestamp, task_type, metrics
        )
        
        # Store for TDC evaluation
        seed_predictions = {name: test_predictions}
        predictions_list.append(seed_predictions)
        all_metrics.append(metrics)
        
        print(f"Completed seed {seed}")
    
    # Compute average metrics across seeds
    print(f"\n{'='*60}")
    print("AVERAGE RESULTS ACROSS SEEDS")
    print(f"{'='*60}")
    
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        print(f"{metric_name}: {avg_metrics[metric_name]['mean']:.4f} ± {avg_metrics[metric_name]['std']:.4f}")
    
    # Evaluate using TDC benchmark group
    print(f"\n{'='*60}")
    print("TDC BENCHMARK EVALUATION")
    print(f"{'='*60}")
    
    try:
        results = group.evaluate_many(predictions_list)
        print("TDC Benchmark Results:")
        if hasattr(results, "items"):
            for metric, (value, std) in results.items():
                print(f"  {metric}: {value:.4f} ± {std:.4f}")
        else:
            print(f"  Results: {results}")
    except Exception as e:
        print(f"TDC evaluation error: {e}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")
    print(f"Property: {args.property}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("All predictions saved with timestamp:", timestamp)
    
    return predictions_list, avg_metrics


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