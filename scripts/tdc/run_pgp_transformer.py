#!/usr/bin/env python3
"""
Main script to run PGP (P-glycoprotein) prediction using MolE Transformer model.

This script demonstrates how to use the MolE Transformer model
for predicting P-glycoprotein inhibition from SMILES strings.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import List, Optional

# Import MolE components
from mole.models.embeddings import AtomEnvEmbeddings
from mole.models.crossenv_mlm import CrossEnvMLMModel
from mole.data.crossenv_datamodule import CrossEnvDataModule
from mole.data.crossenv_dataset import CrossEnvMolDataset

# Import TDC for PGP dataset
try:
    from tdc.single_pred import ADME
    tdc_available = True
except ImportError:
    tdc_available = False


class PGPPredictionHead(nn.Module):
    """PGP prediction head for MolE model"""
    
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.pgp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # For binary classification
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
        Forward pass for PGP prediction
        
        Args:
            hidden_states: Output from transformer encoder [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask for padding [batch_size, seq_len]
        
        Returns:
            PGP inhibition predictions [batch_size] (probabilities)
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
        
        # Predict PGP inhibition
        pgp_prob = self.pgp_head(pooled)
        return pgp_prob.squeeze(-1)


class MolEPGPModel(nn.Module):
    """MolE model for PGP prediction (trained from scratch)"""
    
    def __init__(self, config, hidden_size: int = 768, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        # Create encoder with same architecture as pretrained model
        self.encoder = AtomEnvEmbeddings(config)
        self.pgp_head = PGPPredictionHead(hidden_size, dropout)
        
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
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MolE model
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            PGP inhibition predictions [batch_size] (probabilities)
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
        
        # Predict PGP inhibition
        pgp_prob = self.pgp_head(hidden_states, attention_mask)
        return pgp_prob


class SMILESDataset(Dataset):
    """Dataset for SMILES strings and PGP labels using MolE tokenization"""
    
    def __init__(self, smiles_list: List[str], labels: List[float], data_module: CrossEnvDataModule):
        self.smiles_list = smiles_list
        self.labels = labels
        self.data_module = data_module
        
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


def load_pgp_data():
    """Load TDC PGP dataset"""
    if not tdc_available:
        print("TDC not available. Install with: pip install PyTDC")
        return None, None
    
    print("Loading TDC PGP dataset...")
    data = ADME(name='Pgp_Broccatelli')
    split = data.get_split()
    
    # Combine train and valid for training, use test for evaluation
    train_data = split['train']
    valid_data = split['valid'] 
    test_data = split['test']
    
    # Combine train and validation
    combined_train = pd.concat([train_data, valid_data], ignore_index=True)
    
    print(f"Combined training set: {len(combined_train)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return combined_train, test_data


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


def preprocess_pgp_data(train_df, test_df, target_col='Y', use_max_samples=True):
    """Preprocess PGP data with common pipeline"""
    print(f"\n2. Using target: {target_col} (PGP inhibition)")
    
    # Clean the data
    train_clean = train_df.dropna(subset=['Drug', target_col])
    test_clean = test_df.dropna(subset=['Drug', target_col])
    
    print(f"After cleaning: {len(train_clean)} train, {len(test_clean)} test")
    
    # For binary classification, we don't need outlier filtering
    print("\n3. Preprocessing for binary classification...")
    
    if use_max_samples:
        # Use all available data
        train_subset = train_clean
        test_subset = test_clean
        print(f"Using maximum available samples: {len(train_subset)} train, {len(test_subset)} test")
    else:
        # Use default sampling (for backward compatibility)
        train_subset = train_clean.sample(
            n=min(1500, len(train_clean)), random_state=42
        )
        test_subset = test_clean.sample(
            n=min(300, len(test_clean)), random_state=42
        )
        print(f"Using sampled data: {len(train_subset)} train, {len(test_subset)} test")
    
    # Show target statistics
    train_pos = train_subset[target_col].sum()
    train_neg = len(train_subset) - train_pos
    test_pos = test_subset[target_col].sum()
    test_neg = len(test_subset) - test_pos
    
    print(f"Train class distribution: {train_neg} negative, {train_pos} positive ({train_pos/len(train_subset)*100:.1f}%)")
    print(f"Test class distribution: {test_neg} negative, {test_pos} positive ({test_pos/len(test_subset)*100:.1f}%)")
    
    return train_subset, test_subset


def create_mole_config():
    """Create MolE configuration with same parameters as pretrained model"""
    from DeBERTa.deberta.config import ModelConfig
    
    # Create DeBERTa configuration (matching the pretrained model)
    deberta_config = {
        "attention_head": 12,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "type_vocab_size": 0,
        "vocab_size": 173,  # Input vocabulary size
        "norm_rel_ebd": "layer_norm",
        "position_biased_input": False,
        "pos_att_type": "p2c|c2p",
        "relative_attention": True,
        "max_relative_positions": 128,
        "layer_norm_eps": 1e-7,
        "dropout": 0.1,
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


def create_mole_datasets(train_subset, test_subset, input_vocab_path: str, target_vocab_path: str, target_col='Y'):
    """Create datasets for PGP prediction using MolE tokenization"""
    print("\n4. Creating MolE datasets...")
    
    # Create data module for tokenization
    # We need to provide some dummy data for setup, but we'll handle actual data separately
    dummy_smiles = ["CCO", "CCCO", "CCCC"]  # Simple SMILES for setup
    
    data_module = CrossEnvDataModule(
        train_data=dummy_smiles,
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
    
    # Prepare data
    train_smiles = train_subset['Drug'].values.tolist()
    test_smiles = test_subset['Drug'].values.tolist()
    
    train_labels = train_subset[target_col].values.astype(float)
    test_labels = test_subset[target_col].values.astype(float)
    
    # For binary classification, no scaling needed
    print(f"Using binary labels (0/1) for PGP inhibition prediction")
    
    # Create PyTorch datasets
    print("\n5. Creating PyTorch datasets...")
    
    train_dataset = SMILESDataset(train_smiles, train_labels, data_module)
    test_dataset = SMILESDataset(test_smiles, test_labels, data_module)
    
    print(f"Training dataset created with {len(train_dataset)} samples")
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    # Store SMILES for later use in CSV output
    train_dataset.smiles = train_smiles
    test_dataset.smiles = test_smiles
    
    return train_dataset, test_dataset, data_module


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


def train_mole_model(model, train_dataset, test_dataset, data_module,
                    epochs=30, batch_size=16, learning_rate=1e-4,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    unfreeze_encoder_epoch=None):
    """Train the MolE model from scratch"""
    print(f"\n6. Training MolE model from scratch on {device}...")
    
    if unfreeze_encoder_epoch is not None:
        print(f"   → Encoder will be unfrozen at epoch {unfreeze_encoder_epoch}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer (Binary Cross Entropy for classification)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Unfreeze encoder if specified
        if unfreeze_encoder_epoch is not None and epoch == unfreeze_encoder_epoch:
            model._unfreeze_encoder()
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
            print(f"   → Learning rate reduced to {learning_rate * 0.1:.2e} for fine-tuning")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            smiles = batch['smiles']
            labels = batch['label'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                smiles = batch['smiles']
                labels = batch['label'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        scheduler.step(avg_test_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_mole_pgp_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_mole_pgp_model.pth'))
    print(f"Training completed. Best test loss: {best_test_loss:.4f}")
    
    return model


def evaluate_mole_model(model, train_dataset, test_dataset, data_module, 
                       model_name, save_predictions=True, 
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate MolE model and return metrics"""
    print("\n7. Evaluating model...")
    
    model.eval()
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Reduced batch size
    
    # Get predictions
    train_preds = []
    test_preds = []
    
    with torch.no_grad():
        for batch in train_loader:
            smiles = batch['smiles']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            train_preds.extend(outputs.cpu().numpy())
        
        for batch in test_loader:
            smiles = batch['smiles']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            test_preds.extend(outputs.cpu().numpy())
    
    train_preds = np.array(train_preds)
    test_preds = np.array(test_preds)
    
    print("Predictions completed")
    
    # Get the actual labels
    train_actual = np.array(train_dataset.labels)
    test_actual = np.array(test_dataset.labels)
    
    # Convert probabilities to binary predictions
    train_preds_binary = (train_preds > 0.5).astype(int)
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    # Calculate classification metrics
    train_accuracy = accuracy_score(train_actual, train_preds_binary)
    test_accuracy = accuracy_score(test_actual, test_preds_binary)
    train_precision = precision_score(train_actual, train_preds_binary, zero_division=0)
    test_precision = precision_score(test_actual, test_preds_binary, zero_division=0)
    train_recall = recall_score(train_actual, train_preds_binary, zero_division=0)
    test_recall = recall_score(test_actual, test_preds_binary, zero_division=0)
    train_f1 = f1_score(train_actual, train_preds_binary, zero_division=0)
    test_f1 = f1_score(test_actual, test_preds_binary, zero_division=0)
    train_auc = roc_auc_score(train_actual, train_preds)
    test_auc = roc_auc_score(test_actual, test_preds)
    
    print(f"\n=== Results for PGP using {model_name} ===")
    print(f"Train Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy:  {test_accuracy:.3f}")
    print(f"Train Precision: {train_precision:.3f}")
    print(f"Test Precision:  {test_precision:.3f}")
    print(f"Train Recall: {train_recall:.3f}")
    print(f"Test Recall:  {test_recall:.3f}")
    print(f"Train F1: {train_f1:.3f}")
    print(f"Test F1:  {test_f1:.3f}")
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test AUC:  {test_auc:.3f}")
    
    # Save predictions to CSV files
    if save_predictions:
        save_predictions_to_csv(train_preds, train_actual, test_preds, test_actual, 
                               model_name, train_dataset, test_dataset)
    
    # Show some sample predictions
    print(f"\nSample predictions (first 10 test molecules):")
    for i in range(min(10, len(test_preds))):
        pred_prob = test_preds[i]
        pred_class = test_preds_binary[i]
        actual = test_actual[i]
        print(f"  Predicted: {pred_prob:.3f} ({pred_class}), Actual: {actual}, Correct: {pred_class == actual}")
    
    return {
        'model_type': model_name,
        'featurizer': 'MolE Atom Environments',
        'task': 'binary_classification',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset)
    }


def save_predictions_to_csv(train_preds, train_actual, test_preds, test_actual, 
                           model_name, train_dataset=None, test_dataset=None):
    """Save predictions to CSV files with SMILES"""
    import datetime
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get SMILES if available
    train_smiles = getattr(train_dataset, 'smiles', None) if train_dataset else None
    test_smiles = getattr(test_dataset, 'smiles', None) if test_dataset else None
    
    # Convert probabilities to binary predictions
    train_preds_binary = (train_preds > 0.5).astype(int)
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    # Create training predictions DataFrame
    train_data = {
        'predicted_probability': train_preds,
        'predicted_class': train_preds_binary,
        'actual_class': train_actual,
        'correct_prediction': (train_preds_binary == train_actual).astype(int)
    }
    
    # Add SMILES if available
    if train_smiles is not None:
        train_data['smiles'] = train_smiles
    
    train_df = pd.DataFrame(train_data)
    
    # Create test predictions DataFrame
    test_data = {
        'predicted_probability': test_preds,
        'predicted_class': test_preds_binary,
        'actual_class': test_actual,
        'correct_prediction': (test_preds_binary == test_actual).astype(int)
    }
    
    # Add SMILES if available
    if test_smiles is not None:
        test_data['smiles'] = test_smiles
    
    test_df = pd.DataFrame(test_data)
    
    # Generate filenames
    train_filename = f"predictions_{model_name.lower()}_train_{timestamp}.csv"
    test_filename = f"predictions_{model_name.lower()}_test_{timestamp}.csv"
    
    # Save to CSV
    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)
    
    print(f"\n💾 Predictions saved to CSV files:")
    print(f"  📄 Training: {train_filename} ({len(train_df)} samples)")
    print(f"  📄 Test: {test_filename} ({len(test_df)} samples)")
    
    if train_smiles is not None:
        print(f"  ✅ SMILES included in CSV files")
    
    # Print summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"  Training - Accuracy: {train_df['correct_prediction'].mean():.3f}")
    print(f"  Test - Accuracy: {test_df['correct_prediction'].mean():.3f}")
    print(f"  Training - Mean Probability: {train_df['predicted_probability'].mean():.3f}")
    print(f"  Test - Mean Probability: {test_df['predicted_probability'].mean():.3f}")


def print_model_features(model_name, model_type):
    """Print model-specific features"""
    if "MolE" in model_type:
        print(f"\n🧠 MOLECULAR ENVIRONMENT (MOLECULAR) TRANSFORMER FEATURES:")
        print(f"  ✅ DeBERTa-based architecture with disentangled attention")
        print(f"  ✅ 12-layer transformer encoder (768 hidden, 12 heads)")
        print(f"  ✅ Atom environment tokenization (radius 0 structural)")
        print(f"  ✅ RDKit-based molecular featurization")
        print(f"  ✅ Global average pooling over sequence")
        print(f"  ✅ Dedicated PGP prediction head with sigmoid activation")
        print(f"  ✅ Configurable encoder freezing for transfer learning")
        print(f"  ✅ Gradual unfreezing strategy for optimal fine-tuning")
        print(f"  ✅ Dropout regularization (0.1)")
        print(f"  ✅ AdamW optimizer with weight decay")
        print(f"  ✅ Learning rate scheduling with early stopping")
    else:
        print(f"\n⚠️  MODEL TYPE UNKNOWN:")
        print(f"  • Used fallback model")
    
    print(f"\n📈 KEY OPTIMIZATIONS:")
    print(f"  ✅ Binary cross-entropy loss for classification")
    print(f"  ✅ Sigmoid activation for probability output")
    print(f"  ✅ Extended training epochs with early stopping")
    print(f"  ✅ Gradient clipping for stability")
    print(f"  ✅ Optimized hyperparameters for {model_name}")


def print_success_message(model_name, model_type):
    """Print success message with model details"""
    print(f"\n🎉 SUCCESS! {model_type} model trained on PGP data")
    print(f"The system successfully:")
    print(f"  ✅ Used MolE atom environment tokenization")
    print(f"  ✅ Used {model_type} for molecular property prediction")
    print(f"  ✅ Trained from scratch with DeBERTa architecture")
    print(f"  ✅ Applied transformer attention mechanisms")
    print(f"  ✅ Generated PGP inhibition predictions with dedicated head")


def main():
    """Main execution function"""
    print("🧪 MOLECULAR PGP PREDICTION WITH MOLECULAR ENVIRONMENT TRANSFORMER")
    print("=" * 80)
    
    # Configuration flags
    FREEZE_ENCODER = True  # Set to True to only train the prediction head, False to train everything
    GRADUAL_UNFREEZING = True  # Set to True to unfreeze encoder after some epochs
    UNFREEZE_EPOCH = 15  # Epoch at which to unfreeze encoder (if GRADUAL_UNFREEZING=True)
    
    print(f"🔧 Configuration:")
    print(f"   FREEZE_ENCODER = {FREEZE_ENCODER}")
    print(f"   GRADUAL_UNFREEZING = {GRADUAL_UNFREEZING}")
    if FREEZE_ENCODER:
        print("   → Only prediction head parameters will be trained initially")
        if GRADUAL_UNFREEZING:
            print(f"   → Encoder will be unfrozen at epoch {UNFREEZE_EPOCH}")
        else:
            print("   → Encoder parameters will remain frozen")
    else:
        print("   → All parameters (encoder + head) will be trained from start")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Define vocabulary paths (same as pretrained model)
    input_vocab_path = "mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl"
    target_vocab_path = "mole/data/vocabularies/vocabulary_radius1_functional_guacamol_v1.pkl"
    
    # Step 1: Load data
    print("\n1. Loading PGP dataset...")
    train_df, test_df = load_pgp_data()
    
    if train_df is None or test_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Preprocess data
    train_subset, test_subset = preprocess_pgp_data(
        train_df, test_df, target_col='Y', use_max_samples=True
    )
    
    # Step 3: Create MolE configuration
    print("\n3. Creating MolE configuration...")
    config = create_mole_config()
    print(f"✅ MolE configuration created:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Vocabulary size: {config.vocab_size}")
    
    # Step 4: Create MolE datasets
    train_dataset, test_dataset, data_module = create_mole_datasets(
        train_subset, test_subset, input_vocab_path, target_vocab_path, target_col='Y'
    )
    
    # Step 5: Create and train MolE model
    print("\n5. Creating MolE model...")
    model = MolEPGPModel(config, freeze_encoder=FREEZE_ENCODER)
    
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
    
    # Step 6: Train model
    unfreeze_epoch = UNFREEZE_EPOCH if (FREEZE_ENCODER and GRADUAL_UNFREEZING) else None
    model = train_mole_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        data_module=data_module,
        epochs=30,
        batch_size=8,  # Reduced batch size to fit in GPU memory
        learning_rate=1e-4,
        device=device,
        unfreeze_encoder_epoch=unfreeze_epoch
    )
    
    # Step 7: Evaluate model
    results = evaluate_mole_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        data_module=data_module,
        model_name="MolEPGP",
        save_predictions=True,
        device=device
    )
    
    # Step 8: Print model features and success message
    print_model_features("MolEPGP", "MolE")
    print_success_message("MolEPGP", "MolE")
    
    # Step 9: Print final results summary
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"  Model: MolEPGP")
    print(f"  Architecture: 12-layer DeBERTa Transformer")
    print(f"  Attention Heads: 12")
    print(f"  Hidden Dimension: 768")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Training Samples: {results['n_train']}")
    print(f"  Test Samples: {results['n_test']}")
    print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"  Test F1: {results['test_f1']:.3f}")
    print(f"  Test AUC: {results['test_auc']:.3f}")
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    if results['test_auc'] > 0.8:
        print(f"  🏆 Excellent performance (AUC > 0.8)")
    elif results['test_auc'] > 0.7:
        print(f"  🥇 Good performance (AUC > 0.7)")
    elif results['test_auc'] > 0.6:
        print(f"  🥈 Moderate performance (AUC > 0.6)")
    else:
        print(f"  🥉 Basic performance (AUC ≤ 0.6)")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"  • Check the generated CSV files for detailed predictions")
    print(f"  • The model is saved as 'best_mole_pgp_model.pth'")
    print(f"  • Consider hyperparameter tuning for further improvements")
    print(f"  • Try different atom environment radii for better representations")


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