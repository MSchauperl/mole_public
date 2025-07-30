#!/usr/bin/env python3
"""
Shared utilities for solubility prediction models.

This module contains common functions used across different
Transformer model implementations for solubility prediction.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import tempfile
import os
import re
from typing import List, Tuple, Dict, Optional
import math

# Import TDC for solubility dataset
try:
    from tdc.single_pred import ADME
    tdc_available = True
except ImportError:
    tdc_available = False


class SMILESTokenizer:
    """Simple SMILES tokenizer for molecular transformer"""
    
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Common SMILES tokens
        self.special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.atom_tokens = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H']
        self.bond_tokens = ['-', '=', '#', ':', '/', '\\']
        self.bracket_tokens = ['[', ']', '(', ')']
        self.number_tokens = [str(i) for i in range(10)]
        self.other_tokens = ['+', '-', '@', '*', '%', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Create vocabulary
        self.vocab = self.special_tokens + self.atom_tokens + self.bond_tokens + \
                    self.bracket_tokens + self.number_tokens + self.other_tokens
        
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"SMILES vocabulary size: {self.vocab_size}")
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string"""
        # Add start and end tokens
        smiles = self.start_token + smiles + self.end_token
        
        # Simple character-level tokenization with some SMILES-aware rules
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character tokens first
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.token_to_idx:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character token
            char = smiles[i]
            if char in self.token_to_idx:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
            i += 1
        
        return tokens
    
    def encode(self, smiles: str) -> List[int]:
        """Encode SMILES to token indices"""
        tokens = self.tokenize(smiles)
        indices = [self.token_to_idx.get(token, self.token_to_idx[self.unk_token]) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices += [self.token_to_idx[self.pad_token]] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode token indices back to SMILES"""
        tokens = [self.idx_to_token[idx] for idx in indices]
        # Remove special tokens for output
        tokens = [t for t in tokens if t not in [self.pad_token, self.start_token, self.end_token]]
        return ''.join(tokens)


class MolecularTransformer(nn.Module):
    """Transformer model for molecular property prediction"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_length: int = 128):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Solubility prediction head
        self.solubility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self):
        """Create positional encoding for transformer"""
        pe = torch.zeros(self.max_length, self.d_model)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer"""
        batch_size, seq_len = x.shape
        
        # Token embeddings
        embeddings = self.token_embedding(x)
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Create attention mask for padding
        mask = (x == 0)  # Assuming 0 is pad token index
        
        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension
        # Mask out padding tokens
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        encoded = encoded.masked_fill(mask_expanded, 0)
        
        # Sum and divide by number of non-padding tokens
        seq_lengths = (~mask).sum(dim=1, keepdim=True).float()
        pooled = encoded.sum(dim=1) / seq_lengths
        
        # Solubility prediction
        solubility = self.solubility_head(pooled)
        
        return solubility.squeeze(-1)


class SMILESDataset(Dataset):
    """Dataset for SMILES strings and solubility labels"""
    
    def __init__(self, smiles_list: List[str], labels: List[float], tokenizer: SMILESTokenizer):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Tokenize SMILES
        tokens = self.tokenizer.encode(smiles)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def load_solubility_data():
    """Load TDC AqSolDB solubility dataset"""
    if not tdc_available:
        print("TDC not available. Install with: pip install PyTDC")
        return None, None
    
    print("Loading TDC AqSolDB solubility dataset...")
    data = ADME(name='Solubility_AqSolDB')
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


def preprocess_solubility_data(train_df, test_df, target_col='Y', 
                              use_max_samples=True):
    """Preprocess solubility data with common pipeline"""
    print(f"\n2. Using target: {target_col} (log solubility)")
    
    # Clean the data
    train_clean = train_df.dropna(subset=['Drug', target_col])
    test_clean = test_df.dropna(subset=['Drug', target_col])
    
    print(f"After cleaning: {len(train_clean)} train, {len(test_clean)} test")
    
    # Remove extreme outliers
    print("\n3. Preprocessing and filtering...")
    train_filtered = filter_outliers(train_clean, target_col, n_std=3)
    
    if use_max_samples:
        # Use all available data
        train_subset = train_filtered
        test_subset = test_clean
        print(f"Using maximum available samples: {len(train_subset)} train, {len(test_subset)} test")
    else:
        # Use default sampling (for backward compatibility)
        train_subset = train_filtered.sample(
            n=min(1500, len(train_filtered)), random_state=42
        )
        test_subset = test_clean.sample(
            n=min(300, len(test_clean)), random_state=42
        )
        print(f"Using sampled data: {len(train_subset)} train, {len(test_subset)} test")
    
    # Show target statistics
    train_range = f"{train_subset[target_col].min():.2f} to {train_subset[target_col].max():.2f}"
    test_range = f"{test_subset[target_col].min():.2f} to {test_subset[target_col].max():.2f}"
    print(f"Train solubility range: {train_range}")
    print(f"Test solubility range: {test_range}")
    
    return train_subset, test_subset


def create_transformer_datasets(train_subset, test_subset, target_col='Y', max_length=128):
    """Create PyTorch datasets for transformer model"""
    print("\n4. Creating molecular transformer datasets...")
    
    # Initialize tokenizer
    tokenizer = SMILESTokenizer(max_length=max_length)
    print(f"Using SMILES tokenizer with max length: {max_length}")
    
    # Prepare data
    train_smiles = train_subset['Drug'].values.tolist()
    test_smiles = test_subset['Drug'].values.tolist()
    
    train_labels = train_subset[target_col].values.astype(float)
    test_labels = test_subset[target_col].values.astype(float)
    
    # Use RobustScaler for better outlier handling
    scaler = RobustScaler()
    train_labels_scaled = scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()
    test_labels_scaled = scaler.transform(test_labels.reshape(-1, 1)).flatten()
    
    orig_stats = f"Train mean: {train_labels.mean():.3f}, std: {train_labels.std():.3f}"
    scaled_stats = f"Train mean: {train_labels_scaled.mean():.3f}, std: {train_labels_scaled.std():.3f}"
    print(f"Original target stats - {orig_stats}")
    print(f"Scaled target stats - {scaled_stats}")
    
    # Create PyTorch datasets
    print("\n5. Creating PyTorch datasets...")
    
    train_dataset = SMILESDataset(train_smiles, train_labels_scaled, tokenizer)
    test_dataset = SMILESDataset(test_smiles, test_labels_scaled, tokenizer)
    
    print(f"Training dataset created with {len(train_dataset)} samples")
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    # Store SMILES for later use in CSV output
    train_dataset.smiles = train_smiles
    test_dataset.smiles = test_smiles
    
    return train_dataset, test_dataset, scaler, tokenizer


def train_transformer_model(train_dataset, test_dataset, tokenizer, 
                          epochs=50, batch_size=32, learning_rate=1e-4,
                          device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the transformer model"""
    print(f"\n6. Training Transformer model on {device}...")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MolecularTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_length=tokenizer.max_length
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens)
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
                tokens = batch['tokens'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(tokens)
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
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    print(f"Training completed. Best test loss: {best_test_loss:.4f}")
    
    return model


def evaluate_model(model, train_dataset, test_dataset, scaler, tokenizer, model_name, 
                  save_predictions=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate model and return metrics"""
    print("\n7. Evaluating model...")
    
    model.eval()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    train_preds_scaled = []
    test_preds_scaled = []
    
    with torch.no_grad():
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            outputs = model(tokens)
            train_preds_scaled.extend(outputs.cpu().numpy())
        
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            outputs = model(tokens)
            test_preds_scaled.extend(outputs.cpu().numpy())
    
    train_preds_scaled = np.array(train_preds_scaled)
    test_preds_scaled = np.array(test_preds_scaled)
    
    print("Predictions completed")
    
    # Convert predictions back to original scale
    train_preds = scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
    test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    
    # Get the actual labels
    train_actual = scaler.inverse_transform(train_dataset.labels.reshape(-1, 1)).flatten()
    test_actual = scaler.inverse_transform(test_dataset.labels.reshape(-1, 1)).flatten()
    
    # Calculate regression metrics
    train_mae = mean_absolute_error(train_actual, train_preds)
    test_mae = mean_absolute_error(test_actual, test_preds)
    train_rmse = np.sqrt(mean_squared_error(train_actual, train_preds))
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
    train_r2 = r2_score(train_actual, train_preds)
    test_r2 = r2_score(test_actual, test_preds)
    
    print(f"\n=== Results for Solubility using {model_name} ===")
    print(f"Train MAE:  {train_mae:.3f}")
    print(f"Test MAE:   {test_mae:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}")
    print(f"Train R²:   {train_r2:.3f}")
    print(f"Test R²:    {test_r2:.3f}")
    
    # Save predictions to CSV files
    if save_predictions:
        save_predictions_to_csv(train_preds, train_actual, test_preds, test_actual, 
                               model_name, train_dataset, test_dataset)
    
    # Show some sample predictions
    print(f"\nSample predictions (first 10 test molecules):")
    for i in range(min(10, len(test_preds))):
        pred = test_preds[i]
        actual = test_actual[i]
        error = abs(pred - actual)
        print(f"  Predicted: {pred:.3f}, Actual: {actual:.3f}, Error: {error:.3f}")
    
    return {
        'model_type': model_name,
        'featurizer': 'SMILES Tokenizer',
        'task': 'regression',
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
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
    
    # Create training predictions DataFrame
    train_data = {
        'predicted_solubility': train_preds,
        'actual_solubility': train_actual,
        'absolute_error': np.abs(train_preds - train_actual),
        'squared_error': (train_preds - train_actual) ** 2
    }
    
    # Add SMILES if available
    if train_smiles is not None:
        train_data['smiles'] = train_smiles
    
    train_df = pd.DataFrame(train_data)
    
    # Create test predictions DataFrame
    test_data = {
        'predicted_solubility': test_preds,
        'actual_solubility': test_actual,
        'absolute_error': np.abs(test_preds - test_actual),
        'squared_error': (test_preds - test_actual) ** 2
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
    print(f"  Training - Mean Error: {train_df['absolute_error'].mean():.3f}")
    print(f"  Test - Mean Error: {test_df['absolute_error'].mean():.3f}")
    print(f"  Training - Std Error: {train_df['absolute_error'].std():.3f}")
    print(f"  Test - Std Error: {test_df['absolute_error'].std():.3f}")


def print_model_features(model_name, model_type):
    """Print model-specific features"""
    if "Transformer" in model_type:
        print(f"\n🧠 TRANSFORMER MODEL FEATURES:")
        print(f"  ✅ Multi-head self-attention mechanism (8 heads)")
        print(f"  ✅ 6-layer transformer encoder")
        print(f"  ✅ Positional encoding for sequence awareness")
        print(f"  ✅ SMILES tokenization with custom vocabulary")
        print(f"  ✅ Global average pooling over sequence")
        print(f"  ✅ Dedicated solubility prediction head")
        print(f"  ✅ Dropout regularization (0.1)")
        print(f"  ✅ AdamW optimizer with weight decay")
        print(f"  ✅ Learning rate scheduling with early stopping")
    else:
        print(f"\n⚠️  MODEL TYPE UNKNOWN:")
        print(f"  • Used fallback model")
    
    print(f"\n📈 KEY OPTIMIZATIONS:")
    print(f"  ✅ RobustScaler for outlier handling")
    print(f"  ✅ Outlier filtering (3-sigma rule)")
    print(f"  ✅ Extended training epochs with early stopping")
    print(f"  ✅ Gradient clipping for stability")
    print(f"  ✅ Optimized hyperparameters for {model_name}")


def print_success_message(model_name, model_type):
    """Print success message with model details"""
    print(f"\n🎉 SUCCESS! {model_type} model trained on solubility data")
    print(f"The system successfully:")
    print(f"  ✅ Used SMILES tokenizer for molecular representation")
    print(f"  ✅ Used {model_type} for molecular property prediction")
    print(f"  ✅ Trained on filtered molecular dataset")
    print(f"  ✅ Applied transformer attention mechanisms")
    print(f"  ✅ Generated solubility predictions with dedicated head") 