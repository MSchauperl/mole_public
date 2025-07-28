#!/usr/bin/env python3
"""
Solubility-only training script for MolE.

This script trains a model specifically for molecular solubility prediction,
with a single regression head and dataset filtered to only include molecules
with solubility data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mole.models.embeddings import AtomEnvEmbeddings
from mole.dataset_loader import ADMETDataLoader
from DeBERTa.deberta.config import ModelConfig


class SolubilityRegressionHead(nn.Module):
    """Simple regression head for solubility prediction."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Single output for solubility
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(x)


class SolubilityMolEModel(nn.Module):
    """MolE model for solubility prediction only."""
    
    def __init__(
        self,
        deberta_config: Dict[str, Any],
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Convert dictionary to config object
        config = ModelConfig.from_dict(deberta_config)
        
        # Initialize MolE encoder
        self.encoder = AtomEnvEmbeddings(
            config=config,
            pre_trained=pretrained_path
        )
        
        # Get encoder output dimension
        self.hidden_dim = deberta_config.get('hidden_size', 768)
        
        # Initialize solubility regression head
        self.solubility_head = SolubilityRegressionHead(
            hidden_dim=self.hidden_dim,
            dropout=dropout
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            logging.info("Freezing MolE encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logging.info(f"Initialized SolubilityMolEModel with hidden_dim={self.hidden_dim}")
    
    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        
        # Forward pass through MolE encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_mask=input_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            relative_pos=relative_pos,
            output_all_encoded_layers=True
        )
        
        # Get the last layer hidden states
        hidden_states = encoder_outputs['hidden_states']
        last_hidden_state = hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Use CLS token representation for prediction
        # Handle case where batch might have different structure
        if last_hidden_state.size(0) == 1 and input_ids.size(0) > 1:
            # PyTorch Geometric might batch differently - repeat for all samples
            cls_representation = last_hidden_state[0, 0, :].unsqueeze(0).repeat(input_ids.size(0), 1)
        else:
            cls_representation = last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Forward pass through solubility head
        solubility_output = self.solubility_head(cls_representation)
        
        return {
            'solubility_output': solubility_output,
            'encoder_outputs': encoder_outputs
        }
    
    def compute_loss(
        self,
        solubility_output: torch.Tensor,
        solubility_target: torch.Tensor,
        solubility_mask: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for solubility prediction."""
        
        # Ensure tensors are on the correct device
        solubility_target = solubility_target.to(device)
        solubility_mask = solubility_mask.to(device)
        
        # Compute MSE loss with masking
        mse_loss = nn.functional.mse_loss(
            solubility_output.squeeze() * solubility_mask.float(),
            solubility_target * solubility_mask.float(),
            reduction='sum'
        )
        
        # Normalize by number of valid samples
        if solubility_mask.sum() > 0:
            mse_loss = mse_loss / solubility_mask.sum()
        
        return {
            'total_loss': mse_loss,
            'solubility_loss': mse_loss
        }


class SolubilityLightningModule(pl.LightningModule):
    """Lightning module for solubility training."""
    
    def __init__(
        self,
        model: SolubilityMolEModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # For storing predictions and targets during validation/test
        self.val_predictions = []
        self.val_targets = []
        self.val_masks = []
        
        self.test_predictions = []
        self.test_targets = []
        self.test_masks = []
        
        logging.info(f"Initialized SolubilityLightningModule with lr={learning_rate}")
    
    def forward(self, batch):
        """Forward pass."""
        # Move batch to device
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        # Get input data from PyTorch Geometric batch
        input_ids = batch.x
        
        # PyTorch Geometric concatenates node features across all graphs in batch
        # We need to handle this properly for molecular tokens
        if input_ids.dim() == 1:
            # This might be concatenated features - need to reshape
            if hasattr(batch, 'batch'):
                # Reconstruct individual sequences based on batch indices
                batch_indices = batch.batch
                unique_batch_ids = torch.unique(batch_indices, sorted=True)
                
                # Group tokens by batch
                sequences = []
                for bid in unique_batch_ids:
                    mask = (batch_indices == bid)
                    seq_tokens = input_ids[mask]
                    sequences.append(seq_tokens)
                
                # Pad sequences to same length
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = []
                for seq in sequences:
                    if len(seq) < max_len:
                        padding = torch.zeros(max_len - len(seq), dtype=seq.dtype, device=seq.device)
                        seq = torch.cat([seq, padding])
                    padded_sequences.append(seq)
                
                input_ids = torch.stack(padded_sequences, dim=0)
            else:
                input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        elif input_ids.dim() == 2:
            # Already in correct format [batch_size, seq_len]
            pass
        else:
            raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")
        
        input_mask = (input_ids != 0).long()
        
        return self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            attention_mask=getattr(batch, 'attention_mask', None),
            position_ids=getattr(batch, 'position_ids', None),
            relative_pos=getattr(batch, 'relative_pos', None)
        )
    
    def _extract_solubility_data(self, batch):
        """Extract solubility targets and masks from batch."""
        return batch.Solubility, batch.Solubility_mask
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.forward(batch)
        solubility_target, solubility_mask = self._extract_solubility_data(batch)
        
        # Compute loss
        loss_dict = self.model.compute_loss(
            solubility_output=outputs['solubility_output'],
            solubility_target=solubility_target,
            solubility_mask=solubility_mask,
            device=self.device
        )
        
        # Log training metrics
        self.log('train/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/solubility_loss', loss_dict['solubility_loss'], on_step=True, on_epoch=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(batch)
        solubility_target, solubility_mask = self._extract_solubility_data(batch)
        
        # Compute loss
        loss_dict = self.model.compute_loss(
            solubility_output=outputs['solubility_output'],
            solubility_target=solubility_target,
            solubility_mask=solubility_mask,
            device=self.device
        )
        
        # Store predictions for epoch-level metrics
        if outputs['solubility_output'] is not None:
            # Ensure consistent shapes - flatten to 1D
            preds = outputs['solubility_output'].detach().cpu().squeeze()
            targets = solubility_target.detach().cpu().squeeze()
            masks = solubility_mask.detach().cpu().squeeze()
            
            # Make sure they're 1D (handle single sample case)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            if masks.dim() == 0:
                masks = masks.unsqueeze(0)
                
            self.val_predictions.append(preds)
            self.val_targets.append(targets)
            self.val_masks.append(masks)
        
        # Log validation metrics
        self.log('val/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/solubility_loss', loss_dict['solubility_loss'], on_step=True, on_epoch=True)
        
        return loss_dict['total_loss']
    
    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics."""
        if not self.val_predictions:
            return
        
        try:
            # Concatenate all predictions, targets, and masks (all should be 1D now)
            all_preds = torch.cat(self.val_predictions, dim=0)
            all_targets = torch.cat(self.val_targets, dim=0)
            all_masks = torch.cat(self.val_masks, dim=0)
            
            logging.info(f"Validation shapes - preds: {all_preds.shape}, targets: {all_targets.shape}, masks: {all_masks.shape}")
            
            # Filter to only valid samples
            valid_idx = all_masks.bool()
            if valid_idx.sum() == 0:
                logging.warning("No valid samples found in validation batch")
                return
            
            valid_preds = all_preds[valid_idx]
            valid_targets = all_targets[valid_idx]
            
            logging.info(f"Valid samples - preds: {valid_preds.shape}, targets: {valid_targets.shape}, count: {valid_idx.sum()}")
            
            # Compute MAE
            mae = torch.mean(torch.abs(valid_preds - valid_targets))
            
            # Log metrics
            self.log('val/solubility_MAE', mae, prog_bar=True)
            
            logging.info(f"Validation Solubility MAE: {mae:.4f}")
            
        except Exception as e:
            logging.error(f"Error computing validation metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear stored predictions
        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_masks.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.forward(batch)
        solubility_target, solubility_mask = self._extract_solubility_data(batch)
        
        # Store predictions for test metrics
        if outputs['solubility_output'] is not None:
            # Ensure consistent shapes - flatten to 1D
            preds = outputs['solubility_output'].detach().cpu().squeeze()
            targets = solubility_target.detach().cpu().squeeze()
            masks = solubility_mask.detach().cpu().squeeze()
            
            # Make sure they're 1D (handle single sample case)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            if masks.dim() == 0:
                masks = masks.unsqueeze(0)
                
            self.test_predictions.append(preds)
            self.test_targets.append(targets)
            self.test_masks.append(masks)
        
        # Compute loss for logging
        loss_dict = self.model.compute_loss(
            solubility_output=outputs['solubility_output'],
            solubility_target=solubility_target,
            solubility_mask=solubility_mask,
            device=self.device
        )
        
        self.log('test/total_loss', loss_dict['total_loss'])
        self.log('test/solubility_loss', loss_dict['solubility_loss'])
        
        return loss_dict['total_loss']
    
    def on_test_epoch_end(self):
        """Compute epoch-level test metrics."""
        if not self.test_predictions:
            return
        
        try:
            # Concatenate all predictions, targets, and masks (all should be 1D now)
            all_preds = torch.cat(self.test_predictions, dim=0)
            all_targets = torch.cat(self.test_targets, dim=0)
            all_masks = torch.cat(self.test_masks, dim=0)
            
            logging.info(f"Test shapes - preds: {all_preds.shape}, targets: {all_targets.shape}, masks: {all_masks.shape}")
            
            # Filter to only valid samples
            valid_idx = all_masks.bool()
            if valid_idx.sum() == 0:
                logging.warning("No valid samples found in test batch")
                return
            
            valid_preds = all_preds[valid_idx]
            valid_targets = all_targets[valid_idx]
            
            logging.info(f"Valid test samples - preds: {valid_preds.shape}, targets: {valid_targets.shape}, count: {valid_idx.sum()}")
            
            # Compute MAE
            mae = torch.mean(torch.abs(valid_preds - valid_targets))
            
            # Log metrics
            self.log('test/solubility_MAE', mae)
            
            logging.info(f"Test Solubility MAE: {mae:.4f}")
            
            return {'test/solubility_MAE': mae}
            
        except Exception as e:
            logging.error(f"Error computing test metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.warmup_steps,
            T_mult=2,
            eta_min=self.learning_rate * 0.01,
            last_epoch=-1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def get_arg_parser():
    """Get argument parser for solubility training."""
    parser = argparse.ArgumentParser(description="Train MolE for solubility prediction")
    
    # Dataset arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Table 1 datasets CSV"
    )
    parser.add_argument(
        "--vocab_path", type=str, 
        default="mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to atom environment vocabulary file"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size for transformer"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=None, help="Embedding size (defaults to hidden_size if not specified)"
    )
    parser.add_argument(
        "--max_position_embeddings", type=int, default=2048, help="Maximum sequence length for position embeddings"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--intermediate_size", type=int, default=2048, help="Intermediate size in transformer"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--val_check_interval", type=float, default=0.25, help="Validation check interval"
    )
    
    # Data arguments
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set size"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Validation set size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--precision", type=str, default="32", help="Training precision"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--model_name", type=str, default="solubility_model", help="Model name"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Freeze encoder weights"
    )
    parser.add_argument(
        "--pretrained_path", type=str, help="Path to pretrained model weights"
    )
    
    return parser


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration from arguments."""
    return {
        'hidden_size': args.hidden_size,
        'embedding_size': args.embedding_size if args.embedding_size is not None else args.hidden_size,
        'num_attention_heads': args.num_attention_heads,
        'num_hidden_layers': args.num_hidden_layers,
        'intermediate_size': args.intermediate_size,
        'vocab_size': 1000,  # Will be overridden by vocabulary size
        'max_position_embeddings': args.max_position_embeddings,
        'layer_norm_eps': 1e-12,
        'hidden_dropout_prob': args.dropout,
        'attention_probs_dropout_prob': args.dropout,
        'initializer_range': 0.02,
        'type_vocab_size': 0
    }


def create_solubility_dataloader(
    data_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> ADMETDataLoader:
    """Create dataloader filtered for solubility data only."""
    return ADMETDataLoader(
        data_path=data_path,
        selected_tasks=['Solubility'],  # Only solubility task
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        **kwargs
    )


def main():
    """Main function to run solubility training."""
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s"
    )
    logging.info("Starting Solubility MolE training")
    logging.info(f"Output directory: {args.output_dir}/{args.model_name}")
    
    # Create data loader for solubility only
    logging.info("Setting up data loader for solubility...")
    data_loader = create_solubility_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        vocab_path=args.vocab_path,
        num_workers=args.num_workers
    )
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    logging.info(f"Solubility dataset info: {dataset_info}")
    
    # Create model
    logging.info("Creating solubility model...")
    model_config = create_model_config(args)
    model = SolubilityMolEModel(
        deberta_config=model_config,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        pretrained_path=args.pretrained_path
    )
    
    # Create Lightning module
    lightning_module = SolubilityLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.output_dir}/{args.model_name}/checkpoints",
        filename="{epoch}-{val/solubility_MAE:.4f}",
        monitor="val/solubility_MAE",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    
    if args.patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="val/solubility_MAE",
            patience=args.patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir, name=args.model_name, version="lightning_logs"
    )
    
    # Trainer setup
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=50,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Train the model
    logging.info("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=data_loader.get_dataloader('train'),
        val_dataloaders=data_loader.get_dataloader('val')
    )
    
    # Test the model
    logging.info("Evaluating on test set...")
    test_results = trainer.test(
        lightning_module,
        dataloaders=data_loader.get_dataloader('test')
    )
    
    logging.info("Training completed successfully!")
    logging.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main() 