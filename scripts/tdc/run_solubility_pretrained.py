#!/usr/bin/env python3
"""
Main script to run solubility prediction using pretrained MolE Transformer model.

This script demonstrates how to load a pretrained MolE checkpoint from cross-environment
training and fine-tune it for solubility prediction.
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Suppress RDKit warnings about hydrogen atoms
import warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

from solubility_pretrained_transformer import (
    load_solubility_data,
    preprocess_solubility_data,
    load_pretrained_model,
    create_solubility_datasets,
    train_fine_tuned_model,
    evaluate_fine_tuned_model,
    print_model_features,
    print_success_message
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Molecular solubility prediction using pretrained MolE Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/mschauperl/mole_bucket/outputs/guacamol_crossenv_mlm/guacamol_r0_to_r1_functional_a100_optimized/checkpoints/epoch=11-step=66442-val_loss=2.6733.ckpt",
        help="Path to pretrained MolE checkpoint"
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
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
        default=5,
        help="Number of epochs to keep encoder frozen (only used if --freeze_encoder is set)"
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=0.1,
        help="Learning rate ratio for encoder relative to prediction head when unfrozen"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    print("🧪 MOLECULAR SOLUBILITY PREDICTION WITH PRETRAINED MOLECULAR TRANSFORMER")
    print("=" * 80)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Print configuration
    print(f"📁 Checkpoint: {args.checkpoint_path}")
    print(f"📚 Input vocab: {args.input_vocab}")
    print(f"📚 Target vocab: {args.target_vocab}")
    print(f"🔧 Training epochs: {args.epochs}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"🔧 Learning rate: {args.learning_rate}")
    
    if args.freeze_encoder:
        print(f"🧊 Encoder freezing: ON (first {args.freeze_epochs} epochs)")
        print(f"🧊 Encoder LR ratio: {args.encoder_lr_ratio} (when unfrozen)")
    else:
        print("🔥 Encoder freezing: OFF (full fine-tuning)")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        print("Please ensure the checkpoint file exists and the path is correct.")
        return
    
    print(f"✅ Found checkpoint: {args.checkpoint_path}")
    
    # Step 1: Load data
    print("\n1. Loading solubility dataset...")
    train_df, test_df = load_solubility_data()
    
    if train_df is None or test_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Preprocess data
    train_subset, test_subset = preprocess_solubility_data(
        train_df, test_df, target_col='Y', use_max_samples=True
    )
    
    # Step 3: Load pretrained model
    fine_tuned_model, input_vocab, target_vocab = load_pretrained_model(
        checkpoint_path=args.checkpoint_path,
        input_vocab_path=args.input_vocab,
        target_vocab_path=args.target_vocab
    )
    
    # Step 4: Create datasets
    train_dataset, test_dataset, scaler, data_module = create_solubility_datasets(
        train_subset=train_subset,
        test_subset=test_subset,
        input_vocab_path=args.input_vocab,
        target_vocab_path=args.target_vocab,
        target_col='Y'
    )
    
    # Step 5: Train fine-tuned model
    model = train_fine_tuned_model(
        model=fine_tuned_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        data_module=data_module,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_encoder=args.freeze_encoder,
        freeze_epochs=args.freeze_epochs,
        encoder_lr_ratio=args.encoder_lr_ratio,
        device=device
    )
    
    # Step 6: Evaluate model
    results = evaluate_fine_tuned_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        scaler=scaler,
        data_module=data_module,
        model_name="PretrainedMolESolubility",
        save_predictions=True,
        device=device
    )
    
    # Step 7: Print model features and success message
    print_model_features("PretrainedMolESolubility", "Pretrained")
    print_success_message("PretrainedMolESolubility", "Pretrained")
    
    # Step 8: Print final results summary
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"  Model: PretrainedMolESolubility")
    print(f"  Architecture: 12-layer DeBERTa Encoder")
    print(f"  Attention Heads: 12")
    print(f"  Hidden Dimension: 768")
    print(f"  Input Vocabulary Size: {len(input_vocab)}")
    print(f"  Target Vocabulary Size: {len(target_vocab)}")
    print(f"  Training Samples: {results['n_train']}")
    print(f"  Test Samples: {results['n_test']}")
    print(f"  Test MAE: {results['test_mae']:.3f}")
    print(f"  Test RMSE: {results['test_rmse']:.3f}")
    print(f"  Test R²: {results['test_r2']:.3f}")
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    if results['test_r2'] > 0.8:
        print(f"  🏆 Excellent performance (R² > 0.8)")
    elif results['test_r2'] > 0.6:
        print(f"  🥇 Good performance (R² > 0.6)")
    elif results['test_r2'] > 0.4:
        print(f"  🥈 Moderate performance (R² > 0.4)")
    else:
        print(f"  🥉 Basic performance (R² ≤ 0.4)")
    
    print(f"\n🔬 PRETRAINING DETAILS:")
    print(f"  • Pretrained on GuacaMol dataset (~1.6M molecules)")
    print(f"  • Cross-environment MLM task: Radius 0 → Radius 1")
    print(f"  • Checkpoint: {Path(args.checkpoint_path).name}")
    print(f"  • Fine-tuned for solubility regression")
    
    if args.freeze_encoder:
        print(f"\n🧊 ENCODER FREEZING STRATEGY:")
        print(f"  • Phase 1 (epochs 1-{args.freeze_epochs}): Encoder frozen, train prediction head only")
        print(f"  • Phase 2 (epochs {args.freeze_epochs+1}-{args.epochs}): Encoder unfrozen with {args.encoder_lr_ratio}x learning rate")
        print(f"  • Benefits: Faster convergence, prevents catastrophic forgetting")
    else:
        print(f"\n🔥 FULL FINE-TUNING:")
        print(f"  • All parameters trainable from start")
        print(f"  • Single learning rate for entire model")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"  • Check the generated CSV files for detailed predictions")
    print(f"  • The fine-tuned model is saved as 'best_finetuned_solubility_model.pth'")
    print(f"  • Consider gradual unfreezing for better fine-tuning")
    print(f"  • Try different learning rates and batch sizes")
    print(f"  • Experiment with different prediction head architectures")


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