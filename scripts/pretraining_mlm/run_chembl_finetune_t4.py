#!/usr/bin/env python3
"""
Convenience script to fine-tune a pretrained ChemBL model

This script provides an easy way to fine-tune a previously trained model
on ChemBL data with optimized settings for Tesla T4 GPU.

Usage examples:
  # Fine-tune with frozen encoder (recommended for most cases)
  python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_mlm/best_model.ckpt --freeze_encoder
  
  # Fine-tune with trainable encoder (continued pretraining)
  python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_mlm/best_model.ckpt
  
  # Fine-tune ChemBL-only model
  python run_chembl_finetune_t4.py --pretrained_path outputs/chembl_only/best_model.ckpt --chembl_only
"""

import argparse
import sys
from chembl_config_base import T4FineTuneConfig, T4OnlyFineTuneConfig, run_chembl_training


def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained ChemBL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint (.ckpt file)"
    )
    
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the encoder layers during fine-tuning (recommended)"
    )
    
    parser.add_argument(
        "--chembl_only",
        action="store_true",
        help="Use ChemBL-only configuration (no MLM)"
    )
    
    # Parse known args to allow passing through other arguments
    args, unknown_args = parser.parse_known_args()
    
    # Add unknown args back to sys.argv for the training script
    sys.argv = [sys.argv[0]] + unknown_args
    
    # Choose configuration based on flags
    if args.chembl_only:
        config_class = T4OnlyFineTuneConfig
        title = "Starting MolE ChemBL Fine-Tuning (Classification Only)"
    else:
        config_class = T4FineTuneConfig
        title = "Starting MolE ChemBL Fine-Tuning (MLM + Classification)"
    
    # Add freeze info to title
    if args.freeze_encoder:
        title += " - Frozen Encoder"
    else:
        title += " - Trainable Encoder"
    
    # Run fine-tuning
    run_chembl_training(
        config_class=config_class,
        title=title,
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder
    )


if __name__ == "__main__":
    main() 