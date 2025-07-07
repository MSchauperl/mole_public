"""
Training Script for Multi-Task Pretraining Model

This script trains a MolE model on multiple tasks simultaneously:
- Cross-Environment MLM (primary)
- ClogP Prediction (regression)
- Molecular Weight Prediction (regression)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mole.data.multitask_datamodule import MultiTaskDataModule
from mole.models.multitask_model import (
    MultiTaskModel,
    MultiTaskLightningModule,
)
from mole.cli.train_crossenv_mlm import (
    get_arg_parser as get_base_parser,
    create_model_config,
)


def parse_multitask_args():
    """Parse command line arguments for multi-task training"""
    parser = get_base_parser()

    # Add new arguments for multi-task learning
    parser.add_argument(
        "--mlm_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the MLM loss component.",
    )
    parser.add_argument(
        "--regression_loss_weight",
        type=float,
        default=0.1,
        help="Weight for the regression loss component.",
    )
    parser.set_defaults(
        model_name="multitask_model", output_dir="outputs/multitask_pretraining"
    )

    return parser.parse_args()


def main():
    """Main function to run multi-task training"""
    args = parse_multitask_args()
    pl.seed_everything(args.seed, workers=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s"
    )
    logging.info("Starting multi-task pretraining")
    logging.info(f"Output directory: {args.output_dir}/{args.model_name}")

    # Data module setup
    data_module = MultiTaskDataModule(
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        input_vocab_path=args.input_vocab,
        target_vocab_path=args.target_vocab,
        input_radius=args.input_radius,
        target_radius=args.target_radius,
        input_use_features=args.input_use_features,
        target_use_features=args.target_use_features,
        mask_prob=args.mask_prob,
        replace_prob=args.replace_prob,
        random_prob=args.random_prob,
        max_length=args.max_length,
        cls_token=not args.no_cls_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
    )
    data_module.setup("fit")
    vocab_sizes = data_module.get_vocab_sizes()

    # Model configuration and instantiation
    model_config = create_model_config(args)
    model = MultiTaskModel(
        deberta_config=model_config,
        input_vocab_size=vocab_sizes["input_vocab_size"],
        target_vocab_size=vocab_sizes["target_vocab_size"],
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    # Lightning module setup
    lightning_module = MultiTaskLightningModule(
        model=model,
        optimizer_cfg={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        scheduler_cfg={"warmup_steps": args.warmup_steps},
        log_predictions=args.log_predictions,
        mlm_loss_weight=args.mlm_loss_weight,
        regression_loss_weight=args.regression_loss_weight,
    )

    # Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.output_dir}/{args.model_name}/checkpoints",
        filename="{epoch}-{val/total_loss:.2f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    if args.patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)

    logger = TensorBoardLogger(
        save_dir=args.output_dir, name=args.model_name, version="lightning_logs"
    )

    # Trainer setup
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        deterministic=True,
    )

    # Run training
    logging.info("Starting training...")
    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main() 