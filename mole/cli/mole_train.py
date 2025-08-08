import logging
import os
import argparse
import sys

from lightning_fabric.utilities.seed import seed_everything
import pytorch_lightning as pl

from mole.data.dataloaders import MolDataModule
from mole.models.base import Model
from mole.models.mole import Supervised

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    """Main function to parse arguments and initiate training."""
    train(args=args)


def parse_args(argv=None):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train a MolE model.")

    # --- General ---
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs (logs, checkpoints).",
    )

    # --- Data ---
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data file (e.g., parquet, csv).",
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default=None,
        help="Path to the validation data file.",
    )
    parser.add_argument(
        "--test_data_path", type=str, default=None, help="Path to the test data file."
    )
    parser.add_argument(
        "--vocabulary_path",
        type=str,
        required=True,
        help="Path to the vocabulary pkl file.",
    )
    parser.add_argument(
        "--radius", type=int, default=0, help="Radius for atom environment calculation."
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Flag to use features (if available in data).",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Flag to compute and use class weights for classification.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )

    # --- Model ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="MolE_Supervised",
        help="Name for the model run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
        help="Number of tasks (output dimension for regression, classes for classification).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the prediction head.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint to fine-tune from.",
    )

    # --- Optimizer ---
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Adam"],
        help="Optimizer type.",
    )
    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimizer."
    )

    # --- Scheduler ---
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear_warmup",
        choices=["linear_warmup", "constant_warmup", "none"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps for linear warmup.",
    )

    # --- Trainer ---
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps. Overrides max_epochs.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (e.g., 1). Use 0 for CPU.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Training precision (e.g., 16, 32, bf16).",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients across batches.",
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=None, help="Gradient clipping value."
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="How often to check validation set (float=fraction of epoch, int=steps).",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="Fraction or number of training batches to use per epoch.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Fraction or number of validation batches to use per epoch.",
    )
    parser.add_argument(
        "--checkpoint_monitor",
        type=str,
        default="val_loss",
        help="Metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Mode for checkpointing (min or max).",
    )

    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def train(args: argparse.Namespace):
    """Initiates and runs the training process."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.checkpoint_monitor,
        mode=args.checkpoint_mode,
        dirpath=os.path.join(args.output_dir, args.model_name, "checkpoints"),
        filename="{epoch:02d}-{step:05d}-{" + args.checkpoint_monitor + ":.4f}",
        save_top_k=5,  # Save the 5 best checkpoints
    )

    trainer_args = {
        "max_epochs": args.max_epochs,
        "max_steps": args.max_steps,
        "accelerator": "gpu" if args.gpus is not None and args.gpus > 0 else "cpu",
        "devices": args.gpus if args.gpus is not None and args.gpus > 0 else None,
        "precision": args.precision,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": args.gradient_clip_val,
        "val_check_interval": args.val_check_interval,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "default_root_dir": os.path.join(args.output_dir, args.model_name),
        "callbacks": [checkpoint_callback],
        "logger": pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(args.output_dir, args.model_name), name="logs"
        ),
    }
    trainer: pl.Trainer = pl.Trainer(**trainer_args)

    logging.info(f"Using RANDOM SEED {args.seed} to seed everything!")
    seed_everything(seed=args.seed, workers=True)

    datamodule_args = {
        "data": args.data_path,
        "validation_data": args.validation_data_path,
        "vocabulary_inp": args.vocabulary_path,
        "radius_inp": args.radius,
        "useFeatures_inp": args.use_features,
        "use_class_weights": args.use_class_weights,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    datamodule: MolDataModule = MolDataModule(**datamodule_args)

    num_warmup_steps = 0
    if args.scheduler in ["linear_warmup", "constant_warmup"]:
        num_training_steps, num_warmup_steps_calculated = get_num_warmup_steps(
            trainer, datamodule, frac_warmup_steps=args.warmup_fraction
        )
        num_warmup_steps = num_warmup_steps_calculated
        args.num_training_steps = num_training_steps

    optimizer_config = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    scheduler_config = {
        "scheduler_type": args.scheduler,
        "num_warmup_steps": num_warmup_steps,
        "num_training_steps": getattr(args, "num_training_steps", None),
    }

    model_args = {
        "task": args.task,
        "num_classes" if args.task == "classification" else "num_tasks": args.num_tasks,
        "dropout_rate": args.dropout,
        "checkpoint_path": args.checkpoint_path,
        "optimizer_cfg": optimizer_config,
        "scheduler_cfg": scheduler_config,
    }

    try:
        datamodule.setup("fit")
        model_args["vocab_size_inp"] = len(datamodule.dictionary_inp)
    except AttributeError:
        logger.warning(
            "Could not automatically determine vocab size for model init. Ensure it's handled correctly."
        )

    model: Model = Supervised(**model_args)

    if hasattr(trainer.logger, "log_hyperparams"):
        hparams_to_log = vars(args)
        hparams_to_log["calculated_num_warmup_steps"] = num_warmup_steps
        trainer.logger.log_hyperparams(hparams_to_log)

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


def get_num_warmup_steps(
    trainer: pl.Trainer, datamodule: MolDataModule, frac_warmup_steps: float = 0.1
):
    """
    Returns the total number of training steps and the number of warmup steps
    """
    if frac_warmup_steps < 0 or frac_warmup_steps > 1:
        raise ValueError("frac_warmup_steps should be float between 0 and 1")

    if (
        trainer.estimated_stepping_batches is not None
        and trainer.estimated_stepping_batches > 0
    ):
        num_training_steps = trainer.estimated_stepping_batches
        logger.info(f"Using trainer.estimated_stepping_batches: {num_training_steps}")
    else:
        datamodule.setup("fit")
        num_steps_per_epoch = len(datamodule.train_dataloader())
        if trainer.max_epochs:
            num_training_steps = num_steps_per_epoch * trainer.max_epochs
        elif trainer.max_steps and trainer.max_steps > 0:
            num_training_steps = trainer.max_steps
        else:
            raise ValueError(
                "Cannot determine num_training_steps. Set max_epochs or max_steps in Trainer."
            )

    num_warmup_steps = int(num_training_steps * frac_warmup_steps)

    logging.info(f"Calculated num_warmup_steps: {num_warmup_steps}")
    logging.info(f"Calculated num_training_steps: {num_training_steps}")

    return num_training_steps, num_warmup_steps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
